#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm


def _setup_import_path() -> None:
    tts_root = Path(__file__).resolve().parents[1]
    project_root = tts_root.parent
    dtmae_root = project_root / "DTMAE"
    for p in (tts_root, project_root, dtmae_root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract offline codec tokens (+ VFR spans) from audio list")
    p.add_argument("--run_dir", type=str, required=True, help="Hydra run dir of tokenizer model")
    p.add_argument("--ckpt", type=str, default=None, help="checkpoint path (default: <run_dir>/pl_log/last.ckpt)")
    p.add_argument("--input", type=str, required=True, help="audio file, audio directory, or .txt filelist")
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--output_metadata", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--max_span_len", type=int, default=512)
    p.add_argument("--sample_rate", type=int, default=None, help="override config sample rate")
    p.add_argument("--multiple_of", type=int, default=None, help="override config multiple_of")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--cfg_override", action="append", default=None)
    return p.parse_args()


def patch_legacy_dtp_state_dict(state_dict: Dict[str, torch.Tensor]) -> None:
    legacy_keys = ("dtp.log_tau", "dtp.r_ema", "dtp.steps")
    if not all(k in state_dict for k in legacy_keys):
        return
    log_tau = state_dict.pop("dtp.log_tau")
    tau = torch.exp(log_tau)
    state_dict["dtp.tau_train"] = tau.clone()
    state_dict["dtp.tau_eval"] = tau.clone()

    r_ema = state_dict.pop("dtp.r_ema")
    state_dict["dtp.r_ema_train"] = r_ema.clone()
    state_dict["dtp.r_ema_eval"] = r_ema.clone()

    steps = state_dict.pop("dtp.steps")
    state_dict["dtp.steps_train"] = steps.clone()
    state_dict["dtp.steps_eval"] = steps.clone()


def apply_cfg_overrides(cfg, overrides: Optional[Sequence[str]]):
    if not overrides:
        return cfg
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))


def parse_filelist(path: str) -> List[str]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() in {".wav", ".flac"}:
        return [str(p.resolve())]
    if p.is_file() and p.suffix.lower() == ".txt":
        out = []
        with open(p, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    out.append(str(Path(ln).resolve()))
        return out
    if p.is_dir():
        files = [str(x.resolve()) for x in p.rglob("*") if x.is_file() and x.suffix.lower() in {".wav", ".flac"}]
        files.sort()
        return files
    raise FileNotFoundError(f"Unsupported input path: {path}")


def pad_to_multiple(wav: torch.Tensor, multiple_of: int) -> Tuple[torch.Tensor, int]:
    orig_len = wav.shape[-1]
    if multiple_of <= 1:
        return wav, orig_len
    if orig_len % multiple_of == 0:
        return wav, orig_len
    pad = multiple_of - (orig_len % multiple_of)
    return torch.nn.functional.pad(wav, (0, pad)), orig_len


def mask_to_trailing(mask: torch.Tensor) -> torch.Tensor:
    seq = mask.to(torch.bool).flatten()
    if seq.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=seq.device)
    out = []
    zeros_after = 0
    for keep in reversed(seq.tolist()):
        if keep:
            out.append(zeros_after)
            zeros_after = 0
        else:
            zeros_after += 1
    out.reverse()
    if not out:
        return torch.zeros(0, dtype=torch.long, device=seq.device)
    return torch.tensor(out, dtype=torch.long, device=seq.device)


def extract_text_from_audio_path(path: Path) -> str:
    cands = [
        path.with_name(path.stem + ".normalized.txt"),
        path.with_name(path.stem + ".original.txt"),
    ]
    for c in cands:
        if c.is_file():
            txt = c.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                return txt

    stem = path.stem
    if "-" in stem:
        parts = stem.split("-")
        if len(parts) >= 2:
            trans = path.parent / f"{parts[0]}-{parts[1]}.trans.txt"
            if trans.is_file():
                prefix = f"{stem} "
                with open(trans, "r", encoding="utf-8") as f:
                    for ln in f:
                        if ln.startswith(prefix):
                            return ln[len(prefix) :].strip()
    return ""


def speaker_and_utt_id(path: Path) -> Tuple[str, str]:
    utt_id = path.stem
    if "_" in utt_id:
        return utt_id.split("_")[0], utt_id
    if "-" in utt_id:
        return utt_id.split("-")[0], utt_id
    return "unknown", utt_id


def normalize_codes(codes: torch.Tensor) -> torch.Tensor:
    c = codes.detach().cpu()
    while c.dim() > 2:
        if c.shape[-1] == 1:
            c = c.squeeze(-1)
        elif c.shape[0] == 1:
            c = c.squeeze(0)
        else:
            break
    if c.dim() == 2 and c.shape[0] == 1:
        c = c.squeeze(0)
    if c.dim() != 1:
        raise RuntimeError(f"Unexpected code shape after normalize: {tuple(c.shape)}")
    return c.to(torch.long)


def dtp_active(cfg) -> bool:
    use_dtp = bool(getattr(cfg.model.resampler, "use_dtp", False))
    dtp_cls = str(getattr(cfg.model.resampler, "dtp_cls", ""))
    is_fixed = "fixedpattern" in dtp_cls.replace("_", "").lower()
    return use_dtp and (not is_fixed)


def infer_codebook_size(cfg) -> int:
    qparams = cfg.model.quantizer.params
    if hasattr(qparams, "codebook_size"):
        return int(qparams.codebook_size)
    if hasattr(qparams, "inference_levels") and hasattr(qparams, "codebook_dim"):
        levels = qparams.inference_levels
        if isinstance(levels, (list, tuple)):
            size = 1
            for l in levels:
                size *= int(l)
            return size
        return int(levels) ** int(qparams.codebook_dim)
    return -1


def run_codec_forward(model, wav: torch.Tensor, device: torch.device, dtype: torch.dtype):
    wav = wav.to(device=device, dtype=torch.float32)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    ac = torch.autocast(device_type=device_type, dtype=dtype) if device_type == "cuda" else nullcontext()

    with torch.inference_mode():
        with ac:
            vq_emb = model.encoder(wav.unsqueeze(1), level=1)
            mask = None
            if model.use_dtp:
                dtp_out = model.dtp(vq_emb)
                mask = dtp_out[0]
                down_out = model.downsampler(vq_emb, mask)
                if isinstance(down_out, tuple):
                    if len(down_out) == 5:
                        vq_emb, position_ids, cu_seqlens, max_seqlen, mask = down_out
                    elif len(down_out) == 4:
                        vq_emb, position_ids, cu_seqlens, max_seqlen = down_out
                    else:
                        raise RuntimeError("Unexpected downsampler tuple")
                else:
                    vq_emb = down_out
                    position_ids = cu_seqlens = max_seqlen = None
            else:
                vq_emb = model.downsampler(vq_emb)
                position_ids = cu_seqlens = max_seqlen = None

            vq_emb = model.encoder(vq_emb, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, level=2)
            _, vq_code, _ = model.decoder(vq_emb, vq=True)
    return vq_code, mask


def main() -> None:
    _setup_import_path()
    from DTMAE.lightning_module import CodecLightningModule

    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    ckpt_path = Path(args.ckpt).resolve() if args.ckpt else (run_dir / "pl_log" / "last.ckpt")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = OmegaConf.load(str(cfg_path))
    cfg = apply_cfg_overrides(cfg, args.cfg_override)

    sample_rate = int(args.sample_rate) if args.sample_rate is not None else int(cfg.dataset.sample_rate)
    multiple_of = int(args.multiple_of) if args.multiple_of is not None else int(getattr(cfg.dataset, "multiple_of", 1) or 1)
    use_vfr = dtp_active(cfg)

    model = CodecLightningModule(cfg=cfg)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    patch_legacy_dtp_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if (args.dtype == "bfloat16" and device.type == "cuda") else torch.float32
    model = model.to(device)
    model.eval()

    all_paths = parse_filelist(args.input)
    if args.max_files is not None:
        all_paths = all_paths[: args.max_files]
    if not all_paths:
        raise RuntimeError("No input files found")

    out_path = Path(args.output_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as wf:
        for p in tqdm(all_paths, desc="Extracting tokens"):
            ap = Path(p)
            wav, sr = torchaudio.load(str(ap))
            if wav.dim() == 2 and wav.shape[0] > 1:
                wav = wav[:1, :]
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            if sr != sample_rate:
                wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
            wav = wav[0]
            wav, orig_len = pad_to_multiple(wav, multiple_of)
            wav = wav.unsqueeze(0)

            codes, mask = run_codec_forward(model, wav, device, dtype)
            codes_1d = normalize_codes(codes)
            tokens = [int(x) for x in codes_1d.tolist()]

            spans = None
            if use_vfr:
                if mask is None:
                    continue
                trailing = mask_to_trailing(mask.detach().cpu())
                span_t = torch.clamp(trailing + 1, min=1, max=args.max_span_len)
                if span_t.numel() != len(tokens):
                    n = min(int(span_t.numel()), len(tokens))
                    tokens = tokens[:n]
                    span_t = span_t[:n]
                spans = [int(x) for x in span_t.tolist()]

            speaker_id, utt_id = speaker_and_utt_id(ap)
            text = extract_text_from_audio_path(ap)
            if not text:
                continue

            rec = {
                "audio_path": str(ap),
                "utt_id": utt_id,
                "speaker_id": speaker_id,
                "sample_rate": sample_rate,
                "orig_num_samples": int(orig_len),
                "tokens": tokens,
                "text": text,
            }
            if spans is not None:
                rec["spans"] = spans
            wf.write(json.dumps(rec, ensure_ascii=True) + "\n")
            n_written += 1

    meta = {
        "run_dir": str(run_dir),
        "ckpt": str(ckpt_path),
        "input": str(args.input),
        "output_jsonl": str(out_path),
        "num_utterances": n_written,
        "sample_rate": sample_rate,
        "multiple_of": multiple_of,
        "use_vfr": bool(use_vfr),
        "max_span_len": int(args.max_span_len),
        "codebook_size": infer_codebook_size(cfg),
        "dtype": args.dtype,
        "device": args.device,
    }
    meta_path = Path(args.output_metadata).resolve() if args.output_metadata else out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)
    print(f"Saved {n_written} utterances to {out_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
