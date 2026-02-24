#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm


def _setup_path() -> None:
    tts_root = Path(__file__).resolve().parents[1]
    project_root = tts_root.parent
    dtmae_root = project_root / "DTMAE"
    src_root = tts_root / "src"
    for p in (src_root, tts_root, project_root, dtmae_root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


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


def normalize_codes(codes: torch.Tensor) -> torch.Tensor:
    out = codes.detach().cpu()
    while out.dim() > 2:
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        elif out.shape[0] == 1:
            out = out.squeeze(0)
        else:
            break
    if out.dim() == 2 and out.shape[0] == 1:
        out = out.squeeze(0)
    if out.dim() != 1:
        raise RuntimeError(f"Unexpected code shape: {tuple(out.shape)}")
    return out.to(torch.long)


def parse_bool_mask_to_span(mask: torch.Tensor, max_span_len: int) -> List[int]:
    from tts.span_utils import mask_to_span_lengths

    spans = mask_to_span_lengths(mask.to(torch.bool).flatten(), max_span_len=max_span_len)
    return [int(x) for x in spans.tolist()]


def build_mask_from_spans(spans: Sequence[int], device: torch.device) -> torch.Tensor:
    from tts.span_utils import span_lengths_to_mask

    mask = span_lengths_to_mask([int(x) for x in spans]).unsqueeze(0).to(device=device)
    return mask


def is_vfr_from_cfg(cfg) -> bool:
    use_dtp = bool(getattr(cfg.model.resampler, "use_dtp", False))
    dtp_cls = str(getattr(cfg.model.resampler, "dtp_cls", ""))
    normalized = dtp_cls.lower().replace("_", "")
    is_fixed = "fixedpattern" in normalized
    return use_dtp and (not is_fixed)


def fixed_span_len_from_cfg(cfg) -> int:
    r = None
    dtp_params = getattr(cfg.model.resampler, "dtp_params", None)
    if dtp_params is not None and hasattr(dtp_params, "r"):
        r = float(dtp_params.r)
    if r is None or r <= 0.0 or r > 1.0:
        return 2
    span = int(round(1.0 / r))
    return max(1, span)


def apply_cfg_overrides(cfg, overrides: Optional[Sequence[str]]):
    if not overrides:
        return cfg
    override_conf = OmegaConf.from_dotlist(list(overrides))
    return OmegaConf.merge(cfg, override_conf)


@dataclass
class MetaItem:
    file_id: str
    prompt_text: str
    prompt_audio: Path
    target_text: str
    gt_audio: Optional[Path]


def parse_meta_lst(path: str) -> List[MetaItem]:
    meta_path = Path(path).resolve()
    base = meta_path.parent
    items: List[MetaItem] = []

    with open(meta_path, "r", encoding="utf-8") as f:
        for ln in f:
            raw = ln.strip()
            if not raw:
                continue
            parts = [p.strip() for p in raw.split("|")]
            if len(parts) < 4:
                continue

            file_id = parts[0]
            prompt_text = parts[1]
            prompt_audio = Path(parts[2])
            target_text = parts[3]
            gt_audio = Path(parts[4]) if len(parts) >= 5 and parts[4] else None

            if not prompt_audio.is_absolute():
                prompt_audio = (base / prompt_audio).resolve()
            if gt_audio is not None and not gt_audio.is_absolute():
                gt_audio = (base / gt_audio).resolve()

            items.append(
                MetaItem(
                    file_id=file_id,
                    prompt_text=prompt_text,
                    prompt_audio=prompt_audio,
                    target_text=target_text,
                    gt_audio=gt_audio,
                )
            )
    return items


def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    scaled = logits / max(temperature, 1e-6)
    if top_k > 0:
        k = min(top_k, scaled.numel())
        vals, idxs = torch.topk(scaled, k=k)
        probs = torch.softmax(vals, dim=-1)
        pick = torch.multinomial(probs, num_samples=1)
        return int(idxs[pick].item())

    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def run_codec_encode_prompt(codec, wav: torch.Tensor, device: torch.device, dtype: torch.dtype):
    wav = wav.to(device=device, dtype=torch.float32)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    ac = torch.autocast(device_type=device_type, dtype=dtype) if device_type == "cuda" else nullcontext()

    with torch.inference_mode():
        with ac:
            vq_emb = codec.encoder(wav.unsqueeze(1), level=1)

            if codec.use_dtp:
                dtp_out = codec.dtp(vq_emb)
                mask = dtp_out[0]
                downsample_out = codec.downsampler(vq_emb, mask)
                if isinstance(downsample_out, tuple):
                    if len(downsample_out) == 5:
                        vq_emb, position_ids, cu_seqlens, max_seqlen, mask = downsample_out
                    elif len(downsample_out) == 4:
                        vq_emb, position_ids, cu_seqlens, max_seqlen = downsample_out
                    else:
                        raise RuntimeError("Unexpected downsampler output tuple")
                else:
                    vq_emb = downsample_out
                    position_ids = cu_seqlens = max_seqlen = None
            else:
                mask = None
                vq_emb = codec.downsampler(vq_emb)
                position_ids = cu_seqlens = max_seqlen = None

            vq_emb = codec.encoder(
                vq_emb,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                level=2,
            )
            _, vq_code, _ = codec.decoder(vq_emb, vq=True)

    return vq_code.detach().cpu(), (mask.detach().cpu() if mask is not None else None)


def run_codec_decode_tokens(
    codec,
    tokens: Sequence[int],
    spans: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if len(tokens) == 0:
        return torch.zeros(1, dtype=torch.float32)

    mask = build_mask_from_spans(spans, device)
    codes = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    device_type = "cuda" if device.type == "cuda" else "cpu"
    ac = torch.autocast(device_type=device_type, dtype=dtype) if device_type == "cuda" else nullcontext()

    with torch.inference_mode():
        with ac:
            x = codec.decoder.vq2emb(codes)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = codec.decoder(x, vq=False, level=2)
            if getattr(codec, "upsampler_uses_mask", True):
                x = codec.upsampler(x, mask=mask)
            else:
                x = codec.upsampler(x)
            y = codec.decoder(x, vq=False, level=1)

    wav = y.squeeze(0).squeeze(0).detach().cpu().to(torch.float32)
    return wav


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesize wavs from benchmark meta list")
    p.add_argument("--model_dir", type=str, required=True, help="AR model checkpoint directory")
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--speech_vocab_size", type=int, required=True)
    p.add_argument("--codec_run_dir", type=str, required=True)
    p.add_argument("--codec_ckpt", type=str, default=None)
    p.add_argument("--meta_lst", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--output_manifest", type=str, default=None)
    p.add_argument("--use_vfr", action="store_true", help="Enable VFR span prediction")
    p.add_argument("--max_span_len", type=int, default=512)
    p.add_argument("--fixed_span_len", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--cfg_override", action="append", default=None)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    _setup_path()
    from DTMAE.lightning_module import CodecLightningModule
    from tts.collator import TTSCollator
    from tts.constants import BOS_ID, EOS_ID, SEP_ID
    from tts.modeling_ar_tts import ARTTSForConditionalGeneration
    from tts.text_tokenizer import CharTokenizer

    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if (args.dtype == "bfloat16" and device.type == "cuda") else torch.float32

    # Load AR model
    ar_model = ARTTSForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    ar_model.eval()
    tokenizer = CharTokenizer.load(args.tokenizer_path)
    collator = TTSCollator(
        text_vocab_size=tokenizer.vocab_size,
        speech_vocab_size=args.speech_vocab_size,
        use_vfr=args.use_vfr,
        max_span_len=args.max_span_len,
    )

    # Load codec model
    codec_run_dir = Path(args.codec_run_dir).resolve()
    cfg_path = codec_run_dir / "hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Codec config missing: {cfg_path}")
    codec_ckpt = Path(args.codec_ckpt).resolve() if args.codec_ckpt else (codec_run_dir / "pl_log" / "last.ckpt")
    if not codec_ckpt.is_file():
        raise FileNotFoundError(f"Codec checkpoint missing: {codec_ckpt}")

    cfg = OmegaConf.load(str(cfg_path))
    cfg = apply_cfg_overrides(cfg, args.cfg_override)
    sample_rate = int(cfg.dataset.sample_rate)
    multiple_of = int(getattr(cfg.dataset, "multiple_of", 1) or 1)

    codec = CodecLightningModule(cfg=cfg)
    state = torch.load(str(codec_ckpt), map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    patch_legacy_dtp_state_dict(state_dict)
    codec.load_state_dict(state_dict, strict=False)
    codec = codec.to(device)
    codec.eval()

    use_vfr_from_codec = is_vfr_from_cfg(cfg)
    if args.use_vfr and not use_vfr_from_codec:
        print("[Warn] --use_vfr=True but codec config looks fixed-pattern/non-VFR.")

    fixed_span_len = args.fixed_span_len if args.fixed_span_len is not None else fixed_span_len_from_cfg(cfg)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.output_manifest).resolve() if args.output_manifest else (out_dir / "synthesis_manifest.jsonl")

    items = parse_meta_lst(args.meta_lst)
    if args.limit is not None:
        items = items[: args.limit]

    with open(manifest_path, "w", encoding="utf-8") as mf:
        iterator = tqdm(items, desc="Synthesizing")
        for item in iterator:
            if not item.prompt_audio.is_file():
                continue

            wav_prompt, sr = torchaudio.load(str(item.prompt_audio))
            if wav_prompt.dim() == 2 and wav_prompt.shape[0] > 1:
                wav_prompt = wav_prompt[:1, :]
            if wav_prompt.dim() == 1:
                wav_prompt = wav_prompt.unsqueeze(0)
            if sr != sample_rate:
                wav_prompt = torchaudio.transforms.Resample(sr, sample_rate)(wav_prompt)
            wav_prompt = wav_prompt[0]
            if multiple_of > 1 and wav_prompt.numel() % multiple_of != 0:
                pad = multiple_of - (wav_prompt.numel() % multiple_of)
                wav_prompt = torch.nn.functional.pad(wav_prompt, (0, pad))

            prompt_codes, prompt_mask = run_codec_encode_prompt(codec, wav_prompt.unsqueeze(0), device, dtype)
            prompt_tokens = [int(x) for x in normalize_codes(prompt_codes).tolist()]

            prompt_spans: Optional[List[int]] = None
            if args.use_vfr:
                if prompt_mask is None:
                    prompt_spans = [fixed_span_len] * len(prompt_tokens)
                else:
                    prompt_spans = parse_bool_mask_to_span(prompt_mask, max_span_len=args.max_span_len)
                    if len(prompt_spans) != len(prompt_tokens):
                        n = min(len(prompt_spans), len(prompt_tokens))
                        prompt_spans = prompt_spans[:n]
                        prompt_tokens = prompt_tokens[:n]

            text_ids = tokenizer.encode(item.target_text)
            seq = [BOS_ID] + [collator.text_offset + x for x in text_ids] + [SEP_ID] + [collator.speech_offset + x for x in prompt_tokens] + [SEP_ID]
            speech_mask = [0] * (2 + len(text_ids)) + [1] * len(prompt_tokens) + [0]
            span_ids = [1] * (2 + len(text_ids))
            if args.use_vfr:
                assert prompt_spans is not None
                span_ids += prompt_spans + [1]
            else:
                span_ids += [1] * (len(prompt_tokens) + 1)

            gen_tokens: List[int] = []
            gen_spans: List[int] = []
            for _ in range(args.max_new_tokens):
                input_ids = torch.tensor([seq], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                speech_mask_t = torch.tensor([speech_mask], dtype=torch.bool, device=device)
                span_ids_t = torch.tensor([span_ids], dtype=torch.long, device=device)

                with torch.inference_mode():
                    out = ar_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        speech_mask=speech_mask_t,
                        span_ids=span_ids_t,
                    )

                logits = out["logits"][0, -1].clone()
                allowed = torch.full_like(logits, fill_value=float("-inf"))
                s0 = collator.speech_offset
                s1 = collator.speech_offset + args.speech_vocab_size
                allowed[s0:s1] = logits[s0:s1]
                allowed[EOS_ID] = logits[EOS_ID]

                next_id = sample_from_logits(allowed, temperature=args.temperature, top_k=args.top_k)
                if next_id == EOS_ID:
                    seq.append(EOS_ID)
                    speech_mask.append(0)
                    span_ids.append(1)
                    break

                seq.append(next_id)
                speech_mask.append(1)
                gen_tokens.append(int(next_id - collator.speech_offset))

                if args.use_vfr:
                    span_logits = out["span_logits"][0, -1].clone()
                    span_logits[0] = float("-inf")
                    next_span = sample_from_logits(span_logits, temperature=1.0, top_k=0)
                    next_span = max(1, min(int(next_span), args.max_span_len))
                    gen_spans.append(next_span)
                    span_ids.append(next_span)
                else:
                    span_ids.append(1)

            if len(gen_tokens) == 0:
                continue

            if args.use_vfr:
                decode_spans = gen_spans if len(gen_spans) == len(gen_tokens) else [1] * len(gen_tokens)
            else:
                decode_spans = [fixed_span_len] * len(gen_tokens)

            wav_out = run_codec_decode_tokens(codec, gen_tokens, decode_spans, device, dtype)

            fname = item.file_id
            if not fname.lower().endswith(".wav"):
                fname = f"{fname}.wav"
            out_wav = out_dir / fname
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(out_wav), wav_out.unsqueeze(0), sample_rate=sample_rate)

            rec = {
                "file_id": item.file_id,
                "synth_path": str(out_wav),
                "prompt_audio": str(item.prompt_audio),
                "target_text": item.target_text,
                "gt_audio": str(item.gt_audio) if item.gt_audio is not None else None,
                "num_prompt_tokens": len(prompt_tokens),
                "num_generated_tokens": len(gen_tokens),
                "use_vfr": bool(args.use_vfr),
            }
            if args.use_vfr:
                rec["num_generated_spans"] = len(gen_spans)
            mf.write(json.dumps(rec, ensure_ascii=True) + "\n")

    print(f"Saved synthesized wavs to {out_dir}")
    print(f"Saved synthesis manifest to {manifest_path}")


if __name__ == "__main__":
    main()
