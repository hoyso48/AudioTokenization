#!/usr/bin/env python3
"""
Codec index extraction utility for LM evaluation.

Given LibriSpeech file lists and a trained DTMAE checkpoint, the script
decodes every utterance into codec indices (and optional DTP masks), writes
flattened corpora for train/test, and records the metadata required by the
LM training script.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

EVAL_LM_ROOT = Path(__file__).resolve().parent
EVAL_ROOT = EVAL_LM_ROOT.parent
PROJECT_ROOT = EVAL_ROOT.parent
DTMAE_ROOT = PROJECT_ROOT / "DTMAE"

for path in (EVAL_LM_ROOT, EVAL_ROOT, PROJECT_ROOT, DTMAE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from DTMAE.lightning_module import CodecLightningModule

ALLOWED_AUDIO_EXTS = {".wav", ".flac"}


def read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def parse_filelist(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() == ".txt":
            return [str(Path(x)) for x in read_lines(str(p))]
        if p.suffix.lower() in ALLOWED_AUDIO_EXTS:
            return [str(p.resolve())]
    if p.is_dir():
        entries = [
            str(fp.resolve())
            for fp in p.rglob("*")
            if fp.is_file() and fp.suffix.lower() in ALLOWED_AUDIO_EXTS
        ]
        entries.sort()
        return entries
    raise FileNotFoundError(f"Unsupported input path: {path}")


def pad_to_multiple_1d(waveform: torch.Tensor, multiple_of: int) -> Tuple[torch.Tensor, int]:
    length = waveform.shape[-1]
    if multiple_of <= 0:
        return waveform, length
    if length % multiple_of == 0:
        return waveform, length
    pad_len = multiple_of - (length % multiple_of)
    padded = torch.nn.functional.pad(waveform, (0, pad_len))
    return padded, length


def apply_cfg_overrides(cfg, overrides: Optional[Sequence[str]]):
    if not overrides:
        return cfg
    override_conf = OmegaConf.from_dotlist(list(overrides))
    return OmegaConf.merge(cfg, override_conf)


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


def resolve_with_dataset_roots(paths: Sequence[str], cfg) -> List[str]:
    datasets_cfg = getattr(cfg, "preprocess", None)
    candidate_roots: List[Path] = []
    if datasets_cfg is not None and hasattr(datasets_cfg, "datasets"):
        dcfg = datasets_cfg.datasets
        for name in ("LibriSpeech", "LibriTTS"):
            if hasattr(dcfg, name):
                root = Path(getattr(dcfg, name).root)
                candidate_roots.append(root)

    resolved: List[str] = []
    for item in paths:
        p = Path(item)
        if p.is_absolute() and p.exists():
            resolved.append(str(p.resolve()))
            continue
        if p.exists():
            resolved.append(str(p.resolve()))
            continue
        matched = False
        for root in candidate_roots:
            candidate = root / p
            if candidate.exists():
                resolved.append(str(candidate.resolve()))
                matched = True
                break
        if not matched:
            print(f"[Warning] Unable to resolve path {item}. Skipping.")
    return resolved


class AudioDataset(Dataset):
    def __init__(self, paths: Sequence[str], target_sr: int, multiple_of: int, length_mode: str):
        assert length_mode in ("pad", "truncate"), "length_mode must be 'pad' or 'truncate'"
        self.paths = [str(Path(p)) for p in paths]
        self.target_sr = int(target_sr)
        self.multiple_of = int(multiple_of) if multiple_of else 1
        self.length_mode = length_mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.paths[idx]
        waveform, sr = torchaudio.load(path)
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform[:1, :]
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
        wav = waveform[0]
        orig_len = int(wav.shape[-1])
        if self.multiple_of > 0 and self.length_mode == "pad":
            wav_proc, _ = pad_to_multiple_1d(wav, self.multiple_of)
        elif self.multiple_of > 0 and self.length_mode == "truncate":
            proc_len = (orig_len // self.multiple_of) * self.multiple_of
            wav_proc = wav[:proc_len]
        else:
            wav_proc = wav

        proc_len = int(wav_proc.shape[-1])
        return {
            "wav": wav_proc,
            "path": path,
            "orig_length": orig_len,
            "proc_length": proc_len,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
        assert len(batch) == 1, "Batch size must be 1 for codec inference."
        example = batch[0]
        return {
            "wav": example["wav"].unsqueeze(0),
            "paths": [example["path"]],
            "orig_lengths": torch.tensor([example["orig_length"]], dtype=torch.long),
            "proc_lengths": torch.tensor([example["proc_length"]], dtype=torch.long),
        }


def mask_to_trailing(mask: torch.Tensor) -> torch.Tensor:
    seq = mask.to(torch.bool).squeeze(0)
    if seq.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=seq.device)
    zero_count = 0
    counts: List[int] = []
    for keep in reversed(seq.tolist()):
        if keep:
            counts.append(zero_count)
            zero_count = 0
        else:
            zero_count += 1
    counts.reverse()
    if not counts:
        return torch.zeros(0, dtype=torch.long, device=seq.device)
    return torch.tensor(counts, dtype=torch.long, device=seq.device)


@dataclass
class FileRecord:
    path: str
    token_start: int
    token_end: int
    orig_length: int
    proc_length: int


class CorpusAccumulator:
    def __init__(self, use_dtp: bool):
        self.use_dtp = use_dtp
        self.tokens_parts: List[np.ndarray] = []
        self.trailing_parts: List[np.ndarray] = [] if use_dtp else None
        self.file_records: List[FileRecord] = []
        self.offsets: List[int] = [0]
        self.max_trailing = 0

    def add(
        self,
        path: str,
        codes: np.ndarray,
        orig_length: int,
        proc_length: int,
        trailing: Optional[np.ndarray] = None,
    ) -> None:
        num_tokens = int(codes.size)
        self.tokens_parts.append(codes.astype(np.int32, copy=False))
        if self.use_dtp:
            assert trailing is not None
            trailing = trailing.astype(np.int32, copy=False)
            self.trailing_parts.append(trailing)
            if trailing.size > 0:
                self.max_trailing = max(self.max_trailing, int(trailing.max()))
        else:
            self.max_trailing = 0

        start = self.offsets[-1]
        end = start + num_tokens
        self.offsets.append(end)
        self.file_records.append(
            FileRecord(
                path=str(path),
                token_start=start,
                token_end=end,
                orig_length=int(orig_length),
                proc_length=int(proc_length),
            )
        )

    def finalize(self) -> Dict[str, object]:
        if self.tokens_parts:
            tokens = np.concatenate(self.tokens_parts)
        else:
            tokens = np.zeros(0, dtype=np.int32)
        trailing = None
        if self.use_dtp:
            if self.trailing_parts:
                trailing = np.concatenate(self.trailing_parts)
            else:
                trailing = np.zeros(0, dtype=np.int32)
        return {
            "tokens": tokens,
            "trailing": trailing,
            "offsets": np.asarray(self.offsets, dtype=np.int64),
            "file_records": [asdict(rec) for rec in self.file_records],
            "max_trailing_zero": int(self.max_trailing),
        }


def run_codec_forward(
    model: CodecLightningModule,
    wav: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    wav = wav.to(device=device, dtype=torch.float32)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    autocast_enabled = device_type == "cuda"
    ac_context = torch.autocast(device_type=device_type, dtype=dtype) if autocast_enabled else nullcontext()
    with torch.inference_mode():
        with ac_context:
            vq_emb = model.encoder(wav.unsqueeze(1), level=1)
            if model.use_dtp:
                dtp_out = model.dtp(vq_emb)
                if len(dtp_out) == 4:
                    mask, _, _, _ = dtp_out
                else:
                    mask, _, _ = dtp_out
                vq_emb, position_ids, cu_seqlens, max_seqlen = model.downsampler(vq_emb, mask)
            else:
                mask = None
                vq_emb = model.downsampler(vq_emb)
                position_ids = cu_seqlens = max_seqlen = None

            vq_emb = model.encoder(
                vq_emb,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                level=2,
            )
            _, vq_code, _ = model.decoder(vq_emb, vq=True)
    return vq_code.detach().cpu(), (mask.detach().cpu() if mask is not None else None)


def export_split(
    name: str,
    dataloader: DataLoader,
    model: CodecLightningModule,
    device: torch.device,
    dtype: torch.dtype,
    max_samples: Optional[int],
) -> Dict[str, object]:
    acc = CorpusAccumulator(use_dtp=model.use_dtp)
    iterator = tqdm(dataloader, desc=f"Extracting {name}", total=len(dataloader))
    processed = 0
    for batch in iterator:
        wav = batch["wav"].to(device)
        codes, mask = run_codec_forward(model, wav, device, dtype)
        codes_np = codes.squeeze(0).to(torch.long).numpy()
        trailing_np = None
        if model.use_dtp:
            assert mask is not None
            trailing = mask_to_trailing(mask)
            trailing_np = trailing.to(torch.long).numpy()
            if trailing_np.shape[0] != codes_np.shape[0]:
                raise RuntimeError(
                    f"Mismatch between kept codes ({codes_np.shape[0]}) and trailing counts ({trailing_np.shape[0]})"
                )
        acc.add(
            path=batch["paths"][0],
            codes=codes_np,
            orig_length=int(batch["orig_lengths"][0]),
            proc_length=int(batch["proc_lengths"][0]),
            trailing=trailing_np,
        )
        processed += 1
        if max_samples is not None and processed >= max_samples:
            break
    split_data = acc.finalize()
    print(
        f"[{name}] files={len(split_data['file_records'])} tokens={split_data['tokens'].size} "
        f"max_trailing={split_data['max_trailing_zero']}"
    )
    return split_data


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_npz(path: Path, data: Dict[str, object]) -> None:
    payload = {
        "tokens": data["tokens"],
        "offsets": data["offsets"],
    }
    if data["trailing"] is not None:
        payload["trailing"] = data["trailing"]
    np.savez_compressed(path, **payload)


def save_manifest(path: Path, records: List[Dict[str, object]]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract codec indices for LM evaluation.")
    parser.add_argument("--train_list", type=str, required=True, help="Text file (or dir) with training audio paths.")
    parser.add_argument("--test_list", type=str, required=True, help="Text file (or dir) with evaluation audio paths.")
    parser.add_argument("--run_dir", type=str, required=True, help="Hydra run directory containing config and checkpoint.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional explicit checkpoint path.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: <run_dir>/eval/lm_data).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--length_mode", type=str, choices=["pad", "truncate"], default="pad")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_files", type=int, default=None, help="Optional limit for training files.")
    parser.add_argument("--max_test_files", type=int, default=None, help="Optional limit for test files.")
    parser.add_argument(
        "--cfg_override",
        action="append",
        default=None,
        help="Hydra-style overrides, e.g. dataset.sample_rate=24000",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "hydra" / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Hydra config not found at {cfg_path}")
    ckpt_path = Path(args.ckpt).resolve() if args.ckpt else (run_dir / "pl_log" / "last.ckpt")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    cfg = OmegaConf.load(str(cfg_path))
    cfg = apply_cfg_overrides(cfg, args.cfg_override)

    train_paths = resolve_with_dataset_roots(parse_filelist(args.train_list), cfg)
    test_paths = resolve_with_dataset_roots(parse_filelist(args.test_list), cfg)
    if not train_paths:
        raise RuntimeError("No train files resolved.")
    if not test_paths:
        raise RuntimeError("No test files resolved.")

    target_sr = int(cfg.dataset.sample_rate)
    multiple_of = int(getattr(cfg.dataset, "multiple_of", 1) or 1)

    train_ds = AudioDataset(train_paths, target_sr, multiple_of, args.length_mode)
    test_ds = AudioDataset(test_paths, target_sr, multiple_of, args.length_mode)
    train_dl = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=AudioDataset.collate_fn,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=AudioDataset.collate_fn,
    )

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" and device.type == "cuda" else torch.float32

    model = CodecLightningModule(cfg=cfg).to(device)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    patch_legacy_dtp_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys while loading checkpoint: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys while loading checkpoint: {unexpected}")
    model.eval()

    train_split = export_split("train", train_dl, model, device, dtype, args.max_train_files)
    test_split = export_split("test", test_dl, model, device, dtype, args.max_test_files)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "eval" / "lm_data")
    ensure_dir(output_dir)

    train_npz = output_dir / "train_indices.npz"
    test_npz = output_dir / "test_indices.npz"
    train_manifest = output_dir / "train_manifest.jsonl"
    test_manifest = output_dir / "test_manifest.jsonl"

    save_npz(train_npz, train_split)
    save_npz(test_npz, test_split)
    save_manifest(train_manifest, train_split["file_records"])
    save_manifest(test_manifest, test_split["file_records"])

    metadata = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "config_path": str(cfg_path),
        "cfg_override": args.cfg_override,
        "length_mode": args.length_mode,
        "use_dtp": bool(model.use_dtp),
        "codebook_size": int(cfg.model.codec_decoder.codebook_size),
        "dataset": {
            "sample_rate": target_sr,
            "multiple_of": multiple_of,
        },
        "train_list": str(Path(args.train_list)),
        "test_list": str(Path(args.test_list)),
        "dtype": args.dtype,
        "device": args.device,
        "train": {
            "num_files": len(train_split["file_records"]),
            "num_tokens": int(train_split["tokens"].size),
            "max_trailing_zero": int(train_split["max_trailing_zero"]),
            "npz_path": str(train_npz),
            "manifest_path": str(train_manifest),
        },
        "test": {
            "num_files": len(test_split["file_records"]),
            "num_tokens": int(test_split["tokens"].size),
            "max_trailing_zero": int(test_split["max_trailing_zero"]),
            "npz_path": str(test_npz),
            "manifest_path": str(test_manifest),
        },
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved corpora and metadata under {output_dir}")


if __name__ == "__main__":
    main()

