#!/usr/bin/env python3
"""
Collect detailed statistics about Dynamic Token Pooling (DTP) masks for DTMAE models.

The script mirrors the dataset pipeline from eval.py: it loads a trained checkpoint,
iterates over an input file list (or directory), executes the full generator forward
pass, and aggregates statistics about the mask, tau values, and average masking ratio.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parent
DTMAE_ROOT = PROJECT_ROOT / "DTMAE"

for path in (EVAL_ROOT, PROJECT_ROOT, DTMAE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from DTMAE.lightning_module import CodecLightningModule  # noqa: E402

ALLOWED_AUDIO_EXTS = {".wav", ".flac"}


def read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]


def pad_to_multiple_1d(waveform: torch.Tensor, multiple_of: int) -> Tuple[torch.Tensor, int]:
    length = waveform.shape[-1]
    if multiple_of <= 0 or length % multiple_of == 0:
        return waveform, length
    pad_len = multiple_of - (length % multiple_of)
    padded = torch.nn.functional.pad(waveform, (0, pad_len))
    return padded, length


def parse_input_paths(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_dir():
        files = [
            str(fp.resolve())
            for fp in p.rglob("*")
            if fp.is_file() and fp.suffix.lower() in ALLOWED_AUDIO_EXTS
        ]
        files.sort()
        return files
    if p.is_file():
        if p.suffix.lower() == ".txt":
            return [str(Path(x).as_posix()) for x in read_lines(str(p))]
        if p.suffix.lower() in ALLOWED_AUDIO_EXTS:
            return [str(p.resolve())]
    raise FileNotFoundError(
        f"Invalid --input: {input_path}. Provide a directory, a .txt filelist, or a single audio file."
    )


def apply_cfg_overrides(cfg, overrides: Optional[List[str]]):
    if not overrides:
        return cfg
    override_conf = OmegaConf.from_dotlist(overrides)
    return OmegaConf.merge(cfg, override_conf)


def patch_legacy_dtp_state_dict(state_dict: Dict[str, torch.Tensor]) -> None:
    legacy_keys = ["dtp.log_tau", "dtp.r_ema", "dtp.steps"]
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


def patch_legacy_norm_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, int]:
    """
    Compat patch for older checkpoints where RMSNorm parameters were saved as
    `<module>.weight` (no bias), while current code expects
    `<module>.norm.weight` / `<module>.norm.bias`.
    """
    remapped_norm_weights = 0
    added_norm_biases = 0
    added_optional_defaults = 0

    for old_key in list(state_dict.keys()):
        if not old_key.endswith(".weight"):
            continue

        stem = old_key[: -len(".weight")]
        new_weight_key = f"{stem}.norm.weight"
        if new_weight_key not in model_state_dict or new_weight_key in state_dict:
            continue

        state_dict[new_weight_key] = state_dict.pop(old_key)
        remapped_norm_weights += 1

        new_bias_key = f"{stem}.norm.bias"
        if new_bias_key in model_state_dict and new_bias_key not in state_dict:
            state_dict[new_bias_key] = torch.zeros_like(model_state_dict[new_bias_key])
            added_norm_biases += 1

    # Older checkpoints may not contain this currently-unused projection.
    for key in ("encoder.proj.weight", "encoder.proj.bias"):
        if key in model_state_dict and key not in state_dict:
            state_dict[key] = model_state_dict[key].clone()
            added_optional_defaults += 1

    return {
        "remapped_norm_weights": remapped_norm_weights,
        "added_norm_biases": added_norm_biases,
        "added_optional_defaults": added_optional_defaults,
    }


def pick_tau_value(state: Dict[str, Optional[float]], prefer_eval: bool) -> Optional[float]:
    if not state:
        return None
    order = ("tau_eval", "tau_train") if prefer_eval else ("tau_train", "tau_eval")
    for key in order:
        val = state.get(key)
        if val is not None:
            return float(val)
    return None


def resolve_with_dataset_roots(paths: List[str], cfg) -> List[str]:
    roots: List[Path] = []
    datasets_cfg = getattr(cfg.preprocess, "datasets", None)
    if datasets_cfg is not None:
        for name in ("LibriSpeech", "LibriTTS"):
            if hasattr(datasets_cfg, name):
                roots.append(Path(getattr(datasets_cfg, name).root))

    resolved: List[str] = []
    for p in paths:
        pp = Path(p)
        if pp.is_absolute() and pp.exists():
            resolved.append(str(pp.resolve()))
            continue
        if pp.exists():
            resolved.append(str(pp.resolve()))
            continue
        found = False
        for root in roots:
            candidate = root / p
            if candidate.exists():
                resolved.append(str(candidate.resolve()))
                found = True
                break
        if not found:
            print(f"[Warning] Input path not found: {p}. Skipping.")
    return resolved


class AudioDataset(Dataset):
    def __init__(self, paths: List[str], target_sr: int, multiple_of: int, length_mode: str):
        assert length_mode in ("pad", "truncate"), "length_mode must be 'pad' or 'truncate'"
        self.paths = [str(Path(p)) for p in paths]
        self.target_sr = int(target_sr)
        self.multiple_of = int(multiple_of) if multiple_of is not None else 1
        self.length_mode = length_mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.paths[idx]
        wav, sr = torchaudio.load(path)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav[:1, :]
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if sr != self.target_sr:
            wav = torchaudio.transforms.Resample(sr, self.target_sr)(wav)
        wav_1d = wav[0]
        orig_len = int(wav_1d.shape[-1])

        if self.length_mode == "pad":
            wav_proc, _ = pad_to_multiple_1d(wav_1d, self.multiple_of)
            proc_len = int(wav_proc.shape[-1])
        else:
            proc_len = (
                (orig_len // self.multiple_of) * self.multiple_of if self.multiple_of > 0 else orig_len
            )
            wav_proc = wav_1d[:proc_len]
        return {"wav": wav_proc, "path": path, "orig_length": orig_len, "proc_length": proc_len}

    @staticmethod
    def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
        assert len(batch) == 1, "Batch size must be 1 for this model."
        b = batch[0]
        return {
            "wav": b["wav"].unsqueeze(0),
            "paths": [b["path"]],
            "orig_lengths": torch.tensor([b["orig_length"]], dtype=torch.long),
            "proc_lengths": torch.tensor([b["proc_length"]], dtype=torch.long),
        }


class MaskStatsAggregator:
    def __init__(self):
        self.total_sequences = 0
        self.total_tokens = 0
        self.total_kept_tokens = 0
        self.orig_lengths: List[int] = []
        self.kept_lengths: List[int] = []
        self.zero_run_lengths: List[int] = []
        self.avg_r_values: List[float] = []
        self.kept_length_counter: Counter = Counter()
        self.zero_run_counter: Counter = Counter()

    @staticmethod
    def _zero_runs(seq: np.ndarray) -> List[int]:
        runs: List[int] = []
        keep_idx = np.flatnonzero(seq)
        if keep_idx.size == 0:
            total = int(seq.size)
            if total > 0:
                runs.append(total)
            return runs

        if keep_idx.size > 1:
            gaps = keep_idx[1:] - keep_idx[:-1] - 1
            runs.extend(gaps.astype(int).tolist())

        tail = int(seq.size - keep_idx[-1] - 1)
        if tail > 0:
            runs.append(tail)
        return runs

    def update(
        self,
        mask: torch.Tensor,
        avg_r: float,
        tau: float,
        paths: List[str],
        orig_lengths: Iterable[int],
        proc_lengths: Iterable[int],
    ) -> List[Dict[str, object]]:
        mask_np = mask.detach().to("cpu").numpy().astype(bool)
        avg_r_val = float(avg_r)
        self.avg_r_values.append(avg_r_val)

        records: List[Dict[str, object]] = []
        for idx, seq_mask in enumerate(mask_np):
            keep = int(seq_mask.sum())
            total = int(seq_mask.size)
            self.total_sequences += 1
            self.total_kept_tokens += keep
            self.total_tokens += total
            self.orig_lengths.append(total)
            self.kept_lengths.append(keep)
            self.kept_length_counter[keep] += 1

            zero_runs = self._zero_runs(seq_mask)
            for zr in zero_runs:
                self.zero_run_counter[zr] += 1
            self.zero_run_lengths.extend(zero_runs)

            record = {
                "path": paths[idx] if idx < len(paths) else None,
                "orig_length_tokens": total,
                "dtp_length_tokens": keep,
                "kept_ratio": (keep / total) if total > 0 else None,
                "reduction_ratio": (1.0 - keep / total) if total > 0 else None,
                "avg_r": avg_r_val,
                "tau_used": float(tau),
                "orig_length_samples": int(orig_lengths[idx]) if idx < len(orig_lengths) else None,
                "proc_length_samples": int(proc_lengths[idx]) if idx < len(proc_lengths) else None,
                "max_zero_run": max(zero_runs) if zero_runs else 0,
            }
            records.append(record)
        return records

    @staticmethod
    def _summarize(values: List[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=0)),
        }

    def summary(self) -> Dict[str, object]:
        kept_ratio = (
            (self.total_kept_tokens / max(1, self.total_tokens)) if self.total_tokens > 0 else None
        )
        zero_total = sum(self.zero_run_counter.values())
        zero_dist = [
            {
                "length": int(length),
                "count": int(count),
                "ratio": float(count / zero_total) if zero_total > 0 else 0.0,
            }
            for length, count in sorted(self.zero_run_counter.items())
        ]

        kept_dist_total = sum(self.kept_length_counter.values())
        kept_length_dist = [
            {
                "length": int(length),
                "count": int(count),
                "ratio": float(count / kept_dist_total) if kept_dist_total > 0 else 0.0,
            }
            for length, count in sorted(self.kept_length_counter.items())
        ]

        return {
            "num_sequences": self.total_sequences,
            "total_tokens": self.total_tokens,
            "total_kept_tokens": self.total_kept_tokens,
            "overall_kept_ratio": kept_ratio,
            "avg_r_summary": self._summarize(self.avg_r_values),
            "orig_length_summary": self._summarize(self.orig_lengths),
            "dtp_length_summary": self._summarize(self.kept_lengths),
            "zero_run_summary": self._summarize(self.zero_run_lengths),
            "zero_run_distribution": zero_dist,
            "dtp_length_distribution": kept_length_dist,
        }


def maybe_plot_histogram(
    data: List[int],
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 50,
) -> Optional[str]:
    if not data:
        return None
    try:
        import matplotlib.pyplot as plt  # local import to keep dependency optional
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[Warning] matplotlib not available, skipping plot '{title}': {exc}")
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=bins, color="#2E8B57", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def get_tau_state(dtp_module: torch.nn.Module) -> Dict[str, Optional[float]]:
    state: Dict[str, Optional[float]] = {}
    for attr in ("tau_train", "tau_eval"):
        if hasattr(dtp_module, attr):
            buf = getattr(dtp_module, attr)
            if torch.is_tensor(buf):
                state[attr] = float(buf.detach().cpu().item())
            else:
                state[attr] = float(buf)
    if hasattr(dtp_module, "fixed_tau") and dtp_module.fixed_tau is not None:
        state["fixed_tau"] = float(dtp_module.fixed_tau)
    return state


def run_generator_forward(
    model: CodecLightningModule,
    wav: torch.Tensor,
    device_type: str,
) -> Tuple[torch.Tensor, float, float]:
    # Reproduce CodecLightningModule.forward but also expose the boolean mask.
    with torch.inference_mode():
        autocast_enabled = device_type == "cuda"
        autocast_dtype = torch.bfloat16 if autocast_enabled else None
        ac_context = (
            torch.autocast(device_type=device_type, dtype=autocast_dtype) if autocast_enabled else nullcontext()
        )
        with ac_context:
            vq_emb = model.encoder(wav.unsqueeze(1), level=1)
            dtp_out = model.dtp(vq_emb)
            if len(dtp_out) == 4:
                mask, avg_r, tau_used, _ = dtp_out
            else:
                mask, avg_r, tau_used = dtp_out
            vq_emb, position_ids, cu_seqlens, max_seqlen = model.downsampler(vq_emb, mask)
            vq_emb = model.encoder(
                vq_emb,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                level=2,
            )
            vq_post_emb, _, _ = model.decoder(vq_emb, vq=True)
            vq_post_emb = model.decoder(
                vq_post_emb,
                vq=False,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                level=2,
            )
            vq_post_emb = model.upsampler(vq_post_emb, mask)
            _ = model.decoder(vq_post_emb, vq=False, level=1)
    return mask, float(avg_r.detach().cpu().item()), float(tau_used.detach().cpu().item())


def collect_dtp_stats(args) -> None:
    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "hydra" / "config.yaml"
    ckpt_path = run_dir / "pl_log" / "last.ckpt"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found at {cfg_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    cfg = OmegaConf.load(str(cfg_path))
    cfg = apply_cfg_overrides(cfg, args.cfg_override)
    raw_paths = parse_input_paths(args.input)
    input_paths = resolve_with_dataset_roots(raw_paths, cfg)
    if not input_paths:
        raise RuntimeError("No valid input paths resolved.")

    dataset = AudioDataset(
        input_paths,
        target_sr=int(cfg.dataset.sample_rate),
        multiple_of=int(cfg.dataset.multiple_of),
        length_mode=args.length_mode,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=AudioDataset.collate_fn,
    )

    device = torch.device(args.device)
    model = CodecLightningModule(cfg=cfg).to(device)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    patch_legacy_dtp_state_dict(state_dict)
    compat_stats = patch_legacy_norm_state_dict(state_dict, model.state_dict())
    if any(compat_stats.values()):
        print(
            "[Compat] Applied legacy checkpoint patch: "
            f"remapped_norm_weights={compat_stats['remapped_norm_weights']}, "
            f"added_norm_biases={compat_stats['added_norm_biases']}, "
            f"added_optional_defaults={compat_stats['added_optional_defaults']}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[Warning] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"[Warning] Missing examples: {missing[:8]}")
        if unexpected:
            print(f"[Warning] Unexpected examples: {unexpected[:8]}")
    if not getattr(model, "use_dtp", False):
        raise RuntimeError("The loaded model does not enable DTP (use_dtp=False).")

    dtp_module = model.dtp
    model.eval()

    output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "eval" / "dtp_stats")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    records_path = output_dir / "per_sequence.jsonl"

    tau_state_before = get_tau_state(dtp_module)
    device_type = device.type

    aggregator = MaskStatsAggregator()
    processed = 0

    with open(records_path, "w") as record_file:
        for batch in tqdm(dataloader, total=len(dataloader), desc="Collecting DTP stats"):
            wav = batch["wav"].to(device)
            mask, avg_r_val, tau_val = run_generator_forward(model, wav, device_type)
            records = aggregator.update(
                mask=mask,
                avg_r=avg_r_val,
                tau=tau_val,
                paths=batch["paths"],
                orig_lengths=batch["orig_lengths"].tolist(),
                proc_lengths=batch["proc_lengths"].tolist(),
            )
            for rec in records:
                record_file.write(json.dumps(rec) + "\n")

            processed += len(records)
            if args.max_samples is not None and processed >= args.max_samples:
                break

    tau_state_after = get_tau_state(dtp_module)
    summary = {
        "device": str(device),
        "input_count": len(dataset),
        "num_processed_sequences": processed,
        "records_path": str(records_path),
        "cfg_overrides": args.cfg_override,
        "tau_state_before": tau_state_before,
        "tau_state_after": tau_state_after,
    }
    prefer_eval_tau = bool(getattr(dtp_module, "update_test_time", False))
    tau_start = pick_tau_value(tau_state_before, prefer_eval_tau)
    tau_end = pick_tau_value(tau_state_after, prefer_eval_tau)
    summary["tau_progress"] = {"start": tau_start, "end": tau_end}
    summary.update(aggregator.summary())

    if not args.no_plots:
        kept_plot = maybe_plot_histogram(
            aggregator.kept_lengths,
            title="Kept token lengths after DTP",
            xlabel="Kept tokens",
            output_path=output_dir / "dtp_length_hist.png",
        )
        zero_plot = maybe_plot_histogram(
            aggregator.zero_run_lengths,
            title="Zero-run lengths (consecutive masked tokens)",
            xlabel="Length",
            output_path=output_dir / "zero_run_hist.png",
        )
        summary["plots"] = {"dtp_length_hist": kept_plot, "zero_run_hist": zero_plot}

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect mask/tau statistics for DTMAE DTP modules.")
    parser.add_argument("--input", type=str, required=True, help="Directory, single file, or .txt filelist.")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing hydra/config.yaml and pl_log/last.ckpt.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--length_mode", type=str, choices=["pad", "truncate"], default="pad")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional maximum number of sequences to process.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to store summary/plots (default: run_dir/eval/dtp_stats).")
    parser.add_argument("--no_plots", action="store_true", help="Disable histogram generation even if matplotlib is available.")
    parser.add_argument(
        "--cfg_override",
        action="append",
        default=None,
        help="Hydra-style dotlist override applied after loading hydra/config.yaml "
        "(e.g., --cfg_override dataset.multiple_of=160). Use multiple flags for multiple overrides.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    collect_dtp_stats(args)


if __name__ == "__main__":
    main()
