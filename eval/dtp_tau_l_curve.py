#!/usr/bin/env python3
"""
Analyze DTMAE DTP behavior on LibriSpeech clean (or any input list):

1) Collect sequence-level similarity path length L distribution.
2) Sweep fixed tau values and visualize avg_r(tau).

The script loads one checkpoint once, computes encoder level-1 embeddings once,
then evaluates tau->avg_r quickly with a PLE-equivalent formula.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parent
DTMAE_ROOT = PROJECT_ROOT / "DTMAE"

for path in (EVAL_ROOT, PROJECT_ROOT, DTMAE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from dtp_stats import (  # noqa: E402
    AudioDataset,
    apply_cfg_overrides,
    parse_input_paths,
    patch_legacy_dtp_state_dict,
    patch_legacy_norm_state_dict,
    resolve_with_dataset_roots,
)
from DTMAE.lightning_module import CodecLightningModule  # noqa: E402


@dataclass
class SequencePathLength:
    path: str
    n_tokens: int
    l_value: float
    cumdist: np.ndarray  # shape [N], D[0]=0, D[-1]=L


def compute_cumdist_from_latent(x_btc: torch.Tensor) -> np.ndarray:
    """x_btc: [1, N, C] float tensor on any device."""
    if x_btc.ndim != 3 or x_btc.shape[0] != 1:
        raise ValueError(f"Expected x of shape [1, N, C], got {tuple(x_btc.shape)}")
    _, n_tokens, _ = x_btc.shape
    if n_tokens <= 0:
        return np.zeros((0,), dtype=np.float32)
    if n_tokens == 1:
        return np.zeros((1,), dtype=np.float32)

    sim = F.cosine_similarity(x_btc[:, 1:, :], x_btc[:, :-1, :], dim=-1)[0]  # [N-1]
    d = torch.zeros((n_tokens,), device=x_btc.device, dtype=torch.float32)
    d[1:] = (1.0 - sim).to(torch.float32)
    d = torch.clamp(d, min=0.0)
    cumdist = torch.cumsum(d, dim=0)
    cumdist[0] = 0.0
    return cumdist.detach().cpu().numpy().astype(np.float32)


def apply_max_span_constraint(mask: np.ndarray, max_s: Optional[int]) -> np.ndarray:
    """Mirror DTMAE _apply_max_span_constraint behavior for one sequence."""
    if max_s is None:
        return mask
    n_tokens = int(mask.shape[0])
    if n_tokens == 0:
        return mask

    stride = max(1, int(max_s))
    inserted = np.zeros_like(mask, dtype=bool)
    last_kept = -1
    for i in range(n_tokens):
        if mask[i]:
            last_kept = i
            continue
        run = i - last_kept
        if run > 0 and (run % stride == 0):
            inserted[i] = True
    return np.logical_or(mask, inserted)


def kept_count_from_cumdist(cumdist: np.ndarray, tau: float, max_s: Optional[int]) -> int:
    """
    Exact PLE-style frontier selection for one sequence using cumulative distance D.
    """
    n_tokens = int(cumdist.shape[0])
    if n_tokens <= 0:
        return 0

    mask = np.zeros((n_tokens,), dtype=bool)
    mask[0] = True

    if n_tokens > 1 and math.isfinite(tau) and tau > 0.0:
        l_val = float(cumdist[-1])
        m_raw = int(math.floor(l_val / tau))
        m = max(0, min(m_raw, n_tokens - 1))

        # Clamp-consistent fallback:
        # if requested boundaries exceed available N-1 slots, keep all boundaries.
        if m_raw > (n_tokens - 1):
            mask[1:] = True
            mask = apply_max_span_constraint(mask, max_s=max_s)
            return int(mask.sum())

        for k in range(1, m + 1):
            target = float(k) * float(tau)
            j = int(np.searchsorted(cumdist, target, side="left"))
            j = max(1, min(j, n_tokens - 1))
            mask[j] = True

    mask = apply_max_span_constraint(mask, max_s=max_s)
    return int(mask.sum())


def summarize(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
    }


def build_tau_grid(tau_min: float, tau_max: float, tau_step: float) -> np.ndarray:
    if tau_step <= 0.0:
        raise ValueError("--tau_step must be > 0")
    if tau_min <= 0.0 or tau_max <= 0.0:
        raise ValueError("--tau_min and --tau_max must be > 0")
    if tau_max < tau_min:
        raise ValueError("--tau_max must be >= --tau_min")

    count = int(math.floor((tau_max - tau_min) / tau_step)) + 1
    grid = tau_min + np.arange(count, dtype=np.float64) * tau_step
    if grid[-1] < tau_max - 1e-12:
        grid = np.append(grid, tau_max)
    return grid


def load_model_and_data(args) -> Tuple[CodecLightningModule, object, DataLoader, str, Dict[str, object]]:
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
        length_mode=str(args.length_mode),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=AudioDataset.collate_fn,
    )

    device = torch.device(args.device)
    model = CodecLightningModule(cfg=cfg).to(device).eval()

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

    dtp_cfg = getattr(getattr(cfg, "model", None), "resampler", None)
    dtp_cls = str(getattr(dtp_cfg, "dtp_cls", "")) if dtp_cfg is not None else ""
    dtp_params = getattr(dtp_cfg, "dtp_params", None)
    max_s = getattr(dtp_params, "max_s", None) if dtp_params is not None else None
    dtp_info = {
        "run_dir": str(run_dir),
        "cfg_path": str(cfg_path),
        "ckpt_path": str(ckpt_path),
        "resolved_input_count": len(input_paths),
        "dtp_cls": dtp_cls,
        "max_s": (None if max_s is None else int(max_s)),
        "length_mode": str(args.length_mode),
    }
    return model, cfg, dataloader, str(device), dtp_info


def collect_sequence_lengths(
    model: CodecLightningModule,
    dataloader: DataLoader,
    device: str,
    max_samples: Optional[int],
) -> List[SequencePathLength]:
    records: List[SequencePathLength] = []
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    with torch.inference_mode():
        iterator = tqdm(dataloader, total=len(dataloader), desc="Collect L distribution")
        for batch in iterator:
            wav = batch["wav"].to(device)
            path = str(batch["paths"][0])

            ac = (
                torch.autocast(device_type=device_type, dtype=torch.bfloat16)
                if device_type == "cuda"
                else nullcontext()
            )
            with ac:
                x = model.encoder(wav.unsqueeze(1), level=1)

            cumdist = compute_cumdist_from_latent(x)
            n_tokens = int(cumdist.shape[0])
            l_val = float(cumdist[-1]) if n_tokens > 0 else 0.0
            records.append(SequencePathLength(path=path, n_tokens=n_tokens, l_value=l_val, cumdist=cumdist))

            if max_samples is not None and len(records) >= int(max_samples):
                break

    return records


def sweep_tau_avg_r(
    seq_records: Sequence[SequencePathLength],
    tau_grid: np.ndarray,
    max_s: Optional[int],
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    total_tokens_all = float(sum(int(r.n_tokens) for r in seq_records))
    if total_tokens_all <= 0:
        raise RuntimeError("No tokens found in sequence records.")

    for tau in tqdm(tau_grid, total=len(tau_grid), desc="Sweep tau->avg_r"):
        per_sample_r: List[float] = []
        zeros_total = 0
        token_total = 0

        for rec in seq_records:
            n_tokens = int(rec.n_tokens)
            if n_tokens <= 0:
                continue
            kept = kept_count_from_cumdist(rec.cumdist, float(tau), max_s=max_s)
            masked = int(n_tokens - kept)
            zeros_total += masked
            token_total += n_tokens
            per_sample_r.append(float(masked) / float(max(1, n_tokens)))

        arr = np.asarray(per_sample_r, dtype=np.float64)
        weighted_avg_r = float(zeros_total) / float(max(1, token_total))
        results.append(
            {
                "tau": float(tau),
                "avg_r_weighted": weighted_avg_r,
                "avg_r_sample_mean": float(np.mean(arr)) if arr.size > 0 else float("nan"),
                "avg_r_q10": float(np.quantile(arr, 0.10)) if arr.size > 0 else float("nan"),
                "avg_r_q50": float(np.quantile(arr, 0.50)) if arr.size > 0 else float("nan"),
                "avg_r_q90": float(np.quantile(arr, 0.90)) if arr.size > 0 else float("nan"),
                "avg_r_std": float(np.std(arr, ddof=0)) if arr.size > 0 else float("nan"),
                "num_sequences": int(arr.size),
                "num_tokens": int(token_total),
            }
        )
    return results


def eval_per_seq_formula_avg_r(
    seq_records: Sequence[SequencePathLength],
    target_r: float,
    max_s: Optional[int],
) -> Dict[str, float]:
    """Evaluate realized avg_r for tau_b = (L_b / N_b) / (1 - target_r)."""
    if not (0.0 <= float(target_r) <= 1.0):
        raise ValueError("target_r must be in [0, 1]")

    per_sample_r: List[float] = []
    tau_vals: List[float] = []
    zeros_total = 0
    token_total = 0

    for rec in seq_records:
        n_tokens = int(rec.n_tokens)
        if n_tokens <= 0:
            continue
        l_val = float(rec.l_value)
        denom = max(1e-8, 1.0 - float(target_r))
        tau_b = (l_val / float(max(1, n_tokens))) / denom
        if tau_b <= 0.0:
            tau_b = 1e-12

        kept = kept_count_from_cumdist(rec.cumdist, tau=float(tau_b), max_s=max_s)
        masked = int(n_tokens - kept)
        zeros_total += masked
        token_total += n_tokens
        per_sample_r.append(float(masked) / float(max(1, n_tokens)))
        tau_vals.append(float(tau_b))

    arr_r = np.asarray(per_sample_r, dtype=np.float64)
    arr_tau = np.asarray(tau_vals, dtype=np.float64)
    return {
        "target_r": float(target_r),
        "avg_r_weighted": float(zeros_total) / float(max(1, token_total)),
        "avg_r_sample_mean": float(np.mean(arr_r)) if arr_r.size > 0 else float("nan"),
        "avg_r_q10": float(np.quantile(arr_r, 0.10)) if arr_r.size > 0 else float("nan"),
        "avg_r_q50": float(np.quantile(arr_r, 0.50)) if arr_r.size > 0 else float("nan"),
        "avg_r_q90": float(np.quantile(arr_r, 0.90)) if arr_r.size > 0 else float("nan"),
        "tau_b_mean": float(np.mean(arr_tau)) if arr_tau.size > 0 else float("nan"),
        "tau_b_q10": float(np.quantile(arr_tau, 0.10)) if arr_tau.size > 0 else float("nan"),
        "tau_b_q50": float(np.quantile(arr_tau, 0.50)) if arr_tau.size > 0 else float("nan"),
        "tau_b_q90": float(np.quantile(arr_tau, 0.90)) if arr_tau.size > 0 else float("nan"),
        "num_sequences": int(arr_r.size),
        "num_tokens": int(token_total),
    }


def build_target_r_grid(r_min: float, r_max: float, r_step: float) -> np.ndarray:
    if r_step <= 0.0:
        raise ValueError("--formula_r_step must be > 0")
    if r_max < r_min:
        raise ValueError("--formula_r_max must be >= --formula_r_min")
    r_min = max(0.0, min(1.0, float(r_min)))
    r_max = max(0.0, min(1.0, float(r_max)))
    count = int(math.floor((r_max - r_min) / r_step)) + 1
    grid = r_min + np.arange(count, dtype=np.float64) * r_step
    if grid[-1] < r_max - 1e-12:
        grid = np.append(grid, r_max)
    grid = np.clip(grid, 0.0, 1.0)
    return np.unique(np.round(grid, 10))


def eval_formula_policy_sweep(
    seq_records: Sequence[SequencePathLength],
    target_r_grid: np.ndarray,
    max_s: Optional[int],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for r_t in target_r_grid.tolist():
        out = eval_per_seq_formula_avg_r(seq_records=seq_records, target_r=float(r_t), max_s=max_s)
        row = {
            "target_r": float(r_t),
            "avg_r_weighted": float(out["avg_r_weighted"]),
            "avg_r_sample_mean": float(out["avg_r_sample_mean"]),
            "avg_r_q10": float(out["avg_r_q10"]),
            "avg_r_q50": float(out["avg_r_q50"]),
            "avg_r_q90": float(out["avg_r_q90"]),
            "tau_b_mean": float(out["tau_b_mean"]),
            "tau_b_q10": float(out["tau_b_q10"]),
            "tau_b_q50": float(out["tau_b_q50"]),
            "tau_b_q90": float(out["tau_b_q90"]),
        }
        rows.append(row)
    return rows


def maybe_plot(
    output_path: Path,
    l_values: Sequence[float],
    tau_results: Sequence[Dict[str, float]],
    target_avg_r: Optional[float],
    formula_curve: Optional[Sequence[Dict[str, float]]],
    bins: int,
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Warning] matplotlib unavailable, skip plotting: {exc}")
        return None

    tau = np.asarray([r["tau"] for r in tau_results], dtype=np.float64)
    avg_r = np.asarray([r["avg_r_weighted"] for r in tau_results], dtype=np.float64)
    q10 = np.asarray([r["avg_r_q10"] for r in tau_results], dtype=np.float64)
    q90 = np.asarray([r["avg_r_q90"] for r in tau_results], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(np.asarray(l_values, dtype=np.float64), bins=int(bins), color="#2E8B57", alpha=0.85)
    axes[0].set_title("Distribution of L (sequence path length)")
    axes[0].set_xlabel("L")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    axes[1].plot(tau, avg_r, color="#1f77b4", linewidth=2.0, label="weighted avg_r")
    axes[1].fill_between(tau, q10, q90, color="#1f77b4", alpha=0.20, label="per-seq r q10-q90")
    if target_avg_r is not None:
        axes[1].axhline(float(target_avg_r), color="#d62728", linestyle="--", linewidth=1.2, label="target_avg_r")
    if formula_curve:
        f_tau = np.asarray([row["tau_b_mean"] for row in formula_curve], dtype=np.float64)
        f_avg = np.asarray([row["avg_r_weighted"] for row in formula_curve], dtype=np.float64)
        valid = np.isfinite(f_tau) & np.isfinite(f_avg)
        if np.any(valid):
            f_tau = f_tau[valid]
            f_avg = f_avg[valid]
            order = np.argsort(f_tau)
            axes[1].plot(
                f_tau[order],
                f_avg[order],
                color="#ff7f0e",
                linestyle=":",
                linewidth=2.0,
                marker="o",
                markersize=3,
                label="formula sweep: tau_b=(L/N)/(1-target_r)",
            )
    axes[1].set_title("avg_r vs fixed tau")
    axes[1].set_xlabel("tau")
    axes[1].set_ylabel("avg_r")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[1].legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def write_tau_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tau",
        "avg_r_weighted",
        "avg_r_sample_mean",
        "avg_r_q10",
        "avg_r_q50",
        "avg_r_q90",
        "avg_r_std",
        "num_sequences",
        "num_tokens",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_formula_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_r",
        "avg_r_weighted",
        "avg_r_sample_mean",
        "avg_r_q10",
        "avg_r_q50",
        "avg_r_q90",
        "tau_b_mean",
        "tau_b_q10",
        "tau_b_q50",
        "tau_b_q90",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize L distribution and tau->avg_r for DTMAE DTP.")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing hydra/config.yaml and pl_log/last.ckpt")
    parser.add_argument(
        "--input",
        type=str,
        default="DTMAE/filelists/librispeech_test_clean.txt",
        help="Directory, single file, or .txt filelist (default: LibriSpeech test-clean filelist)",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Default: <run_dir>/eval/dtp_tau_l_curve")
    parser.add_argument("--tau_min", type=float, default=0.001)
    parser.add_argument("--tau_max", type=float, default=1.0)
    parser.add_argument("--tau_step", type=float, default=0.01)
    parser.add_argument("--target_avg_r", type=float, default=None, help="Optional horizontal reference line")
    parser.add_argument("--formula_r_min", type=float, default=0.1, help="Min target_r for formula sweep")
    parser.add_argument("--formula_r_max", type=float, default=0.9, help="Max target_r for formula sweep")
    parser.add_argument("--formula_r_step", type=float, default=0.1, help="Step for formula target_r sweep")
    parser.add_argument("--length_mode", type=str, choices=["pad", "truncate"], default="pad")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bins", type=int, default=80, help="Histogram bins for L plot")
    parser.add_argument("--save_l_npy", action="store_true", help="Save raw L values as l_values.npy")
    parser.add_argument(
        "--cfg_override",
        action="append",
        default=None,
        help="Hydra-style dotlist override, repeatable (e.g., --cfg_override model.resampler.dtp_params.r=0.5)",
    )
    args = parser.parse_args()

    model, _cfg, dataloader, device, dtp_info = load_model_and_data(args)

    dtp_cls = str(dtp_info.get("dtp_cls", ""))
    if "PLEBatchTopK" not in dtp_cls:
        raise RuntimeError(
            f"This script currently supports PLEBatchTopK-family selectors only. Current dtp_cls={dtp_cls!r}."
        )

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (Path(args.run_dir).resolve() / "eval" / "dtp_tau_l_curve")
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_records = collect_sequence_lengths(
        model=model,
        dataloader=dataloader,
        device=device,
        max_samples=args.max_samples,
    )
    if not seq_records:
        raise RuntimeError("No sequence records were collected.")

    l_values = [float(r.l_value) for r in seq_records]
    tau_grid = build_tau_grid(float(args.tau_min), float(args.tau_max), float(args.tau_step))

    max_s_val = dtp_info.get("max_s")
    if max_s_val is not None:
        if isinstance(max_s_val, (int, float, str)):
            max_s_val = int(max_s_val)
        else:
            max_s_val = None

    tau_results = sweep_tau_avg_r(
        seq_records=seq_records,
        tau_grid=tau_grid,
        max_s=max_s_val,
    )

    formula_r_grid = build_target_r_grid(
        r_min=float(args.formula_r_min),
        r_max=float(args.formula_r_max),
        r_step=float(args.formula_r_step),
    )
    formula_sweep = eval_formula_policy_sweep(
        seq_records=seq_records,
        target_r_grid=formula_r_grid,
        max_s=max_s_val,
    )

    csv_path = output_dir / "tau_avg_r_curve.csv"
    write_tau_csv(csv_path, tau_results)
    formula_csv_path = output_dir / "formula_r_sweep.csv"
    write_formula_csv(formula_csv_path, formula_sweep)

    l_jsonl_path = output_dir / "l_per_sequence.jsonl"
    with open(l_jsonl_path, "w") as f:
        for rec in seq_records:
            f.write(
                json.dumps(
                    {
                        "path": rec.path,
                        "n_tokens": rec.n_tokens,
                        "l_value": rec.l_value,
                    }
                )
                + "\n"
            )

    if args.save_l_npy:
        np.save(output_dir / "l_values.npy", np.asarray(l_values, dtype=np.float32))

    plot_path = maybe_plot(
        output_path=output_dir / "l_distribution_and_tau_avg_r.png",
        l_values=l_values,
        tau_results=tau_results,
        target_avg_r=args.target_avg_r,
        formula_curve=formula_sweep,
        bins=args.bins,
    )

    weighted_curve = np.asarray([r["avg_r_weighted"] for r in tau_results], dtype=np.float64)
    tau_vals = np.asarray([r["tau"] for r in tau_results], dtype=np.float64)
    best_idx = None
    if args.target_avg_r is not None and weighted_curve.size > 0:
        best_idx = int(np.argmin(np.abs(weighted_curve - float(args.target_avg_r))))

    summary = {
        "dtp_info": dtp_info,
        "num_sequences": int(len(seq_records)),
        "num_tokens_total": int(sum(r.n_tokens for r in seq_records)),
        "l_summary": summarize(l_values),
        "tau_grid": {
            "tau_min": float(args.tau_min),
            "tau_max": float(args.tau_max),
            "tau_step": float(args.tau_step),
            "num_points": int(len(tau_grid)),
        },
        "target_avg_r": (None if args.target_avg_r is None else float(args.target_avg_r)),
        "closest_tau_to_target": (
            None
            if best_idx is None
            else {
                "tau": float(tau_vals[best_idx]),
                "avg_r_weighted": float(weighted_curve[best_idx]),
                "abs_error": float(abs(weighted_curve[best_idx] - float(args.target_avg_r))),
            }
        ),
        "formula_tau_policy": "tau_b = (L_b / N_b) / (1 - target_r)",
        "formula_target_r_grid": {
            "r_min": float(args.formula_r_min),
            "r_max": float(args.formula_r_max),
            "r_step": float(args.formula_r_step),
            "num_points": int(len(formula_r_grid)),
        },
        "formula_r_sweep": formula_sweep,
        "outputs": {
            "tau_curve_csv": str(csv_path),
            "formula_r_sweep_csv": str(formula_csv_path),
            "l_per_sequence_jsonl": str(l_jsonl_path),
            "plot_png": plot_path,
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
