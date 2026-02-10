from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtp.ops import BatchGreedy, BatchTopK, PLEBatchTopK
from dtp.random_mask_candidates import MaskCandidate, build_default_candidates, sample_candidate_mask


@dataclass
class SimilarityProfile:
    name: str
    freq_min: float
    freq_max: float
    components_min: int
    components_max: int
    bias: float
    scale: float
    noise_std: float


def default_profiles() -> List[SimilarityProfile]:
    return [
        SimilarityProfile("low_freq", 0.4, 2.0, 2, 4, 0.95, 1.2, 0.05),
        SimilarityProfile("mid_freq", 2.0, 6.0, 2, 5, 0.55, 1.3, 0.08),
        SimilarityProfile("high_freq", 6.0, 16.0, 3, 6, 0.25, 1.4, 0.10),
        SimilarityProfile("mixed", 0.5, 12.0, 3, 7, 0.45, 1.5, 0.10),
    ]


def _sample_similarity_sequence(profile: SimilarityProfile, seq_len: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 1.0, seq_len - 1, endpoint=True)
    n_comp = int(rng.integers(profile.components_min, profile.components_max + 1))

    amps = rng.dirichlet(np.ones(n_comp))
    raw = np.zeros(seq_len - 1, dtype=np.float64)
    for i in range(n_comp):
        f = float(rng.uniform(profile.freq_min, profile.freq_max))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        raw += amps[i] * np.sin(2.0 * math.pi * f * t + phase)

    trend = 0.25 * np.sin(2.0 * math.pi * 0.5 * t + float(rng.uniform(0.0, 2.0 * math.pi)))
    raw = raw + trend + rng.normal(0.0, profile.noise_std, size=seq_len - 1)

    s = np.tanh(profile.scale * raw + profile.bias)
    s = np.clip(s, -0.98, 0.999)
    return s.astype(np.float32)


def _orthonormal_columns(c_dim: int, rng: np.random.Generator) -> np.ndarray:
    g = rng.normal(size=(c_dim, 2)).astype(np.float64)
    q, _ = np.linalg.qr(g)
    return q[:, :2]


def _similarity_to_embedding(s: np.ndarray, c_dim: int, rng: np.random.Generator) -> np.ndarray:
    n = s.shape[0] + 1
    delta = np.arccos(np.clip(s.astype(np.float64), -1.0, 1.0))
    signs = rng.choice([-1.0, 1.0], size=n - 1)
    theta = np.zeros(n, dtype=np.float64)
    theta[0] = float(rng.uniform(0.0, 2.0 * math.pi))
    theta[1:] = theta[0] + np.cumsum(signs * delta)

    uv = np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # [N, 2]
    proj = _orthonormal_columns(c_dim, rng)  # [C, 2]
    x = uv @ proj.T  # [N, C]
    return x.astype(np.float32)


def build_synthetic_batches(
    num_batches: int,
    batch_size: int,
    seq_len: int,
    c_dim: int,
    profiles: Sequence[SimilarityProfile],
    rng: np.random.Generator,
) -> Tuple[List[torch.Tensor], List[np.ndarray], List[List[str]], Dict[str, List[np.ndarray]]]:
    x_batches: List[torch.Tensor] = []
    sim_batches: List[np.ndarray] = []
    profile_batches: List[List[str]] = []
    profile_sims: Dict[str, List[np.ndarray]] = {p.name: [] for p in profiles}
    profile_name_list = [p.name for p in profiles]
    profile_lookup = {p.name: p for p in profiles}

    for _ in range(num_batches):
        x_np = np.zeros((batch_size, seq_len, c_dim), dtype=np.float32)
        s_np = np.zeros((batch_size, seq_len - 1), dtype=np.float32)
        names: List[str] = []
        for b in range(batch_size):
            pname = profile_name_list[int(rng.integers(0, len(profile_name_list)))]
            profile = profile_lookup[pname]
            s = _sample_similarity_sequence(profile, seq_len, rng)
            x = _similarity_to_embedding(s, c_dim, rng)
            x_np[b] = x
            s_np[b] = s
            names.append(pname)
            profile_sims[pname].append(s)

        x_batches.append(torch.from_numpy(x_np))
        sim_batches.append(s_np)
        profile_batches.append(names)
    return x_batches, sim_batches, profile_batches, profile_sims


def instantiate_selector(algo_name: str, target_r: float, fixed_tau: float):
    if algo_name == "PLE":
        sel = PLEBatchTopK(r=target_r, fixed_tau=fixed_tau, sample_prob=0.0, max_s=None)
    elif algo_name == "Greedy":
        sel = BatchGreedy(r=target_r, fixed_tau=fixed_tau, sample_prob=0.0, max_s=None)
    elif algo_name == "TopK_maxs8":
        sel = BatchTopK(r=target_r, fixed_tau=fixed_tau, sample_prob=0.0, max_s=8)
    else:
        raise ValueError(f"Unknown algo_name: {algo_name}")
    sel.eval()
    return sel


def run_selector_on_batches(selector, x_batches: Sequence[torch.Tensor], device: torch.device):
    masks: List[np.ndarray] = []
    avg_r_values: List[float] = []
    with torch.no_grad():
        for x in x_batches:
            xx = x.to(device=device, dtype=torch.float32)
            mask, avg_r, _tau = selector(xx)
            masks.append(mask.detach().cpu().numpy().astype(bool))
            avg_r_values.append(float(avg_r.detach().cpu().item()))
    return masks, np.asarray(avg_r_values, dtype=np.float64)


def calibrate_tau(
    algo_name: str,
    target_r: float,
    x_batches_calib: Sequence[torch.Tensor],
    device: torch.device,
) -> Tuple[float, Dict[str, List[float]]]:
    if algo_name == "PLE":
        grid = np.geomspace(1e-3, 20.0, 36)
    elif algo_name == "Greedy":
        grid = np.linspace(1e-4, 0.999, 22)
    elif algo_name == "TopK_maxs8":
        grid = np.linspace(1e-4, 0.999, 36)
    else:
        raise ValueError(algo_name)

    curve_tau: List[float] = []
    curve_avg: List[float] = []

    def eval_tau_values(tau_values: np.ndarray) -> Tuple[float, float]:
        best_tau_local = float(tau_values[0])
        best_err_local = float("inf")
        for tau in tau_values:
            selector = instantiate_selector(algo_name, target_r, float(tau))
            _masks, avg_r_values = run_selector_on_batches(selector, x_batches_calib, device)
            mean_avg = float(avg_r_values.mean())
            err = abs(mean_avg - target_r)

            curve_tau.append(float(tau))
            curve_avg.append(mean_avg)
            if err < best_err_local:
                best_err_local = err
                best_tau_local = float(tau)
        return best_tau_local, best_err_local

    best_tau, best_err = eval_tau_values(grid)

    # Local refinement around the best coarse point
    tau_sorted = np.sort(np.asarray(curve_tau, dtype=np.float64))
    idx_best = int(np.argmin(np.abs(tau_sorted - best_tau)))
    left_idx = max(0, idx_best - 1)
    right_idx = min(len(tau_sorted) - 1, idx_best + 1)
    lo = float(tau_sorted[left_idx])
    hi = float(tau_sorted[right_idx])

    if lo == hi:
        if algo_name == "PLE":
            lo = max(1e-6, lo / 1.8)
            hi = hi * 1.8
        else:
            lo = max(1e-4, lo - 0.08)
            hi = min(0.999, hi + 0.08)

    if algo_name == "PLE":
        refine = np.geomspace(max(1e-6, lo), max(1e-6, hi), 28)
    else:
        refine = np.linspace(max(1e-4, lo), min(0.999, hi), 28)

    best_tau_refine, best_err_refine = eval_tau_values(refine)
    if best_err_refine < best_err:
        best_tau = best_tau_refine

    # Deduplicate for cleaner plots
    seen = set()
    tau_final: List[float] = []
    avg_final: List[float] = []
    for t, a in sorted(zip(curve_tau, curve_avg), key=lambda z: z[0]):
        key = round(float(t), 10)
        if key in seen:
            continue
        seen.add(key)
        tau_final.append(float(t))
        avg_final.append(float(a))

    return best_tau, {"tau": tau_final, "mean_avg_r": avg_final}


def _false_run_lengths(mask_seq: np.ndarray) -> List[int]:
    runs: List[int] = []
    run = 0
    for v in mask_seq:
        if v:
            if run > 0:
                runs.append(run)
                run = 0
        else:
            run += 1
    if run > 0:
        runs.append(run)
    return runs


def _keep_gaps(mask_seq: np.ndarray) -> List[int]:
    idx = np.where(mask_seq)[0]
    if idx.size <= 1:
        return []
    return np.diff(idx).astype(int).tolist()


def stats_from_masks(mask_batches: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    per_sample_r: List[float] = []
    avg_r: List[float] = []
    span_lengths: List[int] = []
    keep_gaps: List[int] = []

    for mb in mask_batches:
        bsz, n = mb.shape
        keep_counts = mb.sum(axis=1).astype(np.float64)
        r_seq = 1.0 - keep_counts / float(n)
        per_sample_r.extend(r_seq.tolist())

        avg_r.append(float((n * bsz - keep_counts.sum()) / float(n * bsz)))

        for b in range(bsz):
            span_lengths.extend(_false_run_lengths(mb[b]))
            keep_gaps.extend(_keep_gaps(mb[b]))

    if len(span_lengths) == 0:
        span_lengths = [0]
    if len(keep_gaps) == 0:
        keep_gaps = [0]

    return {
        "per_sample_r": np.asarray(per_sample_r, dtype=np.float64),
        "avg_r": np.asarray(avg_r, dtype=np.float64),
        "span_lengths": np.asarray(span_lengths, dtype=np.int64),
        "keep_gaps": np.asarray(keep_gaps, dtype=np.int64),
    }


def pmf_from_lengths(lengths: np.ndarray, max_bin: int) -> np.ndarray:
    arr = np.asarray(lengths, dtype=np.int64)
    arr = np.clip(arr, 0, max_bin)
    hist = np.bincount(arr, minlength=max_bin + 1).astype(np.float64)
    hist = hist + 1e-12
    hist = hist / hist.sum()
    return hist


def distribution_distance(
    ref_stats: Dict[str, np.ndarray],
    cand_stats: Dict[str, np.ndarray],
    max_span_bin: int = 20,
    max_gap_bin: int = 20,
) -> Dict[str, float]:
    d_r = float(wasserstein_distance(ref_stats["per_sample_r"], cand_stats["per_sample_r"]))
    d_avg = float(wasserstein_distance(ref_stats["avg_r"], cand_stats["avg_r"]))
    d_span_w1 = float(wasserstein_distance(ref_stats["span_lengths"].astype(np.float64), cand_stats["span_lengths"].astype(np.float64)))

    p_span_ref = pmf_from_lengths(ref_stats["span_lengths"], max_span_bin)
    p_span_cand = pmf_from_lengths(cand_stats["span_lengths"], max_span_bin)
    d_span = float(jensenshannon(p_span_ref, p_span_cand, base=2.0))

    p_gap_ref = pmf_from_lengths(ref_stats["keep_gaps"], max_gap_bin)
    p_gap_cand = pmf_from_lengths(cand_stats["keep_gaps"], max_gap_bin)
    d_gap = float(jensenshannon(p_gap_ref, p_gap_cand, base=2.0))

    d_total = d_r + 0.5 * d_avg + 0.5 * d_span + 0.25 * d_gap
    return {
        "w1_per_sample_r": d_r,
        "w1_avg_r": d_avg,
        "w1_span": d_span_w1,
        "js_span": d_span,
        "js_gap": d_gap,
        "total": float(d_total),
    }


def ensure_dirs(root: Path) -> Dict[str, Path]:
    d = {
        "root": root,
        "figures": root / "figures",
        "tables": root / "tables",
        "metrics": root / "metrics",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def plot_similarity_profiles(profile_sims: Dict[str, List[np.ndarray]], out_path: Path) -> None:
    names = list(profile_sims.keys())
    cols = 2
    rows = int(math.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows), squeeze=False)

    for i, name in enumerate(names):
        ax = axes[i // cols][i % cols]
        sims = profile_sims[name]
        if len(sims) == 0:
            ax.set_title(name)
            ax.grid(True, alpha=0.2)
            continue
        sample = sims[0]
        ax.plot(sample, lw=1.3)
        ax.set_ylim(-1.02, 1.02)
        ax.set_title(f"{name} (example)")
        ax.set_xlabel("t")
        ax.set_ylabel("similarity")
        ax.grid(True, alpha=0.25)

    for j in range(len(names), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_calibration_curves(calib_curves: Dict[Tuple[float, str], Dict[str, List[float]]], target_r: float, out_path: Path) -> None:
    algos = ["PLE", "Greedy", "TopK_maxs8"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    for i, algo in enumerate(algos):
        curve = calib_curves[(target_r, algo)]
        tau = np.asarray(curve["tau"], dtype=np.float64)
        avgv = np.asarray(curve["mean_avg_r"], dtype=np.float64)
        ax = axes[0][i]
        ax.plot(tau, avgv, marker="o", markersize=2.5, lw=1.0)
        if algo == "PLE":
            ax.set_xscale("log")
        ax.axhline(target_r, color="red", linestyle="--", lw=1.0)
        ax.set_title(algo)
        ax.set_xlabel("tau")
        ax.set_ylabel("mean avg_r")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Calibration Curves (target_r={target_r:.2f})")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_heatmap(
    values: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(1.2 * len(col_labels) + 4, 0.45 * len(row_labels) + 2.5))
    im = ax.imshow(values, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_best_candidate_hist(
    target_r: float,
    best_name: str,
    algo_stats: Dict[str, Dict[str, np.ndarray]],
    cand_stats: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0.0, 1.0, 31)

    for algo_name, stats in algo_stats.items():
        ax.hist(
            stats["per_sample_r"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.4,
            label=f"{algo_name}",
        )

    ax.hist(
        cand_stats["per_sample_r"],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        linestyle="--",
        label=f"Candidate: {best_name}",
    )
    ax.set_title(f"Per-sample r distribution (target_r={target_r:.2f})")
    ax.set_xlabel("r")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_span_pmf_target(
    target_r: float,
    algos: Sequence[str],
    algo_stats: Dict[str, Dict[str, np.ndarray]],
    cand_stats_map: Dict[str, Dict[str, np.ndarray]],
    best_candidate: str,
    max_span_bin: int,
    out_path: Path,
) -> None:
    x = np.arange(max_span_bin + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), squeeze=False)

    ax0 = axes[0][0]
    for algo in algos:
        pmf = pmf_from_lengths(algo_stats[algo]["span_lengths"], max_span_bin)
        ax0.plot(x, pmf, lw=1.8, label=algo)
    ax0.set_title("Algorithms")
    ax0.set_xlabel("masked span length")
    ax0.set_ylabel("PMF")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8)

    ax1 = axes[0][1]
    for algo in algos:
        pmf = pmf_from_lengths(algo_stats[algo]["span_lengths"], max_span_bin)
        ax1.plot(x, pmf, lw=1.3, label=algo)
    pmf_best = pmf_from_lengths(cand_stats_map[best_candidate]["span_lengths"], max_span_bin)
    ax1.plot(x, pmf_best, lw=2.2, linestyle="--", label=f"best:{best_candidate}")
    ax1.set_title("Best Candidate vs Algorithms")
    ax1.set_xlabel("masked span length")
    ax1.set_ylabel("PMF")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)

    ax2 = axes[0][2]
    for cname in sorted(cand_stats_map.keys()):
        pmf = pmf_from_lengths(cand_stats_map[cname]["span_lengths"], max_span_bin)
        lw = 2.0 if cname == best_candidate else 1.0
        ls = "--" if cname == best_candidate else "-"
        alpha = 1.0 if cname == best_candidate else 0.9
        ax2.plot(x, pmf, lw=lw, linestyle=ls, alpha=alpha, label=cname)
    ax2.set_title("All Random-Masking Candidates")
    ax2.set_xlabel("masked span length")
    ax2.set_ylabel("PMF")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=7, ncol=1)

    fig.suptitle(f"Span-Length Distributions (target_r={target_r:.2f})")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def spectral_centroid(signal_1d: np.ndarray) -> float:
    x = np.asarray(signal_1d, dtype=np.float64)
    if x.size <= 1:
        return 0.0
    mag = np.abs(np.fft.rfft(x - x.mean()))
    if mag.size <= 1:
        return 0.0
    freq = np.fft.rfftfreq(x.size, d=1.0 / max(2, x.size))
    w = mag + 1e-12
    return float((freq * w).sum() / w.sum())


def summarize_sample_diversity(
    sim_batches: Sequence[np.ndarray],
    profile_batches: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    all_names: List[str] = []
    all_centroids: List[float] = []
    for b_idx, s_batch in enumerate(sim_batches):
        names = profile_batches[b_idx]
        for i in range(s_batch.shape[0]):
            all_names.append(names[i])
            all_centroids.append(spectral_centroid(s_batch[i]))

    counts = Counter(all_names)
    total = int(sum(counts.values()))
    probs = np.asarray([v / max(1, total) for v in counts.values()], dtype=np.float64)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    max_entropy = float(np.log(max(1, len(counts))))
    cent = np.asarray(all_centroids, dtype=np.float64)

    return {
        "total_samples": total,
        "num_profiles": int(len(counts)),
        "profile_counts": dict(counts),
        "profile_entropy": entropy,
        "profile_entropy_normalized": float(entropy / max(1e-12, max_entropy)) if max_entropy > 0 else 1.0,
        "freq_centroid_mean": float(cent.mean()) if cent.size else 0.0,
        "freq_centroid_std": float(cent.std()) if cent.size else 0.0,
        "freq_centroid_q10": float(np.quantile(cent, 0.10)) if cent.size else 0.0,
        "freq_centroid_q50": float(np.quantile(cent, 0.50)) if cent.size else 0.0,
        "freq_centroid_q90": float(np.quantile(cent, 0.90)) if cent.size else 0.0,
    }


def plot_sample_diversity(
    profile_counts: Dict[str, int],
    sim_batches: Sequence[np.ndarray],
    out_path: Path,
) -> None:
    centroids: List[float] = []
    for s_batch in sim_batches:
        for i in range(s_batch.shape[0]):
            centroids.append(spectral_centroid(s_batch[i]))

    labels = sorted(profile_counts.keys())
    vals = [profile_counts[k] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), squeeze=False)
    ax0 = axes[0][0]
    ax1 = axes[0][1]

    ax0.bar(labels, vals)
    ax0.set_title("Profile sample counts")
    ax0.set_ylabel("count")
    ax0.grid(True, axis="y", alpha=0.25)

    ax1.hist(np.asarray(centroids, dtype=np.float64), bins=24, density=True, alpha=0.8)
    ax1.set_title("Similarity spectral-centroid distribution")
    ax1.set_xlabel("centroid (a.u.)")
    ax1.set_ylabel("density")
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_algo_metric_heatmaps(
    summary_rows: Sequence[Dict[str, Any]],
    candidates: Sequence[MaskCandidate],
    target_rs: Sequence[float],
    algos: Sequence[str],
    out_dir: Path,
    metric_prefix: str,
    title_prefix: str,
    file_prefix: str,
) -> None:
    cand_names = [c.name for c in candidates]
    targ_labels = [f"{r:.2f}" for r in target_rs]
    for algo in algos:
        mat = np.zeros((len(cand_names), len(target_rs)), dtype=np.float64)
        for i, cname in enumerate(cand_names):
            for j, rt in enumerate(target_rs):
                row = [r for r in summary_rows if r["candidate"] == cname and float(r["target_r"]) == float(rt)][0]
                mat[i, j] = float(row[f"{metric_prefix}_{algo}"])
        plot_heatmap(
            mat,
            row_labels=cand_names,
            col_labels=targ_labels,
            title=f"{title_prefix} to {algo}",
            out_path=out_dir / f"{file_prefix}_to_{algo}.png",
        )


def plot_best_candidate_algo_distance(
    best_rows: Sequence[Dict[str, Any]],
    algos: Sequence[str],
    out_path: Path,
) -> None:
    targets = [float(r["target_r"]) for r in best_rows]
    x = np.arange(len(targets))
    width = 0.23

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, algo in enumerate(algos):
        vals = [float(r[f"dist_{algo}"]) for r in best_rows]
        ax.bar(x + (i - 1) * width, vals, width=width, label=algo)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in targets])
    ax.set_xlabel("target_r")
    ax.set_ylabel("distance")
    ax.set_title("Best candidate: distance to each algorithm")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _select_example_indices(profile_names: Sequence[str], max_examples: int = 4) -> List[int]:
    picked: List[int] = []
    seen = set()
    for i, name in enumerate(profile_names):
        if name not in seen:
            picked.append(i)
            seen.add(name)
        if len(picked) >= max_examples:
            return picked
    i = 0
    while len(picked) < min(max_examples, len(profile_names)):
        if i not in picked:
            picked.append(i)
        i += 1
    return picked


def plot_mask_example_grid(
    target_r: float,
    sim_batch: np.ndarray,
    profile_names: Sequence[str],
    algo_masks: Dict[str, np.ndarray],
    candidate_name: str,
    candidate_mask: np.ndarray,
    out_path: Path,
) -> None:
    methods = ["PLE", "Greedy", "TopK_maxs8", f"cand:{candidate_name}"]
    idx_list = _select_example_indices(profile_names, max_examples=4)
    n_rows = len(idx_list)

    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 2.7 * n_rows), squeeze=False)
    for r_idx, sample_idx in enumerate(idx_list):
        ax_l = axes[r_idx][0]
        ax_r = axes[r_idx][1]

        s = sim_batch[sample_idx]
        ax_l.plot(np.arange(1, s.size + 1), s, lw=1.2)
        ax_l.set_ylim(-1.02, 1.02)
        ax_l.set_title(f"sample={sample_idx}, profile={profile_names[sample_idx]}")
        ax_l.set_xlabel("position")
        ax_l.set_ylabel("similarity")
        ax_l.grid(True, alpha=0.25)

        rows = []
        for m in methods:
            if m.startswith("cand:"):
                rows.append((~candidate_mask[sample_idx]).astype(np.float64))
            else:
                rows.append((~algo_masks[m][sample_idx]).astype(np.float64))
        raster = np.stack(rows, axis=0)
        ax_r.imshow(raster, aspect="auto", cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_r.set_yticks(np.arange(len(methods)))
        ax_r.set_yticklabels(methods)
        ax_r.set_xlabel("position")
        ax_r.set_title("Masked positions (white=masked)")

    fig.suptitle(f"Mask examples across methods (target_r={target_r:.2f})")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_target_r(text: str) -> List[float]:
    out = []
    for item in text.split(","):
        val = float(item.strip())
        if not (0.0 <= val < 1.0):
            raise ValueError("target_r values must be in [0,1)")
        out.append(val)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="DTP random masking distribution study")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--calib_batches", type=int, default=8)
    parser.add_argument("--eval_batches", type=int, default=48)
    parser.add_argument("--target_r", type=str, default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--neutrality_lambda", type=float, default=1.0)
    parser.add_argument("--span_max_bin", type=int, default=24)
    parser.add_argument("--gap_max_bin", type=int, default=24)
    args = parser.parse_args()

    out_dirs = ensure_dirs(Path(args.out_dir))
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    profiles = default_profiles()
    total_batches = args.calib_batches + args.eval_batches
    x_batches, sim_batches, profile_batches, profile_sims = build_synthetic_batches(
        num_batches=total_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        c_dim=args.dim,
        profiles=profiles,
        rng=rng,
    )
    x_batches_calib = x_batches[: args.calib_batches]
    x_batches_eval = x_batches[args.calib_batches :]
    sim_batches_eval = sim_batches[args.calib_batches :]
    profile_batches_eval = profile_batches[args.calib_batches :]

    plot_similarity_profiles(profile_sims, out_dirs["figures"] / "similarity_profiles.png")
    diversity = summarize_sample_diversity(sim_batches, profile_batches)
    plot_sample_diversity(
        profile_counts=diversity["profile_counts"],
        sim_batches=sim_batches,
        out_path=out_dirs["figures"] / "sample_diversity.png",
    )
    with (out_dirs["tables"] / "sample_diversity.json").open("w", encoding="utf-8") as f:
        json.dump(diversity, f, indent=2)

    target_rs = parse_target_r(args.target_r)
    algos = ["PLE", "Greedy", "TopK_maxs8"]
    candidates: List[MaskCandidate] = build_default_candidates()

    calib_rows: List[Dict[str, Any]] = []
    calib_curves: Dict[Tuple[float, str], Dict[str, List[float]]] = {}
    algo_stats_all: Dict[Tuple[float, str], Dict[str, np.ndarray]] = {}
    algo_tau: Dict[Tuple[float, str], float] = {}
    algo_masks_all: Dict[Tuple[float, str], List[np.ndarray]] = {}

    for r_t in target_rs:
        for algo in algos:
            best_tau, curve = calibrate_tau(algo, r_t, x_batches_calib, device)
            calib_curves[(r_t, algo)] = curve
            algo_tau[(r_t, algo)] = best_tau

            selector = instantiate_selector(algo, r_t, best_tau)
            masks_eval, avg_r_eval = run_selector_on_batches(selector, x_batches_eval, device)
            stats = stats_from_masks(masks_eval)
            algo_stats_all[(r_t, algo)] = stats
            algo_masks_all[(r_t, algo)] = masks_eval

            calib_rows.append(
                {
                    "target_r": r_t,
                    "algorithm": algo,
                    "tau": best_tau,
                    "mean_avg_r_eval": float(avg_r_eval.mean()),
                    "std_avg_r_eval": float(avg_r_eval.std()),
                    "abs_error": abs(float(avg_r_eval.mean()) - r_t),
                }
            )

        plot_calibration_curves(calib_curves, r_t, out_dirs["figures"] / f"calibration_target_{r_t:.2f}.png")

    with (out_dirs["tables"] / "calibration.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(calib_rows[0].keys()))
        writer.writeheader()
        writer.writerows(calib_rows)

    summary_rows: List[Dict[str, Any]] = []
    raw_metrics = {
        "seed": args.seed,
        "target_r": target_rs,
        "algo_tau": {f"{k[0]:.2f}_{k[1]}": v for k, v in algo_tau.items()},
        "sample_diversity": diversity,
        "span_max_bin": int(args.span_max_bin),
        "gap_max_bin": int(args.gap_max_bin),
        "distances": [],
    }
    cand_masks_all: Dict[Tuple[float, str], List[np.ndarray]] = {}
    cand_stats_all: Dict[Tuple[float, str], Dict[str, np.ndarray]] = {}

    for r_t in target_rs:
        cand_stats_map: Dict[str, Dict[str, np.ndarray]] = {}
        for cand_idx, cand in enumerate(candidates):
            rng_cand = np.random.default_rng(args.seed + int(round(r_t * 1000)) + 77 + 997 * (cand_idx + 1))
            mask_batches_cand: List[np.ndarray] = []
            for _ in range(args.eval_batches):
                mask = sample_candidate_mask(cand, args.batch_size, args.seq_len, r_t, rng_cand)
                mask_batches_cand.append(mask)
            cand_stats_map[cand.name] = stats_from_masks(mask_batches_cand)
            cand_masks_all[(r_t, cand.name)] = mask_batches_cand

        for cand in candidates:
            algo_totals: Dict[str, float] = {}
            row = {
                "target_r": r_t,
                "candidate": cand.name,
                "candidate_mean_avg_r": float(cand_stats_map[cand.name]["avg_r"].mean()),
            }
            for algo in algos:
                d = distribution_distance(
                    algo_stats_all[(r_t, algo)],
                    cand_stats_map[cand.name],
                    max_span_bin=int(args.span_max_bin),
                    max_gap_bin=int(args.gap_max_bin),
                )
                total = float(d["total"])
                algo_totals[algo] = total
                row[f"dist_{algo}"] = total
                row[f"w1_r_{algo}"] = float(d["w1_per_sample_r"])
                row[f"w1_avg_{algo}"] = float(d["w1_avg_r"])
                row[f"w1_span_{algo}"] = float(d["w1_span"])
                row[f"js_span_{algo}"] = float(d["js_span"])
                row[f"js_gap_{algo}"] = float(d["js_gap"])

            algo_vals = np.asarray([algo_totals[a] for a in algos], dtype=np.float64)
            row["mean_dist"] = float(algo_vals.mean())
            row["std_dist"] = float(algo_vals.std())
            row["neutral_score"] = float(algo_vals.mean() + args.neutrality_lambda * algo_vals.std())
            summary_rows.append(row)
            raw_metrics["distances"].append(row)

        for cand in candidates:
            cand_stats_all[(r_t, cand.name)] = cand_stats_map[cand.name]

        # Per-target best candidate for histogram figure
        rows_t = [r for r in summary_rows if float(r["target_r"]) == float(r_t)]
        best_row = sorted(rows_t, key=lambda z: float(z["neutral_score"]))[0]
        best_name = str(best_row["candidate"])
        plot_best_candidate_hist(
            target_r=r_t,
            best_name=best_name,
            algo_stats={a: algo_stats_all[(r_t, a)] for a in algos},
            cand_stats=cand_stats_map[best_name],
            out_path=out_dirs["figures"] / f"hist_best_target_{r_t:.2f}.png",
        )

    with (out_dirs["tables"] / "distance_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Best candidates table
    best_rows = []
    for r_t in target_rs:
        rows_t = [r for r in summary_rows if float(r["target_r"]) == float(r_t)]
        best = sorted(rows_t, key=lambda z: float(z["neutral_score"]))[0]
        best_rows.append(best)

    with (out_dirs["tables"] / "best_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
        writer.writeheader()
        writer.writerows(best_rows)

    # Long-format per-algorithm distance table
    dist_algo_rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        for algo in algos:
            dist_algo_rows.append(
                {
                    "target_r": row["target_r"],
                    "candidate": row["candidate"],
                    "algorithm": algo,
                    "distance": row[f"dist_{algo}"],
                    "js_span": row[f"js_span_{algo}"],
                    "w1_span": row[f"w1_span_{algo}"],
                    "w1_r": row[f"w1_r_{algo}"],
                    "w1_avg": row[f"w1_avg_{algo}"],
                }
            )
    with (out_dirs["tables"] / "distance_by_algorithm.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dist_algo_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dist_algo_rows)

    cand_names = [c.name for c in candidates]
    targ_labels = [f"{r:.2f}" for r in target_rs]
    mean_mat = np.zeros((len(cand_names), len(target_rs)), dtype=np.float64)
    std_mat = np.zeros((len(cand_names), len(target_rs)), dtype=np.float64)
    for i, cname in enumerate(cand_names):
        for j, rt in enumerate(target_rs):
            row = [r for r in summary_rows if r["candidate"] == cname and float(r["target_r"]) == float(rt)][0]
            mean_mat[i, j] = float(row["mean_dist"])
            std_mat[i, j] = float(row["std_dist"])

    plot_heatmap(
        mean_mat,
        row_labels=cand_names,
        col_labels=targ_labels,
        title="Mean Distribution Distance to {PLE, Greedy, TopK(max_s=8)}",
        out_path=out_dirs["figures"] / "distance_mean_heatmap.png",
    )
    plot_heatmap(
        std_mat,
        row_labels=cand_names,
        col_labels=targ_labels,
        title="Algorithm Preference (std of distances; lower is more neutral)",
        out_path=out_dirs["figures"] / "distance_std_heatmap.png",
    )
    plot_per_algo_metric_heatmaps(
        summary_rows=summary_rows,
        candidates=candidates,
        target_rs=target_rs,
        algos=algos,
        out_dir=out_dirs["figures"],
        metric_prefix="dist",
        title_prefix="Distance",
        file_prefix="distance",
    )
    plot_per_algo_metric_heatmaps(
        summary_rows=summary_rows,
        candidates=candidates,
        target_rs=target_rs,
        algos=algos,
        out_dir=out_dirs["figures"],
        metric_prefix="js_span",
        title_prefix="JS span distance",
        file_prefix="span_js",
    )
    plot_per_algo_metric_heatmaps(
        summary_rows=summary_rows,
        candidates=candidates,
        target_rs=target_rs,
        algos=algos,
        out_dir=out_dirs["figures"],
        metric_prefix="w1_span",
        title_prefix="W1 span distance",
        file_prefix="span_w1",
    )
    plot_best_candidate_algo_distance(
        best_rows=best_rows,
        algos=algos,
        out_path=out_dirs["figures"] / "best_candidate_algo_distance.png",
    )

    # Mask-location example figures for each target r
    for best in best_rows:
        r_t = float(best["target_r"])
        best_name = str(best["candidate"])
        sim_batch0 = sim_batches_eval[0]
        names_batch0 = profile_batches_eval[0]
        algo_mask_map = {
            "PLE": algo_masks_all[(r_t, "PLE")][0],
            "Greedy": algo_masks_all[(r_t, "Greedy")][0],
            "TopK_maxs8": algo_masks_all[(r_t, "TopK_maxs8")][0],
        }
        cand_mask0 = cand_masks_all[(r_t, best_name)][0]
        plot_mask_example_grid(
            target_r=r_t,
            sim_batch=sim_batch0,
            profile_names=names_batch0,
            algo_masks=algo_mask_map,
            candidate_name=best_name,
            candidate_mask=cand_mask0,
            out_path=out_dirs["figures"] / f"mask_examples_target_{r_t:.2f}.png",
        )

    # Additional mask examples for all random masking candidates
    for r_t in target_rs:
        sim_batch0 = sim_batches_eval[0]
        names_batch0 = profile_batches_eval[0]
        algo_mask_map = {
            "PLE": algo_masks_all[(r_t, "PLE")][0],
            "Greedy": algo_masks_all[(r_t, "Greedy")][0],
            "TopK_maxs8": algo_masks_all[(r_t, "TopK_maxs8")][0],
        }
        for cand in candidates:
            cand_name = cand.name
            cand_mask0 = cand_masks_all[(r_t, cand_name)][0]
            plot_mask_example_grid(
                target_r=r_t,
                sim_batch=sim_batch0,
                profile_names=names_batch0,
                algo_masks=algo_mask_map,
                candidate_name=cand_name,
                candidate_mask=cand_mask0,
                out_path=out_dirs["figures"] / f"mask_examples_target_{r_t:.2f}_cand_{cand_name}.png",
            )

    # Span-length PMF figures across diverse settings
    best_by_target = {float(r["target_r"]): str(r["candidate"]) for r in best_rows}
    for r_t in target_rs:
        plot_span_pmf_target(
            target_r=r_t,
            algos=algos,
            algo_stats={a: algo_stats_all[(r_t, a)] for a in algos},
            cand_stats_map={c.name: cand_stats_all[(r_t, c.name)] for c in candidates},
            best_candidate=best_by_target[float(r_t)],
            max_span_bin=int(args.span_max_bin),
            out_path=out_dirs["figures"] / f"span_pmf_target_{r_t:.2f}.png",
        )

    with (out_dirs["metrics"] / "raw_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(raw_metrics, f, indent=2)

    print(f"Saved study outputs to: {out_dirs['root']}")
    print(f"Figures: {out_dirs['figures']}")
    print(f"Tables: {out_dirs['tables']}")
    print(f"Metrics: {out_dirs['metrics']}")


if __name__ == "__main__":
    main()
