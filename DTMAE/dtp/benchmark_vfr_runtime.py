from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from dtp.ops import (
        BatchClusteringVarsTok,
        BatchDPCodecSlime,
        BatchGreedy,
        BatchTopK,
        PLEBatchTopK,
    )
except ImportError:
    from ops import (
        BatchClusteringVarsTok,
        BatchDPCodecSlime,
        BatchGreedy,
        BatchTopK,
        PLEBatchTopK,
    )


@dataclass(frozen=True)
class AlgoSpec:
    key: str
    label: str
    cls: Any
    selector_kwargs: Dict[str, Any]
    complexity: str


def default_algo_specs() -> List[AlgoSpec]:
    return [
        AlgoSpec(
            key="PLE",
            label="PLE",
            cls=PLEBatchTopK,
            selector_kwargs={"max_s": None},
            complexity="O(N)",
        ),
        AlgoSpec(
            key="PLE_maxs4",
            label="PLE(max_span=4)",
            cls=PLEBatchTopK,
            selector_kwargs={"max_s": 4},
            complexity="O(N)",
        ),
        AlgoSpec(
            key="Greedy",
            label="Greedy",
            cls=BatchGreedy,
            selector_kwargs={"max_s": None},
            complexity="O(N^2)",
        ),
        AlgoSpec(
            key="TopK_maxs4",
            label="TopK(max_span=4)",
            cls=BatchTopK,
            selector_kwargs={"max_s": 4},
            complexity="O(N log N)",
        ),
        AlgoSpec(
            key="TopK_maxs8",
            label="TopK(max_span=8)",
            cls=BatchTopK,
            selector_kwargs={"max_s": 8},
            complexity="O(N log N)",
        ),
        AlgoSpec(
            key="Clustering_maxs4",
            label="Clustering(max_span=4)",
            cls=BatchClusteringVarsTok,
            selector_kwargs={"max_s": 4},
            complexity="O(N^2)",
        ),
        AlgoSpec(
            key="Clustering_maxs8",
            label="Clustering(max_span=8)",
            cls=BatchClusteringVarsTok,
            selector_kwargs={"max_s": 8},
            complexity="O(N^2)",
        ),
        AlgoSpec(
            key="Clustering_unbounded",
            label="Clustering(max_span=None)",
            cls=BatchClusteringVarsTok,
            selector_kwargs={"max_s": None, "cluster_max_span": None},
            complexity="O(N^2)",
        ),
        AlgoSpec(
            key="DP_maxs4",
            label="DP(max_span=4)",
            cls=BatchDPCodecSlime,
            selector_kwargs={"max_s": 4},
            complexity="O(NN'M)",
        ),
        AlgoSpec(
            key="DP_maxs8",
            label="DP(max_span=8)",
            cls=BatchDPCodecSlime,
            selector_kwargs={"max_s": 8},
            complexity="O(NN'M)",
        ),
        AlgoSpec(
            key="DP_unbounded",
            label="DP(max_span=None)",
            cls=BatchDPCodecSlime,
            selector_kwargs={"max_s": None},
            complexity="O(N^2N')",
        ),
    ]


def _normalize_algo_name(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch not in " _-()")


def resolve_algo_specs(algo_text: str, specs: Sequence[AlgoSpec]) -> List[AlgoSpec]:
    by_key = {s.key: s for s in specs}
    alias_to_key = {
        "ple": "PLE",
        "plemaxs4": "PLE_maxs4",
        "plemaxspan4": "PLE_maxs4",
        "greedy": "Greedy",
        "topkmaxs4": "TopK_maxs4",
        "topkmaxspan4": "TopK_maxs4",
        "topk4": "TopK_maxs4",
        "topkmaxs8": "TopK_maxs8",
        "topkmaxspan8": "TopK_maxs8",
        "topk8": "TopK_maxs8",
        "clusteringmaxs4": "Clustering_maxs4",
        "clusteringmaxspan4": "Clustering_maxs4",
        "clustermaxs4": "Clustering_maxs4",
        "kmeansmaxs4": "Clustering_maxs4",
        "varstokmaxs4": "Clustering_maxs4",
        "clusteringmaxs8": "Clustering_maxs8",
        "clusteringmaxspan8": "Clustering_maxs8",
        "clustermaxs8": "Clustering_maxs8",
        "kmeansmaxs8": "Clustering_maxs8",
        "varstokmaxs8": "Clustering_maxs8",
        "clustering": "Clustering_unbounded",
        "clusteringunbounded": "Clustering_unbounded",
        "cluster": "Clustering_unbounded",
        "kmeans": "Clustering_unbounded",
        "varstok": "Clustering_unbounded",
        "dpmaxs4": "DP_maxs4",
        "dpmaxspan4": "DP_maxs4",
        "codecslimemaxs4": "DP_maxs4",
        "dpmaxs8": "DP_maxs8",
        "dpmaxspan8": "DP_maxs8",
        "codecslimemaxs8": "DP_maxs8",
        "dp": "DP_unbounded",
        "dpunbounded": "DP_unbounded",
        "codecslime": "DP_unbounded",
    }

    selected: List[AlgoSpec] = []
    seen = set()
    for raw in algo_text.split(","):
        token = raw.strip()
        if not token:
            continue

        key = by_key.get(token, None)
        if key is not None:
            resolved_key = token
        else:
            norm = _normalize_algo_name(token)
            if norm not in alias_to_key:
                known = ", ".join(s.key for s in specs)
                raise ValueError(f"Unknown algorithm '{token}'. Known keys: {known}")
            resolved_key = alias_to_key[norm]

        if resolved_key in seen:
            continue
        seen.add(resolved_key)
        selected.append(by_key[resolved_key])

    if not selected:
        raise ValueError("No valid algorithm selected.")
    return selected


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for item in text.split(","):
        if not item.strip():
            continue
        v = float(item.strip())
        if not (0.0 <= v < 1.0):
            raise ValueError(f"r must be in [0, 1): got {v}")
        values.append(v)
    if not values:
        raise ValueError("r_list is empty.")
    return values


def make_synthetic_batches(
    batch_size: int,
    seq_len: int,
    dim: int,
    num_batches: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> List[torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    out: List[torch.Tensor] = []
    for _ in range(num_batches):
        x_cpu = torch.randn(batch_size, seq_len, dim, generator=gen, dtype=dtype)
        out.append(x_cpu.to(device=device, non_blocking=False))
    return out


def instantiate_selector(
    spec: AlgoSpec,
    target_r: float,
    fixed_tau: Optional[float],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    kwargs = dict(spec.selector_kwargs)
    kwargs["sample_prob"] = 0.0
    if fixed_tau is not None:
        kwargs["fixed_tau"] = float(fixed_tau)

    selector = spec.cls(r=float(target_r), **kwargs)
    selector.eval()
    selector.to(device=device, dtype=dtype)
    return selector


@torch.no_grad()
def evaluate_mean_avg_r(selector, batches: Sequence[torch.Tensor]) -> float:
    vals: List[float] = []
    for x in batches:
        _mask, avg_r, _tau = selector(x)
        vals.append(float(avg_r.item()))
    return float(np.mean(vals)) if vals else 0.0


def _tau_grid(spec: AlgoSpec, points: int) -> np.ndarray:
    p = max(3, int(points))
    if spec.cls is PLEBatchTopK:
        return np.geomspace(1e-3, 20.0, p)
    return np.linspace(1e-4, 0.999, p)


def calibrate_fixed_tau(
    spec: AlgoSpec,
    target_r: float,
    batches: Sequence[torch.Tensor],
    device: torch.device,
    coarse_points: int = 24,
    refine_points: int = 20,
) -> Tuple[float, float, float]:
    if spec.cls is BatchDPCodecSlime:
        selector = instantiate_selector(spec, target_r=target_r, fixed_tau=1.0, device=device)
        mean_r = evaluate_mean_avg_r(selector, batches)
        return 1.0, mean_r, abs(mean_r - target_r)

    grid = _tau_grid(spec, coarse_points)
    tau_eval: List[float] = []
    avg_eval: List[float] = []

    best_tau = float(grid[0])
    best_avg_r = 0.0
    best_err = float("inf")

    for tau in grid:
        selector = instantiate_selector(spec, target_r=target_r, fixed_tau=float(tau), device=device)
        mean_r = evaluate_mean_avg_r(selector, batches)
        err = abs(mean_r - target_r)
        tau_eval.append(float(tau))
        avg_eval.append(mean_r)
        if err < best_err:
            best_err = err
            best_tau = float(tau)
            best_avg_r = mean_r

    tau_arr = np.asarray(tau_eval, dtype=np.float64)
    tau_sorted = np.sort(tau_arr)
    idx = int(np.argmin(np.abs(tau_sorted - best_tau)))
    left = float(tau_sorted[max(0, idx - 1)])
    right = float(tau_sorted[min(len(tau_sorted) - 1, idx + 1)])

    if left == right:
        if spec.cls is PLEBatchTopK:
            left = max(1e-6, left / 2.0)
            right = right * 2.0
        else:
            left = max(1e-4, left - 0.1)
            right = min(0.999, right + 0.1)

    if spec.cls is PLEBatchTopK:
        refine = np.geomspace(max(1e-6, left), max(1e-6, right), max(3, int(refine_points)))
    else:
        refine = np.linspace(max(1e-4, left), min(0.999, right), max(3, int(refine_points)))

    for tau in refine:
        selector = instantiate_selector(spec, target_r=target_r, fixed_tau=float(tau), device=device)
        mean_r = evaluate_mean_avg_r(selector, batches)
        err = abs(mean_r - target_r)
        if err < best_err:
            best_err = err
            best_tau = float(tau)
            best_avg_r = mean_r

    return best_tau, best_avg_r, best_err


@torch.no_grad()
def _run_selector_pass(selector, batches: Sequence[torch.Tensor]) -> None:
    for x in batches:
        selector(x)


@torch.no_grad()
def benchmark_wall_time(
    selector,
    batches: Sequence[torch.Tensor],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Tuple[float, float, List[float]]:
    for _ in range(max(0, int(warmup))):
        _run_selector_pass(selector, batches)

    times_ms: List[float] = []
    reps = max(1, int(repeats))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        for _ in range(reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _run_selector_pass(selector, batches)
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(float(start.elapsed_time(end)))
    else:
        for _ in range(reps):
            t0 = time.perf_counter()
            _run_selector_pass(selector, batches)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), times_ms


def resolve_devices(device_text: str) -> List[torch.device]:
    devices: List[torch.device] = []
    seen = set()

    for raw in device_text.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token == "cpu":
            if "cpu" not in seen:
                devices.append(torch.device("cpu"))
                seen.add("cpu")
            continue
        if token in {"cuda", "gpu"}:
            if torch.cuda.is_available() and "cuda" not in seen:
                devices.append(torch.device("cuda"))
                seen.add("cuda")
            continue
        raise ValueError(f"Unknown device token '{raw}'. Use cpu and/or cuda.")

    if not devices:
        raise ValueError("No valid device resolved.")
    return devices


def _empty_metrics() -> Dict[str, float]:
    nan = float("nan")
    return {
        "mean_avg_r": nan,
        "ms_total_mean": nan,
        "ms_total_std": nan,
        "ms_per_batch": nan,
        "ms_per_utterance": nan,
        "us_per_token": nan,
    }


def run_single_device(
    spec: AlgoSpec,
    target_r: float,
    fixed_tau: Optional[float],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    batches = make_synthetic_batches(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dim=args.dim,
        num_batches=args.bench_batches,
        device=device,
        dtype=torch.float32,
        seed=seed,
    )
    selector = instantiate_selector(spec, target_r=target_r, fixed_tau=fixed_tau, device=device)
    measured_r = evaluate_mean_avg_r(selector, batches)
    t_mean, t_std, _all = benchmark_wall_time(
        selector,
        batches,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    utterances = float(args.batch_size * args.bench_batches)
    tokens = utterances * float(args.seq_len)
    return {
        "mean_avg_r": measured_r,
        "ms_total_mean": t_mean,
        "ms_total_std": t_std,
        "ms_per_batch": t_mean / float(max(1, args.bench_batches)),
        "ms_per_utterance": t_mean / max(1.0, utterances),
        "us_per_token": (t_mean * 1000.0) / max(1.0, tokens),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark VFR selector wall time (CPU/GPU) for DTP classes")
    parser.add_argument("--out_dir", type=str, default="./benchmark_vfr_runtime_out")
    parser.add_argument(
        "--algorithms",
        type=str,
        default=(
            "PLE,PLE_maxs4,Greedy,TopK_maxs4,TopK_maxs8,"
            "Clustering_maxs4,Clustering_maxs8,DP_maxs4"
        ),
        help=(
            "Comma-separated keys. Examples: "
            "PLE,PLE_maxs4,Greedy,TopK_maxs4,TopK_maxs8,"
            "Clustering_maxs4,Clustering_maxs8,DP_maxs4"
        ),
    )
    parser.add_argument("--r_list", type=str, default="0.5")
    parser.add_argument("--devices", type=str, default="cpu,cuda")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=500)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--calib_batches", type=int, default=4)
    parser.add_argument("--bench_batches", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--no_calibrate_tau", action="store_true")
    parser.add_argument("--tau_coarse_points", type=int, default=24)
    parser.add_argument("--tau_refine_points", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_specs = default_algo_specs()
    selected_specs = resolve_algo_specs(args.algorithms, all_specs)
    target_rs = parse_float_list(args.r_list)
    devices = resolve_devices(args.devices)
    has_cpu = any(d.type == "cpu" for d in devices)
    has_cuda = any(d.type == "cuda" for d in devices)

    rows: List[Dict[str, Any]] = []
    for ridx, r_t in enumerate(target_rs):
        calib_batches_cpu = make_synthetic_batches(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            dim=args.dim,
            num_batches=args.calib_batches,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=args.seed + 1000 * (ridx + 1),
        )

        for aidx, spec in enumerate(selected_specs):
            if args.no_calibrate_tau:
                tau_star = None
                calib_avg = float("nan")
                calib_err = float("nan")
            else:
                tau_star, calib_avg, calib_err = calibrate_fixed_tau(
                    spec=spec,
                    target_r=r_t,
                    batches=calib_batches_cpu,
                    device=torch.device("cpu"),
                    coarse_points=args.tau_coarse_points,
                    refine_points=args.tau_refine_points,
                )

            cpu_metrics = _empty_metrics()
            gpu_metrics = _empty_metrics()

            if has_cpu:
                cpu_metrics = run_single_device(
                    spec=spec,
                    target_r=r_t,
                    fixed_tau=tau_star,
                    args=args,
                    device=torch.device("cpu"),
                    seed=args.seed + 20000 + 101 * ridx + aidx,
                )

            if has_cuda:
                gpu_metrics = run_single_device(
                    spec=spec,
                    target_r=r_t,
                    fixed_tau=tau_star,
                    args=args,
                    device=torch.device("cuda"),
                    seed=args.seed + 40000 + 101 * ridx + aidx,
                )

            row = {
                "algorithm_key": spec.key,
                "algorithm": spec.label,
                "config": json.dumps(spec.selector_kwargs, separators=(",", ":")),
                "complexity": spec.complexity,
                "target_r": float(r_t),
                "tau_fixed": float(tau_star) if tau_star is not None else float("nan"),
                "tau_calib_mean_avg_r": float(calib_avg),
                "tau_calib_abs_err": float(calib_err),
                "cpu_mean_avg_r": cpu_metrics["mean_avg_r"],
                "cpu_ms_total_mean": cpu_metrics["ms_total_mean"],
                "cpu_ms_total_std": cpu_metrics["ms_total_std"],
                "cpu_ms_per_batch": cpu_metrics["ms_per_batch"],
                "cpu_ms_per_utterance": cpu_metrics["ms_per_utterance"],
                "cpu_us_per_token": cpu_metrics["us_per_token"],
                "gpu_mean_avg_r": gpu_metrics["mean_avg_r"],
                "gpu_ms_total_mean": gpu_metrics["ms_total_mean"],
                "gpu_ms_total_std": gpu_metrics["ms_total_std"],
                "gpu_ms_per_batch": gpu_metrics["ms_per_batch"],
                "gpu_ms_per_utterance": gpu_metrics["ms_per_utterance"],
                "gpu_us_per_token": gpu_metrics["us_per_token"],
            }
            rows.append(row)

            cpu_short = row["cpu_ms_per_utterance"]
            gpu_short = row["gpu_ms_per_utterance"]
            tau_txt = "nan" if tau_star is None else f"{tau_star:.6f}"
            print(
                f"[r={r_t:.2f}] {spec.label:16s} tau={tau_txt} | "
                f"CPU ms/utt={cpu_short:.4f} | GPU ms/utt={gpu_short:.4f}"
            )

    csv_path = out_dir / "vfr_runtime_results.csv"
    json_path = out_dir / "vfr_runtime_results.json"
    write_csv(csv_path, rows)

    meta = {
        "config": {
            "algorithms": [s.key for s in selected_specs],
            "target_r": target_rs,
            "devices": [d.type for d in devices],
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "dim": args.dim,
            "calib_batches": args.calib_batches,
            "bench_batches": args.bench_batches,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "calibrate_tau": not args.no_calibrate_tau,
            "tau_coarse_points": args.tau_coarse_points,
            "tau_refine_points": args.tau_refine_points,
        },
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
