#!/usr/bin/env python3
"""
Search a fixed_tau value (DTP threshold) that achieves a target average reduction ratio (avg_r).

This script is intended to replace manually re-running:

  python eval/dtp_stats.py \
    --input <filelist_or_dir> \
    --run_dir <run_dir> \
    --output_dir <out_dir> \
    --cfg_override model.resampler.dtp_params.fixed_tau=<tau>

with an efficient search over tau in [tau_min, tau_max] using a discrete step (default 0.001).

Design goals:
- Load the checkpoint once (fast) and evaluate multiple tau candidates by mutating model.dtp.fixed_tau.
- Use a small number of iterations (bracket + binary search), assuming avg_r is monotonic in fixed_tau.
- Save the best (tau, avg_r) to JSON so you can reuse it as a cfg override during eval.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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

# Reuse the exact dataset/forward pipeline from dtp_stats.py to stay consistent.
from dtp_stats import (  # noqa: E402
    AudioDataset,
    apply_cfg_overrides,
    get_tau_state,
    parse_input_paths,
    patch_legacy_dtp_state_dict,
    pick_tau_value,
    resolve_with_dataset_roots,
    run_generator_forward,
)
from DTMAE.lightning_module import CodecLightningModule  # noqa: E402


@dataclass(frozen=True)
class SearchConfig:
    target_avg_r: float
    tau_min: float
    tau_max: float
    tau_step: float
    max_samples: Optional[int]
    length_mode: str
    bootstrap_update_test_time: bool
    bootstrap_only: bool
    bootstrap_override_update_test_time: bool
    # If true, allow expanding tau bounds when target can't be bracketed.
    auto_expand: bool
    auto_expand_max_tau: float
    # Search behavior knobs (discrete indices)
    direction_probe_step: int


def _ensure_tau_range(cfg: SearchConfig) -> None:
    if not (0.0 < cfg.tau_step):
        raise ValueError("--tau_step must be > 0")
    if not (0.0 < cfg.tau_min <= cfg.tau_max):
        raise ValueError("--tau_min must be > 0 and <= --tau_max")
    if not (0.0 <= cfg.target_avg_r < 1.0):
        raise ValueError("--target_avg_r must be in [0, 1)")
    if cfg.auto_expand:
        if cfg.auto_expand_max_tau <= 0.0:
            raise ValueError("--auto_expand_max_tau must be > 0 when --auto_expand is enabled")
    if cfg.direction_probe_step <= 0:
        raise ValueError("--direction_probe_step must be >= 1")


def _index_bounds(cfg: SearchConfig) -> Tuple[int, int]:
    lo = int(np.ceil(cfg.tau_min / cfg.tau_step))
    hi = int(np.floor(cfg.tau_max / cfg.tau_step))
    if lo < 1:
        lo = 1  # DTP code requires tau > 0
    if hi < lo:
        raise ValueError(
            f"Invalid tau range after discretization: tau_min={cfg.tau_min}, tau_max={cfg.tau_max}, tau_step={cfg.tau_step}"
        )
    return lo, hi


def _idx_to_tau(idx: int, tau_step: float) -> float:
    return float(idx) * float(tau_step)


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


def _set_fixed_tau(dtp_module: torch.nn.Module, fixed_tau: float) -> None:
    if fixed_tau <= 0.0:
        raise ValueError("fixed_tau must be > 0")
    if not hasattr(dtp_module, "fixed_tau"):
        raise AttributeError("Model DTP module does not expose attribute 'fixed_tau'.")

    # The DTP selector uses fixed_tau if not None (see DTMAE/dtp/ops.py).
    dtp_module.fixed_tau = float(fixed_tau)

    # Keep buffers consistent for reporting/debugging (not strictly required for behavior).
    for buf_name in ("tau_train", "tau_eval"):
        if hasattr(dtp_module, buf_name):
            buf = getattr(dtp_module, buf_name)
            if torch.is_tensor(buf):
                buf.fill_(float(fixed_tau))


def _clear_fixed_tau_enable_update_test_time(dtp_module: torch.nn.Module) -> None:
    """
    Prepare the DTP module to adapt tau during eval (update_test_time=True, fixed_tau=None).
    """
    if not hasattr(dtp_module, "fixed_tau") or not hasattr(dtp_module, "update_test_time"):
        raise AttributeError("DTP module does not expose fixed_tau/update_test_time attributes.")
    dtp_module.fixed_tau = None
    dtp_module.update_test_time = True
    # Keep eval buffers consistent for a fresh adaptation run.
    for name in ("r_ema_eval", "steps_eval"):
        if hasattr(dtp_module, name):
            buf = getattr(dtp_module, name)
            if torch.is_tensor(buf):
                buf.zero_()
    if hasattr(dtp_module, "tau_eval") and hasattr(dtp_module, "tau_train"):
        te = getattr(dtp_module, "tau_eval")
        tt = getattr(dtp_module, "tau_train")
        if torch.is_tensor(te) and torch.is_tensor(tt):
            te.copy_(tt)


def _load_existing_trials(trials_path: Path) -> Dict[str, Dict[str, object]]:
    """
    Returns a mapping from a string tau key to the stored trial record.
    The tau key is a string to avoid float-repr issues when resuming.
    """
    cache: Dict[str, Dict[str, object]] = {}
    if not trials_path.is_file():
        return cache
    for line in trials_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        tau_key = rec.get("tau_key")
        if isinstance(tau_key, str):
            cache[tau_key] = rec
    return cache


def _tau_key(tau: float) -> str:
    # Use a stable, human-readable key.
    return f"{tau:.10f}".rstrip("0").rstrip(".")


def reset_trials_file(trials_path: Path) -> Optional[str]:
    """
    Truncate trials.jsonl. If it already exists and is non-empty, keep a timestamped backup.
    Returns the backup path as a string (or None if no backup was created).
    """
    if not trials_path.exists():
        return None
    try:
        size = trials_path.stat().st_size
    except Exception:
        size = 0
    if size <= 0:
        # Just truncate to be safe.
        trials_path.write_text("")
        return None
    backup = trials_path.with_name(f"{trials_path.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}{trials_path.suffix}")
    shutil.copy2(trials_path, backup)
    trials_path.write_text("")
    return str(backup)


def evaluate_tau_avg_r(
    model: CodecLightningModule,
    dataloader: DataLoader,
    device_type: str,
    fixed_tau: float,
    max_samples: Optional[int],
    progress: bool,
) -> Dict[str, object]:
    dtp_module = model.dtp
    _set_fixed_tau(dtp_module, fixed_tau)

    params = list(model.parameters())
    model_device = params[0].device if params else torch.device("cpu")

    avg_r_values: List[float] = []
    tau_used_values: List[float] = []
    processed = 0

    iterator = dataloader
    if progress:
        iterator = tqdm(dataloader, total=len(dataloader), desc=f"tau={fixed_tau:.6f}", leave=False)

    t0 = time.time()
    for batch in iterator:
        wav = batch["wav"].to(model_device)
        _, avg_r_val, tau_used = run_generator_forward(model, wav, device_type)
        avg_r_values.append(float(avg_r_val))
        tau_used_values.append(float(tau_used))
        processed += 1
        if max_samples is not None and processed >= int(max_samples):
            break
    dt = time.time() - t0

    summary = _summarize(avg_r_values)
    tau_used_summary = _summarize(tau_used_values)

    return {
        "fixed_tau": float(fixed_tau),
        "tau_key": _tau_key(float(fixed_tau)),
        "avg_r_summary": summary,
        "avg_r_mean": (summary["mean"] if summary is not None else None),
        "tau_used_summary": tau_used_summary,
        "num_sequences": int(processed),
        "seconds": float(dt),
        "device_type": str(device_type),
        "tau_state": get_tau_state(dtp_module),
    }


def bootstrap_tau_with_update_test_time(
    model: CodecLightningModule,
    dataloader: DataLoader,
    device_type: str,
    max_samples: Optional[int],
) -> Dict[str, object]:
    """
    Run one full pass with update_test_time=True and fixed_tau=None so the internal controller
    adapts tau_eval towards the target r. The resulting tau_end is a good initial guess.
    """
    dtp_module = model.dtp
    tau_state_before = get_tau_state(dtp_module)

    _clear_fixed_tau_enable_update_test_time(dtp_module)

    params = list(model.parameters())
    model_device = params[0].device if params else torch.device("cpu")

    avg_r_values: List[float] = []
    tau_used_values: List[float] = []
    processed = 0

    t0 = time.time()
    for batch in tqdm(dataloader, total=len(dataloader), desc="bootstrap(update_test_time)", leave=False):
        wav = batch["wav"].to(model_device)
        _, avg_r_val, tau_used = run_generator_forward(model, wav, device_type)
        avg_r_values.append(float(avg_r_val))
        tau_used_values.append(float(tau_used))
        processed += 1
        if max_samples is not None and processed >= int(max_samples):
            break
    dt = time.time() - t0

    tau_state_after = get_tau_state(dtp_module)
    prefer_eval_tau = bool(getattr(dtp_module, "update_test_time", False))
    tau_start = pick_tau_value(tau_state_before, prefer_eval_tau)
    tau_end = pick_tau_value(tau_state_after, prefer_eval_tau)

    return {
        "num_sequences": int(processed),
        "seconds": float(dt),
        "avg_r_summary": _summarize(avg_r_values),
        "tau_used_summary": _summarize(tau_used_values),
        "tau_state_before": tau_state_before,
        "tau_state_after": tau_state_after,
        "tau_progress": {"start": tau_start, "end": tau_end},
    }


def _round_tau_to_step(tau: float, tau_step: float) -> float:
    if tau <= 0.0:
        return float(tau_step)
    idx = int(round(float(tau) / float(tau_step)))
    if idx < 1:
        idx = 1
    return _idx_to_tau(idx, tau_step)


def _expand_tau_bounds_if_needed(
    model: CodecLightningModule,
    dataloader: DataLoader,
    device_type: str,
    cfg: SearchConfig,
    trials_path: Path,
    cache: Dict[str, Dict[str, object]],
    lo_idx: int,
    hi_idx: int,
    target: float,
) -> Tuple[int, int]:
    """
    If target cannot be bracketed in [lo_idx, hi_idx], optionally expand the interval.
    Expansion is discrete in tau_step and grows hi_idx (or lo_idx) multiplicatively.
    """

    def eval_idx(idx: int) -> Dict[str, object]:
        tau = _idx_to_tau(idx, cfg.tau_step)
        key = _tau_key(tau)
        if key in cache:
            return cache[key]
        rec = evaluate_tau_avg_r(
            model=model,
            dataloader=dataloader,
            device_type=device_type,
            fixed_tau=tau,
            max_samples=cfg.max_samples,
            progress=False,
        )
        rec["idx"] = int(idx)
        rec["target_avg_r"] = float(cfg.target_avg_r)
        rec["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        trials_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trials_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        cache[key] = rec
        return rec

    rec_lo = eval_idx(lo_idx)
    rec_hi = eval_idx(hi_idx)
    a_lo = float(rec_lo["avg_r_mean"])
    a_hi = float(rec_hi["avg_r_mean"])

    # If already bracketed, return.
    if (a_lo - target) * (a_hi - target) <= 0.0:
        return lo_idx, hi_idx

    if not cfg.auto_expand:
        return lo_idx, hi_idx

    # Determine a coarse direction by probing a mid-point.
    mid = (lo_idx + hi_idx) // 2
    a_mid = float(eval_idx(mid)["avg_r_mean"])
    increasing = a_hi >= a_mid >= a_lo or (a_hi >= a_lo and a_mid >= min(a_lo, a_hi))
    decreasing = not increasing

    # Expand the side that could potentially cross the target.
    max_idx = int(np.floor(float(cfg.auto_expand_max_tau) / float(cfg.tau_step)))
    if max_idx < 1:
        max_idx = 1

    # If both above target, we need to move towards smaller avg_r.
    # - If increasing: shrink tau (move lo down) but lo is already minimal.
    # - If decreasing: increase tau (move hi up).
    # If both below target, the opposite.
    while True:
        rec_lo = eval_idx(lo_idx)
        rec_hi = eval_idx(hi_idx)
        a_lo = float(rec_lo["avg_r_mean"])
        a_hi = float(rec_hi["avg_r_mean"])
        if (a_lo - target) * (a_hi - target) <= 0.0:
            return lo_idx, hi_idx

        if a_lo >= target and a_hi >= target:
            if increasing:
                # Can't go lower than lo_idx=1 in this discrete domain.
                return lo_idx, hi_idx
            # decreasing: raise hi
            if hi_idx >= max_idx:
                return lo_idx, hi_idx
            hi_idx = min(max_idx, int(max(hi_idx + 1, hi_idx * 2)))
            continue

        if a_lo < target and a_hi < target:
            if decreasing:
                # Can't go lower than lo_idx=1.
                return lo_idx, hi_idx
            # increasing: raise hi (since higher tau increases avg_r)
            if hi_idx >= max_idx:
                return lo_idx, hi_idx
            hi_idx = min(max_idx, int(max(hi_idx + 1, hi_idx * 2)))
            continue

        return lo_idx, hi_idx


def _is_monotonic_increasing(a_lo: float, a_hi: float) -> bool:
    # If equal, default to increasing to get "smallest tau" behavior.
    return a_hi >= a_lo


def search_fixed_tau(
    model: CodecLightningModule,
    dataloader: DataLoader,
    device_type: str,
    cfg: SearchConfig,
    trials_path: Path,
    no_resume: bool,
) -> Dict[str, object]:
    _ensure_tau_range(cfg)
    lo_idx, hi_idx = _index_bounds(cfg)

    cache = {} if no_resume else _load_existing_trials(trials_path)

    def eval_idx(idx: int) -> Dict[str, object]:
        tau = _idx_to_tau(idx, cfg.tau_step)
        key = _tau_key(tau)
        if key in cache:
            return cache[key]
        rec = evaluate_tau_avg_r(
            model=model,
            dataloader=dataloader,
            device_type=device_type,
            fixed_tau=tau,
            max_samples=cfg.max_samples,
            progress=False,
        )
        rec["idx"] = int(idx)
        rec["target_avg_r"] = float(cfg.target_avg_r)
        rec["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        trials_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trials_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        cache[key] = rec
        return rec

    bootstrap_info: Optional[Dict[str, object]] = None
    if cfg.bootstrap_update_test_time:
        # Optionally override update_test_time via model mutation even if config doesn't include it.
        if cfg.bootstrap_override_update_test_time:
            _clear_fixed_tau_enable_update_test_time(model.dtp)
        bootstrap_info = bootstrap_tau_with_update_test_time(
            model=model,
            dataloader=dataloader,
            device_type=device_type,
            max_samples=cfg.max_samples,
        )

        tau_end = bootstrap_info.get("tau_progress", {}).get("end")
        if tau_end is not None:
            tau_guess = _round_tau_to_step(float(tau_end), cfg.tau_step)
            guess_idx = int(round(tau_guess / cfg.tau_step))
            if guess_idx < 1:
                guess_idx = 1
            lo_idx = min(lo_idx, guess_idx)
            hi_idx = max(hi_idx, guess_idx)

            # Critical: always evaluate the bootstrap guess as a fixed_tau trial.
            # This prevents the search from only probing endpoints and helps debugging.
            _ = eval_idx(guess_idx)

        if cfg.bootstrap_only:
            tau_end = bootstrap_info.get("tau_progress", {}).get("end")
            if tau_end is None:
                raise RuntimeError("Bootstrap enabled but tau_end is None.")
            tau_guess = _round_tau_to_step(float(tau_end), cfg.tau_step)
            best_rec = evaluate_tau_avg_r(
                model=model,
                dataloader=dataloader,
                device_type=device_type,
                fixed_tau=tau_guess,
                max_samples=cfg.max_samples,
                progress=False,
            )
            return {
                "search_config": {
                    "target_avg_r": cfg.target_avg_r,
                    "tau_min": cfg.tau_min,
                    "tau_max": cfg.tau_max,
                    "tau_step": cfg.tau_step,
                    "max_samples": cfg.max_samples,
                    "length_mode": cfg.length_mode,
                    "bootstrap_update_test_time": cfg.bootstrap_update_test_time,
                    "bootstrap_only": cfg.bootstrap_only,
                },
                "bootstrap": bootstrap_info,
                "best": {
                    "fixed_tau": float(tau_guess),
                    "avg_r_summary": best_rec.get("avg_r_summary"),
                    "avg_r_mean": best_rec.get("avg_r_mean"),
                    "tau_state": best_rec.get("tau_state"),
                },
                "recommended_cfg_override": f"model.resampler.dtp_params.fixed_tau={tau_guess}",
            }

    # Evaluate endpoints to determine feasibility and monotonic direction.
    target = float(cfg.target_avg_r)

    # Helper: fetch avg_r_mean for an idx.
    def val_at(idx: int) -> float:
        rec = eval_idx(idx)
        v = rec.get("avg_r_mean")
        if v is None:
            return float("nan")
        return float(v)

    # Helper: check predicate.
    def meets(idx: int) -> bool:
        v = val_at(idx)
        return bool(np.isfinite(v) and v >= target)

    # ------------------------------------------------------------------
    # Core logic: find tau that yields the smallest avg_r >= target.
    #
    # For monotonic increasing avg_r(tau):
    #   - set S = {tau | avg_r(tau) >= target} is an interval [tau*, +inf)
    #   - best is the *smallest tau* in S (first idx that meets)
    #
    # For monotonic decreasing avg_r(tau):
    #   - S is (-inf, tau*]
    #   - best is the *largest tau* in S (last idx that meets)
    #
    # Direction is inferred from evaluations (prefer near bootstrap tau if available).
    # ------------------------------------------------------------------

    def infer_direction_around(idx0: int) -> bool:
        """
        Infer monotonic direction with ONE additional evaluation at a bigger step than 1.
        Returns True if avg_r increases with tau, else False.
        """
        step = int(cfg.direction_probe_step)
        i0 = int(min(max(idx0, lo_idx), hi_idx))
        i1 = int(min(max(i0 + step, lo_idx), hi_idx))
        if i1 == i0:
            i1 = int(min(max(i0 - step, lo_idx), hi_idx))
        if i1 == i0:
            # Degenerate (range size 1)
            return True
        v0 = val_at(i0)
        v1 = val_at(i1)
        if not (np.isfinite(v0) and np.isfinite(v1)):
            raise RuntimeError("Direction probe produced non-finite avg_r_mean.")
        return v1 > v0

    # Prefer to infer direction around the bootstrap tau_end (if present).
    guess_idx_for_dir: Optional[int] = None
    if cfg.bootstrap_update_test_time and bootstrap_info is not None:
        tau_end = bootstrap_info.get("tau_progress", {}).get("end")
        if tau_end is not None:
            tau_guess = _round_tau_to_step(float(tau_end), cfg.tau_step)
            guess_idx_for_dir = int(round(tau_guess / cfg.tau_step))
            if guess_idx_for_dir < lo_idx:
                guess_idx_for_dir = lo_idx
            if guess_idx_for_dir > hi_idx:
                guess_idx_for_dir = hi_idx

    # If no bootstrap, fall back to endpoints to ensure we can infer direction and feasibility.
    if guess_idx_for_dir is None:
        lo_idx, hi_idx = _expand_tau_bounds_if_needed(
            model=model,
            dataloader=dataloader,
            device_type=device_type,
            cfg=cfg,
            trials_path=trials_path,
            cache=cache,
            lo_idx=lo_idx,
            hi_idx=hi_idx,
            target=float(cfg.target_avg_r),
        )
        a_lo = val_at(lo_idx)
        a_hi = val_at(hi_idx)
        if not (np.isfinite(a_lo) and np.isfinite(a_hi)):
            raise RuntimeError("avg_r_mean is not finite at tau bounds; cannot run search.")
        increasing = _is_monotonic_increasing(a_lo, a_hi)
        start_idx = (lo_idx + hi_idx) // 2
    else:
        start_idx = int(min(max(guess_idx_for_dir, lo_idx), hi_idx))
        increasing = infer_direction_around(start_idx)
        a_lo = float("nan")
        a_hi = float("nan")

    # Bracket the threshold crossing around start_idx using exponential search.
    # We want a pair (i_good, i_bad) such that:
    # - increasing: good = meets, bad = not meets, and bad < good
    # - decreasing: good = meets, bad = not meets, and good < bad
    def bracket_threshold() -> Tuple[int, int]:
        m0 = meets(start_idx)
        step0 = int(cfg.direction_probe_step)
        step = max(1, step0)

        if increasing:
            if m0:
                # Need a NOT-meeting point on the left.
                good = start_idx
                bad = start_idx
                while bad > lo_idx and meets(bad):
                    bad = max(lo_idx, bad - step)
                    step *= 2
                return bad, good
            # Need a meeting point on the right.
            bad = start_idx
            good = start_idx
            while good < hi_idx and (not meets(good)):
                good = min(hi_idx, good + step)
                step *= 2
            return bad, good

        # decreasing
        if m0:
            # Need a NOT-meeting point on the right.
            good = start_idx
            bad = start_idx
            while bad < hi_idx and meets(bad):
                bad = min(hi_idx, bad + step)
                step *= 2
            return good, bad
        # Need a meeting point on the left.
        bad = start_idx
        good = start_idx
        while good > lo_idx and (not meets(good)):
            good = max(lo_idx, good - step)
            step *= 2
        return good, bad

    L, R = bracket_threshold()
    mL, mR = meets(L), meets(R)

    # If we failed to bracket (both same), then either:
    # - Everything meets (best is boundary depending on direction), or
    # - Nothing meets (target not achievable in range).
    if mL == mR:
        if mL:
            best_idx = hi_idx if (not increasing) else lo_idx
        else:
            raise ValueError(
                f"Target avg_r={target} is not achievable in [{_idx_to_tau(lo_idx,cfg.tau_step):.6f}, {_idx_to_tau(hi_idx,cfg.tau_step):.6f}]. "
                f"Try adjusting tau_min/tau_max or check that fixed_tau applies for this dtp_cls."
            )
    else:
        # Binary search within bracket for boundary.
        if increasing:
            # Find first idx that meets (lower bound).
            left, right = L, R
            while left < right:
                mid = (left + right) // 2
                if meets(mid):
                    right = mid
                else:
                    left = mid + 1
            best_idx = left
        else:
            # Find last idx that meets (upper bound).
            left, right = L, R
            while left < right:
                mid = (left + right + 1) // 2
                if meets(mid):
                    left = mid
                else:
                    right = mid - 1
            best_idx = left

    best_tau = _idx_to_tau(best_idx, cfg.tau_step)
    best_rec = eval_idx(best_idx)

    # Neighbor checks (useful for sanity): the previous/next tau around the boundary.
    neighbor_recs: Dict[str, Optional[Dict[str, object]]] = {"prev": None, "next": None}
    if best_idx - 1 >= lo_idx:
        neighbor_recs["prev"] = eval_idx(best_idx - 1)
    if best_idx + 1 <= hi_idx:
        neighbor_recs["next"] = eval_idx(best_idx + 1)

    return {
        "search_config": {
            "target_avg_r": cfg.target_avg_r,
            "tau_min": cfg.tau_min,
            "tau_max": cfg.tau_max,
            "tau_step": cfg.tau_step,
            "max_samples": cfg.max_samples,
            "length_mode": cfg.length_mode,
            "bootstrap_update_test_time": cfg.bootstrap_update_test_time,
            "bootstrap_only": cfg.bootstrap_only,
            "auto_expand": cfg.auto_expand,
            "auto_expand_max_tau": cfg.auto_expand_max_tau,
        },
        "bootstrap": bootstrap_info,
        "bounds": {
            "lo_tau": _idx_to_tau(lo_idx, cfg.tau_step),
            "hi_tau": _idx_to_tau(hi_idx, cfg.tau_step),
            "lo_avg_r_mean": a_lo,
            "hi_avg_r_mean": a_hi,
            "monotonic_increasing": bool(increasing),
            "search_bracket": {
                "L_idx": int(L),
                "R_idx": int(R),
                "L_tau": _idx_to_tau(L, cfg.tau_step),
                "R_tau": _idx_to_tau(R, cfg.tau_step),
                "L_meets": bool(mL),
                "R_meets": bool(mR),
                "start_idx": int(start_idx),
                "direction_probe_step": int(cfg.direction_probe_step),
            },
        },
        "best": {
            "fixed_tau": float(best_tau),
            "idx": int(best_idx),
            "avg_r_summary": best_rec.get("avg_r_summary"),
            "avg_r_mean": best_rec.get("avg_r_mean"),
            "tau_state": best_rec.get("tau_state"),
        },
        "neighbors": neighbor_recs,
        "trials_path": str(trials_path),
        "recommended_cfg_override": f"model.resampler.dtp_params.fixed_tau={best_tau}",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Search fixed_tau to match a target avg_r (reduction ratio) with minimal iterations."
    )
    p.add_argument("--input", type=str, required=True, help="Directory, single file, or .txt filelist.")
    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory containing hydra/config.yaml and pl_log/last.ckpt.",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--length_mode", type=str, choices=["pad", "truncate"], default="pad")
    p.add_argument("--max_samples", type=int, default=None, help="Optional max number of sequences per tau evaluation.")

    p.add_argument("--target_avg_r", type=float, default=0.5, help="Target avg_r (reduction ratio) to reach.")
    p.add_argument("--tau_min", type=float, default=0.001, help="Minimum fixed_tau (must be > 0).")
    p.add_argument("--tau_max", type=float, default=1.0, help="Maximum fixed_tau.")
    p.add_argument("--tau_step", type=float, default=0.001, help="Search step size for fixed_tau.")

    p.add_argument(
        "--bootstrap_update_test_time",
        action="store_true",
        help="Before searching, run one pass with update_test_time=True (and fixed_tau=None) to get a good tau guess.",
    )
    p.add_argument(
        "--bootstrap_only",
        action="store_true",
        help="Only run the update_test_time bootstrap and then evaluate fixed_tau=round(tau_end). No binary search.",
    )
    p.add_argument(
        "--bootstrap_override_update_test_time",
        action="store_true",
        help="Force update_test_time=True by mutating the model, even if cfg doesn't include dtp_params.update_test_time.",
    )
    p.add_argument(
        "--auto_expand",
        action="store_true",
        help="If target cannot be bracketed in [tau_min, tau_max], try expanding tau_max (discrete step) up to auto_expand_max_tau.",
    )
    p.add_argument(
        "--auto_expand_max_tau",
        type=float,
        default=100.0,
        help="Maximum tau allowed when --auto_expand is enabled.",
    )
    p.add_argument(
        "--direction_probe_step",
        type=int,
        default=16,
        help="Step (in tau grid indices) used to infer monotonic direction and as the initial exponential bracketing step.",
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write search summary and trials (default: run_dir/eval/dtp_stats_search).",
    )
    p.add_argument(
        "--cfg_override",
        action="append",
        default=None,
        help="Hydra-style dotlist override applied after loading hydra/config.yaml "
        "(e.g., --cfg_override dataset.multiple_of=160). Use multiple flags for multiple overrides.",
    )
    p.add_argument("--no_resume", action="store_true", help="Do not reuse existing trials.jsonl if present.")
    return p


def main() -> None:
    args = build_parser().parse_args()

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
    model = CodecLightningModule(cfg=cfg).to(device)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    patch_legacy_dtp_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        print(f"[Warning] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if not getattr(model, "use_dtp", False):
        raise RuntimeError("The loaded model does not enable DTP (use_dtp=False).")
    model.eval()

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "eval" / "dtp_stats_search")
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"
    summary_path = out_dir / "summary.json"

    # If --no_resume is set, also reset trials.jsonl so the output directory is effectively "fresh".
    # This matches the intuitive expectation that we won't carry over stale tau measurements.
    backup_trials: Optional[str] = None
    if bool(args.no_resume):
        backup_trials = reset_trials_file(trials_path)

    search_cfg = SearchConfig(
        target_avg_r=float(args.target_avg_r),
        tau_min=float(args.tau_min),
        tau_max=float(args.tau_max),
        tau_step=float(args.tau_step),
        max_samples=(int(args.max_samples) if args.max_samples is not None else None),
        length_mode=str(args.length_mode),
        bootstrap_update_test_time=bool(args.bootstrap_update_test_time),
        bootstrap_only=bool(args.bootstrap_only),
        bootstrap_override_update_test_time=bool(args.bootstrap_override_update_test_time),
        auto_expand=bool(args.auto_expand),
        auto_expand_max_tau=float(args.auto_expand_max_tau),
        direction_probe_step=int(args.direction_probe_step),
    )

    device_type = device.type
    result = search_fixed_tau(
        model=model,
        dataloader=dataloader,
        device_type=device_type,
        cfg=search_cfg,
        trials_path=trials_path,
        no_resume=bool(args.no_resume),
    )

    if backup_trials is not None:
        result["trials_backup"] = backup_trials

    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print("\nUse this in your eval/dtp_stats.py or eval/eval.py runs:")
    print(f"  --cfg_override {result['recommended_cfg_override']}")


if __name__ == "__main__":
    main()


