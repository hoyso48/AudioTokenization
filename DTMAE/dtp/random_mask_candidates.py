from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


def keep_prob_from_target_r(target_r: float, seq_len: int) -> float:
    """Compute keep probability for tokens 1..N-1, with token 0 always kept."""
    if seq_len <= 1:
        return 1.0
    target_keep = float(seq_len) * (1.0 - float(target_r))
    q = (target_keep - 1.0) / float(seq_len - 1)
    return float(np.clip(q, 0.0, 1.0))


def p_start_from_r_and_mean_span(target_r: float, mean_span: float) -> float:
    """Start probability for independent-start interval coverage.

    Poisson-style approximation:
      r ~= 1 - exp(-p_start * mean_span)
      p_start ~= -log(1-r) / mean_span
    """
    r = float(np.clip(target_r, 1e-6, 1.0 - 1e-6))
    m = max(1e-6, float(mean_span))
    p = -np.log1p(-r) / m
    return float(np.clip(p, 1e-6, 1.0))


@dataclass
class MaskCandidate:
    name: str
    kind: str
    params: Dict[str, float]


def build_default_candidates() -> List[MaskCandidate]:
    return [
        MaskCandidate("iid_bernoulli", "iid_bernoulli", {}),
        MaskCandidate("beta_bernoulli_k8", "beta_bernoulli", {"kappa": 8.0}),
        MaskCandidate("beta_bernoulli_k20", "beta_bernoulli", {"kappa": 20.0}),
        MaskCandidate("start_fixed_m4", "start_fixed_span", {"span": 4.0}),
        MaskCandidate("start_geom_m2", "start_geom_span", {"mean_span": 2.0}),
        MaskCandidate("start_geom_m4", "start_geom_span", {"mean_span": 4.0}),
        MaskCandidate("start_geom_m8", "start_geom_span", {"mean_span": 8.0}),
        MaskCandidate(
            "start_geom_m4_jitter36",
            "start_geom_span_jitter",
            {"mean_span_min": 3.0, "mean_span_max": 6.0},
        ),
    ]


def sample_mask_iid_bernoulli(
    batch_size: int,
    seq_len: int,
    target_r: float,
    rng: np.random.Generator,
) -> np.ndarray:
    q = keep_prob_from_target_r(target_r, seq_len)
    mask = rng.random((batch_size, seq_len)) < q
    if seq_len > 0:
        mask[:, 0] = True
    return mask


def sample_mask_beta_bernoulli(
    batch_size: int,
    seq_len: int,
    target_r: float,
    rng: np.random.Generator,
    kappa: float,
) -> np.ndarray:
    q_mean = keep_prob_from_target_r(target_r, seq_len)
    k = max(1e-6, float(kappa))
    alpha = max(1e-6, q_mean * k)
    beta = max(1e-6, (1.0 - q_mean) * k)
    q_b = rng.beta(alpha, beta, size=(batch_size, 1))
    mask = rng.random((batch_size, seq_len)) < q_b
    if seq_len > 0:
        mask[:, 0] = True
    return mask


def _mask_from_intervals(
    batch_size: int,
    seq_len: int,
    start_rows: np.ndarray,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
) -> np.ndarray:
    mask = np.ones((batch_size, seq_len), dtype=bool)
    if seq_len <= 1 or start_rows.size == 0:
        if seq_len > 0:
            mask[:, 0] = True
        return mask

    diff = np.zeros((batch_size, seq_len + 1), dtype=np.int32)
    np.add.at(diff, (start_rows, start_pos), 1)
    np.add.at(diff, (start_rows, end_pos + 1), -1)

    covered = np.cumsum(diff[:, :seq_len], axis=1) > 0
    mask = ~covered
    mask[:, 0] = True
    return mask


def sample_mask_start_fixed_span(
    batch_size: int,
    seq_len: int,
    target_r: float,
    rng: np.random.Generator,
    span: float,
) -> np.ndarray:
    if seq_len <= 1:
        mask = np.ones((batch_size, seq_len), dtype=bool)
        if seq_len > 0:
            mask[:, 0] = True
        return mask

    T = seq_len - 1
    m = max(1.0, float(span))
    p_start = p_start_from_r_and_mean_span(target_r, m)

    starts = rng.random((batch_size, T)) < p_start
    rows, t = np.nonzero(starts)
    if rows.size == 0:
        mask = np.ones((batch_size, seq_len), dtype=bool)
        mask[:, 0] = True
        return mask

    start_pos = t + 1
    span_len = int(round(m))
    span_len = max(1, span_len)
    end_pos = np.minimum(seq_len - 1, start_pos + span_len - 1)
    return _mask_from_intervals(batch_size, seq_len, rows, start_pos, end_pos)


def sample_mask_start_geom_span(
    batch_size: int,
    seq_len: int,
    target_r: float,
    rng: np.random.Generator,
    mean_span: float,
) -> np.ndarray:
    if seq_len <= 1:
        mask = np.ones((batch_size, seq_len), dtype=bool)
        if seq_len > 0:
            mask[:, 0] = True
        return mask

    T = seq_len - 1
    m = max(1.0, float(mean_span))
    p_start = p_start_from_r_and_mean_span(target_r, m)

    starts = rng.random((batch_size, T)) < p_start
    rows, t = np.nonzero(starts)
    if rows.size == 0:
        mask = np.ones((batch_size, seq_len), dtype=bool)
        mask[:, 0] = True
        return mask

    p_end = float(np.clip(1.0 / m, 1e-6, 1.0))
    span_len_all = rng.geometric(p_end, size=(batch_size, T))
    start_pos = t + 1
    end_pos = np.minimum(seq_len - 1, start_pos + span_len_all[rows, t] - 1)
    return _mask_from_intervals(batch_size, seq_len, rows, start_pos, end_pos)


def sample_mask_start_geom_span_jitter(
    batch_size: int,
    seq_len: int,
    target_r: float,
    rng: np.random.Generator,
    mean_span_min: float,
    mean_span_max: float,
) -> np.ndarray:
    if seq_len <= 1:
        mask = np.ones((batch_size, seq_len), dtype=bool)
        if seq_len > 0:
            mask[:, 0] = True
        return mask

    T = seq_len - 1
    m_lo = max(1.0, float(mean_span_min))
    m_hi = max(m_lo, float(mean_span_max))

    m_b = rng.uniform(m_lo, m_hi, size=(batch_size, 1))
    p_start_b = np.clip(-np.log1p(-np.clip(target_r, 1e-6, 1.0 - 1e-6)) / m_b, 1e-6, 1.0)
    starts = rng.random((batch_size, T)) < p_start_b
    rows, t = np.nonzero(starts)
    if rows.size == 0:
        mask = np.ones((batch_size, seq_len), dtype=bool)
        mask[:, 0] = True
        return mask

    p_end_b = np.clip(1.0 / m_b, 1e-6, 1.0)
    p_end_all = p_end_b.repeat(T, axis=1)
    u = np.clip(rng.random((batch_size, T)), 1e-12, 1.0 - 1e-12)
    span_len_all = np.floor(np.log1p(-u) / np.log1p(-p_end_all)).astype(np.int64) + 1

    start_pos = t + 1
    end_pos = np.minimum(seq_len - 1, start_pos + span_len_all[rows, t] - 1)
    return _mask_from_intervals(batch_size, seq_len, rows, start_pos, end_pos)


def sample_candidate_mask(
    candidate: MaskCandidate,
    batch_size: int,
    seq_len: int,
    target_r: float,
    rng: np.random.Generator,
) -> np.ndarray:
    kind = candidate.kind
    p = candidate.params
    if kind == "iid_bernoulli":
        return sample_mask_iid_bernoulli(batch_size, seq_len, target_r, rng)
    if kind == "beta_bernoulli":
        return sample_mask_beta_bernoulli(batch_size, seq_len, target_r, rng, kappa=float(p["kappa"]))
    if kind == "start_fixed_span":
        return sample_mask_start_fixed_span(batch_size, seq_len, target_r, rng, span=float(p["span"]))
    if kind == "start_geom_span":
        return sample_mask_start_geom_span(batch_size, seq_len, target_r, rng, mean_span=float(p["mean_span"]))
    if kind == "start_geom_span_jitter":
        return sample_mask_start_geom_span_jitter(
            batch_size,
            seq_len,
            target_r,
            rng,
            mean_span_min=float(p["mean_span_min"]),
            mean_span_max=float(p["mean_span_max"]),
        )
    raise ValueError(f"Unknown candidate kind: {kind}")
