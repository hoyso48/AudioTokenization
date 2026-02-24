from __future__ import annotations

from typing import List, Sequence

import torch


def mask_to_trailing_zeros(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a binary keep-mask into trailing zero counts per kept token.

    Example:
      mask = [1, 0, 0, 1, 0, 1]
      trailing = [2, 1, 0]

    This matches the "# trailing zeros" convention used in VFR setup.
    """
    flat = mask.to(torch.bool).flatten()
    if flat.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=flat.device)

    trailing_counts: List[int] = []
    zeros_after = 0
    for keep in reversed(flat.tolist()):
        if keep:
            trailing_counts.append(zeros_after)
            zeros_after = 0
        else:
            zeros_after += 1

    if not trailing_counts:
        return torch.zeros(0, dtype=torch.long, device=flat.device)
    trailing_counts.reverse()
    return torch.tensor(trailing_counts, dtype=torch.long, device=flat.device)


def trailing_zeros_to_span_lengths(
    trailing_zeros: torch.Tensor,
    max_span_len: int = 512,
) -> torch.Tensor:
    if max_span_len < 1:
        raise ValueError("max_span_len must be >= 1")
    span = trailing_zeros.to(torch.long) + 1
    return torch.clamp(span, min=1, max=max_span_len)


def mask_to_span_lengths(mask: torch.Tensor, max_span_len: int = 512) -> torch.Tensor:
    trailing = mask_to_trailing_zeros(mask)
    return trailing_zeros_to_span_lengths(trailing, max_span_len=max_span_len)


def span_lengths_to_mask(span_lengths: Sequence[int]) -> torch.Tensor:
    """
    Reconstruct a keep-mask from span lengths using [1, 0, ..., 0] pattern.
    """
    chunks: List[int] = []
    for s in span_lengths:
        if s < 1:
            raise ValueError("span length must be >= 1")
        chunks.append(1)
        if s > 1:
            chunks.extend([0] * (s - 1))
    if not chunks:
        return torch.zeros(0, dtype=torch.bool)
    return torch.tensor(chunks, dtype=torch.bool)
