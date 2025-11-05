import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable, Dict, Optional
import math

import numpy as np

class PLEBatchTopK_old(nn.Module):
    """
    Dense PLE (Path-Length Equalization) selection with optional under-selection (no fallback).

    - r is the prune ratio; per sequence target M = N - floor(r*N).
    - Always keep the first token per sequence; interior boundaries are chosen by
      first-crossing over equal path-length bins (no clamping, no fallback). If a bin
      produces no unique boundary, it contributes nothing (at most M kept overall).

    Input (dense):
      x: [B, N, C]

    Returns:
      mask: [B, N] 0/1 (or bool) frontier indicator over original length per sequence
      avg_r: scalar tensor = (#zeros in mask) / (#total original tokens)
      tau_used: scalar tensor (mean bin width for logging; not used in training)
    """
    def __init__(self, r: float, momentum: float = 0.99):
        super().__init__()
        if not (0.0 <= float(r) < 1.0):
            raise ValueError("PLEBatchTopK_old: r must be in [0, 1)")
        self.r = float(r)
        self.momentum = float(momentum)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        B, N, C = x.shape if x.ndim == 3 else (0, 0, 0)
        total = B * N
        if total == 0:
            mask = torch.zeros(B, N, device=device, dtype=torch.bool)
            avg_r = torch.zeros((), device=device, dtype=dtype)
            tau_used = torch.zeros((), device=device, dtype=dtype)
            return mask, avg_r, tau_used

        # Adjacent dissimilarities per sequence: d[:, 0] = 0, d[:, t] = 1 - cos(x[:,t], x[:,t-1])
        if N > 1:
            sim = F.cosine_similarity(x[:, 1:, :], x[:, :-1, :], dim=-1)
            d = torch.zeros(B, N, device=device, dtype=dtype)
            d[:, 1:] = (1.0 - sim).to(dtype)
        else:
            d = torch.zeros(B, N, device=device, dtype=dtype)

        # Per-sequence target kept count and first-crossing selection (no fallback)
        M = int(N - math.floor(self.r * N)) if N > 0 else 0
        interior = max(M - 1, 0)

        mask = torch.zeros(B, N, device=device, dtype=torch.bool)
        if N > 0:
            mask[:, 0] = True
        tau_used = torch.zeros((), device=device, dtype=dtype)
        if interior > 0 and N > 1:
            # Build cumulative path D with D[:,0]=0 and D accumulating d from index 1
            D = torch.zeros(B, N, device=device, dtype=dtype)
            D[:, 1:] = torch.cumsum(d[:, 1:], dim=1)
            L = D[:, -1]

            # Bin width per sequence (avoid division by zero)
            t_b = torch.where(L > 0, L / float(M), torch.ones_like(L))

            # First-crossing positions j in [1..N-1] per interior boundary
            targets = (torch.arange(1, M, device=device, dtype=dtype).view(1, -1) * t_b.view(B, 1))
            ge = D.unsqueeze(2) >= targets.unsqueeze(1)  # (B, N, interior)
            j = ge.float().argmax(dim=1).clamp_(min=1, max=N - 1)  # (B, interior)

            # Enforce strictly increasing j per row
            ar = torch.arange(interior, device=device, dtype=j.dtype).view(1, -1)
            s = (j - ar)
            smax, _ = torch.cummax(s, dim=1)
            j_strict = (smax + ar).clamp_(min=1, max=N - 1)

            mask.scatter_(1, j_strict, True)
            tau_used = t_b.mean()

        kept_counts = mask.sum(dim=1).to(torch.long)

        kept_total = int(kept_counts.sum().item())
        zeros_total = int(total - kept_total)
        avg_r = torch.tensor(float(zeros_total) / float(max(1, total)), device=device, dtype=dtype)

        return mask, avg_r, tau_used


class PLEBatchTopK(nn.Module):
    """
    Batch-level PLE with absolute threshold (tau) and EMA.

    - Training: choose a single global tau to target K_target ≈ (1-r)*total_tokens
      across the whole batch (not per-sequence). We estimate
        tau_batch = sum(L_b) / max(1, K_target - B)
      so that sum_b floor(L_b / tau_batch) ≲ K_target - B. No bisection.
    - Inference: use stored tau_ema.
    - Always keep token 0 for every sequence; interior boundaries via first-crossing
      with D >= k * tau.

    Input:
      x: [B, N, C]

    Returns:
      mask: [B, N] bool frontier indicator
      avg_r: scalar = (#zeros in mask) / (B*N)
      tau_used: scalar tensor (tau_batch during training, tau_ema during eval)
    """
    def __init__(self, r: float, momentum: float = 0.99):
        super().__init__()
        if not (0.0 <= float(r) < 1.0):
            raise ValueError("PLEBatchTopK: r must be in [0, 1)")
        self.r = float(r)
        self.momentum = float(momentum)
        self.register_buffer("tau_ema", torch.tensor(0.0))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        B, N, C = x.shape if x.ndim == 3 else (0, 0, 0)
        total = B * N
        if total == 0:
            mask = torch.zeros(B, N, device=device, dtype=torch.bool)
            avg_r = torch.zeros((), device=device, dtype=dtype)
            tau_used = self.tau_ema.clone().to(device=device, dtype=dtype)
            return mask, avg_r, tau_used

        # Adjacent dissimilarities and cumulative path per sequence
        if N > 1:
            sim = F.cosine_similarity(x[:, 1:, :], x[:, :-1, :], dim=-1)
            d = torch.zeros(B, N, device=device, dtype=dtype)
            d[:, 1:] = (1.0 - sim).to(dtype)
        else:
            d = torch.zeros(B, N, device=device, dtype=dtype)

        D = torch.zeros(B, N, device=device, dtype=dtype)
        if N > 1:
            D[:, 1:] = torch.cumsum(d[:, 1:], dim=1)
        L = D[:, -1] if N > 0 else torch.zeros(B, device=device, dtype=dtype)

        # Target kept tokens across batch
        K_target = int(math.floor((1.0 - self.r) * float(total)))
        K_target = max(0, min(K_target, total))
        K_extra = max(0, K_target - B/2)  # subtract first-token keeps

        # Determine tau
        if self.training:
            sum_L = L.sum()
            if K_extra > 0 and sum_L.item() > 0.0:
                tau_batch = (sum_L / float(K_extra)).to(dtype)
            else:
                tau_batch = torch.tensor(float('inf'), device=device, dtype=dtype)

            # EMA update (bootstrap on first step)
            if torch.isfinite(tau_batch):
                if self.tau_ema.item() == 0.0:
                    self.tau_ema.copy_(tau_batch)
                else:
                    self.tau_ema.mul_(self.momentum).add_(tau_batch * (1.0 - self.momentum))
            tau_used = tau_batch
        else:
            tau_used = self.tau_ema.to(device=device, dtype=dtype)

        # Build frontier mask via first-crossing with global tau
        mask = torch.zeros(B, N, device=device, dtype=torch.bool)
        if N > 0:
            mask[:, 0] = True

        if N > 1 and torch.isfinite(tau_used) and (tau_used.item() > 0.0):
            # m_b = floor(L_b / tau)
            m_b = torch.floor_divide((L / tau_used).to(torch.long), torch.ones_like(L, dtype=torch.long))
            # clamp m_b to at most N-1
            m_b = torch.clamp(m_b, min=0, max=N - 1)
            max_m = int(m_b.max().item())
            if max_m > 0:
                targets = (torch.arange(1, max_m + 1, device=device, dtype=dtype) * tau_used).view(1, -1)
                # ge: (B, N, max_m)
                ge = D.unsqueeze(2) >= targets.view(1, 1, -1)
                j = ge.float().argmax(dim=1).clamp_(min=1, max=N - 1)  # (B, max_m)

                # strictly increasing per row
                ar = torch.arange(max_m, device=device, dtype=j.dtype).view(1, -1)
                s = (j - ar)
                smax, _ = torch.cummax(s, dim=1)
                j_strict = (smax + ar).clamp_(min=1, max=N - 1)

                # apply only for valid slots k <= m_b
                k_idx = torch.arange(1, max_m + 1, device=device).view(1, -1).expand(B, -1)
                valid = k_idx <= m_b.view(B, 1)
                if valid.any():
                    sel = valid.nonzero(as_tuple=False)
                    b_sel = sel[:, 0]
                    k_sel = sel[:, 1]
                    pos = j_strict[b_sel, k_sel]
                    mask[b_sel, pos] = True

        kept_total = int(mask.sum().item())
        zeros_total = int(total - kept_total)
        avg_r = torch.tensor(float(zeros_total) / float(max(1, total)), device=device, dtype=dtype)

        return mask, avg_r, tau_used