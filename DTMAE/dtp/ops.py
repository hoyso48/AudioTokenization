import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable, Dict, Optional
import math

# class PLEBatchTopK(nn.Module):
#     """
#     Batch-level PLE (Path-Length Equalization) with feedback control.
#     Supports Test-Time Adaptation via update_test_time=True.

#     Design:
#       - Use one global scalar tau for both training and inference.
#       - Separate buffers for training (tau_train) and testing (tau_test).
#       - During training, update tau_train.
#       - During inference (eval mode):
#         - If update_test_time=True: Initialize tau_test from tau_train at the start
#           of eval session, then update tau_test adaptively.
#         - If update_test_time=False: Use tau_train without updating (fixed).
#     """
#     def __init__(
#         self,
#         r: float,
#         initial_tau: float = 1.0,
#         step_size: float = 0.01,
#         momentum: float = 0.9,
#         update_test_time: bool = False,
#         update_every: int = 1,
#         sample_prob: float = 0.0,
#     ):
#         super().__init__()
#         if not (0.0 <= float(r) < 1.0):
#             raise ValueError("PLEBatchTopK: r must be in [0, 1)")
#         if initial_tau <= 0.0:
#             raise ValueError("PLEBatchTopK: initial_tau must be > 0")

#         # Target masked ratio
#         self.r = float(r)

#         # Controller hyperparameters
#         self.step_size = float(step_size)
#         self.momentum = float(momentum)
#         self.update_test_time = bool(update_test_time)
#         self.update_every = int(update_every)
#         self.sample_prob = float(sample_prob)

#         # Train states
#         self.register_buffer("tau_train", torch.tensor(float(initial_tau)))
#         self.register_buffer("r_ema_train", torch.tensor(0.0))
#         self.register_buffer("steps_train", torch.tensor(0, dtype=torch.long))

#         # Test states
#         self.register_buffer("tau_test", torch.tensor(float(initial_tau)))
#         self.register_buffer("r_ema_test", torch.tensor(0.0))
#         self.register_buffer("steps_test", torch.tensor(0, dtype=torch.long))

#     def train(self, mode: bool = True):
#         prev_mode = self.training
#         super().train(mode)
#         # Detect Train -> Eval transition
#         if self.update_test_time and prev_mode and not self.training:
#             # Initialize test states from current train states for the new eval session
#             self.tau_test.copy_(self.tau_train)
#             self.r_ema_test.fill_(0.0)
#             self.steps_test.fill_(0)

#     @torch.no_grad()
#     def _update_tau(self, avg_r: torch.Tensor, is_train: bool) -> None:
#         """
#         Simple feedback control update with multiplicative adjustment and EMA.
#         Uses Adam-style bias correction to handle zero-initialization of r_ema.
#         """
#         if is_train:
#             tau = self.tau_train
#             r_ema = self.r_ema_train
#             steps = self.steps_train
#         else:
#             tau = self.tau_test
#             r_ema = self.r_ema_test
#             steps = self.steps_test

#         steps.add_(1)
#         if steps.item() % self.update_every != 0:
#             return

#         current_r = float(avg_r.item())

#         # 1. Update EMA
#         r_ema.mul_(self.momentum).add_(current_r * (1.0 - self.momentum))

#         # 2. Apply Bias Correction
#         update_count = steps.item() // self.update_every
#         debias_factor = 1.0 - self.momentum ** update_count

#         if debias_factor > 1e-6:
#             r_corrected = float(r_ema.item()) / debias_factor
#         else:
#             r_corrected = current_r

#         error = r_corrected - self.r

#         # 3. Update tau (Multiplicative)
#         factor = math.exp(-self.step_size * error)
#         new_tau = float(tau.item()) * factor
#         new_tau = max(new_tau, 1e-6)

#         tau.fill_(new_tau)

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         device = x.device
#         dtype = x.dtype
#         B, N, C = x.shape if x.ndim == 3 else (0, 0, 0)
#         total = B * N

#         # Determine tau to use
#         if self.training:
#             tau_used = self.tau_train.to(device=device, dtype=dtype)
#         else:
#             if self.update_test_time:
#                 tau_used = self.tau_test.to(device=device, dtype=dtype)
#             else:
#                 tau_used = self.tau_train.to(device=device, dtype=dtype)

#         # Early exit on empty input
#         if total == 0:
#             mask = torch.zeros(B, N, device=device, dtype=torch.bool)
#             avg_r = torch.zeros((), device=device, dtype=dtype)
#             return mask, avg_r, tau_used

#         # Adjacent dissimilarities and cumulative path per sequence
#         if N > 1:
#             sim = F.cosine_similarity(x[:, 1:, :], x[:, :-1, :], dim=-1)
#             d = torch.zeros(B, N, device=device, dtype=dtype)
#             d[:, 1:] = (1.0 - sim).to(dtype)

#             # Apply Distance Normalization (Sequence-wise Mean Normalization)
#             # This stabilizes tau against distribution shifts.
#             # tau now represents "relative change" rather than "absolute cosine distance".
#             d_mean = d.mean(dim=1, keepdim=True)  # [B, 1]
#             d = d / (d_mean + 1e-6)

#         else:
#             d = torch.zeros(B, N, device=device, dtype=dtype)

#         D = torch.zeros(B, N, device=device, dtype=dtype)
#         if N > 1:
#             D[:, 1:] = torch.cumsum(d[:, 1:], dim=1)
#         L = D[:, -1] if N > 0 else torch.zeros(B, device=device, dtype=dtype)

#         # Build frontier mask via first-crossing with global tau
#         mask = torch.zeros(B, N, device=device, dtype=torch.bool)
#         if N > 0:
#             mask[:, 0] = True  # always keep the first token

#         if N > 1 and torch.isfinite(tau_used):
#             # m_b = floor(L_b / tau)
#             val = (L / tau_used).floor()
#             m_b = torch.clamp(val, min=0, max=N - 1).to(torch.long)

#             max_m = int(m_b.max().item())

#             if max_m > 0:
#                 targets = (torch.arange(1, max_m + 1, device=device, dtype=dtype) * tau_used).view(1, -1)
#                 ge = D.unsqueeze(2) >= targets.view(1, 1, -1)
#                 j = ge.float().argmax(dim=1).clamp_(min=1, max=N - 1)

#                 k_idx = torch.arange(1, max_m + 1, device=device).view(1, -1).expand(B, -1)
#                 valid = k_idx <= m_b.view(B, 1)
#                 if valid.any():
#                     sel = valid.nonzero(as_tuple=False)
#                     b_sel = sel[:, 0]
#                     k_sel = sel[:, 1]
#                     pos = j[b_sel, k_sel]
#                     mask[b_sel, pos] = True

#         # Compute masked ratio for this (local) process
#         kept_total = int(mask.sum().item())
#         zeros_total = int(total - kept_total)

#         # Optionally compute global avg_r across distributed workers (if initialized)
#         if torch.distributed.is_available() and torch.distributed.is_initialized():
#             agg = torch.tensor([zeros_total, total], device=device, dtype=torch.long)
#             torch.distributed.all_reduce(agg, op=torch.distributed.ReduceOp.SUM)
#             zeros_total = int(agg[0].item())
#             total = int(agg[1].item())

#         avg_r = torch.tensor(float(zeros_total) / float(max(1, total)), device=device, dtype=dtype)

#         # Controller update (Always using PLE result)
#         if self.training:
#             self._update_tau(avg_r, is_train=True)
#         elif self.update_test_time:
#             self._update_tau(avg_r, is_train=False)

#         # Overwrite with Random Masking if sample_prob > 0
#         if self.sample_prob > 0.0:
#             if torch.rand(1).item() < self.sample_prob:
#                 # Random masking: mask every token with probability r (keep with 1-r)
#                 probs = torch.full((B, N), 1.0 - self.r, device=device, dtype=dtype)
#                 mask = torch.bernoulli(probs).bool()

#         return mask, avg_r, tau_used

class _BatchSelectorBase(nn.Module):
    """
    Shared Robbins–Monro controller and utilities for batch selectors.
    """

    def __init__(
        self,
        r: float,
        initial_tau: float,
        ema_mu: float,
        eta0: float,
        decay_T: float,
        tau_min: float,
        tau_max: float,
        update_every: int,
        sample_prob: float,
        min_mask_prob: float,
        max_mask_prob: float,
        min_mask_span: int,
        max_mask_span: int,
        random_mask_mode: str = "start_geom",
        max_s: Optional[int] = None,
        fixed_tau: Optional[float] = None,
        update_test_time: bool = False,
        sample_in_inference: bool = False,
        invert_update: bool = False,
        fixed_mask_ratio: bool = False,
    ):
        super().__init__()
        if not (0.0 <= float(r) < 1.0):
            raise ValueError("Batch selector: r must be in [0, 1)")

        val_to_check = fixed_tau if fixed_tau is not None else initial_tau
        if val_to_check <= 0.0:
            raise ValueError("Batch selector: tau must be > 0")

        self.r = float(r)
        self.min_mask_prob = float(min_mask_prob)
        self.max_mask_prob = float(max_mask_prob)
        self.min_mask_span = int(min_mask_span)
        self.max_mask_span = int(max_mask_span)
        mode = str(random_mask_mode).strip().lower()
        if mode == "start_geom_span":
            mode = "start_geom"
        if mode not in {"start_geom", "iid_bernoulli"}:
            raise ValueError("Batch selector: random_mask_mode must be 'start_geom' or 'iid_bernoulli'")
        self.random_mask_mode = mode
        self.ema_mu = float(ema_mu)
        self.eta0 = float(eta0)
        self.decay_T = float(decay_T)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.update_every = int(update_every)
        self.sample_prob = float(sample_prob)
        self.max_s = int(max_s) if max_s is not None else None

        self.fixed_tau = float(fixed_tau) if fixed_tau is not None else None
        self.update_test_time = bool(update_test_time)
        self.sample_in_inference = bool(sample_in_inference)
        self.controller_sign = -1.0 if invert_update else 1.0
        self.fixed_mask_ratio = bool(fixed_mask_ratio)

        if self.min_mask_span <= 0 or self.max_mask_span <= 0:
            raise ValueError("Batch selector: mask spans must be >= 1")
        if self.max_mask_span < self.min_mask_span:
            raise ValueError("Batch selector: max_mask_span must be >= min_mask_span")
        if not (0.0 <= self.min_mask_prob <= 1.0):
            raise ValueError("Batch selector: min_mask_prob must be in [0, 1]")
        if not (0.0 <= self.max_mask_prob <= 1.0):
            raise ValueError("Batch selector: max_mask_prob must be in [0, 1]")
        if self.max_mask_prob < self.min_mask_prob:
            raise ValueError("Batch selector: max_mask_prob must be >= min_mask_prob")

        if self.fixed_mask_ratio:
            if self.random_mask_mode != "iid_bernoulli":
                raise ValueError(
                    "Batch selector: fixed_mask_ratio=True requires random_mask_mode='iid_bernoulli'"
                )
            if not (self.min_mask_span == 1 and self.max_mask_span == 1):
                raise ValueError(
                    "Batch selector: fixed_mask_ratio=True requires min_mask_span=max_mask_span=1"
                )
            if abs(self.max_mask_prob - self.min_mask_prob) > 1e-12:
                raise ValueError(
                    "Batch selector: fixed_mask_ratio=True requires min_mask_prob=max_mask_prob"
                )

        init_val = self.fixed_tau if self.fixed_tau is not None else float(initial_tau)
        self.register_buffer("tau_train", torch.tensor(init_val))
        self.register_buffer("r_ema_train", torch.tensor(0.0))
        self.register_buffer("steps_train", torch.tensor(0, dtype=torch.long))

        self.register_buffer("tau_eval", torch.tensor(init_val))
        self.register_buffer("r_ema_eval", torch.tensor(0.0))
        self.register_buffer("steps_eval", torch.tensor(0, dtype=torch.long))

    def train(self, mode: bool = True):
        prev_mode = self.training
        result = super().train(mode)
        if (
            self.fixed_tau is None
            and self.update_test_time
            and prev_mode
            and not self.training
        ):
            self._reset_eval_state()
        return result

    @torch.no_grad()
    def _reset_eval_state(self) -> None:
        self.tau_eval.copy_(self.tau_train)
        self.r_ema_eval.zero_()
        self.steps_eval.zero_()

    def _active_buffers(self):
        if self.training or not self.update_test_time:
            return self.tau_train, self.r_ema_train, self.steps_train
        return self.tau_eval, self.r_ema_eval, self.steps_eval

    @torch.no_grad()
    def _get_tau_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.fixed_tau is not None:
            return torch.as_tensor(self.fixed_tau, device=device, dtype=dtype)
        tau_buf, _, _ = self._active_buffers()
        return tau_buf.to(device=device, dtype=dtype)

    @torch.no_grad()
    def _controller_step(self, avg_r: torch.Tensor) -> None:
        if self.fixed_tau is not None:
            return
        if (not self.training) and (not self.update_test_time):
            return
        tau_buf, r_ema_buf, steps_buf = self._active_buffers()
        self._update_tau(avg_r, tau_buf, r_ema_buf, steps_buf)

    @torch.no_grad()
    def _update_tau(
        self,
        avg_r: torch.Tensor,
        tau_buf: torch.Tensor,
        r_ema_buf: torch.Tensor,
        steps_buf: torch.Tensor,
    ) -> None:
        freq = max(1, self.update_every)
        steps_buf.add_(1)
        if int(steps_buf.item()) % freq != 0:
            return
        updates = int(steps_buf.item()) // freq

        r_hat = float(avg_r.item())
        r_ema_buf.mul_(self.ema_mu).add_((1.0 - self.ema_mu) * r_hat)

        eta_t = self.eta0 / math.sqrt(1.0 + (updates / max(1.0, self.decay_T)))
        error = float(r_ema_buf.item() - self.r)
        error *= self.controller_sign
        factor = math.exp(-eta_t * error)

        new_tau = float(tau_buf.item()) * factor
        new_tau = min(max(new_tau, self.tau_min), self.tau_max)
        tau_buf.fill_(new_tau)

    def _apply_max_span_constraint(self, mask: torch.Tensor) -> torch.Tensor:
        if self.max_s is None:
            return mask

        B, N = mask.shape
        if N == 0:
            return mask

        stride = max(1, self.max_s)
        idx = torch.arange(N, device=mask.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        minus_one = torch.full_like(idx, -1)
        masked_idx = torch.where(mask, idx, minus_one)
        last_kept_idx, _ = torch.cummax(masked_idx, dim=1)
        run = idx - last_kept_idx
        run = torch.where(mask, torch.zeros_like(run), run)

        stride_tensor = torch.tensor(stride, device=mask.device, dtype=run.dtype)
        need_insert = (~mask) & (run > 0) & ((run % stride_tensor) == 0)

        mask = mask | need_insert
        return mask

    def _compute_avg_r(self, mask: torch.Tensor, total: int, dtype: torch.dtype) -> torch.Tensor:
        kept_total = int(mask.sum().item())
        zeros_total = int(total - kept_total)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            agg = torch.tensor([zeros_total, total], device=mask.device, dtype=torch.long)
            torch.distributed.all_reduce(agg, op=torch.distributed.ReduceOp.SUM)
            zeros_total = int(agg[0].item())
            total = int(agg[1].item())
        return torch.tensor(
            float(zeros_total) / float(max(1, total)),
            device=mask.device,
            dtype=dtype,
        )

    def _maybe_apply_random_mask(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if (not self.training and not self.sample_in_inference) or self.sample_prob <= 0.0:
            return mask
        if torch.rand(1, device=mask.device).item() >= self.sample_prob:
            return mask
        B, N = mask.shape
        if B * N == 0:
            return mask
        device = mask.device

        # Default random branch: start_geom_m4 family.
        # - Independent span starts
        # - Geometric span lengths
        # - Interval-union masking (fully vectorized)
        # m is configured by min_mask_span/max_mask_span (default 4).

        if self.max_mask_prob > self.min_mask_prob:
            mask_prob = torch.empty(B, device=device, dtype=dtype).uniform_(
                self.min_mask_prob, self.max_mask_prob
            )
        else:
            base_prob = self.min_mask_prob
            if base_prob == 0.0 and self.max_mask_prob == 0.0:
                # Backward-friendly fallback when only r is configured.
                base_prob = self.r
            mask_prob = torch.full((B,), float(base_prob), device=device, dtype=dtype)

        if self.max_mask_span > self.min_mask_span:
            mean_span = torch.randint(
                low=self.min_mask_span,
                high=self.max_mask_span + 1,
                size=(B,),
                device=device,
            ).to(dtype=dtype)
        else:
            mean_span = torch.full((B,), float(self.min_mask_span), device=device, dtype=dtype)

        final_mask = torch.ones(B, N, device=device, dtype=torch.bool)
        if N <= 1:
            if self.fixed_mask_ratio:
                num_zero_f = float(N) * float(self.min_mask_prob)
                num_zero = int(round(num_zero_f))
                if abs(num_zero_f - float(num_zero)) > 1e-6 or num_zero != 0:
                    raise ValueError(
                        "Batch selector: fixed_mask_ratio=True is invalid when N<=1 unless mask_prob=0"
                    )
            if N > 0:
                final_mask[:, 0] = True
            return final_mask

        if self.random_mask_mode == "iid_bernoulli":
            if self.fixed_mask_ratio:
                # Exact-ratio IID masking:
                # - token 0 is always kept
                # - each sample has exactly (N * mask_prob) masked tokens
                # - masked positions are sampled uniformly without replacement from 1..N-1
                mask_prob = float(self.min_mask_prob)
                num_zero_f = float(N) * mask_prob
                num_zero = int(round(num_zero_f))
                if abs(num_zero_f - float(num_zero)) > 1e-6:
                    raise ValueError(
                        "Batch selector: fixed_mask_ratio=True requires N*mask_prob to be an integer"
                    )
                tail_len = N - 1
                if num_zero < 0 or num_zero > tail_len:
                    raise ValueError(
                        "Batch selector: fixed_mask_ratio=True requires 0 <= N*mask_prob <= N-1 (token0 is always kept)"
                    )

                final_mask = torch.ones(B, N, device=device, dtype=torch.bool)
                if num_zero > 0 and tail_len > 0:
                    rand = torch.rand(B, tail_len, device=device, dtype=torch.float32)
                    zero_idx = torch.topk(rand, k=num_zero, dim=1, largest=True, sorted=False).indices
                    tail_keep = torch.ones(B, tail_len, device=device, dtype=torch.bool)
                    tail_keep.scatter_(1, zero_idx, False)
                    final_mask[:, 1:] = tail_keep
                final_mask[:, 0] = True
                return final_mask

            # Legacy iid_bernoulli branch with span support:
            # - span == 1: pure iid Bernoulli on tokens 1..N-1
            # - span  > 1: wav2vec2-style fixed-span starts (overlap allowed)
            # Token 0 is always kept.
            if self.max_mask_span > self.min_mask_span:
                span_b = torch.randint(
                    low=self.min_mask_span,
                    high=self.max_mask_span + 1,
                    size=(B,),
                    device=device,
                ).to(torch.long)
            else:
                span_b = torch.full((B,), int(self.min_mask_span), device=device, dtype=torch.long)
            span_b = span_b.clamp_min(1)

            T = N - 1
            r_b = mask_prob.clamp(min=0.0, max=1.0)
            # Since token0 is fixed keep, match overall target-r by scaling tail ratio.
            r_tail = (r_b * (float(N) / float(max(1, N - 1)))).clamp(min=0.0, max=1.0)

            span_f = span_b.to(dtype=dtype).clamp(min=1.0)
            r_tail_geom = r_tail.clamp(min=1e-6, max=1.0 - 1e-6)
            p_start_geom = ((-torch.log1p(-r_tail_geom)) / span_f).clamp(min=1e-6, max=1.0)
            p_start_iid = r_tail
            p_start = torch.where(span_b <= 1, p_start_iid, p_start_geom)

            starts = torch.rand(B, T, device=device, dtype=dtype) < p_start.unsqueeze(1)
            if not bool(starts.any().item()):
                final_mask[:, 0] = True
                return final_mask

            pos = torch.arange(1, N, device=device, dtype=torch.long).view(1, T).expand(B, -1)
            b_idx = torch.arange(B, device=device, dtype=torch.long).view(B, 1).expand(B, T)
            len_all = span_b.view(B, 1).expand(B, T)

            start_pos = pos[starts]
            start_len = len_all[starts]
            start_b = b_idx[starts]
            end_pos = (start_pos + start_len - 1).clamp(max=N - 1)

            diff = torch.zeros(B, N + 1, device=device, dtype=dtype)
            diff_flat = diff.view(-1)
            row_stride = N + 1
            flat_start = start_b * row_stride + start_pos
            flat_endp1 = start_b * row_stride + (end_pos + 1)
            ones = torch.ones_like(flat_start, dtype=dtype)
            diff_flat.index_add_(0, flat_start, ones)
            diff_flat.index_add_(0, flat_endp1, -ones)

            covered = diff[:, :N].cumsum(dim=1) > 0
            final_mask = ~covered
            final_mask[:, 0] = True
            return final_mask

        # Fully vectorized geom-span approximation (no Python loop over B or N):
        # 1) Sample span starts at each position with p_start.
        # 2) Sample geometric span length for each position.
        # 3) Union intervals via a difference-array + cumsum trick.
        # This is intentionally approximate for training efficiency.
        r_b = mask_prob.clamp(min=1e-6, max=1.0 - 1e-6)
        m_b = mean_span.clamp(min=1.0)

        p_end = (1.0 / m_b).clamp(min=1e-6, max=1.0)
        # For independent start-process + interval-union construction,
        # use Poisson-style coverage approximation:
        #   r ~= 1 - exp(-p_start * m)  =>  p_start ~= -log(1-r)/m
        # This tracks target r better than Markov-chain stationary mapping in
        # the fully vectorized union model.
        p_start = ((-torch.log1p(-r_b)) / m_b).clamp(min=1e-6, max=1.0)

        T = N - 1
        starts = torch.rand(B, T, device=device, dtype=dtype) < p_start.unsqueeze(1)
        if not bool(starts.any().item()):
            final_mask[:, 0] = True
            return final_mask

        # Geometric length on {1,2,...}: L = floor(log(1-U) / log(1-p_end)) + 1
        u_len = torch.rand(B, T, device=device, dtype=dtype).clamp_(1e-12, 1.0 - 1e-12)
        q = (1.0 - p_end).clamp(min=1e-12, max=1.0 - 1e-6)
        log_q = torch.log(q).unsqueeze(1)
        len_all = torch.floor(torch.log1p(-u_len) / log_q).to(torch.long) + 1

        pos = torch.arange(1, N, device=device, dtype=torch.long).view(1, T).expand(B, -1)
        b_idx = torch.arange(B, device=device, dtype=torch.long).view(B, 1).expand(B, T)

        start_pos = pos[starts]
        start_len = len_all[starts]
        start_b = b_idx[starts]
        end_pos = (start_pos + start_len - 1).clamp(max=N - 1)

        diff = torch.zeros(B, N + 1, device=device, dtype=dtype)
        diff_flat = diff.view(-1)

        row_stride = N + 1
        flat_start = start_b * row_stride + start_pos
        flat_endp1 = start_b * row_stride + (end_pos + 1)
        ones = torch.ones_like(flat_start, dtype=dtype)

        diff_flat.index_add_(0, flat_start, ones)
        diff_flat.index_add_(0, flat_endp1, -ones)

        covered = diff[:, :N].cumsum(dim=1) > 0
        final_mask = ~covered
        final_mask[:, 0] = True
        return final_mask


class FixedPattern(_BatchSelectorBase):
    """
    Fixed-pattern selector.

    - Keeps token 0 and every `round(1 / (1 - r))`-th token.
    - Does not use tau/controller updates.
    - During training, samples per-sequence `r_b ~ Uniform(min_mask_prob, max_mask_prob)`.
      If both bounds are 0, falls back to the configured `r`.

    Tau-related arguments are accepted for config compatibility but ignored.
    """

    def __init__(
        self,
        r: float,
        initial_tau: float = 1.0,
        ema_mu: float = 0.95,
        eta0: float = 0.1,
        decay_T: float = 1000.0,
        tau_min: float = 1e-6,
        tau_max: float = 1e6,
        update_every: int = 1,
        sample_prob: float = 0.0,
        min_mask_prob: float = 0.0,
        max_mask_prob: float = 0.0,
        min_mask_span: int = 4,
        max_mask_span: int = 4,
        random_mask_mode: str = "start_geom",
        max_s: Optional[int] = None,
        fixed_tau: Optional[float] = None,
        update_test_time: bool = False,
        sample_in_inference: bool = False,
        fixed_mask_ratio: bool = False,
    ):
        # Keep the same constructor surface as other selectors,
        # but intentionally ignore tau/controller-related options.
        super().__init__(
            r=r,
            initial_tau=1.0,
            ema_mu=0.95,
            eta0=0.1,
            decay_T=1000.0,
            tau_min=1e-6,
            tau_max=1e6,
            update_every=1,
            sample_prob=sample_prob,
            min_mask_prob=min_mask_prob,
            max_mask_prob=max_mask_prob,
            min_mask_span=min_mask_span,
            max_mask_span=max_mask_span,
            random_mask_mode=random_mask_mode,
            max_s=max_s,
            fixed_tau=1.0,
            update_test_time=False,
            sample_in_inference=sample_in_inference,
            invert_update=False,
            fixed_mask_ratio=fixed_mask_ratio,
        )

    @staticmethod
    def _r_to_stride(r: torch.Tensor) -> torch.Tensor:
        keep_ratio = (1.0 - r).clamp(min=1e-6)
        stride = torch.round(1.0 / keep_ratio).to(torch.long)
        return stride.clamp_min_(1)

    def _sample_pattern_r(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.training:
            return torch.full((batch_size,), float(self.r), device=device, dtype=dtype)

        lo = float(self.min_mask_prob)
        hi = float(self.max_mask_prob)

        if lo == 0.0 and hi == 0.0:
            lo = float(self.r)
            hi = float(self.r)

        if hi > lo:
            r_b = torch.empty(batch_size, device=device, dtype=dtype).uniform_(lo, hi)
        else:
            r_b = torch.full((batch_size,), lo, device=device, dtype=dtype)

        return r_b

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        B, N, _C = x.shape if x.ndim == 3 else (0, 0, 0)
        total = B * N

        tau_used = self._get_tau_tensor(device, dtype)

        if total == 0:
            mask = torch.zeros(B, N, device=device, dtype=torch.bool)
            avg_r = torch.zeros((), device=device, dtype=dtype)
            return mask, avg_r, tau_used

        r_b = self._sample_pattern_r(B, device, dtype)
        r_b = r_b.clamp(min=0.0, max=1.0 - 1e-6)
        stride_b = self._r_to_stride(r_b)

        pos = torch.arange(N, device=device, dtype=torch.long).view(1, N)
        mask = (pos % stride_b.view(B, 1)) == 0

        mask = self._apply_max_span_constraint(mask)
        mask = self._maybe_apply_random_mask(mask, dtype)
        if N > 0:
            mask[:, 0] = True
        mask = self._apply_max_span_constraint(mask)

        avg_r = self._compute_avg_r(mask, total, dtype)
        return mask, avg_r, tau_used


class PLEBatchTopK(_BatchSelectorBase):
    """
    Batch-level PLE (Path-Length Equalization) with a single global tau.

    Design:
      - Use one global scalar tau for both training and inference.
      - Store tau directly instead of its logarithm.
      - If fixed_tau is provided, use it always and ignore updates.
      - Otherwise, update tau using a Robbins–Monro style controller.
      - If update_test_time is True, evaluation starts from the last train tau
        but keeps its own controller state.
    """
    def __init__(
        self,
        r: float,
        initial_tau: float = 1.0,
        ema_mu: float = 0.95,
        eta0: float = 0.1,
        decay_T: float = 1000.0,
        tau_min: float = 1e-6,
        tau_max: float = 1e6,
        update_every: int = 1,
        sample_prob: float = 0.0,
        min_mask_prob: float = 0.0,
        max_mask_prob: float = 0.0,
        min_mask_span: int = 4,
        max_mask_span: int = 4,
        random_mask_mode: str = "start_geom",
        max_s: Optional[int] = None,
        fixed_tau: Optional[float] = None,
        update_test_time: bool = False,
        sample_in_inference: bool = False,
        fixed_mask_ratio: bool = False,
    ):
        super().__init__(
            r=r,
            initial_tau=initial_tau,
            ema_mu=ema_mu,
            eta0=eta0,
            decay_T=decay_T,
            tau_min=tau_min,
            tau_max=tau_max,
            update_every=update_every,
            sample_prob=sample_prob,
            min_mask_prob=min_mask_prob,
            max_mask_prob=max_mask_prob,
            min_mask_span=min_mask_span,
            max_mask_span=max_mask_span,
            random_mask_mode=random_mask_mode,
            max_s=max_s,
            fixed_tau=fixed_tau,
            update_test_time=update_test_time,
            sample_in_inference=sample_in_inference,
            invert_update=False,
            fixed_mask_ratio=fixed_mask_ratio,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        B, N, C = x.shape if x.ndim == 3 else (0, 0, 0)
        total = B * N

        tau_used = self._get_tau_tensor(device, dtype)

        # Early exit on empty input
        if total == 0:
            mask = torch.zeros(B, N, device=device, dtype=torch.bool)
            avg_r = torch.zeros((), device=device, dtype=dtype)
            return mask, avg_r, tau_used

        # Adjacent dissimilarities
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

        # Build frontier mask
        mask = torch.zeros(B, N, device=device, dtype=torch.bool)
        if N > 0:
            mask[:, 0] = True

        if N > 1 and torch.isfinite(tau_used) and (tau_used.item() > 0.0):
            m_raw = torch.floor(L / tau_used).to(torch.long)
            m_b = m_raw
            m_b = torch.clamp(m_b, min=0, max=N - 1)
            saturated = m_raw > (N - 1)
            max_m = int(m_b.max().item())

            if max_m > 0:
                k = torch.arange(1, max_m + 1, device=device, dtype=dtype).view(1, -1)

                # Degenerate small-tau guard:
                # If m_raw is clipped by N-1, k*tau no longer reaches L and many
                # thresholds collapse to early positions. For clipped rows, switch to
                # an evenly-spaced effective tau so the last target reaches L.
                tau_eff = tau_used.expand(B)
                if bool(saturated.any().item()):
                    denom = m_b.to(dtype).clamp_min(1.0)
                    tau_sat = (L / denom).clamp(min=self.tau_min, max=self.tau_max)
                    tau_eff = torch.where(saturated, tau_sat, tau_eff)

                targets = k * tau_eff.view(B, 1)
                ge = D.unsqueeze(2) >= targets.unsqueeze(1)
                j = ge.float().argmax(dim=1).clamp_(min=1, max=N - 1)

                k_idx = torch.arange(1, max_m + 1, device=device).view(1, -1).expand(B, -1)
                valid = k_idx <= m_b.view(B, 1)
                if valid.any():
                    sel = valid.nonzero(as_tuple=False)
                    b_sel = sel[:, 0]
                    k_sel = sel[:, 1]
                    pos = j[b_sel, k_sel]
                    mask[b_sel, pos] = True

            # Clamp consistency policy:
            # if m_raw exceeds the available boundary slots (N-1), keep all boundaries.
            # This avoids pathological duplicate-boundary collapse in the tiny-tau regime.
            if bool(saturated.any().item()):
                mask[saturated, 1:] = True

        mask = self._apply_max_span_constraint(mask)

        avg_r = self._compute_avg_r(mask, total, dtype)
        self._controller_step(avg_r)
        mask = self._maybe_apply_random_mask(mask, dtype)

        return mask, avg_r, tau_used


class PLEBatchTopKJitter(_BatchSelectorBase):
    """
    PLEBatchTopK with train-time per-sample tau jittering for robustness.

    Motivation:
      - Keep a single global tau controlled by Robbins–Monro (same as PLEBatchTopK),
        but during training, apply a small per-sample multiplicative jitter:

            tau_b = clamp(tau * exp(u_b), tau_min, tau_max),  u_b ~ Uniform(-a, a)

      - This makes masking patterns (and thus the reconstruction/SSL signal) robust to tau.

    Notes:
      - Jitter is applied only when self.training is True.
      - Controller update uses the jittered outcome (avg_r), so tau converges in expectation.
      - Return signature matches PLEBatchTopK: (mask, avg_r, tau_used) where tau_used is the
        base (non-jittered) tau tensor on the current device/dtype.
    """

    def __init__(
        self,
        r: float,
        initial_tau: float = 1.0,
        ema_mu: float = 0.95,
        eta0: float = 0.1,
        decay_T: float = 1000.0,
        tau_min: float = 1e-6,
        tau_max: float = 1e6,
        update_every: int = 1,
        sample_prob: float = 0.0,
        min_mask_prob: float = 0.0,
        max_mask_prob: float = 0.0,
        min_mask_span: int = 4,
        max_mask_span: int = 4,
        random_mask_mode: str = "start_geom",
        max_s: Optional[int] = None,
        fixed_tau: Optional[float] = None,
        update_test_time: bool = False,
        sample_in_inference: bool = False,
        jitter_a: float = 0.4,
        fixed_mask_ratio: bool = False,
    ):
        super().__init__(
            r=r,
            initial_tau=initial_tau,
            ema_mu=ema_mu,
            eta0=eta0,
            decay_T=decay_T,
            tau_min=tau_min,
            tau_max=tau_max,
            update_every=update_every,
            sample_prob=sample_prob,
            min_mask_prob=min_mask_prob,
            max_mask_prob=max_mask_prob,
            min_mask_span=min_mask_span,
            max_mask_span=max_mask_span,
            random_mask_mode=random_mask_mode,
            max_s=max_s,
            fixed_tau=fixed_tau,
            update_test_time=update_test_time,
            sample_in_inference=sample_in_inference,
            invert_update=False,
            fixed_mask_ratio=fixed_mask_ratio,
        )
        if jitter_a < 0.0:
            raise ValueError("PLEBatchTopKJitter: jitter_a must be >= 0")
        self.jitter_a = float(jitter_a)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        B, N, C = x.shape if x.ndim == 3 else (0, 0, 0)
        total = B * N

        # Base tau (scalar) on the current device/dtype
        tau_used = self._get_tau_tensor(device, dtype)

        # Early exit on empty input
        if total == 0:
            mask = torch.zeros(B, N, device=device, dtype=torch.bool)
            avg_r = torch.zeros((), device=device, dtype=dtype)
            return mask, avg_r, tau_used

        # Build per-sample tau (train-time jitter; eval-time uses base tau for all)
        if self.training and (self.jitter_a > 0.0) and (self.fixed_tau is None):
            u = torch.empty(B, device=device, dtype=dtype).uniform_(-self.jitter_a, self.jitter_a)
            tau_b = tau_used * torch.exp(u)  # [B]
            tau_b = tau_b.clamp(min=self.tau_min, max=self.tau_max)
        else:
            tau_b = tau_used.expand(B)  # [B]

        # Adjacent dissimilarities
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

        # Build frontier mask (per-sample tau)
        mask = torch.zeros(B, N, device=device, dtype=torch.bool)
        if N > 0:
            mask[:, 0] = True

        if N > 1:
            finite = torch.isfinite(tau_b) & (tau_b > 0.0)
            if bool(finite.any().item()):
                # m_b = floor(L_b / tau_b), clamped
                m_raw = torch.floor(L / tau_b).to(torch.long)
                m_b = m_raw
                m_b = torch.clamp(m_b, min=0, max=N - 1)
                saturated = m_raw > (N - 1)
                max_m = int(m_b.max().item())

                if max_m > 0:
                    k = torch.arange(1, max_m + 1, device=device, dtype=dtype).view(1, -1)  # [1, M]

                    # Degenerate small-tau guard for per-sample tau_b.
                    tau_eff = tau_b
                    if bool(saturated.any().item()):
                        denom = m_b.to(dtype).clamp_min(1.0)
                        tau_sat = (L / denom).clamp(min=self.tau_min, max=self.tau_max)
                        tau_eff = torch.where(saturated, tau_sat, tau_eff)

                    targets = k * tau_eff.view(B, 1)  # [B, M]
                    ge = D.unsqueeze(2) >= targets.unsqueeze(1)  # [B, N, M]
                    j = ge.float().argmax(dim=1).clamp_(min=1, max=N - 1)  # [B, M]

                    k_idx = torch.arange(1, max_m + 1, device=device).view(1, -1).expand(B, -1)  # [B, M]
                    valid = k_idx <= m_b.view(B, 1)
                    valid = valid & finite.view(B, 1)
                    if valid.any():
                        sel = valid.nonzero(as_tuple=False)
                        b_sel = sel[:, 0]
                        k_sel = sel[:, 1]
                        pos = j[b_sel, k_sel]
                        mask[b_sel, pos] = True

                # Clamp consistency policy for tiny tau:
                # if requested boundaries exceed N-1, keep all boundaries.
                if bool(saturated.any().item()):
                    mask[saturated, 1:] = True

        mask = self._apply_max_span_constraint(mask)

        avg_r = self._compute_avg_r(mask, total, dtype)
        self._controller_step(avg_r)
        mask = self._maybe_apply_random_mask(mask, dtype)

        return mask, avg_r, tau_used


class BatchTopK(_BatchSelectorBase):
    """
    One-shot masking: drop every trailing token whose adjacent cosine similarity exceeds tau.
    """

    def __init__(
        self,
        r: float,
        initial_tau: float = 0.6,
        ema_mu: float = 0.95,
        eta0: float = 0.1,
        decay_T: float = 1000.0,
        tau_min: float = 1e-3,
        tau_max: float = 0.999,
        update_every: int = 1,
        sample_prob: float = 0.0,
        min_mask_prob: float = 0.0,
        max_mask_prob: float = 0.0,
        min_mask_span: int = 4,
        max_mask_span: int = 4,
        random_mask_mode: str = "start_geom",
        max_s: Optional[int] = None,
        fixed_tau: Optional[float] = None,
        update_test_time: bool = False,
        sample_in_inference: bool = False,
        fixed_mask_ratio: bool = False,
    ):
        super().__init__(
            r=r,
            initial_tau=initial_tau,
            ema_mu=ema_mu,
            eta0=eta0,
            decay_T=decay_T,
            tau_min=tau_min,
            tau_max=tau_max,
            update_every=update_every,
            sample_prob=sample_prob,
            min_mask_prob=min_mask_prob,
            max_mask_prob=max_mask_prob,
            min_mask_span=min_mask_span,
            max_mask_span=max_mask_span,
            random_mask_mode=random_mask_mode,
            max_s=max_s,
            fixed_tau=fixed_tau,
            update_test_time=update_test_time,
            sample_in_inference=sample_in_inference,
            invert_update=True,
            fixed_mask_ratio=fixed_mask_ratio,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        B, N, C = x.shape if x.ndim == 3 else (0, 0, 0)
        total = B * N
        tau_used = self._get_tau_tensor(device, dtype)

        if total == 0:
            mask = torch.zeros(B, N, device=device, dtype=torch.bool)
            avg_r = torch.zeros((), device=device, dtype=dtype)
            return mask, avg_r, tau_used

        mask = torch.ones(B, N, device=device, dtype=torch.bool)
        if N > 0:
            mask[:, 0] = True

        if N > 1 and bool(torch.isfinite(tau_used).item()):
            sim = F.cosine_similarity(x[:, 1:, :], x[:, :-1, :], dim=-1)
            trailing = sim > tau_used
            mask[:, 1:] = ~trailing

        mask = self._apply_max_span_constraint(mask)

        avg_r = self._compute_avg_r(mask, total, dtype)
        self._controller_step(avg_r)
        mask = self._maybe_apply_random_mask(mask, dtype)
        return mask, avg_r, tau_used


class BatchGreedy(_BatchSelectorBase):
    """
    Greedy masking: iteratively drop the trailing token with the highest similarity above tau.
    """

    def __init__(
        self,
        r: float,
        initial_tau: float = 0.85,
        ema_mu: float = 0.95,
        eta0: float = 0.1,
        decay_T: float = 1000.0,
        tau_min: float = 1e-3,
        tau_max: float = 0.999,
        update_every: int = 1,
        sample_prob: float = 0.0,
        min_mask_prob: float = 0.0,
        max_mask_prob: float = 0.0,
        min_mask_span: int = 4,
        max_mask_span: int = 4,
        random_mask_mode: str = "start_geom",
        max_s: Optional[int] = None,
        fixed_tau: Optional[float] = None,
        update_test_time: bool = False,
        sample_in_inference: bool = False,
        fixed_mask_ratio: bool = False,
    ):
        super().__init__(
            r=r,
            initial_tau=initial_tau,
            ema_mu=ema_mu,
            eta0=eta0,
            decay_T=decay_T,
            tau_min=tau_min,
            tau_max=tau_max,
            update_every=update_every,
            sample_prob=sample_prob,
            min_mask_prob=min_mask_prob,
            max_mask_prob=max_mask_prob,
            min_mask_span=min_mask_span,
            max_mask_span=max_mask_span,
            random_mask_mode=random_mask_mode,
            max_s=max_s,
            fixed_tau=fixed_tau,
            update_test_time=update_test_time,
            sample_in_inference=sample_in_inference,
            invert_update=True,
            fixed_mask_ratio=fixed_mask_ratio,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        B, N, C = x.shape if x.ndim == 3 else (0, 0, 0)
        total = B * N
        tau_used = self._get_tau_tensor(device, dtype)

        if total == 0:
            mask = torch.zeros(B, N, device=device, dtype=torch.bool)
            avg_r = torch.zeros((), device=device, dtype=dtype)
            return mask, avg_r, tau_used

        mask = torch.ones(B, N, device=device, dtype=torch.bool)
        if N > 0:
            mask[:, 0] = True

        if N > 1 and bool(torch.isfinite(tau_used).item()):
            tau_threshold = tau_used
            max_iters = max(0, N - 1)
            for b in range(B):
                seq_mask = mask[b]
                xb = x[b]
                for _ in range(max_iters):
                    kept_idx = torch.where(seq_mask)[0]
                    if kept_idx.numel() <= 1:
                        break
                    seq = xb.index_select(0, kept_idx)
                    sims = F.cosine_similarity(seq[1:], seq[:-1], dim=-1)
                    if sims.numel() == 0:
                        break
                    max_sim, rel_idx = torch.max(sims, dim=0)
                    if (not torch.isfinite(max_sim)) or not bool((max_sim > tau_threshold).item()):
                        break
                    remove_idx = kept_idx[rel_idx + 1]
                    seq_mask[remove_idx] = False

        mask = self._apply_max_span_constraint(mask)

        avg_r = self._compute_avg_r(mask, total, dtype)
        self._controller_step(avg_r)
        mask = self._maybe_apply_random_mask(mask, dtype)
        return mask, avg_r, tau_used


class PLEBatchTopKTrainPerSeq(PLEBatchTopK):
    """Hybrid selector with train/eval split.

    - Training: per-sequence PLE masking with per-sample `r_b ~ Uniform(train_r_min, train_r_max)`.
      This mirrors the per-sequence PLE idea in `AudioTokenization/BigCodec_SSL/dtp/tome_ops.py::PLETopK`,
      but returns DTMAE-style outputs (frontier mask).

    - Eval/Inference: uses the original batch-level global-tau controller from `PLEBatchTopK`.

    Train-time tau initialization policy:
      - Maintain an EMA of normalized path length, `E[L/N]` (valid finite L only).
      - Estimate train tau from statistics (no extra correction term):

          tau_train_hat = clamp( EMA(L/N) / (1 - r_target), tau_min, tau_max )

      - This tau_train is used as the eval start point; if update_test_time=True,
        eval-time Robbins-Monro tuning remains available as in `PLEBatchTopK`.

    Return: (mask, avg_r, tau_used)
      - mask: [B, N] bool, always keeps token 0.
      - avg_r: scalar masked ratio computed from the final mask.
      - tau_used:
          * train(): mean of per-sequence tau values used to generate the PLE mask
          * eval():  scalar tau returned by `PLEBatchTopK`
    """

    def __init__(
        self,
        r: float,
        initial_tau: float = 1.0,
        ema_mu: float = 0.95,
        eta0: float = 0.1,
        decay_T: float = 1000.0,
        tau_min: float = 1e-6,
        tau_max: float = 1e6,
        update_every: int = 1,
        sample_prob: float = 0.0,
        min_mask_prob: float = 0.0,
        max_mask_prob: float = 0.0,
        min_mask_span: int = 4,
        max_mask_span: int = 4,
        random_mask_mode: str = "start_geom",
        max_s: Optional[int] = None,
        fixed_tau: Optional[float] = None,
        update_test_time: bool = False,
        sample_in_inference: bool = False,
        fixed_mask_ratio: bool = False,
        *,
        train_r_min: Optional[float] = None,
        train_r_max: Optional[float] = None,
        train_r_sampling: str = "uniform",
        train_r_beta_alpha: float = 2.0,
        train_r_beta_beta: float = 2.0,
    ):
        super().__init__(
            r=r,
            initial_tau=initial_tau,
            ema_mu=ema_mu,
            eta0=eta0,
            decay_T=decay_T,
            tau_min=tau_min,
            tau_max=tau_max,
            update_every=update_every,
            sample_prob=sample_prob,
            min_mask_prob=min_mask_prob,
            max_mask_prob=max_mask_prob,
            min_mask_span=min_mask_span,
            max_mask_span=max_mask_span,
            random_mask_mode=random_mask_mode,
            max_s=max_s,
            fixed_tau=fixed_tau,
            update_test_time=update_test_time,
            sample_in_inference=sample_in_inference,
            fixed_mask_ratio=fixed_mask_ratio,
        )

        if train_r_min is None:
            train_r_min = float(r)
        if train_r_max is None:
            train_r_max = float(r)
        train_r_min = float(train_r_min)
        train_r_max = float(train_r_max)
        if not (0.0 <= train_r_min <= 1.0):
            raise ValueError("PLEBatchTopKTrainPerSeq: train_r_min must be in [0, 1]")
        if not (0.0 <= train_r_max <= 1.0):
            raise ValueError("PLEBatchTopKTrainPerSeq: train_r_max must be in [0, 1]")
        if train_r_max < train_r_min:
            raise ValueError("PLEBatchTopKTrainPerSeq: train_r_max must be >= train_r_min")
        train_r_sampling = str(train_r_sampling).lower()
        if train_r_sampling not in {"uniform", "beta"}:
            raise ValueError("PLEBatchTopKTrainPerSeq: train_r_sampling must be 'uniform' or 'beta'")
        if float(train_r_beta_alpha) <= 0.0 or float(train_r_beta_beta) <= 0.0:
            raise ValueError("PLEBatchTopKTrainPerSeq: beta parameters must be > 0")
        self.train_r_min = train_r_min
        self.train_r_max = train_r_max
        self.train_r_sampling = train_r_sampling
        self.train_r_beta_alpha = float(train_r_beta_alpha)
        self.train_r_beta_beta = float(train_r_beta_beta)

        # Train-time statistics for tau estimation.
        self.register_buffer("l_over_n_ema", torch.tensor(0.0))
        self.register_buffer("l_over_n_updates", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training:
            return super().forward(x)

        device = x.device
        dtype = x.dtype
        B, N, _C = x.shape if x.ndim == 3 else (0, 0, 0)
        total = B * N

        if total == 0:
            mask = torch.zeros(B, N, device=device, dtype=torch.bool)
            avg_r = torch.zeros((), device=device, dtype=dtype)
            tau_used = self._get_tau_tensor(device, dtype)
            return mask, avg_r, tau_used

        # Adjacent dissimilarities
        if N > 1:
            sim = F.cosine_similarity(x[:, 1:, :], x[:, :-1, :], dim=-1)
            d = torch.zeros(B, N, device=device, dtype=dtype)
            d[:, 1:] = (1.0 - sim).to(dtype)
        else:
            d = torch.zeros(B, N, device=device, dtype=dtype)

        # Cumulative path length
        D = torch.zeros(B, N, device=device, dtype=dtype)
        if N > 1:
            D[:, 1:] = torch.cumsum(d[:, 1:], dim=1)
        L = D[:, -1] if N > 0 else torch.zeros(B, device=device, dtype=dtype)

        # Sample per-sequence target r
        if self.train_r_max > self.train_r_min:
            if self.train_r_sampling == "uniform":
                r_b = torch.empty(B, device=device, dtype=dtype).uniform_(self.train_r_min, self.train_r_max)
            else:  # beta on [train_r_min, train_r_max]
                beta_dist = torch.distributions.Beta(
                    torch.tensor(self.train_r_beta_alpha, device=device, dtype=torch.float32),
                    torch.tensor(self.train_r_beta_beta, device=device, dtype=torch.float32),
                )
                beta_u = beta_dist.sample((B,)).to(device=device, dtype=dtype)
                width = float(self.train_r_max - self.train_r_min)
                r_b = float(self.train_r_min) + width * beta_u
        else:
            r_b = torch.full((B,), self.train_r_min, device=device, dtype=dtype)
        r_upper = 1.0 if N <= 1 else (1.0 - (1.0 / float(N)))
        r_b = r_b.clamp(min=0.0, max=r_upper)

        # Convert r_b -> desired kept count (not enforced strictly; duplicates may reduce actual kept)
        keep_ratio = (1.0 - r_b).clamp(min=0.0, max=1.0)
        desired_keep = torch.round(keep_ratio * float(N)).to(torch.long)
        desired_keep = desired_keep.clamp(min=1, max=N)
        m_target = (desired_keep - 1).clamp(min=0, max=max(0, N - 1))  # number of boundaries

        # Per-sequence tau from target boundary count
        tau_b = torch.full((B,), float(self.tau_max), device=device, dtype=dtype)
        valid_tau = (m_target > 0) & torch.isfinite(L) & (L > 0)
        if bool(valid_tau.any().item()):
            denom = m_target.to(dtype).clamp_min(1.0)
            tau_vals = (L / denom).clamp(min=self.tau_min, max=self.tau_max)
            tau_b = torch.where(valid_tau, tau_vals, tau_b)

        # Build frontier mask
        mask = torch.zeros(B, N, device=device, dtype=torch.bool)
        mask[:, 0] = True

        if N > 1:
            max_m = int(m_target.max().item())
            if max_m > 0 and bool(valid_tau.any().item()):
                k = torch.arange(1, max_m + 1, device=device, dtype=dtype).view(1, -1)  # [1, M]
                targets = k * tau_b.view(B, 1)  # [B, M]
                ge = D.unsqueeze(2) >= targets.unsqueeze(1)  # [B, N, M]
                j = ge.float().argmax(dim=1).clamp_(min=1, max=N - 1)  # [B, M]

                k_idx = torch.arange(1, max_m + 1, device=device).view(1, -1).expand(B, -1)
                valid_k = k_idx <= m_target.view(B, 1)
                valid_k = valid_k & valid_tau.view(B, 1)
                if valid_k.any():
                    sel = valid_k.nonzero(as_tuple=False)
                    b_sel = sel[:, 0]
                    k_sel = sel[:, 1]
                    pos = j[b_sel, k_sel]
                    mask[b_sel, pos] = True

        # Fallback for degenerate sequences (e.g., L==0): keep the first desired_keep tokens.
        fallback = (m_target > 0) & (~valid_tau)
        if bool(fallback.any().item()):
            b_sel = fallback.nonzero(as_tuple=False).squeeze(1)
            pos = torch.arange(N, device=device, dtype=torch.long).view(1, N)
            fb_mask = pos < desired_keep.index_select(0, b_sel).view(-1, 1)
            mask.index_copy_(0, b_sel, fb_mask)
            mask[b_sel, 0] = True

        # Enforce max-span constraint before optional random masking
        mask = self._apply_max_span_constraint(mask)

        # Optional: random masking override/mix (per-sequence prob ~ Uniform(min_mask_prob, max_mask_prob))
        mask = self._maybe_apply_random_mask(mask, dtype)
        if N > 0:
            mask[:, 0] = True
        mask = self._apply_max_span_constraint(mask)

        avg_r = self._compute_avg_r(mask, total, dtype)
        tau_used = tau_b.mean()

        # Estimate tau_train from train-time L statistics (no extra correction).
        # Keep update_test_time behavior from base PLE for eval-time optional tuning.
        if self.fixed_tau is None and N > 0:
            valid_l = torch.isfinite(L) & (L > 0)
            if bool(valid_l.any().item()):
                l_over_n_batch = (L[valid_l] / float(max(1, N))).mean()
                if int(self.l_over_n_updates.item()) == 0:
                    self.l_over_n_ema.fill_(float(l_over_n_batch.item()))
                else:
                    self.l_over_n_ema.mul_(self.ema_mu).add_((1.0 - self.ema_mu) * float(l_over_n_batch.item()))
                self.l_over_n_updates.add_(1)

            if int(self.l_over_n_updates.item()) > 0:
                denom = max(1e-8, 1.0 - float(self.r))
                tau_hat = float(self.l_over_n_ema.item()) / denom
                tau_hat = min(max(tau_hat, self.tau_min), self.tau_max)
                self.tau_train.fill_(tau_hat)
        return mask, avg_r, tau_used


class BatchTopKTrainPerSeq(BatchTopK):
    """Hybrid selector with train/eval split.

    - Training: per-sequence Top-K masking with per-sample `r_b ~ Uniform(train_r_min, train_r_max)`.
      Implementation removes the right token of the top-K adjacent similarities per sequence.
    - Eval/Inference: uses the original batch-level global-tau controller from `BatchTopK`.
    """

    def __init__(
        self,
        r: float,
        initial_tau: float = 0.6,
        ema_mu: float = 0.95,
        eta0: float = 0.1,
        decay_T: float = 1000.0,
        tau_min: float = 1e-3,
        tau_max: float = 0.999,
        update_every: int = 1,
        sample_prob: float = 0.0,
        min_mask_prob: float = 0.0,
        max_mask_prob: float = 0.0,
        min_mask_span: int = 4,
        max_mask_span: int = 4,
        random_mask_mode: str = "start_geom",
        max_s: Optional[int] = None,
        fixed_tau: Optional[float] = None,
        update_test_time: bool = False,
        sample_in_inference: bool = False,
        fixed_mask_ratio: bool = False,
        *,
        train_r_min: Optional[float] = None,
        train_r_max: Optional[float] = None,
        train_r_sampling: str = "uniform",
        train_r_beta_alpha: float = 2.0,
        train_r_beta_beta: float = 2.0,
    ):
        super().__init__(
            r=r,
            initial_tau=initial_tau,
            ema_mu=ema_mu,
            eta0=eta0,
            decay_T=decay_T,
            tau_min=tau_min,
            tau_max=tau_max,
            update_every=update_every,
            sample_prob=sample_prob,
            min_mask_prob=min_mask_prob,
            max_mask_prob=max_mask_prob,
            min_mask_span=min_mask_span,
            max_mask_span=max_mask_span,
            random_mask_mode=random_mask_mode,
            max_s=max_s,
            fixed_tau=fixed_tau,
            update_test_time=update_test_time,
            sample_in_inference=sample_in_inference,
            fixed_mask_ratio=fixed_mask_ratio,
        )

        if train_r_min is None:
            train_r_min = float(r)
        if train_r_max is None:
            train_r_max = float(r)
        train_r_min = float(train_r_min)
        train_r_max = float(train_r_max)
        if not (0.0 <= train_r_min <= 1.0):
            raise ValueError("BatchTopKTrainPerSeq: train_r_min must be in [0, 1]")
        if not (0.0 <= train_r_max <= 1.0):
            raise ValueError("BatchTopKTrainPerSeq: train_r_max must be in [0, 1]")
        if train_r_max < train_r_min:
            raise ValueError("BatchTopKTrainPerSeq: train_r_max must be >= train_r_min")
        train_r_sampling = str(train_r_sampling).lower()
        if train_r_sampling not in {"uniform", "beta"}:
            raise ValueError("BatchTopKTrainPerSeq: train_r_sampling must be 'uniform' or 'beta'")
        if float(train_r_beta_alpha) <= 0.0 or float(train_r_beta_beta) <= 0.0:
            raise ValueError("BatchTopKTrainPerSeq: beta parameters must be > 0")
        self.train_r_min = train_r_min
        self.train_r_max = train_r_max
        self.train_r_sampling = train_r_sampling
        self.train_r_beta_alpha = float(train_r_beta_alpha)
        self.train_r_beta_beta = float(train_r_beta_beta)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training:
            return super().forward(x)

        device = x.device
        dtype = x.dtype
        B, N, _C = x.shape if x.ndim == 3 else (0, 0, 0)
        total = B * N

        if total == 0:
            mask = torch.zeros(B, N, device=device, dtype=torch.bool)
            avg_r = torch.zeros((), device=device, dtype=dtype)
            tau_used = self._get_tau_tensor(device, dtype)
            return mask, avg_r, tau_used

        # Sample per-sequence target r and convert to number of tokens to drop.
        if self.train_r_max > self.train_r_min:
            if self.train_r_sampling == "uniform":
                r_b = torch.empty(B, device=device, dtype=dtype).uniform_(self.train_r_min, self.train_r_max)
            else:  # beta on [train_r_min, train_r_max]
                beta_dist = torch.distributions.Beta(
                    torch.tensor(self.train_r_beta_alpha, device=device, dtype=torch.float32),
                    torch.tensor(self.train_r_beta_beta, device=device, dtype=torch.float32),
                )
                beta_u = beta_dist.sample((B,)).to(device=device, dtype=dtype)
                width = float(self.train_r_max - self.train_r_min)
                r_b = float(self.train_r_min) + width * beta_u
        else:
            r_b = torch.full((B,), self.train_r_min, device=device, dtype=dtype)
        r_upper = 1.0 if N <= 1 else (1.0 - (1.0 / float(N)))
        r_b = r_b.clamp(min=0.0, max=r_upper)
        K_b = torch.floor(r_b * float(N)).to(torch.long)
        K_b = K_b.clamp(min=0, max=max(0, N - 1))

        mask = torch.ones(B, N, device=device, dtype=torch.bool)
        mask[:, 0] = True

        # Implied per-seq tau (for logging/backward-compat): K-th largest similarity cut.
        tau_b = torch.full((B,), float(self.tau_max), device=device, dtype=dtype)

        if N > 1:
            sim = F.cosine_similarity(x[:, 1:, :], x[:, :-1, :], dim=-1)  # [B, N-1]
            Kmax = int(K_b.max().item())
            if Kmax > 0:
                vals, idx = torch.topk(sim, k=Kmax, dim=1)  # vals desc, idx in [0..N-2]

                # Remove the right token of the selected pairs.
                k_idx = torch.arange(Kmax, device=device).view(1, -1).expand(B, -1)
                valid = k_idx < K_b.view(B, 1)
                if valid.any():
                    sel = valid.nonzero(as_tuple=False)
                    b_sel = sel[:, 0]
                    j_sel = sel[:, 1]
                    pos = (idx[b_sel, j_sel] + 1).clamp(min=1, max=N - 1)
                    mask[b_sel, pos] = False

                # tau_b per sequence (only meaningful when K_b>0)
                k_sel = (K_b.clamp(min=1) - 1).view(B, 1)
                tau_pick = vals.gather(1, k_sel).squeeze(1)
                tau_b = torch.where(K_b > 0, tau_pick, tau_b)
                tau_b = tau_b.clamp(min=self.tau_min, max=self.tau_max)

        mask = self._apply_max_span_constraint(mask)

        # Optional random masking override/mix
        mask = self._maybe_apply_random_mask(mask, dtype)
        if N > 0:
            mask[:, 0] = True
        mask = self._apply_max_span_constraint(mask)

        avg_r = self._compute_avg_r(mask, total, dtype)
        tau_used = tau_b.mean()
        if self.fixed_tau is None:
            self.tau_train.fill_(float(tau_used.item()))
        return mask, avg_r, tau_used


class BatchGreedyTrainPerSeq(BatchGreedy):
    """Hybrid selector with train/eval split.

    - Training: per-sequence greedy masking with per-sample `r_b ~ Uniform(train_r_min, train_r_max)`.
      The intended behavior is to iteratively remove K_b tokens per sequence (true greedy),
      which can be expensive.

    - Eval/Inference: uses the original `BatchGreedy` (global tau controller).

    TODO: Implement train-time true greedy K_b removals (per-seq loops).
    """

    def __init__(
        self,
        r: float,
        initial_tau: float = 0.85,
        ema_mu: float = 0.95,
        eta0: float = 0.1,
        decay_T: float = 1000.0,
        tau_min: float = 1e-3,
        tau_max: float = 0.999,
        update_every: int = 1,
        sample_prob: float = 0.0,
        min_mask_prob: float = 0.0,
        max_mask_prob: float = 0.0,
        min_mask_span: int = 4,
        max_mask_span: int = 4,
        random_mask_mode: str = "start_geom",
        max_s: Optional[int] = None,
        fixed_tau: Optional[float] = None,
        update_test_time: bool = False,
        sample_in_inference: bool = False,
        fixed_mask_ratio: bool = False,
        *,
        train_r_min: Optional[float] = None,
        train_r_max: Optional[float] = None,
    ):
        super().__init__(
            r=r,
            initial_tau=initial_tau,
            ema_mu=ema_mu,
            eta0=eta0,
            decay_T=decay_T,
            tau_min=tau_min,
            tau_max=tau_max,
            update_every=update_every,
            sample_prob=sample_prob,
            min_mask_prob=min_mask_prob,
            max_mask_prob=max_mask_prob,
            min_mask_span=min_mask_span,
            max_mask_span=max_mask_span,
            random_mask_mode=random_mask_mode,
            max_s=max_s,
            fixed_tau=fixed_tau,
            update_test_time=update_test_time,
            sample_in_inference=sample_in_inference,
            fixed_mask_ratio=fixed_mask_ratio,
        )

        if train_r_min is None:
            train_r_min = float(r)
        if train_r_max is None:
            train_r_max = float(r)
        train_r_min = float(train_r_min)
        train_r_max = float(train_r_max)
        if not (0.0 <= train_r_min <= 1.0):
            raise ValueError("BatchGreedyTrainPerSeq: train_r_min must be in [0, 1]")
        if not (0.0 <= train_r_max <= 1.0):
            raise ValueError("BatchGreedyTrainPerSeq: train_r_max must be in [0, 1]")
        if train_r_max < train_r_min:
            raise ValueError("BatchGreedyTrainPerSeq: train_r_max must be >= train_r_min")
        self.train_r_min = train_r_min
        self.train_r_max = train_r_max

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            raise NotImplementedError(
                "BatchGreedyTrainPerSeq.train(): TODO implement per-sequence true greedy masking (iteratively remove K_b tokens)."
            )
        return super().forward(x)
