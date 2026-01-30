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

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x, dtype = x.float(), x.dtype
        output = self._norm(x)
        return (output * self.weight).to(dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute caches for initial max seq len
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len: int = None):
        # x is only used for device/dtype; seq_len must be provided explicitly
        if seq_len is None:
            raise ValueError("RotaryEmbedding.forward requires seq_len to be provided")

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SelfAttention(nn.Module):
    def __init__(self, dim, window_size=(64, 64), n_head=8, dropout=0.1, max_position_embeddings=2048, base=10000, causal: bool = False, norm_eps: float = 1e-2):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.causal = causal
        
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.dropout = dropout
        self.window_size = window_size
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, 
                                        max_position_embeddings=max_position_embeddings, 
                                        base=base, 
                                        device=None)
        # Require FlashAttention (dense and varlen kernels). Raise immediately if unavailable.
        try:
            from flash_attn import flash_attn_qkvpacked_func as _fa_dense_qkv
        except Exception as e:
            # Fallback for development environments without flash_attn
            _fa_dense_qkv = None
            # raise ImportError("FlashAttention is required but not installed: missing flash_attn_qkvpacked_func") from e
        try:
            from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as _fa_varlen_qkv
        except Exception as e:
            _fa_varlen_qkv = None
            # raise ImportError("FlashAttention is required but not installed: missing flash_attn_varlen_qkvpacked_func") from e

        self.flash_attn_qkvpacked_func = _fa_dense_qkv
        self.flash_attn_varlen_qkvpacked_func = _fa_varlen_qkv

    def forward(self, x, position_ids=None, cu_seqlens: torch.Tensor = None, max_seqlen: int = None):
        # Packed varlen path: expect x = [total_tokens, C], cu_seqlens int32 [B+1], max_seqlen int
        if cu_seqlens is not None and max_seqlen is not None:
            total, C = x.shape
            # Ensure cu_seqlens is int32 on the same device
            cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32)

            # QKV projection then pack for FA varlen qkvpacked: [total, 3C] -> [total, 3, n_head, head_dim]
            qkv = self.qkv_proj(x)
            qkv = qkv.view(total, 3, self.n_head, self.head_dim)

            # RoPE for varlen packed: build position ids and rotate q/k
            if total > 0 and max_seqlen > 0:
                cos, sin = self.rotary_emb(qkv, seq_len=max_seqlen)
                lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
                if lengths.numel() > 0:
                    if position_ids is None:
                        start_offsets = torch.repeat_interleave(cu_seqlens[:-1].to(torch.long), lengths)
                        token_idx = torch.arange(total, device=qkv.device, dtype=torch.long)
                        position_ids = token_idx - start_offsets  # [total]
                    q, k = self.q_norm(qkv[:, 0]), self.k_norm(qkv[:, 1])
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=1)
                    qkv = torch.stack((q, k, qkv[:, 2]), dim=1)

            # FlashAttention varlen qkvpacked
            if self.flash_attn_varlen_qkvpacked_func is not None:
                out = self.flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_seqlen,
                    dropout_p=self.dropout if self.training else 0.0,
                    softmax_scale=self.head_dim ** -0.5,
                    causal=self.causal,
                    window_size=self.window_size,
                )  # [total, n_head, head_dim]
            else:
                # Fallback? Or assume dense if FA missing
                raise ImportError("FlashAttention required")

            # Merge heads, out proj on packed
            out = out.reshape(total, -1)  # [total, C]
            out = self.out_proj(out)      # [total, C]
            return out

        # Dense path: use FlashAttention dense kernel with (B, T, C)
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)

        # Apply RoPE on qkv packed (dense)
        cos, sin = self.rotary_emb(qkv, seq_len=T)
        q, k = self.q_norm(qkv[:, :, 0]), self.k_norm(qkv[:, :, 1])

        if position_ids is None:
            position_ids = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=2)
        qkv = torch.stack((q, k, qkv[:, :, 2]), dim=2)

        # FlashAttention qkvpacked (dense)
        if self.flash_attn_qkvpacked_func is not None:
            out = self.flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.head_dim ** -0.5,
                causal=self.causal,
                window_size=self.window_size,
            )  # [B, T, n_head, head_dim]
        else:
             raise ImportError("FlashAttention required")

        out = out.reshape(B, T, C)
        out = self.out_proj(out)

        return out

class SigmoidSTE(nn.Module):
    """
    Differentiable Mask Predictor using Self-Attention and Sigmoid + STE.
    Predicts an importance score for each token and selects based on prob.
    
    Args:
        input_dim (int): Input feature dimension.
        r (float): Target masking ratio.
        n_head (int): Number of heads for SelfAttention.
        ste (bool): Whether to use Straight-Through Estimator.
    """
    def __init__(self, input_dim: int, r: float, n_head: int = 8, dropout: float = 0.0, ste: bool = True):
        super().__init__()
        self.r = float(r)
        self.ste = ste
        
        # Lightweight Predictor: SelfAttn -> Linear -> Sigmoid
        self.attn = SelfAttention(
            dim=input_dim,
            n_head=n_head,
            dropout=dropout,
            max_position_embeddings=2048,
            causal=False
        )
        # Final projection to 1 scalar per token
        self.proj = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, C]
        B, T, C = x.shape
        
        # 1. Self-Attention for context
        # Residual connection is often good
        h = x + self.attn(x)
        
        # 2. Predict Probability
        logits = self.proj(h) # [B, T, 1]
        prob = self.sigmoid(logits).squeeze(-1) # [B, T]
        
        # 3. Straight-Through Estimator
        if self.ste:
            # Forward: Binary
            mask_binary = (prob > 0.5).float()
            # Backward: Gradient passes through prob
            mask = prob + (mask_binary - prob).detach()
        else:
            mask = prob
            
        # 4. Compute Aux Loss (Regularize mean(mask) to target ratio)
        # target kept ratio = 1 - r
        # avg_r = 1 - mean(mask)
        kept_count = mask.sum(dim=1) # [B]
        current_kept_ratio = kept_count / T
        avg_r = 1.0 - current_kept_ratio.mean()
        
        # We want current_kept_ratio approx (1 - self.r)
        # Or avg_r approx self.r
        aux_loss = (avg_r - self.r) ** 2
        
        # Dummy tau (not used)
        tau = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        
        bool_mask = mask > 0.5
        
        return bool_mask, avg_r, tau, aux_loss

class FixedPatternMasking(nn.Module):
    """
    Fixed deterministic masking pattern based on target ratio r.
    Selects tokens at regular intervals (stride = 1 / (1-r)).
    Useful for baselines or debugging.
    """
    def __init__(self, r: float, **kwargs):
        super().__init__()
        self.r = float(r)
        # Calculate stride: if r=0.5 (keep 0.5), stride=2. If r=0.75 (keep 0.25), stride=4.
        self.keep_ratio = 1.0 - self.r
        if self.keep_ratio <= 0:
            raise ValueError("r must be < 1.0")
        self.stride = max(1, int(round(1.0 / self.keep_ratio)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, N, C]
        device = x.device
        B, N, C = x.shape
        
        mask = torch.zeros(B, N, device=device, dtype=torch.bool)
        
        # Set mask=True every 'stride' steps
        # Always keep the first token (index 0)
        mask[:, ::self.stride] = True
        
        # Compute actual masked ratio
        kept_total = int(mask.sum().item())
        total = B * N
        zeros_total = int(total - kept_total)
        
        avg_r = torch.tensor(float(zeros_total) / float(max(1, total)), device=device, dtype=x.dtype)
        
        # Dummy tau
        tau_used = torch.tensor(float(self.stride), device=device, dtype=x.dtype)
        
        return mask, avg_r, tau_used

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
        max_s: Optional[int] = None,
        fixed_tau: Optional[float] = None,
        update_test_time: bool = False,
        invert_update: bool = False,
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
        self.controller_sign = -1.0 if invert_update else 1.0

        if self.min_mask_span <= 0 or self.max_mask_span <= 0:
            raise ValueError("Batch selector: mask spans must be >= 1")
        if self.max_mask_span < self.min_mask_span:
            raise ValueError("Batch selector: max_mask_span must be >= min_mask_span")

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
        if not self.training or self.sample_prob <= 0.0:
            return mask
        if torch.rand(1, device=mask.device).item() >= self.sample_prob:
            return mask
        B, N = mask.shape
        if B * N == 0:
            return mask
        device = mask.device

        # Per-sequence sampling for mask probability and span length
        mask_prob = torch.empty(B, device=device, dtype=dtype).uniform_(
            self.min_mask_prob, self.max_mask_prob
        )
        span_lengths = torch.randint(
            low=self.min_mask_span,
            high=self.max_mask_span + 1,
            size=(B,),
            device=device,
        )

        keep_probs = (1.0 - mask_prob).clamp(min=0.0, max=1.0).unsqueeze(1).expand(B, N)
        random_keep = torch.bernoulli(keep_probs).bool()

        if N == 0:
            return random_keep

        final_mask = random_keep.clone()

        # Vectorized span masking
        to_mask = ~random_keep
        bs_idx, pos_idx = to_mask.nonzero(as_tuple=True)
        if bs_idx.numel() == 0:
            return final_mask

        span_offsets = torch.arange(self.max_mask_span, device=device, dtype=torch.long)
        start_expanded = pos_idx.unsqueeze(1) + span_offsets  # [K, max_span]
        span_cap = span_lengths[bs_idx].unsqueeze(1)  # [K, 1]
        valid = span_offsets.unsqueeze(0) < span_cap  # [K, max_span]

        idx = start_expanded[valid]
        bsel = bs_idx.unsqueeze(1).expand_as(start_expanded)[valid]

        within_bounds = idx < N
        if within_bounds.any():
            final_mask[bsel[within_bounds], idx[within_bounds]] = False

        return final_mask


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
        min_mask_span: int = 1,
        max_mask_span: int = 1,
        max_s: Optional[int] = None,
        fixed_tau: Optional[float] = None,
        update_test_time: bool = False,
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
            max_s=max_s,
            fixed_tau=fixed_tau,
            update_test_time=update_test_time,
            invert_update=False,
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
            m_b = torch.floor(L / tau_used).to(torch.long)
            m_b = torch.clamp(m_b, min=0, max=N - 1)
            max_m = int(m_b.max().item())

            if max_m > 0:
                targets = (torch.arange(1, max_m + 1, device=device, dtype=dtype) * tau_used).view(1, -1)
                ge = D.unsqueeze(2) >= targets.view(1, 1, -1)
                j = ge.float().argmax(dim=1).clamp_(min=1, max=N - 1)

                k_idx = torch.arange(1, max_m + 1, device=device).view(1, -1).expand(B, -1)
                valid = k_idx <= m_b.view(B, 1)
                if valid.any():
                    sel = valid.nonzero(as_tuple=False)
                    b_sel = sel[:, 0]
                    k_sel = sel[:, 1]
                    pos = j[b_sel, k_sel]
                    mask[b_sel, pos] = True

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
        min_mask_span: int = 1,
        max_mask_span: int = 1,
        max_s: Optional[int] = None,
        fixed_tau: Optional[float] = None,
        update_test_time: bool = False,
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
            max_s=max_s,
            fixed_tau=fixed_tau,
            update_test_time=update_test_time,
            invert_update=True,
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
        min_mask_span: int = 1,
        max_mask_span: int = 1,
        max_s: Optional[int] = None,
        fixed_tau: Optional[float] = None,
        update_test_time: bool = False,
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
            max_s=max_s,
            fixed_tau=fixed_tau,
            update_test_time=update_test_time,
            invert_update=True,
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