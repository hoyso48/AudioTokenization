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

class DifferentiablePLE(nn.Module):
    """
    Differentiable PLE using Sine-based phase detection and Straight-Through Estimator (STE).
    
    Mechanism:
    1. Compute accumulated distance L_i.
    2. Convert to phase: phi_i = (2 * pi * L_i) / tau
    3. Detect 'pulse' when phase completes a cycle (using Cosine Peak).
    4. Apply Soft-Thresholding (Sigmoid) to generate continuous probabilities.
    5. Use STE to discretize into binary mask during forward pass while passing gradients.
    
    Args:
        input_dim (int): Dimension of input features.
        r (float): Target masking ratio (used for regularization).
        initial_tau (float): Initial value for learnable tau.
        sharpness (float): Scaling factor for sigmoid to approximate step function.
        ste (bool): Whether to use Straight-Through Estimator for binary mask.
    """
    def __init__(
        self, 
        input_dim: int,
        r: float, 
        initial_tau: float = 1.0, 
        sharpness: float = 10.0,
        ste: bool = True
    ):
        super().__init__()
        self.r = r
        self.sharpness = sharpness
        self.ste = ste
        
        # Projection layer for distance calculation
        # Project to lower dimension (e.g., input_dim // 2 or fixed size) or keep same
        # Here we keep same dimension for simplicity, but it adds learnable flexibility.
        self.proj = nn.Linear(input_dim, input_dim)
        
        # Learnable tau (log-parameterized for positivity)
        self.log_tau = nn.Parameter(torch.tensor(math.log(initial_tau)))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        B, N, C = x.shape
        
        # 0. Project input to "Distance Space"
        # This decouples feature representation from selection logic
        x_proj = self.proj(x)
        
        # 1. Compute Distance & Accumulated Path (Same as PLE)
        if N > 1:
            sim = F.cosine_similarity(x_proj[:, 1:, :], x_proj[:, :-1, :], dim=-1)
            d = torch.zeros(B, N, device=device, dtype=dtype)
            d[:, 1:] = (1.0 - sim)
            
            # Distance Normalization (Crucial for stability)
            d_mean = d.mean(dim=1, keepdim=True)
            d = d / (d_mean + 1e-6)
            
            D = torch.cumsum(d, dim=1) # [B, N]
        else:
            D = torch.zeros(B, N, device=device, dtype=dtype)
            
        tau = torch.exp(self.log_tau)
        
        # 2. Phase & Pulse Generation (Differentiable Peak Detection)
        # We want a pulse every time D increases by tau.
        # Cosine(2*pi*D/tau) has peaks at D = 0, tau, 2tau, ...
        
        # Shift phase by pi so peak is at k*tau
        # cos(2pi*x - pi) = -cos(2pi*x)
        phase = (2 * math.pi * D) / (tau + 1e-6)
        scores = -torch.cos(phase) # [B, N]
        
        # 3. Soft Peak Picking (Differentiable Local Maxima)
        # Condition: score[i] > score[i-1] AND score[i] > score[i+1]
        
        # Pad scores to handle boundaries
        # Left: compare with -1 (so first token is always > prev)
        # Right: compare with -1 (so last token is always > next)
        # But actually, we want to force-keep the first token separately,
        # so we just focus on finding peaks in the sequence.
        scores_padded = F.pad(scores, (1, 1), value=-1.1) # [B, N+2]
        
        left = scores_padded[:, :-2]   # [B, N]
        center = scores_padded[:, 1:-1] # [B, N] (== scores)
        right = scores_padded[:, 2:]   # [B, N]
        
        # Soft comparisons using Sigmoid
        # If center > left, diff > 0 => sigmoid > 0.5
        # large sharpness makes it close to step function but differentiable
        diff_left = center - left
        diff_right = center - right
        
        is_peak = torch.sigmoid(self.sharpness * diff_left) * \
                  torch.sigmoid(self.sharpness * diff_right)
                  
        # 4. Soft Thresholding
        # Peak must be high enough (close to 1.0) to be a valid boundary
        # This filters out small wiggles in the valleys
        threshold = 0.0 # Cosine range -1 to 1, 0.0 is mid-point
        is_high = torch.sigmoid(self.sharpness * (center - threshold))
        
        prob = is_peak * is_high
        
        # Always keep first token
        prob = torch.cat([torch.ones(B, 1, device=device, dtype=dtype), prob[:, 1:]], dim=1)

        # 5. Straight-Through Estimator (STE)
        if self.ste:
            # Forward: Binary (0 or 1)
            # Backward: Gradient of probability
            mask_binary = (prob > 0.5).float()
            mask = prob + (mask_binary - prob).detach()
        else:
            mask = prob
            
        # Compute masked ratio (differentiable)
        kept_count = mask.sum(dim=1)
        avg_r = 1.0 - (kept_count / N).mean()
        
        # Convert to bool mask for compatibility
        bool_mask = mask > 0.5
        
        # Compute Auxiliary Loss (MSE between avg_r and target_r)
        # Multiplied by 100 to keep scale reasonable if r is small
        aux_loss = (avg_r - self.r) ** 2
        
        return bool_mask, avg_r, tau, aux_loss

# --- Components for SigmoidSTE (Copied from vq/module.py to avoid dependency/circular import issues) ---

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

class PLEBatchTopK(nn.Module):
    """
    Batch-level PLE (Path-Length Equalization) with a single global tau.

    Design:
      - Use one global scalar tau for both training and inference.
      - During training, update tau once per step with a Robbins–Monro style controller
        in the log-domain using the observed masked ratio (avg_r).
      - Selection uses first-crossing with global tau; no per-sequence budgets and
        no "strictly increasing" post-fix (keep the standard first-crossing behavior).

    Inputs:
      x: [B, N, C]

    Returns:
      mask:   [B, N] bool, True = keep (unmasked)
      avg_r:  scalar tensor, masked ratio = (#zeros in mask)/(B*N)
      tau_used: scalar tensor, the tau used for this step's selection
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
    ):
        super().__init__()
        if not (0.0 <= float(r) < 1.0):
            raise ValueError("PLEBatchTopK: r must be in [0, 1)")
        if initial_tau <= 0.0:
            raise ValueError("PLEBatchTopK: initial_tau must be > 0")

        # Target masked ratio
        self.r = float(r)

        # Controller hyperparameters (Robbins–Monro in log-domain)
        self.ema_mu = float(ema_mu)          # EMA for observed masked ratio
        self.eta0 = float(eta0)              # initial step size for log-tau updates
        self.decay_T = float(decay_T)        # time-scale for step-size decay
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.update_every = int(update_every)
        self.sample_prob = float(sample_prob)

        # Persistent controller state
        self.register_buffer("log_tau", torch.tensor(math.log(float(initial_tau))))
        self.register_buffer("r_ema", torch.tensor(0.0))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _update_tau(self, avg_r: torch.Tensor) -> None:
        """
        Robbins–Monro style update in the log-domain:
          log_tau_{t+1} = log_tau_t - eta_t * (r_ema - r_target)
        with:
          r_ema  = mu * r_ema + (1 - mu) * avg_r
          eta_t  = eta0 / sqrt(1 + steps/decay_T)
        """
        # Increment step counter
        self.steps.add_(1)
        t = float(self.steps.item())

        # EMA of observed masked ratio (scalar in [0, 1])
        r_hat = float(avg_r.item())
        self.r_ema.mul_(self.ema_mu).add_((1.0 - self.ema_mu) * r_hat)

        # Decayed step size (Robbins–Monro schedule)
        eta_t = self.eta0 / math.sqrt(1.0 + (t / max(1.0, self.decay_T)))

        # Gradient step in log-domain toward r_target
        error = float(self.r_ema.item() - self.r)
        new_log_tau = float(self.log_tau.item()) - eta_t * error

        # Clamp and write back
        new_log_tau = min(max(new_log_tau, math.log(self.tau_min)), math.log(self.tau_max))
        self.log_tau.copy_(torch.tensor(new_log_tau, device=self.log_tau.device, dtype=self.log_tau.dtype))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        B, N, C = x.shape if x.ndim == 3 else (0, 0, 0)
        total = B * N

        # Early exit on empty input
        if total == 0:
            mask = torch.zeros(B, N, device=device, dtype=torch.bool)
            avg_r = torch.zeros((), device=device, dtype=dtype)
            tau_used = torch.exp(self.log_tau).to(device=device, dtype=dtype)
            return mask, avg_r, tau_used

        # Adjacent dissimilarities and cumulative path per sequence
        if N > 1:
            sim = F.cosine_similarity(x[:, 1:, :], x[:, :-1, :], dim=-1)
            d = torch.zeros(B, N, device=device, dtype=dtype)
            d[:, 1:] = (1.0 - sim).to(dtype)

            # Apply Distance Normalization (Sequence-wise Mean Normalization)
            # This stabilizes tau against distribution shifts.
            # tau now represents "relative change" rather than "absolute cosine distance".
            # d_mean = d.mean(dim=1, keepdim=True)  # [B, 1]
            # d = d / (d_mean + 1e-6)

        else:
            d = torch.zeros(B, N, device=device, dtype=dtype)

        D = torch.zeros(B, N, device=device, dtype=dtype)
        if N > 1:
            D[:, 1:] = torch.cumsum(d[:, 1:], dim=1)
        L = D[:, -1] if N > 0 else torch.zeros(B, device=device, dtype=dtype)

        # Use the current global tau (same in train/eval for selection)
        tau_used = torch.exp(self.log_tau).to(device=device, dtype=dtype)

        # Build frontier mask via first-crossing with global tau
        mask = torch.zeros(B, N, device=device, dtype=torch.bool)
        if N > 0:
            mask[:, 0] = True  # always keep the first token

        if N > 1 and torch.isfinite(tau_used) and (tau_used.item() > 0.0):
            # m_b = floor(L_b / tau)
            m_b = torch.floor(L / tau_used).to(torch.long)
            # clamp m_b to at most N-1 and at least 0
            m_b = torch.clamp(m_b, min=0, max=N - 1)
            max_m = int(m_b.max().item())

            if max_m > 0:
                # targets: k * tau for k = 1..max_m, shared across batch
                targets = (torch.arange(1, max_m + 1, device=device, dtype=dtype) * tau_used).view(1, -1)
                # ge: (B, N, max_m), True when D >= k * tau
                ge = D.unsqueeze(2) >= targets.view(1, 1, -1)
                # First-crossing index along the N dimension
                j = ge.float().argmax(dim=1).clamp_(min=1, max=N - 1)  # (B, max_m)

                # Apply only for valid slots k <= m_b (no "strictly increasing" post-fix)
                k_idx = torch.arange(1, max_m + 1, device=device).view(1, -1).expand(B, -1)
                valid = k_idx <= m_b.view(B, 1)
                if valid.any():
                    sel = valid.nonzero(as_tuple=False)
                    b_sel = sel[:, 0]
                    k_sel = sel[:, 1]
                    pos = j[b_sel, k_sel]
                    mask[b_sel, pos] = True

        # Compute masked ratio for this (local) process
        kept_total = int(mask.sum().item())
        zeros_total = int(total - kept_total)

        # Optionally compute global avg_r across distributed workers (if initialized)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            agg = torch.tensor([zeros_total, total], device=device, dtype=torch.long)
            torch.distributed.all_reduce(agg, op=torch.distributed.ReduceOp.SUM)
            zeros_total = int(agg[0].item())
            total = int(agg[1].item())

        avg_r = torch.tensor(float(zeros_total) / float(max(1, total)), device=device, dtype=dtype)

        # Training-time controller update (single global tau)
        if self.training and (int(self.steps.item()) % max(1, self.update_every) == 0):
            self._update_tau(avg_r)

        # Overwrite with Random Masking if sample_prob > 0
        if self.training and self.sample_prob > 0.0:
            if torch.rand(1).item() < self.sample_prob:
                # Random masking: mask every token with probability r (keep with 1-r)
                probs = torch.full((B, N), 1.0 - self.r, device=device, dtype=dtype)
                mask = torch.bernoulli(probs).bool()

        return mask, avg_r, tau_used