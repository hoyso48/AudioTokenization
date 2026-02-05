import torch.nn as nn
from einops import rearrange
from . import activations
from .alias_free_torch import *
from torch.nn.utils import weight_norm
from torch import Tensor
import torch.nn.functional as F
import torch
from typing import Tuple, Sequence, Union
import math

class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        device=None,
        dtype=None
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode='zeros',
            device=device,
            dtype=dtype
        )
        
        self.padding_mode = 'constant' if padding_mode == 'zeros' else padding_mode
        self.padding = (kernel_size - stride) * dilation #padding
        
    def forward(self, x):
        x = F.pad(x, (self.padding, 0), mode=self.padding_mode)
        out = self.conv(x)
        return out            

class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype=None):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, bias=bias, device=device, dtype=dtype)
        self.stride = stride

    def forward(self, x):
        return self.conv(x)[..., :-self.stride]

def WNConv1d(*args, causal=False, **kwargs):
    if causal:
        conv = CausalConv1d(*args, **kwargs)
        conv.conv = weight_norm(conv.conv)
        return conv
    else:
        return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, causal=False, **kwargs):
    if causal:
        conv = CausalConvTranspose1d(*args, **kwargs)
        conv.conv = weight_norm(conv.conv)
        return conv
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

class WNConv1dVarlen(nn.Module):
    """
    Weight-normalized Conv1d wrapper for variable-length (packed) sequences.

    Input:
      - x: packed tokens of shape (total_tokens, in_channels)
      - cu_seqlens: int tensor of shape (B + 1,), cumulative sequence lengths (cu_seqlens[0] = 0)
      - max_seqlen: int, max sequence length in the batch

    Behavior:
      1) Pack -> padded: (total_tokens, C) -> (B, max_seqlen, C) with zero-padding
      2) Apply Conv1d over the time axis
      3) Unpad -> packed: (B, T_out, C_out) -> (total_tokens, C_out) by discarding padded positions

    Notes:
      - This assumes the Conv1d preserves the time length (T_out == max_seqlen).
        If your convolution changes the time dimension (e.g., stride != 1), this module
        will raise an error. Extend the mapping logic if you need strided/downsampled outputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding=0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.conv = WNConv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _validate_varlen_inputs(x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> Tuple[int, int, int]:
        if x.dim() != 2:
            raise ValueError(f"WNConv1dVarlen.forward expects x of shape (total_tokens, C), got {tuple(x.shape)}")
        total_tokens, C = x.shape

        if cu_seqlens is None:
            raise ValueError("WNConv1dVarlen.forward requires cu_seqlens (shape: (B+1,))")
        if cu_seqlens.dim() != 1:
            raise ValueError(f"cu_seqlens must be 1D (shape: (B+1,)), got {tuple(cu_seqlens.shape)}")
        if cu_seqlens.numel() < 1:
            raise ValueError("cu_seqlens must have at least one element")
        if cu_seqlens.numel() == 1:
            # B == 0 is not meaningful; treat as empty batch.
            B = 0
        else:
            B = int(cu_seqlens.numel() - 1)

        if not isinstance(max_seqlen, int):
            raise TypeError(f"max_seqlen must be int, got {type(max_seqlen)}")
        if max_seqlen < 0:
            raise ValueError(f"max_seqlen must be >= 0, got {max_seqlen}")

        if cu_seqlens.numel() >= 1:
            if cu_seqlens[0].item() != 0:
                raise ValueError(f"cu_seqlens[0] must be 0, got {cu_seqlens[0].item()}")
            if cu_seqlens[-1].item() != total_tokens:
                raise ValueError(
                    "cu_seqlens[-1] must equal total_tokens. "
                    f"Got cu_seqlens[-1]={cu_seqlens[-1].item()} but total_tokens={total_tokens}."
                )
            if cu_seqlens.numel() > 1:
                diffs = cu_seqlens[1:] - cu_seqlens[:-1]
                if torch.any(diffs < 0):
                    raise ValueError("cu_seqlens must be non-decreasing")
                if torch.any(diffs > max_seqlen):
                    raise ValueError("Found a sequence length > max_seqlen; max_seqlen is inconsistent with cu_seqlens")

        return B, total_tokens, C

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        B, total_tokens, C = self._validate_varlen_inputs(x, cu_seqlens, max_seqlen)

        if C != self.in_channels:
            raise ValueError(
                f"WNConv1dVarlen: channel mismatch. x has C={C}, but conv expects in_channels={self.in_channels}."
            )

        # Empty (no tokens) fast-path
        if total_tokens == 0:
            return x.new_zeros((0, self.out_channels))

        # Ensure cu_seqlens is on the right device and integer type for indexing
        cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.long)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)  # (B,)
        if lengths.numel() != B:
            raise RuntimeError("Internal error: lengths shape mismatch with batch size")

        # Build packed -> padded index mapping (vectorized, no Python loops)
        # batch_idx: (total_tokens,), token positions within each sequence: (total_tokens,)
        batch_idx = torch.repeat_interleave(torch.arange(B, device=x.device, dtype=torch.long), lengths)
        start_offsets = torch.repeat_interleave(cu_seqlens[:-1], lengths)
        token_idx = torch.arange(total_tokens, device=x.device, dtype=torch.long)
        pos_idx = token_idx - start_offsets

        if batch_idx.numel() != total_tokens or pos_idx.numel() != total_tokens:
            raise RuntimeError("Internal error: packed index mapping has incorrect size")
        if torch.any(pos_idx < 0) or torch.any(pos_idx >= max_seqlen):
            raise ValueError("Found token positions out of [0, max_seqlen); check cu_seqlens/max_seqlen consistency")

        # Padded tensor: (B, max_seqlen, C)
        padded = x.new_zeros((B, max_seqlen, C))
        padded[batch_idx, pos_idx] = x

        # Conv over time: (B, C, T)
        y = self.conv(padded.transpose(1, 2)).transpose(1, 2)  # (B, T_out, C_out)

        if y.shape[0] != B:
            raise RuntimeError("Internal error: batch dimension changed by convolution")
        if y.shape[1] != max_seqlen:
            raise ValueError(
                "WNConv1dVarlen expects the convolution to preserve time length (T_out == max_seqlen), "
                f"but got T_out={y.shape[1]} and max_seqlen={max_seqlen}. "
                "Use stride=1 and appropriate padding (e.g., padding='same' or padding=(kernel_size-1)//2)."
            )

        # Unpad back to packed: (total_tokens, C_out)
        out = y[batch_idx, pos_idx]
        return out

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim, eps=eps)
        # self.weight = nn.Parameter(torch.ones(dim))

    # def _norm(self, x):
    #     return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # def forward(self, x):
    #     x, dtype = x.float(), x.dtype
    #     output = self._norm(x)
    #     return (output * self.weight).to(dtype)
    def forward(self, x):
        x, dtype = x.float(), x.dtype
        output = self.norm(x)
        return (output).to(dtype)

# class RMSNorm(torch.nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-2):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))

#     def _norm(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

#     def forward(self, x):
#         x, dtype = x.float(), x.dtype
#         output = self._norm(x)
#         return (output * self.weight).to(dtype)


def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

# Find dim range bounds based on rotations
def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(_yarn_find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)  # Clamp values just in case

def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def _yarn_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

class LlamaDynamicYaRNScaledRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, original_max_position_embeddings=2048, extrapolation_factor=1, attn_factor=1, beta_fast=32, beta_slow=1, finetuned=False, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        if finetuned:
            self.yarn(self.max_position_embeddings / self.original_max_position_embeddings, device)
        else:
            inv_freq = 1.0 / \
                (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.mscale = 1

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()

        self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len

            self.yarn(seq_len / self.max_position_embeddings, x.device)

            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(x.dtype), persistent=False)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

    def yarn(self, scale, device):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float().to(device)) * self.extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mscale = float(_yarn_get_mscale(scale) * self.attn_factor) # Get n-d magnitude scaling corrected for interpolation

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
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
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
        # self.rotary_emb = LlamaDynamicYaRNScaledRotaryEmbedding(self.head_dim, 
        #                                 max_position_embeddings=max_position_embeddings, 
        #                                 base=base, 
        #                                 device=None, 
        #                                 original_max_position_embeddings=original_max_position_embeddings) #1
        self.rotary_emb = RotaryEmbedding(self.head_dim, 
                                        max_position_embeddings=max_position_embeddings, 
                                        base=base, 
                                        device=None)
        # Require FlashAttention (dense and varlen kernels). Raise immediately if unavailable.
        try:
            from flash_attn import flash_attn_qkvpacked_func as _fa_dense_qkv
        except Exception as e:
            raise ImportError("FlashAttention is required but not installed: missing flash_attn_qkvpacked_func") from e
        try:
            from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as _fa_varlen_qkv
        except Exception as e:
            raise ImportError("FlashAttention is required but not installed: missing flash_attn_varlen_qkvpacked_func") from e

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
                    # start_offsets = torch.repeat_interleave(cu_seqlens[:-1].to(torch.long), lengths)
                    # token_idx = torch.arange(total, device=qkv.device, dtype=torch.long)
                    # position_ids = token_idx - start_offsets  # [total]
                    if position_ids is None:
                        start_offsets = torch.repeat_interleave(cu_seqlens[:-1].to(torch.long), lengths)
                        token_idx = torch.arange(total, device=qkv.device, dtype=torch.long)
                        position_ids = token_idx - start_offsets  # [total]
                    q, k = self.q_norm(qkv[:, 0]), self.k_norm(qkv[:, 1])
                    # q = qkv[:, 0]
                    # k = qkv[:, 1]
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=1)
                    qkv = torch.stack((q, k, qkv[:, 2]), dim=1)

            # FlashAttention varlen qkvpacked
            out = self.flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.head_dim ** -0.5,
                causal=self.causal,
                window_size=self.window_size,
            )  # [total, n_head, head_dim]

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
        # q = qkv[:, :, 0]
        # k = qkv[:, :, 1]
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=2)
        qkv = torch.stack((q, k, qkv[:, :, 2]), dim=2)

        # FlashAttention qkvpacked (dense)
        out = self.flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.head_dim ** -0.5,
            causal=self.causal,
            window_size=self.window_size,
        )  # [B, T, n_head, head_dim]

        out = out.reshape(B, T, C)
        out = self.out_proj(out)

        return out

class LayerScale(nn.Module):
    def __init__(self, d_model: int, gamma_init: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model) * gamma_init)

    def forward(self, x):
        return x * self.scale

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(2 * (dim * mult) / 3)
        multiple_of = 256
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # expects (B, T, C)
        out = self.w2(F.silu(self.w1(x)) * self.w3(x))
        out = self.dropout(out)
        return out

class ConformerConvModule(nn.Module):
    def __init__(self, dim, kernel_size=31, dropout=0.1, causal: bool = False, norm_eps: float = 1e-2):
        super().__init__()
        self.pointwise_conv1 = nn.Linear(dim, 2 * dim) #nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=-1)
        if causal:
            self.depthwise_conv = CausalConv1d(dim, dim, kernel_size=kernel_size, groups=dim)
        else:
            self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same')
        self.conv_norm = RMSNorm(dim, eps=norm_eps)
        self.silu = nn.SiLU()
        self.pointwise_conv2 = nn.Linear(dim, dim) #nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        out = self.pointwise_conv1(x)
        out = self.glu(out)
        out = self.depthwise_conv(out.transpose(1, 2)).transpose(1, 2)
        out = self.conv_norm(out)
        out = self.silu(out)
        out = self.pointwise_conv2(out)
        out = self.dropout(out)
        return out

class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        n_head=8,
        ffn_mult=4,
        dropout=0.1,
        max_position_embeddings=2048,
        base=10000,
        causal: bool = False,
        attn_window_size=(64, 64),
        norm_eps: float = 1e-2,
        layerscale_gamma_init: float = 1.0,
    ):
        super().__init__()
        self.ffn1 = FeedForward(dim, mult=ffn_mult, dropout=dropout)
        self.self_attn = SelfAttention(dim, window_size=attn_window_size, n_head=n_head, dropout=dropout, max_position_embeddings=max_position_embeddings, base=base, causal=causal, norm_eps=norm_eps)
        self.attn_scale = LayerScale(dim, gamma_init=layerscale_gamma_init)
        self.ffn_scale = LayerScale(dim, gamma_init=layerscale_gamma_init)
        self.ffn1_norm_in = RMSNorm(dim, eps=norm_eps)
        self.attn_norm_in = RMSNorm(dim, eps=norm_eps)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x, position_ids=None, cu_seqlens: torch.Tensor = None, max_seqlen: int = None):
        x = x + self.attn_scale(self.self_attn(self.attn_norm_in(x), position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen))

        x = x + self.ffn_scale(self.ffn1(self.ffn1_norm_in(x)))

        return x

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        n_layers,
        n_head=8,
        ffn_mult=4,
        dropout=0.1,
        max_position_embeddings=2048,
        base=10000.0,
        causal: bool = False,
        attn_window_size=(64, 64),
        norm_eps: float = 1e-2,
        layerscale_gamma_init: float = 1.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(
                dim,
                n_head,
                ffn_mult,
                dropout,
                max_position_embeddings=max_position_embeddings,
                base=base,
                causal=causal,
                attn_window_size=attn_window_size,
                norm_eps=norm_eps,
                layerscale_gamma_init=layerscale_gamma_init,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim, eps=norm_eps)

    def forward(self, x, position_ids=None, cu_seqlens: torch.Tensor = None, max_seqlen: int = None):
        for layer in self.layers:
            x = layer(x, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = self.norm(x)
        return x

class Patchify2D(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=False)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2) #(B, H, W, C) -> (B, C, H, W)
        x = self.conv(x) #(B, C, H//patch_size, W//patch_size)
        x = x.permute(0, 2, 3, 1) #(B, H//patch_size, W//patch_size, C)
        return x

class UnPatchify2D(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=False)
    
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2) #(B, C, H, W) -> (B, C, H, W)
        x = self.conv(x) #(B, C, H*patch_size, W*patch_size)
        x = x.permute(0, 2, 3, 1) #(B, H*patch_size, W*patch_size, C)
        return x
    
# ---------------------------------------------------------------------------
# Legacy patchify/unpatchify (kept for reference)
#   - Patchify: Conv1d(stride=patch_size) tokenization
#   - UnPatchify: ConvTranspose1d(stride=patch_size) waveform expansion
# ---------------------------------------------------------------------------
# class Patchify1D(nn.Module):
#     def __init__(self, in_channels, out_channels, patch_size):
#         super().__init__()
#         self.patch_size = patch_size
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=False)
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # (B, N, C) -> (B, C, N)
#         x = self.conv(x)        # (B, C', N//patch_size)
#         x = x.permute(0, 2, 1)  # (B, N//patch_size, C')
#         return x
#
# class UnPatchify1D(nn.Module):
#     def __init__(self, in_channels, out_channels, patch_size):
#         super().__init__()
#         self.patch_size = patch_size
#         self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, bias=False)
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # (B, N, C) -> (B, C, N)
#         x = self.conv(x)        # (B, C', N*patch_size)
#         x = x.permute(0, 2, 1)  # (B, N*patch_size, C')
#         return x

# ---------------------------------------------------------------------------
# New patchify/unpatchify: rearrange -> Conv1d(k=7,p=3) / Conv1d(k=7,p=3) -> rearrange
# - Patchify uses a local-context conv projection across patch indices (N).
# - UnPatchify predicts per-token waveform patches (patch_size samples) using Conv1d.
# ---------------------------------------------------------------------------
class Patchify1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int):
        super().__init__()
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError("Patchify1D: patch_size must be > 0")
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        if in_channels <= 0:
            raise ValueError("Patchify1D: in_channels must be > 0")
        if out_channels <= 0:
            raise ValueError("Patchify1D: out_channels must be > 0")

        # Patchify: [B, T, C] -> [B, N, patch_size*C] -> [B, patch_size*C, N] -> conv -> [B, out, N]
        self.conv = WNConv1d(
            in_channels * self.patch_size,
            out_channels,
            kernel_size=7,
            padding=3,
            bias=True,
            causal=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        if x.dim() != 3:
            raise ValueError("Patchify1D.forward expects x of shape (B, T, C)")
        B, T, C = x.shape
        if C * self.patch_size != int(self.conv.in_channels):
            raise ValueError(
                "Patchify1D: input channel mismatch. "
                f"Got C={C}, patch_size={self.patch_size} => {C * self.patch_size} channels, "
                f"but conv expects {int(self.conv.in_channels)}."
            )

        # Keep the same effective rate as stride=patch_size: drop remainder.
        N = T // self.patch_size
        if N <= 0:
            raise ValueError(f"Patchify1D: input too short (T={T}) for patch_size={self.patch_size}")
        T_trim = N * self.patch_size
        if T_trim != T:
            x = x[:, :T_trim, :]

        # [B, T, C] -> [B, N, patch_size*C] -> [B, patch_size*C, N]
        x = x.reshape(B, N, self.patch_size * C).transpose(1, 2)
        x = self.conv(x)              # [B, out_channels, N]
        x = x.transpose(1, 2)         # [B, N, out_channels]
        return x


class UnPatchify1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int):
        super().__init__()
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError("UnPatchify1D: patch_size must be > 0")
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        if in_channels <= 0:
            raise ValueError("UnPatchify1D: in_channels must be > 0")
        if out_channels <= 0:
            raise ValueError("UnPatchify1D: out_channels must be > 0")

        # UnPatchify: [B, N, C] -> [B, C, N] -> conv -> [B, out_channels*patch_size, N]
        #            -> [B, N*patch_size, out_channels]
        self.conv = WNConv1d(
            in_channels,
            out_channels * self.patch_size,
            kernel_size=7,
            padding=3,
            bias=False,
            causal=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        if x.dim() != 3:
            raise ValueError("UnPatchify1D.forward expects x of shape (B, N, C)")
        B, N, C = x.shape
        if C != int(self.conv.in_channels):
            raise ValueError(
                "UnPatchify1D: input channel mismatch. "
                f"Got C={C}, but conv expects in_channels={int(self.conv.in_channels)}."
            )

        x = x.transpose(1, 2)         # [B, C, N]
        x = self.conv(x)              # [B, out_channels*patch_size, N]
        x = x.transpose(1, 2)         # [B, N, out_channels*patch_size]
        x = x.reshape(B, N * self.patch_size, -1)  # [B, N*patch_size, out_channels]
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=nn.SiLU()):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=stride // 2 + stride % 2)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
        self.activation = activation

    def forward(self, x):
        # x: (B, T, C)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        return x

class ConvSubsample1D(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU(), causal=False):
        super().__init__()
        self.causal = causal
        if self.causal:
            self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.activation = activation
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

class ConvSubsample2D(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU(), causal=False):
        super().__init__()
        self.causal = causal
        if self.causal:
            raise NotImplementedError("Causal convolution is not supported for 2D convolution")
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.activation = activation
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x.permute(0, 2, 3, 1) #(B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        return x

class ConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU(), causal=False, norm_eps: float = 1e-2):
        super().__init__()
        self.causal = causal
        if causal:
            self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.activation = activation
        self.norm = RMSNorm(out_channels, eps=norm_eps)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x
    
class ConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU(), causal=False, norm_eps: float = 1e-2):
        super().__init__()
        self.causal = causal
        if self.causal:
            self.conv1 = CausalConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=2)
            self.conv2 = CausalConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=1)
        else:
            self.conv1 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv2 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = activation
        self.norm = RMSNorm(out_channels, eps=norm_eps)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x
        
# class ConvDownsample(nn.Module):
#     def __init__(self, in_channels, out_channels, activation=nn.SiLU(), causal=False, norm_eps: float = 1e-2):
#         super().__init__()
#         self.causal = causal
#         if causal:
#             self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
#             self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
#         else:
#             self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
#             self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
#         self.activation = activation
#         self.norm = RMSNorm(out_channels, eps=norm_eps)
    
#     def forward(self, x):
#         x = x.transpose(1, 2)
#         x = self.conv1(x)
#         x = self.activation(x)
#         x = self.conv2(x)
#         x = self.activation(x)
#         x = x.transpose(1, 2)
#         x = self.norm(x)
#         return x
    
# class ConvUpsample(nn.Module):
#     def __init__(self, in_channels, out_channels, activation=nn.SiLU(), causal=False, norm_eps: float = 1e-2):
#         super().__init__()
#         self.causal = causal
#         if self.causal:
#             self.conv1 = CausalConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=2)
#             self.conv2 = CausalConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2)
#         else:
#             self.conv1 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
#             self.conv2 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.activation = activation
#         self.norm = RMSNorm(out_channels, eps=norm_eps)
    
#     def forward(self, x):
#         x = x.transpose(1, 2)
#         x = self.conv1(x)
#         x = self.activation(x)
#         x = self.conv2(x)
#         x = self.activation(x)
#         x = x.transpose(1, 2)
#         x = self.norm(x)
#         return x
        
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=nn.SiLU()):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=stride // 2 + stride % 2, output_padding=stride % 2)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
        self.activation = activation

    def forward(self, x):
        # x: (B, T, C)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        return x
    
class Upsample1D(nn.Module):
    def __init__(self, scale_factor=2.0, mode='nearest'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        x = self.upsample(x)
        return x

class ConvFeatureEncoder(nn.Module):
    """
    Wav2Vec2-style 1D convolutional feature encoder.

    Args:
        in_channels: input channels, 1 for raw waveform
        conv_dim: output channels for each layer (int or list)
        conv_kernel: kernel sizes
        conv_stride: strides
        activation: activation module class, default nn.SiLU
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_dim: Union[int, Sequence[int]] = 256,
        conv_kernel: Sequence[int] = (10, 3, 3, 3, 3, 2),
        conv_stride: Sequence[int] = (5, 2, 2, 2, 2, 2),
        activation: nn.Module = nn.SiLU,
        norm_eps: float = 1e-2,
    ):
        super().__init__()
        assert len(conv_kernel) == len(conv_stride)

        n_layers = len(conv_kernel)

        if isinstance(conv_dim, int):
            conv_dims = [conv_dim] * n_layers
        else:
            assert len(conv_dim) == n_layers
            conv_dims = list(conv_dim)

        self.activation = activation()

        layers = []
        norms = []
        prev_c = in_channels

        for c, k, s in zip(conv_dims, conv_kernel, conv_stride):
            layers.append(nn.Conv1d(prev_c, c, kernel_size=k, stride=s, bias=False, padding=(k-1)//2))
            norms.append(RMSNorm(c, eps=norm_eps))
            prev_c = c

        self.convs = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)

        # compute receptive field and stride
        self.total_stride = 1
        self.receptive_field = 1
        for k, s in zip(conv_kernel, conv_stride):
            self.receptive_field += (k - 1) * self.total_stride
            self.total_stride *= s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T) or (N, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = norm(x.transpose(1, 2)).transpose(1, 2)
            x = self.activation(x)

        return x.transpose(1, 2) # (N, C, T) -> (N, T, C)