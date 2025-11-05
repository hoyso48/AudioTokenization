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

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x, dtype = x.float(), x.dtype
        output = self._norm(x)
        return (output * self.weight).to(dtype)


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
    def __init__(self, dim, n_head=8, dropout=0.1, max_position_embeddings=2048, original_max_position_embeddings=4096, base=10000, causal: bool = False):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.causal = causal
        
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        # self.q_norm = RMSNorm(self.head_dim)
        # self.k_norm = RMSNorm(self.head_dim)
        self.dropout = dropout
        self.rotary_emb = LlamaDynamicYaRNScaledRotaryEmbedding(self.head_dim, 
                                        max_position_embeddings=max_position_embeddings, 
                                        base=base, 
                                        device=None, 
                                        original_max_position_embeddings=original_max_position_embeddings) #1
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
                    start_offsets = torch.repeat_interleave(cu_seqlens[:-1].to(torch.long), lengths)
                    token_idx = torch.arange(total, device=qkv.device, dtype=torch.long)
                    position_ids = token_idx - start_offsets  # [total]
                    q, k = qkv[:, 0], qkv[:, 1]
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=1)
                    qkv = torch.stack((q, k, qkv[:, 2]), dim=1)

            # FlashAttention varlen qkvpacked
            out = self.flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=None,
                causal=self.causal,
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
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, T)
        cos_bt = cos[position_ids]
        sin_bt = sin[position_ids]
        cos_bt = cos_bt.unsqueeze(2).to(dtype=qkv.dtype)
        sin_bt = sin_bt.unsqueeze(2).to(dtype=qkv.dtype)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        q = (q * cos_bt) + (rotate_half(q) * sin_bt)
        k = (k * cos_bt) + (rotate_half(k) * sin_bt)
        qkv = torch.stack((q, k, qkv[:, :, 2]), dim=2)

        # FlashAttention qkvpacked (dense)
        out = self.flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=None,
            causal=self.causal,
        )  # [B, T, n_head, head_dim]

        out = out.reshape(B, T, C)
        out = self.out_proj(out)

        return out

class LayerScale(nn.Module):
    def __init__(self, d_model: int, gamma_init: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model) * gamma_init)

    def forward(self, x):
        scale = self.scale.view(1, 1, -1)
        return x * scale

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
    def __init__(self, dim, kernel_size=31, dropout=0.1, causal: bool = False):
        super().__init__()
        self.pointwise_conv1 = nn.Linear(dim, 2 * dim) #nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=-1)
        if causal:
            self.depthwise_conv = CausalConv1d(dim, dim, kernel_size=kernel_size, groups=dim)
        else:
            self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same')
        self.conv_norm = RMSNorm(dim)
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
    def __init__(self, dim, n_head=8, ffn_mult=4, dropout=0.1, max_position_embeddings=2048, original_max_position_embeddings=4096, base=10000, causal: bool = False):
        super().__init__()
        self.ffn1 = FeedForward(dim, mult=ffn_mult, dropout=dropout)
        self.self_attn = SelfAttention(dim, n_head=n_head, dropout=dropout, max_position_embeddings=max_position_embeddings, original_max_position_embeddings=original_max_position_embeddings, base=base, causal=causal)

        self.ffn1_norm_in = RMSNorm(dim)
        self.attn_norm_in = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, position_ids=None, cu_seqlens: torch.Tensor = None, max_seqlen: int = None):
        x = x + self.dropout(self.self_attn(self.attn_norm_in(x), position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen))

        x = x + self.ffn1(self.ffn1_norm_in(x))

        return x

class Transformer(nn.Module):
    def __init__(self, dim, n_layers, n_head=8, ffn_mult=4, dropout=0.1, max_position_embeddings=2048, original_max_position_embeddings=4096, base=10000.0, causal: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(dim, n_head, ffn_mult, dropout, max_position_embeddings=max_position_embeddings, original_max_position_embeddings=original_max_position_embeddings, base=base, causal=causal)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x, position_ids=None, cu_seqlens: torch.Tensor = None, max_seqlen: int = None):
        for layer in self.layers:
            x = layer(x, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = self.norm(x)
        return x

class Patchify(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2) #(B, H, W, C) -> (B, C, H, W)
        x = self.conv(x) #(B, C, H//patch_size, W//patch_size)
        x = x.permute(0, 2, 3, 1) #(B, H//patch_size, W//patch_size, C)
        return x

class UnPatchify(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2) #(B, C, H, W) -> (B, C, H, W)
        x = self.conv(x) #(B, C, H*patch_size, W*patch_size)
        x = x.permute(0, 2, 3, 1) #(B, H*patch_size, W*patch_size, C)
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

# class ConvDownsample(nn.Module):
#     def __init__(self, in_channels, out_channels, activation=nn.SiLU(), causal=False):
#         super().__init__()
#         self.causal = causal
#         if causal:
#             self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#             self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
#         else:
#             self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#             self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
#         self.activation = activation
#         self.norm = nn.LayerNorm(out_channels)
    
#     def forward(self, x):
#         x = x.transpose(1, 2)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.activation(x)
#         x = x.transpose(1, 2)
#         x = self.norm(x)
#         return x
    
# class ConvUpsample(nn.Module):
#     def __init__(self, in_channels, out_channels, activation=nn.SiLU(), causal=False):
#         super().__init__()
#         self.causal = causal
#         if self.causal:
#             self.conv1 = CausalConv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
#             self.conv2 = CausalConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2)
#         else:
#             self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
#             self.conv2 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.activation = activation
#         self.norm = nn.LayerNorm(out_channels)
    
#     def forward(self, x):
#         x = x.transpose(1, 2)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.activation(x)
#         x = x.transpose(1, 2)
#         x = self.norm(x)
#         return x
        
class ConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU(), causal=False):
        super().__init__()
        self.causal = causal
        if causal:
            self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.activation = activation
        self.norm = RMSNorm(out_channels)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x
    
class ConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU(), causal=False):
        super().__init__()
        self.causal = causal
        if self.causal:
            self.conv1 = CausalConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=2)
            self.conv2 = CausalConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2)
        else:
            self.conv1 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv2 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.activation = activation
        self.norm = RMSNorm(out_channels)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x
        
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
            norms.append(RMSNorm(c))
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