import torch.nn as nn
from einops import rearrange
from . import activations
from .alias_free_torch import *
from torch.nn.utils import weight_norm
from torch import Tensor
import torch.nn.functional as F
import torch
from typing import Tuple

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

# Conformer Block components

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

import torch
import math

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
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
        except ImportError:
            print("FlashAttention not found, using manual attention")
            self.flash_attn_func = None

    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        cos, sin = self.rotary_emb(v, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)

        if self.flash_attn_func is not None:
            # flash_attn_func expects (B, T, n_head, head_dim)
            out = self.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=self.dropout if self.training else 0.0, causal=self.causal)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            if self.causal:
                mask = torch.ones(T, T, device=q.device, dtype=torch.bool).triu(diagonal=1)
                scores = scores.masked_fill(mask, float('-inf'))

            scores = F.softmax(scores, dim=-1)
            scores = F.dropout(scores, self.dropout, self.training)

            out = torch.matmul(scores, v) # out shape (B, n_head, T, head_dim)
            out = out.transpose(1, 2) # -> (B, T, n_head, head_dim)

        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        
        return out

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
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        if causal:
            self.depthwise_conv = CausalConv1d(dim, dim, kernel_size=kernel_size, groups=dim)
        else:
            self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same')
        self.conv_norm = RMSNorm(dim)
        self.silu = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.pointwise_conv1(x)
        out = self.glu(out)
        out = self.depthwise_conv(out)
        out = self.conv_norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.silu(out)
        out = self.pointwise_conv2(out)
        out = self.dropout(out)
        return out

class ConformerLayer(nn.Module):
    def __init__(self, dim, n_head=8, ffn_mult=4, conv_kernel_size=31, dropout=0.1, max_position_embeddings=2048, original_max_position_embeddings=4096, base=10000, conv_first: bool = False, causal: bool = False):
        super().__init__()
        self.ffn1 = FeedForward(dim, mult=ffn_mult, dropout=dropout)
        self.self_attn = SelfAttention(dim, n_head=n_head, dropout=dropout, max_position_embeddings=max_position_embeddings, original_max_position_embeddings=original_max_position_embeddings, base=base, causal=causal)
        self.conv = ConformerConvModule(dim, kernel_size=conv_kernel_size, dropout=dropout, causal=causal)
        self.ffn2 = FeedForward(dim, mult=ffn_mult, dropout=dropout)
        self.conv_first = conv_first

        self.conv_norm_in = RMSNorm(dim)
        self.ffn1_norm_in = RMSNorm(dim)
        self.attn_norm_in = RMSNorm(dim)
        self.ffn2_norm_in = RMSNorm(dim)
        self.final_norm = RMSNorm(dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.conv_first:
            x = x + self.conv(self.conv_norm_in(x.transpose(1, 2)).transpose(1, 2))
        else:
            x = x + self.dropout(self.self_attn(self.attn_norm_in(x.transpose(1, 2))).transpose(1, 2))

        x = x + self.ffn1(self.ffn1_norm_in(x.transpose(1, 2))).transpose(1, 2)

        if self.conv_first:
            x = x + self.dropout(self.self_attn(self.attn_norm_in(x.transpose(1, 2))).transpose(1, 2))
        else:
            x = x + self.conv(self.conv_norm_in(x.transpose(1, 2)).transpose(1, 2))

        x = x + self.ffn2(self.ffn2_norm_in(x.transpose(1, 2))).transpose(1, 2)
        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
        # x = x + self.dropout(self.self_attn(self.attn_norm_in(x.transpose(1, 2)).transpose(1, 2)))

        # x = x + self.ffn1(self.ffn1_norm_in(x.transpose(1, 2))).transpose(1, 2)

        return x

class ConformerBackbone(nn.Module):
    def __init__(self, dim, n_layers, n_head=8, ffn_mult=4, conv_kernel_size=31, dropout=0.1, max_position_embeddings=2048, original_max_position_embeddings=4096, base=10000.0, conv_first: bool = False, causal: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerLayer(dim, n_head, ffn_mult, conv_kernel_size, dropout, max_position_embeddings=max_position_embeddings, original_max_position_embeddings=original_max_position_embeddings, base=base, conv_first=conv_first, causal=causal)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=nn.SiLU()):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=stride // 2 + stride % 2)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=nn.SiLU()):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=stride // 2 + stride % 2, output_padding=stride % 2)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        return x
    
class Upsample_Interpolate(nn.Module):
    def __init__(self, scale_factor=2.0, mode='nearest'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        x = self.upsample(x)
        return x