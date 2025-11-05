import torch.nn as nn
from einops import rearrange
from . import activations
from .alias_free_torch import *
from torch.nn.utils import weight_norm
from torch import Tensor
import torch.nn.functional as F
import torch
from typing import Tuple, Sequence, Union

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

class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False, antialias: bool = False):
        super().__init__()
        if causal:
            pad = 0
        else:
            pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True), antialias=antialias),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad, causal=causal),
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True), antialias=antialias),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, dilations = (1, 3, 9), causal: bool = False, antialias: bool = False):
        super().__init__()
        runits = [ResidualUnit(dim // 2, dilation=d, causal=causal, antialias=antialias) for d in dilations]
        if causal:
            pad = 0
        else:
            pad = stride // 2 + stride % 2 if stride != 1 else 0
        self.block = nn.Sequential(
            *runits,
            Activation1d(activation=activations.SnakeBeta(dim//2, alpha_logscale=True), antialias=antialias),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride if stride != 1 else 1,
                stride=stride,
                padding=pad,
                causal=causal
            ),
        )

    def forward(self, x):
        return self.block(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, dilations = (1, 3, 9), causal: bool = False, antialias: bool = False):
        super().__init__()
        
        if causal:
            tconv_kwargs = {}
        else:
            tconv_kwargs = {
                "padding": stride // 2 + stride % 2 if stride != 1 else 0,
                "output_padding": stride % 2 if stride != 1 else 0
            }

        self.block = nn.Sequential(
            Activation1d(activation=activations.SnakeBeta(input_dim, alpha_logscale=True), antialias=antialias),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride if stride != 1 else 1,
                stride=stride,
                causal=causal,
                **tconv_kwargs,
            )
        )
        self.block.extend([ResidualUnit(output_dim, dilation=d, causal=causal, antialias=antialias) for d in dilations])

    def forward(self, x):
        return self.block(x)
    
class ResLSTM(nn.Module):
    def __init__(self, dimension: int,
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension if not bidirectional else dimension // 2,
                            num_layers, batch_first=True,
                            bidirectional=bidirectional)

    def forward(self, x):
        """
        Args:
            x: [B, F, T]

        Returns:
            y: [B, F, T]
        """
        x = rearrange(x, "b f t -> b t f")
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = rearrange(y, "b t f -> b f t")
        return y

class ECA(nn.Module):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding="same", bias=False)

    def forward(self, inputs):
        x = inputs.mean(2)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(1)
        x = torch.sigmoid(x)
        x = x.unsqueeze(-1)
        return inputs * x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x, mask=None):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class ScaleBiasLayer(nn.Module):
    """
    Computes an affine transformation y = x * scale + bias, either learned via adaptive weights, or fixed.
    Efficient alternative to LayerNorm where we can avoid computing the mean and variance of the input, and
    just rescale the output of the previous layer.

    Args:
        d_model (int): input dimension of layer.
        adaptive_scale (bool): whether to learn the affine transformation parameters or not. If set to False,
            the scale is fixed to 1 and bias to 0, effectively performing a No-Op on the input.
            This is done for export compatibility.
    """

    def __init__(self, d_model: int, adaptive_scale: bool):
        super().__init__()
        self.adaptive_scale = adaptive_scale
        if adaptive_scale:
            self.scale = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer('scale', torch.ones(d_model), persistent=True)
            self.register_buffer('bias', torch.zeros(d_model), persistent=True)

    def forward(self, x):
        scale = self.scale.view(1, 1, -1)
        bias = self.bias.view(1, 1, -1)
        return x * scale + bias

class SemanticEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        code_dim: int,
        encode_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super(SemanticEncoder, self).__init__()

        # 初始卷积，将 input_channels 映射到 encode_channels
        self.initial_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=encode_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        # 残差块
        self.residual_blocks = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(
                encode_channels,
                encode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                encode_channels,
                encode_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias
            )
        )

        # 最终卷积，将 encode_channels 映射到 code_dim
        self.final_conv = nn.Conv1d(
            in_channels=encode_channels,
            out_channels=code_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

    def forward(self, x):
        """
        前向传播方法。

        Args:
            x (Tensor): 输入张量，形状为 (Batch, Input_channels, Length)

        Returns:
            Tensor: 编码后的张量，形状为 (Batch, Code_dim, Length)
        """
        x = self.initial_conv(x)           # (Batch, Encode_channels, Length)
        x = self.residual_blocks(x) + x   # 残差连接
        x = self.final_conv(x)             # (Batch, Code_dim, Length)
        return x

class SemanticDecoder(nn.Module):
    def __init__(
        self,
        code_dim: int,
        output_channels: int,
        decode_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super(SemanticDecoder, self).__init__()
        
        # Initial convolution to map code_dim to decode_channels
        self.initial_conv = nn.Conv1d(
            in_channels=code_dim,
            out_channels=decode_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        
        # Residual Blocks
        self.residual_blocks = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(decode_channels, decode_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv1d(decode_channels, decode_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias)
        )
        
        # Final convolution to map decode_channels to output_channels
        self.final_conv = nn.Conv1d(
            in_channels=decode_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        
    def forward(self, z):
        # z: (Batch, Code_dim, Length)
        x = self.initial_conv(z)  # (Batch, Decode_channels, Length)
        x = self.residual_blocks(x) + x  # Residual connection
        x = self.final_conv(x)  # (Batch, Output_channels, Length)
        return x

# Conformer Block components

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

import torch
import math

# # Inverse dim formula to find dim based on number of rotations
# def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
#     return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

# # Find dim range bounds based on rotations
# def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
#     low = math.floor(find_correction_dim(
#         low_rot, dim, base, max_position_embeddings))
#     high = math.ceil(find_correction_dim(
#         high_rot, dim, base, max_position_embeddings))
#     return max(low, 0), min(high, dim-1)  # Clamp values just in case

# def linear_ramp_mask(min, max, dim):
#     if min == max:
#         max += 0.001  # Prevent singularity

#     linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
#     ramp_func = torch.clamp(linear_func, 0, 1)
#     return ramp_func

# def get_mscale(scale=1):
#     if scale <= 1:
#         return 1.0
#     return 0.1 * math.log(scale) + 1.0

# class LlamaDynamicYaRNScaledRotaryEmbedding(torch.nn.Module):
#     def __init__(self, dim, max_position_embeddings=2048, base=10000, original_max_position_embeddings=2048, extrapolation_factor=1, attn_factor=1, beta_fast=32, beta_slow=1, finetuned=False, device=None):
#         super().__init__()

#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
#         self.original_max_position_embeddings = original_max_position_embeddings
#         self.extrapolation_factor = extrapolation_factor
#         self.attn_factor = attn_factor
#         self.beta_fast = beta_fast
#         self.beta_slow = beta_slow

#         if finetuned:
#             self.yarn(self.max_position_embeddings / self.original_max_position_embeddings, device)
#         else:
#             inv_freq = 1.0 / \
#                 (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
#             self.register_buffer("inv_freq", inv_freq)
#             self.mscale = 1

#         # Build here to make `torch.jit.trace` work.
#         self.max_seq_len_cached = max_position_embeddings
#         t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
#         freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#         # Different from paper, but it uses a different permutation in order to obtain the same calculation
#         emb = torch.cat((freqs, freqs), dim=-1)
#         dtype = torch.get_default_dtype()

#         self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(dtype), persistent=False)
#         self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(dtype), persistent=False)

#     def forward(self, x, seq_len=None):
#         # x: [bs, num_attention_heads, seq_len, head_size]
#         # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
#         if seq_len > self.max_seq_len_cached:
#             self.max_seq_len_cached = seq_len

#             self.yarn(seq_len / self.original_max_position_embeddings, x.device)

#             t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
#             freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#             # Different from paper, but it uses a different permutation in order to obtain the same calculation
#             emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

#             self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(x.dtype), persistent=False)
#             self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(x.dtype), persistent=False)
#         return (
#             self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
#             self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
#         )

#     def yarn(self, scale, device):
#         pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
#         inv_freq_extrapolation = 1.0 / pos_freqs
#         inv_freq_interpolation = 1.0 / (scale * pos_freqs)

#         low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
#         inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim // 2).float().to(device)) * self.extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
#         inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

#         self.register_buffer("inv_freq", inv_freq)
#         self.mscale = float(get_mscale(scale) * self.attn_factor) # Get n-d magnitude scaling corrected for interpolation
# Inverse dim formula to find dim based on number of rotations

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

# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     t = torch.arange(end, device=freqs.device, dtype=torch.float32)
#     freqs = torch.outer(t, freqs)
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
#     return freqs_cis

# def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
#     ndim = x.ndim
#     assert 0 <= 1 < ndim
#     shape = [1] * ndim
#     shape[1] = x.shape[1]
#     shape[-1] = x.shape[-1]
#     return freqs_cis.view(*shape)

# def apply_rotary_emb(
#     xq: torch.Tensor,
#     xk: torch.Tensor,
#     freqs_cis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
#     freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
#     return xq_out.type_as(xq), xk_out.type_as(xk)

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
        q = q.transpose(1, 2) #self.q_norm(q).transpose(1, 2)
        k = k.transpose(1, 2) #self.k_norm(k).transpose(1, 2)
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

class ConformerLayer(nn.Module):
    def __init__(self, dim, n_head=8, ffn_mult=4, conv_kernel_size=31, dropout=0.1, max_position_embeddings=2048, original_max_position_embeddings=4096, base=10000, conv_first: bool = False, causal: bool = False):
        super().__init__()
        self.ffn1 = FeedForward(dim, mult=ffn_mult, dropout=dropout)
        self.self_attn = SelfAttention(dim, n_head=n_head, dropout=dropout, max_position_embeddings=max_position_embeddings, original_max_position_embeddings=original_max_position_embeddings, base=base, causal=causal)
        # self.conv = ConformerConvModule(dim, kernel_size=conv_kernel_size, dropout=dropout, causal=causal)
        # self.ffn2 = FeedForward(dim, mult=ffn_mult, dropout=dropout)
        self.conv_first = conv_first

        # self.ffn1_norm_in = RMSNorm(dim)
        # self.attn_norm_in = RMSNorm(dim)

        # self.conv_norm_in = RMSNorm(dim)
        self.ffn1_norm_in = RMSNorm(dim)
        self.attn_norm_in = RMSNorm(dim)
        # self.ffn2_norm_in = RMSNorm(dim)
        # self.final_norm = RMSNorm(dim)
        # self.attn_scale = LayerScale(dim, gamma_init=1e-0)
        # self.ffn1_scale = LayerScale(dim, gamma_init=1e-0)

        # self.conv_norm_out = RMSNorm(dim)
        # self.ffn1_norm_out = RMSNorm(dim)
        # self.attn_norm_out = RMSNorm(dim)
        # self.ffn2_norm_out = RMSNorm(dim)
        # self.conv_scale = LayerScale(dim)
        # self.ffn1_scale = LayerScale(dim)
        # self.attn_scale = LayerScale(dim)
        # self.ffn2_scale = LayerScale(dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # if self.conv_first:
        #     x = x + self.conv(self.conv_norm_in(x))
        # else:
        #     x = x + self.dropout(self.self_attn(self.attn_norm_in(x)))

        # x = x + self.ffn1(self.ffn1_norm_in(x))

        # if self.conv_first:
        #     x = x + self.dropout(self.self_attn(self.attn_norm_in(x)))
        # else:
        #     x = x + self.conv(self.conv_norm_in(x))

        # x = x + self.ffn2(self.ffn2_norm_in(x))
        # x = self.final_norm(x)

        # if self.conv_first:
        #     x = x + self.conv_scale(self.conv(x))
        #     x = self.conv_norm_out(x)
        # else:
        #     x = x + self.attn_scale(self.dropout(self.self_attn(x)))
        #     x = self.attn_norm_out(x)

        # x = x + self.ffn1_scale(self.ffn1(x))
        # x = self.ffn1_norm_out(x)

        # if self.conv_first:
        #     x = x + self.attn_scale(self.dropout(self.self_attn(x)))
        #     x = self.attn_norm_out(x)
        # else:
        #     x = x + self.conv_scale(self.conv(x))
        #     x = self.conv_norm_out(x)

        # x = x + self.ffn2_scale(self.ffn2(x))
        # x = self.ffn2_norm_out(x)

        x = x + self.dropout(self.self_attn(self.attn_norm_in(x)))

        x = x + self.ffn1(self.ffn1_norm_in(x))

        return x

class ConformerBackbone(nn.Module):
    def __init__(self, dim, n_layers, n_head=8, ffn_mult=4, conv_kernel_size=31, dropout=0.1, max_position_embeddings=2048, original_max_position_embeddings=4096, base=10000.0, conv_first: bool = False, causal: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerLayer(dim, n_head, ffn_mult, conv_kernel_size, dropout, max_position_embeddings=max_position_embeddings, original_max_position_embeddings=original_max_position_embeddings, base=base, conv_first=conv_first, causal=causal)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
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