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
def rmsnorm(x, eps):
    def _norm(y):
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)

    return _norm(x.float()).type_as(x)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rmsnorm(x, self.eps) * self.weight
    
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    shape = [1] * ndim
    shape[1] = x.shape[1]
    shape[-1] = x.shape[-1]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SelfAttention(nn.Module):
    def __init__(self, dim, n_head=8, dropout=0.1, causal: bool = False):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.causal = causal
        
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
        except ImportError:
            print("FlashAttention not found, using manual attention")
            self.flash_attn_func = None

    def forward(self, x, freqs_cis):
        B, C, T = x.shape
        x = x.transpose(1, 2)
        
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = rmsnorm(q, 1e-6)
        k = rmsnorm(k, 1e-6)
        
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        if self.flash_attn_func is not None:
            # flash_attn_func expects (B, T, n_head, head_dim)
            out = self.flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=self.causal)
        else:
            # Fallback to manual attention. Transpose to (B, n_head, T, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            if self.causal:
                mask = torch.ones(T, T, device=q.device, dtype=torch.bool).triu(diagonal=1)
                scores = scores.masked_fill(mask, float('-inf'))

            scores = F.softmax(scores, dim=-1, dtype=torch.float32)
            scores = F.dropout(scores, self.dropout, self.training)

            out = torch.matmul(scores, v).to(q.dtype) # out shape (B, n_head, T, head_dim)
            out = out.transpose(1, 2) # -> (B, T, n_head, head_dim)

        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        out = out.transpose(1, 2)
        
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
    def __init__(self, dim, n_head=8, ffn_mult=4, conv_kernel_size=31, dropout=0.1, conv_first: bool = False, causal: bool = False):
        super().__init__()
        self.ffn1 = FeedForward(dim, mult=ffn_mult, dropout=dropout)
        self.self_attn = SelfAttention(dim, n_head=n_head, dropout=dropout, causal=causal)
        self.conv = ConformerConvModule(dim, kernel_size=conv_kernel_size, dropout=dropout, causal=causal)
        self.ffn2 = FeedForward(dim, mult=ffn_mult, dropout=dropout)
        self.conv_first = conv_first

        self.conv_norm_in = RMSNorm(dim)
        self.ffn1_norm_in = RMSNorm(dim)
        self.attn_norm_in = RMSNorm(dim)
        self.ffn2_norm_in = RMSNorm(dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cis):
        if self.conv_first:
            x = x + self.conv(self.conv_norm_in(x.transpose(1, 2)).transpose(1, 2))
        else:
            x = x + self.dropout(self.self_attn(self.attn_norm_in(x.transpose(1, 2)).transpose(1, 2), freqs_cis))

        x = x + self.ffn1(self.ffn1_norm_in(x.transpose(1, 2))).transpose(1, 2)

        if self.conv_first:
            x = x + self.dropout(self.self_attn(self.attn_norm_in(x.transpose(1, 2)).transpose(1, 2), freqs_cis))
        else:
            x = x + self.conv(self.conv_norm_in(x.transpose(1, 2)).transpose(1, 2))

        x = x + self.ffn2(self.ffn2_norm_in(x.transpose(1, 2))).transpose(1, 2)
        return x

class ConformerBackbone(nn.Module):
    def __init__(self, dim, n_layers, n_head=8, ffn_mult=4, conv_kernel_size=31, dropout=0.1, max_seq_len=8192, rope_theta=10000.0, conv_first: bool = False, causal: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerLayer(dim, n_head, ffn_mult, conv_kernel_size, dropout, conv_first=conv_first, causal=causal)
            for _ in range(n_layers)
        ])
        self.freqs_cis = precompute_freqs_cis(
            dim // n_head,
            max_seq_len,
            rope_theta,
        )

    def forward(self, x):
        self.freqs_cis = self.freqs_cis.to(x.device)
        freqs_cis = self.freqs_cis[:x.shape[-1]]

        for layer in self.layers:
            x = layer(x, freqs_cis)
        return x