import torch.nn as nn
from einops import rearrange
from . import activations
from .alias_free_torch import *
from torch.nn.utils import weight_norm
from torch import Tensor
import torch.nn.functional as F
import torch

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
    
class ConformerEncoderLayer(nn.Module):
    def __init__(self, dim: int = 16, 
                 stride: int = 1, 
                 causal: bool = False, 
                 antialias: bool = False):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, eps=1e-5)
        self.pointwise_conv1 = nn.Conv1d(
            dim,
            2 * dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.glu = nn.GLU(dim=1)
        self.conv = nn.Conv1d(
            dim,
            dim,
            3,
            stride=1,
            padding=0,
            groups=dim,
            bias=False,
        )

        self.conv_layer_norm = nn.LayerNorm(dim, eps=1e-5)
        self.activation = activations.SnakeBeta(dim, alpha_logscale=True)
        self.pointwise_conv2 = nn.Conv1d(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.conv_layer_norm(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x

class Wav2Vec2BertConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(self, config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=0,
            groups=config.hidden_size,
            bias=False,
        )

        self.depthwise_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.activation = ACT2FN[config.hidden_act]
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(config.conformer_conv_dropout)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.layer_norm(hidden_states)

        # Ensure that we do not leak padded positions in depthwise convolution if attention mask is passed.
        # Put 0 where necessary
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        # exchange the temporal dimension and the feature dimension
        hidden_states = hidden_states.transpose(1, 2)

        # GLU mechanism
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # Pad the sequence entirely on the left because of causal convolution.
        hidden_states = torch.nn.functional.pad(hidden_states, (self.depthwise_conv.kernel_size[0] - 1, 0))

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)

        hidden_states = self.depthwise_layer_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

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