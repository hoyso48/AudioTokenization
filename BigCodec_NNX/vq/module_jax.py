import math
from typing import Callable, Optional, Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax as lax
from flax import nnx
try:
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as tpu_flash_attention
except Exception:
    tpu_flash_attention = None

class CausalConv1d(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: Optional[Any] = None,
        param_dtype: Any = jnp.float32,
        precision: Any = lax.Precision.DEFAULT,
        kernel_init: Callable = None,
        bias_init: Callable = None,
        rngs: nnx.Rngs = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.use_bias = bias

        # Conv without built-in padding; we'll do manual causal pad (left side)
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            kernel_dilation=dilation,
            feature_group_count=groups,
            padding=((0, 0),),  # no automatic padding
            precision=self.precision,
            rngs=rngs,
        )

        # compute amount of left padding to achieve causal convolution
        self._left_pad = max((kernel_size - 1) * dilation, 0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (N, L, C) channels-last
        pad = ((0, 0), (self._left_pad, 0), (0, 0))
        x = jnp.pad(x, pad, mode='constant')
        return self.conv(x)


class CausalConvTranspose1d(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        dtype: Optional[Any] = None,
        param_dtype: Any = jnp.float32,
        precision: Any = lax.Precision.DEFAULT,
        kernel_init: Callable = None,
        bias_init: Callable = None,
        rngs: nnx.Rngs = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.use_bias = bias
        # Mirror PyTorch ConvTranspose1d default padding=0
        self.deconv = nnx.ConvTranspose(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding='VALID',
            transpose_kernel=True,
            precision=self.precision,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (N, L, C)
        y = self.deconv(x)
        # Torch impl slices last self.stride elements: y[..., :-self.stride]
        if self.stride > 0:
            y = y[:, : (y.shape[1] - self.stride), :]
        return y


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, eps: float = 1e-6, rngs: nnx.Rngs = None):
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., dim)
        x_f = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f * x_f, axis=-1, keepdims=True) + self.eps)
        y = (x_f / rms).astype(x.dtype)
        return y * self.weight


class LayerScale(nnx.Module):
    def __init__(self, dim: int, rngs: nnx.Rngs = None):
        self.scale = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * self.scale.reshape((1, 1, -1))


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nnx.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        self.inv_freq = nnx.Variable(inv_freq)
        # build cache
        t = jnp.arange(max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.einsum('i,j->ij', t, self.inv_freq.value)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = nnx.Variable(jnp.cos(emb))
        self.sin_cached = nnx.Variable(jnp.sin(emb))

    def __call__(self, seq_len: int, dtype=jnp.float32):
        return self.cos_cached.value[:seq_len].astype(dtype), self.sin_cached.value[:seq_len].astype(dtype)


class LlamaDynamicYaRNScaledRotaryEmbedding(nnx.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        original_max_position_embeddings: int = 2048,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        finetuned: bool = False,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # init inv_freq and mscale
        if finetuned:
            self._yarn(scale=max_position_embeddings / original_max_position_embeddings)
        else:
            inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
            self.inv_freq = nnx.Variable(inv_freq)
            self.mscale = 1.0

        # build caches
        self.max_seq_len_cached = max_position_embeddings
        t = jnp.arange(self.max_seq_len_cached, dtype=jnp.float32)
        freqs = jnp.einsum('i,j->ij', t, self.inv_freq.value)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = nnx.Variable(jnp.cos(emb) * self.mscale)
        self.sin_cached = nnx.Variable(jnp.sin(emb) * self.mscale)

    def _yarn(self, scale: float):
        pos_freqs = self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale * pos_freqs)

        def _find_correction_dim(num_rotations, dim, base, max_pos):
            return (dim * jnp.log(max_pos / (num_rotations * 2 * jnp.pi))) / (2 * jnp.log(base))

        def _find_correction_range(low_rot, high_rot, dim, base, max_pos):
            low = jnp.floor(_find_correction_dim(low_rot, dim, base, max_pos))
            high = jnp.ceil(_find_correction_dim(high_rot, dim, base, max_pos))
            low = jnp.maximum(low, 0)
            high = jnp.minimum(high, dim - 1)
            return low.astype(jnp.int32), high.astype(jnp.int32)

        def _linear_ramp_mask(mi, ma, dim):
            mi_f = jnp.where(mi == ma, mi + 1e-3, mi)
            lin = (jnp.arange(dim, dtype=jnp.float32) - mi_f) / (ma - mi_f)
            return jnp.clip(lin, 0.0, 1.0)

        low, high = _find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        mask = (1.0 - _linear_ramp_mask(low, high, self.dim // 2)) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (1.0 - mask) + inv_freq_extrapolation * mask
        self.inv_freq = nnx.Variable(inv_freq)

        def _get_mscale(scale=1.0):
            return 1.0 if scale <= 1.0 else (0.1 * jnp.log(scale) + 1.0)

        self.mscale = float(_get_mscale(scale) * self.attn_factor)

    def __call__(self, seq_len: int, dtype=jnp.float32):
        # extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = jnp.arange(self.max_seq_len_cached, dtype=jnp.float32)
            freqs = jnp.einsum('i,j->ij', t, self.inv_freq.value)
            emb = jnp.concatenate([freqs, freqs], axis=-1)
            self.cos_cached.value = jnp.cos(emb) * self.mscale
            self.sin_cached.value = jnp.sin(emb) * self.mscale
        return self.cos_cached.value[:seq_len].astype(dtype), self.sin_cached.value[:seq_len].astype(dtype)


class SelfAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        n_head: int = 8,
        dropout: float = 0.0,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        causal: bool = False,
        precision: Any = lax.Precision.DEFAULT,
        rngs: nnx.Rngs = None,
        original_max_position_embeddings: int = 4096,
        use_flash_attention: bool = False,
    ):
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.causal = causal
        self.precision = precision
        self.qkv_proj = nnx.Linear(dim, 3 * dim, use_bias=False, precision=self.precision, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, use_bias=False, precision=self.precision, rngs=rngs)
        self.dropout_p = dropout
        self.attn_dropout = nnx.Dropout(dropout, rngs=rngs)
        # Use YaRN-scaled rotary embedding to match PyTorch reference
        self.rotary_emb = LlamaDynamicYaRNScaledRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            original_max_position_embeddings=original_max_position_embeddings,
        )
        self.use_flash_attention = use_flash_attention

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, T, C)
        b, t, c = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(b, t, 3, self.n_head, self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.squeeze(2).transpose(0, 2, 1, 3)  # (B, H, T, D)
        k = k.squeeze(2).transpose(0, 2, 1, 3)
        v = v.squeeze(2).transpose(0, 2, 1, 3)

        cos, sin = self.rotary_emb(t, dtype=x.dtype)
        # Broadcast to (1,1,T,D)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        use_tpu_flash = (
            self.use_flash_attention
            and tpu_flash_attention is not None
            and (jax.default_backend() == 'tpu' or (len(jax.devices()) > 0 and jax.devices()[0].platform == 'tpu'))
        )
        if use_tpu_flash:
            sm_scale = 1.0 / math.sqrt(self.head_dim)
            try:
                out = tpu_flash_attention(q, k, v, ab=None, segment_ids=None, causal=self.causal, sm_scale=sm_scale)
            except Exception:
                attn_logits = jnp.einsum('bhtd,bhsd->bhts', q, k) / math.sqrt(self.head_dim)
                if self.causal:
                    mask = jnp.triu(jnp.ones((t, t), dtype=bool), k=1)
                    attn_logits = jnp.where(mask[None, None, :, :], -jnp.inf, attn_logits)
                attn = jax.nn.softmax(attn_logits, axis=-1)
                if self.dropout_p > 0.0:
                    attn = self.attn_dropout(attn)
                out = jnp.einsum('bhts,bhsd->bhtd', attn, v)
        else:
            attn_logits = jnp.einsum('bhtd,bhsd->bhts', q, k) / math.sqrt(self.head_dim)
            if self.causal:
                mask = jnp.triu(jnp.ones((t, t), dtype=bool), k=1)
                attn_logits = jnp.where(mask[None, None, :, :], -jnp.inf, attn_logits)
            attn = jax.nn.softmax(attn_logits, axis=-1)
            if self.dropout_p > 0.0:
                attn = self.attn_dropout(attn)
            out = jnp.einsum('bhts,bhsd->bhtd', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, c)
        if use_tpu_flash and self.dropout_p > 0.0:
            out = self.attn_dropout(out)
        out = self.out_proj(out)
        return out


class FeedForward(nnx.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0, precision: Any = lax.Precision.DEFAULT, rngs: nnx.Rngs = None):
        hidden_dim = int(2 * (dim * mult) / 3)
        multiple_of = 256
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.precision = precision
        self.w1 = nnx.Linear(dim, hidden_dim, use_bias=False, precision=self.precision, rngs=rngs)
        self.w2 = nnx.Linear(hidden_dim, dim, use_bias=False, precision=self.precision, rngs=rngs)
        self.w3 = nnx.Linear(dim, hidden_dim, use_bias=False, precision=self.precision, rngs=rngs)
        self.dropout_p = dropout
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))
        if self.dropout_p > 0.0:
            out = self.dropout(out)
        return out


def _glu(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    a, b = jnp.split(x, 2, axis=axis)
    return a * jax.nn.sigmoid(b)


class DepthwiseConv1d(nnx.Module):
    def __init__(self, channels: int, kernel_size: int, causal: bool = False, precision: Any = lax.Precision.DEFAULT, rngs: nnx.Rngs = None):
        self.channels = channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.precision = precision
        # weight: (H, in_ch/groups, out_ch) -> for depthwise, groups=channels, in_ch/groups=1, out_ch=channels
        k_shape = (kernel_size, 1, channels)
        self.kernel = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), k_shape))
        self.bias = nnx.Param(jnp.zeros((channels,), dtype=jnp.float32))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (N, L, C)
        if self.causal:
            pad_left = self.kernel_size - 1
            x = jnp.pad(x, ((0, 0), (pad_left, 0), (0, 0)))
        y = lax.conv_general_dilated(
            lhs=x,
            rhs=self.kernel.value,
            window_strides=(1,),
            padding='SAME' if not self.causal else 'VALID',
            dimension_numbers=('NHC', 'HIO', 'NHC'),
            feature_group_count=self.channels,
            precision=self.precision,
        )
        return y + self.bias.value.reshape((1, 1, -1))


class ConformerConvModule(nnx.Module):
    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.0, causal: bool = False, precision: Any = lax.Precision.DEFAULT, rngs: nnx.Rngs = None):
        self.precision = precision
        self.pointwise_conv1 = nnx.Conv(in_features=dim, out_features=2 * dim, kernel_size=1, precision=self.precision, rngs=rngs)
        self.glu = True
        self.depthwise_conv = DepthwiseConv1d(dim, kernel_size=kernel_size, causal=causal, precision=self.precision, rngs=rngs)
        self.conv_norm = RMSNorm(dim)
        self.silu = True
        self.pointwise_conv2 = nnx.Conv(in_features=dim, out_features=dim, kernel_size=1, precision=self.precision, rngs=rngs)
        self.dropout_p = dropout
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, T, C)
        out = self.pointwise_conv1(x)
        out = _glu(out, axis=-1)
        out = self.depthwise_conv(out)
        out = self.conv_norm(out)
        out = jax.nn.silu(out)
        out = self.pointwise_conv2(out)
        if self.dropout_p > 0.0:
            out = self.dropout(out)
        return out


class ConformerLayer(nnx.Module):
    def __init__(
        self,
        dim: int,
        n_head: int = 8,
        ffn_mult: float = 4.0,
        conv_kernel_size: int = 31,
        dropout: float = 0.0,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        conv_first: bool = False,
        causal: bool = False,
        precision: Any = lax.Precision.DEFAULT,
        rngs: nnx.Rngs = None,
        original_max_position_embeddings: int = 4096,
    ):
        self.ffn1 = FeedForward(dim, mult=ffn_mult, dropout=dropout, precision=precision, rngs=rngs)
        self.self_attn = SelfAttention(
            dim,
            n_head=n_head,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
            base=base,
            causal=causal,
            precision=precision,
            rngs=rngs,
            original_max_position_embeddings=original_max_position_embeddings,
        )
        self.conv = ConformerConvModule(dim, kernel_size=conv_kernel_size, dropout=dropout, causal=causal, precision=precision, rngs=rngs)
        self.ffn2 = FeedForward(dim, mult=ffn_mult, dropout=dropout, precision=precision, rngs=rngs)
        self.conv_first = conv_first

        # self.conv_norm_in = RMSNorm(dim)
        # self.ffn1_norm_in = RMSNorm(dim)
        # self.attn_norm_in = RMSNorm(dim)
        # self.ffn2_norm_in = RMSNorm(dim)
        # self.final_norm = RMSNorm(dim)

        self.conv_norm_out = RMSNorm(dim)
        self.ffn1_norm_out = RMSNorm(dim)
        self.attn_norm_out = RMSNorm(dim)
        self.ffn2_norm_out = RMSNorm(dim)
        self.conv_scale = LayerScale(dim)
        self.ffn1_scale = LayerScale(dim)
        self.attn_scale = LayerScale(dim)
        self.ffn2_scale = LayerScale(dim)

        self.dropout_p = dropout
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, T, C)
        # if self.conv_first:
        #     x = x + self.conv(self.conv_norm_in(x))
        # else:
        #     attn_out = self.self_attn(self.attn_norm_in(x))
        #     if self.dropout_p > 0.0:
        #         attn_out = self.dropout(attn_out)
        #     x = x + attn_out
        # x = x + self.ffn1(self.ffn1_norm_in(x))
        # if self.conv_first:
        #     attn_out = self.self_attn(self.attn_norm_in(x))
        #     if self.dropout_p > 0.0:
        #         attn_out = self.dropout(attn_out)
        #     x = x + attn_out
        # else:
        #     x = x + self.conv(self.conv_norm_in(x))
        # x = x + self.ffn2(self.ffn2_norm_in(x))
        # x = self.final_norm(x)
        # x: (B, T, C)
        if self.conv_first:
            x = x + self.conv_scale(self.conv(x))
            x = self.conv_norm_out(x)
        else:
            x = x + self.attn_scale(self.dropout(self.self_attn(x)))
            x = self.attn_norm_out(x)

        x = x + self.ffn1_scale(self.ffn1(x))
        x = self.ffn1_norm_out(x)

        if self.conv_first:
            x = x + self.attn_scale(self.dropout(self.self_attn(x)))
            x = self.attn_norm_out(x)
        else:
            x = x + self.conv_scale(self.conv(x))
            x = self.conv_norm_out(x)

        x = x + self.ffn2_scale(self.ffn2(x))
        x = self.ffn2_norm_out(x)
        return x


class ConformerBackbone(nnx.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_head: int = 8,
        ffn_mult: float = 4.0,
        conv_kernel_size: int = 31,
        dropout: float = 0.0,
        max_position_embeddings: int = 2048,
        original_max_position_embeddings: int = 4096,
        base: float = 10000.0,
        conv_first: bool = False,
        causal: bool = False,
        precision: Any = lax.Precision.DEFAULT,
        rngs: nnx.Rngs = None,
    ):
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(
                ConformerLayer(
                    dim,
                    n_head,
                    ffn_mult,
                    conv_kernel_size,
                    dropout,
                    max_position_embeddings=max_position_embeddings,
                    original_max_position_embeddings=original_max_position_embeddings,
                    base=base,
                    conv_first=conv_first,
                    causal=causal,
                    precision=precision,
                    rngs=rngs,
                )
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


class Downsample(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, activation: Callable = jax.nn.silu, precision: Any = lax.Precision.DEFAULT, rngs: nnx.Rngs = None):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=2 * stride,
            strides=stride,
            padding=((stride // 2 + stride % 2, stride // 2),),
            precision=precision,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(out_channels, rngs=rngs)
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Upsample(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, activation: Callable = jax.nn.silu, precision: Any = lax.Precision.DEFAULT, rngs: nnx.Rngs = None):
        self.stride = stride
        self.kernel_size = 2 * stride
        self.padding_pt = stride // 2 + stride % 2
        self.output_padding_pt = stride % 2
        # Use VALID in ConvTranspose and apply exact Torch-style crop later
        self.conv = nnx.ConvTranspose(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=self.kernel_size,
            strides=stride,
            padding='VALID',
            transpose_kernel=True,
            precision=precision,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(out_channels, rngs=rngs)
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, L, C)
        y = self.conv(x)  # VALID -> length = (L_in - 1)*s + k
        p = self.padding_pt
        op = self.output_padding_pt
        # Exact Torch crop: remove p from left and (p - op) from right
        if p > 0:
            start = p
        else:
            start = 0
        end_trim = max(p - op, 0)
        if end_trim > 0:
            y = y[:, start: y.shape[1] - end_trim, :]
        else:
            y = y[:, start:, :]
        y = self.norm(y)
        y = self.activation(y)
        return y


class Upsample_Interpolate(nnx.Module):
    def __init__(self, scale_factor: float = 2.0, mode: str = 'nearest'):
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Only support nearest for now
        sf = int(self.scale_factor)
        x = jnp.repeat(x, sf, axis=1)
        return x