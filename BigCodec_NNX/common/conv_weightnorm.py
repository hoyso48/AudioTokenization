import math
from typing import Callable, Optional, Any, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax as lax
from flax import nnx
from flax.linen.dtypes import promote_dtype

def _as_padding_lax(padding: Union[str, int, Sequence[Tuple[int, int]]]) -> Union[str, Sequence[Tuple[int, int]]]:
    if isinstance(padding, str):
        return padding
    if isinstance(padding, int):
        return [(padding, padding)]
    if isinstance(padding, Sequence):
        if len(padding) == 1:
            p0 = padding[0]
            if isinstance(p0, int):
                return [(p0, p0)]
            if isinstance(p0, tuple) and len(p0) == 2:
                return padding
        if len(padding) == 2 and isinstance(padding[0], int) and isinstance(padding[1], int):
            return [(padding[0], padding[1])]
    raise ValueError(f"Invalid padding format: {padding}")


class WNConv1d(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        *,
        strides: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[Tuple[int, int]]] = 'SAME',
        kernel_dilation: Union[int, Sequence[int]] = 1,
        feature_group_count: int = 1,
        use_bias: bool = True,
        dtype: Optional[Any] = None,
        param_dtype: Any = jnp.float32,
        precision: Any = lax.Precision.HIGHEST,
        kernel_init: Callable | None = None,
        bias_init: Callable | None = None,
        rngs: nnx.Rngs | None = None,
        eps: float = 1e-8,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_dilation = kernel_dilation
        self.feature_group_count = feature_group_count
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.eps = eps

        if kernel_init is None:
            kernel_init = nnx.initializers.lecun_normal()
        if bias_init is None:
            bias_init = nnx.initializers.zeros

        v_shape = (kernel_size, in_features // feature_group_count, out_features)
        v_init = kernel_init(rngs.params(), v_shape)
        self.weight_v = nnx.Param(v_init.astype(self.param_dtype))
        v_norm = jnp.sqrt(jnp.sum(jnp.square(self.weight_v.value), axis=(0, 1)))
        self.weight_g = nnx.Param(v_norm.astype(self.param_dtype))

        if use_bias:
            self.bias = nnx.Param(bias_init(rngs.params(), (out_features,)).astype(self.param_dtype))

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # inputs: (N, L, C)
        strides = (self.strides,) if isinstance(self.strides, int) else tuple(self.strides)
        rhs_dilation = (self.kernel_dilation,) if isinstance(self.kernel_dilation, int) else tuple(self.kernel_dilation)
        padding_lax = _as_padding_lax(self.padding)

        v = self.weight_v.value
        v_norm = jnp.sqrt(jnp.sum(jnp.square(v), axis=(0, 1))) + self.eps
        scale = (self.weight_g.value / v_norm).reshape((1, 1, -1))
        w = v * scale

        bias = self.bias.value if self.use_bias else None
        x, w, bias = promote_dtype(inputs, w, bias, dtype=self.dtype)

        y = lax.conv_general_dilated(
            lhs=x,
            rhs=w,
            window_strides=strides,
            padding=padding_lax,
            lhs_dilation=None,
            rhs_dilation=rhs_dilation,
            dimension_numbers=('NHC', 'HIO', 'NHC'),
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )
        if self.use_bias:
            y = y + bias.reshape((1, 1, -1))
        return y


class WNConvTranspose1d(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        *,
        strides: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[Tuple[int, int]]] = 'SAME',
        kernel_dilation: Union[int, Sequence[int]] = 1,
        use_bias: bool = True,
        dtype: Optional[Any] = None,
        param_dtype: Any = jnp.float32,
        precision: Any = lax.Precision.HIGHEST,
        kernel_init: Callable | None = None,
        bias_init: Callable | None = None,
        rngs: nnx.Rngs | None = None,
        eps: float = 1e-8,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_dilation = kernel_dilation
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.eps = eps

        if kernel_init is None:
            kernel_init = nnx.initializers.lecun_normal()
        if bias_init is None:
            bias_init = nnx.initializers.zeros

        v_shape = (kernel_size, in_features, out_features)
        v_init = kernel_init(rngs.params(), v_shape)
        self.weight_v = nnx.Param(v_init.astype(self.param_dtype))
        v_norm = jnp.sqrt(jnp.sum(jnp.square(self.weight_v.value), axis=(0, 1)))
        self.weight_g = nnx.Param(v_norm.astype(self.param_dtype))

        if use_bias:
            self.bias = nnx.Param(bias_init(rngs.params(), (out_features,)).astype(self.param_dtype))

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # inputs: (N, L, C)
        strides = (self.strides,) if isinstance(self.strides, int) else tuple(self.strides)
        rhs_dilation = (self.kernel_dilation,) if isinstance(self.kernel_dilation, int) else tuple(self.kernel_dilation)
        padding_lax = _as_padding_lax(self.padding)

        v = self.weight_v.value
        v_norm = jnp.sqrt(jnp.sum(jnp.square(v), axis=(0, 1))) + self.eps
        scale = (self.weight_g.value / v_norm).reshape((1, 1, -1))
        w = v * scale

        bias = self.bias.value if self.use_bias else None
        x, w, bias = promote_dtype(inputs, w, bias, dtype=self.dtype)

        y = lax.conv_transpose(
            lhs=x,
            rhs=w,
            strides=strides,
            padding=padding_lax,
            rhs_dilation=rhs_dilation,
            dimension_numbers=('NHC', 'HIO', 'NHC'),
            precision=self.precision,
        )
        if self.use_bias:
            y = y + bias.reshape((1, 1, -1))
        return y


class WNConv2d(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: Tuple[int, int],
        *,
        strides: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int, Tuple[int, int], Sequence[Tuple[int, int]]] = 'SAME',
        kernel_dilation: Union[int, Tuple[int, int]] = 1,
        feature_group_count: int = 1,
        use_bias: bool = True,
        dtype: Optional[Any] = None,
        param_dtype: Any = jnp.float32,
        precision: Any = lax.Precision.HIGHEST,
        kernel_init: Callable | None = None,
        bias_init: Callable | None = None,
        rngs: nnx.Rngs | None = None,
        eps: float = 1e-8,
    ):
        self.in_features = in_features
        self.out_features = out_features
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        # Normalize padding to 2D spec for lax
        if isinstance(padding, str):
            self.padding = padding
        elif isinstance(padding, int):
            self.padding = [(padding, padding), (padding, padding)]
        elif isinstance(padding, tuple) and len(padding) == 2 and isinstance(padding[0], int):
            self.padding = [(padding[0], padding[0]), (padding[1], padding[1])]
        else:
            self.padding = padding
        self.kernel_dilation = kernel_dilation if isinstance(kernel_dilation, tuple) else (kernel_dilation, kernel_dilation)
        self.feature_group_count = feature_group_count
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.eps = eps

        if kernel_init is None:
            kernel_init = nnx.initializers.lecun_normal()
        if bias_init is None:
            bias_init = nnx.initializers.zeros

        kh, kw = self.kernel_size
        v_shape = (kh, kw, in_features // feature_group_count, out_features)
        v_init = kernel_init(rngs.params(), v_shape)
        self.weight_v = nnx.Param(v_init.astype(self.param_dtype))
        v_norm = jnp.sqrt(jnp.sum(jnp.square(self.weight_v.value), axis=(0, 1, 2)))
        self.weight_g = nnx.Param(v_norm.astype(self.param_dtype))

        if use_bias:
            self.bias = nnx.Param(bias_init(rngs.params(), (out_features,)).astype(self.param_dtype))

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # inputs: (N, H, W, C)
        v = self.weight_v.value
        v_norm = jnp.sqrt(jnp.sum(jnp.square(v), axis=(0, 1, 2))) + self.eps
        scale = (self.weight_g.value / v_norm).reshape((1, 1, 1, -1))
        w = v * scale

        bias = self.bias.value if self.use_bias else None
        x, w, bias = promote_dtype(inputs, w, bias, dtype=self.dtype)

        y = lax.conv_general_dilated(
            lhs=x,
            rhs=w,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=None,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )
        if self.use_bias:
            y = y + bias.reshape((1, 1, 1, -1))
        return y