"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
JAX/Flax NNX로 구현한 버전
"""

from __future__ import annotations
from functools import wraps, partial
from contextlib import nullcontext
from typing import List, Tuple, Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax

from einops import rearrange, pack, unpack

import random

class Constant(nnx.Variable):
    pass

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def round_ste(z):
    """Round with straight through gradients."""
    zhat = jnp.round(z)
    return z + jax.lax.stop_gradient(zhat - z)

# NNX 버전의 Identity 모듈
class Identity(nnx.Module):
    """Identity module for Flax NNX"""
    
    def __init__(self):
        pass
        
    def __call__(self, x):
        return x

# main class

class FSQ(nnx.Module):
    def __init__(
        self, 
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        channel_first: bool = False,
        projection_has_bias: bool = True,
        return_indices: bool = True,
        force_quantization_f32: bool = True,
        preserve_symmetry: bool = False,
        noise_approx_prob: float = 0.0,
        rngs: nnx.Rngs = None
    ):
        # 설정값 저장
        self.levels = levels
        self.dim = dim
        self.num_codebooks = num_codebooks
        self.keep_num_codebooks_dim = keep_num_codebooks_dim
        self.scale = scale
        self.channel_first = channel_first
        self.projection_has_bias = projection_has_bias
        self.return_indices = return_indices
        self.force_quantization_f32 = force_quantization_f32
        self.preserve_symmetry = preserve_symmetry
        self.noise_approx_prob = noise_approx_prob
        
        # 내부 계산값 초기화
        self._levels = Constant(jnp.array(self.levels, dtype=jnp.int32))
        self._basis = Constant(jnp.cumprod(jnp.array([1] + list(self.levels)[:-1]), axis=0, dtype=jnp.int32))
        
        codebook_dim = len(self.levels)
        self.codebook_dim = codebook_dim
        
        effective_codebook_dim = codebook_dim * self.num_codebooks
        self.effective_codebook_dim = effective_codebook_dim
        
        keep_num_codebooks_dim = default(self.keep_num_codebooks_dim, self.num_codebooks > 1)
        assert not (self.num_codebooks > 1 and not keep_num_codebooks_dim)
        
        dim_size = default(self.dim, len(self._levels) * self.num_codebooks)
        
        has_projections = dim_size != effective_codebook_dim
        
        # 프로젝션 레이어 생성
        if has_projections:
            self.project_in = nnx.Linear(
                dim_size, 
                effective_codebook_dim, 
                use_bias=self.projection_has_bias, 
                rngs=rngs
            )
            self.project_out = nnx.Linear(
                effective_codebook_dim, 
                dim_size, 
                use_bias=self.projection_has_bias, 
                rngs=rngs
            )
        else:
            self.project_in = Identity()
            self.project_out = Identity()
        
        self.has_projections = has_projections
        
        if self.return_indices:
            # JIT 컴파일 시 동적 크기 배열 생성 방지
            self.codebook_size = int(jnp.prod(self._levels))
            self.implicit_codebook = None
        
        # 훈련 모드 플래그 (NNX에서는 명시적으로 관리)
        self.training = True

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels.value - 1) * (1 + eps) / 2
        offset = jnp.where(self._levels.value % 2 == 0, 0.5, 0.0)
        shift = jnp.arctanh(offset / half_l)
        return jnp.tanh(z + shift) * half_l - offset

    def symmetry_preserving_bound(self, z):
        """
        QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1
        """
        levels_minus_1 = (self._levels.value - 1)
        scale = 2.0 / levels_minus_1
        bracket = (levels_minus_1 * (jnp.tanh(z) + 1) / 2.0) + 0.5
        return scale * bracket - 1.0

    def noise_approx_bound(self, z):
        """
        simulates quantization using noise -> Q_L(x) ~= tanh(x) + U{-1,1} / (L-1)
        """
        noise = jax.random.uniform(
            jax.random.PRNGKey(0),
            z.shape, 
            minval=-1.0, 
            maxval=1.0
        )
        return jnp.tanh(z) + noise / (self._levels.value - 1)

    def quantize(self, z, preserve_symmetry=False):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        key = jax.random.PRNGKey(0)
        use_noise_prob = jnp.array(self.noise_approx_prob, dtype=jnp.float32)
        random_val = jax.random.uniform(key, shape=())
        use_noise = self.training and (random_val < use_noise_prob)
        
        bounded = jax.lax.cond(
            use_noise,
            lambda x: self.noise_approx_bound(x),
            lambda x: jax.lax.cond(
                preserve_symmetry,
                lambda y: self.symmetry_preserving_bound(y),
                lambda y: self.bound(y),
                x
            ),
            z
        )
            
        quantized = round_ste(bounded)
        half_width = self._levels.value // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels.value // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels.value // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return jnp.sum(zhat * self._basis.value, axis=-1).astype(jnp.int32)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis.value) % self._levels.value
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def __call__(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        keep_num_codebooks_dim = default(self.keep_num_codebooks_dim, self.num_codebooks > 1)
        dim_size = default(self.dim, len(self._levels.value) * self.num_codebooks)
        
        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)
        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == dim_size, f'expected dimension of {dim_size} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        # whether to force quantization step to be full precision or not
        force_f32 = self.force_quantization_f32
        
        orig_dtype = z.dtype
        
        if force_f32 and orig_dtype != jnp.float32:
            z = z.astype(jnp.float32)

        codes = self.quantize(z, preserve_symmetry=self.preserve_symmetry)

        # returning indices could be optional
        indices = None

        if self.return_indices:
            indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        codes = codes.astype(orig_dtype)

        # project out
        out = self.project_out(codes)

        # reconstitute image or video dimensions
        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            if indices is not None:
                indices = maybe(unpack_one)(indices, ps, 'b * c')

        if not keep_num_codebooks_dim and self.return_indices:
            if indices is not None:
                indices = maybe(rearrange)(indices, '... 1 -> ...')

        # return quantized output and indices
        return out, indices
    
    # def set_training(self, training: bool):
    #     """훈련 모드 설정 메서드"""
    #     self.training = training