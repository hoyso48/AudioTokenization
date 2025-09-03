import jax
import jax.numpy as jnp
from typing import Optional
from flax import nnx

from AudioTokenization.BigCodec_NNX.common.spectral import Spectrogram
from AudioTokenization.BigCodec_NNX.common.conv_weightnorm import WNConv1d
from .module_jax import ConformerBackbone, RMSNorm


class ConformerEncoderSTFT(nnx.Module):
    def __init__(
        self,
        *,
        hop_length: int = 256,
        n_fft: int = 1024,
        window_size: int = 1024,
        dim: int = 512,
        n_layers_stage0: int = 12,
        n_layers_stage1: int = 12,
        r: float = 0.5,
        n_head: int = 8,
        ffn_mult: float = 4.0,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        original_max_position_embeddings: int = 4096,
        base: float = 10000.0,
        causal: bool = False,
        out_channels: int = 1024,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.hop_length = hop_length
        self.n_fft = n_fft

        pad = (window_size - hop_length) // 2
        self.stft = Spectrogram(
            n_fft=n_fft,
            win_length=window_size,
            hop_length=hop_length,
            pad=pad,
            window_type='hann',
            power=None,  # return complex
            normalized=False,
            center=False,
            pad_mode='constant',
            rngs=rngs,
        )

        stft_dim = n_fft // 2 + 1
        self.input_proj = nnx.Linear(2 * stft_dim, dim, rngs=rngs)
        self.input_norm = RMSNorm(dim)

        class Identity(nnx.Module):
            def __call__(self, x):
                return x

        if n_layers_stage0 > 0:
            self.conformer_backbone_stage0 = ConformerBackbone(
                dim=dim,
                n_layers=n_layers_stage0,
                n_head=n_head,
                ffn_mult=ffn_mult,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                max_position_embeddings=max_position_embeddings,
                original_max_position_embeddings=original_max_position_embeddings,
                base=base,
                causal=causal,
                conv_first=True,
                rngs=rngs,
            )
        else:
            self.conformer_backbone_stage0 = Identity()

        if n_layers_stage1 > 0:
            self.conformer_backbone_stage1 = ConformerBackbone(
                dim=dim,
                n_layers=n_layers_stage1,
                n_head=n_head,
                ffn_mult=ffn_mult,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                max_position_embeddings=int(max_position_embeddings * r),
                original_max_position_embeddings=int(original_max_position_embeddings * r),
                base=base,
                causal=causal,
                conv_first=True,
                rngs=rngs,
            )
        else:
            self.conformer_backbone_stage1 = Identity()

        # self.norm = RMSNorm(dim)

        if out_channels != dim:
            self.output_proj = nnx.Linear(dim, out_channels, rngs=rngs)
        else:
            self.output_proj = Identity()

    def __call__(self, x, stage: int = 0):
        # Accept (B, T), (B, 1, T), or (B, T, 1) and normalize to (B, T)
        if x.ndim == 3:
            b, a, c = x.shape
            if a == 1:
                x_wave = x[:, 0, :]
            elif c == 1:
                x_wave = x[:, :, 0]
            else:
                x_wave = x
        elif x.ndim == 2:
            x_wave = x
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        if stage == 0:
            # STFT: (B, F, T_frames) complex
            stft_result = self.stft(x_wave)
            # Concatenate real/imag on frequency axis -> (B, 2F, T_frames)
            real_part = jnp.real(stft_result)
            imag_part = jnp.imag(stft_result)
            stft_features = jnp.concatenate([real_part, imag_part], axis=1)

            # Project to model dim: convert to (B, T_frames, C)
            x_feats = jnp.swapaxes(stft_features, 1, 2)
            x_feats = self.input_proj(x_feats)
            x_feats = self.input_norm(x_feats)

            # Conformer backbone (expects (B, T, C))
            x_feats = self.conformer_backbone_stage0(x_feats)
            return x_feats  # (B, T_frames, dim)

        elif stage == 1:
            # Expect features (B, T, C)
            x_feats = x
            if x_feats.ndim != 3:
                raise ValueError(f"Stage 1 expects (B, T, C), got {x_feats.shape}")
            x_feats = self.conformer_backbone_stage1(x_feats)
            # x_feats = self.norm(x_feats)
            # Output projection to out_channels if needed (1x1 conv)
            x_out = self.output_proj(x_feats)
            return x_out  # (B, T, out_channels)

        else:
            raise ValueError(f"Unsupported stage: {stage}")
