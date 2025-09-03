import jax
import jax.numpy as jnp
from typing import Optional, Any, Tuple
from flax import nnx

from .fsq_jax import FSQ
from .residual_vq_jax import ResidualVQ
from .module_jax import ConformerBackbone, RMSNorm
from common.conv_weightnorm import WNConv1d
from common.spectral import _get_window


class ISTFT(nnx.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        if padding not in ("same", "center"):
            raise ValueError("padding must be 'same' or 'center'")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.padding = padding
        # build window using periodic Hann consistent with torch.hann_window(periodic=True)
        self.window = nnx.Variable(_get_window(win_length, 'hann', True).astype(jnp.float32))

    def __call__(self, spec: jnp.ndarray) -> jnp.ndarray:
        # spec: (B, N, T) complex64/complex128
        if self.padding == "center":
            # jnp.fft.irfft across frequency axis (N)
            ifft = jnp.fft.irfft(spec, n=self.n_fft, axis=1)
            ifft = ifft * self.window.value[None, :, None]
            # Overlap-add via conv of frames
            B, N, T = ifft.shape
            out_len = (T - 1) * self.hop_length + self.win_length
            y = jnp.zeros((B, out_len), dtype=ifft.dtype)
            for t in range(T):
                start = t * self.hop_length
                end = start + self.win_length
                y = y.at[:, start:end].add(ifft[:, :, t])
            return y
        else:
            # 'same' behavior: mirror the PyTorch reference as close as feasible
            ifft = jnp.fft.irfft(spec, n=self.n_fft, axis=1)
            ifft = ifft * self.window.value[None, :, None]
            B, N, T = ifft.shape
            out_len = (T - 1) * self.hop_length + self.win_length
            pad = (self.win_length - self.hop_length) // 2
            y = jnp.zeros((B, out_len), dtype=ifft.dtype)
            for t in range(T):
                start = t * self.hop_length
                end = start + self.win_length
                y = y.at[:, start:end].add(ifft[:, :, t])
            # trim pad on both sides
            y = y[:, pad: out_len - pad]
            # window envelope normalization
            window_sq = (self.window.value ** 2)[None, :, None].repeat(T, axis=2)
            # build the folding envelope like above
            env = jnp.zeros((out_len,), dtype=ifft.dtype)
            for t in range(T):
                start = t * self.hop_length
                end = start + self.win_length
                env = env.at[start:end].add(self.window.value ** 2)
            env = env[pad: out_len - pad]
            env = jnp.clip(env, 1e-11, None)
            y = y / env[None, :]
            return y


class ISTFTHead(nnx.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same", rngs: Optional[nnx.Rngs] = None):
        out_dim = n_fft + 2
        self.out = nnx.Linear(dim, out_dim, rngs=rngs)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, T, H)
        x_pred = self.out(x)  # (B, T, n_fft+2)
        x_pred = x_pred.swapaxes(1, 2)  # (B, n_fft+2, T)
        mag, p = jnp.split(x_pred, 2, axis=1)
        mag = jnp.exp(mag)
        mag = jnp.clip(mag, a_max=1e2)
        real = jnp.cos(p)
        imag = jnp.sin(p)
        spec = mag * (real + 1j * imag)  # (B, n_fft+1, T)
        audio = self.istft(spec)
        return audio[:, None, :]


class ConformerDecoderISTFT(nnx.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1024,
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
        fsq: bool = False,
        fsq_levels: Tuple[int, ...] = (4, 4, 4, 8),
        vq_num_quantizers: int = 1,
        vq_commit_weight: float = 0.25,
        vq_weight_init: bool = False,
        vq_full_commit_loss: bool = False,
        codebook_size: int = 8192,
        codebook_dim: int = 8,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fsq = fsq
        self.in_channels = in_channels

        if fsq:
            self.quantizer = FSQ(levels=list(fsq_levels), channel_first=True, dim=in_channels)
            # commit loss shape compatibility handled in call
        else:
            self.quantizer = ResidualVQ(
                num_quantizers=vq_num_quantizers,
                codebook_size=codebook_size,
                rngs=rngs,
                dim=in_channels,
                codebook_dim=codebook_dim,
                commitment=vq_commit_weight,
            )

        class Identity(nnx.Module):
            def __call__(self, x):
                return x

        if in_channels != dim:
            self.input_proj = nnx.Linear(in_channels, dim, rngs=rngs)
        else:
            self.input_proj = Identity()
        # self.input_norm = RMSNorm(dim)

        if n_layers_stage0 > 0:
            self.conformer_backbone_stage0 = ConformerBackbone(
                dim=dim,
                n_layers=n_layers_stage0,
                n_head=n_head,
                ffn_mult=ffn_mult,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                original_max_position_embeddings=int(original_max_position_embeddings * r),
                max_position_embeddings=int(max_position_embeddings * r),
                base=base,
                causal=causal,
                conv_first=False,
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
                original_max_position_embeddings=original_max_position_embeddings,
                max_position_embeddings=max_position_embeddings,
                base=base,
                causal=causal,
                conv_first=False,
                rngs=rngs,
            )
        else:
            self.conformer_backbone_stage1 = Identity()

        # self.norm = RMSNorm(dim)
        self.head = ISTFTHead(dim=dim, n_fft=n_fft, hop_length=hop_length, padding="same", rngs=rngs)

    def __call__(self, x, vq: bool = True, stage: int = 0):
        if vq:
            if self.fsq:
                x_q, q = self.quantizer(x)
                commit_loss = jnp.zeros((x_q.shape[0],), dtype=x_q.dtype)
            else:
                # Expect BTC for ResidualVQ; transpose if given BCT
                x_in = x
                if x_in.ndim == 3 and x_in.shape[1] == self.in_channels:
                    x_in = x_in.swapaxes(1, 2)
                x_q, q, commit_loss = self.quantizer(x_in)
            return x_q, q, commit_loss

        if stage == 0:
            x = self.input_proj(x)
            # x = self.input_norm(x)
            x = self.conformer_backbone_stage0(x)  # (B, T, dim)
            return x
        elif stage == 1:
            x = self.conformer_backbone_stage1(x)  # (B, T, dim)
            # x = self.norm(x)  # normalize last dim
            audio = self.head(x)
            return audio
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    # Helper passthroughs to mirror Torch API
    def vq2emb(self, vq):
        return self.quantizer.vq2emb(vq)

    def get_emb(self):
        return self.quantizer.get_emb()

    def inference_vq(self, vq):
        x = vq[None, :, :]
        audio = self.__call__(x, vq=False)
        return audio

    def inference_0(self, x):
        x_q, q, loss = self.__call__(x, vq=True)
        audio = self.__call__(x_q, vq=False)
        return audio, None

    def inference(self, x):
        audio = self.__call__(x, vq=False)
        return audio, None