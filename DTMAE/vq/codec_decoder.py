import numpy as np
import torch
import torch.nn as nn
from .module import Transformer, ConvUpsample, UnPatchify1D
from .alias_free_torch import *

# Quantizer imports - for dynamic instantiation
import vq.quantizers as quantizers

class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y

class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

class ISTFTHead(FourierHead):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        if dim != out_dim:
            self.out = nn.Linear(dim, out_dim)
        else:
            self.out = nn.Identity()
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    @torch.compiler.disable(recursive=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x_pred = self.out(x)
        # x_pred = x
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value 
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio.unsqueeze(1),x_pred


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class TransformerDecoderISTFT(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 hop_length=256,
                 n_fft=1024,
                 window_size=1024,
                 dim=512,
                 n_layers_level1=12,
                 n_layers_level2=12,
                 r=0.5,
                 n_head=8,
                 ffn_mult=4,
                 dropout=0.1,
                 max_position_embeddings=2048,
                 base=10000.0,
                 causal=False,
                 # Quantizer config (extensible pattern like resampler)
                 quantizer_cls='ResidualVQ',
                 quantizer_params=None,
                 # Legacy quantizer parameters (kept for reference, use quantizer_cls/params instead)
                 # fsq=False,
                 # simvq=False,
                 # fsq_levels=[4,4,4,8],
                 # vq_num_quantizers=1,
                 # vq_commit_weight=0.25,
                 # vq_weight_init=False,
                 # vq_full_commit_loss=False,
                 # codebook_size=8192,
                 # codebook_dim=8,
                 norm_eps: float = 1e-2,
                 attn_window_size=(64, 64),
                ):
        super().__init__()
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.quantizer_cls = quantizer_cls
        
        # Default quantizer params if not provided
        if quantizer_params is None:
            quantizer_params = {
                'dim': in_channels,
                'codebook_size': 16384,
                'codebook_dim': 8,
                'num_quantizers': 1,
                'commitment': 0.5,
            }
        
        # Dynamic quantizer instantiation (like resampler pattern)
        quantizer_class = getattr(quantizers, quantizer_cls)
        self.quantizer = quantizer_class(**quantizer_params)
        
        # Input projection from quantized features to conformer dimension
        if in_channels != dim:
            self.input_proj = nn.Linear(in_channels, dim) #nn.Conv1d(in_channels, dim, kernel_size=1)
        else:
            self.input_proj = nn.Identity()
        
        if n_layers_level1 > 0:
            self.transformer_backbone_level2 = Transformer(
                dim=dim,
                n_layers=n_layers_level2,
                n_head=n_head,
                ffn_mult=ffn_mult,
                dropout=dropout,
                max_position_embeddings=int(max_position_embeddings*r),
                base=base,
                causal=causal,
                attn_window_size=attn_window_size,
                norm_eps=norm_eps,
                )
        else:
            self.transformer_backbone_level2 = nn.Identity()
        
        if n_layers_level1 > 0:
            self.transformer_backbone_level1 = Transformer(
                dim=dim,
                n_layers=n_layers_level1,
                n_head=n_head,
                ffn_mult=ffn_mult,
                dropout=dropout,
                max_position_embeddings=max_position_embeddings,
                base=base,
                causal=causal,
                attn_window_size=attn_window_size,
                norm_eps=norm_eps,
                )
        else:
            self.transformer_backbone_level1 = nn.Identity()
        
        # Use existing ISTFTHead
        self.head = ISTFTHead(dim=n_fft+2, n_fft=n_fft, hop_length=hop_length, padding="same")
        # self.head = UnPatchify1D(dim, 1, hop_length)

        self.conv = ConvUpsample(dim, n_fft+2, norm_eps=norm_eps)
        
        self.reset_parameters()

    def forward(self, x, vq=True, position_ids=None, cu_seqlens=None, max_seqlen=None, level=1):
        if vq is True:
            # Unified quantizer interface
            # All quantizers return: (quantized, indices, commit_loss_or_none)
            result = self.quantizer(x)
            
            # Handle different return formats
            if len(result) == 2:
                # FSQ-style: (quantized, indices)
                x, q = result
                commit_loss = None
            else:
                # VQ-style: (quantized, indices, commit_loss)
                x, q, commit_loss = result
                # Ensure commit_loss is a list for consistency
                if commit_loss is not None and not isinstance(commit_loss, list):
                    if isinstance(commit_loss, torch.Tensor) and commit_loss.dim() == 0:
                        commit_loss = [commit_loss]
            
            return x, q, commit_loss
        
        if level == 2:
            # Input projection
            x = self.input_proj(x)  # (B, T, dim)

            x = self.transformer_backbone_level2(x, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)  # (B, T, dim)
            return x
        elif level == 1:
            x = self.transformer_backbone_level1(x)  # (B, T, dim)

            x = self.conv(x)
            
            audio, x_pred = self.head(x)

            # audio = self.head(x)
            # audio = audio.permute(0, 2, 1)

            return audio
        else:
            raise ValueError(f"Unsupported level: {level}")

    def vq2emb(self, vq):
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self):
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    def inference_vq(self, vq):
        x = vq[None,:,:]
        audio = self.forward(x, vq=False)
        return audio

    def inference_0(self, x):
        x, q, loss = self.forward(x, vq=True)
        audio = self.forward(x, vq=False)
        return audio, None
    
    def inference(self, x):
        audio = self.forward(x, vq=False)
        return audio, None

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)
