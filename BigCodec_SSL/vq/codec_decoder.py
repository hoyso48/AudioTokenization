import numpy as np
import torch
import torch.nn as nn
from .residual_vq import ResidualVQ
from .module import WNConv1d, DecoderBlock, ResLSTM, ConformerBackbone, RMSNorm
from .alias_free_torch import *
from . import activations
from .vector_quantize_pytorch_lucidrains import VectorQuantize, FSQ

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class BigCodecDecoder(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 upsample_initial_channel=1536,
                 ngf=48,
                 use_rnn=True,
                 rnn_bidirectional=False,
                 rnn_num_layers=2,
                 up_ratios=(5, 5, 2, 2, 2),
                 dilations=(1, 3, 9),
                 causal=False,
                 antialias=False,
                 fsq=False,
                 fsq_levels=[4,4,4,8],
                 vq_num_quantizers=1,
                 vq_commit_weight=0.25,
                 vq_weight_init=False,
                 vq_full_commit_loss=False,
                 codebook_size=8192,
                 codebook_dim=8,
                ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios
        self.fsq = fsq
        if fsq:
            self.quantizer = FSQ(
                levels = fsq_levels,
                channel_first = True,
                dim = in_channels,
            )
            assert codebook_size == np.prod(fsq_levels), "codebook_size must be equal to the product of fsq_levels"
        else:
            self.quantizer = ResidualVQ(
                num_quantizers=vq_num_quantizers,
                dim=in_channels,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                threshold_ema_dead_code=2,
                commitment=vq_commit_weight,
                weight_init=vq_weight_init,
                full_commit_loss=vq_full_commit_loss,
            )
        channels = upsample_initial_channel
        layers = [WNConv1d(in_channels, channels, kernel_size=7, padding=3, causal=causal)]
        
        if use_rnn:
            layers += [
                ResLSTM(channels,
                        num_layers=rnn_num_layers,
                        bidirectional=rnn_bidirectional
                    )
            ]
        
        for i, stride in enumerate(up_ratios):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, dilations, causal=causal, antialias=antialias)]

        layers += [
            Activation1d(activation=activations.SnakeBeta(output_dim, alpha_logscale=True), antialias=antialias),
            WNConv1d(output_dim, 1, kernel_size=7, padding=3, causal=causal),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        
        self.reset_parameters()

    def forward(self, x, vq=True):
        if vq is True:
            if self.fsq:
                x, q = self.quantizer(x)
                commit_loss = torch.zeros(x.shape[0], device = x.device)
            else:
                x, q, commit_loss = self.quantizer(x)
            return x, q, commit_loss
        x = self.model(x)
        return x

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
        x = self.model(x)
        return x

    def inference_0(self, x):
        x, q, loss, perp = self.quantizer(x)
        x = self.model(x)
        return x, None
    
    def inference(self, x):
        x = self.model(x)
        return x, None


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
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x_pred = self.out(x )
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
        nn.init.constant_(m.bias, 0)

# class CodecDecoderVocos(nn.Module):
#     def __init__(self,
#                  hidden_dim=1024,
#                  depth=12,
#                  heads=16,
#                  pos_meb_dim=64,
#                  hop_length=320,
#                  vq_num_quantizers=1,
#                  vq_dim=2048, #1024 2048
#                  vq_commit_weight=0.25,
#                  vq_weight_init=False,
#                  vq_full_commit_loss=False,
#                  codebook_size=16384,
#                  codebook_dim=16,
#                 ):
#         super().__init__()
#         self.hop_length = hop_length
 
#         self.quantizer = ResidualFSQ(
#             dim = vq_dim,
#             levels = [4, 4, 4, 4, 4,4,4,4],
#             num_quantizers = 1
#         )
        
#         # self.quantizer = ResidualVQ(
#         #     num_quantizers=vq_num_quantizers,
#         #     dim=vq_dim,  
#         #     codebook_size=codebook_size,
#         #     codebook_dim=codebook_dim,
#         #     threshold_ema_dead_code=2,
#         #     commitment=vq_commit_weight,
#         #     weight_init=vq_weight_init,
#         #     full_commit_loss=vq_full_commit_loss,
#         # )
 
 
#         self.backbone = VocosBackbone( hidden_dim=hidden_dim,depth=depth,heads=heads,pos_meb_dim=pos_meb_dim)

#         self.head = ISTFTHead(dim=hidden_dim, n_fft=self.hop_length*4, hop_length=self.hop_length, padding="same")
 
#         self.reset_parameters()

#     def forward(self, x, vq=True):
#         if vq is True:
#             # x, q, commit_loss = self.quantizer(x)
#             x = x.permute(0, 2, 1)
#             x, q = self.quantizer(x)
#             x = x.permute(0, 2, 1)
#             q = q.permute(0, 2, 1)
#             return x, q, None
#         x = self.backbone(x)
#         x,_  = self.head(x)
 
#         return x ,_

#     def vq2emb(self, vq):
#         self.quantizer = self.quantizer.eval()
#         x = self.quantizer.vq2emb(vq)
#         return x

#     def get_emb(self):
#         self.quantizer = self.quantizer.eval()
#         embs = self.quantizer.get_emb()
#         return embs

#     def inference_vq(self, vq):
#         x = vq[None,:,:]
#         x = self.model(x)
#         return x

#     def inference_0(self, x):
#         x, q, loss, perp = self.quantizer(x)
#         x = self.model(x)
#         return x, None
    
#     def inference(self, x):
#         x = self.model(x)
#         return x, None


#     def remove_weight_norm(self):
#         """Remove weight normalization module from all of the layers."""

#         def _remove_weight_norm(m):
#             try:
#                 torch.nn.utils.remove_weight_norm(m)
#             except ValueError:  # this module didn't have weight norm
#                 return

#         self.apply(_remove_weight_norm)

#     def apply_weight_norm(self):
#         """Apply weight normalization module from all of the layers."""

#         def _apply_weight_norm(m):
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
#                 torch.nn.utils.weight_norm(m)

#         self.apply(_apply_weight_norm)

#     def reset_parameters(self):
#         self.apply(init_weights)

class ConformerDecoderISTFT(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 hop_length=256,
                 n_fft=1024,
                 window_size=1024,
                 dim=512,
                 n_layers=12,
                 n_head=8,
                 ffn_mult=4,
                 conv_kernel_size=31,
                 dropout=0.1,
                 max_seq_len=8192,
                 rope_theta=10000.0,
                 causal=False,
                 # Quantizer parameters
                 fsq=False,
                 fsq_levels=[4,4,4,8],
                 vq_num_quantizers=1,
                 vq_commit_weight=0.25,
                 vq_weight_init=False,
                 vq_full_commit_loss=False,
                 codebook_size=8192,
                 codebook_dim=8,
                ):
        super().__init__()
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fsq = fsq
        
        # Quantizer
        if fsq:
            self.quantizer = FSQ(
                levels = fsq_levels,
                channel_first = True,
                dim = in_channels,
            )
            assert codebook_size == np.prod(fsq_levels), "codebook_size must be equal to the product of fsq_levels"
        else:
            self.quantizer = ResidualVQ(
                num_quantizers=vq_num_quantizers,
                dim=in_channels,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                threshold_ema_dead_code=2,
                commitment=vq_commit_weight,
                weight_init=vq_weight_init,
                full_commit_loss=vq_full_commit_loss,
            )
        
        # Input projection from quantized features to conformer dimension
        if in_channels != dim:
            self.input_proj = torch.nn.utils.weight_norm(nn.Conv1d(in_channels, dim, kernel_size=1))
        else:
            self.input_proj = nn.Identity()
        
        # Conformer backbone
        self.conformer_backbone = ConformerBackbone(
            dim=dim,
            n_layers=n_layers,
            n_head=n_head,
            ffn_mult=ffn_mult,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            causal=causal,
            conv_first=False
        )

        self.norm = RMSNorm(dim)
        
        # Use existing ISTFTHead
        self.head = ISTFTHead(dim=dim, n_fft=n_fft, hop_length=hop_length, padding="same")
        
        self.reset_parameters()

    def forward(self, x, vq=True):
        if vq is True:
            if self.fsq:
                x, q = self.quantizer(x)
                commit_loss = torch.zeros(x.shape[0], device = x.device)
            else:
                x, q, commit_loss = self.quantizer(x)
            return x, q, commit_loss
        
        # Input projection
        x = self.input_proj(x)  # (B, dim, T)
        
        # Conformer backbone
        x = self.conformer_backbone(x)  # (B, dim, T)
        
        # Convert to (B, T, dim) for ISTFTHead
        x = x.transpose(1, 2)  # (B, T, dim)

        x = self.norm(x)
        
        # Use ISTFTHead
        audio, x_pred = self.head(x)
        
        return audio

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
