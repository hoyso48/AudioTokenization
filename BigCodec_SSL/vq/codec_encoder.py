import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .module import WNConv1d, EncoderBlock, ResLSTM, ConformerBackbone, RMSNorm, ConvDownsample, Patchify
from .alias_free_torch import *
from . import activations

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class BigCodecEncoder(nn.Module):
    def __init__(self,
                ngf=48,
                use_rnn=True,
                rnn_bidirectional=False,
                causal=False,
                antialias=False,
                rnn_num_layers=2,
                up_ratios=(2, 2, 2, 5, 5),
                dilations=(1, 3, 9),
                out_channels=1024):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        if causal:
            assert not rnn_bidirectional

        # Create first convolution
        d_model = ngf
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3, causal=causal)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for i, stride in enumerate(up_ratios):
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride, dilations=dilations, causal=causal, antialias=antialias)]
        # RNN
        if use_rnn:
            self.block += [
                ResLSTM(d_model,
                        num_layers=rnn_num_layers,
                        bidirectional=rnn_bidirectional
                    )
            ]
        # Create last convolution

        self.block += [
            Activation1d(activation=activations.SnakeBeta(d_model, alpha_logscale=True), antialias=antialias),
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1, causal=causal),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model
        
        self.reset_parameters()

    def forward(self, x):
        out = self.block(x)
        return out

    def inference(self, x):
        return self.block(x)

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
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class STFT(nn.Module):
    def __init__(self,
                 hop_length=256,
                 n_fft=1024,
                 window_size=1024,
                 window_fn=torch.hann_window,
                 ):
        super().__init__()
        self.register_buffer("window", window_fn(window_size))
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.window_size = window_size
        self.pad_mode = "constant"
        self.center = False
        self.return_complex = True
    
    def forward(self, x):
        # x: (B, 1, T) -> STFT -> (B, n_fft//2+1, n_frames)
        x = x.squeeze(1)  # (B, T)
        pad = (self.window_size - self.hop_length) // 2
        x = F.pad(x, (pad, pad), mode=self.pad_mode)
        stft_result = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window, 
            center=self.center, 
            pad_mode=self.pad_mode, 
            return_complex=self.return_complex
        )
        return stft_result

class ConformerEncoderSTFT(nn.Module):
    def __init__(self,
                 hop_length=256,
                 n_fft=1024,
                 window_size=1024,
                 dim=512,
                 n_layers_stage0=12,
                 n_layers_stage1=12,
                 r=0.5,
                 n_head=8,
                 ffn_mult=4,
                 conv_kernel_size=31,
                 dropout=0.1,
                 max_position_embeddings=2048,
                 original_max_position_embeddings=4096,
                 base=10000.0,
                 causal=False,
                 out_channels=1024):
        super().__init__()
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # STFT module
        self.stft = STFT(
            hop_length=hop_length,
            n_fft=n_fft,
            window_size=window_size
        )

        # self.patchify = nn.Conv1d(1, dim, kernel_size=hop_length, stride=hop_length)
        # self.conv = ConvDownsample(dim, dim)

        self.patchify = Patchify(2, dim, (4, (n_fft // 2) // 4)) #ex. (4, 80) for n_fft=320
        
        # Input projection: complex STFT -> real features
        # STFT output: (B, n_fft//2+1, n_frames)
        # We convert complex to real+imag: (B, 2*(n_fft//2+1), n_frames)
        stft_dim = n_fft // 2 + 1
        # self.input_proj = nn.Linear(2 * stft_dim, dim) #nn.Conv1d(2 * stft_dim, dim, kernel_size=1)

        # self.input_proj = ConvEncoderBlock(2 * stft_dim, dim)
        # self.input_norm = RMSNorm(dim)
        # self.conv = ConvDownsample(2 * stft_dim, dim)
        
        # Conformer backbone
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
                conv_first=True
            )
        else:
            self.conformer_backbone_stage0 = nn.Identity()
        # self.conformer_backbone2 = nn.Identity()
        if n_layers_stage1 > 0:
            self.conformer_backbone_stage1 = ConformerBackbone(
                dim=dim,
                n_layers=n_layers_stage1,
                n_head=n_head,
                ffn_mult=ffn_mult,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                max_position_embeddings=int(max_position_embeddings*r),
                original_max_position_embeddings=int(original_max_position_embeddings*r),
                base=base,
                causal=causal,
                conv_first=True
            )
        else:
            self.conformer_backbone_stage1 = nn.Identity()

        self.norm = RMSNorm(dim)

        # Output projection
        if out_channels != dim:
            self.output_proj = nn.Linear(dim, out_channels) #nn.Conv1d(dim, out_channels, kernel_size=1)
        else:
            self.output_proj = nn.Identity()
        
        self.reset_parameters()

    def forward(self, x, stage=0):
        # x = self.patchify(x)
        # x: (B, 1, T) - raw audio
        
        # STFT
        if stage == 0:
            # stft_result = self.stft(x)  # (B, n_fft//2+1, n_frames)
            
            # # # Convert complex to real and imaginary parts
            # # real_part = stft_result.real  # (B, n_fft//2+1, n_frames)
            # # imag_part = stft_result.imag  # (B, n_fft//2+1, n_frames)
            
            # # # Concatenate real and imaginary parts
            # # stft_features = torch.cat([real_part, imag_part], dim=1)  # (B, 2*(n_fft//2+1), n_frames)
            # stft_features = torch.view_as_real(stft_result).permute(0, 2, 1, 3).flatten(2)
            
            # # Input projection
            # # x = self.input_proj(stft_features)  # (B, n_frames, dim)
            # # x = self.input_norm(x)
            # x = self.conv(stft_features)

            # 1d-patchify waveform
            # x = self.patchify(x).transpose(1, 2) # (B, 1, n_frames) -> (B, n_frames//patch_size, dim)
            # x = self.conv(x)

            # 2d-patchify stft
            x = self.stft(x) # (B, n_fft//2+1, n_frames)
            x = torch.stack([x.real[:, :-1, :], x.imag[:, :-1, :]], dim=-1).transpose(1, 2) # (B, n_frames, n_fft//2, 2), drop nyquist frequency. TODO: more correct way?
            x = self.patchify(x) # (B, T//patch_size, F//patch_size, D)
            # print(x.shape)
            x = x.flatten(1, 2) # (B, L, D)
            x = self.norm(x)
            # print(x.shape)

            # Conformer backbone
            x = self.conformer_backbone_stage0(x)  # (B, n_frames, dim)

        elif stage == 1:
            x = self.conformer_backbone_stage1(x)  # (B, n_frames, dim)
            
            # Output projection
            x = self.output_proj(x)  # (B, n_frames, out_channels)
        
        return x

    def reset_parameters(self):
        self.apply(init_weights)