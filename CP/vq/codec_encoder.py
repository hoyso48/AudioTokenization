import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .module import ConformerBackbone, RMSNorm

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def set_module_trainable(module, trainable):
    module.requires_grad_(bool(trainable))

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
                 n_layers=12,
                 n_head=8,
                 ffn_mult=4,
                 conv_kernel_size=31,
                 dropout=0.1,
                 max_position_embeddings=2048,
                 original_max_position_embeddings=4096,
                 base=10000.0,
                 causal=False,
                 out_channels=1024,
                 trainable=True):
        super().__init__()
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.trainable = bool(trainable)
        
        # STFT module
        self.stft = STFT(
            hop_length=hop_length,
            n_fft=n_fft,
            window_size=window_size
        )
        # self.patchify = nn.Conv1d(1, dim, kernel_size=hop_length, stride=hop_length)
        
        # Input projection: complex STFT -> real features
        # STFT output: (B, n_fft//2+1, n_frames)
        # We convert complex to real+imag: (B, 2*(n_fft//2+1), n_frames)
        stft_dim = n_fft // 2 + 1
        self.input_proj = nn.Conv1d(2 * stft_dim, dim, kernel_size=1)
        self.input_norm = RMSNorm(dim)
        
        # Conformer backbone
        self.conformer_backbone1 = ConformerBackbone(
            dim=dim,
            n_layers=n_layers,#-n_layers//2,
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
        self.conformer_backbone2 = nn.Identity()
        # self.conformer_backbone2 = ConformerBackbone(
        #     dim=dim,
        #     n_layers=n_layers//2,
        #     n_head=n_head,
        #     ffn_mult=ffn_mult,
        #     conv_kernel_size=conv_kernel_size,
        #     dropout=dropout,
        #     max_position_embeddings=max_position_embeddings,
        #     original_max_position_embeddings=original_max_position_embeddings//2,
        #     base=base,
        #     causal=causal,
        #     conv_first=True
        # )

        self.norm = RMSNorm(dim)

        # Output projection
        if out_channels != dim:
            self.output_proj = torch.nn.utils.weight_norm(nn.Conv1d(dim, out_channels, kernel_size=1))
        else:
            self.output_proj = nn.Identity()
        
        self.reset_parameters()
        set_module_trainable(self, self.trainable)

    def forward(self, x, stage=0):
        # x = self.patchify(x)
        # x: (B, 1, T) - raw audio
        
        # STFT
        if stage == 0:
            stft_result = self.stft(x)  # (B, n_fft//2+1, n_frames)
            
            # Convert complex to real and imaginary parts
            real_part = stft_result.real  # (B, n_fft//2+1, n_frames)
            imag_part = stft_result.imag  # (B, n_fft//2+1, n_frames)
            
            # Concatenate real and imaginary parts
            stft_features = torch.cat([real_part, imag_part], dim=1)  # (B, 2*(n_fft//2+1), n_frames)
            
            # Input projection
            x = self.input_proj(stft_features)  # (B, dim, n_frames)
            x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
            
            # Conformer backbone
            x = self.conformer_backbone1(x)  # (B, dim, n_frames)

        elif stage == 1:
            x = self.conformer_backbone2(x)  # (B, dim, n_frames)
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
            
            # Output projection
            x = self.output_proj(x)  # (B, out_channels, n_frames)
        
        return x

    def reset_parameters(self):
        self.apply(init_weights)
