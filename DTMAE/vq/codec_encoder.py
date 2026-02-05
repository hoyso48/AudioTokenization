import torch
from torch import nn
import torch.nn.functional as F
from .module import  Transformer, ConvDownsample, Patchify1D, RMSNorm, WNConv1d#, WNConv1dVarlen
from .alias_free_torch import *

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

class TransformerEncoderSTFT(nn.Module):
    def __init__(self,
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
                 out_channels=1024,
                 norm_eps: float = 1e-2,
                 attn_window_size=(64, 64),
                 layerscale_gamma_init: float = 1.0):
        super().__init__()
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # STFT module
        self.stft = STFT(
            hop_length=hop_length,
            n_fft=n_fft,
            window_size=window_size
        )

        stft_dim = n_fft // 2 + 1
        self.conv = ConvDownsample(2 * stft_dim, dim, norm_eps=norm_eps)
        # self.norm = RMSNorm(dim)

        # self.patchify = Patchify1D(1, dim, hop_length)
        # self.conv = WNConv1dVarlen(dim, dim, kernel_size=3, stride=1, padding=1, causal=causal, bias=False)

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
                    layerscale_gamma_init=layerscale_gamma_init,
                )
        else:
            self.transformer_backbone_level1 = nn.Identity()
        if n_layers_level2 > 0:
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
                layerscale_gamma_init=layerscale_gamma_init,
            )
        else:
            self.transformer_backbone_level2 = nn.Identity()


        # Output projection
        if out_channels != dim:
            self.output_proj = nn.Linear(dim, out_channels) #nn.Conv1d(dim, out_channels, kernel_size=1)
        else:
            self.output_proj = nn.Identity()
        
        self.reset_parameters()

    def forward(self, x, position_ids=None, cu_seqlens=None, max_seqlen=None, level=1):
        # x = self.patchify(x)
        # x: (B, 1, T) - raw audio
        
        # STFT
        if level == 1:
            stft_result = self.stft(x)  # (B, n_fft//2+1, n_frames)
            x = torch.view_as_real(stft_result).permute(0, 2, 1, 3).flatten(2)
            # x = self.patchify(x.permute(0, 2, 1))    
            # x = self.norm(x)
            x = self.conv(x)

            x = self.transformer_backbone_level1(x)  # (B, n_frames, dim)

        elif level == 2:
            x = self.transformer_backbone_level2(x, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)  # (B, n_frames, dim)
            
            # Output projection
            x = self.output_proj(x)  # (B, n_frames, out_channels)
        
        return x

    def reset_parameters(self):
        self.apply(init_weights)