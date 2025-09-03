import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Tuple, Dict, Any, Optional, Callable, Union, Sequence
from functools import partial
from common.spectral import stft, _get_window
from dataclasses import field
from common.conv_weightnorm import WNConv2d

class NLayerSpecDiscriminator(nnx.Module):
    """Multi-layer spectrogram discriminator (JAX/Flax NNX)."""
    
    def __init__(self, 
                in_channels: int = 1, 
                out_channels: int = 1,
                kernel_sizes: Tuple[int, int] = (5, 3),
                channels: int = 32,
                max_downsample_channels: int = 512,
                downsample_scales: Tuple[int, ...] = (2, 2, 2),
                rngs: nnx.Rngs = None):
        
        # Validate kernel sizes
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.max_downsample_channels = max_downsample_channels
        self.downsample_scales = downsample_scales
        
        # First layer
        self.layer_0 = WNConv2d(
            in_features=self.in_channels,
            out_features=self.channels,
            kernel_size=(self.kernel_sizes[0], self.kernel_sizes[0]),
            strides=(2, 2),
            padding=((self.kernel_sizes[0] // 2, self.kernel_sizes[0] // 2), 
                    (self.kernel_sizes[0] // 2, self.kernel_sizes[0] // 2)),
            rngs=rngs
        )
        # Downsampling conv layers
        self.down_layers = []
        in_chs = self.channels
        for i, downsample_scale in enumerate(self.downsample_scales):
            out_chs = min(in_chs * downsample_scale, self.max_downsample_channels)
            
            layer = WNConv2d(
                in_features=in_chs,
                out_features=out_chs,
                kernel_size=(downsample_scale * 2 + 1, downsample_scale * 2 + 1),
                strides=(downsample_scale, downsample_scale),
                padding=((downsample_scale, downsample_scale), 
                        (downsample_scale, downsample_scale)),
                rngs=rngs
            )
            self.down_layers.append(layer)
            # Alias names to mirror Torch's layer_i
            setattr(self, f"layer_{i + 1}", layer)
            in_chs = out_chs
        
        # Additional conv layer
        out_chs = min(in_chs * 2, self.max_downsample_channels)
        self.additional_layer = WNConv2d(
            in_features=in_chs,
            out_features=out_chs,
            kernel_size=(self.kernel_sizes[1], self.kernel_sizes[1]),
            strides=(1, 1),
            padding=((self.kernel_sizes[1] // 2, self.kernel_sizes[1] // 2), 
                    (self.kernel_sizes[1] // 2, self.kernel_sizes[1] // 2)),
            rngs=rngs
        )
        setattr(self, f"layer_{len(self.downsample_scales) + 1}", self.additional_layer)

        # Output conv layer
        self.output_layer = WNConv2d(
            in_features=out_chs,
            out_features=self.out_channels,
            kernel_size=(self.kernel_sizes[1], self.kernel_sizes[1]),
            strides=(1, 1),
            padding=((self.kernel_sizes[1] // 2, self.kernel_sizes[1] // 2), 
                    (self.kernel_sizes[1] // 2, self.kernel_sizes[1] // 2)),
            rngs=rngs
        )
        setattr(self, f"layer_{len(self.downsample_scales) + 2}", self.output_layer)
        
    def __call__(self, x):
        """
        Forward pass.
        Args:
            x (Array): input spectrogram (B, F, T, C) in channels-last format.
        Returns:
            list: list of each layer's outputs.
        """
        results = []
        
        # First layer
        x = self.layer_0(x)
        x = jax.nn.leaky_relu(x, negative_slope=0.2)
        results.append(x)
        
        # Downsampling layers
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
            results.append(x)
            # if jax.process_index() == 0:
            #     print(f"NLayerSpecDiscriminator conv_{i} out_chs: {x.shape}")
        
        # Additional layer
        x = self.additional_layer(x)
        x = jax.nn.leaky_relu(x, negative_slope=0.2)
        results.append(x)
        
        # Output layer
        x = self.output_layer(x)
        results.append(x)
        # if jax.process_index() == 0:
        #     print(f"NLayerSpecDiscriminator output out_chs: {x.shape}")
        return results


class SpecDiscriminator(nnx.Module):
    """Spectrogram discriminator (JAX/Flax NNX)."""
    
    def __init__(self, 
                stft_params: Dict[str, Any] = None,
                in_channels: int = 1,
                out_channels: int = 1,
                kernel_sizes: Tuple[int, int] = (7, 3),
                channels: int = 32,
                max_downsample_channels: int = 512,
                downsample_scales: Tuple[int, ...] = (2, 2, 2),
                rngs: nnx.Rngs = None):
        
        if stft_params is None:
            stft_params = {
                'fft_sizes': [1024, 2048, 512],
                'hop_sizes': [120, 240, 50],
                'win_lengths': [600, 1200, 240],
                'window': 'hann_window'
            }
        
        self.stft_params = stft_params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.max_downsample_channels = max_downsample_channels
        self.downsample_scales = downsample_scales
        
        # Build discriminators for each STFT config
        self.discriminators = []
        for i in range(len(self.stft_params['fft_sizes'])):
            disc = NLayerSpecDiscriminator(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_sizes=self.kernel_sizes,
                channels=self.channels,
                max_downsample_channels=self.max_downsample_channels,
                downsample_scales=self.downsample_scales,
                rngs=rngs
            )
            self.discriminators.append(disc)
            setattr(self, f"disc_{i}", disc)
    
    def __call__(self, x):
        """
        Forward pass.
        Args:
            x (Array): input audio (B, T) or (B, T, 1)
        Returns:
            List: list of discriminator outputs per STFT config.
        """
        results = []
        
        # If (B, T, 1) -> (B, T)
        if len(x.shape) == 3:
            x = x.squeeze(-1)  # (B, T)
        
        # For each STFT config, create spectrogram and apply discriminator
        for i, disc in enumerate(self.discriminators):
            fft_size = self.stft_params['fft_sizes'][i]
            win_length = self.stft_params['win_lengths'][i]
            hop_length = self.stft_params['hop_sizes'][i]
            
            # Match torch.stft default args: center=True, reflect pad, normalized=False, onesided=True
            window = _get_window(win_length, 'hann')
            
            spec_complex = stft(
                waveform=x.astype(jnp.float32),
                n_fft=fft_size,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                pad_mode="reflect",
                normalized=False
            )
            
            # Torch CP uses magnitude as sqrt(clamp(real^2+imag^2, 1e-7, 1e3))
            power = (spec_complex.real ** 2 + spec_complex.imag ** 2)
            power = jnp.clip(power, 1e-7, 1e3)
            spec = jnp.sqrt(power).astype(x.dtype)
            
            # (B, F, T) -> (B, F, T, 1) channels-last
            spec = spec[..., jnp.newaxis]
            
            results.append(disc(spec))
        
        return results