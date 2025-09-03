import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence
from functools import partial
from ..common.conv_weightnorm import WNConv2d

class HiFiGANPeriodDiscriminator(nnx.Module):
    """HiFiGAN period discriminator (JAX/Flax NNX)."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        period: int = 3,
        kernel_sizes: Tuple[int, int] = (5, 3),
        channels: int = 32,
        downsample_scales: Tuple[int, ...] = (3, 3, 3, 3, 1),
        channel_increasing_factor: int = 4,
        max_downsample_channels: int = 1024,
        nonlinear_activation_params: Dict[str, float] = None,
        rngs: Optional[nnx.Rngs] = None
    ):
        super().__init__()
        
        if nonlinear_activation_params is None:
            nonlinear_activation_params = {"negative_slope": 0.1}
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.period = period
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.downsample_scales = downsample_scales
        self.channel_increasing_factor = channel_increasing_factor
        self.max_downsample_channels = max_downsample_channels
        self.nonlinear_activation_params = nonlinear_activation_params
        
        assert len(self.kernel_sizes) == 2
        assert self.kernel_sizes[0] % 2 == 1, "커널 크기는 홀수여야 합니다."
        assert self.kernel_sizes[1] % 2 == 1, "커널 크기는 홀수여야 합니다."
        
        in_chs = self.in_channels
        out_chs = self.channels
        
        # Downsampling conv stack (store actual module objects)
        self.conv_layers = []
        for i, downsample_scale in enumerate(self.downsample_scales):
            layer = WNConv2d(
                in_features=in_chs,
                out_features=out_chs,
                kernel_size=(self.kernel_sizes[0], 1),
                strides=(downsample_scale, 1),
                padding=(((self.kernel_sizes[0] - 1) // 2, (self.kernel_sizes[0] - 1) // 2), (0, 0)),
                rngs=rngs
            )
            self.conv_layers.append(layer)
            
            in_chs = out_chs
            out_chs = min(out_chs * self.channel_increasing_factor, self.max_downsample_channels)
        # Alias to match Torch child name for weight copy
        self.convs = self.conv_layers

        # Output convolution (align kernel/padding with Torch: kernel_sizes[1] - 1)
        self.output_conv = WNConv2d(
            in_features=in_chs,
            out_features=self.out_channels,
            kernel_size=(self.kernel_sizes[1] - 1, 1),
            strides=(1, 1),
            padding=(((self.kernel_sizes[1] - 1) // 2, (self.kernel_sizes[1] - 1) // 2), (0, 0)),
            rngs=rngs
        )
        # self.output_conv = WeightNorm(self.output_conv)
    def __call__(self, x):
        """
        Forward pass.
        Args:
            x (Array): input tensor (B, T, C), channels-last.
        Returns:
            list: list of intermediate tensors including final flattened output.
        """
        # Input is (B, T, C)
        b, t, c = x.shape
        
        # Pad time dimension to match period
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = jnp.pad(x, ((0, 0), (0, n_pad), (0, 0)), mode='reflect')
            t += n_pad
        
        # (B, T, C) -> (B, T//period, period, C)
        x = x.reshape(b, t // self.period, self.period, c)
        
        outs = []

        # Apply conv stack
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            
            # LeakyReLU
            x = jax.nn.leaky_relu(x, negative_slope=self.nonlinear_activation_params["negative_slope"])
            
            # if jax.process_index() == 0:
            #     print(f"MPD conv_{i} out shape: {x.shape}")

            outs.append(x)
        
        x = self.output_conv(x)
        
        # Flatten
        x = x.reshape(b, -1)
        outs.append(x)
        
        return outs


class HiFiGANMultiPeriodDiscriminator(nnx.Module):
    """HiFiGAN multi-period discriminator (JAX/Flax NNX)."""
    
    def __init__(
        self,
        periods: Tuple[int, ...] = (2, 3, 5, 7, 11),
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Tuple[int, int] = (5, 3),
        channels: int = 32,
        downsample_scales: Tuple[int, ...] = (3, 3, 3, 3, 1),
        channel_increasing_factor: int = 4,
        max_downsample_channels: int = 1024,
        nonlinear_activation_params: Dict[str, float] = None,
        rngs: Optional[nnx.Rngs] = None
    ):
        super().__init__()
        
        if nonlinear_activation_params is None:
            nonlinear_activation_params = {"negative_slope": 0.1}
            
        self.periods = periods
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.downsample_scales = downsample_scales
        self.channel_increasing_factor = channel_increasing_factor
        self.max_downsample_channels = max_downsample_channels
        self.nonlinear_activation_params = nonlinear_activation_params
        
        # Build per-period discriminators
        self.discriminators = []
        for i, period in enumerate(self.periods):
            disc = HiFiGANPeriodDiscriminator(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                period=period,
                kernel_sizes=self.kernel_sizes,
                channels=self.channels,
                downsample_scales=self.downsample_scales,
                channel_increasing_factor=self.channel_increasing_factor,
                max_downsample_channels=self.max_downsample_channels,
                nonlinear_activation_params=self.nonlinear_activation_params,
                rngs=rngs
            )
            self.discriminators.append(disc)
    
    def __call__(self, x):
        """
        Forward pass.
        Args:
            x (Array): (B, T, C) or (B, T)
        Returns:
            List: list of per-period discriminator outputs
        """
        outs = []
        
        # If (B, T), add channel dim
        if len(x.shape) == 2:
            x = x[..., jnp.newaxis]  # (B, T, 1)
        
        # Apply each period disc
        for discriminator in self.discriminators:
            outs.append(discriminator(x))
        
        return outs