# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch.nn as nn
from .resample import UpSample1d, DownSample1d

class Activation1d(nn.Module):
    def __init__(self,
                 activation,
                 antialias: bool = False,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12):
        super().__init__()
        self.antialias = antialias
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        if antialias:
            self.upsample = UpSample1d(up_ratio, up_kernel_size)
            self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        if self.antialias:
            x = self.upsample(x)
        x = self.act(x)
        if self.antialias:
            x = self.downsample(x)

        return x
