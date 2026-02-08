"""
Quantizers module for DTMAE.

This module provides a unified interface for different quantization methods,
allowing dynamic instantiation via Hydra config.

Available quantizers:
- ResidualVQ: Residual Vector Quantization
- DitheredFSQ: Dithered Finite Scalar Quantization (from TAAE)
                Supports multi-level training and post-hoc residual decomposition
- TAAEDitheredFSQ: Exact dithered FSQ class copied from official TAAE repo
- FSQ: Finite Scalar Quantization (from vector_quantize_pytorch)
- SimVQ: Simple Vector Quantization (from vector_quantize_pytorch)

Usage in config:
    quantizer:
      cls: DitheredFSQ
      params:
        dim: 256
        codebook_dim: 6
        train_levels: [17, 9, 5]    # Multi-level training
        inference_levels: 17         # Post-hoc adjustable
        num_residuals: 1             # 1=single, 2+=residual
        ...
"""

# Internal implementations
from .residual_vq import ResidualVQ
from .dithered_fsq import DitheredFSQ, TAAEDitheredFSQ

# External implementations (vector_quantize_pytorch)
from vector_quantize_pytorch import FSQ, SimVQ

__all__ = [
    'ResidualVQ',
    'DitheredFSQ',
    'TAAEDitheredFSQ',
    'FSQ',
    'SimVQ',
]
