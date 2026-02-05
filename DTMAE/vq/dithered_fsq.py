"""
Dithered Finite Scalar Quantization
Adapted from TAAE (Stability AI) implementation:
https://github.com/Stability-AI/stable-codec

This module provides a dithered FSQ quantizer compatible with the DTMAE quantizer interface.

Key features:
- Symmetric quantization around origin
- Hybrid training: 50% uniform noise, 50% straight-through estimation
- Multi-level training for post-hoc flexibility
- Near-perfect codebook utilization
- Automatic train/eval mode switching

Reference:
    TAAE: "Scaling Transformers for Low-Bitrate High-Quality Speech Coding"
    Section 3.2: Discrete bottleneck
    Section 3.2.1: Post-training bottleneck modification

Usage:
    # model.train() → uses train_levels, train_num_residuals
    # model.eval()  → uses inference_levels, inference_num_residuals
"""

from typing import List, Optional, Tuple, Union
import math
import random

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

from einops import rearrange
import numpy as np


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class DitheredFSQ(nn.Module):
    """
    Dithered Finite Scalar Quantization module with automatic train/eval mode switching.
    
    Behavior automatically changes based on model mode:
    - model.train(): Uses train_levels (random sampling) and train_num_residuals
    - model.eval():  Uses inference_levels (per-dim) and inference_num_residuals
    
    Args:
        dim: Input feature dimension
        codebook_dim: Number of FSQ dimensions (default: 6)
        
        # Training settings (used when self.training=True)
        train_levels: List of level values to randomly sample during training (e.g., [17, 9, 5])
        train_num_residuals: Number of residual stages during training (default: 1, no residual)
        
        # Inference settings (used when self.training=False, i.e., after .eval())
        inference_levels: Level(s) for inference. Can be:
            - int: Same level for all dimensions (e.g., 5)
            - List[int]: Per-dimension levels (e.g., [5, 5, 5, 5, 5, 5])
            - None: Uses max(train_levels) for all dimensions
        inference_num_residuals: Number of residual stages at inference (default: 1)
        
        # Common settings
        num_codebooks: Number of parallel codebooks (default: 1)
        noise_dropout: Probability of using noise-based quantization vs STE (default: 0.5)
        scale: Scale factor for quantization (default: 1.0)
        channel_first: If True, expects input as (B, D, T), else (B, T, D) (default: False)
    
    Example config:
        quantizer:
          cls: DitheredFSQ
          params:
            dim: 256
            codebook_dim: 6
            # Training (model.train())
            train_levels: [17, 9, 5]
            train_num_residuals: 1
            # Inference (model.eval())
            inference_levels: [5, 5, 5, 5, 5, 5]
            inference_num_residuals: 2  # 700bps @ 25Hz
            # Common
            num_codebooks: 1
            noise_dropout: 0.5
            scale: 1.0
    
    BPS Calculation @25Hz:
        bps = 25 * num_residuals * ceil(log2(codebook_size))
        
        Examples:
        - inference_levels=[6,6,6,6,6,6], num_residuals=1 → 400bps
        - inference_levels=[5,5,5,5,5,5], num_residuals=2 → 700bps
        - inference_levels=[5,5,5,5,5,5], num_residuals=1 → 350bps
    """
    
    def __init__(
        self,
        dim: int,
        codebook_dim: int = 6,
        # Training settings
        train_levels: List[int] | None = None,
        train_num_residuals: int = 1,
        # Inference settings
        inference_levels: int | List[int] | None = None,
        inference_num_residuals: int = 1,
        # Common settings
        num_codebooks: int = 1,
        noise_dropout: float = 0.5,
        scale: float = 1.0,
        channel_first: bool = False,
    ):
        super().__init__()
        
        # Default train_levels if not provided
        if train_levels is None:
            train_levels = [17, 9, 5]

        if train_num_residuals != 1:
            raise ValueError(
                "Post-hoc residual FSQ per TAAE requires train_num_residuals=1. "
                "Use residual stages only at inference."
            )
        
        # Process inference_levels: convert to list of per-dimension levels
        if inference_levels is None:
            inference_levels_list = [max(train_levels)] * codebook_dim
        elif isinstance(inference_levels, int):
            inference_levels_list = [inference_levels] * codebook_dim
        else:
            inference_levels_list = list(inference_levels)
            assert len(inference_levels_list) == codebook_dim, \
                f"inference_levels list length ({len(inference_levels_list)}) must match codebook_dim ({codebook_dim})"

        if inference_num_residuals > 1:
            if len(set(inference_levels_list)) != 1:
                raise ValueError(
                    "Residual FSQ requires uniform inference_levels when using post-hoc decomposition."
                )
            level = inference_levels_list[0]
            if level < 3:
                raise ValueError("Residual FSQ requires levels >= 3.")
            intervals = level - 1
            if intervals & (intervals - 1) != 0:
                raise ValueError(
                    "Residual FSQ requires levels of the form L=2^n+1 (so L-1 is a power of two)."
                )
        
        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks
        self.noise_dropout = noise_dropout
        self.scale = scale
        self.channel_first = channel_first
        
        # Training settings
        self.train_levels = train_levels
        self.train_num_residuals = train_num_residuals
        
        # Inference settings
        self.inference_levels_list = inference_levels_list
        self.inference_num_residuals = inference_num_residuals
        
        # FSQ dimension = codebook_dim * num_codebooks
        self.fsq_dim = codebook_dim * num_codebooks
        
        # Compute codebook size based on inference_levels (for metrics)
        self.codebook_size = int(np.prod(inference_levels_list))
        
        # Input/output projections
        if dim != self.fsq_dim:
            self.in_proj = nn.Linear(dim, self.fsq_dim)
            self.out_proj = nn.Linear(self.fsq_dim, dim)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        
        # Register inference levels as buffer (per-dimension)
        _levels = torch.tensor(inference_levels_list, dtype=torch.int64)
        self.register_buffer("_levels", _levels, persistent=False)
        
        # Compute basis for index calculation: [1, L0, L0*L1, L0*L1*L2, ...]
        _basis = torch.cumprod(
            torch.tensor([1] + inference_levels_list[:-1], dtype=torch.int64), 
            dim=0
        )
        self.register_buffer("_basis", _basis, persistent=False)
        
        # Precompute half_l for inference (per-dimension)
        _half_l = self.scale * 2 / (_levels.float() - 1)
        self.register_buffer("_half_l_inference", _half_l, persistent=False)
        
        self.allowed_dtypes = (torch.float32, torch.float64)
    
    @property
    def num_residuals(self) -> int:
        """Returns num_residuals based on current mode (train/eval)."""
        return self.train_num_residuals if self.training else self.inference_num_residuals
    
    def _get_half_l_uniform(self, L: int, device: torch.device) -> Tensor:
        """Compute uniform half_l for a given level L (used in training)."""
        return torch.tensor(self.scale * 2 / (L - 1), device=device)
    
    def _scale_and_shift(self, z: Tensor, half_l: Tensor) -> Tensor:
        """Scale and shift z from [-scale, scale] to [0, L-1] range."""
        level_indices = (z + self.scale) / half_l
        return level_indices
    
    def _scale_and_shift_inverse(self, level_indices: Tensor, half_l: Tensor) -> Tensor:
        """Inverse of _scale_and_shift: from [0, L-1] to [-scale, scale]."""
        z = level_indices * half_l - self.scale
        return z
    
    def _quantize_inference(self, z: Tensor) -> Tensor:
        """
        Quantize tensor for inference with per-dimension levels.
        Handles both single-stage and residual decomposition.
        """
        num_residuals = self.inference_num_residuals
        
        if num_residuals == 1:
            # Single-stage quantization with per-dimension levels
            half_l = self._half_l_inference
            quantized = self._scale_and_shift_inverse(
                round_ste(self._scale_and_shift(z, half_l)),
                half_l
            )
        else:
            # Residual decomposition
            quantized = torch.zeros_like(z)
            residual = z
            
            for k in range(num_residuals):
                stage_scale = self.scale / (2 ** k)
                stage_half_l = stage_scale * 2 / (self._levels.float() - 1)
                
                level_indices = (residual + stage_scale) / stage_half_l
                level_indices = round_ste(level_indices)
                stage_quantized = level_indices * stage_half_l - stage_scale
                
                quantized = quantized + stage_quantized
                residual = residual - stage_quantized
        
        return quantized
    
    def _quantize_training(self, z: Tensor) -> Tensor:
        """
        Quantize tensor for training with multi-level dithering.
        Uses uniform level across all dimensions (randomly sampled).
        """
        batch_size = z.shape[0]
        
        # Randomly select level for this batch
        L = random.choice(self.train_levels)
        half_l = self._get_half_l_uniform(L, z.device)
        
        # Hybrid dithering approach (TAAE style): 50/50 noise vs STE when noise_dropout=0.5
        ste_quantized = self._scale_and_shift_inverse(
            round_ste(self._scale_and_shift(z, half_l)),
            half_l
        )

        noisy = z + (torch.rand_like(z) - 0.5) * half_l

        noise_mask = torch.bernoulli(
            torch.full([batch_size, 1, 1, 1], self.noise_dropout, device=z.device)
        ).bool().expand_as(z)

        quantized = torch.where(noise_mask, noisy, ste_quantized)

        return quantized
    
    def quantize(self, z: Tensor) -> Tensor:
        """
        Quantize the input tensor.
        
        Automatically uses:
        - Training mode: train_levels (random), train_num_residuals
        - Eval mode: inference_levels (per-dim), inference_num_residuals
        """
        z = torch.tanh(z)
        
        if self.training:
            return self._quantize_training(z)
        else:
            return self._quantize_inference(z)
    
    def _codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Convert quantized codes to flat indices (per-dimension levels)."""
        level_indices = self._scale_and_shift(zhat, self._half_l_inference)
        level_indices = level_indices.round().to(torch.int64)
        level_indices = level_indices.clamp(min=0)
        level_indices = torch.min(level_indices, self._levels - 1)
        out = (level_indices * self._basis).sum(dim=-1)
        return out
    
    def _codes_to_indices_residual(self, zhat: Tensor) -> Tensor:
        """Convert quantized codes to indices for residual decomposition."""
        all_indices = []
        residual = zhat
        num_residuals = self.num_residuals
        
        for k in range(num_residuals):
            stage_scale = self.scale / (2 ** k)
            stage_half_l = stage_scale * 2 / (self._levels.float() - 1)
            
            level_indices = (residual + stage_scale) / stage_half_l
            level_indices = level_indices.round().to(torch.int64)
            level_indices = level_indices.clamp(min=0)
            level_indices = torch.min(level_indices, self._levels - 1)
            
            stage_indices = (level_indices * self._basis).sum(dim=-1)
            all_indices.append(stage_indices)
            
            stage_quantized = level_indices.float() * stage_half_l - stage_scale
            residual = residual - stage_quantized
        
        return torch.stack(all_indices, dim=-1).flatten(start_dim=-2)
    
    def _indices_to_level_indices(self, indices: Tensor) -> Tensor:
        """Convert flat indices to per-dimension level indices."""
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered
    
    def _indices_to_codes(self, indices: Tensor, residual_stage: int = 0) -> Tensor:
        """Convert flat indices back to quantized codes (per-dimension levels)."""
        level_indices = self._indices_to_level_indices(indices)
        stage_scale = self.scale / (2 ** residual_stage)
        stage_half_l = stage_scale * 2 / (self._levels.float() - 1)
        codes = level_indices.float() * stage_half_l - stage_scale
        return codes
    
    @autocast(device_type="cuda", enabled=False)
    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor, Optional[List[Tensor]]]:
        """
        Forward pass of the DitheredFSQ quantizer.
        
        Behavior depends on mode:
        - train(): Uses train_levels, train_num_residuals
        - eval():  Uses inference_levels, inference_num_residuals
        
        Args:
            z: Input tensor of shape (B, T, D) or (B, D, T) if channel_first
               For varlen path: (total_tokens, D)
        
        Returns:
            quantized: Quantized output, same shape as input
            indices: Codebook indices
                - Single stage: shape (B, T, num_codebooks)
                - Residual: shape (B, T, num_codebooks * num_residuals)
            commit_loss: None (FSQ doesn't need commitment loss)
        """
        # Handle varlen path (2D input)
        is_varlen = (z.dim() == 2)
        if is_varlen:
            z = z.unsqueeze(0)
        
        # Handle channel_first format
        if self.channel_first:
            z = rearrange(z, 'b d t -> b t d')
        
        orig_dtype = z.dtype
        
        # Project to FSQ dimension
        z_proj = self.in_proj(z)
        
        # Reshape for multi-codebook
        z_proj = rearrange(z_proj, 'b t (c d) -> b t c d', c=self.num_codebooks)
        
        # Ensure correct dtype for quantization
        if z_proj.dtype not in self.allowed_dtypes:
            z_proj = z_proj.to(torch.float32)
        
        # Quantize (auto-switches based on self.training)
        codes = self.quantize(z_proj)
        
        # Get indices
        num_residuals = self.num_residuals
        if num_residuals == 1:
            indices = self._codes_to_indices(codes)
        else:
            z_tanh = torch.tanh(z_proj)
            indices = self._codes_to_indices_residual(z_tanh)
        
        # Reshape back
        codes = rearrange(codes, 'b t c d -> b t (c d)')
        
        # Project back to original dimension
        quantized = self.out_proj(codes)
        
        # Cast back to original dtype
        if quantized.dtype != orig_dtype:
            quantized = quantized.to(orig_dtype)
        
        # Handle channel_first format
        if self.channel_first:
            quantized = rearrange(quantized, 'b t d -> b d t')
        
        # Handle varlen path
        if is_varlen:
            quantized = quantized.squeeze(0)
            indices = indices.squeeze(0)
        
        return quantized, indices, None
    
    def vq2emb(self, vq: Tensor, proj: bool = True) -> Tensor:
        """
        Convert indices back to embeddings.
        
        Args:
            vq: Indices tensor
                - Single stage: shape (B, T) or (B, T, num_codebooks)
                - Residual: shape (B, T, num_codebooks * num_residuals)
            proj: Whether to apply output projection (default: True)
        
        Returns:
            Embeddings tensor of shape (B, T, D)
        """
        if vq.dim() == 2:
            vq = vq.unsqueeze(-1)
        
        num_residuals = self.num_residuals
        
        if num_residuals == 1:
            codes = self._indices_to_codes(vq.to(torch.int64))
            codes = rearrange(codes, 'b t c d -> b t (c d)')
        else:
            total_indices = self.num_codebooks * num_residuals
            assert vq.shape[-1] == total_indices, \
                f"Expected {total_indices} indices but got {vq.shape[-1]}"
            
            codes = torch.zeros(
                (*vq.shape[:-1], self.fsq_dim), 
                device=vq.device, dtype=torch.float32
            )
            
            for k in range(num_residuals):
                start_idx = k * self.num_codebooks
                end_idx = start_idx + self.num_codebooks
                stage_indices = vq[..., start_idx:end_idx]
                stage_codes = self._indices_to_codes(stage_indices.to(torch.int64), residual_stage=k)
                stage_codes = rearrange(stage_codes, 'b t c d -> b t (c d)')
                codes = codes + stage_codes
        
        if proj:
            codes = self.out_proj(codes)
        
        return codes
    
    def get_emb(self) -> Tensor:
        """
        Get the implicit codebook embeddings.
        
        For FSQ, this generates all possible quantization points.
        Warning: Can be large for big codebooks!
        
        Returns:
            Tensor of shape (codebook_size, codebook_dim)
        """
        indices = torch.arange(self.codebook_size, device=self._levels.device)
        codes = self._indices_to_codes(indices)
        return codes
    
    def get_bps(self, frame_rate: float = 25.0) -> float:
        """
        Calculate bits-per-second for current mode's configuration.
        
        Args:
            frame_rate: Frame rate in Hz (default: 25)
        
        Returns:
            Bits per second
        """
        num_residuals = self.num_residuals
        bits_per_frame = math.ceil(math.log2(self.codebook_size))
        bps = frame_rate * num_residuals * self.num_codebooks * bits_per_frame
        return bps
    
    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Public method to convert indices to codes (for compatibility)."""
        return self.vq2emb(indices, proj=False)
    
    def get_config_info(self) -> dict:
        """Get current configuration info for debugging."""
        return {
            'mode': 'train' if self.training else 'eval',
            'levels': self.train_levels if self.training else self.inference_levels_list,
            'num_residuals': self.num_residuals,
            'codebook_size': self.codebook_size,
            'bps_at_25hz': self.get_bps(25.0),
        }


# Backward compatibility alias
ResidualDitheredFSQ = DitheredFSQ
