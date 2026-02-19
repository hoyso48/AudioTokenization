import torch
import torch.nn as nn
from typing import Optional

from vq.module import WNConv1d, WNConvTranspose1d


def _fixed_pattern_stride(r: float) -> int:
    keep_ratio = 1.0 - float(r)
    if keep_ratio <= 0.0:
        raise ValueError("FixedPatternMasking: r must be < 1.0")
    return max(1, int(round(1.0 / keep_ratio)))


def _build_fixed_pattern_mask(batch_size: int, seq_len: int, stride: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
    mask[:, ::stride] = True
    return mask


def _pack_with_mask(x: torch.Tensor, mask: torch.Tensor):
    if x.dim() != 3:
        raise ValueError("Downsampler expects dense x of shape [B, N, C]")
    device = x.device
    batch_size, _ = mask.shape
    channels = int(x.shape[-1])
    y_packed = x[mask].view(-1, channels)
    counts = mask.sum(dim=1).to(torch.long)
    cu_kept = torch.zeros(batch_size + 1, device=device, dtype=torch.long)
    cu_kept[1:] = torch.cumsum(counts, dim=0)
    keep_rank = mask.to(torch.long).cumsum(dim=1) - 1
    keep_rank = keep_rank.clamp_min(0)
    position_ids = keep_rank[mask].to(torch.long)
    max_seqlen = int(counts.max().item()) if batch_size > 0 else 0
    return y_packed, position_ids, cu_kept, max_seqlen


def _scatter_dense_kept_tokens(y: torch.Tensor, x_dense: torch.Tensor, mask: torch.Tensor, module_name: str) -> None:
    batch_size, _, channels = y.shape
    if x_dense.dim() != 3:
        raise ValueError(f"{module_name}: dense input must be [B, N_kept, C]")
    if int(x_dense.shape[0]) != int(batch_size) or int(x_dense.shape[2]) != int(channels):
        raise ValueError(f"{module_name}: shape mismatch between x and mask target")

    total_kept = int(mask.sum().item())
    if total_kept != int(x_dense.shape[0] * x_dense.shape[1]):
        raise ValueError(f"{module_name}: kept count from mask != dense token count")

    y[mask] = x_dense.reshape(-1, channels)


class FixedPatternMasking(nn.Module):
    """
    Fixed deterministic masking for non-DTP mode.
    Returns only mask [B, N].
    """

    def __init__(self, r: float, **kwargs):
        super().__init__()
        self.r = float(r)
        self.stride = _fixed_pattern_stride(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("FixedPatternMasking expects dense x of shape [B, N, C]")
        batch_size, seq_len, _ = x.shape
        return _build_fixed_pattern_mask(batch_size, seq_len, self.stride, x.device)


class TAAEConvDownsampler(nn.Module):
    """
    TAAE-style dense conv downsampler.
      - weight-normalized Conv1d
      - activation before strided conv (Identity by default)
      - input/output in [B, N, C]
    """

    def __init__(
        self,
        dim: int,
        stride: int = 2,
        activation: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        dim = int(dim)
        if dim <= 0:
            raise ValueError("TAAEConvDownsampler: dim must be positive")
        if int(stride) < 1:
            raise ValueError("TAAEConvDownsampler: stride must be >= 1")

        stride = int(stride)
        kernel_size = 2 * stride if stride > 1 else 1
        padding = stride // 2 + stride % 2 if stride > 1 else 0

        self.activation = nn.Identity() if activation is None else activation
        self.conv = WNConv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError("TAAEConvDownsampler expects dense x of shape [B, N, C]")
        x = x.transpose(1, 2)
        x = self.activation(x)
        x = self.conv(x)
        return x.transpose(1, 2)


class TAAEConvUpsampler(nn.Module):
    """
    TAAE-style dense conv upsampler.
      - weight-normalized ConvTranspose1d
      - activation before transposed conv (Identity by default)
      - input/output in [B, N, C]
    """

    def __init__(
        self,
        dim: int,
        stride: int = 2,
        activation: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        dim = int(dim)
        if dim <= 0:
            raise ValueError("TAAEConvUpsampler: dim must be positive")
        if int(stride) < 1:
            raise ValueError("TAAEConvUpsampler: stride must be >= 1")

        stride = int(stride)
        kernel_size = 2 * stride if stride > 1 else 1
        padding = stride // 2 + stride % 2 if stride > 1 else 0

        self.activation = nn.Identity() if activation is None else activation
        self.conv = WNConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError("TAAEConvUpsampler expects dense x of shape [B, N, C]")
        x = x.transpose(1, 2)
        x = self.activation(x)
        x = self.conv(x)
        return x.transpose(1, 2)


class RepeatUpsampler(nn.Module):
    def __init__(self, r: Optional[float] = None, **kwargs):
        super().__init__()
        self.stride = _fixed_pattern_stride(float(r)) if r is not None else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, cu_seqlens: torch.Tensor = None):
        """
        DTP mode:
          - x: [total_kept, C], mask: [B, N] -> y: [B, N, C]
        Non-DTP mode:
          - x: [B, N_kept, C], mask: [B, N] (or inferred by r) -> y: [B, N, C]
        """
        if x.dim() == 2:
            if mask is None:
                raise ValueError("RepeatUpsampler(varlen): mask is required")
            batch_size, seq_len = mask.shape
            channels = int(x.shape[-1])
            y = torch.zeros(batch_size, seq_len, channels, device=x.device, dtype=x.dtype)
            lin_idx = mask.view(-1).nonzero(as_tuple=False).squeeze(1)
            if int(lin_idx.numel()) != int(x.shape[0]):
                raise ValueError("RepeatUpsampler(varlen): kept count != mask trues")
            y.view(-1, channels)[lin_idx] = x
        elif x.dim() == 3:
            batch_size, n_kept, channels = x.shape
            if mask is None:
                if self.stride is None:
                    raise ValueError("RepeatUpsampler(dense): mask is required when r is not configured")
                mask = _build_fixed_pattern_mask(batch_size, n_kept * self.stride, self.stride, x.device)
            _, seq_len = mask.shape
            y = torch.zeros(batch_size, seq_len, channels, device=x.device, dtype=x.dtype)
            _scatter_dense_kept_tokens(y, x, mask, "RepeatUpsampler")
        else:
            raise ValueError("RepeatUpsampler expects x shape [total_kept, C] or [B, N_kept, C]")

        positions = torch.arange(y.shape[1], device=y.device).view(1, y.shape[1]).expand(y.shape[0], y.shape[1])
        keep_pos = torch.where(mask, positions, torch.full_like(positions, -1))
        src_pos = torch.cummax(keep_pos, dim=1).values.clamp(min=0)
        y = y.gather(1, src_pos.unsqueeze(-1).expand(-1, -1, y.shape[-1]))
        return y


class MaskUpsampler(nn.Module):
    def __init__(self, dim: int, r: Optional[float] = None, **kwargs):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(dim))
        self.stride = _fixed_pattern_stride(float(r)) if r is not None else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, cu_seqlens: torch.Tensor = None):
        """
        DTP mode:
          - x: [total_kept, C], mask: [B, N] -> y: [B, N, C]
        Non-DTP mode:
          - x: [B, N_kept, C], mask: [B, N] (or inferred by r) -> y: [B, N, C]
        """
        if x.dim() == 2:
            if mask is None:
                raise ValueError("MaskUpsampler(varlen): mask is required")
            batch_size, seq_len = mask.shape
            channels = int(x.shape[-1])
            if channels != int(self.mask_token.numel()):
                raise ValueError("MaskUpsampler: dim mismatch between x and mask_token")
            y = self.mask_token.to(device=x.device, dtype=x.dtype).view(1, 1, channels).expand(batch_size, seq_len, channels).clone()
            lin_idx = mask.view(-1).nonzero(as_tuple=False).squeeze(1)
            if int(lin_idx.numel()) != int(x.shape[0]):
                raise ValueError("MaskUpsampler(varlen): kept count != mask trues")
            y.view(-1, channels)[lin_idx] = x
            return y

        if x.dim() != 3:
            raise ValueError("MaskUpsampler expects x shape [total_kept, C] or [B, N_kept, C]")

        batch_size, n_kept, channels = x.shape
        if channels != int(self.mask_token.numel()):
            raise ValueError("MaskUpsampler: dim mismatch between x and mask_token")
        if mask is None:
            if self.stride is None:
                raise ValueError("MaskUpsampler(dense): mask is required when r is not configured")
            mask = _build_fixed_pattern_mask(batch_size, n_kept * self.stride, self.stride, x.device)

        _, seq_len = mask.shape
        y = self.mask_token.to(device=x.device, dtype=x.dtype).view(1, 1, channels).expand(batch_size, seq_len, channels).clone()
        _scatter_dense_kept_tokens(y, x, mask, "MaskUpsampler")
        return y


class FrontierDownsampler(nn.Module):
    def __init__(self, r: Optional[float] = None, use_varlen_path: Optional[bool] = None, **kwargs):
        super().__init__()
        self.stride = _fixed_pattern_stride(float(r)) if r is not None else None
        if use_varlen_path is not None and not isinstance(use_varlen_path, bool):
            raise ValueError("FrontierDownsampler: use_varlen_path must be bool or None")
        self.use_varlen_path = use_varlen_path

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        DTP mode:
          - default: returns [B, N_kept, C] when all per-sequence kept lengths are equal
          - otherwise: returns packed varlen (y_packed, position_ids, cu_seqlens, max_seqlen)
        Non-DTP mode:
          - x: [B, N, C] -> [B, N_kept, C] (fixed pattern)
        use_varlen_path:
          - True: always use packed varlen path when mask is provided
          - None/False: prefer fixed-len [B, N_kept, C] when lengths are uniform
        """
        if x.dim() != 3:
            raise ValueError("FrontierDownsampler expects dense x of shape [B, N, C]")

        if mask is None:
            if self.stride is None:
                raise ValueError("FrontierDownsampler(dense): r is required when mask is not provided")
            return x[:, ::self.stride, :]

        counts = mask.sum(dim=1).to(torch.long)
        uniform_kept_len = counts.numel() == 0 or bool(torch.all(counts == counts[0]))
        if uniform_kept_len and self.use_varlen_path is not True:
            if counts.numel() == 0:
                return x[:, :0, :]
            n_kept = int(counts[0].item())
            if n_kept == 0:
                return x[:, :0, :]
            return x[mask].view(x.shape[0], n_kept, x.shape[-1])

        return _pack_with_mask(x, mask)


class AverageDownsampler(nn.Module):
    def __init__(self, r: Optional[float] = None, **kwargs):
        super().__init__()
        self.stride = _fixed_pattern_stride(float(r)) if r is not None else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        DTP mode:
          - x: [B, N, C], mask: [B, N] -> (packed varlen, position_ids, cu_seqlens, max_seqlen)
        Non-DTP mode:
          - x: [B, N, C] -> [B, N_kept, C] (fixed-pattern pooled dense)
        """
        if x.dim() != 3:
            raise ValueError("AverageDownsampler expects dense x of shape [B, N, C]")

        if mask is None:
            if self.stride is None:
                raise ValueError("AverageDownsampler(dense): r is required when mask is not provided")
            batch_size, seq_len, channels = x.shape
            chunks = []
            for start in range(0, seq_len, self.stride):
                end = min(seq_len, start + self.stride)
                chunks.append(x[:, start:end, :].mean(dim=1, keepdim=True))
            if len(chunks) == 0:
                return x.new_zeros((batch_size, 0, channels))
            return torch.cat(chunks, dim=1)

        device = x.device
        batch_size, seq_len = mask.shape
        channels = int(x.shape[-1])

        group_id = mask.to(torch.long).cumsum(dim=1) - 1
        group_id = group_id.clamp_min(0)

        b_idx = torch.arange(batch_size, device=device).view(batch_size, 1).expand(batch_size, seq_len)
        key = (b_idx * seq_len + group_id).reshape(-1)
        x_flat = x.reshape(batch_size * seq_len, channels)

        sum_flat = torch.zeros(batch_size * seq_len, channels, device=device, dtype=x.dtype)
        cnt_flat = torch.zeros(batch_size * seq_len, device=device, dtype=x.dtype)
        sum_flat.index_add_(0, key, x_flat)
        cnt_flat.index_add_(0, key, torch.ones(batch_size * seq_len, device=device, dtype=x.dtype))

        sums = sum_flat.view(batch_size, seq_len, channels)
        cnts = cnt_flat.view(batch_size, seq_len).clamp_min(1e-12).unsqueeze(-1)
        group_means = sums / cnts

        if int(mask.sum().item()) == 0:
            y_packed = x.new_zeros((0, channels))
        else:
            b_sel, _ = mask.nonzero(as_tuple=True)
            keep_group = group_id[mask].to(torch.long)
            flat_idx = (b_sel.to(torch.long) * seq_len + keep_group).to(torch.long)
            y_packed = group_means.view(batch_size * seq_len, channels).index_select(0, flat_idx)

        counts = mask.sum(dim=1).to(torch.long)
        cu_kept = torch.zeros(batch_size + 1, device=device, dtype=torch.long)
        cu_kept[1:] = torch.cumsum(counts, dim=0)
        keep_rank = mask.to(torch.long).cumsum(dim=1) - 1
        keep_rank = keep_rank.clamp_min(0)
        position_ids = keep_rank[mask].to(torch.long)
        max_seqlen = int(counts.max().item()) if batch_size > 0 else 0
        return y_packed, position_ids, cu_kept, max_seqlen


class FixedPatternMaskingDownsampler(nn.Module):
    """
    Fixed deterministic downsampler (single input/output).
      in:  [B, N, C]
      out: [B, N_kept, C]
    """

    def __init__(self, r: float, **kwargs):
        super().__init__()
        self.stride = _fixed_pattern_stride(float(r))

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError("FixedPatternMaskingDownsampler expects dense x of shape [B, N, C]")
        return x[:, ::self.stride, :]


class FixedPatternMaskingUpsampler(nn.Module):
    """
    Fixed deterministic mask-token upsampler (single input/output).
      in:  [B, N_kept, C]
      out: [B, N, C]
    """

    def __init__(self, r: float, dim: int, **kwargs):
        super().__init__()
        self.stride = _fixed_pattern_stride(float(r))
        self.mask_token = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor):
        if x.dim() != 3:
            raise ValueError("FixedPatternMaskingUpsampler expects dense x of shape [B, N, C]")
        batch_size, n_kept, channels = x.shape
        if channels != int(self.mask_token.numel()):
            raise ValueError("FixedPatternMaskingUpsampler: dim mismatch between x and mask_token")

        seq_len = n_kept * self.stride
        mask = _build_fixed_pattern_mask(batch_size, seq_len, self.stride, x.device)
        y = self.mask_token.to(device=x.device, dtype=x.dtype).view(1, 1, channels).expand(batch_size, seq_len, channels).clone()
        _scatter_dense_kept_tokens(y, x, mask, "FixedPatternMaskingUpsampler")
        return y
