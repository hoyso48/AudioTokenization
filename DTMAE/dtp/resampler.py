import torch
import torch.nn as nn

class RepeatUpsampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cu_seqlens: torch.Tensor = None):
        """
        Varlen -> Dense upsampling.
        Inputs:
          - x: [total_kept, C]
          - mask: [B, N] (frontier indicator)
        Output:
          - y: [B, N, C]
        """
        device = x.device
        B, N = mask.shape

        if x.dim() != 2:
            raise ValueError("RepeatUpsampler expects varlen x of shape [total_kept, C]")
        C = int(x.shape[-1])
        y = torch.zeros(B, N, C, device=device, dtype=x.dtype)
        lin_idx = mask.view(-1).nonzero(as_tuple=False).squeeze(1)
        if int(lin_idx.numel()) != int(x.shape[0]):
            raise ValueError("RepeatUpsampler: kept count != mask trues")
        y.view(-1, C)[lin_idx] = x

        # Compute nearest-left frontier indices via cummax
        positions = torch.arange(N, device=device).view(1, N).expand(B, N)
        keep_pos = torch.where(mask, positions, torch.full_like(positions, -1))
        src_pos = torch.cummax(keep_pos, dim=1).values.clamp(min=0)
        y = y.gather(1, src_pos.unsqueeze(-1).expand(-1, -1, C))
        return y


class MaskUpsampler(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cu_seqlens: torch.Tensor = None):
        """
        Varlen -> Dense upsampling with a learned mask token.
        Inputs:
          - x: [total_kept, C]
          - mask: [B, N]
        Output y: [B, N, C]
        """
        device = x.device
        B, N = mask.shape
        if x.dim() != 2:
            raise ValueError("MaskUpsampler expects varlen x of shape [total_kept, C]")
        C = int(x.shape[-1])
        if C != int(self.mask_token.numel()):
            raise ValueError("MaskUpsampler: dim mismatch between x and mask_token")
        y = torch.empty(B, N, C, device=device, dtype=x.dtype)
        y[~mask] = self.mask_token.to(device=device, dtype=y.dtype)
        lin_idx = mask.view(-1).nonzero(as_tuple=False).squeeze(1)
        if int(lin_idx.numel()) != int(x.shape[0]):
            raise ValueError("MaskUpsampler: kept count != mask trues")
        y.view(-1, C)[lin_idx] = x

        return y

class FrontierDownsampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Dense -> Varlen downsampling (frontiers only).
        Inputs:
          - x: [B, N, C]
          - mask: [B, N]
        Returns:
          - y_packed: [total_kept, C]
          - cu_seqlens_kept: [B+1]
          - position_ids: [total_kept] per-sequence ranks 0..len_kept-1
          - max_seqlen: int
        """
        if x.dim() != 3:
            raise ValueError("FrontierDownsampler expects dense x of shape [B, N, C]")
        device = x.device
        B, N = mask.shape
        C = int(x.shape[-1])
        y_packed = x[mask].view(-1, C)
        counts = mask.sum(dim=1).to(torch.long)
        cu_kept = torch.zeros(B + 1, device=device, dtype=torch.long)
        cu_kept[1:] = torch.cumsum(counts, dim=0)
        keep_rank = mask.to(torch.long).cumsum(dim=1) - 1
        keep_rank = keep_rank.clamp_min(0)
        position_ids = keep_rank[mask].to(torch.long)
        max_seqlen = int(counts.max().item()) if B > 0 else 0
        return y_packed, position_ids, cu_kept, max_seqlen

class AverageDownsampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Dense -> Varlen downsampling.
        For each bin delimited by frontiers, average tokens within the bin and
        emit the pooled value at each frontier, packed across batch.

        Inputs:
          - x: [B, N, C]
          - mask: [B, N]
        Returns:
          - y_packed: [total_kept, C]
          - cu_seqlens_kept: [B+1]
          - position_ids: [total_kept] per-sequence ranks 0..len_kept-1
          - max_seqlen: int
        """
        if x.dim() != 3:
            raise ValueError("AverageDownsampler expects dense x of shape [B, N, C]")
        device = x.device
        B, N = mask.shape
        C = int(x.shape[-1])

        # Group id per token: cumulative count of frontiers - 1
        group_id = mask.to(torch.long).cumsum(dim=1) - 1
        group_id = group_id.clamp_min(0)

        # Flatten for scatter-add per (b, group)
        b_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)
        key = (b_idx * N + group_id).reshape(-1)
        x_flat = x.reshape(B * N, C)

        sum_flat = torch.zeros(B * N, C, device=device, dtype=x.dtype)
        cnt_flat = torch.zeros(B * N, device=device, dtype=x.dtype)

        sum_flat.index_add_(0, key, x_flat)
        cnt_flat.index_add_(0, key, torch.ones(B * N, device=device, dtype=x.dtype))

        sums = sum_flat.view(B, N, C)
        cnts = cnt_flat.view(B, N).clamp_min(1e-12).unsqueeze(-1)
        means = sums / cnts

        # Extract pooled values at frontier positions and pack
        y_packed = means[mask].view(-1, C)
        counts = mask.sum(dim=1).to(torch.long)
        cu_kept = torch.zeros(B + 1, device=device, dtype=torch.long)
        cu_kept[1:] = torch.cumsum(counts, dim=0)
        keep_rank = mask.to(torch.long).cumsum(dim=1) - 1
        keep_rank = keep_rank.clamp_min(0)
        position_ids = keep_rank[mask].to(torch.long)
        max_seqlen = int(counts.max().item()) if B > 0 else 0
        return y_packed, position_ids, cu_kept, max_seqlen