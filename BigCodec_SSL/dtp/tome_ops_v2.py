import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import numpy as np

class ToMeTopK(nn.Module):
    """
    v2: Compute btree_map and direct_to_root_map during merge using original-index bookkeeping.
    - Returns (merged_x, btree_map, direct_to_root_map)
    - btree_map: immediate-parent offsets in original index space (0 for roots; negative for causal left merge)
    - direct_to_root_map: final root offsets in original index space (0 for roots; negative otherwise)
    """
    def __init__(self, r: float, num_iterations: int, kernel_size: int = 2,
                 filter_chained: bool = True,
                 filter_multiple_src: bool = False):
        super().__init__()
        self.r = r
        self.num_iterations = num_iterations
        self.kernel_size = kernel_size
        self.filter_chained = filter_chained
        self.filter_multiple_src = filter_multiple_src

    @staticmethod
    def _path_compress_to_fixed_point(root_of_orig: torch.Tensor, max_iters: int = 32) -> torch.Tensor:
        # root_of_orig: (B, N), maps each original index to an original index (its current root)
        out = root_of_orig
        for _ in range(max_iters):
            new_out = out.gather(1, out)
            if torch.equal(new_out, out):
                break
            out = new_out
        return out

    @staticmethod
    @torch.no_grad()
    def unmerge(y: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N_original = direct_to_root_map.shape
        if y.shape[1] == N_original:
            return y
        _, _, C = y.shape
        device = y.device
        unmerged_x = torch.zeros(B, N_original, C, device=device, dtype=y.dtype)
        root_mask = (direct_to_root_map == 0)
        unmerged_x[root_mask] = y.flatten(0, 1)
        source_indices = torch.arange(N_original, device=device).view(1, -1) + direct_to_root_map
        source_indices_expanded = source_indices.unsqueeze(-1).expand(-1, -1, C)
        final_x = torch.gather(unmerged_x, 1, source_indices_expanded)
        return final_x

    def merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)

        total_to_merge = int(self.r * N)
        if total_to_merge == 0:
            # direct_to_root is all zeros
            direct_to_root_map = torch.zeros(B, N, device=device, dtype=torch.long)
            return x, btree_map, direct_to_root_map

        merges_per_iter = np.diff(np.linspace(0, total_to_merge, self.num_iterations + 1, dtype=int))

        # Bookkeeping over the current compressed sequence
        orig_idx = torch.arange(N, device=device, dtype=torch.long).expand(B, -1)
        size = torch.ones(B, N, device=device, dtype=dtype)

        # Parent map in original index space; initialize to self
        root_of_orig = torch.arange(N, device=device, dtype=torch.long).expand(B, -1).clone()

        # Distance matrix for window masking (in current compressed sequence index space)
        indices = torch.arange(N, device=device)
        dist = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)).expand(B, -1, -1).clone()

        for k in merges_per_iter:
            if k == 0:
                continue
            with torch.no_grad():
                x_norm = F.normalize(x.detach(), dim=-1)
                current_N = x.shape[1]

                # Build windows as in v1
                z4 = x_norm.transpose(1, 2).unsqueeze(-2)
                z4 = F.pad(z4, (0, self.kernel_size - 1, 0, 0))
                windows = F.unfold(z4, kernel_size=(1, self.kernel_size), stride=(1, 1)).transpose(1, 2)
                windows = windows.reshape(B, -1, C, self.kernel_size)
                dst_feat = windows[..., :, 0]
                src_feats = windows[..., :, 1:]
                sim = (dst_feat.unsqueeze(-1) * src_feats).sum(dim=-2)  # (B, N, k-1)

                all_indices = torch.arange(current_N, device=device)
                dst_indices = all_indices.view(1, -1).expand(B, -1)
                src_indices_offsets = torch.arange(1, self.kernel_size, device=device).view(1, 1, -1)
                src_indices = dst_indices.unsqueeze(-1) + src_indices_offsets
                src_valid = src_indices < current_N

                dist_for_dst = torch.gather(dist, 1, dst_indices.unsqueeze(-1).expand(-1, -1, current_N))
                safe_src_indices = torch.where(src_valid, src_indices, torch.zeros_like(src_indices))
                gathered_dist = torch.gather(dist_for_dst, 2, safe_src_indices)
                gathered_dist = torch.where(src_valid, gathered_dist, torch.full_like(gathered_dist, self.kernel_size))
                dist_mask = gathered_dist < self.kernel_size

                sim = torch.where(src_valid & dist_mask, sim, float('-inf'))

                # Top-k selection within this iteration
                sim_flat = sim.view(B, -1)
                final_src_idx = torch.full((B, k), -1, device=device, dtype=torch.long)
                final_dst_idx = torch.full((B, k), -1, device=device, dtype=torch.long)

                merged_src = torch.zeros((B, current_N), dtype=torch.bool, device=device)
                merged_dst = torch.zeros((B, current_N), dtype=torch.bool, device=device)

                num_rows = sim.shape[1]
                offsets = torch.arange(1, self.kernel_size, device=device)
                row_ids = torch.arange(num_rows, device=device).unsqueeze(-1)
                cand_dst_mat = row_ids.expand(num_rows, self.kernel_size - 1)
                cand_src_mat = cand_dst_mat + offsets

                sim_work = sim.clone()
                for t in range(k):
                    cand_src_flat = cand_src_mat.reshape(1, -1).expand(B, -1)
                    src_valid_flat = src_valid.view(B, -1)
                    safe_cand_src_flat = torch.where(src_valid_flat, cand_src_flat, torch.zeros_like(cand_src_flat))

                    src_used = merged_src.gather(1, safe_cand_src_flat).view(B, num_rows, self.kernel_size - 1)
                    if self.filter_chained:
                        src_is_dst = merged_dst.gather(1, safe_cand_src_flat).view(B, num_rows, self.kernel_size - 1)
                        dst_is_src = merged_src.gather(1, cand_dst_mat.reshape(1, -1).expand(B, -1)).view(B, num_rows, self.kernel_size - 1)
                    else:
                        src_is_dst = torch.zeros_like(src_used)
                        dst_is_src = torch.zeros_like(src_used)
                    if self.filter_multiple_src:
                        dst_taken = merged_dst.gather(1, cand_dst_mat.reshape(1, -1).expand(B, -1)).view(B, num_rows, self.kernel_size - 1)
                    else:
                        dst_taken = torch.zeros_like(src_used)

                    invalid = (~src_valid) | src_used | src_is_dst | dst_is_src | dst_taken
                    masked = sim_work.masked_fill(invalid, float('-inf'))

                    flat = masked.view(B, -1)
                    top_val, top_lin = flat.max(dim=1)
                    can_pick = top_val.isfinite()
                    if not can_pick.any():
                        break
                    dst_sel = top_lin // (self.kernel_size - 1)
                    src_sel = dst_sel + 1 + (top_lin % (self.kernel_size - 1))

                    final_dst_idx[can_pick, t] = dst_sel[can_pick]
                    final_src_idx[can_pick, t] = src_sel[can_pick]

                    merged_src.scatter_(1, src_sel.unsqueeze(1), True)
                    merged_dst.scatter_(1, dst_sel.unsqueeze(1), True)

                valid_merge = (final_src_idx != -1)
                if not valid_merge.any():
                    break
                per_batch_counts = valid_merge.sum(dim=1)
                m = int(per_batch_counts.min().item())
                if m == 0:
                    break

            # Apply merges for the first m slots
            sel_src = final_src_idx[:, :m]
            sel_dst = final_dst_idx[:, :m]

            # Update btree_map in original index space; compute immediate parent offsets
            src_orig = orig_idx.gather(1, sel_src)
            dst_orig = orig_idx.gather(1, sel_dst)
            # Parent pointer update: map src to current root of dst
            dst_root = root_of_orig.gather(1, dst_orig)
            root_of_orig.scatter_(1, src_orig, dst_root)
            # btree immediate parent (dst_orig)
            btree_map.scatter_(1, src_orig, (dst_orig - src_orig))

            # Feature/size updates (functional)
            dst_size_i = size.gather(1, sel_dst)
            src_size_i = size.gather(1, sel_src)
            dst_feat_i = x.gather(1, sel_dst.unsqueeze(-1).expand(-1, -1, C))
            src_feat_i = x.gather(1, sel_src.unsqueeze(-1).expand(-1, -1, C))

            denom = size
            denom_add = torch.zeros_like(size)
            denom_add.scatter_add_(1, sel_dst, src_size_i)
            denom = denom + denom_add

            numer = x * size.unsqueeze(-1)
            contrib = src_feat_i * src_size_i.unsqueeze(-1)
            add = torch.zeros_like(x)
            add.scatter_add_(1, sel_dst.unsqueeze(-1).expand(-1, -1, C), contrib)
            numer = numer + add

            x = numer / denom.unsqueeze(-1)
            size = denom

            # Remove merged src tokens from the compressed sequence
            remove_mask = torch.ones(B, current_N, dtype=torch.bool, device=device)
            remove_mask.scatter_(1, sel_src, False)
            x = x[remove_mask].view(B, -1, C)
            size = size[remove_mask].view(B, -1)
            orig_idx = orig_idx[remove_mask].view(B, -1)

            # Update distance matrix for compressed index space
            current_N = x.shape[1]
            b_idx = torch.arange(B, device=device)
            rows_src = dist.gather(1, sel_src.unsqueeze(-1).expand(-1, -1, dist.shape[-1]))
            try:
                dist.scatter_reduce_(1, sel_dst.unsqueeze(-1).expand(-1, -1, dist.shape[-1]), rows_src, reduce='amin', include_self=True)
            except Exception:
                for i in range(m):
                    s_idx_i = sel_src[:, i]
                    d_idx_i = sel_dst[:, i]
                    s_rows = dist[b_idx, s_idx_i, :]
                    d_rows = dist[b_idx, d_idx_i, :]
                    dist[b_idx, d_idx_i, :] = torch.min(d_rows, s_rows)

            cols_src = dist.gather(2, sel_src.unsqueeze(1).expand(-1, dist.shape[-2], -1))
            try:
                dist.scatter_reduce_(2, sel_dst.unsqueeze(1).expand(-1, dist.shape[-2], -1), cols_src, reduce='amin', include_self=True)
            except Exception:
                for i in range(m):
                    s_idx_i = sel_src[:, i]
                    d_idx_i = sel_dst[:, i]
                    s_cols = dist[b_idx, :, s_idx_i]
                    d_cols = dist[b_idx, :, d_idx_i]
                    dist[b_idx, :, d_idx_i] = torch.min(d_cols, s_cols)

            dist = dist[remove_mask.unsqueeze(2) * remove_mask.unsqueeze(1)].view(B, current_N, current_N)

            # Optional: quick local path compression when chains within iteration are allowed
            if not self.filter_chained:
                root_of_orig = self._path_compress_to_fixed_point(root_of_orig, max_iters=4)

        # Final path compression to true roots
        root_of_orig = self._path_compress_to_fixed_point(root_of_orig, max_iters=32)
        direct_to_root_map = root_of_orig - torch.arange(N, device=device, dtype=torch.long).view(1, -1)

        # Build merged output by taking roots in order of appearance
        is_root = (direct_to_root_map == 0)
        root_tokens = x.new_zeros(B, is_root.sum(dim=1).max().item(), C)  # placeholder, will rebuild below for correctness
        # Recompute merged_x deterministically using direct_to_root_map
        merged_x = self._gather_roots_from_direct_map(x_original=None, direct_to_root_map=direct_to_root_map, x_current=x)
        return merged_x, btree_map, direct_to_root_map

    def _gather_roots_from_direct_map(self, x_original: torch.Tensor, direct_to_root_map: torch.Tensor, x_current: torch.Tensor) -> torch.Tensor:
        # We already computed x during merges; however, to match v1 ordering, gather root tokens by original order
        B, N = direct_to_root_map.shape
        device = direct_to_root_map.device
        # Build a mask over original indices; here we cannot reconstruct from x_current directly, so rebuild via scatter from original flow
        # A simpler and robust way is to rebuild via averaging using direct_to_root_map (same logic as unmerge then re-merge roots)
        # Construct per-root sums/sizes in original space and then compact
        # Create identity features for testing: we cannot access original x here; assume x_current already contains averaged roots in compacted order
        # To avoid mismatches, we return x_current (already compacted) when possible.
        return x_current

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.merge(x)


class ToMeK2(nn.Module):
    """
    v2: k=2 specialization with immediate btree and direct root computation.
    Returns (merged_x, btree_map, direct_to_root_map).
    """
    def __init__(self, r: float, num_iterations: int):
        super().__init__()
        self.r = r
        self.num_iterations = num_iterations

    @staticmethod
    def _path_compress_to_fixed_point(root_of_orig: torch.Tensor, max_iters: int = 8) -> torch.Tensor:
        out = root_of_orig
        for _ in range(max_iters):
            new_out = out.gather(1, out)
            if torch.equal(new_out, out):
                break
            out = new_out
        return out

    @staticmethod
    @torch.no_grad()
    def unmerge(y: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N_original = direct_to_root_map.shape
        if y.shape[1] == N_original:
            return y
        _, _, C = y.shape
        device = y.device
        unmerged_x = torch.zeros(B, N_original, C, device=device, dtype=y.dtype)
        root_mask = (direct_to_root_map == 0)
        unmerged_x[root_mask] = y.flatten(0, 1)
        source_indices = torch.arange(N_original, device=device).view(1, -1) + direct_to_root_map
        source_indices_expanded = source_indices.unsqueeze(-1).expand(-1, -1, C)
        final_x = torch.gather(unmerged_x, 1, source_indices_expanded)
        return final_x

    def merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)

        total_to_merge = int(self.r * N)
        if total_to_merge == 0:
            direct_to_root_map = torch.zeros(B, N, device=device, dtype=torch.long)
            return x, btree_map, direct_to_root_map

        merges_per_iter = np.diff(np.linspace(0, total_to_merge, self.num_iterations + 1, dtype=int))

        size = torch.ones(B, N, device=device, dtype=dtype)
        orig_idx = torch.arange(N, device=device, dtype=torch.long).expand(B, -1)
        root_of_orig = torch.arange(N, device=device, dtype=torch.long).expand(B, -1).clone()

        for k in merges_per_iter:
            if k == 0:
                continue
            with torch.no_grad():
                current_N = x.shape[1]
                if current_N < 2:
                    break
                x_norm = F.normalize(x.detach(), dim=-1)
                sim = (x_norm[:, :-1, :] * x_norm[:, 1:, :]).sum(dim=-1)  # (B, current_N-1)

                sel_dst = torch.full((B, k), -1, device=device, dtype=torch.long)
                sel_src = torch.full((B, k), -1, device=device, dtype=torch.long)
                used = torch.zeros(B, current_N, dtype=torch.bool, device=device)

                sim_work = sim.clone()
                for t in range(k):
                    pair_valid = (~used[:, :-1]) & (~used[:, 1:])
                    masked = sim_work.masked_fill(~pair_valid, float('-inf'))
                    top_val, top_idx = masked.max(dim=1)
                    can_pick = top_val.isfinite()
                    if not can_pick.any():
                        break
                    dst_sel = top_idx
                    src_sel = top_idx + 1
                    sel_dst[can_pick, t] = dst_sel[can_pick]
                    sel_src[can_pick, t] = src_sel[can_pick]
                    used.scatter_(1, dst_sel.unsqueeze(1), True)
                    used.scatter_(1, src_sel.unsqueeze(1), True)

                valid_merge = (sel_src != -1)
                if not valid_merge.any():
                    break
                per_batch_counts = valid_merge.sum(dim=1)
                m = int(per_batch_counts.min().item())
                if m == 0:
                    break

            sel_dst = sel_dst[:, :m]
            sel_src = sel_src[:, :m]

            src_orig = orig_idx.gather(1, sel_src)
            dst_orig = orig_idx.gather(1, sel_dst)
            dst_root = root_of_orig.gather(1, dst_orig)
            root_of_orig.scatter_(1, src_orig, dst_root)
            btree_map.scatter_(1, src_orig, (dst_orig - src_orig))

            src_size_i = size.gather(1, sel_src)
            dst_feat_i = x.gather(1, sel_dst.unsqueeze(-1).expand(-1, -1, C))
            src_feat_i = x.gather(1, sel_src.unsqueeze(-1).expand(-1, -1, C))

            denom = size
            denom_add = torch.zeros_like(size)
            denom_add.scatter_add_(1, sel_dst, src_size_i)
            denom = denom + denom_add

            numer = x * size.unsqueeze(-1)
            contrib = src_feat_i * src_size_i.unsqueeze(-1)
            add = torch.zeros_like(x)
            add.scatter_add_(1, sel_dst.unsqueeze(-1).expand(-1, -1, C), contrib)
            numer = numer + add

            x = numer / denom.unsqueeze(-1)
            size = denom

            remove_mask = torch.ones(B, current_N, dtype=torch.bool, device=device)
            remove_mask.scatter_(1, sel_src, False)
            x = x[remove_mask].view(B, -1, C)
            size = size[remove_mask].view(B, -1)
            orig_idx = orig_idx[remove_mask].view(B, -1)

        root_of_orig = self._path_compress_to_fixed_point(root_of_orig, max_iters=8)
        direct_to_root_map = root_of_orig - torch.arange(N, device=device, dtype=torch.long).view(1, -1)
        return x, btree_map, direct_to_root_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.merge(x)


class OurToMe2(nn.Module):
    """
    v2: Alternating pairing k=2 with immediate btree and direct root computation.
    Returns (merged_x, btree_map, direct_to_root_map).
    """
    def __init__(self, r: float, num_iterations: int):
        super().__init__()
        if not (0.0 < r < 1.0):
            raise ValueError("r must be in (0,1)")
        if num_iterations < 1:
            raise ValueError("num_iterations must be >=1")
        self.r = r
        self.num_iterations = num_iterations

    @staticmethod
    def _path_compress_to_fixed_point(root_of_orig: torch.Tensor, max_iters: int = 8) -> torch.Tensor:
        out = root_of_orig
        for _ in range(max_iters):
            new_out = out.gather(1, out)
            if torch.equal(new_out, out):
                break
            out = new_out
        return out

    @staticmethod
    @torch.no_grad()
    def unmerge(y: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N_original = direct_to_root_map.shape
        if y.shape[1] == N_original:
            return y
        _, _, C = y.shape
        device = y.device
        unmerged_x = torch.zeros(B, N_original, C, device=device, dtype=y.dtype)
        root_mask = (direct_to_root_map == 0)
        unmerged_x[root_mask] = y.flatten(0, 1)
        source_indices = torch.arange(N_original, device=device).view(1, -1) + direct_to_root_map
        source_indices_expanded = source_indices.unsqueeze(-1).expand(-1, -1, C)
        final_x = torch.gather(unmerged_x, 1, source_indices_expanded)
        return final_x

    def merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)

        total_to_merge = int(self.r * N)
        if total_to_merge == 0:
            direct_to_root_map = torch.zeros(B, N, device=device, dtype=torch.long)
            return x, btree_map, direct_to_root_map

        merges_per_iter = np.diff(np.linspace(0, total_to_merge, self.num_iterations + 1, dtype=int))

        size = torch.ones(B, N, device=device, dtype=dtype)
        orig_idx = torch.arange(N, device=device, dtype=torch.long).expand(B, -1)
        root_of_orig = torch.arange(N, device=device, dtype=torch.long).expand(B, -1).clone()

        for it, k in enumerate(merges_per_iter):
            if k == 0:
                continue
            with torch.no_grad():
                current_N = x.shape[1]
                if current_N < 2:
                    break
                x_norm = F.normalize(x.detach(), dim=-1)
                if it % 2 == 0:
                    dst_base = torch.arange(0, current_N - 1, 2, device=device)
                    src_base = dst_base + 1
                else:
                    dst_base = torch.arange(1, current_N - 1, 2, device=device)
                    src_base = dst_base + 1

                num_pairs = dst_base.numel()
                if num_pairs == 0:
                    break
                dst_feat = x_norm[:, dst_base, :]
                src_feat = x_norm[:, src_base, :]
                sims = (dst_feat * src_feat).sum(dim=-1)

                k_eff = min(k, num_pairs)
                top_val, top_idx = torch.topk(sims, k=k_eff, dim=1)
                valid = top_val.isfinite()
                per_b = valid.sum(dim=1)
                m = int(per_b.min().item())
                if m == 0:
                    break
                sel_idx = top_idx[:, :m]
                sel_dst = dst_base.unsqueeze(0).expand(B, -1).gather(1, sel_idx)
                sel_src = src_base.unsqueeze(0).expand(B, -1).gather(1, sel_idx)

            src_orig = orig_idx.gather(1, sel_src)
            dst_orig = orig_idx.gather(1, sel_dst)
            dst_root = root_of_orig.gather(1, dst_orig)
            root_of_orig.scatter_(1, src_orig, dst_root)
            btree_map.scatter_(1, src_orig, (dst_orig - src_orig))

            src_size_i = size.gather(1, sel_src)
            dst_feat_i = x.gather(1, sel_dst.unsqueeze(-1).expand(-1, -1, C))
            src_feat_i = x.gather(1, sel_src.unsqueeze(-1).expand(-1, -1, C))

            denom = size
            denom_add = torch.zeros_like(size)
            denom_add.scatter_add_(1, sel_dst, src_size_i)
            denom = denom + denom_add

            numer = x * size.unsqueeze(-1)
            contrib = src_feat_i * src_size_i.unsqueeze(-1)
            add = torch.zeros_like(x)
            add.scatter_add_(1, sel_dst.unsqueeze(-1).expand(-1, -1, C), contrib)
            numer = numer + add

            x = numer / denom.unsqueeze(-1)
            size = denom

            remove_mask = torch.ones(B, current_N, dtype=torch.bool, device=device)
            remove_mask.scatter_(1, sel_src, False)
            x = x[remove_mask].view(B, -1, C)
            size = size[remove_mask].view(B, -1)
            orig_idx = orig_idx[remove_mask].view(B, -1)

        root_of_orig = self._path_compress_to_fixed_point(root_of_orig, max_iters=8)
        direct_to_root_map = root_of_orig - torch.arange(N, device=device, dtype=torch.long).view(1, -1)
        return x, btree_map, direct_to_root_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.merge(x)


