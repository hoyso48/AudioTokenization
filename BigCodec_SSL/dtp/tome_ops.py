import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable, Dict, Optional
import math

import numpy as np

class ToMeChained(nn.Module):
    """
    A ToMe layer that merges tokens based on a sliding window approach.

    This implementation aims to be a highly flexible and efficient version of token merging,
    incorporating ideas from ToMe, ToMeSD, and a custom audio-focused version.
    The core logic for resolving merge chains and unmerging is adapted from a robust
    reference implementation.
    """
    def __init__(self, r: float = 0.5, kernel_size: int = 2):
        """
        Initializes the module.

        Args:
            r (float): The ratio of tokens to reduce. Must be between 0.0 and 1.0.
            kernel_size (int): The size of the causal sliding window for merging.
                               A token can merge into another token within this window.
                               Must be at least 2.
        """
        super().__init__()
        if not (0.0 <= r <= 1.0):
            raise ValueError("r must be between 0.0 and 1.0")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2.")

        self.r = r
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        """
        Applies the merging process to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (B, N, C).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
                - The merged tensor.
                - The merge_btree tensor in linked-list format.
                - The unmerge function.
        """
        # merged_x, btree = self.merge(x)
        merge_btree = self._create_merge_btree(x)
        # btree = self._convert_root_map_to_btree(direct_to_root_map)
        direct_to_root_map = self._resolve_chains(merge_btree)
        merged_x, _avg_sim = self.merge(x, direct_to_root_map)

        def unmerge_fn(y: torch.Tensor) -> torch.Tensor:
            return self.unmerge(y, direct_to_root_map)

        return merged_x, merge_btree, unmerge_fn

    @torch.no_grad()
    def _create_merge_btree(self, metric: torch.Tensor) -> torch.Tensor:
        """
        Calculates pairwise similarities, selects merge candidates, and creates the one-step merge map.
        This version uses a full similarity matrix for simplicity and correctness, at the cost of memory.
        """
        B, N, C = metric.shape
        device = metric.device

        num_tokens_to_merge = int(self.r * N)
        if num_tokens_to_merge == 0:
            return torch.zeros(B, N, dtype=torch.long, device=device)

        # metric = metric / metric.norm(dim=-1, keepdim=True)

        metric_a = metric.unsqueeze(2).expand(-1, -1, N, -1)
        metric_b = metric.unsqueeze(1).expand(-1, N, -1, -1)
        sim_matrix = torch.nn.functional.cosine_similarity(metric_a, metric_b, dim=-1)

        indices = torch.arange(N, device=device)
        src_indices = indices.view(1, N, 1)
        dst_indices = indices.view(1, 1, N)
        
        causal_mask = dst_indices < src_indices
        kernel_mask = (src_indices - dst_indices) < self.kernel_size
        identity_mask = dst_indices != src_indices
        final_mask = causal_mask & kernel_mask & identity_mask
        
        sim_matrix.masked_fill_(~final_mask, -torch.inf)

        best_sim_for_src, best_dst_for_src = torch.max(sim_matrix, dim=2)
        
        k = num_tokens_to_merge
        
        invalid_mask = torch.isinf(best_sim_for_src)
        best_sim_for_src[invalid_mask] = -torch.inf
        
        _, top_k_src_indices = torch.topk(best_sim_for_src, k=k, dim=1)

        num_valid_candidates = (~invalid_mask).sum(dim=1)
        num_merges_to_perform = min(k, num_valid_candidates.min().item()) if k > 0 else 0
        
        actual_top_k_src = top_k_src_indices[:, :num_merges_to_perform]
        
        merge_btree = torch.zeros(B, N, dtype=torch.long, device=device)

        if num_merges_to_perform > 0:
            top_k_dst_indices = best_dst_for_src.gather(1, actual_top_k_src)
            offsets = top_k_dst_indices - actual_top_k_src
            merge_btree.scatter_(1, actual_top_k_src, offsets)

        return merge_btree

    @staticmethod
    @torch.no_grad()
    def _resolve_chains(merge_btree: torch.Tensor) -> torch.Tensor:
        """
        Converts a partial merge map (showing one-step merges) to a full merge map
        that points every merged token directly to its final root token.
        """
        B, N = merge_btree.shape
        device = merge_btree.device
        
        direct_to_root_map = merge_btree.clone()
        
        b_idx = torch.arange(B, device=device).view(B, 1)
        arange_N = torch.arange(N, device=device).view(1, N)

        for _ in range(N):
            current_dest = arange_N + direct_to_root_map
            next_hop_offsets = merge_btree[b_idx, current_dest]
            needs_update = next_hop_offsets != 0
            if not needs_update.any():
                break
            direct_to_root_map += next_hop_offsets
        
        return direct_to_root_map

    @staticmethod
    @torch.no_grad()
    def _convert_root_map_to_btree(direct_to_root_map: torch.Tensor) -> torch.Tensor:
        """
        Converts a map where each token points directly to its final root
        into a b-tree like map where each token points to its immediate parent.
        """
        B, N = direct_to_root_map.shape
        device = direct_to_root_map.device

        arange_b_n = torch.arange(N, device=device).expand(B, -1)
        dst_indices_b_n = arange_b_n + direct_to_root_map
        sort_key = dst_indices_b_n * N + arange_b_n
        
        sorted_order = torch.argsort(sort_key, dim=1)
        original_indices_sorted = arange_b_n.gather(1, sorted_order)
        prev_indices_in_list = torch.roll(original_indices_sorted, shifts=1, dims=1)
        is_root_sorted = direct_to_root_map.gather(1, sorted_order) == 0
        
        parent_indices_sorted = torch.where(
            is_root_sorted,
            original_indices_sorted,
            prev_indices_in_list
        )
        
        btree_values_sorted = parent_indices_sorted - original_indices_sorted
        inverse_sort_order = torch.argsort(sorted_order, dim=1)
        merge_btree = btree_values_sorted.gather(1, inverse_sort_order)
        return merge_btree

    def merge(self, x: torch.Tensor, direct_to_root_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merges tokens based on similarity, resolves merge chains, and returns
        the merged tensor and the direct-to-root mapping.
        """
        B, N, C = x.shape
        size = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)
        
        root_indices = torch.arange(N, device=x.device).expand(B, -1) + direct_to_root_map
        
        merged_x = torch.zeros_like(x)
        merged_size = torch.zeros_like(size)
        
        root_indices_expanded = root_indices.unsqueeze(-1).expand(-1, -1, C)
        merged_x.scatter_add_(1, root_indices_expanded, x * size)
        merged_size.scatter_add_(1, root_indices.unsqueeze(-1), size)
        
        merged_x = merged_x / (merged_size + 1e-8)
        
        is_final_root = (direct_to_root_map == 0)
        
        # This part is tricky. We need to gather the root tokens in a predictable order.
        # Let's gather all root tokens and then select the right number.
        root_tokens = merged_x[is_final_root].view(B, -1, C)

        # Compute min similarity of used merges per batch using immediate-parent links
        with torch.no_grad():
            btree = self._convert_root_map_to_btree(direct_to_root_map)
            offsets = btree
            has_parent = offsets != 0
            if has_parent.any():
                indices = torch.arange(N, device=x.device).view(1, N)
                parent_indices = (indices + offsets).clamp_min(0)
                x_norm = F.normalize(x, dim=-1)
                dst_vecs = x_norm.gather(1, parent_indices.unsqueeze(-1).expand(-1, -1, C))
                sims = (x_norm * dst_vecs).sum(dim=-1)
                sims = sims.masked_fill(~has_parent, float('inf'))
                min_vals = sims.amin(dim=1)
                has_any = has_parent.any(dim=1)
                avg_sim = torch.where(has_any, min_vals.to(x.dtype), torch.zeros(B, device=x.device, dtype=x.dtype))
            else:
                avg_sim = torch.zeros(B, device=x.device, dtype=x.dtype)

        return root_tokens, avg_sim

    @staticmethod
    def unmerge(y: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the original tensor from the merged tensor and a direct-to-root merge map.
        """
        B, N_original = direct_to_root_map.shape
        if y.shape[1] == N_original: # No merging was done
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


class GeneralizedToMeMaskingUpsampler(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.num_embeddings = kernel_size - 1
        self.embedding = nn.Embedding(self.num_embeddings, dim)

    def forward(self, metric: torch.Tensor, merge_btree: torch.Tensor) -> torch.Tensor:
        """
        Upsamples the merged tensor back to the original sequence length.
        Root tokens are filled from the input metric, while merged tokens
        are filled from a learned embedding based on the merge offset.
        This implementation is fully vectorized.
        """
        B, N_original, C = metric.shape[0], merge_btree.shape[1], metric.shape[2]
        device = metric.device

        # Create the output tensor.
        out = torch.zeros(B, N_original, C, device=device, dtype=metric.dtype)

        # 1. Vectorized placement of root tokens (where merge_btree == 0)
        root_mask = (merge_btree == 0)
        num_roots = root_mask.sum()
        if num_roots > 0:
            # The metric tensor (B, R, C) contains the root tokens in order.
            # We reshape it and place it into the output tensor at the root locations.
            out[root_mask] = metric.reshape(-1, C)

        # 2. Vectorized placement of merged tokens from embeddings
        merged_mask = ~root_mask
        if merged_mask.any():
            # Get the non-zero offset values from the merge_btree
            offsets = merge_btree[merged_mask]

            # Map these offsets to valid embedding indices
            embedding_indices = -offsets - 1
            
            embedding_indices = embedding_indices.long()
            
            # Retrieve the learned embeddings and place them
            # print(embedding_indices.unique())
            learned_vectors = self.embedding(embedding_indices).to(metric.dtype)
            out[merged_mask] = learned_vectors

        return out


class ToMeGreedy(nn.Module):
    def __init__(self, r: float, kernel_size: int = 2):
        super().__init__()
        self.r = r
        self.kernel_size = kernel_size
        
    def merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)
        num_to_merge = int(self.r * N)
        indices = torch.arange(N, device=device)
        dist = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)).expand(B, -1, -1).clone()
        size = torch.ones(B, N, device=device, dtype=dtype)
        batch_idx = torch.arange(B)
        orig_idx = torch.arange(N, device=device).expand(B, -1)

        min_sim = torch.full((B,), float('inf'), device=device, dtype=dtype)
        has_merge = torch.zeros(B, device=device, dtype=torch.bool)
        merges_done = 0
        for i in range(num_to_merge):
            
            with torch.no_grad():
                x_norm = F.normalize(x, dim=-1)
                current_N = x.shape[1]
                windows = F.unfold(x_norm.transpose(1, 2).unsqueeze(-2), kernel_size=(1, self.kernel_size), stride=(1, 1)).transpose(1, 2)
                windows = windows.reshape(B, -1, C, self.kernel_size)
            
                dst_feat = windows[..., :, 0]
                src_feats = windows[..., :, 1:]
                sim = (dst_feat.unsqueeze(-1) * src_feats).sum(dim=-2)
                
                # --- Start of fix ---
                # The original unfold logic for dist_mask is incorrect because `dist` is modified
                # in each iteration, breaking the assumptions of fixed strides.
                # We need to gather the distances corresponding to the (dst, src) pairs used in `sim`.
                
                current_N = x.shape[1]
                # Indices for all tokens currently in the sequence
                all_indices = torch.arange(current_N, device=device)

                # Indices for `dst` tokens. These are the first N - kernel_size + 1 tokens.
                # Shape: (B, N-k+1)
                dst_indices = all_indices[:current_N - self.kernel_size + 1].view(1, -1).expand(B, -1)

                # For each `dst` token, get the indices of the `src` tokens in its window.
                # Shape: (B, N-k+1, k-1)
                src_indices_offsets = torch.arange(1, self.kernel_size, device=device).view(1, 1, -1)
                src_indices = dst_indices.unsqueeze(-1) + src_indices_offsets
                
                # Use gather to get the distances from the `dist` tensor.
                # `dist` has shape (B, current_N, current_N)
                # We need to get dist[b, dst, src] for all pairs.
                
                # Expand dst_indices to gather along the second dimension of dist
                # Shape: (B, N-k+1, current_N)
                dist_for_dst = torch.gather(dist, 1, dst_indices.unsqueeze(-1).expand(-1, -1, current_N))
                
                # Now gather the src distances from the result
                # Shape: (B, N-k+1, k-1)
                gathered_dist = torch.gather(dist_for_dst, 2, src_indices)

                dist_mask = gathered_dist < self.kernel_size
                # --- End of fix ---

                if not torch.any(dist_mask):
                    print("No further valid merges found, iteration stopped at", i)
                    break
                sim = torch.where(dist_mask, sim, float('-inf'))
                sim_flat = sim.view(B, -1)
                top_val, candidate_idx = sim_flat.max(dim=1)
                can_pick = top_val.isfinite()
                # track per-batch min similarity across merges performed
                min_sim = torch.where(can_pick, torch.minimum(min_sim, top_val.to(dtype)), min_sim)
                has_merge = has_merge | can_pick
                dst_idx = candidate_idx // (self.kernel_size - 1)
                src_idx = dst_idx + 1 + candidate_idx % (self.kernel_size - 1)
                # dst_orig_idx = orig_idx.gather(1, dst_idx.unsqueeze(-1))
                src_orig_idx = orig_idx.gather(1, src_idx.unsqueeze(-1))
                src_vals = dist.gather(2, src_idx.view(B, 1, 1).expand(B, current_N, 1)).squeeze(-1)
                dst_vals = dist.gather(2, dst_idx.view(B, 1, 1).expand(B, current_N, 1)).squeeze(-1)

                remove_mask = torch.ones(B, current_N, dtype=torch.bool, device=device)
                remove_mask.scatter_(1, src_idx.unsqueeze(1), False)

                dst_size = size.gather(1, dst_idx.unsqueeze(-1))
                src_size = size.gather(1, src_idx.unsqueeze(-1))
                dst_feat = x.gather(1, dst_idx.view(B, 1, 1).expand(B, 1, C))
                src_feat = x.gather(1, src_idx.view(B, 1, 1).expand(B, 1, C))
                link = - dist[batch_idx, src_idx.unsqueeze(-1), dst_idx.unsqueeze(-1)]
                btree_map.scatter_(1, src_orig_idx, link)
                dist.scatter_(dim=2, index=dst_idx.view(B, 1, 1).expand(B, current_N, 1), src=torch.min(dst_vals, src_vals).unsqueeze(-1))
            size.scatter_(1, dst_idx.unsqueeze(-1), dst_size + src_size)
            x.scatter_(1, dst_idx.view(B, 1, 1).expand(B, 1, C), (dst_feat * dst_size + src_feat * src_size) / (dst_size + src_size))

            x = x[remove_mask].view(B, -1, C)
            orig_idx = orig_idx[remove_mask].view(B, -1)
            size = size[remove_mask].view(B, -1)
            dist = dist[remove_mask.unsqueeze(2)*remove_mask.unsqueeze(1)].view(B, current_N-1, current_N-1)
            merges_done += 1
            
        if not has_merge.any():
            avg_sim = torch.zeros(B, device=device, dtype=dtype)
        else:
            avg_sim = torch.where(has_merge, min_sim, torch.zeros_like(min_sim))
        return x, btree_map, avg_sim

    @staticmethod
    @torch.no_grad()
    def btree_to_root_map(merge_btree: torch.Tensor) -> torch.Tensor:
        B, N = merge_btree.shape
        device = merge_btree.device
        
        direct_to_root_map = merge_btree.clone()
        
        b_idx = torch.arange(B, device=device).view(B, 1)
        arange_N = torch.arange(N, device=device).view(1, N)

        for _ in range(N):
            current_dest = arange_N + direct_to_root_map
            next_hop_offsets = merge_btree[b_idx, current_dest]
            needs_update = next_hop_offsets != 0
            if not needs_update.any():
                break
            direct_to_root_map += next_hop_offsets
        
        return direct_to_root_map

    @staticmethod
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


class ToMeTopK(nn.Module):
    def __init__(self, r: float, num_iterations: int, kernel_size: int = 2,
                 filter_chained: bool = True,
                 filter_multiple_src: bool = False):
        super().__init__()
        self.r = r
        self.num_iterations = num_iterations
        self.kernel_size = kernel_size
        self.filter_chained = filter_chained
        self.filter_multiple_src = filter_multiple_src
    
    def merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)
        
        total_to_merge = int(self.r * N)
        if total_to_merge == 0:
            return x, btree_map, torch.zeros(B, device=device, dtype=dtype)

        merges_per_iter = np.diff(np.linspace(0, total_to_merge, self.num_iterations + 1, dtype=int))

        indices = torch.arange(N, device=device)
        dist = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)).expand(B, -1, -1).clone()
        size = torch.ones(B, N, device=device, dtype=dtype)
        orig_idx = torch.arange(N, device=device).expand(B, -1)

        min_sim = torch.full((B,), float('inf'), device=device, dtype=dtype)
        total_selected = 0
        for k in merges_per_iter:
            if k == 0:
                continue

            with torch.no_grad():
                # Compute selection features from a detached view to avoid creating graph on selection
                x_norm = F.normalize(x.detach(), dim=-1)
                current_N = x.shape[1]

                # --- Similarity and Distance Masking (same as ToMe) ---
                # Right-pad so every real token can be dst;
                z4 = x_norm.transpose(1, 2).unsqueeze(-2)
                z4 = F.pad(z4, (0, self.kernel_size - 1, 0, 0))
                windows = F.unfold(z4, kernel_size=(1, self.kernel_size), stride=(1, 1)).transpose(1, 2)
                windows = windows.reshape(B, -1, C, self.kernel_size)
                dst_feat = windows[..., :, 0]
                src_feats = windows[..., :, 1:]
                sim = (dst_feat.unsqueeze(-1) * src_feats).sum(dim=-2)

                all_indices = torch.arange(current_N, device=device)
                # Allow all N positions as dst
                dst_indices = all_indices.view(1, -1).expand(B, -1)
                src_indices_offsets = torch.arange(1, self.kernel_size, device=device).view(1, 1, -1)
                src_indices = dst_indices.unsqueeze(-1) + src_indices_offsets  # (B, N, k-1)
                src_valid = src_indices < current_N
                
                dist_for_dst = torch.gather(dist, 1, dst_indices.unsqueeze(-1).expand(-1, -1, current_N))
                safe_src_indices = torch.where(src_valid, src_indices, torch.zeros_like(src_indices))
                gathered_dist = torch.gather(dist_for_dst, 2, safe_src_indices)
                # Invalidate padded src
                gathered_dist = torch.where(src_valid, gathered_dist, torch.full_like(gathered_dist, self.kernel_size))
                dist_mask = gathered_dist < self.kernel_size
                
                sim = torch.where(src_valid & dist_mask, sim, float('-inf'))
                sim_flat = sim.view(B, -1)
                
                # --- Top-k Selection via k masked top-1 steps ---
                final_src_idx = torch.full((B, k), -1, device=device, dtype=torch.long)
                final_dst_idx = torch.full((B, k), -1, device=device, dtype=torch.long)
                
                merged_src = torch.zeros((B, current_N), dtype=torch.bool, device=device)
                merged_dst = torch.zeros((B, current_N), dtype=torch.bool, device=device)
                b_idx = torch.arange(B, device=device)
                
                # Precompute candidate dst/src matrices
                num_rows = sim.shape[1]
                offsets = torch.arange(1, self.kernel_size, device=device)
                row_ids = torch.arange(num_rows, device=device).unsqueeze(-1)
                cand_dst_mat = row_ids.expand(num_rows, self.kernel_size - 1)  # (R, k-1)
                cand_src_mat = cand_dst_mat + offsets  # (R, k-1)
                
                sim_work = sim.clone()
                vals_per_slot = torch.full((B, k), float('-inf'), device=device, dtype=sim.dtype) if k > 0 else None
                for t in range(k):
                    # Build invalid mask based on current merged_src/dst
                    cand_src_flat = cand_src_mat.reshape(1, -1).expand(B, -1)  # (B, R*(k-1))
                    src_valid_flat = src_valid.view(B, -1)
                    safe_cand_src_flat = torch.where(src_valid_flat, cand_src_flat, torch.zeros_like(cand_src_flat))
                    src_used = merged_src.gather(1, safe_cand_src_flat).view(B, num_rows, self.kernel_size - 1)
                    # chained filters
                    if self.filter_chained:
                        src_is_dst = merged_dst.gather(1, safe_cand_src_flat).view(B, num_rows, self.kernel_size - 1)
                        dst_is_src = merged_src.gather(1, cand_dst_mat.reshape(1, -1).expand(B, -1)).view(B, num_rows, self.kernel_size - 1)
                    else:
                        src_is_dst = torch.zeros_like(src_used)
                        dst_is_src = torch.zeros_like(src_used)
                    # multiple-src per dst filter
                    if self.filter_multiple_src:
                        dst_taken = merged_dst.gather(1, cand_dst_mat.reshape(1, -1).expand(B, -1)).view(B, num_rows, self.kernel_size - 1)
                    else:
                        dst_taken = torch.zeros_like(src_used)
                    # also invalidate originally padded src positions (src_valid)
                    src_valid_rows = src_valid  # (B, N, k-1), broadcast ok
                    invalid = (~src_valid_rows) | src_used | src_is_dst | dst_is_src | dst_taken
                    masked = sim_work.masked_fill(invalid, float('-inf'))

                    # Pick top-1 per batch
                    flat = masked.view(B, -1)
                    top_val, top_lin = flat.max(dim=1)
                    can_pick = top_val.isfinite()
                    if not can_pick.any():
                        break
                    # Decode to (dst, src)
                    dst_sel = top_lin // (self.kernel_size - 1)
                    src_sel = dst_sel + 1 + (top_lin % (self.kernel_size - 1))

                    # Write selected indices for batches that can pick
                    final_dst_idx[can_pick, t] = dst_sel[can_pick]
                    final_src_idx[can_pick, t] = src_sel[can_pick]
                    vals_per_slot[can_pick, t] = top_val[can_pick]

                    # Update merged masks
                    merged_src.scatter_(1, src_sel.unsqueeze(1), True)
                    merged_dst.scatter_(1, dst_sel.unsqueeze(1), True)
 
                # Valid mask per slot (avoid flattening across batch)
                valid_merge = (final_src_idx != -1)
                if not valid_merge.any():
                    break
                # Ensure equal removals across batch by taking the minimum valid count
                per_batch_counts = valid_merge.sum(dim=1)
                m = int(per_batch_counts.min().item())
                if m == 0:
                    break
                # accumulate similarity of actually applied merges (first m slots)
                if m > 0:
                    step_min = vals_per_slot[:, :m].min(dim=1).values
                    min_sim = torch.minimum(min_sim, step_min.to(dtype))
                    total_selected += m
 
            # --- Batched Merge per fixed number m of slots ---
            remove_mask = torch.ones(B, current_N, dtype=torch.bool, device=device)
            b_idx = torch.arange(B, device=device)
            # Gather first m slots
            sel_src = final_src_idx[:, :m]  # (B, m)
            sel_dst = final_dst_idx[:, :m]  # (B, m)

            # btree_map update
            src_orig_idx_i = orig_idx.gather(1, sel_src)
            link_i = -dist.gather(1, sel_src.unsqueeze(-1).expand(-1, -1, current_N)).gather(2, sel_dst.unsqueeze(-1)).squeeze(-1)
            btree_map.scatter_(1, src_orig_idx_i, link_i)

            # dist row/col updates using scatter_reduce_ if available
            rows_src = dist.gather(1, sel_src.unsqueeze(-1).expand(-1, -1, current_N))  # (B, m, N)
            try:
                dist.scatter_reduce_(1, sel_dst.unsqueeze(-1).expand(-1, -1, current_N), rows_src, reduce='amin', include_self=True)
            except Exception:
                # Fallback per slot
                for i in range(m):
                    s_idx_i = sel_src[:, i]
                    d_idx_i = sel_dst[:, i]
                    s_rows = dist[b_idx, s_idx_i, :]
                    d_rows = dist[b_idx, d_idx_i, :]
                    dist[b_idx, d_idx_i, :] = torch.min(d_rows, s_rows)

            cols_src = dist.gather(2, sel_src.unsqueeze(1).expand(-1, current_N, -1))  # (B, N, m)
            try:
                dist.scatter_reduce_(2, sel_dst.unsqueeze(1).expand(-1, current_N, -1), cols_src, reduce='amin', include_self=True)
            except Exception:
                for i in range(m):
                    s_idx_i = sel_src[:, i]
                    d_idx_i = sel_dst[:, i]
                    s_cols = dist[b_idx, :, s_idx_i]
                    d_cols = dist[b_idx, :, d_idx_i]
                    dist[b_idx, :, d_idx_i] = torch.min(d_cols, s_cols)

            # features and sizes updates (functional, differentiable)
            dst_size_i = size.gather(1, sel_dst)
            src_size_i = size.gather(1, sel_src)
            dst_feat_i = x.gather(1, sel_dst.unsqueeze(-1).expand(-1, -1, C))
            src_feat_i = x.gather(1, sel_src.unsqueeze(-1).expand(-1, -1, C))

            # Denominator: start from current sizes and add src sizes into their dsts
            denom = size
            denom_add = torch.zeros_like(size)
            denom_add.scatter_add_(1, sel_dst, src_size_i)
            denom = denom + denom_add

            # Numerator: start from x*size and add src contributions into dst locations
            numer = x * size.unsqueeze(-1)
            contrib = src_feat_i * src_size_i.unsqueeze(-1)
            add = torch.zeros_like(x)
            add.scatter_add_(1, sel_dst.unsqueeze(-1).expand(-1, -1, C), contrib)
            numer = numer + add

            x = numer / denom.unsqueeze(-1)
            size = denom

            # mark src for removal
            remove_mask.scatter_(1, sel_src, False)
 
            x = x[remove_mask].view(B, -1, C)
            size = size[remove_mask].view(B, -1)
            orig_idx = orig_idx[remove_mask].view(B, -1)
            dist = dist[remove_mask.unsqueeze(2) * remove_mask.unsqueeze(1)].view(B, x.shape[1], x.shape[1])

        if total_selected == 0:
            avg_sim = torch.zeros(B, device=device, dtype=dtype)
        else:
            avg_sim = min_sim
        return x, btree_map, avg_sim
        
    @staticmethod
    @torch.no_grad()
    def btree_to_root_map(merge_btree: torch.Tensor) -> torch.Tensor:
        B, N = merge_btree.shape
        device = merge_btree.device
        
        direct_to_root_map = merge_btree.clone()
        
        b_idx = torch.arange(B, device=device).view(B, 1)
        arange_N = torch.arange(N, device=device).view(1, N)

        for _ in range(N):
            current_dest = arange_N + direct_to_root_map
            next_hop_offsets = merge_btree[b_idx, current_dest]
            needs_update = next_hop_offsets != 0
            if not needs_update.any():
                break
            direct_to_root_map += next_hop_offsets
        
        return direct_to_root_map

    @staticmethod
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


class ToMeK2(nn.Module):
    """
    Efficient special-case of ToMeTopK for kernel_size==2 with filter_chained=True.
    - Only adjacent pairs (i, i+1) are considered
    - No chain within an iteration (non-overlapping pairs)
    - No dist/unfold bookkeeping
    - Outputs btree_map with values in {0, -1}
    """
    def __init__(self, r: float, num_iterations: int):
        super().__init__()
        self.r = r
        self.num_iterations = num_iterations

    def merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)

        total_to_merge = int(self.r * N)
        if total_to_merge == 0:
            return x, btree_map, torch.zeros(B, device=device, dtype=dtype)

        merges_per_iter = np.diff(np.linspace(0, total_to_merge, self.num_iterations + 1, dtype=int))

        size = torch.ones(B, N, device=device, dtype=dtype)
        orig_idx = torch.arange(N, device=device).expand(B, -1)

        min_sim = torch.full((B,), float('inf'), device=device, dtype=dtype)
        total_selected = 0
        for k in merges_per_iter:
            if k == 0:
                continue

            with torch.no_grad():
                current_N = x.shape[1]
                # Similarity between adjacent tokens
                x_norm = F.normalize(x.detach(), dim=-1)
                sim = (x_norm[:, :-1, :] * x_norm[:, 1:, :]).sum(dim=-1)  # (B, current_N-1)

                # Greedy non-overlapping selection: pick top-1 per step, k steps
                sel_dst = torch.full((B, k), -1, device=device, dtype=torch.long)
                sel_src = torch.full((B, k), -1, device=device, dtype=torch.long)

                # Token usage mask to prevent chains and multiple usage within the iteration
                used = torch.zeros(B, current_N, dtype=torch.bool, device=device)

                sim_work = sim.clone()
                vals_per_slot = torch.full((B, k), float('-inf'), device=device, dtype=sim.dtype) if k > 0 else None
                for t in range(k):
                    # Invalidate pairs touching already used tokens
                    pair_valid = (~used[:, :-1]) & (~used[:, 1:])
                    masked = sim_work.masked_fill(~pair_valid, float('-inf'))

                    # Top-1 per batch
                    top_val, top_idx = masked.max(dim=1)
                    can_pick = top_val.isfinite()
                    if not can_pick.any():
                        break

                    dst_sel = top_idx
                    src_sel = top_idx + 1

                    sel_dst[can_pick, t] = dst_sel[can_pick]
                    sel_src[can_pick, t] = src_sel[can_pick]
                    vals_per_slot[can_pick, t] = top_val[can_pick]

                    # Mark tokens as used
                    used.scatter_(1, dst_sel.unsqueeze(1), True)
                    used.scatter_(1, src_sel.unsqueeze(1), True)

                # Equalize across batch
                valid_merge = (sel_src != -1)
                if not valid_merge.any():
                    break
                per_batch_counts = valid_merge.sum(dim=1)
                m = int(per_batch_counts.min().item())
                if m == 0:
                    break
                step_min = vals_per_slot[:, :m].min(dim=1).values
                min_sim = torch.minimum(min_sim, step_min.to(dtype))
                total_selected += m

            # Apply merges (vectorized)
            sel_dst = sel_dst[:, :m]
            sel_src = sel_src[:, :m]

            # Update btree_map (offset is always -1 for adjacent merge)
            src_orig_idx = orig_idx.gather(1, sel_src)
            btree_map.scatter_(1, src_orig_idx, torch.full_like(src_orig_idx, -1))

            # Update features and sizes in a differentiable, functional way
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

            # Remove merged src tokens
            remove_mask = torch.ones(B, current_N, dtype=torch.bool, device=device)
            remove_mask.scatter_(1, sel_src, False)
            x = x[remove_mask].view(B, -1, C)
            size = size[remove_mask].view(B, -1)
            orig_idx = orig_idx[remove_mask].view(B, -1)

        if total_selected == 0:
            avg_sim = torch.zeros(B, device=device, dtype=dtype)
        else:
            avg_sim = min_sim
        return x, btree_map, avg_sim

    @staticmethod
    @torch.no_grad()
    def btree_to_root_map(merge_btree: torch.Tensor) -> torch.Tensor:
        B, N = merge_btree.shape
        device = merge_btree.device
        
        direct_to_root_map = merge_btree.clone()
        
        b_idx = torch.arange(B, device=device).view(B, 1)
        arange_N = torch.arange(N, device=device).view(1, N)

        for _ in range(N):
            current_dest = arange_N + direct_to_root_map
            next_hop_offsets = merge_btree[b_idx, current_dest]
            needs_update = next_hop_offsets != 0
            if not needs_update.any():
                break
            direct_to_root_map += next_hop_offsets
        
        return direct_to_root_map

    @staticmethod
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
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.merge(x)


class ToMeK2New(nn.Module):
    """
    New k=2 ToMe variant with two APIs:
    - compute_merge(x): identical behavior to ToMeK2.merge, but the entire process runs under torch.no_grad.
      Returns (merged_x, btree_map, avg_sim) with the same semantics (avg_sim is per-batch min similarity).
    - merge(x, direct_to_root_map): differentiable single-step merge using a provided direct-to-root map.
      Performs token averaging to roots in one vectorized pass and returns only the merged tensor.
    """
    def __init__(self, r: float, num_iterations: int):
        super().__init__()
        self.r = r
        self.num_iterations = num_iterations

    def compute_merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        with torch.no_grad():
            btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)

            total_to_merge = int(self.r * N)
            if total_to_merge == 0:
                return x, btree_map, torch.zeros(B, device=device, dtype=dtype)

            merges_per_iter = np.diff(np.linspace(0, total_to_merge, self.num_iterations + 1, dtype=int))

            size = torch.ones(B, N, device=device, dtype=dtype)
            orig_idx = torch.arange(N, device=device).expand(B, -1)

            min_sim = torch.full((B,), float('inf'), device=device, dtype=dtype)
            total_selected = 0
            for k in merges_per_iter:
                if k == 0:
                    continue

                current_N = x.shape[1]
                if current_N < 2:
                    break
                # Similarity between adjacent tokens
                x_norm = F.normalize(x.detach(), dim=-1)
                sim = (x_norm[:, :-1, :] * x_norm[:, 1:, :]).sum(dim=-1)  # (B, current_N-1)

                # Greedy non-overlapping selection: pick top-1 per step, k steps
                sel_dst = torch.full((B, k), -1, device=device, dtype=torch.long)
                sel_src = torch.full((B, k), -1, device=device, dtype=torch.long)

                # Token usage mask to prevent chains and multiple usage within the iteration
                used = torch.zeros(B, current_N, dtype=torch.bool, device=device)

                sim_work = sim.clone()
                vals_per_slot = torch.full((B, k), float('-inf'), device=device, dtype=sim.dtype) if k > 0 else None
                for t in range(k):
                    # Invalidate pairs touching already used tokens
                    pair_valid = (~used[:, :-1]) & (~used[:, 1:])
                    masked = sim_work.masked_fill(~pair_valid, float('-inf'))

                    # Top-1 per batch
                    top_val, top_idx = masked.max(dim=1)
                    can_pick = top_val.isfinite()
                    if not can_pick.any():
                        break

                    dst_sel = top_idx
                    src_sel = top_idx + 1

                    sel_dst[can_pick, t] = dst_sel[can_pick]
                    sel_src[can_pick, t] = src_sel[can_pick]
                    vals_per_slot[can_pick, t] = top_val[can_pick]

                    # Mark tokens as used
                    used.scatter_(1, dst_sel.unsqueeze(1), True)
                    used.scatter_(1, src_sel.unsqueeze(1), True)

                # Equalize across batch
                valid_merge = (sel_src != -1)
                if not valid_merge.any():
                    break
                per_batch_counts = valid_merge.sum(dim=1)
                m = int(per_batch_counts.min().item())
                if m == 0:
                    break
                step_min = vals_per_slot[:, :m].min(dim=1).values
                min_sim = torch.minimum(min_sim, step_min.to(dtype))
                total_selected += m

                # Apply merges (vectorized)
                sel_dst = sel_dst[:, :m]
                sel_src = sel_src[:, :m]

                # Update btree_map (offset is always -1 for adjacent merge)
                src_orig_idx = orig_idx.gather(1, sel_src)
                btree_map.scatter_(1, src_orig_idx, torch.full_like(src_orig_idx, -1))

                # Update features and sizes
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

                # Remove merged src tokens
                remove_mask = torch.ones(B, current_N, dtype=torch.bool, device=device)
                remove_mask.scatter_(1, sel_src, False)
                x = x[remove_mask].view(B, -1, C)
                size = size[remove_mask].view(B, -1)
                orig_idx = orig_idx[remove_mask].view(B, -1)

            if total_selected == 0:
                avg_sim = torch.zeros(B, device=device, dtype=dtype)
            else:
                avg_sim = min_sim
            return x, btree_map, avg_sim

    def merge(self, x: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        """
        Differentiable single-step merge given a direct-to-root map.
        Returns the merged tensor containing only root tokens, in order.
        """
        B, N, C = x.shape
        size = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)

        root_indices = torch.arange(N, device=x.device).expand(B, -1) + direct_to_root_map

        merged_x = torch.zeros_like(x)
        merged_size = torch.zeros_like(size)

        root_indices_expanded = root_indices.unsqueeze(-1).expand(-1, -1, C)
        merged_x.scatter_add_(1, root_indices_expanded, x * size)
        merged_size.scatter_add_(1, root_indices.unsqueeze(-1), size)

        merged_x = merged_x / (merged_size + 1e-8)

        root_mask = (direct_to_root_map == 0)
        root_tokens = merged_x[root_mask].view(B, -1, C)
        return root_tokens

    @torch.no_grad()
    def btree_to_root_map(self, merge_btree: torch.Tensor) -> torch.Tensor:
        B, N = merge_btree.shape
        device = merge_btree.device
        # For k=2, merge_btree contains only {0 (root), -1 (merged into left neighbor)}.
        # The direct-to-root offset for each position i is (last_root_pos(i) - i),
        # where last_root_pos(i) is the index of the nearest root token at or to the left of i.
        # Compute last_root_pos via a cummax over masked indices.
        idx = torch.arange(N, device=device, dtype=merge_btree.dtype).view(1, N).expand(B, -1)
        is_root = (merge_btree == 0)
        masked = torch.where(is_root, idx + 1, torch.zeros_like(merge_btree))  # use 1-based to keep zeros for non-roots
        last_root_pos_plus1, _ = torch.cummax(masked, dim=1)
        last_root_pos = last_root_pos_plus1 - 1
        direct_to_root_map = last_root_pos - idx
        return direct_to_root_map

    def unmerge(self, y: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N_original = direct_to_root_map.shape
        if y.shape[1] == N_original:
            return y
        
        _, _, C = y.shape
        device = y.device

        root_mask = (direct_to_root_map == 0)
        root_rank_map = torch.cumsum(root_mask.to(torch.int32), dim=1) - 1
        source_indices = torch.arange(N_original, device=device).view(1, -1) + direct_to_root_map
        root_ranks = torch.gather(root_rank_map, 1, source_indices).clamp_min(0)
        final_x = torch.gather(y, 1, root_ranks.unsqueeze(-1).expand(-1, -1, C))
        return final_x


class ToPrK2New(nn.Module):
    """
    Pruning-based k=2 variant mirroring ToMeK2New's selection but removes src tokens
    instead of averaging them into dst tokens.

    APIs (kept identical to ToMeK2New):
    - compute_merge(x): runs full pruning selection under no_grad, returns
      (merged_x, btree_map, avg_sim), where merged_x contains only root (kept) tokens
      in order; btree_map has -1 for pruned src positions; avg_sim is per-batch min
      similarity over selected pairs.
    - merge(x, direct_to_root_map): returns only root tokens gathered from x (no
      averaging), in order.
    - btree_to_root_map(merge_btree): resolves immediate-parent offsets to direct-to-root.
    - unmerge(y, direct_to_root_map): reconstructs sequence by copying root token values
      into pruned positions.
    """
    def __init__(self, r: float, num_iterations: int):
        super().__init__()
        self.r = r
        self.num_iterations = num_iterations

    def compute_merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        with torch.no_grad():
            btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)

            total_to_prune = int(self.r * N)
            if total_to_prune == 0:
                return x, btree_map, torch.zeros(B, device=device, dtype=dtype)

            merges_per_iter = np.diff(np.linspace(0, total_to_prune, self.num_iterations + 1, dtype=int))

            # We do not change token values on dst; just track and remove srcs
            orig_idx = torch.arange(N, device=device).expand(B, -1)

            # Precompute normalization once on the original sequence; reuse via gathers per iteration
            base_norm = F.normalize(x.detach(), dim=-1)

            min_sim = torch.full((B,), float('inf'), device=device, dtype=dtype)
            total_selected = 0
            for k in merges_per_iter:
                if k == 0:
                    continue

                current_N = x.shape[1]
                if current_N < 2:
                    break

                # Similarity between adjacent tokens using precomputed normalization
                curr_norm = base_norm.gather(1, orig_idx.unsqueeze(-1).expand(-1, -1, C))
                sim = (curr_norm[:, :-1, :] * curr_norm[:, 1:, :]).sum(dim=-1)  # (B, current_N-1)

                # Greedy non-overlapping selection: pick top-1 per step, k steps
                sel_dst = torch.full((B, k), -1, device=device, dtype=torch.long)
                sel_src = torch.full((B, k), -1, device=device, dtype=torch.long)

                # Token usage mask to prevent chains and multiple usage within the iteration
                used = torch.zeros(B, current_N, dtype=torch.bool, device=device)

                sim_work = sim.clone()
                vals_per_slot = torch.full((B, k), float('-inf'), device=device, dtype=sim.dtype) if k > 0 else None
                for t in range(k):
                    # Invalidate pairs touching already used tokens
                    pair_valid = (~used[:, :-1]) & (~used[:, 1:])
                    masked = sim_work.masked_fill(~pair_valid, float('-inf'))

                    # Top-1 per batch
                    top_val, top_idx = masked.max(dim=1)
                    can_pick = top_val.isfinite()
                    if not can_pick.any():
                        break

                    dst_sel = top_idx
                    src_sel = top_idx + 1

                    sel_dst[can_pick, t] = dst_sel[can_pick]
                    sel_src[can_pick, t] = src_sel[can_pick]
                    vals_per_slot[can_pick, t] = top_val[can_pick]

                    # Mark tokens as used
                    used.scatter_(1, dst_sel.unsqueeze(1), True)
                    used.scatter_(1, src_sel.unsqueeze(1), True)

                # Equalize across batch
                valid_merge = (sel_src != -1)
                if not valid_merge.any():
                    break
                per_batch_counts = valid_merge.sum(dim=1)
                m = int(per_batch_counts.min().item())
                if m == 0:
                    break
                step_min = vals_per_slot[:, :m].min(dim=1).values
                min_sim = torch.minimum(min_sim, step_min.to(dtype))
                total_selected += m

                # Apply pruning (remove src tokens only; keep dst values unchanged)
                sel_dst = sel_dst[:, :m]
                sel_src = sel_src[:, :m]

                # Update btree_map (offset is always -1 for adjacent prune)
                src_orig_idx = orig_idx.gather(1, sel_src)
                btree_map.scatter_(1, src_orig_idx, torch.full_like(src_orig_idx, -1))

                # Remove selected src tokens from x and orig_idx
                remove_mask = torch.ones(B, current_N, dtype=torch.bool, device=device)
                remove_mask.scatter_(1, sel_src, False)
                x = x[remove_mask].view(B, -1, C)
                orig_idx = orig_idx[remove_mask].view(B, -1)

            if total_selected == 0:
                avg_sim = torch.zeros(B, device=device, dtype=dtype)
            else:
                avg_sim = min_sim
            return x, btree_map, avg_sim

    def merge(self, x: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        """
        Differentiable single-step pruning result given a direct-to-root map.
        Returns the gathered root tokens (no averaging), in order.
        """
        B, N, C = x.shape
        root_mask = (direct_to_root_map == 0)
        root_tokens = x[root_mask].view(B, -1, C)
        return root_tokens

    @torch.no_grad()
    def btree_to_root_map(self, merge_btree: torch.Tensor) -> torch.Tensor:
        B, N = merge_btree.shape
        device = merge_btree.device
        idx = torch.arange(N, device=device, dtype=merge_btree.dtype).view(1, N).expand(B, -1)
        is_root = (merge_btree == 0)
        masked = torch.where(is_root, idx + 1, torch.zeros_like(merge_btree))
        last_root_pos_plus1, _ = torch.cummax(masked, dim=1)
        last_root_pos = last_root_pos_plus1 - 1
        direct_to_root_map = last_root_pos - idx
        return direct_to_root_map

    def unmerge(self, y: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N_original = direct_to_root_map.shape
        if y.shape[1] == N_original:
            return y
        _, _, C = y.shape
        root_mask = (direct_to_root_map == 0)
        root_rank_map = torch.cumsum(root_mask.to(torch.int32), dim=1) - 1
        source_indices = torch.arange(N_original, device=y.device).view(1, -1) + direct_to_root_map
        root_ranks = torch.gather(root_rank_map, 1, source_indices).clamp_min(0)
        final_x = torch.gather(y, 1, root_ranks.unsqueeze(-1).expand(-1, -1, C))
        return final_x


class ToPrK2NewChunk(nn.Module):
    """
    Chunked pruning variant of k=2 ToPrK2New.

    - Splits the input sequence into fixed-length chunks (last chunk may be shorter)
    - For each chunk with length L_c, targets q_c = floor(r*L_c) prunes, then distributes
      the remaining global quota by largest-remainder so that sum_c q_c = floor(r*N)
    - Within each chunk, performs the same adjacent-pair greedy selection as ToPrK2New
      (no within-iteration chaining), but reuses precomputed normalization from the
      original sequence via gathers (no re-normalize per iteration)
    - After all chunks are processed, strictly checks that the total number of kept
      tokens equals N - floor(r*N); if not, raises an error (deterministic behavior)

    APIs mirror ToPrK2New:
      - compute_merge(x) -> (merged_x, btree_map, avg_sim)
      - merge(x, direct_to_root_map) -> root tokens (no averaging)
      - btree_to_root_map(merge_btree) -> left-chain direct offsets
      - unmerge(y, direct_to_root_map) -> copy kept tokens back to original length
    """
    def __init__(self, r: float, num_iterations: int, chunk_size: int = 100):
        super().__init__()
        if not (0.0 <= r <= 1.0):
            raise ValueError("r must be between 0.0 and 1.0")
        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        if chunk_size < 2:
            raise ValueError("chunk_size must be >= 2")
        self.r = float(r)
        self.num_iterations = int(num_iterations)
        self.chunk_size = int(chunk_size)

    @torch.no_grad()
    def compute_merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype

        # Trivial cases
        total_to_prune = int(self.r * N)
        if N == 0:
            btree_map = torch.zeros(B, 0, device=device, dtype=torch.int64)
            avg_sim = torch.zeros(B, device=device, dtype=dtype)
            return x, btree_map, avg_sim
        if total_to_prune == 0:
            btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)
            avg_sim = torch.zeros(B, device=device, dtype=dtype)
            return x, btree_map, avg_sim

        # Precompute normalization once on the original sequence; reuse via gathers
        base_norm = F.normalize(x.detach(), dim=-1)

        # Prepare global outputs and statistics
        btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)
        kept_chunks: list[torch.Tensor] = []
        min_sim = torch.full((B,), float('inf'), device=device, dtype=dtype)
        total_selected = 0

        # Chunk partition
        starts = list(range(0, N, self.chunk_size))
        lens = [min(self.chunk_size, N - s) for s in starts]

        # Largest-remainder distribution of per-chunk prune quotas
        # q_floor_c = floor(r * L_c); distribute remaining to largest remainders
        raw = [self.r * float(Lc) for Lc in lens]
        q_floor = [int(math.floor(v)) for v in raw]
        sum_floor = int(sum(q_floor))
        remainder = total_to_prune - sum_floor
        if remainder > 0:
            frac = [(raw[i] - float(q_floor[i]), i) for i in range(len(lens))]
            frac.sort(key=lambda t: t[0], reverse=True)
            for k in range(min(remainder, len(frac))):
                idx = frac[k][1]
                q_floor[idx] += 1

        # Light safeguard: if the LAST chunk is too small to realize its assigned quota,
        # shift just enough deletions to the previous chunk (if it has spare capacity).
        # Capacity upper bound for k=2 and T iterations: cap = L - ceil(L / 2^T)
        if len(lens) >= 2:
            T = max(1, int(self.num_iterations))
            pow2T = 1 << T
            caps = [int(Lc - math.ceil(Lc / float(pow2T))) for Lc in lens]
            last = len(lens) - 1
            over = q_floor[last] - max(0, caps[last])
            if over > 0:
                prev = last - 1
                spare_prev = max(0, caps[prev] - q_floor[prev])
                shift = int(min(over, spare_prev))
                if shift > 0:
                    q_floor[last] -= shift
                    q_floor[prev] += shift

        # Process each chunk independently, concatenating kept tokens
        for ci, s in enumerate(starts):
            Lc = lens[ci]
            e = s + Lc
            x_chunk = x[:, s:e, :]
            # orig indices for this chunk; track deletions within the chunk
            orig_idx = torch.arange(s, e, device=device).expand(B, -1)

            q_c = int(q_floor[ci])
            if q_c <= 0 or Lc <= 1:
                kept_chunks.append(x_chunk)
                continue

            # Distribute q_c across iterations like the base implementation
            merges_per_iter = np.diff(np.linspace(0, q_c, self.num_iterations + 1, dtype=int))

            for k in merges_per_iter:
                if k == 0:
                    continue
                current_N = x_chunk.shape[1]
                if current_N < 2:
                    break

                # Reuse precomputed normalization on current tokens
                curr_norm = base_norm.gather(1, orig_idx.unsqueeze(-1).expand(-1, -1, C))
                sim = (curr_norm[:, :-1, :] * curr_norm[:, 1:, :]).sum(dim=-1)  # (B, current_N-1)

                # Greedy non-overlapping selection: pick top-1 per step, k steps
                sel_dst = torch.full((B, k), -1, device=device, dtype=torch.long)
                sel_src = torch.full((B, k), -1, device=device, dtype=torch.long)

                used = torch.zeros(B, current_N, dtype=torch.bool, device=device)
                sim_work = sim.clone()
                vals_per_slot = torch.full((B, k), float('-inf'), device=device, dtype=sim.dtype) if k > 0 else None
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
                    vals_per_slot[can_pick, t] = top_val[can_pick]

                    used.scatter_(1, dst_sel.unsqueeze(1), True)
                    used.scatter_(1, src_sel.unsqueeze(1), True)

                valid_merge = (sel_src != -1)
                if not valid_merge.any():
                    break
                per_batch_counts = valid_merge.sum(dim=1)
                m = int(per_batch_counts.min().item())
                if m == 0:
                    break

                step_min = vals_per_slot[:, :m].min(dim=1).values
                min_sim = torch.minimum(min_sim, step_min.to(dtype))
                total_selected += m

                # Apply pruning (remove src tokens only; keep dst values unchanged)
                sel_dst = sel_dst[:, :m]
                sel_src = sel_src[:, :m]

                src_orig_idx = orig_idx.gather(1, sel_src)
                btree_map.scatter_(1, src_orig_idx, torch.full_like(src_orig_idx, -1))

                remove_mask = torch.ones(B, current_N, dtype=torch.bool, device=device)
                remove_mask.scatter_(1, sel_src, False)
                x_chunk = x_chunk[remove_mask].view(B, -1, C)
                orig_idx = orig_idx[remove_mask].view(B, -1)

            kept_chunks.append(x_chunk)

        # Concatenate kept tokens from all chunks (preserve chunk order)
        merged_x = torch.cat(kept_chunks, dim=1) if len(kept_chunks) > 0 else x[:, :0, :]

        # Strict post-check: enforce exact target kept length
        expected_kept = N - total_to_prune
        actual_kept = int(merged_x.shape[1])
        if actual_kept != expected_kept:
            raise RuntimeError(
                f"ToPrK2NewChunk: kept length mismatch (actual={actual_kept}, expected={expected_kept}); "
                f"r={self.r}, N={N}, num_iterations={self.num_iterations}, chunk_size={self.chunk_size}"
            )

        # avg_sim per-batch min over all used pairs; if none selected, zeros
        if total_selected == 0:
            avg_sim = torch.zeros(B, device=device, dtype=dtype)
        else:
            avg_sim = min_sim
        return merged_x, btree_map, avg_sim

    def merge(self, x: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        root_mask = (direct_to_root_map == 0)
        root_tokens = x[root_mask].view(B, -1, C)
        return root_tokens

    @torch.no_grad()
    def btree_to_root_map(self, merge_btree: torch.Tensor) -> torch.Tensor:
        B, N = merge_btree.shape
        device = merge_btree.device
        idx = torch.arange(N, device=device, dtype=merge_btree.dtype).view(1, N).expand(B, -1)
        is_root = (merge_btree == 0)
        masked = torch.where(is_root, idx + 1, torch.zeros_like(merge_btree))
        last_root_pos_plus1, _ = torch.cummax(masked, dim=1)
        last_root_pos = last_root_pos_plus1 - 1
        direct_to_root_map = last_root_pos - idx
        return direct_to_root_map

    def unmerge(self, y: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N_original = direct_to_root_map.shape
        if y.shape[1] == N_original:
            return y
        _, _, C = y.shape
        root_mask = (direct_to_root_map == 0)
        root_rank_map = torch.cumsum(root_mask.to(torch.int32), dim=1) - 1
        source_indices = torch.arange(N_original, device=y.device).view(1, -1) + direct_to_root_map
        root_ranks = torch.gather(root_rank_map, 1, source_indices).clamp_min(0)
        final_x = torch.gather(y, 1, root_ranks.unsqueeze(-1).expand(-1, -1, C))
        return final_x

class ToPrTopK(nn.Module):
    """
    Simple pruning based on dissimilarity to the previous token.

    - For each token i, compute s_i = cosine_similarity(x[i], x[i-1]); define s_0 := 0
    - Keep exactly int(N*r) tokens with the largest (1 - s_i); delete the rest (no iterations)
    - btree_map has -1 for pruned positions and 0 for kept positions
    - avg_sim is the per-batch minimum of selected (1 - s_i)

    Methods and return values mirror ToPrK2New:
    - compute_merge(x) -> (merged_x, btree_map, avg_sim)
    - merge(x, direct_to_root_map) -> gathered kept tokens (no averaging)
    - btree_to_root_map(merge_btree) -> resolves offsets to direct-to-root map
    - unmerge(y, direct_to_root_map) -> reconstructs sequence by copying kept tokens
    """
    def __init__(self, r: float, num_iterations: int):
        super().__init__()
        self.r = r
        self.num_iterations = num_iterations  # kept for API symmetry

    def compute_merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        with torch.no_grad():
            if N == 0:
                btree_map = torch.zeros(B, 0, device=device, dtype=torch.int64)
                avg_sim = torch.zeros(B, device=device, dtype=dtype)
                return x, btree_map, avg_sim

            # Number of tokens to keep
            keep_k = int(self.r * N)
            keep_k = int(max(0, min(keep_k, N)))

            # Cosine similarity with previous token; s_0 := 0
            x_norm = F.normalize(x.detach(), dim=-1)
            if N > 1:
                sim_rest = (x_norm[:, 1:, :] * x_norm[:, :-1, :]).sum(dim=-1)  # (B, N-1)
                s = torch.cat([torch.zeros(B, 1, device=device, dtype=sim_rest.dtype), sim_rest], dim=1)
            else:
                s = torch.zeros(B, 1, device=device, dtype=x_norm.dtype)
            d = (1.0 - s.to(dtype))

            if keep_k == N:
                # Keep all tokens
                btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)
                avg_sim = d.min(dim=1).values
                return x, btree_map, avg_sim

            if keep_k > 0:
                top_vals, top_idx = torch.topk(d, k=keep_k, dim=1, largest=True, sorted=True)
                avg_sim = top_vals.min(dim=1).values.to(dtype)
                keep_idx_sorted, _ = torch.sort(top_idx, dim=1)

                # Gather kept tokens in original order
                x_kept = x.gather(1, keep_idx_sorted.unsqueeze(-1).expand(-1, -1, C))

                # Build btree_map: -1 for pruned positions, 0 for kept
                btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)
                keep_mask = torch.zeros(B, N, device=device, dtype=torch.bool)
                keep_mask.scatter_(1, keep_idx_sorted, True)
                btree_map.masked_fill_(~keep_mask, -1)
            else:
                # Keep none: return empty sequence; btree_map all -1; avg_sim zeros
                x_kept = x[:, :0, :]
                btree_map = torch.full((B, N), -1, device=device, dtype=torch.int64)
                avg_sim = torch.zeros(B, device=device, dtype=dtype)

            return x_kept, btree_map, avg_sim

    def merge(self, x: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        root_mask = (direct_to_root_map == 0)
        root_tokens = x[root_mask].view(B, -1, C)
        return root_tokens

    @torch.no_grad()
    def btree_to_root_map(self, merge_btree: torch.Tensor) -> torch.Tensor:
        B, N = merge_btree.shape
        device = merge_btree.device
        idx = torch.arange(N, device=device, dtype=merge_btree.dtype).view(1, N).expand(B, -1)
        is_root = (merge_btree == 0)
        masked = torch.where(is_root, idx + 1, torch.zeros_like(merge_btree))
        last_root_pos_plus1, _ = torch.cummax(masked, dim=1)
        last_root_pos = last_root_pos_plus1 - 1
        direct_to_root_map = last_root_pos - idx
        return direct_to_root_map

    def unmerge(self, y: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N_original = direct_to_root_map.shape
        if y.shape[1] == N_original:
            return y
        _, _, C = y.shape
        root_mask = (direct_to_root_map == 0)
        root_rank_map = torch.cumsum(root_mask.to(torch.int32), dim=1) - 1
        source_indices = torch.arange(N_original, device=y.device).view(1, -1) + direct_to_root_map
        root_ranks = torch.gather(root_rank_map, 1, source_indices).clamp_min(0)
        final_x = torch.gather(y, 1, root_ranks.unsqueeze(-1).expand(-1, -1, C))
        return final_x

class ToMeK2V2(nn.Module):
    """
    Batched k=2 variant using greedy non-adjacent top-k selection without per-slot max loops.
    - Considers adjacent pairs (i, i+1)
    - Per iteration selects up to k non-adjacent pairs by scanning a single top-T list (T<=3k-2)
    - Equalizes number of merges across batch (use minimum m across batch)
    - Returns (merged_x, btree_map, avg_sim)
    """
    def __init__(self, r: float, num_iterations: int):
        super().__init__()
        self.r = r
        self.num_iterations = num_iterations

    def merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)

        total_to_merge = int(self.r * N)
        if total_to_merge == 0:
            return x, btree_map, torch.zeros(B, device=device, dtype=dtype)

        merges_per_iter = np.diff(np.linspace(0, total_to_merge, self.num_iterations + 1, dtype=int))

        size = torch.ones(B, N, device=device, dtype=dtype)
        orig_idx = torch.arange(N, device=device).expand(B, -1)

        min_sim = torch.full((B,), float('inf'), device=device, dtype=dtype)
        total_selected = 0

        for k in merges_per_iter:
            if k == 0:
                continue
            with torch.no_grad():
                current_N = x.shape[1]
                if current_N < 2:
                    break
                # Adjacent similarities
                x_norm = F.normalize(x.detach(), dim=-1)
                sims = (x_norm[:, :-1, :] * x_norm[:, 1:, :]).sum(dim=-1)  # (B, M) with M=current_N-1

                M = sims.shape[1]
                if M == 0:
                    break
                T = int(min(M, max(0, 3 * k - 2)))
                if T == 0:
                    continue
                vals, cand = torch.topk(sims, T, dim=1, largest=True, sorted=True)  # (B, T)

                # Greedy non-adjacent selection across the pre-sorted candidate list
                blocked = torch.zeros(B, M, dtype=torch.bool, device=device)
                sel = torch.full((B, k), -1, dtype=torch.long, device=device)
                sel_vals = torch.full((B, k), float('-inf'), dtype=vals.dtype, device=device)
                counts = torch.zeros(B, dtype=torch.long, device=device)

                for t in range(T):
                    p = cand[:, t]  # (B,)
                    # check not blocked and still need
                    can_take = (~blocked.gather(1, p.unsqueeze(1)).squeeze(1)) & (counts < k)
                    if not can_take.any():
                        # Still need to advance blocking for those we skip? No, only on taken
                        continue
                    # write selection at position counts[b] for each batch
                    idx_b = torch.nonzero(can_take, as_tuple=False).squeeze(1)
                    pos = counts[idx_b]
                    sel[idx_b, pos] = p[idx_b]
                    sel_vals[idx_b, pos] = vals[idx_b, t]
                    counts[idx_b] = pos + 1

                    # block p and neighbors ONLY for batches that took this candidate
                    pb = p[idx_b]
                    blocked[idx_b, pb] = True
                    left_mask = pb > 0
                    if left_mask.any():
                        blocked[idx_b[left_mask], (pb[left_mask] - 1)] = True
                    right_mask = pb + 1 < M
                    if right_mask.any():
                        blocked[idx_b[right_mask], (pb[right_mask] + 1)] = True

                valid_counts = counts
                if (valid_counts == 0).all():
                    continue
                m = int(valid_counts.min().item())
                if m == 0:
                    continue

                sel = sel[:, :m]
                sel_vals = sel_vals[:, :m]

                # Convert pair index p -> (dst=p, src=p+1)
                sel_dst = sel
                sel_src = sel + 1

            # Update btree_map (offset -1 for adjacent merge)
            src_orig_idx = orig_idx.gather(1, sel_src)
            btree_map.scatter_(1, src_orig_idx, torch.full_like(src_orig_idx, -1))

            # Differentiable feature/size updates
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

            # Remove src tokens
            remove_mask = torch.ones(B, current_N, dtype=torch.bool, device=device)
            remove_mask.scatter_(1, sel_src, False)
            x = x[remove_mask].view(B, -1, C)
            size = size[remove_mask].view(B, -1)
            orig_idx = orig_idx[remove_mask].view(B, -1)

            # min similarity accumulation
            step_min = sel_vals.min(dim=1).values
            min_sim = torch.minimum(min_sim, step_min.to(dtype))
            total_selected += m

        if total_selected == 0:
            avg_sim = torch.zeros(B, device=device, dtype=dtype)
        else:
            avg_sim = min_sim
        return x, btree_map, avg_sim

    @staticmethod
    @torch.no_grad()
    def btree_to_root_map(merge_btree: torch.Tensor) -> torch.Tensor:
        return ToMeK2.btree_to_root_map(merge_btree)

    @staticmethod
    def unmerge(y: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        return ToMeK2.unmerge(y, direct_to_root_map)

class ToPrPLETopK(nn.Module):
    """
    Pruning via Path-Length Equalization (PLE-TopK) with left-chain semantics.

    keep_last is disabled (False): token 0 is always kept; the last token may be pruned.

    - compute_merge(x): select exactly M = N - floor(r*N) frontiers, where the interior M-1
      frontiers are chosen per equal path-length bins. Two modes:
        - first-crossing (use_bin_argmax=False): select the first token index where cumulative
          path length crosses each target.
        - bin-wise argmax (use_bin_argmax=True): within each bin, select the pair with largest
          dissimilarity and keep its left token as the frontier.
      Returns (merged_x, merge_btree, avg_sim) with merge_btree in {0, -1}.
    - merge(x, direct_to_root_map): gather kept tokens (offset==0) in order.
    - btree_to_root_map(merge_btree): resolve immediate -1 links to nearest left root (vectorized).
    - unmerge(y, direct_to_root_map): copy kept tokens into pruned slots via left-chain.
    """
    def __init__(self, r: float, beta: float = 1.0, eps: float = 1e-12, use_bin_argmax: bool = False):
        super().__init__()
        self.r = float(r)
        self.beta = float(beta)
        self.eps = float(eps)
        self.use_bin_argmax = bool(use_bin_argmax)

    @torch.no_grad()
    def compute_merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype

        if N == 0:
            btree_map = torch.zeros(B, 0, device=device, dtype=torch.int64)
            avg_sim = torch.zeros(B, device=device, dtype=dtype)
            return x, btree_map, avg_sim

        # Keep/prune counts (keep_last=False): always keep token 0
        K = int(min(max(math.floor(self.r * N), 0), N - 1))
        M = N - K

        x_norm = F.normalize(x, dim=-1)
        if N > 1:
            d = 1.0 - (x_norm[:, :-1, :] * x_norm[:, 1:, :]).sum(dim=-1)  # (B, N-1)
        else:
            d = torch.zeros(B, 0, device=device, dtype=dtype)

        w = (d + self.eps).pow(self.beta)

        if N > 1:
            D = torch.zeros(B, N, device=device, dtype=dtype)
            D[:, 1:] = torch.cumsum(w, dim=1)
            L = D[:, -1]  # (B,)
        else:
            D = torch.zeros(B, 1, device=device, dtype=dtype)
            L = torch.zeros(B, device=device, dtype=dtype)

        btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)

        if M <= 1:
            if N > 1:
                btree_map[:, 1:] = -1
            merged_x = x[:, :1, :]
            avg_sim = torch.zeros(B, device=device, dtype=dtype)
            return merged_x, btree_map, avg_sim

        keep_int = torch.zeros(B, N, device=device, dtype=torch.int64)
        keep_int[:, 0] = 1

        interior = M - 1

        # Vectorized first-crossing selection (default)
        if not self.use_bin_argmax:
            # targets: (B, interior) with per-batch spacing t = L/M
            t = torch.where(L > 0, L / float(M), torch.zeros_like(L))  # (B,)
            targets = (torch.arange(1, M, device=device, dtype=dtype).view(1, -1) * t.view(B, 1))  # (B, interior)
            # For L==0, handle later with uniform fallback
            # Compute j = smallest index with D >= target (broadcasted argmax over ge-mask)
            if N > 1 and interior > 0:
                ge = D.unsqueeze(2) >= targets.unsqueeze(1)  # (B, N, interior)
                # If a column is all False (e.g., L==0), argmax gives 0; clamp to at least 1 later
                j = ge.float().argmax(dim=1).clamp_(min=1, max=N - 1)  # (B, interior)
            else:
                j = torch.ones(B, interior, device=device, dtype=torch.long)

            # Fallback uniform selection for rows with L==0
            if interior > 0:
                uniform_j = torch.linspace(1, N - 1, steps=interior, device=device).round().long().clamp(1, N - 1)
                j = torch.where((L > 0).view(B, 1), j, uniform_j.view(1, -1).expand(B, -1))

            # Enforce strictly increasing j per row (dedup) using cummax trick
            if interior > 0:
                ar = torch.arange(interior, device=device, dtype=j.dtype).view(1, -1)
                s = (j - ar)
                smax, _ = torch.cummax(s, dim=1)
                j_strict = smax + ar
                j_strict = j_strict.clamp_(min=1, max=N - 1)
            else:
                j_strict = j

            # Keep the RIGHT token of each boundary crossing: j -> j (token index already on right side)
            keep_int.scatter_(1, j_strict, 1)

            # Fill missing selections to hit exactly M per batch
            kept_counts = keep_int.sum(dim=1)
            need = (M - kept_counts).clamp_min(0)  # (B,)
            need_max = int(need.max().item()) if need.numel() > 0 else 0
            if need_max > 0 and N > 1:
                available = (keep_int[:, 1:] == 0)
                pos = torch.arange(1, N, device=device).view(1, -1).expand(B, -1)
                scores = torch.where(available, -pos.to(dtype), torch.full((B, N - 1), float('-inf'), device=device, dtype=dtype))
                vals, extra_cols = torch.topk(scores, k=need_max, dim=1)
                extra_j = (extra_cols + 1).long()
                use_mask = (torch.arange(need_max, device=device).view(1, -1) < need.view(B, 1))
                extras_use = torch.where(use_mask, extra_j, torch.zeros_like(extra_j))
                vals_use = use_mask.long()
                keep_int.scatter_add_(1, extras_use, vals_use)

                # For avg_sim, compute boundary pairs for primary and extras
                if d.numel() > 0 and interior > 0:
                    i_idx = (j_strict - 1).clamp(0, N - 2)
                    sel_d_primary = torch.gather(d, 1, i_idx)
                    extra_pairs = (extra_j - 1).clamp(0, N - 2)
                    sel_d_extras = torch.gather(d, 1, extra_pairs)
                    sel_d_extras = torch.where(use_mask, sel_d_extras, torch.full_like(sel_d_extras, float('inf')))
                    all_d = torch.cat([sel_d_primary, sel_d_extras], dim=1)
                    avg_sim = all_d.min(dim=1).values.to(dtype)
                else:
                    avg_sim = torch.zeros(B, device=device, dtype=dtype)
            else:
                # No extras needed
                if d.numel() > 0 and interior > 0:
                    i_idx = (j_strict - 1).clamp(0, N - 2)
                    sel_d = torch.gather(d, 1, i_idx)
                    avg_sim = sel_d.min(dim=1).values.to(dtype)
                else:
                    avg_sim = torch.zeros(B, device=device, dtype=dtype)
        else:
            # Bin-wise argmax-left selection across bins k in [1..M-1]
            # Assign each pair to bins by right cumulative position: floor(D[i+1]/t)
            t = torch.where(L > 0, L / float(M), torch.ones_like(L))  # avoid div0
            if N > 1 and interior > 0:
                pair_right = D[:, 1:]  # (B, N-1)
                bin_idx = torch.clamp((pair_right / t.view(B, 1)).floor().long(), min=0, max=M - 1)
                # Exclude bin 0 from interior selection and exclude pair index 0 (duplicate of seed)
                d_mask = d.clone()
                if d_mask.numel() > 0:
                    d_mask[:, 0] = float('-inf')
                # Vectorized per-bin argmax over pairs
                bins = torch.arange(1, M, device=device, dtype=bin_idx.dtype).view(1, interior, 1)
                mask3 = (bin_idx.unsqueeze(1) == bins)  # (B, interior, N-1)
                d_exp = d_mask.unsqueeze(1).expand(-1, interior, -1)
                neg_inf = torch.full_like(d_exp, float('-inf'))
                masked_scores = torch.where(mask3, d_exp, neg_inf)
                vals, idxs = masked_scores.max(dim=2)  # (B, interior)
                chosen = idxs
                has_any = vals.isfinite()
                # Scatter only where a valid candidate exists; invalid bins write to col 0 (already kept)
                chosen_use = torch.where(has_any, chosen, torch.zeros_like(chosen))
                # Keep RIGHT token of boundary: i -> i+1
                chosen_tokens = (chosen_use + 1).clamp(1, N - 1)
                keep_int.scatter_(1, chosen_tokens, 1)

                # Ensure exactly M kept: fill missing with earliest available indices
                kept_counts = keep_int.sum(dim=1)
                need = (M - kept_counts).clamp_min(0)
                need_max = int(need.max().item()) if need.numel() > 0 else 0
                if need_max > 0 and N > 1:
                    available = (keep_int[:, 1:] == 0)
                    pos = torch.arange(1, N, device=device).view(1, -1).expand(B, -1)
                    scores = torch.where(available, -pos.to(dtype), torch.full((B, N - 1), float('-inf'), device=device, dtype=dtype))
                    vals, extra_cols = torch.topk(scores, k=need_max, dim=1)
                    extra_j = (extra_cols + 1).long()
                    use_mask = (torch.arange(need_max, device=device).view(1, -1) < need.view(B, 1))
                    extras_use = torch.where(use_mask, extra_j, torch.zeros_like(extra_j))
                    vals_use = use_mask.long()
                    keep_int.scatter_add_(1, extras_use, vals_use)

                    if d.numel() > 0 and interior > 0:
                        # avg_sim over boundary pairs (still i)
                        sel_d_primary = torch.gather(d, 1, chosen_use.clamp(0, N - 2))
                        extra_pairs = (extra_j - 1).clamp(0, N - 2)
                        sel_d_extras = torch.gather(d, 1, extra_pairs)
                        sel_d_extras = torch.where(use_mask, sel_d_extras, torch.full_like(sel_d_extras, float('inf')))
                        all_d = torch.cat([sel_d_primary, sel_d_extras], dim=1)
                        avg_sim = all_d.min(dim=1).values.to(dtype)
                    else:
                        avg_sim = torch.zeros(B, device=device, dtype=dtype)
                else:
                    if d.numel() > 0 and interior > 0:
                        sel_d = torch.gather(d, 1, chosen_use.clamp(0, N - 2))
                        # ignore bins without any candidate
                        sel_d = torch.where(has_any, sel_d, torch.full_like(sel_d, float('inf')))
                        avg_sim = sel_d.min(dim=1).values.to(dtype)
                    else:
                        avg_sim = torch.zeros(B, device=device, dtype=dtype)
            else:
                avg_sim = torch.zeros(B, device=device, dtype=dtype)

        keep = keep_int.bool()
        btree_map[~keep] = -1
        merged_x = x[keep].view(B, M, C)
        return merged_x, btree_map, avg_sim

    def merge(self, x: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        root_mask = (direct_to_root_map == 0)
        root_tokens = x[root_mask].view(B, -1, C)
        return root_tokens

    @torch.no_grad()
    def btree_to_root_map(self, merge_btree: torch.Tensor) -> torch.Tensor:
        B, N = merge_btree.shape
        device = merge_btree.device
        idx = torch.arange(N, device=device, dtype=merge_btree.dtype).view(1, N).expand(B, -1)
        is_root = (merge_btree == 0)
        masked = torch.where(is_root, idx + 1, torch.zeros_like(merge_btree))
        last_root_pos_plus1, _ = torch.cummax(masked, dim=1)
        last_root_pos = last_root_pos_plus1 - 1
        direct_to_root_map = last_root_pos - idx
        return direct_to_root_map

    def unmerge(self, y: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N_original = direct_to_root_map.shape
        if y.shape[1] == N_original:
            return y
        _, _, C = y.shape
        device = y.device
        root_mask = (direct_to_root_map == 0)
        root_rank_map = torch.cumsum(root_mask.to(torch.int32), dim=1) - 1
        source_indices = torch.arange(N_original, device=device).view(1, -1) + direct_to_root_map
        root_ranks = torch.gather(root_rank_map, 1, source_indices).clamp_min(0)
        final_x = torch.gather(y, 1, root_ranks.unsqueeze(-1).expand(-1, -1, C))
        return final_x


class ToPrCPRRTopK(nn.Module):
    """
    Pruning via Chunked + Round-Robin Top-K over equal path-length bins.

    keep_last is disabled (False): token 0 is always kept; the last token may be pruned.

    - compute_merge(x): build S equal-length pair bins along the cumulative path-length axis,
      rank pairs within each bin by dissimilarity, and select exactly M-1 interior frontiers
      by a global lexicographic order (round asc, score desc). Returns (merged_x, merge_btree, avg_sim).
    - merge / btree_to_root_map / unmerge: same semantics as ToPrPLETopK.
    """
    def __init__(self, r: float, beta: float = 1.0, eps: float = 1e-12,
                 bins: Optional[int] = None,
                 bin_size: Optional[int] = None):
        super().__init__()
        self.r = float(r)
        self.beta = float(beta)
        self.eps = float(eps)
        # bins: number of bins along path-length (if provided)
        # bin_size: desired number of pairs (approx. tokens) per bin; takes precedence if set
        self.bins = bins
        self.bin_size = bin_size

    @torch.no_grad()
    def compute_merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype

        if N == 0:
            btree_map = torch.zeros(B, 0, device=device, dtype=torch.int64)
            avg_sim = torch.zeros(B, device=device, dtype=dtype)
            return x, btree_map, avg_sim

        K = int(min(max(math.floor(self.r * N), 0), N - 1))
        M = N - K

        x_norm = F.normalize(x, dim=-1)
        if N > 1:
            d = 1.0 - (x_norm[:, :-1, :] * x_norm[:, 1:, :]).sum(dim=-1)  # (B, N-1)
        else:
            d = torch.zeros(B, 0, device=device, dtype=dtype)

        w = (d + self.eps).pow(self.beta)
        if N > 1:
            D = torch.zeros(B, N, device=device, dtype=dtype)
            D[:, 1:] = torch.cumsum(w, dim=1)
            L = D[:, -1]
        else:
            D = torch.zeros(B, 1, device=device, dtype=dtype)
            L = torch.zeros(B, device=device, dtype=dtype)

        btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)

        if M <= 1:
            if N > 1:
                btree_map[:, 1:] = -1
            merged_x = x[:, :1, :]
            avg_sim = torch.zeros(B, device=device, dtype=dtype)
            return merged_x, btree_map, avg_sim

        keep_int = torch.zeros(B, N, device=device, dtype=torch.int64)
        keep_int[:, 0] = 1
        interior = M - 1

        if N > 1 and interior > 0:
            # Choose number of bins S (constant across batch)
            M_int = int(M)
            if self.bin_size is not None and self.bin_size > 0:
                # Target pairs per bin -> number of bins = ceil((N-1)/bin_size)
                S = int(math.ceil((N - 1) / float(self.bin_size))) if N > 1 else 1
                S = max(1, min(S, N - 1))
            elif self.bins is not None and self.bins > 0:
                S = int(min(max(self.bins, 1), N - 1))
            else:
                S = int(min(M_int, max(1, int(math.isqrt(M_int)))))
                S = max(1, min(S, N - 1))

            pair_right = D[:, 1:]  # (B, N-1)
            bin_size = torch.where(L > 0, L / float(S), torch.ones_like(L))  # (B,)
            bin_idx = torch.clamp((pair_right / bin_size.view(B, 1)).floor().long(), min=0, max=S - 1)

            # Avoid selecting token index 0 twice: set score of pair 0 to -inf globally
            d_mask = d.clone()
            if d_mask.numel() > 0:
                d_mask[:, 0] = float('-inf')

            kbin = min(interior, N - 1)
            LARGE = 1_000_000.0
            keys_all = []
            idxs_all = []
            for s in range(S):
                mask_s = (bin_idx == s)
                scores_s = torch.where(mask_s, d_mask, torch.full_like(d_mask, float('-inf')))
                vals_s, idx_s = torch.topk(scores_s, k=kbin, dim=1)
                rounds = torch.arange(kbin, device=device, dtype=dtype).view(1, -1).expand(B, -1)
                key_s = (-rounds) * LARGE + vals_s
                keys_all.append(key_s)
                idxs_all.append(idx_s)

            keys_cat = torch.cat(keys_all, dim=1) if len(keys_all) > 0 else torch.full((B, interior), float('-inf'), device=device, dtype=dtype)
            idxs_cat = torch.cat(idxs_all, dim=1) if len(idxs_all) > 0 else torch.zeros(B, interior, device=device, dtype=torch.long)

            # Global top-k across bins
            topk = min(interior, keys_cat.shape[1])
            vals, ord_lin = torch.topk(keys_cat, k=topk, dim=1)
            sel_pairs = torch.gather(idxs_cat, 1, ord_lin)

            # Ensure exactly interior selections; if needed, pad from remaining candidates (rare)
            if sel_pairs.shape[1] < interior:
                pad = interior - sel_pairs.shape[1]
                extra_ord = torch.topk(keys_cat, k=min(keys_cat.shape[1], interior), dim=1).indices
                extra = torch.gather(idxs_cat, 1, extra_ord)[:, :pad]
                sel_pairs = torch.cat([sel_pairs, extra], dim=1)

            # Keep RIGHT token of boundary: i -> i+1
            sel_tokens = (sel_pairs + 1).clamp(1, N - 1)
            keep_int.scatter_(1, sel_tokens, 1)

            # avg_sim
            if d.numel() > 0:
                sel_d = torch.gather(d, 1, sel_pairs.clamp(0, N - 2))
                avg_sim = sel_d.min(dim=1).values.to(dtype)
            else:
                avg_sim = torch.zeros(B, device=device, dtype=dtype)
        else:
            # Degenerate: uniform by index
            if interior > 0:
                j = torch.linspace(1, N - 1, steps=interior, device=device).round().long().clamp(1, N - 1)
                keep_int.scatter_(1, j.view(1, -1).expand(B, -1), 1)
            avg_sim = torch.zeros(B, device=device, dtype=dtype)

        keep = keep_int.bool()
        btree_map[~keep] = -1
        merged_x = x[keep].view(B, M, C)
        return merged_x, btree_map, avg_sim

    def merge(self, x: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        root_mask = (direct_to_root_map == 0)
        root_tokens = x[root_mask].view(B, -1, C)
        return root_tokens

    @torch.no_grad()
    def btree_to_root_map(self, merge_btree: torch.Tensor) -> torch.Tensor:
        B, N = merge_btree.shape
        device = merge_btree.device
        idx = torch.arange(N, device=device, dtype=merge_btree.dtype).view(1, N).expand(B, -1)
        is_root = (merge_btree == 0)
        masked = torch.where(is_root, idx + 1, torch.zeros_like(merge_btree))
        last_root_pos_plus1, _ = torch.cummax(masked, dim=1)
        last_root_pos = last_root_pos_plus1 - 1
        direct_to_root_map = last_root_pos - idx
        return direct_to_root_map

    def unmerge(self, y: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N_original = direct_to_root_map.shape
        if y.shape[1] == N_original:
            return y
        _, _, C = y.shape
        device = y.device
        root_mask = (direct_to_root_map == 0)
        root_rank_map = torch.cumsum(root_mask.to(torch.int32), dim=1) - 1
        source_indices = torch.arange(N_original, device=device).view(1, -1) + direct_to_root_map
        root_ranks = torch.gather(root_rank_map, 1, source_indices).clamp_min(0)
        final_x = torch.gather(y, 1, root_ranks.unsqueeze(-1).expand(-1, -1, C))
        return final_x

class OurToMe2(nn.Module):
    """
    Efficient k=2 ToMe variant with alternating pairing patterns per iteration.
    - Iteration 0,2,...: dst at even indices, src at the next odd indices: (0<-1),(2<-3),...
    - Iteration 1,3,...: dst at odd indices,  src at the next even indices: (1<-2),(3<-4),...
    - Causal and non-overlapping by construction; no dist bookkeeping.
    - Outputs btree_map with offsets in {0, -1}.
    - Unmerge is compatible with ToMeTopK.btree_to_root_map()/unmerge.
    """
    def __init__(self, r: float, num_iterations: int, shift_offset: bool = True):
        super().__init__()
        if not (0.0 < r < 1.0):
            raise ValueError("r must be in (0,1)")
        if num_iterations < 1:
            raise ValueError("num_iterations must be >=1")
        self.r = r
        self.num_iterations = num_iterations
        self.shift_offset = shift_offset

    def compute_merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        dtype = x.dtype
        btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)

        total_to_merge = int(self.r * N)
        if total_to_merge == 0:
            return x, btree_map, torch.zeros(B, device=device, dtype=dtype)

        merges_per_iter = np.diff(np.linspace(0, total_to_merge, self.num_iterations + 1, dtype=int))

        size = torch.ones(B, N, device=device, dtype=dtype)
        orig_idx = torch.arange(N, device=device).expand(B, -1)

        min_sim = torch.full((B,), float('inf'), device=device, dtype=dtype)
        total_selected = 0
        for it, k in enumerate(merges_per_iter):
            if k == 0:
                continue
            with torch.no_grad():
                current_N = x.shape[1]
                if current_N < 2:
                    break
                # Normalize features for similarity calculation (no grad)
                x_norm = F.normalize(x.detach(), dim=-1)
                if it % 2 == 0 or not self.shift_offset:
                    # Even dst: pairs (0,1),(2,3),...
                    dst_base = torch.arange(0, current_N - 1, 2, device=device)
                    src_base = dst_base + 1
                else:
                    # Odd dst: pairs (1,2),(3,4),...
                    dst_base = torch.arange(1, current_N - 1, 2, device=device)
                    src_base = dst_base + 1

                num_pairs = dst_base.numel()
                if num_pairs == 0:
                    break
                # Gather pair features (B, P, C)
                dst_feat = x_norm[:, dst_base, :]
                src_feat = x_norm[:, src_base, :]
                sims = (dst_feat * src_feat).sum(dim=-1)  # (B, P)

                # Select top-k pairs per batch; equalize across batch
                k_eff = min(k, num_pairs)
                top_val, top_idx = torch.topk(sims, k=k_eff, dim=1)
                valid = top_val.isfinite()
                per_b = valid.sum(dim=1)
                m = int(per_b.min().item())
                if m == 0:
                    break
                sel_idx = top_idx[:, :m]  # (B, m)
                sel_dst = dst_base.unsqueeze(0).expand(B, -1).gather(1, sel_idx)
                sel_src = src_base.unsqueeze(0).expand(B, -1).gather(1, sel_idx)
                step_min = top_val[:, :m].min(dim=1).values
                min_sim = torch.minimum(min_sim, step_min.to(dtype))
                total_selected += m

                # Update btree_map: offset is always -1
                src_orig_idx = orig_idx.gather(1, sel_src)
                btree_map.scatter_(1, src_orig_idx, torch.full_like(src_orig_idx, -1))

                # Differentiable feature/size updates
                # dst_size_i = size.gather(1, sel_dst)
                src_size_i = size.gather(1, sel_src)
                # dst_feat_i = x.gather(1, sel_dst.unsqueeze(-1).expand(-1, -1, C))
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

                # Remove merged src tokens
                remove_mask = torch.ones(B, current_N, dtype=torch.bool, device=device)
                remove_mask.scatter_(1, sel_src, False)
                x = x[remove_mask].view(B, -1, C)
                size = size[remove_mask].view(B, -1)
                orig_idx = orig_idx[remove_mask].view(B, -1)

        if total_selected == 0:
            avg_sim = torch.zeros(B, device=device, dtype=dtype)
        else:
            avg_sim = min_sim
        return x, btree_map, avg_sim

    def merge(self, x: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        """
        Differentiable single-step merge given a direct-to-root map.
        Returns the merged tensor containing only root tokens, in order.
        """
        B, N, C = x.shape
        size = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)

        root_indices = torch.arange(N, device=x.device).expand(B, -1) + direct_to_root_map

        merged_x = torch.zeros_like(x)
        merged_size = torch.zeros_like(size)

        root_indices_expanded = root_indices.unsqueeze(-1).expand(-1, -1, C)
        merged_x.scatter_add_(1, root_indices_expanded, x * size)
        merged_size.scatter_add_(1, root_indices.unsqueeze(-1), size)

        merged_x = merged_x / (merged_size + 1e-8)

        root_mask = (direct_to_root_map == 0)
        root_tokens = merged_x[root_mask].view(B, -1, C)
        return root_tokens

    @torch.no_grad()
    def btree_to_root_map(self, merge_btree: torch.Tensor) -> torch.Tensor:
        B, N = merge_btree.shape
        device = merge_btree.device
        idx = torch.arange(N, device=device, dtype=merge_btree.dtype).view(1, N).expand(B, -1)
        is_root = (merge_btree == 0)
        masked = torch.where(is_root, idx + 1, torch.zeros_like(merge_btree))
        last_root_pos_plus1, _ = torch.cummax(masked, dim=1)
        last_root_pos = last_root_pos_plus1 - 1
        direct_to_root_map = last_root_pos - idx
        return direct_to_root_map

    def unmerge(self, y: torch.Tensor, direct_to_root_map: torch.Tensor) -> torch.Tensor:
        B, N_original = direct_to_root_map.shape
        if y.shape[1] == N_original:
            return y
        
        _, _, C = y.shape
        device = y.device

        root_mask = (direct_to_root_map == 0)
        root_rank_map = torch.cumsum(root_mask.to(torch.int32), dim=1) - 1
        source_indices = torch.arange(N_original, device=device).view(1, -1) + direct_to_root_map
        root_ranks = torch.gather(root_rank_map, 1, source_indices).clamp_min(0)
        final_x = torch.gather(y, 1, root_ranks.unsqueeze(-1).expand(-1, -1, C))
        return final_x


# Generalized subgroup-based variant (group_size>=2). For group_size==2 it matches OurToMe2.
class OurToMeK(nn.Module):
    """
    Subgroup-based adjacent merging with cyclic offsets and chaining allowed within subgroup.
    - Partition sequence into non-overlapping subgroups of length `group_size` with a start offset cycling each iteration
    - Consider only adjacent pairs within each subgroup
    - Selection allows chains and multiple sources to the same destination within an iteration (no masking)
    - Apply merges in a single batched step per iteration; outputs btree_map with offsets in {0, -1}
    - For group_size==2 this reduces exactly to OurToMe2
    """
    def __init__(self, r: float, num_iterations: int, group_size: int):
        super().__init__()
        if not (0.0 < r < 1.0):
            raise ValueError("r must be in (0,1)")
        if num_iterations < 1:
            raise ValueError("num_iterations must be >=1")
        if group_size < 2:
            raise ValueError("group_size must be >=2")
        self.r = r
        self.num_iterations = num_iterations
        self.group_size = group_size

    def merge(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        device = x.device
        btree_map = torch.zeros(B, N, device=device, dtype=torch.int64)

        total_to_merge = int(self.r * N)
        if total_to_merge == 0:
            return x, btree_map, torch.zeros(B, device=device, dtype=x.dtype)

        merges_per_iter = np.diff(np.linspace(0, total_to_merge, self.num_iterations + 1, dtype=int))

        size = torch.ones(B, N, device=device, dtype=x.dtype)
        orig_idx = torch.arange(N, device=device).expand(B, -1)

        min_sim = torch.full((B,), float('inf'), device=device, dtype=x.dtype)
        total_selected = 0
        for it, k in enumerate(merges_per_iter):
            if k == 0:
                continue
            with torch.no_grad():
                current_N = x.shape[1]
                if current_N < 2:
                    break

                # Normalize features for cosine similarity
                x_norm = F.normalize(x.detach(), dim=-1)

                # Cyclic subgroup start offset
                offset = it % self.group_size

                # Candidate pairs within each subgroup
                group_starts = torch.arange(offset, max(current_N - 1, 0), self.group_size, device=device)
                if group_starts.numel() == 0:
                    break
                rel = torch.arange(0, self.group_size - 1, device=device)
                if rel.numel() == 0:
                    break
                dst_mat = group_starts.unsqueeze(1) + rel.unsqueeze(0)
                src_mat = dst_mat + 1
                valid_mat = src_mat < current_N
                if not valid_mat.any():
                    break
                dst_flat = dst_mat[valid_mat]
                src_flat = src_mat[valid_mat]

                # Pair similarities
                sims = (x_norm[:, dst_flat, :] * x_norm[:, src_flat, :]).sum(dim=-1)  # (B, P)

                # Top-k across candidates (same candidate set for all batches)
                P = sims.shape[1]
                k_eff = min(int(k), int(P))
                if k_eff == 0:
                    continue
                top_val, top_idx = torch.topk(sims, k=k_eff, dim=1)
                sel_dst = dst_flat.unsqueeze(0).expand(B, -1).gather(1, top_idx)
                sel_src = src_flat.unsqueeze(0).expand(B, -1).gather(1, top_idx)

                # Record links in global btree (offset -1 for adjacent)
                src_orig_idx = orig_idx.gather(1, sel_src)
                btree_map.scatter_(1, src_orig_idx, torch.full_like(src_orig_idx, -1))

                # Build per-iteration btree and resolve chains to direct-to-root
                iter_btree = torch.zeros(B, current_N, device=device, dtype=torch.int64)
                iter_btree.scatter_(1, sel_src, torch.full_like(sel_src, -1))
                iter_root = OurToMeK.btree_to_root_map(iter_btree)  # offsets to final root within this iteration

                # Merge along chains: scatter-add features and sizes to roots
                roots = torch.arange(current_N, device=device).unsqueeze(0).expand(B, -1) + iter_root
                merged_x = torch.zeros_like(x)
                merged_size = torch.zeros_like(size)
                merged_x.scatter_add_(1, roots.unsqueeze(-1).expand(-1, -1, C), x * size.unsqueeze(-1))
                merged_size.scatter_add_(1, roots, size)
                x = merged_x / (merged_size.unsqueeze(-1) + 1e-8)
                size = merged_size

                # Keep only roots for next iteration
                root_mask = (iter_root == 0)
                x = x[root_mask].view(B, -1, C)
                size = size[root_mask].view(B, -1)
                orig_idx = orig_idx[root_mask].view(B, -1)

                # Record min similarity values actually used this iteration
                step_min = top_val.min(dim=1).values
                min_sim = torch.minimum(min_sim, step_min.to(min_sim.dtype))
                total_selected += int(top_val.shape[1])

        if total_selected == 0:
            avg_sim = torch.zeros(B, device=device, dtype=x.dtype)
        else:
            avg_sim = min_sim
        return x, btree_map, avg_sim

    # Reuse helpers
    btree_to_root_map = staticmethod(ToMeTopK.btree_to_root_map)
    unmerge = staticmethod(ToMeTopK.unmerge)

# Backward-compatibility alias for tests expecting GeneralizedToMe
class GeneralizedToMe(ToMeChained):
    pass