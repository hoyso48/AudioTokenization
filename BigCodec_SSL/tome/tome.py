from typing import Tuple, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

class GeneralizedTokenMerging(nn.Module):
    """Token-merging layer for 1-D sequences.

    Parameters
    ----------
    r : float
        Fraction of tokens to *remove* (0 ≤ *r* ≤ 1).
    kernel_size : int, optional
        Size of the sliding window.  ``2`` reproduces adjacent-pair merging.
    num_iterations : int, optional
        Progressive merging steps.  The overall *r* is split as evenly as
        possible across these iterations.
    causal : bool, optional
        *True* enforces left-to-right merging.  In this simplified
        implementation only affects the sign of the produced ``merge_map``.
    generator : torch.Generator, optional
        Used for deterministic random choices.
    """

    def __init__(
        self,
        r: float,
        *,
        kernel_size: int = 2,
        num_iterations: int = 1,
        causal: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()

        if not 0.0 <= r <= 1.0:
            raise ValueError("r must lie in [0, 1]")
        if kernel_size < 2:
            raise ValueError("kernel_size must be ≥ 2")
        if num_iterations < 1:
            raise ValueError("num_iterations must be ≥ 1")

        self.r = r
        self.kernel_size = kernel_size
        self.strides = kernel_size  # non-overlapping windows
        self.num_iterations = num_iterations
        self.causal = causal
        self.generator = generator

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def merge(self, metric: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform merging.

        Returns
        -------
        merged_x : torch.Tensor
            The merged representation.
        merge_map : torch.Tensor
            Integer tensor of shape ``(B, N_original)`` describing all merge
            operations (see specification for the exact semantics).
        """
        print(f"--- Initializing merge ---")
        B, N, C = metric.shape
        print(f"Input metric shape: B={B}, N={N}, C={C}")
        device, dtype = metric.device, metric.dtype

        # Early exit ------------------------------------------------------
        if self.r == 0.0:
            return metric, torch.zeros(B, N, device=device, dtype=torch.int64)

        # How many *pairs* to merge in total?
        total_pairs = int(self.r * N)
        print(f"Total pairs to merge: {total_pairs}")
        if total_pairs == 0:
            return metric, torch.zeros(B, N, device=device, dtype=torch.int64)

        # Distribute work across iterations (as evenly as possible)
        pairs_per_iter = [total_pairs // self.num_iterations] * self.num_iterations
        for i in range(total_pairs % self.num_iterations):
            pairs_per_iter[i] += 1
        print(f"Pairs per iteration: {pairs_per_iter}")

        # Book-keeping structures
        merge_map_history: List[torch.Tensor] = []
        distance_history: List[torch.Tensor] = []

        # Working tensors -------------------------------------------------
        x = metric  # will shrink
        size = torch.ones(B, N, 1, device=device)  # token sizes
        # Track mapping from current index to original index
        orig_idx = torch.arange(N, device=device).expand(B, -1).clone()
        print(f"Initial x shape: {x.shape}")
        print(f"Initial size shape: {size.shape}")
        print(f"Initial orig_idx shape: {orig_idx.shape}")

        # The *merge_map* is defined w.r.t the *original* sequence length.
        merge_map_final = torch.zeros(B, N, device=device, dtype=torch.int64)

        if isinstance(self.dst_offset, int):
            dst_offsets = [self.dst_offset] * self.num_iterations
        else:
            dst_offsets = [0] * self.num_iterations

        # -----------------------------------------------------------------
        # Iterations
        # -----------------------------------------------------------------
        start_index = 0  # offset cycles through windows implicitly
        for iter_idx, (k_pairs, dst_offset) in enumerate(zip(pairs_per_iter, dst_offsets)):
            print(f"\n--- Iteration {iter_idx+1}/{self.num_iterations}, merging {k_pairs} pairs ---")
            if k_pairs == 0 or x.shape[1] < 2:
                break

            with torch.no_grad():
                # ----------------------------------------------------------------
                # Partition sequence into windows (non-overlapping)
                # ----------------------------------------------------------------
                L = x.shape[1]
                print(f"Current sequence length L: {L}")
                remainder = L % self.kernel_size
                valid_len = L - remainder  # trailing tokens ignored (spec)
                print(f"Valid length for windows: {valid_len}")

                idx = torch.arange(valid_len, device=device)
                windows = idx.view(-1, self.kernel_size)  # (num_windows, k)
                num_windows = windows.shape[0]
                print(f"Windows shape: {windows.shape}")

                # Destination token = first element of every window (simplified)
                dst_indices = windows[:, dst_offset]  # (num_windows,)
                src_indices = windows[:, :dst_offset].reshape(-1)  # flat list of src candidates
                print(f"dst_indices shape: {dst_indices.shape}")
                print(f"src_indices shape: {src_indices.shape}")

                # Cosine similarity between *every* src token and its dst token
                x_norm = F.normalize(x, dim=-1)
                src_feat = x_norm[:, src_indices, :]  # (B, S, C)
                dst_feat = x_norm[:, dst_indices.repeat_interleave(self.kernel_size - 1), :]
                sim = (src_feat * dst_feat).sum(dim=-1)  # (B, S)
                print(f"Similarity matrix `sim` shape: {sim.shape}")

                # Each src token has exactly one dst – similarity already unique.
                # Pick globally best `k_pairs` across tokens per batch.
                # ------------------------------------------------------------
                k_effective = min(k_pairs, src_indices.numel())
                scores, topk_local = sim.topk(k_effective, dim=1)  # (B, k)
                print(f"topk scores shape: {scores.shape}, topk_local indices shape: {topk_local.shape}")

                # Map local indices back to *absolute* indices
                gather_src = src_indices[topk_local]  # (B, k)
                gather_dst = dst_indices.repeat_interleave(self.kernel_size - 1)[topk_local]
                print(f"gather_src shape: {gather_src.shape}")
                print(f"gather_dst shape: {gather_dst.shape}")

                # Map to original indices before any removal
                orig_src = orig_idx.gather(1, gather_src)
                orig_dst = orig_idx.gather(1, gather_dst)

            # --------------------------------------------------------------------
            # Actual merge (with gradient) – weighted average by `size`
            # --------------------------------------------------------------------
            src_feat = x.gather(1, gather_src.unsqueeze(-1).expand(-1, -1, C))
            dst_feat = x.gather(1, gather_dst.unsqueeze(-1).expand(-1, -1, C))
            src_size = size.gather(1, gather_src.unsqueeze(-1))
            dst_size = size.gather(1, gather_dst.unsqueeze(-1))

            new_dst_feat = (src_feat * src_size + dst_feat * dst_size) / (src_size + dst_size)
            new_dst_size = src_size + dst_size

            # Write updated dst tokens back
            x.scatter_(1, gather_dst.unsqueeze(-1).expand(-1, -1, C), new_dst_feat)
            size.scatter_(1, gather_dst.unsqueeze(-1), new_dst_size)

            # ----------------------------------------------------------------
            # Build merge_map entry BEFORE removing src tokens
            # ----------------------------------------------------------------
            rel_offset_orig = orig_dst - orig_src  # sign preserved
            if self.causal:
                rel_offset_orig = rel_offset_orig.abs() * -1

            merge_map_final.scatter_(1, orig_src, rel_offset_orig)
            print(f"Updated merge_map_final (partial): \n{merge_map_final}")

            # Remove src tokens -------------------------------------------
            mask_remove = torch.zeros(B, x.shape[1], dtype=torch.bool, device=device)
            mask_remove.scatter_(1, gather_src, True)
            keep_mask = ~mask_remove

            x = x[keep_mask].view(B, -1, C)
            size = size[keep_mask].view(B, -1, 1)
            orig_idx = orig_idx[keep_mask].view(B, -1)
            print(f"Shape after removing src tokens: x={x.shape}, size={size.shape}, orig_idx={orig_idx.shape}")

            # No need for per-iteration distance tracking in this simplified
            # non-overlapping implementation.

        print("\n--- Merge finished ---")
        return x, merge_map_final

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------
    @staticmethod
    def unmerge(merged_x: torch.Tensor, merge_map: torch.Tensor) -> torch.Tensor:
        """Invert the merge operation given the ``merge_map`` from :py:meth:`merge`."""
        B, N_original = merge_map.shape
        _, R, C = merged_x.shape
        device = merged_x.device

        # Output tensor – we will scatter values into it.
        out = torch.zeros(B, N_original, C, device=device, dtype=merged_x.dtype)

        # Find the root indices for each batch
        root_mask = merge_map == 0
        # We need to place merged_x tokens in the order they appear in the
        # *merged* tensor per batch so that the scatter assignment is correct.
        for b in range(B):
            batch_root_idx = torch.nonzero(root_mask[b], as_tuple=False).squeeze(1)
            out[b, batch_root_idx] = merged_x[b]

        # Iterate to fill in the rest of the output tensor
        max_steps = int(torch.abs(merge_map).max().item()) + 1
        filled = root_mask.clone()
        arange = torch.arange(N_original, device=device).expand(B, -1)
        for _ in range(max_steps):
            unfinished = ~filled
            if not unfinished.any():
                break
            dst_idx = arange + merge_map
            can_fill = unfinished & filled.gather(1, dst_idx)
            if not can_fill.any():
                break
            src_values = out.gather(1, dst_idx.unsqueeze(-1).expand(-1, -1, C))
            out = torch.where(can_fill.unsqueeze(-1), src_values, out)
            filled = filled | can_fill

        return out 