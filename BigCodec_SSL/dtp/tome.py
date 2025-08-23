from typing import Tuple, List, Optional, Union, Literal

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["GeneralizedTokenMerging"]

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
        B, N, C = metric.shape
        device, dtype = metric.device, metric.dtype

        # Early exit ------------------------------------------------------
        if self.r == 0.0:
            return metric, torch.zeros(B, N, device=device, dtype=torch.int64)

        # How many *pairs* to merge in total?
        total_pairs = int(self.r * N)
        if total_pairs == 0:
            return metric, torch.zeros(B, N, device=device, dtype=torch.int64)

        # Distribute work across iterations (as evenly as possible)
        pairs_per_iter = [total_pairs // self.num_iterations] * self.num_iterations
        for i in range(total_pairs % self.num_iterations):
            pairs_per_iter[i] += 1

        # Book-keeping structures
        merge_map_history: List[torch.Tensor] = []
        distance_history: List[torch.Tensor] = []

        # Working tensors -------------------------------------------------
        x = metric  # will shrink
        size = torch.ones(B, N, 1, device=device)  # token sizes
        # Track mapping from current index to original index
        orig_idx = torch.arange(N, device=device).expand(B, -1).clone()

        # The *merge_map* is defined w.r.t the *original* sequence length.
        merge_map_final = torch.zeros(B, N, device=device, dtype=torch.int64)
        merge_maps = []
        # -----------------------------------------------------------------
        # Iterations
        # -----------------------------------------------------------------
        start_index = 0  # offset cycles through windows implicitly
        for iter_idx, k_pairs in enumerate(pairs_per_iter):
            if k_pairs == 0 or x.shape[1] < 2:
                break

            with torch.no_grad():
                # ----------------------------------------------------------------
                # Partition sequence into windows (non-overlapping)
                # ----------------------------------------------------------------
                L = x.shape[1]
                remainder = L % self.kernel_size
                valid_len = L - remainder  # trailing tokens ignored (spec)

                idx = torch.arange(valid_len, device=device)
                windows = idx.view(-1, self.kernel_size)  # (num_windows, k)
                num_windows = windows.shape[0]

                # Destination token = first element of every window (simplified)
                dst_indices = windows[:, 0]  # (num_windows,)
                src_indices = windows[:, 1:].reshape(-1)  # flat list of src candidates

                # Cosine similarity between *every* src token and its dst token
                x_norm = F.normalize(x, dim=-1)
                src_feat = x_norm[:, src_indices, :]  # (B, S, C)
                dst_feat = x_norm[:, dst_indices.repeat_interleave(self.kernel_size - 1), :]
                sim = (src_feat * dst_feat).sum(dim=-1)  # (B, S)

                # Each src token has exactly one dst – similarity already unique.
                # Pick globally best `k_pairs` across tokens per batch.
                # ------------------------------------------------------------
                k_effective = min(k_pairs, src_indices.numel())
                scores, topk_local = sim.topk(k_effective, dim=1)  # (B, k)

                # Map local indices back to *absolute* indices
                gather_src = src_indices[topk_local]  # (B, k)
                gather_dst = dst_indices.repeat_interleave(self.kernel_size - 1)[topk_local]
                print(f"gather_src: {gather_src}")
                print(f"gather_dst: {gather_dst}")

                # # Map to original indices before any removal
                orig_src = orig_idx.gather(1, gather_src)
                orig_dst = orig_idx.gather(1, gather_dst)
                print(f"orig_src: {orig_src}")
                print(f"orig_dst: {orig_dst}")

                #calculate current merge map
                merge_map = torch.zeros(B, L, device=device, dtype=torch.long)
                merge_map.scatter_(1, gather_src, gather_dst - gather_src)
                print(f"merge_map: {merge_map}")
                merge_maps.append(merge_map)

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
            # if self.causal:
            #     rel_offset_orig = rel_offset_orig.abs() * -1

            merge_map_final.scatter_(1, orig_src, rel_offset_orig)

            # Remove src tokens -------------------------------------------
            mask_remove = torch.zeros(B, x.shape[1], dtype=torch.bool, device=device)
            mask_remove.scatter_(1, gather_src, True)
            keep_mask = ~mask_remove

            x = x[keep_mask].view(B, -1, C)
            size = size[keep_mask].view(B, -1, 1)
            orig_idx = orig_idx[keep_mask].view(B, -1)

            # No need for per-iteration distance tracking in this simplified
            # non-overlapping implementation.
        
        # for i in range(len(merge_maps)-1, 0, -1):
        #     print(f"{merge_maps[i-1]}, merge_maps[i-1]")
        #     prev_dst_idx = torch.where(merge_maps[i-1] == 0)[1]
        #     prev_dst_idx = prev_dst_idx.reshape(B, merge_maps[i].shape[1])
        #     print(f"{prev_dst_idx}, prev_dst_idx")
        #     distance = torch.cat([torch.zeros((B, 1), device=device, dtype=torch.long), prev_dst_idx[:,1:] - prev_dst_idx[:,:-1] - 1], dim=1)
        #     distance = torch.sign(merge_maps[i]) * distance
        #     print(f"{distance}, distance")
        #     print(f"{merge_maps[i]}, merge_maps[i]")
        #     # merge_maps[i-1][merge_maps[i-1] == 0] += merge_maps[i] + (merge_maps[i] > 0).long() * distance
        #     # merge_maps[i-1] = torch.where(merge_maps[i-1] == 0, merge_maps[i] + distance, merge_maps[i-1])
        #     merge_maps[i-1].masked_scatter_(merge_maps[i-1] == 0, merge_maps[i] + distance)
        #     print(f"{merge_maps[i-1]}, merge_maps[i-1]")

        return x, final_merge_map #merge_maps[0]

    def merge_map_to_btree(self, merge_map: torch.Tensor) -> torch.Tensor:
        B, N = merge_map.shape
        for i in range(self.num_iterations):
            is_root = merge_map == 0
            merge_map

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

# class GeneralizedTokenMerging(nn.Module):
#     """Token-merging layer for 1-D sequences.

#     Parameters
#     ----------
#     r : float
#         Fraction of tokens to *remove* (0 ≤ *r* ≤ 1).
#     kernel_size : int, optional
#         Size of the sliding window.  ``2`` reproduces adjacent-pair merging.
#     num_iterations : int, optional
#         Progressive merging steps.  The overall *r* is split as evenly as
#         possible across these iterations.
#     causal : bool, optional
#         *True* enforces left-to-right merging.  In this simplified
#         implementation only affects the sign of the produced ``merge_map``.
#     generator : torch.Generator, optional
#         Used for deterministic random choices.
#     """

#     def __init__(
#         self,
#         r: float,
#         *,
#         kernel_size: int = 2,
#         num_iterations: int = 1,
#         causal: bool = False,
#         generator: Optional[torch.Generator] = None,
#         dst_offset: Union[int, str] = 0,
#     ) -> None:
#         super().__init__()

#         if not 0.0 <= r <= 1.0:
#             raise ValueError("r must lie in [0, 1]")
#         if kernel_size < 2:
#             raise ValueError("kernel_size must be ≥ 2")
#         if num_iterations < 1:
#             raise ValueError("num_iterations must be ≥ 1")

#         self.r = r
#         self.kernel_size = kernel_size
#         self.strides = kernel_size  # non-overlapping windows
#         self.num_iterations = num_iterations
#         self.causal = causal
#         self.generator = generator
#         self.dst_offset = dst_offset
#     # ------------------------------------------------------------------
#     # Public interface
#     # ------------------------------------------------------------------
#     def merge(self, metric: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Perform merging.

#         Returns
#         -------
#         merged_x : torch.Tensor
#             The merged representation.
#         merge_map : torch.Tensor
#             Integer tensor of shape ``(B, N_original)`` describing all merge
#             operations (see specification for the exact semantics).
#         """
#         print(f"--- Initializing merge ---")
#         B, N, C = metric.shape
#         print(f"Input metric shape: B={B}, N={N}, C={C}")
#         device, dtype = metric.device, metric.dtype

#         # Early exit ------------------------------------------------------
#         if self.r == 0.0:
#             return metric, torch.zeros(B, N, device=device, dtype=torch.int64)

#         # How many *pairs* to merge in total?
#         total_pairs = int(self.r * N)
#         print(f"Total pairs to merge: {total_pairs}")
#         if total_pairs == 0:
#             return metric, torch.zeros(B, N, device=device, dtype=torch.int64)

#         # Distribute work across iterations (as evenly as possible)
#         pairs_per_iter = [total_pairs // self.num_iterations] * self.num_iterations
#         for i in range(total_pairs % self.num_iterations):
#             pairs_per_iter[i] += 1
#         print(f"Pairs per iteration: {pairs_per_iter}")

#         # Book-keeping structures
#         merge_map_history: List[torch.Tensor] = []
#         distance_history: List[torch.Tensor] = []

#         # Working tensors -------------------------------------------------
#         x = metric  # will shrink
#         size = torch.ones(B, N, 1, device=device)  # token sizes
#         # Track mapping from current index to original index
#         orig_idx = torch.arange(N, device=device).expand(B, -1).clone()
#         print(f"Initial x shape: {x.shape}")
#         print(f"Initial size shape: {size.shape}")
#         print(f"Initial orig_idx shape: {orig_idx.shape}")

#         # The *merge_map* is defined w.r.t the *original* sequence length.
#         merge_map_final = torch.zeros(B, N, device=device, dtype=torch.int64)

#         # Pre-calculate dst_offsets for each iteration
#         if isinstance(self.dst_offset, int):
#             dst_offsets = [self.dst_offset] * self.num_iterations
#         elif self.dst_offset == "cycle":
#             dst_offsets = [i % self.kernel_size for i in range(self.num_iterations)]
#         elif self.dst_offset == "random":
#             dst_offsets = torch.randint(
#                 0, self.kernel_size, (self.num_iterations,), generator=self.generator
#             ).tolist()
#         # -----------------------------------------------------------------
#         # Iterations
#         # -----------------------------------------------------------------
#         start_index = 0  # offset cycles through windows implicitly
#         for iter_idx, (k_pairs, dst_offset) in enumerate(zip(pairs_per_iter, dst_offsets)):
#             # print(f"\n--- Iteration {iter_idx+1}/{self.num_iterations}, merging {k_pairs} pairs ---")
#             if k_pairs == 0 or x.shape[1] < 2:
#                 break

#             with torch.no_grad():
#                 # ----------------------------------------------------------------
#                 # Partition sequence into windows (non-overlapping)
#                 # ----------------------------------------------------------------
#                 L = x.shape[1]
#                 # print(f"Current sequence length L: {L}")
#                 remainder = L % self.kernel_size
#                 valid_len = L - remainder  # trailing tokens ignored (spec)
#                 # print(f"Valid length for windows: {valid_len}")

#                 idx = torch.arange(valid_len, device=device)
#                 windows = idx.view(-1, self.kernel_size)  # (num_windows, k)
#                 num_windows = windows.shape[0]
#                 # print(f"Windows shape: {windows.shape}")

#                 # Destination token = first element of every window (simplified)
#                 # dst_indices = windows[:, dst_offset]  # (num_windows,)
#                 # src_indices = windows[:, :dst_offset].reshape(-1)  # flat list of src candidates
#                 dst_indices = windows[:, dst_offset]
#                 src_mask = torch.ones(self.kernel_size, dtype=torch.bool)
#                 src_mask[dst_offset] = False
#                 src_indices = windows[:, src_mask].reshape(-1)
#                 # print(f"dst_indices shape: {dst_indices.shape}")
#                 # print(f"src_indices shape: {src_indices.shape}")

#                 # Cosine similarity between *every* src token and its dst token
#                 x_norm = F.normalize(x, dim=-1)
#                 src_feat = x_norm[:, src_indices, :]  # (B, S, C)
#                 dst_feat = x_norm[:, dst_indices.repeat_interleave(self.kernel_size - 1), :]
#                 sim = (src_feat * dst_feat).sum(dim=-1)  # (B, S)
#                 # print(f"Similarity matrix `sim` shape: {sim.shape}")

#                 # Each src token has exactly one dst – similarity already unique.
#                 # Pick globally best `k_pairs` across tokens per batch.
#                 # ------------------------------------------------------------
#                 k_effective = min(k_pairs, src_indices.numel())
#                 scores, topk_local = sim.topk(k_effective, dim=1)  # (B, k)
#                 # print(f"topk scores shape: {scores.shape}, topk_local indices shape: {topk_local.shape}")

#                 # Map local indices back to *absolute* indices
#                 gather_src = src_indices[topk_local]  # (B, k)
#                 gather_dst = dst_indices.repeat_interleave(self.kernel_size - 1)[topk_local]
#                 # print(f"gather_src shape: {gather_src.shape}")
#                 # print(f"gather_dst shape: {gather_dst.shape}")

#                 # Map to original indices before any removal
#                 orig_src = orig_idx.gather(1, gather_src)
#                 orig_dst = orig_idx.gather(1, gather_dst)

#             # --------------------------------------------------------------------
#             # Actual merge (with gradient) – weighted average by `size`
#             # --------------------------------------------------------------------
#             src_feat = x.gather(1, gather_src.unsqueeze(-1).expand(-1, -1, C))
#             dst_feat = x.gather(1, gather_dst.unsqueeze(-1).expand(-1, -1, C))
#             src_size = size.gather(1, gather_src.unsqueeze(-1))
#             dst_size = size.gather(1, gather_dst.unsqueeze(-1))

#             new_dst_feat = (src_feat * src_size + dst_feat * dst_size) / (src_size + dst_size)
#             new_dst_size = src_size + dst_size

#             # Write updated dst tokens back
#             x.scatter_(1, gather_dst.unsqueeze(-1).expand(-1, -1, C), new_dst_feat)
#             size.scatter_(1, gather_dst.unsqueeze(-1), new_dst_size)

#             # ----------------------------------------------------------------
#             # Build merge_map entry BEFORE removing src tokens
#             # ----------------------------------------------------------------
#             rel_offset_orig = orig_dst - orig_src  # sign preserved
#             if self.causal:
#                 rel_offset_orig = rel_offset_orig.abs() * -1

#             merge_map_final.scatter_(1, orig_src, rel_offset_orig)
#             # print(f"Updated merge_map_final (partial): \n{merge_map_final}")

#             # Remove src tokens -------------------------------------------
#             mask_remove = torch.zeros(B, x.shape[1], dtype=torch.bool, device=device)
#             mask_remove.scatter_(1, gather_src, True)
#             keep_mask = ~mask_remove

#             x = x[keep_mask].view(B, -1, C)
#             size = size[keep_mask].view(B, -1, 1)
#             orig_idx = orig_idx[keep_mask].view(B, -1)
#             print(f"Shape after removing src tokens: x={x.shape}, size={size.shape}, orig_idx={orig_idx.shape}")

#             # No need for per-iteration distance tracking in this simplified
#             # non-overlapping implementation.

#         # print("\n--- Merge finished ---")
#         return x, merge_map_final

#     # ------------------------------------------------------------------
#     # Reconstruction
#     # ------------------------------------------------------------------
#     @staticmethod
#     def unmerge(merged_x: torch.Tensor, merge_map: torch.Tensor) -> torch.Tensor:
#         """Invert the merge operation given the ``merge_map`` from :py:meth:`merge`."""
#         B, N_original = merge_map.shape
#         _, R, C = merged_x.shape
#         device = merged_x.device

#         # Output tensor – we will scatter values into it.
#         out = torch.zeros(B, N_original, C, device=device, dtype=merged_x.dtype)

#         # Find the root indices for each batch
#         root_mask = merge_map == 0
#         # We need to place merged_x tokens in the order they appear in the
#         # *merged* tensor per batch so that the scatter assignment is correct.
#         for b in range(B):
#             batch_root_idx = torch.nonzero(root_mask[b], as_tuple=False).squeeze(1)
#             out[b, batch_root_idx] = merged_x[b]

#         # Iterate to fill in the rest of the output tensor
#         max_steps = int(torch.abs(merge_map).max().item()) + 1
#         filled = root_mask.clone()
#         arange = torch.arange(N_original, device=device).expand(B, -1)
#         for _ in range(max_steps):
#             unfinished = ~filled
#             if not unfinished.any():
#                 break
#             dst_idx = arange + merge_map
#             can_fill = unfinished & filled.gather(1, dst_idx)
#             if not can_fill.any():
#                 break
#             src_values = out.gather(1, dst_idx.unsqueeze(-1).expand(-1, -1, C))
#             out = torch.where(can_fill.unsqueeze(-1), src_values, out)
#             filled = filled | can_fill

#         return out

#     def forward(self, metric: torch.Tensor) -> torch.Tensor:
#         return self.merge(metric)


# class GeneralizedTokenMerging(nn.Module):
#     """Token-merging layer for 1-D sequences.

#     Parameters
#     ----------
#     r : float
#         Fraction of tokens to *remove* (0 ≤ *r* ≤ 1).
#     kernel_size : int, optional
#         Size of the sliding window.  ``2`` reproduces adjacent-pair merging.
#     num_iterations : int, optional
#         Progressive merging steps.  The overall *r* is split as evenly as
#         possible across these iterations.
#     causal : bool, optional
#         *True* enforces left-to-right merging.  In this simplified
#         implementation only affects the sign of the produced ``merge_map``.
#     dst_offset : Union[int, str], optional
#         Offset of the destination token in the window. If an int, it must be in [0, kernel_size - 1].
#         If 'random', the destination token is chosen randomly. If 'cycle', the destination token is
#         chosen in a cyclic manner. Default is 0.
#     generator : torch.Generator, optional
#         Used for deterministic random choices.
#     """

#     def __init__(
#         self,
#         r: float,
#         *,
#         kernel_size: int = 2,
#         num_iterations: int = 1,
#         causal: bool = False,
#         dst_offset: Union[int, str] = 0,
#         padding: Optional[Literal['same', 'pre', 'post', 'random']] = None,
#         generator: Optional[torch.Generator] = None,
#     ) -> None:
#         super().__init__()

#         if not 0.0 <= r <= 1.0:
#             raise ValueError("r must lie in [0, 1]")
#         if kernel_size < 2:
#             raise ValueError("kernel_size must be ≥ 2")
#         if num_iterations < 1:
#             raise ValueError("num_iterations must be ≥ 1")

#         self.r = r
#         self.kernel_size = kernel_size
#         self.strides = kernel_size  # non-overlapping windows
#         self.num_iterations = num_iterations
#         self.causal = causal
#         self.dst_offset = dst_offset
#         self.padding = padding
#         self.generator = generator

#         if self.causal:
#             if self.dst_offset != 0:
#                 print(f"Warning: dst_offset is set to 0 because causal is True")
#             self.dst_offset = 0

#         if isinstance(self.dst_offset, int):
#             if not 0 <= self.dst_offset < self.kernel_size:
#                 raise ValueError(
#                     f"dst_offset as an int must be in [0, {kernel_size - 1}], but got {self.dst_offset}"
#                 )
#         elif isinstance(self.dst_offset, str):
#             if self.dst_offset not in ("random", "cycle"):
#                 raise ValueError(
#                     f"dst_offset as a string must be 'random' or 'cycle', but got {self.dst_offset!r}"
#                 )
#         else:
#             raise TypeError(
#                 f"dst_offset must be an int or a string, not {type(self.dst_offset).__name__}"
#             )

#         if self.padding is not None and self.padding not in ('same', 'pre', 'post', 'random'):
#             raise ValueError(
#                 f"padding must be one of 'same', 'pre', 'post', 'random', or None, but got {self.padding!r}"
#             )

#     # ------------------------------------------------------------------
#     # Public interface
#     # ------------------------------------------------------------------
#     def merge(self, metric: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Perform merging.

#         Returns
#         -------
#         merged_x : torch.Tensor
#             The merged representation.
#         merge_map : torch.Tensor
#             Integer tensor of shape ``(B, N_original)`` describing all merge
#             operations (see specification for the exact semantics).
#         """
#         # print(f"--- Initializing merge ---")
#         B, N, C = metric.shape
#         # print(f"Input metric shape: B={B}, N={N}, C={C}")
#         device, dtype = metric.device, metric.dtype

#         # Early exit ------------------------------------------------------
#         if self.r == 0.0:
#             return metric, torch.zeros(B, N, device=device, dtype=torch.int64)

#         # How many *pairs* to merge in total?
#         total_pairs = int(self.r * N)
#         # print(f"Total pairs to merge: {total_pairs}")
#         if total_pairs == 0:
#             return metric, torch.zeros(B, N, device=device, dtype=torch.int64)

#         # Distribute work across iterations (as evenly as possible)
#         pairs_per_iter = [total_pairs // self.num_iterations] * self.num_iterations
#         for i in range(total_pairs % self.num_iterations):
#             pairs_per_iter[i] += 1
#         print(f"Pairs per iteration: {pairs_per_iter}")

#         # Book-keeping structures
#         merge_map_history: List[torch.Tensor] = []
#         distance_history: List[torch.Tensor] = []

#         # Working tensors -------------------------------------------------
#         x = metric  # will shrink
#         size = torch.ones(B, N, 1, device=device)  # token sizes
#         # Track mapping from current index to original index
#         orig_idx = torch.arange(N, device=device).expand(B, -1).clone()
#         # print(f"Initial x shape: {x.shape}")
#         # print(f"Initial size shape: {size.shape}")
#         # print(f"Initial orig_idx shape: {orig_idx.shape}")

#         # The *merge_map* is defined w.r.t the *original* sequence length.
#         merge_map_final = torch.zeros(B, N, device=device, dtype=torch.int64)

#         # Pre-calculate dst_offsets for each iteration
#         if isinstance(self.dst_offset, int):
#             dst_offsets = [self.dst_offset] * self.num_iterations
#         elif self.dst_offset == "cycle":
#             dst_offsets = [i % self.kernel_size for i in range(self.num_iterations)]
#         elif self.dst_offset == "random":
#             dst_offsets = torch.randint(
#                 0, self.kernel_size, (self.num_iterations,), generator=self.generator
#             ).tolist()

#         # -----------------------------------------------------------------
#         # Iterations
#         # -----------------------------------------------------------------
#         for iter_idx, (k_pairs, dst_offset) in enumerate(zip(pairs_per_iter, dst_offsets)):
#             # print(f"\n--- Iteration {iter_idx+1}/{self.num_iterations}, merging {k_pairs} pairs, dst_offset={dst_offset} ---")
#             if k_pairs == 0 or x.shape[1] < 2:
#                 break

#             with torch.no_grad():
#                 # ----------------------------------------------------------------
#                 # Partition sequence into windows
#                 # ----------------------------------------------------------------
#                 L = x.shape[1]
#                 # print(f"Current sequence length L: {L}")
#                 pad_pre = 0

#                 if self.padding is not None:
#                     padding_needed = (self.kernel_size - (L % self.kernel_size)) % self.kernel_size
#                     if padding_needed > 0:
#                         if self.padding == 'pre':
#                             pad_pre = padding_needed
#                             pad_post = 0
#                         elif self.padding == 'post':
#                             pad_pre = 0
#                             pad_post = padding_needed
#                         elif self.padding == 'same':
#                             pad_pre = padding_needed // 2
#                             pad_post = padding_needed - pad_pre
#                         elif self.padding == 'random':
#                             pad_pre = torch.randint(0, padding_needed + 1, (1,), generator=self.generator).item()
#                             pad_post = padding_needed - pad_pre
                        
#                         x = F.pad(x, (0, 0, pad_pre, pad_post), value=0)
#                         size = F.pad(size, (0, 0, pad_pre, pad_post), value=0) # Also pad size
#                         orig_idx = F.pad(orig_idx, (pad_pre, pad_post), value=-1) # Pad orig_idx with -1
#                         L = x.shape[1]
#                         # print(f"Padded length: {L}, Pre: {pad_pre}, Post: {pad_post}")
                
#                 padding_mask = (orig_idx != -1) # Padded tokens are -1

#                 remainder = L % self.kernel_size
#                 valid_len = L - remainder
#                 # print(f"Valid length for windows: {valid_len}")

#                 if valid_len == 0: continue

#                 idx = torch.arange(valid_len, device=device)
#                 windows = idx.view(-1, self.kernel_size)
#                 num_windows = windows.shape[0]
#                 # print(f"Windows shape: {windows.shape}")

#                 # Designate dst and src tokens
#                 dst_indices = windows[:, dst_offset]
#                 src_mask = torch.ones(self.kernel_size, dtype=torch.bool)
#                 src_mask[dst_offset] = False
#                 src_indices = windows[:, src_mask].reshape(-1)

#                 # Cosine similarity calculation
#                 x_norm = F.normalize(x, dim=-1)
#                 src_feat = x_norm[:, src_indices, :]
#                 dst_feat = x_norm[:, dst_indices.repeat_interleave(self.kernel_size - 1), :]
#                 sim = (src_feat * dst_feat).sum(dim=-1)
#                 # print(f"Similarity matrix `sim` shape: {sim.shape}")

#                 # Mask out padded tokens from being candidates
#                 # A token is invalid if it, or its destination, was padding
#                 src_padding_mask = padding_mask.gather(1, src_indices.expand(B, -1))
#                 dst_padding_mask = padding_mask.gather(1, dst_indices.repeat_interleave(self.kernel_size-1).expand(B,-1))
                
#                 # Invalidate similarities involving padded tokens
#                 sim[~(src_padding_mask & dst_padding_mask)] = -float('inf')

#                 # Top-K selection
#                 k_effective = min(k_pairs, src_indices.numel())
#                 if sim.numel() == 0: continue # No valid pairs to merge
                
#                 scores, topk_local = sim.topk(k_effective, dim=1)

#                 # Filter out -inf scores from padding
#                 valid_merges = scores > -float('inf')
                
#                 topk_local = topk_local[valid_merges]
                
#                 if topk_local.numel() == 0: continue

#                 # Map local indices back to absolute indices in the current tensor x
#                 gather_src = src_indices[topk_local]
#                 gather_dst = dst_indices.repeat_interleave(self.kernel_size - 1)[topk_local]

#                 # Expand to batch size
#                 gather_src = gather_src.expand(B,-1)
#                 gather_dst = gather_dst.expand(B,-1)

#                 # Map to original indices before any removal
#                 orig_src = orig_idx.gather(1, gather_src)
#                 orig_dst = orig_idx.gather(1, gather_dst)

#             # --------------------------------------------------------------------
#             # Actual merge (with gradient)
#             # --------------------------------------------------------------------
#             src_feat = x.gather(1, gather_src.unsqueeze(-1).expand(-1, -1, C))
#             dst_feat = x.gather(1, gather_dst.unsqueeze(-1).expand(-1, -1, C))
#             src_size = size.gather(1, gather_src.unsqueeze(-1))
#             dst_size = size.gather(1, gather_dst.unsqueeze(-1))

#             new_dst_feat = (src_feat * src_size + dst_feat * dst_size) / (src_size + dst_size)
#             new_dst_size = src_size + dst_size

#             x.scatter_(1, gather_dst.unsqueeze(-1).expand(-1, -1, C), new_dst_feat)
#             size.scatter_(1, gather_dst.unsqueeze(-1), new_dst_size)

#             # ----------------------------------------------------------------
#             # Build merge_map entry and remove src tokens
#             # ----------------------------------------------------------------
#             rel_offset_orig = orig_dst - orig_src
#             if self.causal:
#                 rel_offset_orig = rel_offset_orig.abs() * -1
            
#             merge_map_final.scatter_(1, orig_src, rel_offset_orig)
#             # print(f"Updated merge_map_final (partial): \n{merge_map_final}")

#             mask_remove = torch.zeros(B, L, dtype=torch.bool, device=device)
#             mask_remove.scatter_(1, gather_src, True)
            
#             # Also remove the padding that was added in this iteration
#             if pad_pre > 0:
#                 mask_remove[:, :pad_pre] = True
#             if 'pad_post' in locals() and pad_post > 0:
#                 mask_remove[:, -pad_post:] = True

#             keep_mask = ~mask_remove
#             x = x[keep_mask].view(B, -1, C)
#             size = size[keep_mask].view(B, -1, 1)
#             orig_idx = orig_idx[keep_mask].view(B, -1)
#             print(f"Shape after removing src tokens & padding: x={x.shape}")

#         # print("\n--- Merge finished ---")
#         return x, merge_map_final

#     # ------------------------------------------------------------------
#     # Reconstruction
#     # ------------------------------------------------------------------
#     @staticmethod
#     def unmerge(merged_x: torch.Tensor, merge_map: torch.Tensor) -> torch.Tensor:
#         """Invert the merge operation given the ``merge_map`` from :py:meth:`merge`."""
#         B, N_original = merge_map.shape
#         _, R, C = merged_x.shape
#         device = merged_x.device

#         # Output tensor – we will scatter values into it.
#         out = torch.zeros(B, N_original, C, device=device, dtype=merged_x.dtype)

#         # Find the root indices for each batch
#         root_mask = merge_map == 0
#         # We need to place merged_x tokens in the order they appear in the
#         # *merged* tensor per batch so that the scatter assignment is correct.
#         for b in range(B):
#             batch_root_idx = torch.nonzero(root_mask[b], as_tuple=False).squeeze(1)
#             out[b, batch_root_idx] = merged_x[b]

#         # Iterate to fill in the rest of the output tensor
#         max_steps = int(torch.abs(merge_map).max().item()) + 1
#         filled = root_mask.clone()
#         arange = torch.arange(N_original, device=device).expand(B, -1)
#         for _ in range(max_steps):
#             unfinished = ~filled
#             if not unfinished.any():
#                 break
#             dst_idx = arange + merge_map
#             # Clamp dst_idx to be within valid range to avoid errors from large offsets
#             # dst_idx.clamp_(0, N_original - 1)
#             can_fill = unfinished & filled.gather(1, dst_idx)
#             if not can_fill.any():
#                 break
#             src_values = out.gather(1, dst_idx.unsqueeze(-1).expand(-1, -1, C))
#             out = torch.where(can_fill.unsqueeze(-1), src_values, out)
#             filled = filled | can_fill

#         return out

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.merge(x)



# from typing import Tuple, List, Optional, Union, Literal

# import torch
# from torch import nn
# from torch.nn import functional as F

# class GeneralizedTokenMerging(nn.Module):
#     """Token-merging layer for 1-D sequences.

#     Parameters
#     ----------
#     r : float
#         Fraction of tokens to *remove* (0 ≤ *r* ≤ 1).
#     kernel_size : int, optional
#         Size of the sliding window.  ``2`` reproduces adjacent-pair merging.
#     num_iterations : int, optional
#         Progressive merging steps.  The overall *r* is split as evenly as
#         possible across these iterations.
#     causal : bool, optional
#         *True* enforces left-to-right merging.  In this simplified
#         implementation only affects the sign of the produced ``merge_map``.
#     dst_offset : Union[int, str], optional
#         Offset of the destination token in the window. If an int, it must be in [0, kernel_size - 1].
#         If 'random', the destination token is chosen randomly. If 'cycle', the destination token is
#         chosen in a cyclic manner. Default is 0.
#     generator : torch.Generator, optional
#         Used for deterministic random choices.
#     """

#     def __init__(
#         self,
#         r: float,
#         *,
#         kernel_size: int = 2,
#         strides: int = 1,
#         num_iterations: int = 1,
#         causal: bool = False,
#         dst_offset: Union[int, str] = 0,
#         padding: Optional[Literal['same', 'pre', 'post', 'random']] = None,
#         generator: Optional[torch.Generator] = None,
#     ) -> None:
#         super().__init__()

#         if not 0.0 <= r <= 1.0:
#             raise ValueError("r must lie in [0, 1]")
#         if kernel_size < 2:
#             raise ValueError("kernel_size must be ≥ 2")
#         if strides < 1:
#             raise ValueError("strides must be ≥ 1")
#         if num_iterations < 1:
#             raise ValueError("num_iterations must be ≥ 1")

#         self.r = r
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.num_iterations = num_iterations
#         self.causal = causal
#         self.dst_offset = dst_offset
#         self.padding = padding
#         self.generator = generator

#         if self.causal:
#             if self.dst_offset != 0:
#                 print(f"Warning: dst_offset is set to 0 because causal is True")
#             self.dst_offset = 0

#         if isinstance(self.dst_offset, int):
#             if not 0 <= self.dst_offset < self.kernel_size:
#                 raise ValueError(
#                     f"dst_offset as an int must be in [0, {kernel_size - 1}], but got {self.dst_offset}"
#                 )
#         elif isinstance(self.dst_offset, str):
#             if self.dst_offset not in ("random", "cycle"):
#                 raise ValueError(
#                     f"dst_offset as a string must be 'random' or 'cycle', but got {self.dst_offset!r}"
#                 )
#         else:
#             raise TypeError(
#                 f"dst_offset must be an int or a string, not {type(self.dst_offset).__name__}"
#             )

#         if self.padding is not None and self.padding not in ('same', 'pre', 'post', 'random'):
#             raise ValueError(
#                 f"padding must be one of 'same', 'pre', 'post', 'random', or None, but got {self.padding!r}"
#             )

#     # ------------------------------------------------------------------
#     # Public interface
#     # ------------------------------------------------------------------
#     def merge(self, metric: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Perform merging.

#         Returns
#         -------
#         merged_x : torch.Tensor
#             The merged representation.
#         merge_map : torch.Tensor
#             Integer tensor of shape ``(B, N_original)`` describing all merge
#             operations (see specification for the exact semantics).
#         """
#         print(f"--- Initializing merge ---")
#         B, N, C = metric.shape
#         print(f"Input metric shape: B={B}, N={N}, C={C}")
#         device, dtype = metric.device, metric.dtype

#         # Early exit ------------------------------------------------------
#         if self.r == 0.0:
#             return metric, torch.zeros(B, N, device=device, dtype=torch.int64)

#         # How many *pairs* to merge in total?
#         total_pairs = int(self.r * N)
#         print(f"Total pairs to merge: {total_pairs}")
#         if total_pairs == 0:
#             return metric, torch.zeros(B, N, device=device, dtype=torch.int64)

#         # Distribute work across iterations (as evenly as possible)
#         pairs_per_iter = [total_pairs // self.num_iterations] * self.num_iterations
#         for i in range(total_pairs % self.num_iterations):
#             pairs_per_iter[i] += 1
#         print(f"Pairs per iteration: {pairs_per_iter}")

#         # Book-keeping structures
#         merge_map_history: List[torch.Tensor] = []
#         distance_history: List[torch.Tensor] = []

#         # Working tensors -------------------------------------------------
#         x = metric  # will shrink
#         size = torch.ones(B, N, 1, device=device)  # token sizes
#         # Track mapping from current index to original index
#         orig_idx = torch.arange(N, device=device).expand(B, -1).clone()
#         print(f"Initial x shape: {x.shape}")
#         print(f"Initial size shape: {size.shape}")
#         print(f"Initial orig_idx shape: {orig_idx.shape}")

#         # The *merge_map* is defined w.r.t the *original* sequence length.
#         merge_map_final = torch.zeros(B, N, device=device, dtype=torch.int64)

#         # Pre-calculate dst_offsets for each iteration
#         if isinstance(self.dst_offset, int):
#             dst_offsets = [self.dst_offset] * self.num_iterations
#         elif self.dst_offset == "cycle":
#             dst_offsets = [i % self.kernel_size for i in range(self.num_iterations)]
#         elif self.dst_offset == "random":
#             dst_offsets = torch.randint(
#                 0, self.kernel_size, (self.num_iterations,), generator=self.generator
#             ).tolist()

#         # -----------------------------------------------------------------
#         # Iterations
#         # -----------------------------------------------------------------
#         for iter_idx, (k_pairs, dst_offset) in enumerate(zip(pairs_per_iter, dst_offsets)):
#             print(f"\n--- Iteration {iter_idx+1}/{self.num_iterations}, merging {k_pairs} pairs, dst_offset={dst_offset} ---")
#             if k_pairs == 0 or x.shape[1] < 2:
#                 break

#             with torch.no_grad():
#                 # ----------------------------------------------------------------
#                 # Partition sequence into windows
#                 # ----------------------------------------------------------------
#                 L = x.shape[1]
#                 print(f"Current sequence length L: {L}")
#                 pad_pre = 0

#                 if self.padding is not None:
#                     # To be perfectly divisible, the padded length L' must satisfy (L' - kernel_size) % strides == 0
#                     # L' = L + pad_pre + pad_post
#                     # (L + padding - kernel_size) % strides == 0
#                     # Let offset = (L - kernel_size) % strides. We need to add `(strides - offset) % strides`
#                     offset = (L - self.kernel_size) % self.strides
#                     padding_needed = (self.strides - offset) % self.strides

#                     if padding_needed > 0:
#                         if self.padding == 'pre':
#                             pad_pre = padding_needed
#                             pad_post = 0
#                         elif self.padding == 'post':
#                             pad_pre = 0
#                             pad_post = padding_needed
#                         elif self.padding == 'same':
#                             pad_pre = padding_needed // 2
#                             pad_post = padding_needed - pad_pre
#                         elif self.padding == 'random':
#                             pad_pre = torch.randint(0, padding_needed + 1, (1,), generator=self.generator).item()
#                             pad_post = padding_needed - pad_pre
                        
#                         x = F.pad(x, (0, 0, pad_pre, pad_post), value=0)
#                         size = F.pad(size, (0, 0, pad_pre, pad_post), value=0) # Also pad size
#                         orig_idx = F.pad(orig_idx, (pad_pre, pad_post), value=-1) # Pad orig_idx with -1
#                         L = x.shape[1]
#                         print(f"Padded length: {L}, Pre: {pad_pre}, Post: {pad_post}")
                
#                 padding_mask = (orig_idx != -1) # Padded tokens are -1

#                 # Use unfold to create overlapping windows
#                 # (L - kernel_size) must be divisible by strides
#                 valid_len = L - (L - self.kernel_size) % self.strides
                
#                 # Check if there's enough length to create at least one window
#                 if L < self.kernel_size:
#                     continue

#                 idx = torch.arange(L, device=device)
#                 windows = idx.unfold(0, self.kernel_size, self.strides) # This is the key change
#                 num_windows = windows.shape[0]
#                 print(f"Windows shape (from unfold): {windows.shape}")

#                 # --- Optimized Masking Strategy ---
#                 # 1. Create a single integer "status" mask for the whole sequence
#                 #    0: Padded, 1: Normal, 2: Destination
#                 status_mask = torch.ones(B, L, dtype=torch.int8, device=device)
#                 status_mask[~padding_mask] = 0  # Mark padded tokens

#                 all_dst_indices = windows[:, dst_offset]
#                 is_dst_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
#                 is_dst_mask.scatter_(1, all_dst_indices.expand(B, -1), True)
#                 status_mask[is_dst_mask] = 2  # Mark destination tokens

#                 # 2. Unfold the single status mask
#                 windowed_status_mask = status_mask.squeeze(0).unfold(0, self.kernel_size, self.strides)

#                 # 3. Determine validity from the unfolded status mask
#                 src_mask_kernel = torch.ones(self.kernel_size, dtype=torch.bool)
#                 src_mask_kernel[dst_offset] = False
                
#                 src_statuses = windowed_status_mask[:, src_mask_kernel]
#                 dst_statuses = windowed_status_mask[:, dst_offset]

#                 # A src is valid ONLY if its status is 1 (Normal)
#                 valid_src_mask = (src_statuses == 1)
#                 # A dst is valid if its status is 1 (Normal) or 2 (Destination)
#                 valid_dst_mask = (dst_statuses >= 1)
                
#                 # 4. Flatten masks and indices for similarity calculation
#                 src_indices_in_window = windows[:, src_mask_kernel]
#                 src_indices = src_indices_in_window[valid_src_mask]
                
#                 # Find corresponding destinations
#                 dst_indices_in_window = all_dst_indices.unsqueeze(1).expand(-1, self.kernel_size - 1)
#                 dst_indices = dst_indices_in_window[valid_src_mask]

#                 # --- End of Optimized Masking ---

#                 # Cosine similarity calculation for all VALID pairs
#                 x_norm = F.normalize(x, dim=-1)
#                 src_feat = x_norm[:, src_indices, :]
#                 dst_feat = x_norm[:, dst_indices, :]
#                 sim = (src_feat * dst_feat).sum(dim=-1)
#                 print(f"Similarity matrix `sim` shape: {sim.shape}")


#                 # Top-K selection
#                 k_effective = min(k_pairs, sim.numel())
#                 if sim.numel() == 0: continue
                
#                 scores, topk_indices = sim.topk(k_effective, dim=-1)
                
#                 gather_src = src_indices[topk_indices].expand(B,-1)
#                 gather_dst = dst_indices[topk_indices].expand(B,-1)
                
#                 # Map to original indices before any removal
#                 orig_src = orig_idx.gather(1, gather_src)
#                 orig_dst = orig_idx.gather(1, gather_dst)
                
#                 # Check for and remove self-merges (if src and dst are same original token)
#                 not_self_merge = (orig_src != orig_dst)
#                 if not not_self_merge.all():
#                     orig_src = orig_src[not_self_merge]
#                     orig_dst = orig_dst[not_self_merge]
#                     gather_src = gather_src[not_self_merge]
#                     gather_dst = gather_dst[not_self_merge]

#             # --------------------------------------------------------------------
#             # Actual merge (with gradient)
#             # --------------------------------------------------------------------
#             if gather_src.numel() == 0: continue

#             src_feat = x.gather(1, gather_src.unsqueeze(-1).expand(-1, -1, C))
#             dst_feat = x.gather(1, gather_dst.unsqueeze(-1).expand(-1, -1, C))
#             src_size = size.gather(1, gather_src.unsqueeze(-1))
#             dst_size = size.gather(1, gather_dst.unsqueeze(-1))

#             new_dst_feat = (src_feat * src_size + dst_feat * dst_size) / (src_size + dst_size)
#             new_dst_size = src_size + dst_size

#             x.scatter_(1, gather_dst.unsqueeze(-1).expand(-1, -1, C), new_dst_feat)
#             size.scatter_(1, gather_dst.unsqueeze(-1), new_dst_size)

#             # ----------------------------------------------------------------
#             # Build merge_map entry and remove src tokens
#             # ----------------------------------------------------------------
#             rel_offset_orig = orig_dst - orig_src
#             if self.causal:
#                 rel_offset_orig = rel_offset_orig.abs() * -1
            
#             merge_map_final.scatter_(1, orig_src, rel_offset_orig)
#             print(f"Updated merge_map_final (partial): \n{merge_map_final}")

#             mask_remove = torch.zeros(B, L, dtype=torch.bool, device=device)
#             mask_remove.scatter_(1, gather_src, True)
            
#             # Also remove the padding that was added in this iteration
#             if pad_pre > 0:
#                 mask_remove[:, :pad_pre] = True
#             if 'pad_post' in locals() and pad_post > 0:
#                 mask_remove[:, -pad_post:] = True

#             keep_mask = ~mask_remove
#             x = x[keep_mask].view(B, -1, C)
#             size = size[keep_mask].view(B, -1, 1)
#             orig_idx = orig_idx[keep_mask].view(B, -1)
#             print(f"Shape after removing src tokens & padding: x={x.shape}")

#         print("\n--- Merge finished ---")
#         return x, merge_map_final

#     # ------------------------------------------------------------------
#     # Reconstruction
#     # ------------------------------------------------------------------
#     @staticmethod
#     def unmerge(merged_x: torch.Tensor, merge_map: torch.Tensor) -> torch.Tensor:
#         """Invert the merge operation given the ``merge_map`` from :py:meth:`merge`."""
#         B, N_original = merge_map.shape
#         _, R, C = merged_x.shape
#         device = merged_x.device

#         # Output tensor – we will scatter values into it.
#         out = torch.zeros(B, N_original, C, device=device, dtype=merged_x.dtype)

#         # Find the root indices for each batch
#         root_mask = merge_map == 0
#         # We need to place merged_x tokens in the order they appear in the
#         # *merged* tensor per batch so that the scatter assignment is correct.
#         for b in range(B):
#             batch_root_idx = torch.nonzero(root_mask[b], as_tuple=False).squeeze(1)
#             out[b, batch_root_idx] = merged_x[b]

#         # Iterate to fill in the rest of the output tensor
#         max_steps = int(torch.abs(merge_map).max().item()) + 1
#         filled = root_mask.clone()
#         arange = torch.arange(N_original, device=device).expand(B, -1)
#         for _ in range(max_steps):
#             unfinished = ~filled
#             if not unfinished.any():
#                 break
#             dst_idx = arange + merge_map
#             # Clamp dst_idx to be within valid range to avoid errors from large offsets
#             # dst_idx.clamp_(0, N_original - 1)
#             can_fill = unfinished & filled.gather(1, dst_idx)
#             if not can_fill.any():
#                 break
#             src_values = out.gather(1, dst_idx.unsqueeze(-1).expand(-1, -1, C))
#             out = torch.where(can_fill.unsqueeze(-1), src_values, out)
#             filled = filled | can_fill

#         return out


class GeneralizedToMeMaskingUpsampler(nn.Module):
    def __init__(self, dim: int, kernel_size: int, causal: bool = False):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.causal = causal

        if self.causal:
            # For causal, offsets are {-1, ..., -(kernel_size - 1)}
            num_embeddings = kernel_size - 1
        else:
            # For non-causal, offsets are {+/-1, ..., +/-(kernel_size - 1)}
            num_embeddings = 2 * (kernel_size - 1)

        # Ensure num_embeddings is at least 1 to avoid errors with nn.Embedding
        if num_embeddings <= 0:
            num_embeddings = 1
            
        self.embedding = nn.Embedding(num_embeddings, dim)

    def forward(self, metric: torch.Tensor, merge_map: torch.Tensor) -> torch.Tensor:
        """
        Upsamples the merged tensor back to the original sequence length.
        Root tokens are filled from the input metric, while merged tokens
        are filled from a learned embedding based on the merge offset.
        This implementation is fully vectorized.
        """
        B, N_original, C = metric.shape[0], merge_map.shape[1], metric.shape[2]
        device = metric.device

        # Create the output tensor.
        out = torch.zeros(B, N_original, C, device=device, dtype=metric.dtype)

        # 1. Vectorized placement of root tokens (where merge_map == 0)
        root_mask = (merge_map == 0)
        num_roots = root_mask.sum()
        if num_roots > 0:
            # The metric tensor (B, R, C) contains the root tokens in order.
            # We reshape it and place it into the output tensor at the root locations.
            out[root_mask] = metric.reshape(-1, C)

        # 2. Vectorized placement of merged tokens from embeddings
        merged_mask = ~root_mask
        if merged_mask.any():
            # Get the non-zero offset values from the merge_map
            offsets = merge_map[merged_mask]

            # Map these offsets to valid embedding indices
            if self.causal:
                # Offsets: [-k+1, ..., -1] -> Indices: [k-2, ..., 0]
                embedding_indices = -offsets - 1
            else:
                # Map negative and positive offsets to different embedding ranges
                # Negative offsets: [-k+1, ..., -1] -> Indices: [0, ..., k-2]
                neg_indices = -offsets - 1
                # Positive offsets: [1, ..., k-1] -> Indices: [k-1, ..., 2k-3]
                pos_indices = offsets - 1 + (self.kernel_size - 1)
                
                embedding_indices = torch.where(offsets < 0, neg_indices, pos_indices)
            
            embedding_indices = embedding_indices.long()
            
            # Retrieve the learned embeddings and place them
            print(embedding_indices.unique())
            learned_vectors = self.embedding(embedding_indices)
            out[merged_mask] = learned_vectors

        return out