import math
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx


def _normalize(x: jax.Array, axis: int = -1, eps: float = 1e-8) -> jax.Array:
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    norm = jnp.clip(norm, a_min=eps)
    return x / norm


def _take_along_axis_3d(x: jax.Array, idx: jax.Array) -> jax.Array:
    # x: (B, N, C), idx: (B, K)
    B, _, C = x.shape
    idx_exp = jnp.expand_dims(idx, axis=-1)
    idx_exp = jnp.broadcast_to(idx_exp, (B, idx.shape[1], C))
    return jnp.take_along_axis(x, idx_exp, axis=1)


def _compact_by_keep_mask_2d(arr: jax.Array, keep_mask: jax.Array, keep_count: int) -> jax.Array:
    # arr: (B, N), keep_mask: (B, N) bool -> returns (B, keep_count)
    B, N = keep_mask.shape
    base = jnp.broadcast_to(jnp.arange(N)[None, :], (B, N))
    # Move kept (0) before removed (1) and keep stable order with base index
    key = jnp.where(keep_mask, 0, 1) * N + base
    order = jnp.argsort(key, axis=1)
    gather_idx = order[:, :keep_count]
    return jnp.take_along_axis(arr, gather_idx, axis=1)


def _compact_by_keep_mask_3d(arr: jax.Array, keep_mask: jax.Array, keep_count: int) -> jax.Array:
    # arr: (B, N, C), keep_mask: (B, N) bool -> returns (B, keep_count, C)
    B, N = keep_mask.shape
    base = jnp.broadcast_to(jnp.arange(N)[None, :], (B, N))
    key = jnp.where(keep_mask, 0, 1) * N + base
    order = jnp.argsort(key, axis=1)
    gather_idx = order[:, :keep_count]
    return _take_along_axis_3d(arr, gather_idx)


class ToMeK2New(nnx.Module):
    """
    JAX/Flax NNX implementation of the PyTorch ToMeK2New (k=2) token merging variant.

    API parity with torch version:
    - compute_merge(x): returns (merged_x, btree_map, avg_sim)
    - merge(x, direct_to_root_map): returns merged root tokens
    - btree_to_root_map(merge_btree): resolves chained merges to direct-to-root map
    - unmerge(y, direct_to_root_map): reconstructs original sequence
    """

    def __init__(self, r: float, num_iterations: int):
        if r < 0.0 or r > 1.0:
            raise ValueError("r must be in [0, 1]")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        self.r = r
        self.num_iterations = num_iterations

    def compute_merge(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Greedy adjacent merges in multiple iterations (non-overlapping within each iteration).

        Args:
            x: (B, N, C) float array

        Returns:
            merged_x: (B, R, C)
            btree_map: (B, N) int32, offsets (-1 for merged src tokens, 0 otherwise)
            avg_sim: (B,) float, per-batch min similarity across iterations (like torch impl)
        """
        B, N, C = x.shape
        dtype = x.dtype

        total_to_merge = int(self.r * N)
        btree_map = jnp.zeros((B, N), dtype=jnp.int32)
        if total_to_merge == 0 or N < 2:
            return x, btree_map, jnp.zeros((B,), dtype=dtype)

        # Compute merges per iteration as in torch version
        steps = self.num_iterations + 1
        # Equivalent of np.diff(np.linspace(0, total_to_merge, steps, dtype=int))
        boundaries = jnp.linspace(0, float(total_to_merge), steps)
        boundaries = jnp.floor(boundaries).astype(jnp.int32)
        merges_per_iter = jnp.diff(boundaries)  # (num_iterations,)

        size = jnp.ones((B, N), dtype=dtype)
        orig_idx = jnp.broadcast_to(jnp.arange(N, dtype=jnp.int32)[None, :], (B, N))

        min_sim = jnp.full((B,), jnp.inf, dtype=dtype)
        total_selected = 0

        # We'll keep x, size, orig_idx as Python variables updated iteratively (no jit required)
        x_curr: jax.Array = x
        size_curr: jax.Array = size
        orig_idx_curr: jax.Array = orig_idx

        for k in list(map(int, list(merges_per_iter))):
            if k == 0:
                continue
            Bc, Nc, Cc = x_curr.shape
            if Nc < 2:
                break

            # Compute adjacency similarities
            x_norm = _normalize(x_curr, axis=-1)
            sim = jnp.sum(x_norm[:, :-1, :] * x_norm[:, 1:, :], axis=-1)  # (B, Nc-1)

            # Batched greedy selection up to k non-overlapping adjacent pairs
            used = jnp.zeros((Bc, Nc), dtype=jnp.bool_)
            sel_dst = jnp.full((Bc, k), -1, dtype=jnp.int32)
            sel_src = jnp.full((Bc, k), -1, dtype=jnp.int32)
            vals_per_slot = jnp.full((Bc, k), -jnp.inf, dtype=dtype)
            b_arange = jnp.arange(Bc)

            for t in range(k):
                pair_valid = (~used[:, :-1]) & (~used[:, 1:])  # (B, Nc-1)
                masked = jnp.where(pair_valid, sim, -jnp.inf)
                top_val = jnp.max(masked, axis=1)  # (B,)
                top_idx = jnp.argmax(masked, axis=1)  # (B,)
                can_pick = jnp.isfinite(top_val)

                dst_col = jnp.where(can_pick, top_idx, -1).astype(jnp.int32)
                src_col = jnp.where(can_pick, dst_col + 1, -1).astype(jnp.int32)

                sel_dst = sel_dst.at[:, t].set(dst_col)
                sel_src = sel_src.at[:, t].set(src_col)
                vals_per_slot = vals_per_slot.at[:, t].set(jnp.where(can_pick, top_val, -jnp.inf))

                # Update used mask only for valid picks via scatter on auxiliary updates
                upd_dst = jnp.zeros_like(used)
                upd_src = jnp.zeros_like(used)
                dst_pos = jnp.where(can_pick, dst_col, 0)
                src_pos = jnp.where(can_pick, src_col, 0)
                upd_dst = upd_dst.at[b_arange, dst_pos].set(can_pick)
                upd_src = upd_src.at[b_arange, src_pos].set(can_pick)
                used = used | upd_dst | upd_src

            valid_merge = sel_src != -1
            if not bool(jnp.any(valid_merge)):
                break
            per_batch_counts = jnp.sum(valid_merge, axis=1)  # (B,)
            m = int(jnp.min(per_batch_counts))
            if m == 0:
                break

            step_min = jnp.min(vals_per_slot[:, :m], axis=1)
            min_sim = jnp.minimum(min_sim, step_min)
            total_selected += m

            # Apply merges: vectorized across batches for the first m slots
            sel_dst_m = sel_dst[:, :m].astype(jnp.int32)
            sel_src_m = sel_src[:, :m].astype(jnp.int32)

            # Update btree_map at original indices of selected src
            src_orig_idx_i = jnp.take_along_axis(orig_idx_curr, sel_src_m.astype(jnp.int32), axis=1)
            b_idx = jnp.arange(Bc)[:, None]
            btree_map = btree_map.at[b_idx, src_orig_idx_i].set(-1)

            # sizes and features updates
            dst_size_i = jnp.take_along_axis(size_curr, sel_dst_m, axis=1)
            src_size_i = jnp.take_along_axis(size_curr, sel_src_m, axis=1)
            dst_feat_i = _take_along_axis_3d(x_curr, sel_dst_m)
            src_feat_i = _take_along_axis_3d(x_curr, sel_src_m)

            denom_add = jnp.zeros_like(size_curr)
            denom_add = denom_add.at[b_idx, sel_dst_m].add(src_size_i)
            denom = size_curr + denom_add

            numer = x_curr * jnp.expand_dims(size_curr, axis=-1)
            contrib = src_feat_i * jnp.expand_dims(src_size_i, axis=-1)
            add = jnp.zeros_like(x_curr)
            add = add.at[b_idx, sel_dst_m, :].add(contrib)
            numer = numer + add

            x_next = numer / jnp.expand_dims(denom, axis=-1)
            size_next = denom

            # Remove the selected src positions
            keep_mask = jnp.ones((Bc, Nc), dtype=jnp.bool_)
            keep_mask = keep_mask.at[b_idx, sel_src_m].set(False)
            new_N = Nc - m
            x_curr = _compact_by_keep_mask_3d(x_next, keep_mask, new_N)
            size_curr = _compact_by_keep_mask_2d(size_next, keep_mask, new_N)
            orig_idx_curr = _compact_by_keep_mask_2d(orig_idx_curr, keep_mask, new_N)

        if total_selected == 0:
            avg_sim = jnp.zeros((B,), dtype=dtype)
        else:
            avg_sim = min_sim
        return x_curr, btree_map, avg_sim

    def merge(self, x: jax.Array, direct_to_root_map: jax.Array) -> jax.Array:
        """
        Differentiable-like single-step merge given a direct-to-root map.
        Returns root tokens, in order.
        """
        B, N, C = x.shape
        root_indices = jnp.broadcast_to(jnp.arange(N, dtype=jnp.int32)[None, :], (B, N)) + direct_to_root_map
        # Accumulate features per root via scatter-add
        def merge_batch(x_b: jax.Array, root_idx_b: jax.Array) -> jax.Array:
            accum = jnp.zeros_like(x_b)
            accum = accum.at[root_idx_b, :].add(x_b)
            counts = jnp.zeros((x_b.shape[0],), dtype=x_b.dtype).at[root_idx_b].add(1.0)
            counts = jnp.clip(counts, a_min=1e-8)
            return accum / counts[:, None]

        merged_x = jax.vmap(merge_batch)(x, root_indices)
        root_mask = direct_to_root_map == 0
        root_tokens = merged_x[root_mask].reshape(B, -1, C)
        return root_tokens

    def btree_to_root_map(self, merge_btree: jax.Array) -> jax.Array:
        """
        Resolve one-step btree offsets to direct-to-root map by repeated pointer jumping.
        """
        B, N = merge_btree.shape
        direct = merge_btree.astype(jnp.int32)
        arange_N = jnp.arange(N, dtype=jnp.int32)[None, :]
        b_idx = jnp.arange(B, dtype=jnp.int32)[:, None]

        for _ in range(self.num_iterations):
            current_dest = arange_N + direct
            next_hop = merge_btree[b_idx, current_dest]
            needs = next_hop != 0
            if not bool(jnp.any(needs)):
                break
            direct = direct + next_hop
        return direct

    def unmerge(self, y: jax.Array, direct_to_root_map: jax.Array) -> jax.Array:
        """
        Reconstruct original tokens from merged roots and mapping.
        """
        B, N = direct_to_root_map.shape
        if y.shape[1] == N:
            return y
        C = y.shape[2]
        unmerged = jnp.zeros((B, N, C), dtype=y.dtype)
        root_mask = direct_to_root_map == 0
        # Emulate PyTorch's boolean assignment: unmerged[root_mask] = y.flatten(0, 1)
        # by flattening along (B, N) in row-major order.
        flat = unmerged.reshape(B * N, C)
        mask_flat = root_mask.reshape(B * N)
        pos = jnp.nonzero(mask_flat, size=int(y.shape[0] * y.shape[1]), fill_value=0)[0]
        flat = flat.at[pos].set(y.reshape(-1, C))
        unmerged = flat.reshape(B, N, C)

        source_indices = jnp.arange(N, dtype=jnp.int32)[None, :] + direct_to_root_map
        source_indices_expanded = jnp.broadcast_to(source_indices[:, :, None], (B, N, C))
        final_x = jnp.take_along_axis(unmerged, source_indices_expanded, axis=1)
        return final_x


if __name__ == "__main__":
    # Self-test to compare JAX vs PyTorch implementations
    import sys
    import numpy as np
    import importlib.util
    import time

    def load_module_from_path(name: str, path: str):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    torch_module = load_module_from_path(
        "torch_tome_ops",
        "/home/hoyeol/AudioTokenization/CP/dtp/tome_ops.py",
    )

    TorchToMeK2New = torch_module.ToMeK2New

    def run_case(B: int, N: int, C: int, r: float, iters: int):
        import torch

        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((B, N, C), dtype=np.float32)
        x_j = jnp.array(x_np)
        x_t = torch.tensor(x_np, dtype=torch.float32)

        j_model = ToMeK2New(r=r, num_iterations=iters)
        t_model = TorchToMeK2New(r=r, num_iterations=iters)

        j_merged, j_btree, j_avg = j_model.compute_merge(x_j)
        t_merged, t_btree, t_avg = t_model.compute_merge(x_t)

        j_root_map = j_model.btree_to_root_map(j_btree)
        t_root_map = t_model.btree_to_root_map(t_btree)

        # Compare direct_to_root_map exactly
        j_rm = np.array(j_root_map, dtype=np.int32)
        t_rm = t_root_map.detach().cpu().numpy().astype(np.int32)
        assert np.array_equal(j_rm, t_rm), "direct_to_root_map mismatch"

        # Compare merged root tokens
        j_root = j_model.merge(x_j, j_root_map)
        t_root = t_model.merge(x_t, t_root_map)
        j_root_np = np.array(j_root)
        t_root_np = t_root.detach().cpu().numpy()
        np.testing.assert_allclose(j_root_np, t_root_np, atol=1e-6)

        # Compare the merged sequences from compute_merge directly
        np.testing.assert_allclose(np.array(j_merged), t_merged.detach().cpu().numpy(), atol=1e-6)

        # Compare unmerge to originals
        j_rec = j_model.unmerge(j_root, j_root_map)
        t_rec = t_model.unmerge(t_root, t_root_map)
        np.testing.assert_allclose(np.array(j_rec), t_rec.detach().cpu().numpy(), atol=1e-6)

    cases = [
        (2, 16, 8, 0.0, 4),
        (2, 17, 7, 0.3, 4),
        (1, 10, 5, 0.5, 3),
        (3, 12, 9, 0.25, 5),
    ]

    all_ok = True
    for case in cases:
        try:
            run_case(*case)
            print(f"OK: B={case[0]} N={case[1]} C={case[2]} r={case[3]} iters={case[4]}")
        except Exception as e:
            all_ok = False
            print(f"FAIL: {case}: {e}")
            raise
    if all_ok:
        print("All JAX vs PyTorch ToMeK2New equivalence tests passed.")
        # ------------------------
        # Simple runtime benchmark
        # ------------------------
        def _ms(x: float) -> float:
            return x * 1000.0

        def bench_case(B: int, N: int, C: int, r: float, iters: int, runs: int = 30):
            import torch
            rng = np.random.default_rng(0)
            x_np = rng.standard_normal((B, N, C), dtype=np.float32)
            x_j = jnp.array(x_np)
            x_t = torch.tensor(x_np, dtype=torch.float32)

            j_model = ToMeK2New(r=r, num_iterations=iters)
            t_model = TorchToMeK2New(r=r, num_iterations=iters)

            # 1) JAX compute_merge (eager)
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                j_merged, j_btree, j_avg = j_model.compute_merge(x_j)
                _ = jax.block_until_ready(j_merged)
                _ = jax.block_until_ready(j_btree)
                _ = jax.block_until_ready(j_avg)
                times.append(time.perf_counter() - t0)
            j_compute_merge_ms = _ms(np.mean(times))

            # 2) Torch compute_merge
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                tm, tb, ta = t_model.compute_merge(x_t)
                times.append(time.perf_counter() - t0)
            t_compute_merge_ms = _ms(np.mean(times))

            # Root maps
            j_root_map = j_model.btree_to_root_map(j_btree)
            t_root_map = t_model.btree_to_root_map(tb)

            # 3) JAX merge/unmerge eager
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                y = j_model.merge(x_j, j_root_map)
                _ = jax.block_until_ready(y)
                times.append(time.perf_counter() - t0)
            j_merge_ms = _ms(np.mean(times))

            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                z = j_model.unmerge(y, j_root_map)
                _ = jax.block_until_ready(z)
                times.append(time.perf_counter() - t0)
            j_unmerge_ms = _ms(np.mean(times))

            # 4) Torch merge/unmerge
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                y_t = t_model.merge(x_t, t_root_map)
                times.append(time.perf_counter() - t0)
            t_merge_ms = _ms(np.mean(times))

            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                z_t = t_model.unmerge(y_t, t_root_map)
                times.append(time.perf_counter() - t0)
            t_unmerge_ms = _ms(np.mean(times))

            # 5) JAX jitted merge/unmerge (boolean-indexing-free, fixed packed length)
            R = int(np.array((j_root_map == 0)).sum(axis=1).min())

            def jitted_merge_impl(x_in: jax.Array, rm: jax.Array) -> jax.Array:
                B_loc, N_loc, C_loc = x_in.shape
                root_indices = jnp.arange(N_loc, dtype=jnp.int32)[None, :] + rm.astype(jnp.int32)

                def merge_batch(x_b: jax.Array, root_idx_b: jax.Array) -> jax.Array:
                    accum = jnp.zeros_like(x_b)
                    accum = accum.at[root_idx_b, :].add(x_b)
                    counts = jnp.zeros((N_loc,), dtype=x_b.dtype).at[root_idx_b].add(1.0)
                    counts = jnp.clip(counts, a_min=1e-8)
                    return accum / counts[:, None]

                merged_x_loc = jax.vmap(merge_batch)(x_in, root_indices)
                keep_mask = (rm == 0)
                base = jnp.broadcast_to(jnp.arange(N_loc)[None, :], (B_loc, N_loc))
                key = jnp.where(keep_mask, 0, 1) * N_loc + base
                order = jnp.argsort(key, axis=1)
                gather_idx = order[:, :R]
                return _take_along_axis_3d(merged_x_loc, gather_idx)

            def jitted_unmerge_impl(y_in: jax.Array, rm: jax.Array) -> jax.Array:
                B_loc, N_loc = rm.shape
                C_loc = y_in.shape[2]
                keep_mask = (rm == 0)
                base = jnp.broadcast_to(jnp.arange(N_loc)[None, :], (B_loc, N_loc))
                key = jnp.where(keep_mask, 0, 1) * N_loc + base
                order = jnp.argsort(key, axis=1)
                root_pos = order[:, :R]
                unmerged = jnp.zeros((B_loc, N_loc, C_loc), dtype=y_in.dtype)
                b_ids = jnp.arange(B_loc)[:, None]
                unmerged = unmerged.at[b_ids, root_pos, :].set(y_in)
                source_indices = jnp.arange(N_loc, dtype=jnp.int32)[None, :] + rm.astype(jnp.int32)
                src_exp = jnp.broadcast_to(source_indices[:, :, None], (B_loc, N_loc, C_loc))
                return jnp.take_along_axis(unmerged, src_exp, axis=1)

            j_merge_jit = jax.jit(jitted_merge_impl)
            j_unmerge_jit = jax.jit(jitted_unmerge_impl)

            # Compile + steady-state for merge
            t0 = time.perf_counter()
            y0 = j_merge_jit(x_j, j_root_map)
            _ = jax.block_until_ready(y0)
            merge_compile_ms = _ms(time.perf_counter() - t0)

            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                y1 = j_merge_jit(x_j, j_root_map)
                _ = jax.block_until_ready(y1)
                times.append(time.perf_counter() - t0)
            j_merge_jit_ms = _ms(np.mean(times))

            # Compile + steady-state for unmerge
            t0 = time.perf_counter()
            z0 = j_unmerge_jit(y0, j_root_map)
            _ = jax.block_until_ready(z0)
            unmerge_compile_ms = _ms(time.perf_counter() - t0)

            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                z1 = j_unmerge_jit(y0, j_root_map)
                _ = jax.block_until_ready(z1)
                times.append(time.perf_counter() - t0)
            j_unmerge_jit_ms = _ms(np.mean(times))

            print(f"Benchmark (B={B}, N={N}, C={C}, r={r}, iters={iters}, runs={runs})")
            print(f"  JAX  compute_merge (eager): {j_compute_merge_ms:.3f} ms")
            print(f"  Torch compute_merge       : {t_compute_merge_ms:.3f} ms")
            print(f"  JAX  merge (eager)        : {j_merge_ms:.3f} ms")
            print(f"  JAX  unmerge (eager)      : {j_unmerge_ms:.3f} ms")
            print(f"  Torch merge               : {t_merge_ms:.3f} ms")
            print(f"  Torch unmerge             : {t_unmerge_ms:.3f} ms")
            print(f"  JAX  merge jit compile    : {merge_compile_ms:.3f} ms")
            print(f"  JAX  merge jit steady     : {j_merge_jit_ms:.3f} ms")
            print(f"  JAX  unmerge jit compile  : {unmerge_compile_ms:.3f} ms")
            print(f"  JAX  unmerge jit steady   : {j_unmerge_jit_ms:.3f} ms")

        bench_case(2, 128, 64, 0.5, 4, runs=30)


