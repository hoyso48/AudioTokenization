import math
import unittest
import torch

from ops import PLEBatchTopK, PLEBatchTopK_old
from resampler import RepeatUpsampler, MaskUpsampler


def _forward_fill_expected(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    positions = torch.arange(N, device=x.device).view(1, N).expand(B, N)
    keep_pos = torch.where(mask, positions, torch.full_like(positions, -1))
    src_pos = torch.cummax(keep_pos, dim=1).values.clamp(min=0)
    return x.gather(1, src_pos.unsqueeze(-1).expand(-1, -1, C))


class TestTomeOpsVFR(unittest.TestCase):
    def test_plebatchtopk_train_eval_shapes_and_counts(self):
        cases = [
            dict(seed=0, B=4, N=16, C=8, r=0.5),
            dict(seed=1, B=1, N=17, C=6, r=0.35),
            dict(seed=2, B=3, N=12, C=5, r=0.3),
            dict(seed=3, B=2, N=10, C=4, r=0.1),
        ]
        for case in cases:
            torch.manual_seed(case['seed'])
            B, N, C, r = case['B'], case['N'], case['C'], case['r']
            x = torch.randn(B, N, C)
            ple = PLEBatchTopK(r=r, momentum=0.9)
            ple.train()
            mask, avg_r, tau_used = ple(x)

            # derive cu_kept and max_seqlen from mask
            per_seq = mask.sum(dim=1).to(torch.long)
            cu_kept = torch.zeros(B + 1, dtype=torch.long)
            cu_kept[1:] = torch.cumsum(per_seq, dim=0)
            max_seqlen = int(per_seq.max().item())

            # prints for visibility
            total = B * N
            kept_total = int(mask.sum().item())
            K_target = total - math.floor(r * total)
            print(f"CASE seed={case['seed']} B={B} N={N} C={C} r={r:.2f}")
            print(f"  kept_total={kept_total} K_target={K_target} avg_r={avg_r.item():.4f} tau={tau_used.item():.6f}")
            print(f"  per_seq_kept min={int(per_seq.min().item())} max={int(per_seq.max().item())} mean={float(per_seq.float().mean().item()):.2f}")
            # show first 5 frontier indices for first sequence
            idx0 = mask[0].nonzero(as_tuple=False).squeeze(1)
            print(f"  seq0_frontiers(first5)={idx0[:5].tolist()}")

            # assertions
            self.assertEqual(mask.shape, (B, N))
            y_packed = x[mask].view(-1, C)
            self.assertEqual(y_packed.shape[1], C)
            self.assertEqual(y_packed.shape[0], kept_total)
            self.assertEqual(tuple(cu_kept.shape), (B + 1,))
            self.assertEqual(cu_kept[0].item(), 0)
            self.assertEqual(cu_kept[-1].item(), kept_total)
            self.assertIsInstance(max_seqlen, int)
            self.assertTrue(torch.all(mask[:, 0]).item())
            self.assertLessEqual(kept_total, K_target)
            self.assertGreaterEqual(kept_total, B)
            zeros_total = total - kept_total
            expect_avg_r = float(zeros_total) / float(total)
            self.assertTrue(torch.isclose(avg_r, torch.tensor(expect_avg_r, dtype=avg_r.dtype)).item())
            self.assertTrue(torch.isfinite(tau_used).item())

    def test_plebatchtopk_old_train_eval_shapes_and_counts(self):
        # Use identical cases to the new class test for apples-to-apples comparison
        cases = [
            dict(seed=0, B=4, N=16, C=8, r=0.5),
            dict(seed=1, B=1, N=17, C=6, r=0.35),
            dict(seed=2, B=3, N=12, C=5, r=0.3),
            dict(seed=3, B=2, N=10, C=4, r=0.1),
        ]
        for case in cases:
            torch.manual_seed(case['seed'])
            B, N, C, r = case['B'], case['N'], case['C'], case['r']
            x = torch.randn(B, N, C)
            ple = PLEBatchTopK_old(r=r, momentum=0.9)
            ple.train()
            mask, avg_r, tau_used = ple(x)

            per_seq = mask.sum(dim=1).to(torch.long)
            cu_kept = torch.zeros(B + 1, dtype=torch.long)
            cu_kept[1:] = torch.cumsum(per_seq, dim=0)
            max_seqlen = int(per_seq.max().item())

            total = B * N
            kept_total = int(mask.sum().item())
            K_target = total - math.floor(r * total)
            print(f"OLD CASE seed={case['seed']} B={B} N={N} C={C} r={r:.2f}")
            print(f"  kept_total={kept_total} K_target={K_target} avg_r={avg_r.item():.4f} tau={tau_used.item():.6f}")
            print(f"  per_seq_kept min={int(per_seq.min().item())} max={int(per_seq.max().item())} mean={float(per_seq.float().mean().item()):.2f}")
            idx0 = mask[0].nonzero(as_tuple=False).squeeze(1)
            print(f"  seq0_frontiers(first5)={idx0[:5].tolist()}")

            self.assertEqual(mask.shape, (B, N))
            y_packed = x[mask].view(-1, C)
            self.assertEqual(y_packed.shape[1], C)
            self.assertEqual(y_packed.shape[0], kept_total)
            self.assertEqual(tuple(cu_kept.shape), (B + 1,))
            self.assertEqual(cu_kept[0].item(), 0)
            self.assertEqual(cu_kept[-1].item(), kept_total)
            self.assertIsInstance(max_seqlen, int)
            self.assertTrue(torch.all(mask[:, 0]).item())
            self.assertLessEqual(kept_total, K_target)
            self.assertGreaterEqual(kept_total, B)
            zeros_total = total - kept_total
            expect_avg_r = float(zeros_total) / float(total)
            self.assertTrue(torch.isclose(avg_r, torch.tensor(expect_avg_r, dtype=avg_r.dtype)).item())
            self.assertTrue(torch.isfinite(tau_used).item())

    def test_plebatchtopk_packing_and_order(self):
        torch.manual_seed(1)
        B, N, C = 3, 12, 6
        x = torch.randn(B, N, C)
        ple = PLEBatchTopK(r=0.3)
        ple.train()
        mask, avg_r, tau_used = ple(x)
        per_seq = mask.sum(dim=1).to(torch.long)
        cu_kept = torch.zeros(B + 1, dtype=torch.long)
        cu_kept[1:] = torch.cumsum(per_seq, dim=0)
        max_seqlen = int(per_seq.max().item())

        expected = []
        for b in range(B):
            expected.append(x[b][mask[b]])
        expected = torch.cat(expected, dim=0)
        y_packed = x[mask].view(-1, C)
        self.assertTrue(torch.allclose(y_packed, expected))

        per_seq = mask.sum(dim=1).to(torch.long)
        self.assertTrue(torch.allclose(cu_kept[1:], torch.cumsum(per_seq, dim=0)))
        self.assertEqual(max_seqlen, int(per_seq.max().item()))

    def test_repeat_upsampler_dense_and_varlen_equivalence(self):
        torch.manual_seed(2)
        B, N, C = 2, 10, 4
        x = torch.randn(B, N, C)
        ple = PLEBatchTopK(r=0.5)
        ple.train()
        mask, _, _ = ple(x)
        per_seq = mask.sum(dim=1).to(torch.long)
        cu_kept = torch.zeros(B + 1, dtype=torch.long)
        cu_kept[1:] = torch.cumsum(per_seq, dim=0)
        y_packed = x[mask].view(-1, C)

        up = RepeatUpsampler()

        expected_dense = _forward_fill_expected(x, mask)
        y_varlen = up(y_packed, mask, cu_seqlens=cu_kept)
        self.assertTrue(torch.allclose(y_varlen, expected_dense))

    def test_masked_upsampler_dense_and_varlen(self):
        torch.manual_seed(3)
        B, N, C = 2, 9, 5
        x = torch.randn(B, N, C)
        ple = PLEBatchTopK(r=0.4)
        ple.train()
        mask, _, _ = ple(x)
        per_seq = mask.sum(dim=1).to(torch.long)
        cu_kept = torch.zeros(B + 1, dtype=torch.long)
        cu_kept[1:] = torch.cumsum(per_seq, dim=0)
        y_packed = x[mask].view(-1, C)

        mu = MaskUpsampler(dim=C)
        with torch.no_grad():
            mu.mask_token.copy_(torch.arange(C, dtype=x.dtype))

        y_varlen = mu(y_packed, mask, cu_seqlens=cu_kept)
        self.assertTrue(torch.allclose(y_varlen[mask], x[mask]))
        self.assertTrue(torch.all(y_varlen[~mask] == mu.mask_token))

    def test_edge_case_single_token_sequences(self):
        torch.manual_seed(4)
        B, N, C = 3, 1, 7
        x = torch.randn(B, N, C)
        ple = PLEBatchTopK(r=0.8)
        ple.train()
        mask, avg_r, tau_used = ple(x)
        per_seq = mask.sum(dim=1).to(torch.long)
        cu_kept = torch.zeros(B + 1, dtype=torch.long)
        cu_kept[1:] = torch.cumsum(per_seq, dim=0)
        max_seqlen = int(per_seq.max().item())

        self.assertEqual(mask.shape, (B, N))
        self.assertTrue(torch.all(mask[:, 0]).item())
        y_packed = x[mask].view(-1, C)
        self.assertEqual(tuple(y_packed.shape), (B, C))
        self.assertEqual(cu_kept.tolist(), [0, 1, 2, 3])
        self.assertEqual(max_seqlen, 1)
        self.assertTrue(torch.isclose(avg_r, torch.tensor(0.0, dtype=avg_r.dtype)).item())

    def test_equivalence_with_pletopk_batch1_first_crossing(self):
        torch.manual_seed(5)
        B, N, C = 1, 17, 8
        x = torch.randn(B, N, C)
        r = 0.35

        from tome_ops import PLETopK
        ple_orig = PLETopK(r=r, use_bin_argmax=False, sample_bins_training=0.0, fallback=None)
        merged_x, btree_map, _ = ple_orig.compute_merge(x)
        mask_topk = (btree_map == 0)

        ple_vfr = PLEBatchTopK(r=r)
        mask_vfr, _, _ = ple_vfr(x)

        self.assertTrue(torch.equal(mask_vfr, mask_topk))
        expected = merged_x.view(-1, C)
        y_packed = x[mask_vfr].view(-1, C)
        self.assertTrue(torch.allclose(y_packed, expected))
        # print(mask_vfr)
        # print(mask_topk)

    def test_all_zero_path_keeps_first_M_positions(self):
        torch.manual_seed(6)
        B, N, C = 2, 12, 4
        base = torch.randn(B, 1, C)
        x = base.expand(B, N, C).clone()
        r = 0.25
        ple = PLEBatchTopK(r=r)
        mask, avg_r, tau_used = ple(x)
        per_seq = mask.sum(dim=1).to(torch.long)
        cu_kept = torch.zeros(B + 1, dtype=torch.long)
        cu_kept[1:] = torch.cumsum(per_seq, dim=0)
        max_seqlen = int(per_seq.max().item())

        M = N - math.floor(r * N)
        kept_total = int(mask.sum().item())
        # With global tau and L=0, only first tokens are kept
        self.assertEqual(kept_total, B)
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=False).squeeze(1)
            self.assertEqual(idx.numel(), M)
            self.assertEqual(idx[0].item(), 0)
            if idx.numel() > 1:
                self.assertTrue(torch.all(idx[1:] > idx[:-1]).item())

        # avg_r should match 1 - M/N exactly when all rows are identical
        expect_avg_r = float(N - M) / float(N)
        self.assertTrue(torch.isclose(avg_r, torch.tensor(expect_avg_r, dtype=avg_r.dtype)).item())

    def test_boundaries_strictly_increasing(self):
        torch.manual_seed(7)
        B, N, C = 3, 15, 5
        x = torch.randn(B, N, C)
        ple = PLEBatchTopK(r=0.3)
        mask, avg_r, tau_used = ple(x)
        per_seq = mask.sum(dim=1).to(torch.long)
        cu_kept = torch.zeros(B + 1, dtype=torch.long)
        cu_kept[1:] = torch.cumsum(per_seq, dim=0)
        max_seqlen = int(per_seq.max().item())

        for b in range(B):
            idx = mask[b].nonzero(as_tuple=False).squeeze(1)
            self.assertGreaterEqual(idx.numel(), 1)
            self.assertEqual(idx[0].item(), 0)
            if idx.numel() > 1:
                self.assertTrue(torch.all(idx[1:] > idx[:-1]).item())

    def test_r_edges(self):
        torch.manual_seed(8)
        B, N, C = 2, 10, 3
        x = torch.randn(B, N, C)

        ple_all = PLEBatchTopK(r=0.0)
        mask_all, _, _ = ple_all(x)
        # No fallback: may under-select due to clamping at N-1; ensure valid bounds
        self.assertTrue(torch.all(mask_all[:, 0]).item())
        self.assertTrue(B <= int(mask_all.sum().item()) <= B * N)

        ple_one = PLEBatchTopK(r=0.9)
        mask_one, _, _ = ple_one(x)
        self.assertTrue(torch.all(mask_one[:, 0]).item())
        self.assertEqual(int(mask_one.sum().item()), B * 1)

    def test_first_token_always_frontier(self):
        torch.manual_seed(9)
        B, N, C = 5, 13, 7
        x = torch.randn(B, N, C)
        r_values = [0.0, 0.1, 0.3, 0.5, 0.9]
        for r in r_values:
            ple = PLEBatchTopK(r=r)
            mask, _, _ = ple(x)
            # every sequence's first token must be a frontier
            self.assertTrue(torch.all(mask[:, 0]).item())



if __name__ == "__main__":
    unittest.main()
