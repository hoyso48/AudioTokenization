import unittest
import itertools
import torch
from .tome_ops import ToMeTopK, ToMeK2, ToMeK2V2, OurToMe2, OurToMeK, ToMeK2New

class TestCompatibilityK2(unittest.TestCase):
    def test_k2_topk_old_vs_tomek2_match(self):
        torch.manual_seed(0)

        seeds = [0, 1, 2]
        batches = [1, 2, 3]
        lengths = [7, 16]
        channels = [8, 16]
        ratios = [0.1, 0.25, 0.5]
        iterations = [1, 2, 4]
        k = 2

        for seed in seeds:
            g = torch.Generator().manual_seed(seed)
            for B, N, C, r, iters in itertools.product(batches, lengths, channels, ratios, iterations):
                with self.subTest(seed=seed, B=B, N=N, C=C, r=r, iters=iters):
                    x = torch.randn(B, N, C, generator=g)

                    k2_impl = ToMeK2(r=r, num_iterations=iters)
                    k2_v2_impl = ToMeK2V2(r=r, num_iterations=iters)

                    x_k2, btree_k2, avg_k2 = k2_impl.merge(x.clone())
                    x_v2, btree_v2, avg_v2 = k2_v2_impl.merge(x.clone())

                    self.assertEqual(x_k2.shape, x_v2.shape, "Merged shapes differ between K2 and K2V2")
                    self.assertTrue(torch.allclose(x_k2, x_v2, atol=1e-6, rtol=1e-5),
                                    f"Merged tensors differ for config (B={B}, N={N}, C={C}, r={r}, iters={iters})")

                    root_k2 = ToMeK2.btree_to_root_map(btree_k2)
                    root_v2 = ToMeK2V2.btree_to_root_map(btree_v2)
                    self.assertTrue(torch.equal(root_k2, root_v2),
                                    f"Direct-to-root maps differ for k=2 config (B={B}, N={N}, C={C}, r={r}, iters={iters})")

                    x_k2_un = ToMeK2.unmerge(x_k2, root_k2)
                    x_v2_un = ToMeK2V2.unmerge(x_v2, root_v2)

                    self.assertEqual(x_k2_un.shape, x_v2_un.shape, "Unmerged shapes differ between K2 and K2V2")
                    self.assertTrue(torch.allclose(x_k2_un, x_v2_un, atol=1e-6, rtol=1e-5),
                                    f"Unmerged tensors differ for config (B={B}, N={N}, C={C}, r={r}, iters={iters})")

                    self.assertTrue(torch.allclose(avg_k2, avg_v2, atol=1e-6, rtol=1e-5),
                                    f"avg_sim differ for config (B={B}, N={N}, C={C}, r={r}, iters={iters})")


class TestCompatibilityK2VsTopK(unittest.TestCase):
    def test_k2_k2v2_vs_topk_k2_filters_match(self):
        torch.manual_seed(0)

        seeds = [0, 1, 2]
        batches = [1, 2]
        lengths = [7, 16]
        channels = [8, 16]
        ratios = [0.1, 0.25, 0.5]
        iterations = [1, 2, 4]
        k = 2

        for seed in seeds:
            g = torch.Generator().manual_seed(seed)
            for B, N, C, r, iters in itertools.product(batches, lengths, channels, ratios, iterations):
                with self.subTest(seed=seed, B=B, N=N, C=C, r=r, iters=iters):
                    x = torch.randn(B, N, C, generator=g)

                    k2_impl = ToMeK2(r=r, num_iterations=iters)
                    k2_v2_impl = ToMeK2V2(r=r, num_iterations=iters)
                    topk_impl = ToMeTopK(r=r, num_iterations=iters, kernel_size=k, filter_chained=True, filter_multiple_src=False)

                    x_k2, btree_k2, avg_k2 = k2_impl.merge(x.clone())
                    x_v2, btree_v2, avg_v2 = k2_v2_impl.merge(x.clone())
                    x_tk, btree_tk, avg_tk = topk_impl.merge(x.clone())

                    # Shapes
                    self.assertEqual(x_k2.shape, x_v2.shape)
                    self.assertEqual(x_k2.shape, x_tk.shape)

                    # Merged tensors equality
                    self.assertTrue(torch.allclose(x_k2, x_v2, atol=1e-6, rtol=1e-5))
                    self.assertTrue(torch.allclose(x_k2, x_tk, atol=1e-6, rtol=1e-5))

                    # Root maps equality
                    root_k2 = ToMeK2.btree_to_root_map(btree_k2)
                    root_v2 = ToMeK2V2.btree_to_root_map(btree_v2)
                    root_tk = ToMeTopK.btree_to_root_map(btree_tk)
                    self.assertTrue(torch.equal(root_k2, root_v2))
                    self.assertTrue(torch.equal(root_k2, root_tk))

                    # Unmerge equality
                    x_k2_un = ToMeK2.unmerge(x_k2, root_k2)
                    x_v2_un = ToMeK2V2.unmerge(x_v2, root_v2)
                    x_tk_un = ToMeTopK.unmerge(x_tk, root_tk)
                    self.assertEqual(x_k2_un.shape, x_v2_un.shape)
                    self.assertEqual(x_k2_un.shape, x_tk_un.shape)
                    self.assertTrue(torch.allclose(x_k2_un, x_v2_un, atol=1e-6, rtol=1e-5))
                    self.assertTrue(torch.allclose(x_k2_un, x_tk_un, atol=1e-6, rtol=1e-5))

                    # Min-sim compatibility
                    self.assertTrue(torch.allclose(avg_k2, avg_v2, atol=1e-6, rtol=1e-5))
                    self.assertTrue(torch.allclose(avg_k2, avg_tk, atol=1e-6, rtol=1e-5))

class TestCompatibilityOurToMe2VsOurToMeK(unittest.TestCase):
    def test_ourtome2_matches_ourtomek_group2(self):
        torch.manual_seed(0)

        seeds = [0, 1, 2]
        batches = [1, 2]
        lengths = [7, 16]
        channels = [8, 16]
        ratios = [0.1, 0.25, 0.5]
        iterations = [1, 2, 4]

        for seed in seeds:
            g = torch.Generator().manual_seed(seed)
            for B, N, C, r, iters in itertools.product(batches, lengths, channels, ratios, iterations):
                with self.subTest(seed=seed, B=B, N=N, C=C, r=r, iters=iters):
                    x = torch.randn(B, N, C, generator=g)

                    impl2 = OurToMe2(r=r, num_iterations=iters)
                    implK = OurToMeK(r=r, num_iterations=iters, group_size=2)

                    x2, b2, avg2 = impl2.merge(x.clone())
                    xk, bk, avgk = implK.merge(x.clone())

                    self.assertEqual(x2.shape, xk.shape, "Merged shapes differ between OurToMe2 and OurToMeK(group_size=2)")
                    self.assertTrue(torch.allclose(x2, xk, atol=1e-6, rtol=1e-5),
                                    f"Merged tensors differ for config (B={B}, N={N}, C={C}, r={r}, iters={iters})")

                    r2 = OurToMe2.btree_to_root_map(b2)
                    rk = OurToMeK.btree_to_root_map(bk)
                    self.assertTrue(torch.equal(r2, rk),
                                    f"Direct-to-root maps differ for config (B={B}, N={N}, C={C}, r={r}, iters={iters})")

                    xu2 = OurToMe2.unmerge(x2, r2)
                    xuk = OurToMeK.unmerge(xk, rk)
                    self.assertEqual(xu2.shape, xuk.shape, "Unmerged shapes differ between OurToMe2 and OurToMeK(group_size=2)")
                    self.assertTrue(torch.allclose(xu2, xuk, atol=1e-6, rtol=1e-5))


class TestCompatibilityToMeK2New(unittest.TestCase):
    def test_compute_merge_matches_k2(self):
        torch.manual_seed(0)
        seeds = [0, 1]
        batches = [1, 2]
        lengths = [7, 16]
        channels = [8, 16]
        ratios = [0.1, 0.25, 0.5]
        iterations = [1, 2]
        for seed in seeds:
            g = torch.Generator().manual_seed(seed)
            for B, N, C, r, iters in itertools.product(batches, lengths, channels, ratios, iterations):
                with self.subTest(seed=seed, B=B, N=N, C=C, r=r, iters=iters):
                    x = torch.randn(B, N, C, generator=g)
                    k2 = ToMeK2(r=r, num_iterations=iters)
                    k2n = ToMeK2New(r=r, num_iterations=iters)
                    x_k2, b_k2, m_k2 = k2.merge(x.clone())
                    x_n, b_n, m_n = k2n.compute_merge(x.clone())
                    self.assertEqual(x_k2.shape, x_n.shape)
                    self.assertTrue(torch.allclose(x_k2, x_n, atol=1e-6, rtol=1e-5))
                    self.assertTrue(torch.equal(ToMeK2.btree_to_root_map(b_k2), ToMeK2New.btree_to_root_map(b_n)))
                    self.assertTrue(torch.allclose(m_k2, m_n, atol=1e-6, rtol=1e-5))

    def test_map_based_merge_matches_unmerge_roundtrip(self):
        torch.manual_seed(0)
        B, N, C = 2, 10, 8
        r, iters = 0.5, 2
        x = torch.randn(B, N, C)
        k2 = ToMeK2(r=r, num_iterations=iters)
        k2n = ToMeK2New(r=r, num_iterations=iters)
        x_k2, b_k2, _ = k2.merge(x.clone())
        root = ToMeK2.btree_to_root_map(b_k2)
        x_map = k2n.merge(x.clone(), root)
        self.assertTrue(torch.allclose(x_k2, x_map, atol=1e-6, rtol=1e-5))
        # roundtrip
        x_un = ToMeK2New.unmerge(x_map, root)
        self.assertTrue(torch.allclose(x_un, ToMeK2.unmerge(x_k2, root), atol=1e-6, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()


