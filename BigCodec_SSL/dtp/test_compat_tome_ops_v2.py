import unittest
import itertools
import torch
from .tome_ops import ToMeTopK as ToMeTopK_v1, ToMeK2 as ToMeK2_v1, OurToMe2 as OurToMe2_v1
from .tome_ops_v2 import ToMeTopK as ToMeTopK_v2, ToMeK2 as ToMeK2_v2, OurToMe2 as OurToMe2_v2


class TestCompatibilityV2TopK(unittest.TestCase):
    def test_v1_v2_topk_match(self):
        torch.manual_seed(0)
        seeds = [0, 1]
        batches = [1, 2]
        lengths = [8, 17]
        channels = [8]
        ratios = [0.0, 0.1, 0.25]
        iterations = [1, 3]
        kernel_sizes = [2, 3]

        for seed in seeds:
            g = torch.Generator().manual_seed(seed)
            for B, N, C, r, iters, k in itertools.product(batches, lengths, channels, ratios, iterations, kernel_sizes):
                with self.subTest(seed=seed, B=B, N=N, C=C, r=r, iters=iters, k=k):
                    x = torch.randn(B, N, C, generator=g)

                    v1 = ToMeTopK_v1(r=r, num_iterations=iters, kernel_size=k, filter_chained=True, filter_multiple_src=False)
                    v2 = ToMeTopK_v2(r=r, num_iterations=iters, kernel_size=k, filter_chained=True, filter_multiple_src=False)

                    x1, btree1 = v1.merge(x.clone())
                    x2, btree2, root2 = v2.merge(x.clone())

                    self.assertEqual(x1.shape, x2.shape, "Merged shapes differ (v1 vs v2)")
                    self.assertTrue(torch.allclose(x1, x2, atol=1e-6, rtol=1e-5),
                                    f"Merged tensors differ for config (B={B}, N={N}, C={C}, r={r}, iters={iters}, k={k})")

                    root1 = ToMeTopK_v1.btree_to_root_map(btree1)
                    self.assertTrue(torch.equal(root1, root2), "Direct-to-root maps differ (v1 vs v2)")

                    x1_un = ToMeTopK_v1.unmerge(x1, root1)
                    x2_un = ToMeTopK_v2.unmerge(x2, root2)
                    self.assertTrue(torch.allclose(x1_un, x2_un, atol=1e-6, rtol=1e-5),
                                    "Unmerged tensors differ (v1 vs v2)")


class TestCompatibilityV2K2(unittest.TestCase):
    def test_v1_v2_k2_match(self):
        torch.manual_seed(0)
        seeds = [0, 1]
        batches = [1, 2, 3]
        lengths = [7, 16]
        channels = [8]
        ratios = [0.1, 0.25]
        iterations = [1, 2]

        for seed in seeds:
            g = torch.Generator().manual_seed(seed)
            for B, N, C, r, iters in itertools.product(batches, lengths, channels, ratios, iterations):
                with self.subTest(seed=seed, B=B, N=N, C=C, r=r, iters=iters):
                    x = torch.randn(B, N, C, generator=g)

                    v1 = ToMeK2_v1(r=r, num_iterations=iters)
                    v2 = ToMeK2_v2(r=r, num_iterations=iters)

                    x1, btree1 = v1.merge(x.clone())
                    x2, btree2, root2 = v2.merge(x.clone())

                    self.assertEqual(x1.shape, x2.shape, "Merged shapes differ (v1 vs v2)")
                    self.assertTrue(torch.allclose(x1, x2, atol=1e-6, rtol=1e-5),
                                    f"Merged tensors differ for config (B={B}, N={N}, C={C}, r={r}, iters={iters})")

                    root1 = ToMeK2_v1.btree_to_root_map(btree1)
                    self.assertTrue(torch.equal(root1, root2), "Direct-to-root maps differ (v1 vs v2)")

                    x1_un = ToMeK2_v1.unmerge(x1, root1)
                    x2_un = ToMeK2_v2.unmerge(x2, root2)
                    self.assertTrue(torch.allclose(x1_un, x2_un, atol=1e-6, rtol=1e-5),
                                    "Unmerged tensors differ (v1 vs v2)")


class TestCompatibilityV2OurToMe2(unittest.TestCase):
    def test_v1_v2_our2_match(self):
        torch.manual_seed(0)
        seeds = [0, 1]
        batches = [1, 2]
        lengths = [8, 17]
        channels = [8]
        ratios = [0.1, 0.25]
        iterations = [1, 2]

        for seed in seeds:
            g = torch.Generator().manual_seed(seed)
            for B, N, C, r, iters in itertools.product(batches, lengths, channels, ratios, iterations):
                with self.subTest(seed=seed, B=B, N=N, C=C, r=r, iters=iters):
                    x = torch.randn(B, N, C, generator=g)

                    v1 = OurToMe2_v1(r=r, num_iterations=iters)
                    v2 = OurToMe2_v2(r=r, num_iterations=iters)

                    x1, btree1 = v1.merge(x.clone())
                    x2, btree2, root2 = v2.merge(x.clone())

                    self.assertEqual(x1.shape, x2.shape, "Merged shapes differ (v1 vs v2)")
                    self.assertTrue(torch.allclose(x1, x2, atol=1e-6, rtol=1e-5),
                                    f"Merged tensors differ for config (B={B}, N={N}, C={C}, r={r}, iters={iters})")

                    root1 = OurToMe2_v1.btree_to_root_map(btree1)
                    self.assertTrue(torch.equal(root1, root2), "Direct-to-root maps differ (v1 vs v2)")

                    x1_un = OurToMe2_v1.unmerge(x1, root1)
                    x2_un = OurToMe2_v2.unmerge(x2, root2)
                    self.assertTrue(torch.allclose(x1_un, x2_un, atol=1e-6, rtol=1e-5),
                                    "Unmerged tensors differ (v1 vs v2)")


if __name__ == '__main__':
    unittest.main()


