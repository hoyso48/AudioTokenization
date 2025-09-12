import unittest
import torch
from tome_ops import ToPrK2New, ToPrK2NewChunk

class TestToPrK2NewChunkCompatibility(unittest.TestCase):
    def test_chunk_equals_full_when_chunk_size_is_N(self):
        torch.manual_seed(0)
        seeds = [0, 1]
        batches = [1, 2]
        lengths = [7, 16, 31]
        channels = [8, 16]
        ratios = [0.1, 0.25, 0.5]
        iterations = [2, 4, 8]

        for seed in seeds:
            g = torch.Generator().manual_seed(seed)
            for B in batches:
                for N in lengths:
                    for C in channels:
                        for r in ratios:
                            for iters in iterations:
                                with self.subTest(seed=seed, B=B, N=N, C=C, r=r, iters=iters):
                                    x = torch.randn(B, N, C, generator=g)
                                    full = ToPrK2New(r=r, num_iterations=iters)
                                    chunk = ToPrK2NewChunk(r=r, num_iterations=iters, chunk_size=N)
                                    with torch.no_grad():
                                        x_full, b_full, m_full = full.compute_merge(x.clone())
                                        x_chunk, b_chunk, m_chunk = chunk.compute_merge(x.clone())
                                    # Shapes and tensors must match exactly
                                    self.assertEqual(x_full.shape, x_chunk.shape)
                                    self.assertTrue(torch.allclose(x_full, x_chunk, atol=1e-6, rtol=1e-5))
                                    # Root maps equality
                                    root_full = full.btree_to_root_map(b_full)
                                    root_chunk = chunk.btree_to_root_map(b_chunk)
                                    self.assertTrue(torch.equal(root_full, root_chunk))
                                    # Unmerge equality
                                    x_full_un = full.unmerge(x_full, root_full)
                                    x_chunk_un = chunk.unmerge(x_chunk, root_chunk)
                                    self.assertEqual(x_full_un.shape, x_chunk_un.shape)
                                    self.assertTrue(torch.allclose(x_full_un, x_chunk_un, atol=1e-6, rtol=1e-5))
                                    # avg_sim equality
                                    self.assertTrue(torch.allclose(m_full, m_chunk, atol=1e-6, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()


