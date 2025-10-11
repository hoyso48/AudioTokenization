import unittest
import torch
from tome_ops import GeneralizedToMe, OurToMeK, OurToMe2, ToPrK2New, ToPrPLETopK, ToPrCPRRTopK, ToPrK2NewChunk, ToPrGreedy, PLETopK2D
from typing import Tuple, Callable

class TestGeneralizedToMe(unittest.TestCase):

    def setUp(self):
        """Set up a hook to print the test name."""
        print(f"\n--- Running test: {self._testMethodName} ---")

    def test_initialization(self):
        """Tests that the module initializes correctly and validates arguments."""
        try:
            GeneralizedToMe(r=0.5, kernel_size=2)
        except ValueError:
            self.fail("GeneralizedToMe raised ValueError unexpectedly.")

    def test_r_zero_case(self):
        """Tests the behavior when r=0 (no merging)."""
        B, N, C = 4, 16, 32
        tome = GeneralizedToMe(r=0, kernel_size=4)
        x = torch.randn(B, N, C)
        
        merged_x, btree, unmerge_fn = tome(x)
        unmerged_x = unmerge_fn(merged_x)
        
        self.assertEqual(merged_x.shape, x.shape, "With r=0, merged_x shape should be same as input.")
        self.assertTrue(torch.equal(x, merged_x), "With r=0, merged_x should be identical to input.")
        self.assertTrue(torch.all(btree == 0), "With r=0, btree should be all zeros.")
        self.assertTrue(torch.equal(x, unmerged_x), "With r=0, unmerged_x should be identical to input.")
        print("With r=0, operations correctly result in no change.")

    def test_chain_resolution(self):
        """Tests the _resolve_chains helper function."""
        # 3 -> 2, 2 -> 1, 1 -> 0 (a single long chain)
        # 7 -> 6, 6 -> 5 (a shorter chain)
        # 4 is a root
        partial_map = torch.tensor([[0, -1, -1, -1, 0, -1, -1, -1]], dtype=torch.long)
        
        expected_root_map = torch.tensor([[0, -1, -2, -3, 0, -1, -2, -3]], dtype=torch.long)
        
        # This is a static method, so we can call it directly
        resolved_map = GeneralizedToMe._resolve_chains(partial_map)
        
        self.assertTrue(torch.equal(resolved_map, expected_root_map), 
                        f"Chain resolution failed. Got:\n{resolved_map}\nExpected:\n{expected_root_map}")
        print("Chain resolution test passed.")

    def test_btree_conversion(self):
        """Tests the _convert_root_map_to_btree helper function."""
        # Group 1: 0, 1, 2, 3 -> all merge to 0
        # Group 2: 4, 5 -> merge to 4
        # Group 3: 6, 7 -> merge to 6
        direct_to_root_map = torch.tensor([[0, -1, -2, -3, 0, -1, 0, -1]], dtype=torch.long)
        
        # Expected btree: 3->2, 2->1, 1->0 | 5->4 | 7->6
        expected_btree = torch.tensor([[0, -1, -1, -1, 0, -1, 0, -1]], dtype=torch.long)
        
        btree = GeneralizedToMe._convert_root_map_to_btree(direct_to_root_map)
        
        self.assertTrue(torch.equal(btree, expected_btree),
                        f"B-tree conversion failed. Got:\n{btree}\nExpected:\n{expected_btree}")
        print("B-tree conversion test passed.")

    def test_full_merge_unmerge_logic(self):
        """
        End-to-end test for the full merge and unmerge pipeline with predictable inputs.
        """
        B, N, C = 1, 8, 4
        r = 0.25  # Merge 2 tokens
        kernel_size = 3
        
        tome = GeneralizedToMe(r=r, kernel_size=kernel_size)

        # Create a deterministic, numerically stable input.
        # Use orthogonal vectors for pairs and distinct orthogonal vectors for noise
        # to prevent NaN issues from normalizing zero vectors.
        x = torch.zeros(B, N, C, dtype=torch.float32)
        
        # Pairs with perfect similarity
        x[0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0]) # Pair 1
        x[0, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 4] = torch.tensor([0.0, 1.0, 0.0, 0.0]) # Pair 2
        x[0, 5] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        
        # Noise vectors that are orthogonal to everything else
        x[0, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x[0, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        x[0, 6] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x[0, 7] = torch.tensor([0.0, 0.0, 0.0, 1.0])


        merged_x, btree, unmerge_fn = tome(x)

        # 1. Test btree (the final output)
        # Top 2 pairs by similarity should be (3->2) and (5->4)
        expected_btree = torch.zeros(B, N, dtype=torch.long)
        expected_btree[0, 3] = -1 # 3 -> 2
        expected_btree[0, 5] = -1 # 5 -> 4
        self.assertTrue(torch.equal(btree.cpu(), expected_btree),
                        f"B-tree is not as expected. Got:\n{btree}\nExpected:\n{expected_btree}")

        # 2. Test merged_x
        expected_n_merged = N - int(r * N)
        self.assertEqual(merged_x.shape, (B, expected_n_merged, C))
        
        # Roots are [0, 1, 2, 4, 6, 7]
        expected_val1 = (x[0, 2] + x[0, 3]) / 2
        expected_val2 = (x[0, 4] + x[0, 5]) / 2
        expected_merged_x = torch.stack(
            [x[0, 0], x[0, 1], expected_val1, expected_val2, x[0, 6], x[0, 7]]
        ).unsqueeze(0)
        self.assertTrue(torch.allclose(merged_x.cpu(), expected_merged_x),
                        f"Merged tensor is not as expected. Got:\n{merged_x}\nExpected:\n{expected_merged_x}")

        # 3. Test unmerged_x
        unmerged_x = unmerge_fn(merged_x)
        expected_unmerged_x = x.clone()
        expected_unmerged_x[0, 2] = expected_val1
        expected_unmerged_x[0, 3] = expected_val1
        expected_unmerged_x[0, 4] = expected_val2
        expected_unmerged_x[0, 5] = expected_val2
        self.assertTrue(torch.allclose(unmerged_x.cpu(), expected_unmerged_x),
                        f"Unmerged tensor is not as expected. Got:\n{unmerged_x}\nExpected:\n{expected_unmerged_x}")
        
        print("Full merge/unmerge logic test passed.")


class TestOurToMeK(unittest.TestCase):

    def setUp(self):
        print(f"\n--- Running test: {self._testMethodName} ---")

    def test_initialization(self):
        try:
            OurToMeK(r=0.5, num_iterations=2, group_size=2)
        except ValueError:
            self.fail("OurToMeK raised ValueError unexpectedly.")

    def test_merge_unmerge_k2_matches_expectation(self):
        B, N, C = 1, 8, 4
        r = 0.25  # Merge 2 tokens
        group_size = 2

        tome = OurToMeK(r=r, num_iterations=1, group_size=group_size)

        # Deterministic input: two strong adjacent pairs
        x = torch.zeros(B, N, C, dtype=torch.float32)
        x[0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 4] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        x[0, 5] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        x[0, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x[0, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        x[0, 6] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x[0, 7] = torch.tensor([0.0, 0.0, 0.0, 1.0])

        merged_x, btree, avg = tome.merge(x)

        expected_btree = torch.zeros(B, N, dtype=torch.long)
        expected_btree[0, 3] = -1
        expected_btree[0, 5] = -1
        self.assertTrue(torch.equal(btree.cpu(), expected_btree))

        expected_n_merged = N - int(r * N)
        self.assertEqual(merged_x.shape, (B, expected_n_merged, C))

        expected_val1 = (x[0, 2] + x[0, 3]) / 2
        expected_val2 = (x[0, 4] + x[0, 5]) / 2
        expected_merged_x = torch.stack(
            [x[0, 0], x[0, 1], expected_val1, expected_val2, x[0, 6], x[0, 7]]
        ).unsqueeze(0)

        self.assertTrue(torch.allclose(merged_x.cpu(), expected_merged_x))



class TestToPrK2New(unittest.TestCase):

    def setUp(self):
        print(f"\n--- Running test: {self._testMethodName} ---")

    def test_initialization(self):
        try:
            ToPrK2New(r=0.25, num_iterations=2)
        except ValueError:
            self.fail("ToPrK2New raised ValueError unexpectedly.")

    def test_r_zero_case(self):
        B, N, C = 2, 10, 4
        x = torch.randn(B, N, C)
        model = ToPrK2New(r=0.0, num_iterations=2)
        with torch.no_grad():
            merged_x, btree, avg_sim = model.compute_merge(x.clone())
            direct = model.btree_to_root_map(btree)
            unmerged_x = model.unmerge(merged_x, direct)
        self.assertEqual(merged_x.shape, x.shape)
        self.assertTrue(torch.all(btree == 0))
        self.assertTrue(torch.allclose(unmerged_x, x))

    def test_prune_behavior_simple(self):
        B, N, C = 1, 8, 4
        r = 0.25  # prune 2 tokens
        model = ToPrK2New(r=r, num_iterations=1)

        # Deterministic input: two strong adjacent pairs (2,3) and (4,5)
        x = torch.zeros(B, N, C, dtype=torch.float32)
        x[0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 4] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        x[0, 5] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        x[0, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x[0, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        x[0, 6] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x[0, 7] = torch.tensor([0.0, 0.0, 0.0, 1.0])

        with torch.no_grad():
            merged_x, btree, avg_sim = model.compute_merge(x.clone())
            direct = model.btree_to_root_map(btree)
            unmerged_x = model.unmerge(merged_x, direct)

        # Expect pruning of src indices 3 and 5 (adjacent merges into 2 and 4), so btree has -1 there
        expected_btree = torch.zeros(B, N, dtype=torch.long)
        expected_btree[0, 3] = -1
        expected_btree[0, 5] = -1
        self.assertTrue(torch.equal(btree.cpu(), expected_btree))

        # merged_x keeps original root tokens [0,1,2,4,6,7] with original values
        expected_merged_x = torch.stack(
            [x[0, 0], x[0, 1], x[0, 2], x[0, 4], x[0, 6], x[0, 7]]
        ).unsqueeze(0)
        self.assertEqual(merged_x.shape, expected_merged_x.shape)
        self.assertTrue(torch.allclose(merged_x.cpu(), expected_merged_x))

        # Unmerged should copy root token into pruned positions (no averaging)
        expected_unmerged_x = x.clone()
        expected_unmerged_x[0, 3] = x[0, 2]
        expected_unmerged_x[0, 5] = x[0, 4]
        self.assertTrue(torch.allclose(unmerged_x.cpu(), expected_unmerged_x))


class TestToPrGreedy(unittest.TestCase):

    def setUp(self):
        print(f"\n--- Running test: {self._testMethodName} ---")

    def test_initialization(self):
        try:
            ToPrGreedy(r=0.25)
        except ValueError:
            self.fail("ToPrGreedy raised ValueError unexpectedly.")

    def test_r_zero_case(self):
        B, N, C = 2, 10, 4
        x = torch.randn(B, N, C)
        model = ToPrGreedy(r=0.0)
        with torch.no_grad():
            merged_x, btree, avg_sim = model.compute_merge(x.clone())
            direct = model.btree_to_root_map(btree)
            unmerged_x = model.unmerge(merged_x, direct)
        self.assertEqual(merged_x.shape, x.shape)
        self.assertTrue(torch.all(btree == 0))
        self.assertTrue(torch.allclose(unmerged_x, x))

    def test_prune_behavior_simple(self):
        B, N, C = 1, 8, 4
        r = 0.25  # prune 2 tokens
        model = ToPrGreedy(r=r)

        # Deterministic input: two strong adjacent pairs (2,3) and (4,5)
        x = torch.zeros(B, N, C, dtype=torch.float32)
        x[0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 4] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        x[0, 5] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        x[0, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x[0, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        x[0, 6] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x[0, 7] = torch.tensor([0.0, 0.0, 0.0, 1.0])

        with torch.no_grad():
            merged_x, btree, avg_sim = model.compute_merge(x.clone())
            direct = model.btree_to_root_map(btree)
            unmerged_x = model.unmerge(merged_x, direct)

        # Expect pruning of src indices 3 and 5
        expected_btree = torch.zeros(B, N, dtype=torch.long)
        expected_btree[0, 3] = -1
        expected_btree[0, 5] = -1
        self.assertTrue(torch.equal(btree.cpu(), expected_btree))

        # merged_x keeps original root tokens [0,1,2,4,6,7] with original values
        expected_merged_x = torch.stack(
            [x[0, 0], x[0, 1], x[0, 2], x[0, 4], x[0, 6], x[0, 7]]
        ).unsqueeze(0)
        self.assertEqual(merged_x.shape, expected_merged_x.shape)
        self.assertTrue(torch.allclose(merged_x.cpu(), expected_merged_x))

        # Unmerged should copy root token into pruned positions (no averaging)
        expected_unmerged_x = x.clone()
        expected_unmerged_x[0, 3] = x[0, 2]
        expected_unmerged_x[0, 5] = x[0, 4]
        self.assertTrue(torch.allclose(unmerged_x.cpu(), expected_unmerged_x))

class TestToPrK2NewChunk(unittest.TestCase):

    def setUp(self):
        print(f"\n--- Running test: {self._testMethodName} ---")

    def test_initialization(self):
        try:
            ToPrK2NewChunk(r=0.25, num_iterations=2, chunk_size=4)
        except ValueError:
            self.fail("ToPrK2NewChunk raised ValueError unexpectedly.")

    def test_r_zero_case(self):
        B, N, C = 2, 10, 4
        x = torch.randn(B, N, C)
        model = ToPrK2NewChunk(r=0.0, num_iterations=2, chunk_size=4)
        with torch.no_grad():
            merged_x, btree, avg_sim = model.compute_merge(x.clone())
            direct = model.btree_to_root_map(btree)
            unmerged_x = model.unmerge(merged_x, direct)
        self.assertEqual(merged_x.shape, x.shape)
        self.assertTrue(torch.all(btree == 0))
        self.assertTrue(torch.allclose(unmerged_x, x))

    def test_prune_behavior_simple_chunked(self):
        B, N, C = 1, 8, 4
        r = 0.25  # prune 2 tokens
        chunk_size = 4  # two chunks of length 4
        model = ToPrK2NewChunk(r=r, num_iterations=1, chunk_size=chunk_size)

        # Deterministic input: two strong adjacent pairs (2,3) and (4,5)
        x = torch.zeros(B, N, C, dtype=torch.float32)
        x[0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 4] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        x[0, 5] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        x[0, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x[0, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        x[0, 6] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x[0, 7] = torch.tensor([0.0, 0.0, 0.0, 1.0])

        with torch.no_grad():
            merged_x, btree, avg_sim = model.compute_merge(x.clone())
            direct = model.btree_to_root_map(btree)
            unmerged_x = model.unmerge(merged_x, direct)

        # Expect pruning of src indices 3 and 5 (one per chunk)
        expected_btree = torch.zeros(B, N, dtype=torch.long)
        expected_btree[0, 3] = -1
        expected_btree[0, 5] = -1
        self.assertTrue(torch.equal(btree.cpu(), expected_btree))

        # merged_x keeps original root tokens [0,1,2,4,6,7] with original values
        expected_merged_x = torch.stack(
            [x[0, 0], x[0, 1], x[0, 2], x[0, 4], x[0, 6], x[0, 7]]
        ).unsqueeze(0)
        self.assertEqual(merged_x.shape, expected_merged_x.shape)
        self.assertTrue(torch.allclose(merged_x.cpu(), expected_merged_x))

        # Unmerged should copy root token into pruned positions (no averaging)
        expected_unmerged_x = x.clone()
        expected_unmerged_x[0, 3] = x[0, 2]
        expected_unmerged_x[0, 5] = x[0, 4]
        self.assertTrue(torch.allclose(unmerged_x.cpu(), expected_unmerged_x))

class TestToPrPLETopK(unittest.TestCase):

    def setUp(self):
        print(f"\n--- Running test: {self._testMethodName} ---")

    def test_counts_and_unmerge(self):
        B, N, C = 1, 8, 4
        r = 0.25  # prune 2 tokens => keep 6
        model = ToPrPLETopK(r=r, beta=1.0, eps=1e-9, use_bin_argmax=False)

        # Deterministic input
        x = torch.zeros(B, N, C, dtype=torch.float32)
        x[0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 4] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        x[0, 5] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        x[0, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x[0, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        x[0, 6] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        x[0, 7] = torch.tensor([0.0, 0.0, 0.0, 1.0])

        with torch.no_grad():
            merged_x, btree, avg_sim = model.compute_merge(x.clone())
            direct = model.btree_to_root_map(btree)
            unmerged_x = model.unmerge(merged_x, direct)

        # counts
        kept = (btree == 0).sum().item()
        self.assertEqual(kept, N - int(r * N))
        self.assertEqual(merged_x.shape, (B, N - int(r * N), C))
        # semantics: 0/-1 only, first token kept
        self.assertTrue(torch.all((btree == 0) | (btree == -1)))
        self.assertEqual(btree[0, 0].item(), 0)

        # unmerge correctness: each pruned position copies from nearest kept on the left
        for j in range(N):
            if btree[0, j].item() == -1:
                k = j - 1
                while k >= 0 and btree[0, k].item() != 0:
                    k -= 1
                self.assertTrue(torch.allclose(unmerged_x[0, j], x[0, k]))

        # avg_sim non-negative
        self.assertGreaterEqual(avg_sim.item(), 0.0)


class TestToPrCPRRTopK(unittest.TestCase):

    def setUp(self):
        print(f"\n--- Running test: {self._testMethodName} ---")

    def test_counts_and_unmerge(self):
        B, N, C = 1, 12, 4
        r = 0.33  # floor(r*N) = floor(3.96)=3, keep 9
        model = ToPrCPRRTopK(r=r, beta=1.0, eps=1e-9, bins=None)

        # Deterministic input with multiple transitions
        x = torch.zeros(B, N, C, dtype=torch.float32)
        for i in range(N):
            x[0, i, i % C] = 1.0

        with torch.no_grad():
            merged_x, btree, avg_sim = model.compute_merge(x.clone())
            direct = model.btree_to_root_map(btree)
            unmerged_x = model.unmerge(merged_x, direct)

        kept = (btree == 0).sum().item()
        self.assertEqual(kept, N - int(r * N))
        self.assertEqual(merged_x.shape, (B, N - int(r * N), C))
        self.assertTrue(torch.all((btree == 0) | (btree == -1)))
        self.assertEqual(btree[0, 0].item(), 0)

        # unmerge correctness under left-chain
        for j in range(N):
            if btree[0, j].item() == -1:
                k = j - 1
                while k >= 0 and btree[0, k].item() != 0:
                    k -= 1
                self.assertTrue(torch.allclose(unmerged_x[0, j], x[0, k]))

        self.assertGreaterEqual(avg_sim.item(), 0.0)


class TestPLETopK2D(unittest.TestCase):

    def setUp(self):
        print(f"\n--- Running test: {self._testMethodName} ---")

    def test_counts_and_unmerge_basic(self):
        B, C = 1, 4
        W = 4
        N = 8  # 2 rows of width 4
        r = 0.25  # prune 2 -> keep 6
        model = PLETopK2D(r=r, token_width=W, use_bin_argmax=False)

        # Construct a deterministic grid-like pattern in raster order
        x = torch.zeros(B, N, C, dtype=torch.float32)
        # Row 0: indices 0..3, Row 1: indices 4..7
        x[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 2] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        x[0, 3] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        x[0, 4] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 5] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        x[0, 6] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        x[0, 7] = torch.tensor([0.0, 1.0, 0.0, 0.0])

        with torch.no_grad():
            merged_x, btree, avg_sim = model.compute_merge(x.clone())
            direct = model.btree_to_root_map(btree)
            unmerged_x = model.unmerge(merged_x, direct)

        kept = (btree == 0).sum().item()
        self.assertEqual(kept, N - int(r * N))
        self.assertEqual(merged_x.shape, (B, N - int(r * N), C))
        self.assertTrue(torch.all((btree == 0) | (btree == -1)))
        self.assertEqual(btree[0, 0].item(), 0)

        # Unmerge: pruned positions should copy from nearest kept on the left
        for j in range(N):
            if btree[0, j].item() == -1:
                k = j - 1
                while k >= 0 and btree[0, k].item() != 0:
                    k -= 1
                self.assertTrue(torch.allclose(unmerged_x[0, j], x[0, k]))

    def test_non_divisible_width(self):
        B, C = 1, 4
        W = 4
        N = 10  # not divisible by W
        r = 0.2  # prune 2 -> keep 8
        model = PLETopK2D(r=r, token_width=W, use_bin_argmax=True, fallback='pre')

        # Build simple pattern; ensure code handles missing right/last row cells without padding
        x = torch.zeros(B, N, C, dtype=torch.float32)
        for i in range(N):
            x[0, i, i % C] = 1.0

        with torch.no_grad():
            merged_x, btree, avg_sim = model.compute_merge(x.clone())
            direct = model.btree_to_root_map(btree)
            unmerged_x = model.unmerge(merged_x, direct)

        kept = (btree == 0).sum().item()
        self.assertEqual(kept, N - int(r * N))
        self.assertEqual(merged_x.shape, (B, N - int(r * N), C))
        self.assertTrue(torch.all((btree == 0) | (btree == -1)))
        self.assertEqual(btree[0, 0].item(), 0)

        # Unmerge left-chain correctness
        for j in range(N):
            if btree[0, j].item() == -1:
                k = j - 1
                while k >= 0 and btree[0, k].item() != 0:
                    k -= 1
                self.assertTrue(torch.allclose(unmerged_x[0, j], x[0, k]))

if __name__ == '__main__':
    unittest.main()
