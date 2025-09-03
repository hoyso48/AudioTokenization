import unittest
import torch
from .tome_ops import GeneralizedToMe, OurToMeK, OurToMe2
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



if __name__ == '__main__':
    unittest.main()
