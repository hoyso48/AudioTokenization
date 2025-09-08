import os
import unittest
import torch

from .benchmark_tome import run_benchmark


class TestBenchmarkToMe(unittest.TestCase):

    def setUp(self):
        print(f"\n--- Running test: {self._testMethodName} ---")

    def test_benchmark_cpu_small(self):
        filelist = "/home/hoyso/projects/AudioTokenization/BigCodec_SSL/filelists/librispeech_test_clean.txt"
        if not os.path.exists(filelist):
            self.skipTest(f"Missing filelist: {filelist}")

        device_str = "cpu"
        results = run_benchmark(
            filelist=filelist,
            device_str=device_str,
            sample_rate=16000,
            seconds=4,
            batch_size=8,
            r=0.5,
            iterations_list=[2],
            hf_model_name="microsoft/wavlm-large",
            use_sv_loader=True,
            sv_checkpoint="/home/hoyso/projects/AudioTokenization/BigCodec_SSL/wavlm_large_finetune.pth",
            seed=123,
        )

        self.assertTrue(len(results) >= 3)  # chained, greedy, and at least one iter-based
        for r in results:
            self.assertIn("method", r)
            self.assertIn("runtime_ms", r)
            self.assertIn("avg_sim_mean", r)
            self.assertIn("cos_sim_unmerged_vs_original", r)
            # sanity checks
            self.assertGreaterEqual(r["n_before"], r["n_after"])
            self.assertTrue(torch.isfinite(torch.tensor(r["avg_sim_mean"])) )
            self.assertTrue(torch.isfinite(torch.tensor(r["cos_sim_unmerged_vs_original"])) )


if __name__ == "__main__":
    unittest.main(verbosity=2)


