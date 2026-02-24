import math
import unittest

import torch

try:
    from dtp.benchmark_vfr_runtime import (
        calibrate_fixed_tau,
        default_algo_specs,
        evaluate_mean_avg_r,
        instantiate_selector,
        benchmark_wall_time,
        make_synthetic_batches,
    )
except ImportError:
    from benchmark_vfr_runtime import (
        calibrate_fixed_tau,
        default_algo_specs,
        evaluate_mean_avg_r,
        instantiate_selector,
        benchmark_wall_time,
        make_synthetic_batches,
    )


class TestVFRBenchmarkSmoke(unittest.TestCase):
    def test_selector_forward_shapes(self):
        specs = default_algo_specs()
        x = torch.randn(2, 32, 16)
        for spec in specs:
            with self.subTest(algo=spec.key):
                selector = instantiate_selector(
                    spec=spec,
                    target_r=0.5,
                    fixed_tau=None,
                    device=torch.device("cpu"),
                )
                with torch.no_grad():
                    mask, avg_r, tau = selector(x)

                self.assertEqual(mask.shape, (2, 32))
                self.assertEqual(mask.dtype, torch.bool)
                self.assertTrue(torch.isfinite(avg_r).item())
                self.assertTrue(torch.isfinite(tau).item())
                self.assertTrue(torch.all(mask[:, 0]).item())

    def test_calibration_and_timing_smoke(self):
        specs = default_algo_specs()
        batches = make_synthetic_batches(
            batch_size=2,
            seq_len=32,
            dim=16,
            num_batches=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=123,
        )

        for spec in specs:
            with self.subTest(algo=spec.key):
                tau_star, calib_avg, calib_err = calibrate_fixed_tau(
                    spec=spec,
                    target_r=0.5,
                    batches=batches,
                    device=torch.device("cpu"),
                    coarse_points=6,
                    refine_points=5,
                )
                self.assertTrue(math.isfinite(tau_star))
                self.assertTrue(math.isfinite(calib_avg))
                self.assertGreaterEqual(calib_err, 0.0)

                selector = instantiate_selector(
                    spec=spec,
                    target_r=0.5,
                    fixed_tau=tau_star,
                    device=torch.device("cpu"),
                )
                measured_r = evaluate_mean_avg_r(selector, batches)
                self.assertTrue(0.0 <= measured_r <= 1.0)

                mean_ms, std_ms, raw_ms = benchmark_wall_time(
                    selector=selector,
                    batches=batches,
                    device=torch.device("cpu"),
                    warmup=1,
                    repeats=2,
                )
                self.assertEqual(len(raw_ms), 2)
                self.assertGreaterEqual(mean_ms, 0.0)
                self.assertGreaterEqual(std_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
