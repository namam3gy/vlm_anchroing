import unittest

import numpy as np

from vlm_anchor.judge_pilot_data import paired_bootstrap_ci


class TestPairedBootstrapCI(unittest.TestCase):
    def test_zero_difference_ci_includes_zero(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, size=200)
        y = x.copy()
        ci = paired_bootstrap_ci(x, y, n_resamples=2000, alpha=0.05, rng_seed=42)
        # Non-strict bracketing: y = x.copy() yields width-0 CI at exactly 0.0,
        # which still satisfies "CI includes zero" semantics.
        self.assertLessEqual(ci.lo, 0.0)
        self.assertGreaterEqual(ci.hi, 0.0)
        self.assertAlmostEqual(ci.point, 0.0, places=10)

    def test_constant_negative_difference_ci_excludes_zero(self) -> None:
        x = np.zeros(100)
        y = x - 1.0  # y - x = -1 everywhere (anchor=1 floor push direction)
        ci = paired_bootstrap_ci(x, y, n_resamples=1000, alpha=0.05, rng_seed=42)
        self.assertAlmostEqual(ci.point, -1.0, places=10)
        self.assertAlmostEqual(ci.lo, -1.0, places=10)
        self.assertAlmostEqual(ci.hi, -1.0, places=10)

    def test_drops_pairs_with_nan(self) -> None:
        x = np.array([1.0, 2.0, 3.0, np.nan])
        y = np.array([0.0, 1.0, 2.0, 5.0])
        ci = paired_bootstrap_ci(x, y, n_resamples=500, alpha=0.05, rng_seed=42)
        self.assertAlmostEqual(ci.point, -1.0, places=10)
        self.assertEqual(ci.n_pairs, 3)


if __name__ == "__main__":
    unittest.main()
