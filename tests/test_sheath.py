"""Sheath module checks against analytical limits."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blackout.sheath import (  # noqa: E402
    IonMobility,
    child_law_uniform,
    collisional_sheath,
    high_field_child_uniform,
    matrix_sheath,
    matrix_sheath_uniform,
)

N_UNIFORM = 5e21


class TestMatrixSheath(unittest.TestCase):
    def test_uniform_profile_matches_analytical(self):
        y = np.linspace(0, 0.02, 20001)
        n = np.full_like(y, 2.31e17)
        self.assertAlmostEqual(matrix_sheath(7500, y, n) / matrix_sheath_uniform(7500, 2.31e17), 1.0, places=4)

    def test_value_at_arizona_conditions(self):
        # 7.5 kV into 2.31e17 m^-3: about 1.9 mm
        self.assertAlmostEqual(matrix_sheath_uniform(7500, 2.31e17) * 1e3, 1.89, places=2)


class TestCollisionalSheath(unittest.TestCase):
    def test_low_field_reduces_to_child_law(self):
        mob = IonMobility("low")
        mu = mob.low_field_mobility(N_UNIFORM)
        d, _ = collisional_sheath(7500, 150.0, mob, lambda y: N_UNIFORM)
        self.assertAlmostEqual(d / child_law_uniform(7500, 150.0, mu), 1.0, places=3)

    def test_high_field_reduces_to_analytical(self):
        mob = IonMobility("high", K0=1e3)  # huge low-field mobility: always saturated
        d, _ = collisional_sheath(7500, 150.0, mob, lambda y: N_UNIFORM)
        self.assertAlmostEqual(d / high_field_child_uniform(7500, 150.0, N_UNIFORM), 1.0, places=3)

    def test_child_law_scalings(self):
        # d ~ mu^(1/3), d ~ J^(-1/3), d ~ V^(2/3)
        d0 = child_law_uniform(5000, 100, 1.0)
        self.assertAlmostEqual(child_law_uniform(5000, 100, 8.0) / d0, 2.0, places=10)
        self.assertAlmostEqual(child_law_uniform(5000, 800, 1.0) / d0, 0.5, places=10)
        self.assertAlmostEqual(child_law_uniform(5000 * 8**1.5, 100, 1.0) / d0, 8.0, places=8)

    def test_high_field_mobility_shrinks_sheath(self):
        N = lambda y: N_UNIFORM
        d_low, _ = collisional_sheath(7500, 150.0, IonMobility("low"), N)
        d_high, _ = collisional_sheath(7500, 150.0, IonMobility("high"), N)
        self.assertLess(d_high, d_low)


if __name__ == "__main__":
    unittest.main()
