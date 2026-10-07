"""Physics checks for the plasma and propagation modules (unittest / pytest)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blackout.plasma import C0, critical_density, debye_length, plasma_frequency  # noqa: E402
from blackout.propagation import (  # noqa: E402
    slab_absorption_transmission,
    transfer_matrix,
)


class TestPlasma(unittest.TestCase):
    def test_plasma_frequency_rule_of_thumb(self):
        # f_p ~ 8.98 sqrt(n_e) Hz
        for n in (1e16, 1e17, 1e18, 1e19):
            self.assertAlmostEqual(plasma_frequency(n) / (8.98 * np.sqrt(n)), 1.0, places=3)

    def test_critical_density_inverse(self):
        f = 4e9
        self.assertAlmostEqual(plasma_frequency(critical_density(f)) / f, 1.0, places=10)

    def test_debye_length_order(self):
        # 1e17 m^-3, 1 eV electrons -> ~23 micrometres
        lam = debye_length(1e17, 11604.5)
        self.assertTrue(20e-6 < lam < 26e-6)


class TestTransferMatrix(unittest.TestCase):
    def test_vacuum_slab_fully_transmits(self):
        R, T, A = transfer_matrix([1.0], [0.05], 3e9)
        self.assertAlmostEqual(T, 1.0, places=12)
        self.assertAlmostEqual(R, 0.0, places=12)

    def test_half_wave_dielectric_is_transparent(self):
        f, n = 3e9, 2.0
        d = C0 / f / (2 * n)
        R, T, _ = transfer_matrix([n**2], [d], f)
        self.assertAlmostEqual(T, 1.0, places=10)

    def test_quarter_wave_dielectric_reflectance(self):
        # Free space / n / free space, quarter wave: R = ((1 - n^2)/(1 + n^2))^2
        f, n = 3e9, 2.0
        d = C0 / f / (4 * n)
        R, T, _ = transfer_matrix([n**2], [d], f)
        self.assertAlmostEqual(R, ((1 - n**2) / (1 + n**2)) ** 2, places=10)
        self.assertAlmostEqual(R + T, 1.0, places=10)

    def test_energy_conservation_with_loss(self):
        eps = 1 - 3.0 / (1 + 0.5j)  # lossy, physics convention Im > 0
        R, T, A = transfer_matrix([eps], [0.02], 4e9)
        self.assertGreater(A, 0.0)
        self.assertAlmostEqual(R + T + A, 1.0, places=12)

    def test_layer_splitting_invariance(self):
        eps = 1 - 2.0 / (1 + 0.1j)
        one = transfer_matrix([eps], [0.03], 2e9)
        many = transfer_matrix([eps] * 30, [0.001] * 30, 2e9)
        np.testing.assert_allclose(one, many, atol=1e-10)

    def test_thick_lossy_slab_tracks_absorption_formula(self):
        # Underdense, weakly reflecting: transfer matrix ~ exp(-2 alpha d)
        from blackout.propagation import _eps_physics

        n_e, nu, d, f = 2e16, 1e9, 0.1, 8e9
        _, T, _ = transfer_matrix([_eps_physics(n_e, nu, f)], [d], f)
        self.assertAlmostEqual(T, slab_absorption_transmission(n_e, nu, d, f), places=2)


if __name__ == "__main__":
    unittest.main()
