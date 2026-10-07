"""Checks for the current closure, fast sheath, batched propagation and window model."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackout.atmosphere import atmosphere, radiative_equilibrium_temperature, sutton_graves  # noqa: E402
from blackout.closure import calibrate  # noqa: E402
from blackout.plasma import KB  # noqa: E402
from blackout.propagation import _eps_physics, transfer_matrix, transfer_matrix_batch  # noqa: E402
from blackout.sheath import HighFieldSheathProfile, IonMobility, collisional_sheath  # noqa: E402
from blackout.window import FlightPlasma, nominal  # noqa: E402


class TestClosure(unittest.TestCase):
    def setUp(self):
        self.c = calibrate()

    def test_fit_quality(self):
        self.assertLess(self.c.fit_rms, 0.05)

    def test_reproduces_cfd_current_at_7p5kV(self):
        J = self.c.current_density(7500, 2.38e17, 8100, gamma_e=0.1)
        self.assertAlmostEqual(J / 150.0, 1.0, delta=0.05)

    def test_gamma_matches_fig18b_at_7p5kV(self):
        J1 = self.c.current_density(7500, 2.38e17, 8100, gamma_e=0.1)
        J5 = self.c.current_density(7500, 2.38e17, 8100, gamma_e=0.5)
        self.assertAlmostEqual(J5 / J1, 3.74, delta=0.4)

    def test_supply_coefficient_is_physical(self):
        self.assertTrue(0.3 < self.c.c_s < 1.0)


class TestFastSheath(unittest.TestCase):
    def test_matches_ode_on_variable_density(self):
        y = np.linspace(0, 0.03, 241)
        T = 1400 + 6800 * np.sin(np.clip(y / 0.02, 0, 1) * np.pi / 2)
        N = 500 / (KB * T)
        hf = HighFieldSheathProfile(y, N)
        d_ode, _ = collisional_sheath(7500, 150, IonMobility("high", K0=1e3),
                                      lambda q: 500 / (KB * np.interp(q, y, T)))
        self.assertAlmostEqual(float(hf.thickness(7500, 150)) / d_ode, 1.0, places=3)


class TestBatchPropagation(unittest.TestCase):
    def test_batch_equals_single(self):
        f, d = 2.25e9, np.full(20, 0.002)
        rng = np.random.default_rng(0)
        n = 10 ** rng.uniform(15, 17.5, (5, 20))
        eps = _eps_physics(n, 1e8, f)
        Tb = transfer_matrix_batch(eps, d, f)
        for i in range(5):
            self.assertAlmostEqual(Tb[i], transfer_matrix(eps[i], d, f)[1], places=10)


class TestWindow(unittest.TestCase):
    def test_underdense_needs_no_sheath(self):
        d_req, loss = FlightPlasma(82.0, nominal()).required_depth()
        self.assertEqual(d_req, 0.0)
        self.assertLess(loss, 1.0)

    def test_blackout_onset_near_ramc2(self):
        # Nominal S-band loss crosses 10 dB within a few km of RAM-C II station 3 S-band cut-off (73.9 km)
        h = np.arange(80, 64, -0.5)
        loss = np.array([FlightPlasma(x, nominal()).required_depth()[1] for x in h])
        onset = h[np.argmax(loss > 10.0)]
        self.assertTrue(68.0 <= onset <= 76.0)

    def test_heating_increases_with_duty(self):
        fp = FlightPlasma(70.0, nominal())
        self.assertGreater(fp.wall_temperature(7500, 1.0), fp.wall_temperature(7500, 0.05))


class TestAtmosphere(unittest.TestCase):
    def test_density_at_70km(self):
        self.assertAlmostEqual(atmosphere(70)[0] / 8.283e-5, 1.0, places=6)

    def test_heating_and_radiation(self):
        q = sutton_graves(8.283e-5, 7650.0)
        self.assertTrue(1.5e6 < q < 2.2e6)
        self.assertAlmostEqual(radiative_equilibrium_temperature(0.85 * 5.670374e-8 * 1500**4), 1500, places=6)


if __name__ == "__main__":
    unittest.main()
