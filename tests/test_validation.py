"""Regression checks for the validation cases."""

import csv
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "data"))

from blackout.propagation import slab_absorption_transmission  # noqa: E402
import v2_arizona_table4 as v2  # noqa: E402
import build_ramc2  # noqa: E402


class TestV2(unittest.TestCase):
    def test_table4_within_5_percent(self):
        model = slab_absorption_transmission(v2.N_P, 1e7, v2.D_P, v2.F_GHZ * 1e9)
        err = np.abs(model - v2.I_OVER_I0) / v2.I_OVER_I0
        self.assertLess(err.max(), 0.05)

    def test_eq18_as_printed_does_not_reproduce_table(self):
        from blackout.propagation import slab_absorption_transmission_as_printed

        model = slab_absorption_transmission_as_printed(v2.N_P, 1e7, v2.D_P, v2.F_GHZ * 1e9)
        err = np.abs(model - v2.I_OVER_I0) / v2.I_OVER_I0
        self.assertGreater(err.max(), 0.5)


class TestV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import v3_arizona_7p5kV as v3

        cls.v3 = v3
        cls.y, cls.n, cls.T = v3.load_profile()

    def test_corrected_sheath_inside_cfd_range(self):
        d_mid, d_lo, d_hi, _ = self.v3.sheath_band("corrected", self.y, self.T)
        lo, hi = self.v3.CFD_DS_MM["corrected"]
        self.assertTrue(lo <= d_lo * 1e3 and d_hi * 1e3 <= hi)

    def test_mobility_correction_shrinks_sheath(self):
        d_u = self.v3.sheath_band("uncorrected", self.y, self.T)[0]
        d_c = self.v3.sheath_band("corrected", self.y, self.T)[0]
        self.assertTrue(0.33 <= d_c / d_u <= 0.75)

    def test_density_above_sheath_matches_table4(self):
        d_c = self.v3.sheath_band("corrected", self.y, self.T)[0]
        _, N_p, _ = self.v3.window_after_sheath(d_c, self.y, self.n)
        self.assertAlmostEqual(N_p / 1.14e17, 1.0, delta=0.15)


class TestRamC2(unittest.TestCase):
    def test_time_derived_altitudes_match_printed(self):
        rows = build_ramc2.build()
        diffs = [abs(r["altitude_km"] - r["printed_altitude_km"]) for r in rows
                 if r["printed_altitude_km"] != "" and "c:" not in r["notes"]]
        self.assertGreaterEqual(len(diffs), 10)
        self.assertLess(max(diffs), 0.5)

    def test_density_increases_as_vehicle_descends(self):
        rows = list(csv.DictReader(open(ROOT / "data" / "ramc2_reflectometer.csv")))
        st2 = sorted((r for r in rows if r["station"] == "2" and r["use_for_fit"] == "yes"),
                     key=lambda r: -float(r["altitude_km"]))
        n = [float(r["n_e_peak_m3"]) for r in st2]
        self.assertEqual(n, sorted(n))


if __name__ == "__main__":
    unittest.main()
