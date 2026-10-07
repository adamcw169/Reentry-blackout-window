"""Calibrated cathode current closure.

    J(V) = J_s * [1 + g(gamma_e) * (V / V_m)^m]
    J_s  = c_s * e * n_0 * u_B,     u_B = sqrt(k T / m_i)

* J_s is an ion supply term: a Bohm-like ion flux from the plasma at the sheath
  edge, reduced by collisions (c_s < 1).
* The power-law term is the extra current from ionisation inside the sheath.
  It grows roughly as V^2, as a collisional Child law does at fixed thickness.
* g(gamma_e) scales the ionisation term with the cathode's secondary emission
  coefficient, so that total power at 7.5 kV follows the gamma_e sweep in
  Rodriguez Fuentes & Parent (2026), Fig. 18b.

Calibration data: the high-field-corrected curve of their Fig. 16 (68 km,
gamma_e = 0.1 baseline), digitised in data/arizona2026/. Only one flight
condition is available, so the density and temperature scaling through J_s is
a physical assumption; its uncertainty (factor c_s) is carried into the QMC.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from .plasma import KB, QE

AMU = 1.66054e-27
M_ION = 30.0 * AMU          # NO+, dominant ion in air re-entry plasma
L_CATHODE_CFD = 0.040        # m, Arizona cathode length
N0_CFD = 2.38e17             # m^-3, peak of the digitised no-pulse profile
T_CFD = 8100.0               # K, gas temperature at that peak

# Fig. 18b, power deposited at 7.5 kV relative to gamma_e = 0.1
# (curves integrated by eye, about +/-10%)
GAMMA_POWER_RATIO = {0.1: 1.00, 0.2: 1.39, 0.3: 2.02, 0.5: 3.74}

DATA = Path(__file__).resolve().parents[2] / "data" / "arizona2026" / "cathode_iv_7p5kV.csv"


def bohm_speed(T, m_i=M_ION):
    return np.sqrt(KB * np.asarray(T, dtype=float) / m_i)


def _power_law(V, Js, Vm, m):
    return Js * (1.0 + (V / Vm) ** m)


@dataclass
class CurrentClosure:
    c_s: float
    V_m: float
    m: float
    gamma_slope: float      # ln(power ratio) per unit (gamma_e - 0.1)
    fit_rms: float

    def gamma_factor(self, gamma_e, V_ref=7500.0):
        """Multiplier g on the ionisation term, matched to Fig. 18b total power at V_ref."""
        M = np.exp(self.gamma_slope * (np.asarray(gamma_e, dtype=float) - 0.1))
        R = (V_ref / self.V_m) ** self.m
        return (M * (1.0 + R) - 1.0) / R

    def current_density(self, V, n0, T, gamma_e=0.1, c_s_factor=1.0):
        """Cathode ion current density [A/m^2] at sheath voltage V [V]."""
        Js = c_s_factor * self.c_s * QE * n0 * bohm_speed(T)
        return Js * (1.0 + self.gamma_factor(gamma_e) * (np.asarray(V, dtype=float) / self.V_m) ** self.m)


def calibrate(path=DATA):
    rows = list(csv.DictReader(open(path)))
    V = -np.array([float(r["V_kV"]) for r in rows]) * 1e3
    J = np.array([float(r["I_corrected_A_per_m"]) for r in rows]) / L_CATHODE_CFD
    (Js, Vm, m), _ = curve_fit(_power_law, V, J, p0=(40.0, 5000.0, 2.0), maxfev=20000)
    rms = float(np.sqrt(np.mean(((_power_law(V, Js, Vm, m) - J) / J) ** 2)))
    c_s = Js / (QE * N0_CFD * bohm_speed(T_CFD))
    g = np.array([k for k in GAMMA_POWER_RATIO if k > 0.1]) - 0.1
    lnM = np.log([GAMMA_POWER_RATIO[k] for k in GAMMA_POWER_RATIO if k > 0.1])
    slope = float(np.sum(g * lnM) / np.sum(g * g))
    return CurrentClosure(c_s=float(c_s), V_m=float(Vm), m=float(m), gamma_slope=slope, fit_rms=rms)
