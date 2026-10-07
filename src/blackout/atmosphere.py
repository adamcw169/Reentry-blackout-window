"""Atmosphere, RAM-C II trajectory and aerothermal heating."""

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R_AIR = 287.05
SIGMA_SB = 5.670374e-8
RAMC_NOSE_RADIUS = 0.1524  # m

# US Standard Atmosphere 1976: altitude [km], density [kg/m^3], temperature [K]
_US76 = np.array([
    [40, 3.996e-3, 250.4], [45, 1.966e-3, 264.2], [50, 1.027e-3, 270.7],
    [55, 5.680e-4, 260.8], [60, 3.097e-4, 247.0], [65, 1.632e-4, 233.3],
    [70, 8.283e-5, 219.6], [75, 3.992e-5, 208.4], [80, 1.846e-5, 198.6],
    [85, 8.220e-6, 188.9], [90, 3.416e-6, 186.9],
])


def atmosphere(h_km):
    """(density kg/m^3, temperature K, pressure Pa) at altitude h_km (40-90 km)."""
    rho = 10 ** np.interp(h_km, _US76[:, 0], np.log10(_US76[:, 1]))
    T = np.interp(h_km, _US76[:, 0], _US76[:, 2])
    return rho, T, rho * R_AIR * T


def ramc2_velocity(h_km):
    """RAM-C II flight velocity [m/s] at altitude, from NASA TN D-6062 Table IX."""
    rows = list(csv.DictReader(open(ROOT / "data" / "ramc2_trajectory.csv")))
    h = np.array([float(r["altitude_km"]) for r in rows])[::-1]
    v = np.array([float(r["velocity_km_s"]) for r in rows])[::-1]
    return 1e3 * np.interp(h_km, h, v)


def ramc2_time(h_km):
    rows = list(csv.DictReader(open(ROOT / "data" / "ramc2_trajectory.csv")))
    h = np.array([float(r["altitude_km"]) for r in rows])[::-1]
    t = np.array([float(r["time_s"]) for r in rows])[::-1]
    return np.interp(h_km, h, t)


def sutton_graves(rho, V, R_n=RAMC_NOSE_RADIUS):
    """Stagnation-point convective heat flux [W/m^2] for Earth air."""
    return 1.7415e-4 * np.sqrt(rho / R_n) * V**3


def radiative_equilibrium_temperature(q, emissivity=0.85):
    """Wall temperature [K] at which re-radiation balances heat flux q [W/m^2]."""
    return (np.asarray(q, dtype=float) / (emissivity * SIGMA_SB)) ** 0.25
