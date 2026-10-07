"""Communication-window model along a RAM-C II-type trajectory.

For one flight condition and one set of uncertain inputs, this module answers:
  1. how deep must the electron-free sheath be for the link to close? (d_req)
  2. what sheath does a pulse of voltage V make?                      (d_s)
  3. what does it cost: current, power, electrode heating, arc margin?

Plasma above the cathode: the digitised Arizona 68 km profile shape, scaled to the
RAM-C II peak electron density at station 3 (x/D = 2.30, aft cone, where an
antenna and electrode pair would sit), and stretched by an uncertain thickness
factor. Gas temperature keeps the Arizona shape. Wall pressure p = p_inf + Cp q_inf.

Pulses are square: on for a fraction D of each period. Sheath formation (ion
transit, ~1 us) and collapse (electron return, ~10 ns) are fast against ms pulses,
so the window is open for ~D of the time (see ``timescales``).
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .atmosphere import atmosphere, radiative_equilibrium_temperature, ramc2_velocity, sutton_graves
from .closure import AMU, M_ION, calibrate
from .plasma import EPS0, KB, ME, QE, critical_density, plasma_angular_frequency
from .propagation import C0, _refractive_index, transfer_matrix_batch
from .sheath import K_HIGH_FIELD, HighFieldSheathProfile

ROOT = Path(__file__).resolve().parents[2]
CLOSURE = calibrate()

# Design and modelling constants
F_LINK = 2.25e9            # Hz, S-band telemetry
LOSS_MAX_DB = 10.0         # one-way plasma loss the link budget can absorb
L_ELECTRODE = 0.040        # m
T_WALL_MAX = 1800.0        # K, electrode / insulator surface limit
EMISSIVITY = 0.85
NEUTRALISATION_EV = 5.0    # energy released per ion neutralised at the cathode (9.26 eV - work function)
M_NEUTRAL = 29.0 * AMU
Y_GRID = np.linspace(0.0, 0.130, 261)  # m, 0.5 mm cells

# Uncertain inputs: name -> (low, high, scale). Ranges and sources in the README.
UNCERTAIN = {
    "log10_n_factor": (-0.30, 0.30, "lin"),   # RAM-C II reflectometer factor-of-2 bar
    "thickness_factor": (2.00, 4.00, "lin"),  # RAM-C II aft layer >7 cm (probe rake) vs 2.5 cm Arizona wedge
    "Cp_wall": (0.04, 0.08, "lin"),           # wall pressure coefficient, 9 deg cone
    "K_factor": (0.80, 1.20, "lin"),          # high-field ion drift constant
    "gamma_e": (0.10, 0.50, "lin"),           # secondary emission coefficient (Arizona range)
    "cs_factor": (0.50, 1.50, "lin"),         # ion supply scaling away from the calibration point
    "phi_ionisation": (0.00, 1.00, "lin"),    # share of sheath-generated ions that shield the sheath
    "sigma_factor": (0.50, 2.00, "log"),      # electron-neutral cross-section (collision frequency)
    "kappa_energy": (0.25, 1.00, "lin"),      # fraction of ion drift energy deposited in the cathode
    "q_ratio": (0.05, 0.12, "lin"),           # aft-cone / stagnation heating ratio at x/D = 2.3
    "log10_J_arc": (4.00, 5.00, "lin"),       # glow-to-arc current density, A/m^2
}
LINK_INPUTS = list(UNCERTAIN)[:8]


def scale_unit(u):
    """Map unit-cube samples (n, k) to physical values, one column per UNCERTAIN entry."""
    u = np.atleast_2d(u)
    out = {}
    for j, (name, (lo, hi, sc)) in enumerate(UNCERTAIN.items()):
        x = u[:, j] if j < u.shape[1] else np.full(u.shape[0], 0.5)
        out[name] = lo * (hi / lo) ** x if sc == "log" else lo + (hi - lo) * x
    return out


def nominal():
    return {k: (lo * hi) ** 0.5 if sc == "log" else 0.5 * (lo + hi) for k, (lo, hi, sc) in UNCERTAIN.items()}


# ---------------------------------------------------------------------------
# Plasma at a flight condition
# ---------------------------------------------------------------------------

def _load_shape():
    rows = list(csv.DictReader(open(ROOT / "data" / "arizona2026" / "profile_68km_no_pulse.csv")))
    y = np.array([float(r["y_m"]) for r in rows])
    n = np.array([float(r["n_e_m3"]) for r in rows])
    T = np.array([float(r["T_gas_K"]) for r in rows])
    return y, n / n.max(), T


def _load_ramc_peak():
    rows = [r for r in csv.DictReader(open(ROOT / "data" / "ramc2_reflectometer.csv"))
            if r["station"] == "3" and r["event"] == "onset"]
    h = np.array([float(r["altitude_km"]) for r in rows])
    n = np.array([float(r["n_e_peak_m3"]) for r in rows])
    o = np.argsort(h)
    return h[o], n[o]


SHAPE_Y, SHAPE_N, SHAPE_T = _load_shape()
RAMC_H, RAMC_N = _load_ramc_peak()


def ramc_peak_density(h_km):
    """Peak n_e at RAM-C II station 3, log-linear through the reflectometer onset points."""
    logn = np.log10(RAMC_N)
    if h_km > RAMC_H[-1]:   # extrapolate above the highest point with the upper slope
        slope = (logn[-1] - logn[-2]) / (RAMC_H[-1] - RAMC_H[-2])
        return 10 ** (logn[-1] + slope * (h_km - RAMC_H[-1]))
    return 10 ** np.interp(h_km, RAMC_H, logn)


@dataclass
class FlightPlasma:
    h_km: float
    p: dict
    y: np.ndarray = field(default_factory=lambda: Y_GRID)

    def __post_init__(self):
        p = self.p
        self.V_inf = ramc2_velocity(self.h_km)
        rho, T_inf, p_inf = atmosphere(self.h_km)
        self.rho_inf, self.q_inf = rho, 0.5 * rho * self.V_inf**2
        self.p_wall = p_inf + p["Cp_wall"] * self.q_inf
        self.n_peak = ramc_peak_density(self.h_km) * 10 ** p["log10_n_factor"]
        s = p["thickness_factor"]
        self.n_e = self.n_peak * np.interp(self.y / s, SHAPE_Y, SHAPE_N, right=0.0)
        self.T = np.interp(self.y / s, SHAPE_Y, SHAPE_T)
        self.N = self.p_wall / (KB * self.T)
        v_e = np.sqrt(8 * KB * self.T / (np.pi * ME))
        self.nu = self.N * 1e-19 * p["sigma_factor"] * v_e
        i_pk = int(np.argmax(self.n_e))
        self.n_edge, self.T_edge = self.n_peak, self.T[i_pk]
        self.sheath = HighFieldSheathProfile(self.y, self.N, K_HIGH_FIELD["NO+"] * p["K_factor"])

    # --- link -------------------------------------------------------------
    def required_depth(self, f=F_LINK, loss_max_db=LOSS_MAX_DB, n_cand=131):  # 1 mm steps
        """Smallest electron-free depth [m] from the wall that keeps loss <= loss_max for all deeper cuts."""
        d_c = np.linspace(0.0, self.y[-1], n_cand)
        mids = 0.5 * (self.y[1:] + self.y[:-1])
        n_mid = np.interp(mids, self.y, self.n_e)
        nu_mid = np.interp(mids, self.y, self.nu)
        omega = 2 * np.pi * f
        n_cut = np.where(mids[None, :] < d_c[:, None], 0.0, n_mid[None, :])
        eps = 1.0 - plasma_angular_frequency(n_cut) ** 2 / (omega * (omega + 1j * nu_mid[None, :]))
        Tr = transfer_matrix_batch(eps, np.diff(self.y), f)
        loss = -10 * np.log10(np.maximum(Tr, 1e-30))
        ok = loss <= loss_max_db
        if ok[0]:
            return 0.0, float(loss[0])
        bad = np.where(~ok)[0]
        if bad[-1] == len(d_c) - 1:
            return np.inf, float(loss[0])
        return float(d_c[bad[-1] + 1]), float(loss[0])

    # --- discharge ----------------------------------------------------------
    def currents(self, V):
        p = self.p
        J_tot = CLOSURE.current_density(V, self.n_edge, self.T_edge, p["gamma_e"], p["cs_factor"])
        J_s = CLOSURE.current_density(0.0, self.n_edge, self.T_edge, p["gamma_e"], p["cs_factor"])
        J_eff = J_s + p["phi_ionisation"] * (J_tot - J_s)
        return J_tot, J_eff

    def sheath_thickness(self, V):
        _, J_eff = self.currents(V)
        return self.sheath.thickness(V, J_eff)

    def cathode_heat_flux(self, V):
        """Instantaneous heat flux into the cathode from ion impact and neutralisation [W/m^2]."""
        J_tot, J_eff = self.currents(V)
        d = np.minimum(self.sheath.thickness(V, J_eff), self.y[-1])
        E_w = self.sheath.wall_field(d, J_eff)
        v_d = self.sheath.K * np.sqrt(E_w / self.N[0])
        eps_ion = 0.5 * (M_ION + M_NEUTRAL) * v_d**2           # Wannier mean ion energy
        return J_tot / QE * (self.p["kappa_energy"] * eps_ion + NEUTRALISATION_EV * QE)

    def aero_heat_flux(self):
        return self.p["q_ratio"] * sutton_graves(self.rho_inf, self.V_inf)

    def wall_temperature(self, V, D):
        return radiative_equilibrium_temperature(self.aero_heat_flux() + D * self.cathode_heat_flux(V), EMISSIVITY)

    def gas_heating(self, V):
        """Upper-bound gas temperature rise in the sheath over one electrode residence time [K]."""
        J_tot, J_eff = self.currents(V)
        d = np.minimum(self.sheath.thickness(V, J_eff), self.y[-1])
        q_gas = V * J_tot - self.cathode_heat_flux(V)
        m = self.y <= d
        rho = np.mean(self.p_wall / (R_DISSOC * self.T[m])) if m.any() else self.p_wall / (R_DISSOC * self.T[0])
        tau = L_ELECTRODE / (0.5 * self.V_inf)
        return q_gas / d * tau / (rho * CP_GAS)


R_DISSOC = 320.0   # J/kg/K, partly dissociated air
CP_GAS = 1500.0    # J/kg/K


def evaluate(h_km, params, V_grid, D_grid):
    """Per-sample feasibility over a voltage x duty-cycle grid.

    Returns dict with arrays shaped (n_samples, nV) or (n_samples, nV, nD).
    """
    n = len(next(iter(params.values())))
    nV, nD = len(V_grid), len(D_grid)
    link = np.zeros((n, nV), bool)
    Tw = np.zeros((n, nV, nD))
    arc_ok = np.zeros((n, nV), bool)
    P = np.zeros((n, nV))
    d_req = np.zeros(n)
    loss0 = np.zeros(n)
    for i in range(n):
        fp = FlightPlasma(h_km, {k: v[i] for k, v in params.items()})
        d_req[i], loss0[i] = fp.required_depth()
        d_s = fp.sheath_thickness(V_grid)
        J_tot, _ = fp.currents(V_grid)
        link[i] = d_s >= d_req[i]
        arc_ok[i] = J_tot <= 10 ** fp.p["log10_J_arc"]
        P[i] = V_grid * J_tot
        q_c = fp.cathode_heat_flux(V_grid)
        Tw[i] = radiative_equilibrium_temperature(fp.aero_heat_flux() + q_c[:, None] * D_grid[None, :], EMISSIVITY)
    feasible = link[:, :, None] & arc_ok[:, :, None] & (Tw <= T_WALL_MAX)
    return {"link": link, "arc_ok": arc_ok, "T_wall": Tw, "feasible": feasible,
            "P_peak": P, "d_req": d_req, "loss_no_pulse_db": loss0}


def timescales(fp, V):
    """Characteristic times [s] that justify the quasi-steady square-pulse window model."""
    J_tot, J_eff = fp.currents(V)
    d = float(np.minimum(fp.sheath.thickness(V, J_eff), fp.y[-1]))
    E_w = fp.sheath.wall_field(d, J_eff)
    v_mean = fp.sheath.K * np.sqrt(0.5 * E_w / np.mean(fp.N[fp.y <= d]))
    return {
        "electron_expulsion": 1.0 / float(plasma_angular_frequency(fp.n_peak)) * 10,
        "ion_transit": d / v_mean,
        "flow_residence": L_ELECTRODE / (0.5 * fp.V_inf),
        "electron_return": d / np.sqrt(8 * KB * fp.T_edge / (np.pi * ME)),
    }
