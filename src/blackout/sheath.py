"""Cathode sheath models for a negatively pulsed electrode under a re-entry plasma.

Planar (1D, wall-normal) geometry. y is distance from the cathode [m].

Three estimates of the electron-free sheath thickness d_s:

1. ``matrix_sheath``: the instant response. Electrons are expelled on the
   electron timescale and ions have not yet moved, so the sheath is set by the
   frozen ion profile n(y):  V = (e/eps0) * int_0^s y n(y) dy.

2. ``collisional_sheath`` with a constant-mobility ion model: steady, ions drift
   at v = mu*E. With uniform gas density this reduces to the collisional
   Child (Mott-Gurney) law  J = (9/8) eps0 mu V^2 / d^3.

3. ``collisional_sheath`` with the high-field ion model: at reduced fields of
   thousands of Td, ion drift saturates as v = K*sqrt(E/N) (Sinnott et al. 1968,
   as used by Rodriguez Fuentes & Parent 2026). With uniform N this gives
   d = (5V/3)^(3/5) * (2 eps0 c / (3J))^(2/5),  c = K/sqrt(N).

Models 2 and 3 take the ion current density J as an input. In the Arizona CFD
the cathode extracts several times more ions than the flow supplies, so J is a
discharge property (ionisation inside the sheath), not a plasma-supply property.
Closing J is the next modelling step; here it is supplied.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .plasma import EPS0, KB, QE

# Reduced ion mobility at STP, NO+ in air (m^2/V/s); Loschmidt number (m^-3)
K0_NO_PLUS = 2.5e-4
N_LOSCHMIDT = 2.6868e25

# High-field drift constants K [m^-1 V^-1/2 s^-1 ... in SI so that v = K*sqrt(E/N)],
# from Rodriguez Fuentes & Parent (2026), Table 2 (after Sinnott et al. 1968)
K_HIGH_FIELD = {"NO+": 4.47e12, "O2+": 3.61e12, "N2+": 2.03e12}


# ---------------------------------------------------------------------------
# Ion transport
# ---------------------------------------------------------------------------

class IonMobility:
    """Ion drift velocity v(E, N).

    model = "low"  : v = mu0*E with mu0 = K0*N_L/N (polarisation limit)
    model = "high" : v = min(mu0*E, K*sqrt(E/N))   (high-field saturation)
    """

    def __init__(self, model="high", K0=K0_NO_PLUS, ion="NO+"):
        if model not in ("low", "high"):
            raise ValueError("model must be 'low' or 'high'")
        self.model = model
        self.K0 = K0
        self.K = K_HIGH_FIELD[ion]

    def low_field_mobility(self, N):
        return self.K0 * N_LOSCHMIDT / N

    def drift_velocity(self, E, N):
        E = np.abs(E)
        v_low = self.low_field_mobility(N) * E
        if self.model == "low":
            return v_low
        return np.minimum(v_low, self.K * np.sqrt(E / N))


def gas_number_density(p, T):
    """N = p / (k T) [m^-3]."""
    return p / (KB * np.asarray(T, dtype=float))


# ---------------------------------------------------------------------------
# Sheath models
# ---------------------------------------------------------------------------

def matrix_sheath(V, y, n_ion):
    """Ion-matrix sheath thickness [m] for a frozen ion profile.

    V      : sheath voltage magnitude [V]
    y      : grid of wall distances [m], increasing from 0
    n_ion  : ion density on that grid [m^-3] (taken equal to the pre-pulse n_e)
    """
    y = np.asarray(y, dtype=float)
    n = np.asarray(n_ion, dtype=float)
    integrand = y * n
    phi = (QE / EPS0) * np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(y))]
    )
    if V > phi[-1]:
        return np.inf
    return float(np.interp(V, phi, y))


def matrix_sheath_uniform(V, n0):
    """Analytical matrix sheath for uniform density: s = sqrt(2 eps0 V / (e n0))."""
    return np.sqrt(2 * EPS0 * V / (QE * n0))


def child_law_uniform(V, J, mu):
    """Collisional Child (Mott-Gurney) law, uniform mobility: d = (9 eps0 mu V^2 / 8J)^(1/3)."""
    return (9 * EPS0 * mu * V**2 / (8 * J)) ** (1.0 / 3.0)


def high_field_child_uniform(V, J, N, K=K_HIGH_FIELD["NO+"]):
    """High-field collisional sheath, uniform gas density, v = K sqrt(E/N)."""
    c = K / np.sqrt(N)
    return (5 * V / 3) ** 0.6 * (2 * EPS0 * c / (3 * J)) ** 0.4


class HighFieldSheathProfile:
    """Fast steady sheath for saturated ion drift v = K sqrt(E/N) over a profile N(y).

    With J constant across the sheath, Poisson's equation integrates exactly:
        E(s)^(3/2) = (3 J / (2 eps0 K)) * G(s),   G(s) = int_edge^s sqrt(N) ds'
    so the sheath voltage is V(d) = (3 J / (2 eps0 K))^(2/3) * W(d), where W depends
    only on the gas-density profile. W is tabulated once; d(V, J) is an interpolation.
    The low-field limit near the sheath edge is neglected (it carries little voltage).
    """

    def __init__(self, y, N, K=K_HIGH_FIELD["NO+"]):
        y = np.asarray(y, dtype=float)
        sq = np.sqrt(np.asarray(N, dtype=float))
        S = np.concatenate([[0.0], np.cumsum(0.5 * (sq[1:] + sq[:-1]) * np.diff(y))])
        diff = np.clip(S[:, None] - S[None, :], 0.0, None) ** (2.0 / 3.0)   # (S(d_i) - S(y_j))
        diff = np.tril(diff)
        dy = np.diff(y)
        W = np.concatenate([[0.0], [np.sum(0.5 * (diff[i, 1:i + 1] + diff[i, :i]) * dy[:i])
                                    for i in range(1, len(y))]])
        self.y, self.S, self.W, self.K = y, S, W, K
        self._sqrtN = sq

    def _coef(self, J):
        return (3.0 * np.asarray(J, dtype=float) / (2.0 * EPS0 * self.K)) ** (2.0 / 3.0)

    def voltage(self, d, J):
        return self._coef(J) * np.interp(d, self.y, self.W)

    def thickness(self, V, J):
        """Sheath thickness [m]; inf if beyond the tabulated profile."""
        w = np.asarray(V, dtype=float) / self._coef(J)
        return np.where(w > self.W[-1], np.inf, np.interp(w, self.W, self.y))

    def wall_field(self, d, J):
        """Electric field at the cathode [V/m] and the gas density there."""
        G = np.interp(d, self.y, self.S)
        return (3.0 * J * G / (2.0 * EPS0 * self.K)) ** (2.0 / 3.0)


def _sheath_voltage(d, J, mobility, N_of_y, E_seed=1.0):
    """Voltage across a steady collisional sheath of thickness d carrying J.

    Integrates from the sheath edge (s = 0, y = d) to the wall (s = d, y = 0):
        dE/ds = J / (eps0 * v(E, N(d - s))),   dphi/ds = E.
    """

    def rhs(s, z):
        E = max(z[0], E_seed)
        N = N_of_y(d - s)
        return [J / (EPS0 * mobility.drift_velocity(E, N)), E]

    sol = solve_ivp(rhs, (0.0, d), [E_seed, 0.0], method="LSODA", rtol=1e-7, atol=[1e-3, 1e-6])
    return sol.y[1, -1], sol.y[0, -1]


def collisional_sheath(V, J, mobility, N_of_y, d_bounds=(1e-6, 0.1)):
    """Steady collisional sheath thickness [m] for voltage V [V] and ion current density J [A/m^2].

    N_of_y : callable giving gas number density [m^-3] at wall distance y [m]
    Returns (d, E_wall) with E_wall the field at the cathode [V/m].
    """
    f = lambda d: _sheath_voltage(d, J, mobility, N_of_y)[0] - V
    lo, hi = d_bounds
    if f(hi) < 0:
        return np.inf, np.nan
    d = brentq(f, lo, hi, xtol=1e-7)
    return d, _sheath_voltage(d, J, mobility, N_of_y)[1]
