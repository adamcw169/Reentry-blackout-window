"""Cold, collisional, unmagnetised plasma properties.

All quantities in SI units. Sign convention: fields vary as exp(+j*omega*t),
so a lossy medium has a relative permittivity with a negative imaginary part.
"""

import numpy as np
from scipy import constants as c

EPS0 = c.epsilon_0
QE = c.elementary_charge
ME = c.electron_mass
C0 = c.speed_of_light
KB = c.Boltzmann


def plasma_angular_frequency(n_e):
    """Electron plasma angular frequency omega_p [rad/s] for density n_e [m^-3]."""
    return np.sqrt(np.asarray(n_e, dtype=float) * QE**2 / (EPS0 * ME))


def plasma_frequency(n_e):
    """Electron plasma frequency f_p [Hz]; approximately 8.98*sqrt(n_e)."""
    return plasma_angular_frequency(n_e) / (2 * np.pi)


def critical_density(f):
    """Electron density [m^-3] at which f_p equals the signal frequency f [Hz]."""
    omega = 2 * np.pi * np.asarray(f, dtype=float)
    return EPS0 * ME * omega**2 / QE**2


def debye_length(n_e, T_e):
    """Electron Debye length [m] for density n_e [m^-3] and temperature T_e [K]."""
    return np.sqrt(EPS0 * KB * T_e / (np.asarray(n_e, dtype=float) * QE**2))


def relative_permittivity(n_e, nu_en, f):
    """Complex relative permittivity of a cold collisional plasma.

    eps_r = 1 - omega_p^2 / (omega * (omega - j*nu))

    n_e   : electron density [m^-3]
    nu_en : electron-neutral collision frequency [1/s]
    f     : signal frequency [Hz]
    """
    omega = 2 * np.pi * np.asarray(f, dtype=float)
    wp2 = plasma_angular_frequency(n_e) ** 2
    return 1.0 - wp2 / (omega * (omega - 1j * np.asarray(nu_en, dtype=float)))
