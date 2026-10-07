"""Plane-wave propagation through re-entry plasma layers at normal incidence.

Two models:

* ``slab_absorption_transmission`` - power transmitted through a uniform slab,
  counting only in-slab decay exp(-2*alpha*d) and ignoring reflections. This is
  the metric used by Rodriguez Fuentes & Parent (2026), their Eq. (18).
* ``transfer_matrix`` - exact 1D multilayer solution (characteristic matrix
  method) that includes reflection at every interface. Use this for real,
  non-uniform profiles with the sheath cut out.

Internally the transfer matrix uses the physics convention exp(-i*omega*t),
in which a lossy medium has Im(n) > 0.
"""

import numpy as np

from .plasma import C0, plasma_angular_frequency


def _eps_physics(n_e, nu_en, f):
    """Relative permittivity in the exp(-i*omega*t) convention (Im >= 0 for loss)."""
    omega = 2 * np.pi * np.asarray(f, dtype=float)
    wp2 = plasma_angular_frequency(n_e) ** 2
    return 1.0 - wp2 / (omega * (omega + 1j * np.asarray(nu_en, dtype=float)))


def _refractive_index(eps):
    """Principal square root with Im(n) >= 0 (decaying wave)."""
    n = np.sqrt(np.asarray(eps, dtype=complex))
    return np.where(n.imag < 0, -n, n)


def attenuation_constant(n_e, nu_en, f):
    """Field attenuation constant alpha [Np/m] of a uniform cold plasma."""
    omega = 2 * np.pi * np.asarray(f, dtype=float)
    return (omega / C0) * _refractive_index(_eps_physics(n_e, nu_en, f)).imag


def slab_absorption_transmission(n_e, nu_en, d, f):
    """Transmitted power fraction I/I0 = exp(-2*alpha*d), reflections ignored."""
    return np.exp(-2.0 * attenuation_constant(n_e, nu_en, f) * d)


def slab_absorption_transmission_as_printed(n_e, nu_en, d, f):
    """Eq. (18) of Rodriguez Fuentes & Parent (2026) exactly as typeset.

    The printed real-part term uses omega_p^4 where the standard cold-plasma
    result has omega_p^2 * omega_s^2. Kept only to show which form reproduces
    their Table 4.
    """
    ws = 2 * np.pi * np.asarray(f, dtype=float)
    wp = plasma_angular_frequency(n_e)
    nu = np.asarray(nu_en, dtype=float)
    den = ws**4 + nu**2 * ws**2
    A = 1.0 - wp**4 / den
    B = nu * wp**2 * ws / den
    inner = np.sqrt(np.sqrt(A**2 + B**2) - A) / np.sqrt(2.0)
    return np.exp(-2.0 * d * ws / C0 * inner)


def transfer_matrix(eps_layers, d_layers, f, eps_in=1.0, eps_out=1.0):
    """Power reflectance and transmittance of a stack of uniform layers.

    eps_layers : relative permittivities, physics convention (Im >= 0 for loss)
    d_layers   : layer thicknesses [m]
    f          : frequency [Hz] (scalar)
    eps_in, eps_out : media either side (default free space)

    Returns (R, T, A) with A = 1 - R - T the absorbed fraction.
    """
    k0 = 2 * np.pi * f / C0
    M = np.eye(2, dtype=complex)
    for eps, d in zip(eps_layers, d_layers):
        n = _refractive_index(eps)
        delta = k0 * n * d
        layer = np.array(
            [[np.cos(delta), -1j * np.sin(delta) / n],
             [-1j * n * np.sin(delta), np.cos(delta)]],
            dtype=complex,
        )
        M = M @ layer
    y0 = _refractive_index(eps_in)
    ys = _refractive_index(eps_out)
    denom = y0 * M[0, 0] + y0 * ys * M[0, 1] + M[1, 0] + ys * M[1, 1]
    r = (y0 * M[0, 0] + y0 * ys * M[0, 1] - M[1, 0] - ys * M[1, 1]) / denom
    t = 2 * y0 / denom
    R = float(abs(r) ** 2)
    T = float((ys.real / y0.real) * abs(t) ** 2)
    return R, T, 1.0 - R - T


def transfer_matrix_batch(eps, d, f):
    """Vectorised transfer matrix for B stacks sharing layer thicknesses.

    eps : (B, L) complex permittivities (physics convention), free space either side
    d   : (L,) thicknesses [m]
    Returns transmittance T, shape (B,).
    """
    k0 = 2 * np.pi * f / C0
    n = _refractive_index(eps)
    delta = k0 * n * d[None, :]
    c, s = np.cos(delta), np.sin(delta)
    B = eps.shape[0]
    m00 = np.ones(B, complex); m01 = np.zeros(B, complex)
    m10 = np.zeros(B, complex); m11 = np.ones(B, complex)
    for j in range(eps.shape[1]):
        a00, a01 = c[:, j], -1j * s[:, j] / n[:, j]
        a10, a11 = -1j * n[:, j] * s[:, j], c[:, j]
        m00, m01, m10, m11 = (m00 * a00 + m01 * a10, m00 * a01 + m01 * a11,
                              m10 * a00 + m11 * a10, m10 * a01 + m11 * a11)
    t = 2.0 / (m00 + m01 + m10 + m11)
    return np.abs(t) ** 2


def profile_transmission(y, n_e, nu_en, f):
    """Transfer-matrix (R, T, A) for a sampled profile n_e(y), nu_en(y).

    y must be increasing cell-edge positions [m] (length N+1); n_e and nu_en are
    cell values (length N), or scalars for nu_en.
    """
    y = np.asarray(y, dtype=float)
    d = np.diff(y)
    n_e = np.asarray(n_e, dtype=float)
    nu = np.broadcast_to(np.asarray(nu_en, dtype=float), n_e.shape)
    eps = _eps_physics(n_e, nu, f)
    return transfer_matrix(eps, d, f)
