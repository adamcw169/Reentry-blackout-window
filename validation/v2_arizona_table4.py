"""Validation V2: reproduce the no-field baseline of Rodriguez Fuentes & Parent (2026), Table 4.

Baseline plasma above the cathode (no applied voltage):
    N_p = 2.31e17 m^-3, d_p = 2.53 cm  (f_p = 4.33 GHz)
Transmitted power fraction I/I0 at six signal frequencies is tabulated.

The paper does not list its electron-neutral collision frequency, so we fit a
single nu_en to all six points and then check (a) the fit residuals and
(b) whether the fitted nu_en is physically plausible at 68 km.

Run:  python validation/v2_arizona_table4.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackout.plasma import plasma_frequency  # noqa: E402
from blackout.propagation import (  # noqa: E402
    _eps_physics,
    slab_absorption_transmission,
    slab_absorption_transmission_as_printed,
    transfer_matrix,
)

# Arizona Table 4, applied voltage = 0 kV (identical for both mobility models)
F_GHZ = np.array([0.10, 0.50, 1.00, 2.00, 4.00, 8.00])
I_OVER_I0 = np.array([0.010, 0.011, 0.012, 0.018, 0.185, 0.995])
N_P = 2.31e17  # m^-3
D_P = 0.0253   # m


def fit_collision_frequency():
    f = F_GHZ * 1e9

    def cost(log10_nu):
        model = slab_absorption_transmission(N_P, 10**log10_nu, D_P, f)
        return np.sum((np.log(model) - np.log(I_OVER_I0)) ** 2)

    res = minimize_scalar(cost, bounds=(5, 11), method="bounded")
    return 10**res.x


def main():
    f = F_GHZ * 1e9
    nu = fit_collision_frequency()
    absorb = slab_absorption_transmission(N_P, nu, D_P, f)
    printed = slab_absorption_transmission_as_printed(N_P, nu, D_P, f)
    tm = np.array([transfer_matrix([_eps_physics(N_P, nu, fi)], [D_P], fi)[1] for fi in f])

    rel_err = (absorb - I_OVER_I0) / I_OVER_I0
    passed = bool(np.all(np.abs(rel_err) < 0.05))

    print(f"Baseline: n_e = {N_P:.3g} m^-3, f_p = {plasma_frequency(N_P)/1e9:.2f} GHz, d = {D_P*100:.2f} cm")
    print(f"Fitted collision frequency nu_en = {nu:.3g} 1/s\n")
    print(f"{'f [GHz]':>8} {'Table 4':>9} {'exp(-2ad)':>10} {'err %':>7} {'Eq18 printed':>13} {'transfer-mtx':>13}")
    for row in zip(F_GHZ, I_OVER_I0, absorb, rel_err * 100, printed, tm):
        print(f"{row[0]:8.2f} {row[1]:9.3f} {row[2]:10.3f} {row[3]:7.1f} {row[4]:13.3f} {row[5]:13.3f}")
    print(f"\nV2 {'PASS' if passed else 'FAIL'}: all six points within 5% (max {np.max(np.abs(rel_err))*100:.1f}%)")

    # The fit is flat for small nu: report the range Table 4 actually supports.
    grid = np.logspace(4, 10, 601)
    ok = [g for g in grid
          if np.max(np.abs(slab_absorption_transmission(N_P, g, D_P, f) - I_OVER_I0) / I_OVER_I0) < 0.05]
    nu_max = max(ok)
    # Hard-sphere estimate behind the shock at 68 km (q = 2.5 kPa, V = 7.45 km/s)
    rho_inf = 2 * 2500 / 7450**2
    n_inf = rho_inf / (28.96e-3 / 6.02214e23)
    v_e = lambda T: np.sqrt(8 * 1.380649e-23 * T / (np.pi * 9.10938e-31))
    nu_phys = (6 * n_inf * 1e-19 * v_e(3000), 10 * n_inf * 1e-19 * v_e(8000))
    print(f"Table 4 supports nu_en < {nu_max:.1e} 1/s (all points within 5%).")
    print(f"Hard-sphere estimate behind the shock: {nu_phys[0]:.1e} to {nu_phys[1]:.1e} 1/s "
          "-> open question; S/C-band results are insensitive to nu in this range.")

    out = {
        "nu_en_fit_per_s": nu,
        "nu_en_max_supported_per_s": nu_max,
        "nu_en_hard_sphere_estimate_per_s": list(nu_phys),
        "max_rel_error": float(np.max(np.abs(rel_err))),
        "pass": passed,
        "rows": [
            {"f_GHz": float(a), "table4": float(b), "absorption_only": float(c),
             "eq18_as_printed": float(d), "transfer_matrix": float(e)}
            for a, b, c, d, e in zip(F_GHZ, I_OVER_I0, absorb, printed, tm)
        ],
    }
    (ROOT / "validation" / "v2_results.json").write_text(json.dumps(out, indent=2))
    make_figure(nu)


def make_figure(nu):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink, quiet, grid = "#0b0b0b", "#52514e", "#e4e3df"

    fs = np.logspace(np.log10(0.08e9), np.log10(12e9), 400)
    absorb = slab_absorption_transmission(N_P, nu, D_P, fs)
    tm = np.array([transfer_matrix([_eps_physics(N_P, nu, fi)], [D_P], fi)[1] for fi in fs])

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(fs / 1e9, absorb, color=blue, lw=2, label="Absorption only, exp(-2αd)")
    ax.plot(fs / 1e9, tm, color=aqua, lw=2, label="Transfer matrix (with reflection)")
    ax.plot(F_GHZ, I_OVER_I0, "o", ms=8, color=orange, mec="#fcfcfb", mew=2,
            label="Rodríguez Fuentes & Parent 2026, Table 4")
    ax.axvline(plasma_frequency(N_P) / 1e9, color=quiet, lw=1, ls=(0, (3, 3)))
    ax.text(plasma_frequency(N_P) / 1e9 * 0.95, 0.62, f"plasma frequency\n{plasma_frequency(N_P)/1e9:.2f} GHz",
            color=quiet, fontsize=9, ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Signal frequency (GHz)", color=ink)
    ax.set_ylabel("Transmitted power fraction, I/I₀", color=ink)
    ax.set_title("V2: absorption-only model matches all six Table 4 points (ν < 7×10⁷ s⁻¹);\n"
                 "including reflection changes the answer near the plasma frequency",
                 color=ink, fontsize=10.5, loc="left")
    ax.grid(True, which="major", color=grid, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=quiet)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=ink)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "v2_attenuation.png")


if __name__ == "__main__":
    main()
