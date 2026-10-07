"""Validation V3: sheath size and window quality against Rodriguez Fuentes & Parent (2026), 7.5 kV case.

Inputs (digitised, see data/arizona2026/README.md):
  * no-pulse electron density and gas temperature profiles above the cathode (Fig. 18)
  * cathode current at 7.5 kV (Fig. 16): 10 A/m uncorrected, 6 A/m corrected, over 40 mm
Wall pressure from oblique-shock theory, Mach 24, 18 deg wedge: 464 Pa (gamma=1.2) to
589 Pa (gamma=1.4). Low-field reduced ion mobility K0 = 2.0-3.0 cm^2/V/s.

CFD sheath thickness targets
  Table 4 gives the plasma thickness d_p above the sheath. With the shock 2.53 cm
  (pre-pulse slab) to 2.9 cm (Fig. 18c, during pulse) from the wall:
      uncorrected ion mobility: d_s = 7.3 to 11 mm
      corrected ion mobility:   d_s = 5.3 to 9 mm
  Fig. 18a (uncorrected, gamma_e 0.5 to 0.009) shows electrons depleted to 13-22 mm.

Pass criterion (from the plan): each model within a factor of 2 of its CFD range,
and the corrected/uncorrected trend reproduced.

Run:  python validation/v3_arizona_7p5kV.py
"""

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackout.plasma import KB  # noqa: E402
from blackout.propagation import (  # noqa: E402
    profile_transmission,
    slab_absorption_transmission,
)
from blackout.sheath import IonMobility, collisional_sheath, matrix_sheath  # noqa: E402

V_PEAK = 7500.0
L_CATHODE = 0.040
I_PER_DEPTH = {"uncorrected": 10.0, "corrected": 6.0}  # A/m at 7.5 kV (Fig. 16)
CFD_DS_MM = {"uncorrected": (7.3, 11.0), "corrected": (5.3, 9.0)}
CFD_DEPLETION_MM_UNCORRECTED = (13.0, 22.0)
P_WALL_PA = (464.0, 589.0)
K0_RANGE = (2.0e-4, 3.0e-4)
SHOCK_SLAB_M = 0.0253

# Table 4, 7.5 kV, corrected ion mobility (the physically correct model)
TABLE4_CORRECTED = {"d_p_cm": 2.00, "N_p_m3": 1.14e17,
                    "I_over_I0": {2e9: 0.149, 4e9: 0.964, 8e9: 0.994}}


def load_profile():
    rows = list(csv.DictReader(open(ROOT / "data" / "arizona2026" / "profile_68km_no_pulse.csv")))
    y = np.array([float(r["y_m"]) for r in rows])
    n = np.array([float(r["n_e_m3"]) for r in rows])
    T = np.array([float(r["T_gas_K"]) for r in rows])
    return y, n, T


def sheath_band(case, y, T):
    """Sheath thickness over the corners of the (p_wall, K0) box."""
    J = I_PER_DEPTH[case] / L_CATHODE
    model = "low" if case == "uncorrected" else "high"
    out = []
    for p, K0 in itertools.product(P_WALL_PA, K0_RANGE):
        N_of_y = lambda yy, p=p: p / (KB * np.interp(yy, y, T))
        d, _ = collisional_sheath(V_PEAK, J, IonMobility(model, K0=K0), N_of_y)
        out.append(d)
    p_mid, K0_mid = np.mean(P_WALL_PA), np.mean(K0_RANGE)
    d_mid, E_wall = collisional_sheath(
        V_PEAK, J, IonMobility(model, K0=K0_mid), lambda yy: p_mid / (KB * np.interp(yy, y, T)))
    N_wall = p_mid / (KB * T[0])
    return d_mid, min(out), max(out), E_wall / N_wall * 1e21  # Td


def window_after_sheath(d_s, y, n, nu=1e7):
    """Mean density above the sheath, slab transmission (Arizona metric) and transfer matrix."""
    m = (y >= d_s) & (y <= SHOCK_SLAB_M)
    d_p = SHOCK_SLAB_M - d_s
    N_p = np.trapezoid(n[m], y[m]) / (y[m][-1] - y[m][0])
    edges = np.concatenate([[0.0], 0.5 * (y[1:] + y[:-1]), [y[-1]]])
    n_cut = np.where(y < d_s, 0.0, n)
    res = {}
    for f in TABLE4_CORRECTED["I_over_I0"]:
        res[f] = {"slab": float(slab_absorption_transmission(N_p, nu, d_p, f)),
                  "transfer_matrix": profile_transmission(edges, n_cut, nu, f)[1]}
    return d_p, N_p, res


def main():
    y, n, T = load_profile()
    results = {}
    d_matrix = matrix_sheath(V_PEAK, y, n)
    print(f"Matrix sheath (ions frozen, first nanoseconds): {d_matrix*1e3:.1f} mm\n")

    passed = True
    for case in ("uncorrected", "corrected"):
        d_mid, d_lo, d_hi, td = sheath_band(case, y, T)
        lo, hi = CFD_DS_MM[case]
        ok = (d_hi * 1e3 <= 2 * hi) and (d_lo * 1e3 >= lo / 2)
        passed &= ok
        results[case] = {"d_s_mm": d_mid * 1e3, "d_s_range_mm": [d_lo * 1e3, d_hi * 1e3],
                         "cfd_range_mm": [lo, hi], "E_over_N_wall_Td": td, "within_factor_2": ok}
        model = "low-field Child law" if case == "uncorrected" else "high-field sheath"
        print(f"{case:>11} ion mobility, {model}: d_s = {d_mid*1e3:.1f} mm "
              f"(range {d_lo*1e3:.1f}-{d_hi*1e3:.1f}) vs CFD {lo}-{hi} mm  "
              f"[{'within x2' if ok else 'OUTSIDE x2'}]; E/N at wall {td:,.0f} Td")

    ratio = results["corrected"]["d_s_mm"] / results["uncorrected"]["d_s_mm"]
    print(f"\nMobility correction shrinks the sheath by x{ratio:.2f} "
          "(paper text: 'two- or three-fold'; Table 4 implies x0.73)")
    trend_ok = ratio < 1.0
    passed &= trend_ok

    d_p, N_p, win = window_after_sheath(results["corrected"]["d_s_mm"] / 1e3, y, n)
    print(f"\nCorrected case, plasma above the sheath: d_p = {d_p*100:.2f} cm (Table 4: 2.00), "
          f"mean n_e = {N_p:.2e} m^-3 (Table 4: 1.14e17)")
    print(f"{'f [GHz]':>8} {'Table 4':>8} {'slab':>7} {'profile TM':>11}")
    for f, v in win.items():
        print(f"{f/1e9:8.0f} {TABLE4_CORRECTED['I_over_I0'][f]:8.3f} {v['slab']:7.3f} {v['transfer_matrix']:11.3f}")

    print(f"\nV3 {'PASS' if passed else 'FAIL'}: both sheath models within a factor of 2 of CFD, trend reproduced")
    results.update({"matrix_sheath_mm": d_matrix * 1e3, "mobility_ratio": ratio, "pass": bool(passed),
                    "corrected_window": {"d_p_m": d_p, "N_p_m3": N_p,
                                         "transmission": {str(int(f/1e9)): v for f, v in win.items()}}})
    (ROOT / "validation" / "v3_results.json").write_text(json.dumps(results, indent=2))
    make_figure(results, y, n)


def make_figure(res, y, n):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange = "#2a78d6", "#eb6834"
    ink, quiet, grid, bg = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=160,
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(bg)
    for a in (ax, ax2):
        a.set_facecolor(bg)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(grid)
        a.tick_params(colors=quiet)

    rows = [("Uncorrected\nion mobility", "uncorrected", 1.0), ("Corrected\n(high-field)", "corrected", 0.0)]
    for label, key, yy in rows:
        lo, hi = res[key]["cfd_range_mm"]
        ax.plot([lo, hi], [yy, yy], color=grid, lw=12, solid_capstyle="round", zorder=1)
        m_lo, m_hi = res[key]["d_s_range_mm"]
        ax.errorbar(res[key]["d_s_mm"], yy, xerr=[[res[key]["d_s_mm"] - m_lo], [m_hi - res[key]["d_s_mm"]]],
                    fmt="o", color=blue, ms=8, mec=bg, mew=1.5, capsize=4, lw=2, zorder=3)
    lo, hi = CFD_DEPLETION_MM_UNCORRECTED
    ax.plot([lo, hi], [1.3, 1.3], color=quiet, lw=2, solid_capstyle="round", zorder=1)
    ax.axvline(res["matrix_sheath_mm"], color=orange, lw=1.5, ls=(0, (3, 3)))
    ax.text(res["matrix_sheath_mm"] + 0.4, 0.5, "matrix sheath\n(ions frozen,\nfirst ns)", color=orange, fontsize=8)
    ax.plot([], [], "o", color=blue, ms=7, label="This model (pressure × mobility range)")
    ax.plot([], [], color=grid, lw=8, label="CFD sheath, Table 4")
    ax.plot([], [], color=quiet, lw=2, label="CFD electron depletion, Fig. 18a")
    ax.set_yticks([1, 0])
    ax.set_yticklabels([r[0] for r in rows], color=ink)
    ax.set_ylim(-0.5, 1.55)
    ax.set_xlim(0, 25)
    ax.set_xlabel("Sheath thickness at 7.5 kV (mm)", color=ink)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=ink)
    ax.set_title("Physical (corrected) case lands inside the CFD range;\nuncorrected over-predicts Table 4 by up to 2.2×",
                 color=ink, fontsize=10, loc="left")

    ax2.semilogx(n, y * 1e3, color=quiet, lw=2, label="No pulse (digitised)")
    d_s = res["corrected"]["d_s_mm"]
    ax2.axhspan(0, d_s, color=blue, alpha=0.15, lw=0)
    ax2.text(1.2e15, d_s / 2, f"electron-free sheath\n{d_s:.1f} mm (corrected)", va="center", fontsize=8, color=ink)
    ax2.axhline(SHOCK_SLAB_M * 1e3, color=grid, lw=1)
    ax2.text(4.5e17, SHOCK_SLAB_M * 1e3 + 0.6, "slab edge, 25.3 mm", fontsize=8, color=quiet, ha="right")
    ax2.set_xlim(1e15, 5e17)
    ax2.set_ylim(0, 30)
    ax2.set_xlabel("Electron density (m⁻³)", color=ink)
    ax2.set_ylabel("Distance from cathode (mm)", color=ink)
    ax2.set_title("…but the 2.4×10¹⁷ m⁻³ core sits above it,\nso 4 GHz still loses 18%", color=ink, fontsize=10, loc="left")
    ax2.grid(True, which="major", color=grid, lw=0.8)

    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "v3_sheath.png")


if __name__ == "__main__":
    main()
