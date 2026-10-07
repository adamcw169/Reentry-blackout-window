"""Figures from analysis/results/ -> figures/."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "analysis" / "results"
FIG = ROOT / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, QUIET, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f6fb", "#c9dbf3", "#86b1e8", "#2a78d6", "#123f7a"])

LABELS = {
    "log10_n_factor": "Peak electron density", "thickness_factor": "Plasma layer thickness",
    "Cp_wall": "Wall pressure", "K_factor": "High-field ion drift", "gamma_e": "Secondary emission γe",
    "cs_factor": "Ion supply scaling", "phi_ionisation": "Ionisation shielding φ",
    "sigma_factor": "Collision frequency",
}


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=QUIET)


def fig_maps():
    m = np.load(RES / "maps.npz")
    V, D = m["V"] / 1e3, m["D"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), dpi=160, sharey=True)
    fig.patch.set_facecolor(BG)
    vmax = 0.15
    for ax, h in zip(axes, ("74", "70", "66")):
        style(ax)
        gain = (m[f"P_feasible_{h}"] - float(m[f"P_none_{h}"])).T
        pc = ax.pcolormesh(V, D, np.clip(gain, 0, None), cmap=SEQ, vmin=0, vmax=vmax, shading="gouraud")
        heat = m[f"Tw_ok_{h}"].T
        ax.contourf(V, D, heat, levels=[0, 0.9], colors="none", hatches=["////"])
        ch = ax.contour(V, D, heat, levels=[0.9], colors=[INK], linewidths=1.6)
        Pavg = (m[f"P_peak_median_{h}"][:, None] * D[None, :]).T / 1e4
        cp = ax.contour(V, D, Pavg, levels=[1, 10, 100], colors=[ORANGE], linewidths=1, linestyles=":")
        ax.clabel(cp, fmt=lambda x: f"{x:g} W/cm²", fontsize=7.5, colors=ORANGE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        base, best = float(m[f"P_none_{h}"]), m[f"P_feasible_{h}"].max()
        ax.set_title(f"{h} km: link {base:.0%} with no pulse, {best:.0%} at best", color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("Pulse voltage (kV)", color=INK)
        ax.set_xticks([0.5, 1, 2, 5, 10, 20])
        ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    axes[0].set_ylabel("Duty cycle (fraction of time on)", color=INK)
    axes[0].text(0.6, 0.55, "hatched: electrode\n>1800 K in >10%\nof cases", fontsize=7.5, color=INK,
                 bbox=dict(facecolor=BG, edgecolor="none", pad=1))
    cb = fig.colorbar(pc, ax=axes, fraction=0.025, pad=0.02, extend="max")
    cb.set_label("Link probability added by pulsing", color=INK)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    cb.ax.tick_params(colors=QUIET)
    fig.suptitle("Blunt body (RAM-C II class), S-band: pulsing adds at most ~10 points; "
                 "beyond ~10 kV more voltage adds nothing", x=0.01, y=1.02, ha="left", color=INK, fontsize=11)
    fig.text(0.01, -0.06, "Dotted orange: time-averaged electrical power per cm² of cathode (median). "
             "QMC: 1024 scrambled Sobol samples over 11 uncertain inputs.", color=QUIET, fontsize=8)
    fig.savefig(FIG / "feasibility_maps.png", bbox_inches="tight")


def fig_trajectory():
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=160)
    fig.patch.set_facecolor(BG)
    style(ax)
    for veh, col, lab in (("blunt", BLUE, "Blunt (RAM-C II, >7 cm layer)"),
                          ("slender", ORANGE, "Slender (Arizona-like, 2.5 cm layer)")):
        r = np.load(RES / f"trajectory_{veh}.npz")
        h = r["h"]
        ax.plot(r["P_none"], h, color=col, lw=1.5, ls=(0, (4, 3)))
        ax.plot(r["P_pulse"], h, color=col, lw=2.2)
        ax.fill_betweenx(h, r["P_none"], r["P_pulse"], color=col, alpha=0.15, lw=0)
        ax.plot([], [], color=col, lw=6, alpha=0.5, label=lab)
    ax.plot([], [], color=QUIET, lw=1.5, ls=(0, (4, 3)), label="No pulses")
    ax.plot([], [], color=QUIET, lw=2.2, label="Pulsed, ≤20 kV, duty 0.1")
    ax.axvline(0.9, color=GRID, lw=1)
    ax.set_xlabel("P(S-band link closes, plasma loss ≤ 10 dB)", color=INK)
    ax.set_ylabel("Altitude (km)", color=INK)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(58, 82)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    ax.set_title("Pulsing adds 10–20 points of link probability: it delays blackout,\n"
                 "it does not remove it", color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "trajectory_link_probability.png")


def fig_sobol():
    s = np.load(RES / "sobol.npz")
    names, S1, ST = list(s["names"]), s["S1"], s["ST"]
    o = np.argsort(ST)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=160)
    fig.patch.set_facecolor(BG)
    style(ax)
    ax.barh(y + 0.18, ST[o], height=0.34, color=BLUE, label="Total effect (incl. interactions)")
    ax.barh(y - 0.18, np.clip(S1[o], 0, None), height=0.34, color=AQUA, label="First-order effect")
    ci = s["ST_ci"]
    ax.errorbar(ST[o], y + 0.18, xerr=[ST[o] - ci[0][o], ci[1][o] - ST[o]], fmt="none", ecolor=INK, lw=0.8, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[names[i]] for i in o], color=INK)
    ax.set_xlabel("Share of variance in sheath margin (10 kV, 70 km)", color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    ax.set_title(f"Plasma uncertainty, not discharge physics, decides the window\n"
                 f"(P(margin ≥ 0) = {float(s['P_positive']):.0%})", color=INK, fontsize=10.5, loc="left")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "sobol_indices.png")


def fig_convergence():
    c = np.load(RES / "convergence.npz")
    N, eq, em = c["N"], c["err_qmc"], c["err_mc"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=160)
    fig.patch.set_facecolor(BG)
    style(ax)
    ax.loglog(N, em, "o-", color=ORANGE, lw=2, ms=6, mec=BG, label="Plain Monte Carlo")
    ax.loglog(N, eq, "o-", color=BLUE, lw=2, ms=6, mec=BG, label="Scrambled Sobol (QMC)")
    ax.loglog(N, em[0] * (N / N[0]) ** -0.5, color=QUIET, lw=1, ls=":", label="N^-1/2")
    ax.loglog(N, em[0] * (N / N[0]) ** -1.0, color=QUIET, lw=1, ls="--", label="N^-1")
    ax.set_xticks(N)
    ax.set_xticklabels([str(int(n)) for n in N])
    ax.minorticks_off()
    ax.set_xlabel("Samples N (RMS over 8 independent seeds; reference: 8192 Sobol samples)", color=INK)
    ax.set_ylabel("RMS error in P(link), 70 km, 7.5 kV", color=INK)
    gain = (em / eq)[-1]
    ax.set_title(f"QMC reaches the same accuracy with fewer runs\n(×{gain:.1f} lower error at N = {N[-1]})",
                 color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    ax.grid(True, which="both", color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "qmc_convergence.png")


FIGS = {"maps": fig_maps, "trajectory": fig_trajectory, "sobol": fig_sobol, "convergence": fig_convergence}

if __name__ == "__main__":
    for k in (sys.argv[1:] or FIGS):
        FIGS[k]()
        print("wrote", k)
