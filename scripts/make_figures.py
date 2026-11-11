"""Generate every README figure and the animation into docs/assets/.

    python scripts/make_figures.py            # everything (about a minute)
    python scripts/make_figures.py hero gif   # selected figures only

Sources
  hero, trajectory, sobol, convergence   cached results in analysis/results/ (python analysis/run_analysis.py)
  validation                             validation/v2_results.json, v3_results.json and the model
  tradeoff, gif                          the model, run here at nominal inputs

Each static figure is written as PNG (200 dpi) and SVG in a light and a dark variant
(<name>-light.*, <name>-dark.*) so the README can serve the right one per GitHub theme.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackout.plasma import critical_density, plasma_angular_frequency  # noqa: E402
from blackout.propagation import slab_absorption_transmission, transfer_matrix_batch  # noqa: E402
from blackout.window import F_LINK, LOSS_MAX_DB, FlightPlasma, nominal  # noqa: E402

RES = ROOT / "analysis" / "results"
VAL = ROOT / "validation"
OUT = ROOT / "docs" / "assets"
STYLE = Path(__file__).with_name("figures.mplstyle")

# Okabe-Ito colour-blind-safe palette; blue is lightened on the dark background.
PALETTE = {
    "light": dict(blue="#0072B2", orange="#D55E00", green="#009E73", yellow="#E69F00", purple="#CC79A7"),
    "dark": dict(blue="#56B4E9", orange="#D55E00", green="#009E73", yellow="#E69F00", purple="#CC79A7"),
}
DARK_RC = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117", "savefig.facecolor": "#0d1117",
    "text.color": "#e6edf3", "axes.labelcolor": "#e6edf3", "axes.edgecolor": "#6e7681",
    "xtick.color": "#9198a1", "ytick.color": "#9198a1", "grid.color": "#21262d", "hatch.color": "#e6edf3",
}
THEMES = {"light": {}, "dark": DARK_RC}

LABELS = {
    "log10_n_factor": "Peak electron density", "thickness_factor": "Plasma layer thickness",
    "Cp_wall": "Wall pressure", "K_factor": "High-field ion drift", "gamma_e": r"Secondary emission $\gamma_e$",
    "cs_factor": "Ion supply scaling", "phi_ionisation": r"Ionisation shielding $\varphi$",
    "sigma_factor": "Collision frequency",
}


def ink():
    return plt.rcParams["text.color"]


def quiet():
    return plt.rcParams["xtick.color"]


def save(fig, name, theme, svg=True):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme}.png")
    if svg:
        fig.savefig(OUT / f"{name}-{theme}.svg")
    plt.close(fig)


def loss_db(fp, d_cut=0.0, f=F_LINK):
    """One-way plasma loss [dB] with electrons removed below d_cut (same layer model as required_depth)."""
    mids = 0.5 * (fp.y[1:] + fp.y[:-1])
    n = np.where(mids < d_cut, 0.0, np.interp(mids, fp.y, fp.n_e))
    nu = np.interp(mids, fp.y, fp.nu)
    omega = 2 * np.pi * f
    eps = 1.0 - plasma_angular_frequency(n) ** 2 / (omega * (omega + 1j * nu))
    Tr = transfer_matrix_batch(eps[None, :], np.diff(fp.y), f)
    return float(-10 * np.log10(np.maximum(Tr, 1e-30))[0])


# ---------------------------------------------------------------------------
# Hero: feasibility maps
# ---------------------------------------------------------------------------

def fig_hero(theme, C):
    m = np.load(RES / "maps.npz")
    V, D = m["V"] / 1e3, m["D"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True, constrained_layout=True)
    for ax, h in zip(axes, ("74", "70", "66")):
        gain = (m[f"P_feasible_{h}"] - float(m[f"P_none_{h}"])).T
        pc = ax.pcolormesh(V, D, np.clip(gain, 0, None), cmap="cividis", vmin=0, vmax=0.15,
                           shading="gouraud", rasterized=True)
        heat = m[f"Tw_ok_{h}"].T
        with plt.rc_context({"hatch.color": "#ffffff"}):
            ax.contourf(V, D, heat, levels=[0, 0.9], colors="none", hatches=["////"])
        ax.contour(V, D, heat, levels=[0.9], colors=["#ffffff"], linewidths=1.4)
        Pavg = (m[f"P_peak_median_{h}"][:, None] * D[None, :]).T / 1e4
        cp = ax.contour(V, D, Pavg, levels=[1, 10, 100], colors=[C["yellow"]], linewidths=1.1, linestyles=":")
        ax.clabel(cp, fmt=lambda x: f"{x:g} W cm$^{{-2}}$", fontsize=7.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(False)
        base, best = float(m[f"P_none_{h}"]), float(m[f"P_feasible_{h}"].max())
        ax.set_title(f"{h} km   link {base:.0%} unpulsed, {best:.0%} best pulsed", fontsize=9.5)
        ax.set_xlabel("Pulse voltage (kV)")
        ax.set_xticks([0.5, 1, 2, 5, 10, 20])
        ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
        ax.minorticks_off()
    axes[0].set_ylabel("Duty cycle (–)")
    for ax in axes:
        ax.set_yticks([0.01, 0.1, 1])
        ax.set_yticklabels(["0.01", "0.1", "1"])
    cb = fig.colorbar(pc, ax=axes, fraction=0.025, pad=0.015, extend="max")
    cb.set_label("Link probability added by pulsing")
    cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    cb.outline.set_visible(False)
    save(fig, "hero_feasibility_maps", theme, svg=False)


# ---------------------------------------------------------------------------
# Trajectory and blackout window
# ---------------------------------------------------------------------------

def fig_trajectory(theme, C):
    rows = list(csv.DictReader(open(ROOT / "data" / "ramc2_trajectory.csv")))
    t = np.array([float(r["time_s"]) for r in rows])
    h = np.array([float(r["altitude_km"]) for r in rows])
    v = np.array([float(r["velocity_km_s"]) for r in rows])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True,
                               gridspec_kw=dict(width_ratios=[1, 1.25]))
    a.axhspan(58, 84, color=C["blue"], alpha=0.15, lw=0)
    a.text(t.min() + 1, 81.5, "analysed band, 58–84 km", va="top", fontsize=8.5, color=quiet())
    a.plot(t, h, color=C["blue"])
    a.set_xlabel("Flight time (s)")
    a.set_ylabel("Altitude (km)", color=C["blue"])
    a2 = a.twinx()
    a2.plot(t, v, color=C["orange"], ls=(0, (4, 2)))
    a2.set_ylabel("Velocity (km s$^{-1}$)", color=C["orange"])
    a2.spines["right"].set_visible(True)
    a2.grid(False)
    a2.set_ylim(0, 8.5)
    a.set_title("RAM-C II trajectory (NASA TN D-6062)")

    for veh, col, lab in (("blunt", C["blue"], "Blunt, RAM-C II class (>7 cm layer)"),
                          ("slender", C["orange"], "Slender, thin layer (2.5 cm)")):
        r = np.load(RES / f"trajectory_{veh}.npz")
        b.plot(r["P_none"], r["h"], color=col, lw=1.4, ls=(0, (4, 3)))
        b.plot(r["P_pulse"], r["h"], color=col, lw=2.2)
        b.fill_betweenx(r["h"], r["P_none"], r["P_pulse"], color=col, alpha=0.18, lw=0)
        b.plot([], [], color=col, lw=6, alpha=0.6, label=lab)
    b.plot([], [], color=quiet(), lw=1.4, ls=(0, (4, 3)), label="No pulses")
    b.plot([], [], color=quiet(), lw=2.2, label="Pulsed, ≤ 20 kV, duty cycle 0.1")
    b.axvline(0.5, color=quiet(), lw=0.8, ls=":")
    b.set_xlabel("Probability that the S-band link closes (plasma loss ≤ 10 dB)")
    b.set_ylabel("Altitude (km)")
    b.set_xlim(0, 1.02)
    b.set_ylim(58, 82)
    b.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    b.legend(loc="upper left")
    b.set_title("Blackout window with and without pulses")
    save(fig, "trajectory_blackout_window", theme)


# ---------------------------------------------------------------------------
# Validation against CFD
# ---------------------------------------------------------------------------

def fig_validation(theme, C):
    v2 = json.loads((VAL / "v2_results.json").read_text())
    v3 = json.loads((VAL / "v3_results.json").read_text())
    f = np.array([r["f_GHz"] for r in v2["rows"]])
    ref = np.array([r["table4"] for r in v2["rows"]])
    mod = np.array([r["absorption_only"] for r in v2["rows"]])
    err = 100 * (mod - ref) / ref

    fig = plt.figure(figsize=(10.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1], width_ratios=[1.35, 1])
    a, e, s = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

    ff = np.geomspace(0.08, 10, 300)
    curve = slab_absorption_transmission(2.31e17, v2["nu_en_fit_per_s"], 0.0253, ff * 1e9)
    a.plot(ff, curve, color=C["blue"], label="Reduced-order model (absorption, fitted $\\nu_{en}$)")
    a.plot(f, ref, "o", color=C["orange"], ms=7, mec=plt.rcParams["axes.facecolor"], mew=1.2,
           label="Coupled CFD, Table 4 of [1]")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylabel("Transmitted power fraction $I/I_0$ (–)")
    a.set_title("V2: no-field attenuation, 68 km")
    a.legend(loc="upper left")
    a.tick_params(labelbottom=False)

    e.bar(f, err, width=0.22 * f, color=C["blue"])
    e.axhline(0, color=quiet(), lw=0.8)
    for x, y in zip(f, err):
        e.annotate(f"{y:+.1f}", (x, y), xytext=(0, 3 if y >= 0 else -3), textcoords="offset points",
                   ha="center", va="bottom" if y >= 0 else "top", fontsize=8)
    e.set_xscale("log")
    e.set_xlim(a.get_xlim())
    e.set_ylim(-6, 6)
    e.set_xticks([0.1, 0.5, 1, 2, 4, 8])
    e.set_xticklabels(["0.1", "0.5", "1", "2", "4", "8"])
    e.minorticks_off()
    e.set_xlabel("Signal frequency (GHz)")
    e.set_ylabel("Error (%)")

    for i, (key, lab) in enumerate((("uncorrected", "Low-field\nion mobility"),
                                    ("corrected", "High-field\nion mobility"))):
        lo, hi = v3[key]["cfd_range_mm"]
        s.bar(i - 0.18, hi - lo, bottom=lo, width=0.3, color=C["orange"], alpha=0.85,
              label="Coupled CFD range [1]" if i == 0 else None)
        mlo, mhi = v3[key]["d_s_range_mm"]
        s.bar(i + 0.18, mhi - mlo, bottom=mlo, width=0.3, color=C["blue"],
              label="Reduced-order model range" if i == 0 else None)
        s.plot([i + 0.03, i + 0.33], [v3[key]["d_s_mm"]] * 2, color=ink(), lw=1.2)
    s.set_xticks([0, 1])
    s.set_xticklabels(["Low-field\nion mobility", "High-field\nion mobility"])
    s.set_ylabel("Sheath thickness at 7.5 kV (mm)")
    s.set_ylim(0, 18)
    s.grid(axis="x", visible=False)
    s.legend(loc="upper right")
    s.set_title("V3: sheath thickness, 68 km")
    save(fig, "validation_cfd", theme)


# ---------------------------------------------------------------------------
# Sobol indices
# ---------------------------------------------------------------------------

def fig_sobol(theme, C):
    s = np.load(RES / "sobol.npz")
    names, S1, ST = list(s["names"]), s["S1"], s["ST"]
    o = np.argsort(ST)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    for vals, ci, off, col, lab in ((ST, s["ST_ci"], 0.19, C["blue"], "Total-order $S_T$"),
                                    (S1, s["S1_ci"], -0.19, C["yellow"], "First-order $S_1$")):
        x = np.clip(vals[o], 0, None)
        ax.barh(y + off, x, height=0.34, color=col, label=lab)
        ax.errorbar(x, y + off, xerr=[np.clip(x - ci[0][o], 0, None), np.clip(ci[1][o] - x, 0, None)],
                    fmt="none", ecolor=ink(), lw=0.9, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[names[i]] for i in o])
    ax.set_xlabel("Share of variance in sheath margin at 70 km, 10 kV (–)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "sobol_indices", theme)


# ---------------------------------------------------------------------------
# Power / attenuation trade-off (nominal inputs)
# ---------------------------------------------------------------------------

def fig_tradeoff(theme, C, duty=0.1):
    V = np.geomspace(500.0, 20000.0, 60)
    marks = np.array([1e3, 2e3, 5e3, 10e3, 20e3])
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    for h, col in zip((74.0, 72.0, 70.0, 68.0), (C["green"], C["blue"], C["orange"], C["purple"])):
        fp = FlightPlasma(h, nominal())

        def point(v):
            J, _ = fp.currents(v)
            return duty * v * float(J) / 1e4, loss_db(fp, float(fp.sheath_thickness(np.array([v]))[0]))

        P, L = np.array([point(v) for v in V]).T
        ax.plot(P, L, color=col, label=f"{h:.0f} km (unpulsed {loss_db(fp):.0f} dB)")
        Pm, Lm = np.array([point(v) for v in marks]).T
        ax.plot(Pm, Lm, "o", color=col, mec=plt.rcParams["axes.facecolor"], mew=1.2)
        if h == 68.0:
            for p, l, v in zip(Pm, Lm, marks):
                ax.annotate(f"{v/1e3:g} kV", (p, l), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=8, color=quiet())
    ax.axhline(LOSS_MAX_DB, color=ink(), lw=1, ls=(0, (4, 3)))
    ax.set_xscale("log")
    ax.text(0.01, LOSS_MAX_DB, "10 dB link limit", va="bottom", fontsize=8.5, transform=ax.get_yaxis_transform())
    ax.set_xlabel(f"Time-averaged electrical power at duty cycle {duty:g} (W cm$^{{-2}}$ of cathode)")
    ax.set_ylabel("One-way S-band plasma loss during pulse (dB)")
    ax.set_ylim(bottom=0)
    ax.legend(title="Altitude, nominal inputs", loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "power_attenuation_tradeoff", theme)


# ---------------------------------------------------------------------------
# QMC convergence
# ---------------------------------------------------------------------------

def fig_convergence(theme, C):
    c = np.load(RES / "convergence.npz")
    N, eq, em = c["N"], c["err_qmc"], c["err_mc"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    bg = plt.rcParams["axes.facecolor"]
    ax.loglog(N, em, "o-", color=C["orange"], mec=bg, label="Plain Monte Carlo")
    ax.loglog(N, eq, "o-", color=C["blue"], mec=bg, label="Scrambled Sobol (QMC)")
    ax.loglog(N, em[0] * (N / N[0]) ** -0.5, color=quiet(), lw=1, ls=":", label="$N^{-1/2}$")
    ax.loglog(N, em[0] * (N / N[0]) ** -1.0, color=quiet(), lw=1, ls="--", label="$N^{-1}$")
    ax.set_xticks(N)
    ax.set_xticklabels([str(int(n)) for n in N])
    ax.minorticks_off()
    ax.set_xlabel("Samples $N$ (–)")
    ax.set_ylabel("RMS error in link probability, 70 km, 7.5 kV (–)")
    ax.legend()
    save(fig, "qmc_convergence", theme)


# ---------------------------------------------------------------------------
# Animation: descent along the RAM-C II trajectory (nominal inputs)
# ---------------------------------------------------------------------------

def descent_data(V=10000.0, h_grid=np.arange(84.0, 57.9, -0.5)):
    out = []
    for h in h_grid:
        fp = FlightPlasma(float(h), nominal())
        d_req, loss0 = fp.required_depth()
        d_s = float(fp.sheath_thickness(np.array([V]))[0])
        out.append(dict(h=float(h), y=fp.y * 100, n=fp.n_e.copy(), d_req=d_req * 100, d_s=d_s * 100,
                        loss0=loss0, loss1=loss_db(fp, d_s)))
    return out


def gif_descent(theme, C, data, V=10000.0, cap=60.0):
    n_crit = critical_density(F_LINK)
    h = np.array([d["h"] for d in data])
    L0 = np.minimum([d["loss0"] for d in data], cap)
    L1 = np.minimum([d["loss1"] for d in data], cap)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 4.2), dpi=100, gridspec_kw=dict(width_ratios=[1, 1.3]))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, wspace=0.28)

    a.axvline(LOSS_MAX_DB, color=ink(), lw=1, ls=(0, (4, 3)))
    a.text(LOSS_MAX_DB + 1, 83.6, "10 dB limit", fontsize=8, va="top")
    l0, = a.plot([], [], color=C["orange"], label="No pulse")
    l1, = a.plot([], [], color=C["blue"], label=f"Pulsed, {V/1e3:g} kV")
    m0, = a.plot([], [], "o", color=C["orange"])
    m1, = a.plot([], [], "o", color=C["blue"])
    a.set_xlim(0, cap)
    a.set_ylim(58, 84)
    a.set_xlabel(f"One-way S-band loss (dB, capped at {cap:g})")
    a.set_ylabel("Altitude (km)")
    a.legend(loc="lower right")

    prof, = b.semilogy([], [], color=C["orange"])
    b.axhline(n_crit, color=ink(), lw=1, ls=(0, (4, 3)))
    b.text(12.8, n_crit * 1.15, "critical density, 2.25 GHz", fontsize=8, ha="right")
    span = b.axvspan(0, 0, color=C["blue"], alpha=0.3, lw=0, label="Electron-free sheath (pulse on)")
    req = b.axvline(0, color=C["green"], lw=1.6, label="Sheath depth required for link")
    b.set_xlim(0, 13)
    b.set_ylim(1e15, 3e18)
    b.set_xlabel("Distance from wall (cm)")
    b.set_ylabel("Electron number density (m$^{-3}$)")
    b.legend(loc="upper right")
    title = fig.text(0.08, 0.93, "", fontsize=11, fontweight="bold")
    status = fig.text(0.98, 0.93, "", fontsize=11, fontweight="bold", ha="right")

    frames = list(range(len(data))) + [len(data) - 1] * 12   # hold the last frame

    def draw(i):
        d = data[i]
        l0.set_data(L0[: i + 1], h[: i + 1])
        l1.set_data(L1[: i + 1], h[: i + 1])
        m0.set_data([L0[i]], [h[i]])
        m1.set_data([L1[i]], [h[i]])
        prof.set_data(d["y"], np.maximum(d["n"], 1.0))
        span.set_width(min(d["d_s"], 13.0))
        if np.isfinite(d["d_req"]) and d["d_req"] > 0:
            req.set_xdata([d["d_req"]] * 2)
            req.set_visible(True)
        else:
            req.set_visible(False)
        title.set_text(f"RAM-C II descent   {d['h']:.1f} km")
        if d["loss0"] <= LOSS_MAX_DB:
            status.set_text("link open without pulses")
            status.set_color(C["green"])
        elif d["loss1"] <= LOSS_MAX_DB:
            status.set_text("link open only while pulsed")
            status.set_color(C["blue"])
        else:
            status.set_text("blackout")
            status.set_color(C["orange"])
        return l0, l1, m0, m1, prof, span, req, title, status

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    OUT.mkdir(parents=True, exist_ok=True)
    anim.save(OUT / f"descent-{theme}.gif", writer=PillowWriter(fps=10), savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)


FIGS = {"hero": fig_hero, "trajectory": fig_trajectory, "validation": fig_validation, "sobol": fig_sobol,
        "tradeoff": fig_tradeoff, "convergence": fig_convergence, "gif": gif_descent}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(FIGS)
    data = descent_data() if "gif" in todo else None
    for theme, rc in THEMES.items():
        with plt.style.context([str(STYLE), rc]):
            for k in todo:
                if k == "gif":
                    gif_descent(theme, PALETTE[theme], data)
                else:
                    FIGS[k](theme, PALETTE[theme])
                print(f"wrote {k} ({theme})")
    if data:
        print("descent summary (h km, loss unpulsed dB, loss pulsed dB, d_s cm, d_req cm):")
        for d in data[::4]:
            print(f"  {d['h']:5.1f} {d['loss0']:8.1f} {d['loss1']:8.1f} {d['d_s']:6.2f} {d['d_req']:6.2f}")
