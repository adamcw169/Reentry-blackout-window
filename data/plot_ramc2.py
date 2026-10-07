"""Plot RAM-C II reflectometer-derived peak electron density vs altitude."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

COLOURS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
LABELS = {1: "Station 1, x/D = 0.15 (nose)", 2: "Station 2, x/D = 0.76", 3: "Station 3, x/D = 2.30"}
INK, QUIET, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main():
    rows = list(csv.DictReader(open(HERE / "ramc2_reflectometer.csv")))
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for st in (1, 2, 3):
        pts = [r for r in rows if int(r["station"]) == st and r["use_for_fit"] == "yes"]
        pts.sort(key=lambda r: float(r["altitude_km"]))
        n = [float(r["n_e_peak_m3"]) for r in pts]
        h = [float(r["altitude_km"]) for r in pts]
        lo = [v - float(r["n_e_low_m3"]) for v, r in zip(n, pts)]
        hi = [float(r["n_e_high_m3"]) - v for v, r in zip(n, pts)]
        ax.errorbar(n, h, xerr=[lo, hi], fmt="o-", color=COLOURS[st], ms=7, lw=2,
                    capsize=3, mec=BG, mew=1.5, label=LABELS[st])
    ax.axhline(68, color=QUIET, lw=1, ls=(0, (3, 3)))
    ax.text(1.3e16, 66.6, "Arizona 2026 CFD condition, 68 km", fontsize=9, color=QUIET)
    ax.set_xscale("log")
    ax.set_xlabel("Peak electron density along station normal (m⁻³)", color=INK)
    ax.set_ylabel("Altitude (km)", color=INK)
    ax.set_title("RAM-C II: peak density rises ~70x between 85 and 70 km\n"
                 "(reflectometer cut-off events, 7.6 km/s, clean-flow period; bars = factor-of-2 error)",
                 color=INK, fontsize=10.5, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=QUIET)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 0.2), labelcolor=INK)
    ax.set_ylim(64, 86)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "v1_ramc2_density.png")


if __name__ == "__main__":
    main()
