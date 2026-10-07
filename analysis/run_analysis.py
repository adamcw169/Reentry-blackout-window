"""Run the uncertainty analysis and save results to analysis/results/.

Stages (each can be run alone: python analysis/run_analysis.py maps sobol ...)
  trajectory   P(link) vs altitude, with and without pulses             (QMC, 512 per altitude)
  slender      the same for a slender vehicle with an Arizona-like thin plasma layer
  maps         P(feasible) over voltage x duty cycle at 74, 70, 66 km    (QMC, 1024 per altitude)
  sobol        Sobol indices of the sheath margin at 70 km, 10 kV       (Saltelli, N = 1024)
  convergence  QMC vs plain MC error for P(link) at 70 km, 7.5 kV
  costs        power, heating, arc margin and timescales at 70 km, 7.5 kV
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackout.atmosphere import ramc2_time  # noqa: E402
from blackout.uq import bootstrap_ci, random_points, saltelli_matrices, sobol_indices, sobol_points  # noqa: E402
from blackout.window import (  # noqa: E402
    LINK_INPUTS, T_WALL_MAX, UNCERTAIN, FlightPlasma, evaluate, nominal, scale_unit, timescales)

OUT = ROOT / "analysis" / "results"
OUT.mkdir(parents=True, exist_ok=True)
K = len(UNCERTAIN)
V_GRID = np.geomspace(500.0, 20000.0, 41)
D_GRID = np.geomspace(0.01, 1.0, 31)
D_REF = 0.1          # duty cycle used for the trajectory sweep
V_REF = 7500.0


def _params(u, vehicle):
    params = scale_unit(u)
    if vehicle == "slender":   # thin layer like the Arizona sharp wedge (2.5 cm) instead of RAM-C II (>7 cm)
        params["thickness_factor"] = 0.75 + 0.75 * u[:, list(UNCERTAIN).index("thickness_factor")]
    return params


def stage_trajectory(n=512, vehicle="blunt"):
    h_grid = np.arange(84.0, 57.9, -1.0)
    P_none, P_pulse, P_link_any = [], [], []
    u = sobol_points(n, K, seed=11)
    params = _params(u, vehicle)
    iD = int(np.argmin(abs(D_GRID - D_REF)))
    for h in h_grid:
        r = evaluate(h, params, V_GRID, D_GRID)
        P_none.append(np.mean(r["d_req"] == 0.0))
        P_link_any.append(np.mean(r["link"].any(axis=1)))
        P_pulse.append(np.mean(r["feasible"][:, :, iD].any(axis=1)))
        print(f"  {h:.0f} km: no pulse {P_none[-1]:.2f}, pulsed link {P_link_any[-1]:.2f}, feasible at D={D_REF} {P_pulse[-1]:.2f}")
    np.savez(OUT / f"trajectory_{vehicle}.npz", h=h_grid, t=ramc2_time(h_grid), P_none=P_none,
             P_link_any=P_link_any, P_pulse=P_pulse)


def stage_maps(n=1024, altitudes=(74.0, 70.0, 66.0)):
    params = scale_unit(sobol_points(n, K, seed=21))
    out = {"V": V_GRID, "D": D_GRID}
    for h in altitudes:
        r = evaluate(h, params, V_GRID, D_GRID)
        key = f"{h:.0f}"
        out[f"P_feasible_{key}"] = r["feasible"].mean(axis=0)
        out[f"P_none_{key}"] = np.mean(r["d_req"] == 0.0)
        out[f"P_link_{key}"] = r["link"].mean(axis=0)
        out[f"P_peak_median_{key}"] = np.median(r["P_peak"], axis=0)
        out[f"Tw_ok_{key}"] = (r["T_wall"] <= T_WALL_MAX).mean(axis=0)
        print(f"  {key} km: max P(feasible) {out[f'P_feasible_{key}'].max():.2f}, "
              f"max P(link) {out[f'P_link_{key}'].max():.2f}")
    np.savez(OUT / "maps.npz", **out)


def _margin(params, h=70.0, V=10000.0):
    n = len(next(iter(params.values())))
    m = np.empty(n)
    for i in range(n):
        fp = FlightPlasma(h, {k: v[i] for k, v in params.items()})
        d_req, _ = fp.required_depth()
        d_s = float(fp.sheath_thickness(np.array([V]))[0])
        m[i] = (min(d_s, 0.13) - min(d_req, 0.13)) * 1e3   # mm
    return m


def stage_sobol(n=1024):
    d = len(LINK_INPUTS)
    A, B, AB = saltelli_matrices(n, d, seed=31)
    pad = lambda U: np.hstack([U, np.full((U.shape[0], K - d), 0.5)])
    t = time.time()
    fA, fB = _margin(scale_unit(pad(A))), _margin(scale_unit(pad(B)))
    fAB = np.array([_margin(scale_unit(pad(AB[i]))) for i in range(d)])
    S1, ST = sobol_indices(fA, fB, fAB)
    (S1lo, S1hi), (STlo, SThi) = bootstrap_ci(fA, fB, fAB)
    print(f"  {n*(d+2)} model runs in {time.time()-t:.0f} s")
    for name, a, b in zip(LINK_INPUTS, S1, ST):
        print(f"  {name:>18}: S1 {a:5.2f}  ST {b:5.2f}")
    np.savez(OUT / "sobol.npz", names=LINK_INPUTS, S1=S1, ST=ST, S1_ci=np.array([S1lo, S1hi]),
             ST_ci=np.array([STlo, SThi]), margin_mean=np.mean(fA), margin_sd=np.std(fA),
             P_positive=np.mean(fA >= 0))


def _p_link(U, h=70.0, V=V_REF):
    params = scale_unit(U)
    n = U.shape[0]
    ok = np.empty(n, bool)
    for i in range(n):
        fp = FlightPlasma(h, {k: v[i] for k, v in params.items()})
        ok[i] = float(fp.sheath_thickness(np.array([V]))[0]) >= fp.required_depth()[0]
    return ok.mean()


def stage_convergence(reps=8, n_ref=8192):
    ref = _p_link(sobol_points(n_ref, K, seed=999))
    Ns = 2 ** np.arange(5, 10)
    err_q, err_m = [], []
    for N in Ns:
        eq = [_p_link(sobol_points(N, K, seed=100 + r)) - ref for r in range(reps)]
        em = [_p_link(random_points(N, K, seed=200 + r)) - ref for r in range(reps)]
        err_q.append(np.sqrt(np.mean(np.square(eq))))
        err_m.append(np.sqrt(np.mean(np.square(em))))
        print(f"  N={N:4d}: RMSE QMC {err_q[-1]:.4f}, MC {err_m[-1]:.4f}")
    np.savez(OUT / "convergence.npz", N=Ns, err_qmc=err_q, err_mc=err_m, ref=ref)


def stage_costs(h=70.0, V=V_REF):
    p = nominal()
    fp = FlightPlasma(h, p)
    J_tot, _ = fp.currents(V)
    q_c = float(fp.cathode_heat_flux(V))
    D_max = np.clip((fp.aero_heat_flux() * 0 + (0.85 * 5.670374e-8 * T_WALL_MAX**4) - fp.aero_heat_flux()) / q_c, 0, 1)
    res = {
        "altitude_km": h, "V": V, "J_A_m2": float(J_tot), "P_peak_W_cm2": float(V * J_tot / 1e4),
        "q_cathode_peak_W_cm2": q_c / 1e4, "q_aero_W_cm2": float(fp.aero_heat_flux() / 1e4),
        "T_wall_D0p05_K": float(fp.wall_temperature(V, 0.05)), "T_wall_D1_K": float(fp.wall_temperature(V, 1.0)),
        "D_max_heating": float(D_max), "gas_dT_upper_bound_K": float(fp.gas_heating(V)),
        "arc_margin_nominal": float(10 ** p["log10_J_arc"] / J_tot),
        "timescales_s": {k: float(v) for k, v in timescales(fp, V).items()},
    }
    for D in (0.05, 0.1, 1.0):
        res[f"P_avg_W_cm2_D{D}"] = float(D * V * J_tot / 1e4)
    (OUT / "costs.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


STAGES = {"trajectory": stage_trajectory,
          "slender": lambda: stage_trajectory(vehicle="slender"), "maps": stage_maps, "sobol": stage_sobol,
          "convergence": stage_convergence, "costs": stage_costs}

if __name__ == "__main__":
    for name in (sys.argv[1:] or STAGES):
        print(f"[{name}]")
        t = time.time()
        STAGES[name]()
        print(f"  done in {time.time()-t:.0f} s")
