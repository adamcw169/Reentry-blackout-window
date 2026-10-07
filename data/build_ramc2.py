"""Build RAM-C II datasets from NASA TN D-6062 (Grantham, 1970).

Source (public domain): https://ntrs.nasa.gov/citations/19710004000

Outputs
-------
ramc2_trajectory.csv     Table IX: time from lift-off, altitude, velocity
ramc2_reflectometer.csv  Table X: times at which the PEAK electron density along a
                         station's normal crossed 0.63 x critical density of each
                         reflectometer band (onset = rising, decay = falling),
                         converted to altitude and peak density.

Method notes
------------
* Table X was transcribed from an OCR'd scan. Event TIMES are legible; printed
  altitudes are partly garbled. Altitude is therefore interpolated from Table IX
  using the event time, and the printed altitude (where legible) is kept only
  as a cross-check column.
* Peak density at each event: n_pk = 0.63 * n_cr(f), with the report's factor-of-2
  error bar. n_cr is computed from first principles (eps0 * m_e * omega^2 / e^2).
* Grantham's "primary data period" (clean, ablation-free nose) ends at beryllium
  cap ejection, 56.39 km. Events below that are flagged `ablation_period`.
* Station 4 entries and some footnoted cells are ambiguous in the scan; they are
  flagged `low_confidence` and excluded from model fitting.
"""

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from blackout.plasma import critical_density  # noqa: E402

FT = 0.3048
CAP_EJECTION_KM = 56.39

# Table IX, upper part: 450 to 235 kft in 5 kft steps
_alt_kft_hi = np.arange(450, 234, -5)
_t_hi = [363.9, 364.6, 365.4, 366.2, 366.9, 367.7, 368.4, 369.3, 370.0, 370.8, 371.6,
         372.3, 373.1, 373.8, 374.7, 375.4, 376.2, 376.9, 377.7, 378.5, 379.3, 380.1,
         380.9, 381.6, 382.3, 383.1, 383.8, 384.7, 385.4, 386.2, 386.9, 387.7, 388.5,
         389.3, 390.0, 390.8, 391.6, 392.3, 393.1, 393.8, 394.7, 395.4, 396.2, 396.9]
# Note: the scan prints 381.6 / 380.5 out of order at 340/335 kft; 380.9 is the
# monotonic interpolation used here (affects no Table X event).
_v_hi = [7.59] * 11 + [7.62] * 15 + [7.65] * 18

# Table IX, lower part: 70.10 km down to 12.19 km in 5 kft steps (km column is clean)
_alt_km_lo = [70.10, 68.58, 67.06, 65.53, 64.01, 62.48, 60.96, 59.44, 57.91, 56.39,
              54.86, 53.34, 51.82, 50.29, 48.77, 47.24, 45.72, 44.20, 42.67, 41.15,
              39.62, 38.10, 36.58, 35.05, 33.53, 32.00, 30.48, 28.96, 27.43, 25.91,
              24.38, 22.86, 21.34, 19.81, 18.29, 16.76, 15.24, 13.72, 12.19]
_t_lo = [397.7, 398.4, 399.3, 400.0, 400.8, 401.6, 402.3, 403.1, 403.8, 404.7,
         405.4, 406.2, 406.9, 407.8, 408.5, 409.3, 410.1, 410.8, 411.7, 412.4,
         413.2, 414.0, 414.8, 415.7, 416.5, 417.3, 418.3, 419.2, 420.1, 421.1,
         422.2, 423.3, 424.8, 426.3, 428.3, 430.8, 434.1, 438.9, 446.2]
# (scan prints 400.8 before 400.0; swapped back to monotonic order)
_v_lo = ([7.65] * 11 + [7.62] * 3 + [7.59] * 2 +
         [7.56, 7.53, 7.47, 7.44, 7.38, 7.32, 7.22, 7.10, 6.95, 6.80, 6.55] +
         [6.31, 6.00, 5.61, 5.12, 4.60, 3.93, 3.29, 2.56, 1.86, 1.28, 0.76, 0.39])

TIME = np.array(_t_hi + _t_lo)
ALT_KM = np.concatenate([_alt_kft_hi * 1e3 * FT / 1e3, _alt_km_lo])
VEL = np.array(_v_hi + _v_lo)
assert len(TIME) == len(ALT_KM) == len(VEL)
assert np.all(np.diff(TIME) > 0)

FREQ_HZ = {"L": 1228e6, "S": 3348e6, "X": 10044e6, "Ka": 35000e6}
STATION_XD = {1: 0.15, 2: 0.76, 3: 2.30, 4: 3.48}

# Table X events: (station, band, event, time_s or None, printed_alt_kft or None, notes)
EVENTS = [
    (1, "S", "onset", 390.3, 277, ""),
    (1, "S", "decay", 424.3, None, "b: decay inferred from end of modulation"),
    (1, "X", "onset", 392.3, 264, ""),
    (1, "X", "decay", 420.0, None, ""),
    (1, "Ka", "onset", None, 236, "c: from text, about 236 kft"),
    (2, "L", "onset", 390.3, 277, ""),
    (2, "L", "decay", 423.4, None, ""),
    (2, "S", "onset", 393.8, 254.8, ""),
    (2, "S", "decay", 422.3, None, ""),
    (2, "X", "onset", 397.8, 229, ""),
    (2, "X", "decay", 419.2, None, ""),
    (2, "Ka", "onset", 414.0, 125, ""),
    (2, "Ka", "decay", 417.8, 102, ""),
    (3, "L", "onset", 391.6, 269, ""),
    (3, "L", "decay", 421.8, 81.8, ""),
    (3, "S", "onset", 395.8, 242, ""),
    (3, "S", "decay", 420.8, None, ""),
    (3, "X", "onset", 406.8, 170, "e: windward/leeward oscillation"),
    (3, "X", "decay", 417.8, None, "e: windward/leeward oscillation"),
    (4, "S", "onset", 397.0, 234.8, "low_confidence: band assignment unclear in scan"),
    (4, "S", "decay", 419.6, 92, "low_confidence: band assignment unclear in scan"),
    (4, "X", "decay", 415.3, 117, "low_confidence; e"),
]


def altitude_at(t):
    return float(np.interp(t, TIME, ALT_KM))


def velocity_at(t):
    return float(np.interp(t, TIME, VEL))


def time_at_altitude(alt_km):
    return float(np.interp(alt_km, ALT_KM[::-1], TIME[::-1]))


def build():
    with open(HERE / "ramc2_trajectory.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "altitude_km", "velocity_km_s"])
        for row in zip(TIME, ALT_KM, VEL):
            w.writerow([f"{row[0]:.1f}", f"{row[1]:.2f}", f"{row[2]:.2f}"])

    rows = []
    for st, band, ev, t, alt_kft, notes in EVENTS:
        if t is None:
            alt = alt_kft * 1e3 * FT / 1e3
            t = time_at_altitude(alt)
        else:
            alt = altitude_at(t)
        printed_km = alt_kft * 1e3 * FT / 1e3 if alt_kft else None
        n_pk = 0.63 * critical_density(FREQ_HZ[band])
        flags = []
        if alt < CAP_EJECTION_KM:
            flags.append("ablation_period")
        if "low_confidence" in notes or notes.startswith(("b", "c", "e")):
            flags.append("low_confidence")
        rows.append({
            "station": st, "x_over_D": STATION_XD[st], "band": band,
            "freq_MHz": FREQ_HZ[band] / 1e6, "event": ev,
            "time_s": round(t, 1), "altitude_km": round(alt, 2),
            "printed_altitude_km": round(printed_km, 2) if printed_km else "",
            "velocity_km_s": round(velocity_at(t), 2),
            "n_e_peak_m3": f"{n_pk:.3e}", "n_e_low_m3": f"{n_pk/2:.3e}",
            "n_e_high_m3": f"{n_pk*2:.3e}",
            "use_for_fit": "no" if flags else "yes",
            "flags": ";".join(flags), "notes": notes,
        })

    with open(HERE / "ramc2_reflectometer.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    rows = build()
    print(f"{'st':>2} {'band':>4} {'event':>6} {'t [s]':>6} {'alt [km]':>8} {'printed':>8} {'n_pk [m^-3]':>12}  fit")
    for r in rows:
        print(f"{r['station']:>2} {r['band']:>4} {r['event']:>6} {r['time_s']:>6} "
              f"{r['altitude_km']:>8} {str(r['printed_altitude_km']):>8} {r['n_e_peak_m3']:>12}  {r['use_for_fit']}")
    diffs = [abs(r["altitude_km"] - r["printed_altitude_km"]) for r in rows
             if r["printed_altitude_km"] != "" and "c:" not in r["notes"]]
    print(f"\nTime-derived vs printed altitude: max difference {max(diffs):.2f} km over {len(diffs)} legible cells")
