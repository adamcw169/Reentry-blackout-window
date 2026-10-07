# Digitised data: Rodríguez Fuentes & Parent (2026)

Source: *Electron density depletion in reentry plasma flows using pulsed electric fields*, Physics of Fluids 38, 036114 (2026), [arXiv:2512.18163v2](https://arxiv.org/abs/2512.18163). Case: 2D wedge, Mach 24, 68 km, q = 2.5 kPa, wall 1400 K, cathode 40 mm long.

| File | Figure | Content |
|---|---|---|
| `profile_68km_no_pulse.csv` | Fig. 18a and 18c, blue curves | Electron density and gas temperature vs distance from the cathode midpoint, before any pulse |
| `cathode_iv_7p5kV.csv` | Fig. 16 | Cathode current per unit depth (A/m) vs cathode voltage, uncorrected and high-field-corrected ion mobility |
| `closure.py` constants | Fig. 18b | Power deposited at 7.5 kV for γe = 0.1, 0.2, 0.3, 0.5, integrated by eye (ratios 1, 1.39, 2.02, 3.74; about ±10%) |

Method: pages rasterised at 300 dpi; axes calibrated from detected frame and tick marks; curves extracted by colour (blue) or by run position (Fig. 16, dashed curve left of solid). The 7.5 kV end points of Fig. 16 (10 and 6 A/m) were read directly at the frame edge.

Checks:
- Fig. 16 integrated over the triangle waveform gives 58 W/cm² time-averaged power for the uncorrected case. The paper reports 66 W/cm² (from E·J over the domain): 12% apart.
- Mean density of the profile from 0 to 2.53 cm is 1.35×10¹⁷ m⁻³. The paper's slab value is 2.31×10¹⁷ m⁻³, nearer the profile peak (2.4×10¹⁷), so their "average" is closer to a peak value.

Accuracy: about ±2% of axis span in position; density read on a log axis, roughly ±10% in value. Above 2.7 cm the temperature curve is hidden under the pulsed curves; it is held at 4200 K there.
