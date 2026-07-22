# Reference benchmark metrics

Validated results for the `im_model` CSEM inversion benchmark. `analyze_inversion.py`
reproduces these values from a completed run and checks them against the
tolerances in `reference_metrics.json`.

## True model

| quantity | value |
|---|---|
| background resistivity | 100 Ω·m |
| anomaly resistivity | 10 Ω·m |
| anomaly centroid | (0, 0, 200) m |
| anomaly volume | 8.0 × 10⁶ m³ (200 m cube) |

## Observed data

| quantity | value |
|---|---|
| noise level | 1 % (`error_level = 0.01`) |
| noise seed | 20260720 |
| RMS of the true model | ≈ 1.0 |

RMS ≈ 1 for the true model confirms the observed data lies exactly one
noise-level away from the truth - i.e. the noise model is statistically
consistent with how the inversion weights the misfit.

## Convergence

| quantity | value |
|---|---|
| initial RMS | 11.35 |
| final RMS | 1.05 |
| iterations | 88 (93 objective-gradient evaluations) |
| termination | `CONVERGED_RMSTOL` (reached the RMS = 1.05 target) |

RMS decreases monotonically at every iteration.

## Recovered model

| quantity | value | true |
|---|---|---|
| background resistivity | 100 Ω·m | 100 |
| peak (minimum) resistivity | ≈ 11 Ω·m | 10 |
| conductor centroid | (-3, -6, 159) m | (0, 0, 200) |
| lateral offset | 6.9 m | 0 |
| vertical offset | -41 m (shallow) | 0 |
| conductor volume (ρ < 30 Ω·m) | 3.5 × 10⁶ m³ | 8.0 × 10⁶ |
| fraction inside the true box | 73 % | - |

The inversion fits the data to the noise floor and recovers a conductor of the
correct magnitude at the correct lateral position. The shallow bias and reduced
volume reflect the limited vertical resolution of a single surface receiver
plane - the expected behaviour of a regularized CSEM inversion.
