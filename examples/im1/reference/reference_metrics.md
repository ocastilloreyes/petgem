# Reference benchmark metrics

Validated results for the `im1` CSEM inversion benchmark. `analyze_inversion.py`
reproduces these values from a completed run and checks them against the
tolerances in `reference_metrics.json`.

Coordinates use **z negative downward** (depth is a negative z), the same
convention as `examples/fm`. A *positive* vertical offset therefore means the
recovered body sits shallower than the truth.

Produced on the current `geometry/mesh.geo` (58022 cells, 374144 DOFs at order
2) with the PCBDDC iterative solver (`configs/solver_bddc.txt`) on **112 MPI
tasks** (one node), in 01:13:12.

The domain decomposition is part of the configuration: the L-BFGS path follows
the preconditioned solves, so reproduce these metrics at 112 tasks.

## True model

| quantity | value |
|---|---|
| background resistivity | 100 Ω·m |
| anomaly resistivity | 10 Ω·m |
| anomaly centroid | (0, 0, -200) m |
| anomaly volume | 8.0 × 10⁶ m³ (200 m cube) |

## Observed data

| quantity | value |
|---|---|
| noise level | 1 % (`error_level = 0.01`) |
| noise seed | 20260720 |
| RMS of the true model | 0.994 |

RMS ≈ 1 for the true model confirms the observed data lies exactly one
noise-level away from the truth - i.e. the noise model is statistically
consistent with how the inversion weights the misfit. The value is computed by
`utils/make_observed.py` and stored as the `rms_true_model` attribute of
`observed.h5`, so it is checkable rather than nominal.

## Convergence

| quantity | value |
|---|---|
| initial RMS | 11.795 |
| final RMS | 1.0499 |
| iterations | 94 accepted L-BFGS steps (105 objective-gradient evaluations) |
| termination | `CONVERGED_RMSTOL` (reached the RMS = 1.05 target) |

The 105 evaluations are the initial one at the starting model plus 104
line-search trials, of which 94 were accepted and 10 rejected: 1.117 evaluations
per accepted step. Each evaluation issues `2 x n_freq` = 14 linear solves
(forward and adjoint share the KSP), so the run performed 1470 solves.

RMS decreases monotonically over the **accepted** L-BFGS steps. The
`/rms_history` dataset in `responses_im_p2.h5` is not monotone, because it
records every objective-gradient evaluation, including the ten line-search
trials that were rejected. `responses_im_p2.h5` stores both counters separately -
`num_iterations` (accepted steps) and `num_objgrad_evaluations` (length of
`rms_history`) - and `analyze_inversion.py` prints them on separate lines.

## Recovered model

| quantity | value | true |
|---|---|---|
| background resistivity | 100 Ω·m | 100 |
| peak (minimum) resistivity | 9.00 Ω·m | 10 |
| conductor centroid | (-1.6, -1.0, -166.4) m | (0, 0, -200) |
| lateral offset | 1.8 m | 0 |
| vertical offset | +33.6 m (shallower than true) | 0 |
| conductor volume (ρ < 30 Ω·m) | 4.35 × 10⁶ m³ | 8.0 × 10⁶ |
| fraction inside the true box | 72 % | - |

Each entry carries its definition. The peak is the minimum cell resistivity
anywhere in the earth, a single-cell extremum rather than an average over the
body. The centroid is weighted by cell volume / cell resistivity over the cells
with ρ < 30 Ω·m, and both offsets follow from it. The volume is the sum of those
same cells, 54 % of the true 8 × 10⁶ m³. "Fraction inside the true box" is the
share of the recovered volume that falls within the true cube, so the overlap
with the true body is 0.72 × 4.35 × 10⁶ / 8 × 10⁶ ≈ 39 %.

The inversion fits the data to the noise floor and recovers a conductor of the
correct magnitude at the correct lateral position. The shallow bias and reduced
volume reflect the limited vertical resolution of this acquisition: source and
receivers both sit on the air/earth interface at a 4 km offset, so the recorded
field is dominated by the airwave and the amplitude spectrum is nearly flat
above 10 Hz. The seven frequencies therefore carry less independent
depth information than their range suggests.
