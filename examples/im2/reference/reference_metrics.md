# Reference benchmark metrics

Acceptance description for the `im2` CSEM inversion benchmark (synthetic model
2). `scripts/analyze_inversion.py` reproduces the measured values from a
completed run and checks them against the tolerances in
`reference_metrics.json`.

Coordinates use **z negative downward** (depth is a negative z), the same
convention as `examples/fm` and `examples/im1`. A *positive* vertical offset
therefore means the recovered body sits shallower than the truth.

Produced on the current `geometry/mesh.geo` (57276 cells, 355502 DOFs at order
2) with the PCBDDC iterative solver (`configs/solver_bddc.txt`) on **112 MPI
tasks** (one node), in 00:57:28.

The domain decomposition is part of the configuration: the L-BFGS path follows
the preconditioned solves, so reproduce these metrics at 112 tasks.

## True model

A checkerboard of four 200 m cubes at the same depth in a uniform halfspace.

| quantity | value |
|---|---|
| background resistivity | 100 Ω·m |
| conductive bodies | 2 × 10 Ω·m |
| resistive bodies | 2 × 1000 Ω·m |
| body size | 200 × 200 × 200 m (8.0 × 10⁶ m³ each) |
| burial depth | 100 m (top at z = −100 m, bottom at z = −300 m) |
| gap between adjacent bodies | 100 m |

| body | polarity | resistivity | centre (x, y, z) | footprint |
|---|---|---|---|---|
| C1 | conductor | 10 Ω·m | (+150, −150, −200) m | x ∈ [50, 250], y ∈ [−250, −50] |
| C2 | conductor | 10 Ω·m | (−150, +150, −200) m | x ∈ [−250, −50], y ∈ [50, 250] |
| R1 | resistor | 1000 Ω·m | (+150, +150, −200) m | x ∈ [50, 250], y ∈ [50, 250] |
| R2 | resistor | 1000 Ω·m | (−150, −150, −200) m | x ∈ [−250, −50], y ∈ [−250, −50] |

The conductors sit on the (+,−)/(−,+) diagonal and the resistors on the
(+,+)/(−,−) diagonal, so each quadrant of the survey holds exactly one body.
That is what makes the quadrant a valid search region for attributing a smeared
reconstruction to the right target, and it is how `search_box_m` is set in
`reference_metrics.json`.

## Survey

| quantity | value |
|---|---|
| source | x-directed electric dipole at (0, −4000, 0) m, unit current and length |
| receivers | 441 (21 × 21), x, y ∈ [−500, 500] m, 50 m spacing, z = 0 |
| frequencies | 1, 10, 50, 100, 300, 800, 1500 Hz |
| data | real and imaginary parts of Ex |
| basis order | 2 (Nédélec edge elements) |

## Mesh

Built from `geometry/mesh.geo` by `scripts/build_meshes.py`.

| mesh | cells | tags |
|---|---|---|
| `mesh.msh` (starting model) | 57276 | AIR = 20040, BG = 15387, INVERT = 21849 |
| `mesh_true.msh` (true model) | 57276 | AIR = 20040, BG = 15387, INVERT = 15519, COND = 3173, RES = 3157 |

Both meshes share one set of nodes and cells and differ only in their material
labels. COND carries both 10 Ω·m cubes and RES both 1000 Ω·m cubes, so the four
bodies need two tags rather than four.

## Observed data

| quantity | value |
|---|---|
| noise level | 1 % (`error_level = 0.01`) |
| noise seed | 20260922 |
| RMS of the true model | 0.9864 |

RMS ≈ 1 for the true model confirms the observed data lies exactly one
noise-level away from the truth - i.e. the noise model is statistically
consistent with how the inversion weights the misfit. The value is computed by
`utils/make_observed.py` and stored as the `rms_true_model` attribute of
`observed.h5`, so it is checkable rather than nominal.

## Inversion configuration

Table 2 of the paper, as set in `configs/params_im.txt`:

| parameter | value |
|---|---|
| starting model | 100 Ω·m uniform halfspace |
| error level | 1 % |
| regularization factor λ | 0.1 |
| target RMS threshold | 1.05 |
| maximum iterations | 150 (Table 2 says 80; see below) |

## Convergence

| quantity | value |
|---|---|
| initial RMS | 7.0604 |
| final RMS | 1.0491 |
| iterations | 105 accepted L-BFGS steps (112 objective-gradient evaluations) |
| termination | `CONVERGED_RMSTOL` (reached the RMS = 1.05 target) |
| wall time | 00:57:28 on 112 MPI tasks (32.8 s per step) |

Each evaluation issues `2 x n_freq` = 14 linear solves (forward and adjoint
share the KSP), so the run performed 1568 solves at 2.20 s each.

**Table 2 of the paper caps the run at 80 iterations and reports 38; this kernel
needs 105.** At a cap of 80 the run exits `DIVERGED_MAXITS` at RMS 1.1145 with
the misfit still descending 0.4 % per step, so `configs/params_im.txt` sets 150.
The discrepancy is not the regularization - the Section 2 gradient smoother is
implemented and active (`src/inversion_smoother.c`: vertex-star adjacency,
normalised 1/distance weights, forward then reverse sweep, no diagonal
self-term, matching eq. 15) - so it lies in the optimisation path. Open.

**The step count is not reproducible to the unit.** Two runs from the same
bundle on 112 tasks agree to four decimals through iteration 20, then drift to
about 1 % in RMS by iteration 70: round-off non-determinism in the MPI
reductions of the iterative solver, amplified by L-BFGS. Expect 100-115 steps.

`num_iterations` (accepted L-BFGS steps) and `num_objgrad_evaluations` (length
of `/rms_history`) are two different counters, and conflating them is the
classic mistake here. `/rms_history` also records rejected line-search trials,
so it is not monotone even though the accepted steps are.

## Recovered model

Background: 100.0 Ω·m, unchanged - only the `INVERT` region is free, and the
recovered model holds air at 1e8 and BG at 100.0000 exactly.

| body | true ρ | peak ρ | mean ρ | contrast | centroid | lat. offset | vert. offset | volume | in box |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 10 Ω·m | 4.84 | 28.3 | 55 % | (147, −158, −157) | 8.3 m | +43.1 m | 3.57 × 10⁶ m³ | 94 % |
| C2 | 10 Ω·m | 2.96 | 25.7 | 59 % | (−156, 173, −164) | 23.4 m | +36.5 m | 4.50 × 10⁶ m³ | 84 % |
| R1 | 1000 Ω·m | 363.7 | 178.5 | 25 % | (174, 220, −173) | 73.7 m | +27.4 m | 2.86 × 10⁵ m³ | 95 % |
| R2 | 1000 Ω·m | 360.6 | 178.2 | 25 % | (−145, −156, −149) | 7.8 m | +50.6 m | 3.19 × 10⁵ m³ | 100 % |

Definitions, as the analyzer computes them. **"Peak"** is the extremum of cell
resistivity inside the body's quadrant - the minimum for a conductor, the
maximum for a resistor - a single-cell value, and for the conductors it
*overshoots* past the true 10 Ω·m. **"Mean"** is the volume-weighted geometric
mean over the true body box and is what actually describes the body, so it is
the quantity the amplitude tolerance is set on. **"Contrast"** is the fraction
of the true |log₁₀(ρ_body/ρ_background)| that the mean represents. The centroid
is weighted by cell volume × contrast (1/ρ for a conductor, ρ for a resistor)
over the cells that cross the detection threshold, and both offsets follow from
it. The volume is the sum of those same cells. "In box" is the share of that
volume falling inside the true cube.

### Cross-contamination: none

Of the 6330 cells inside the four true boxes, **zero** have the wrong sign - no
resistive cell in a conductor, no conductive cell in a resistor. Outside the
boxes, only 75 conductive and 1 resistive cell are anomalous, and all of them
lie 104-148 m from their own body centre; a cube corner is at 173 m, so that is
the smearing halo, not an artifact elsewhere in the model.

### Amplitude: the expected asymmetry

The conductors recover 55-59 % of their true log-contrast, the resistors 25 % -
a factor 2.3. This is physics, not a solver deficiency: a CSEM field is far more
sensitive to a conductor than to a resistor of the same contrast. Between step 80
(RMS 1.1315) and step 105 (RMS 1.0491) the body means change by 11-14 %.

The depth profile shows the same limitation and the shallow bias, as volume-
weighted mean ρ in each body column (|x−x_c| ≤ 100, |y−y_c| ≤ 100):

| band | C1 | C2 | R1 | R2 |
|---|---|---|---|---|
| z ∈ [−100, 0] | 80.1 | 83.3 | 107.7 | 110.2 |
| **z ∈ [−200, −100]** | **18.9** | **16.6** | **201.7** | **213.9** |
| z ∈ [−300, −200] | 42.5 | 41.4 | 158.1 | 147.2 |
| z ∈ [−400, −300] | 56.6 | 56.7 | 121.6 | 114.6 |

The anomaly concentrates in the upper half of the true body and decays below
it - source and receivers both sit on the air/earth interface at a 4 km offset,
so the airwave dominates and |Ex| is nearly flat above 10 Hz (median 5.15e-10 at
10 Hz against 4.03e-10 at 1500 Hz).

### What this case tests, and how it answered

`examples/im1` asks whether a single conductor is recovered at all. This one asks
three further questions. All three are answered by the run above.

1. **Polarity.** Does each quadrant come back with the correct sign? **Yes**, all
   four. The analyzer checks this first, before any quantitative tolerance.
2. **Cross-contamination.** Does a conductor leak into the neighbouring
   resistor's quadrant? **No** - zero wrong-sign cells among the 6330 inside the
   true boxes.
3. **Separation.** Are the bodies resolved as four distinct anomalies across
   their 100 m gaps, or smeared into one structure? **Distinct.** Every body's
   recovered volume is at or below the true 8 x 10^6 m^3 with an in-box fraction
   of 84-100 %, so nothing has merged; the failure mode to watch for is the
   opposite one, a volume much larger than the truth with a low in-box fraction.

What the case does *not* demonstrate is amplitude recovery. The conductors come
back at 55-59 % of their true log-contrast and the resistors at 25 %, so any
claim that the inversion "recovers the amplitude" of these bodies is not
supported by these numbers. The location and the sign are recovered; the
magnitude is not, and the 2.3x conductor/resistor asymmetry is the expected
physics rather than a defect.

## Convergence record

`reference/` also ships the optimisation history of this run, so a new run can
be compared step by step and not only on its final metrics:

| file | content |
|---|---|
| `lbfgs_log.txt` | one row per accepted step with RMS, objective `F`, regularization term, |g| and step length, plus the exit reason. 106 rows, iteration 0 (the starting model) to 105. |
| `rms_history.txt` | the 112 entries of `/rms_history`, one per objective-gradient evaluation, including the 6 rejected line-search trials. Not monotone, while the accepted steps are. |

## Regenerating this file

After a new complete pipeline run:

```bash
python3 examples/im2/scripts/analyze_inversion.py \
    -run_dir   examples/im2/outputs \
    -reference examples/im2/reference/reference_metrics.json
```

It reports PASS when every value falls inside the tolerances in
`reference_metrics.json`. Those bands are set around the run recorded here, wide
enough for the run-to-run drift noted above and for a re-meshed discretisation,
tight enough to fail if a body is lost, misplaced or comes back with the wrong
sign. Re-meshing changes the discretisation, so it requires the whole pipeline
to be rerun and this file regenerated.

The convergence record is extracted from a finished run rather than written by
the kernel; the two commands are in the case README under "Convergence record".
The step count in `lbfgs_log.txt` must match `iterations` in
`reference_metrics.json`, and the row count in `rms_history.txt` must match
`objgrad_evaluations`.
