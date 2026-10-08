# im2 - CSEM inversion benchmark

The second controlled-source electromagnetic (CSEM) inversion benchmark for
PETGEM. It is a checkerboard of four buried cubes, two conductive and two 
resistive. It runs the same end-to-end
workflow as [`examples/im1`](../im1), on a harder target:

```
true model --> forward modelling --> synthetic observations --> noise --> inversion --> recovered model
```

A known resistivity model is forward-modelled with `fm.csem`, contaminated with
1 % Gaussian noise, and inverted with `im.csem` from a homogeneous starting
model.

## Overview

`examples/im1` (synthetic model 1) asks whether a single conductor is recovered
at all. This case asks whether **multiple high-contrast targets of opposite
sign** are recovered *simultaneously*, at the right places, without leaking
into one another - the questions that make a nonlinear 3-D inversion hard.
It is otherwise the same benchmark: same solvers, same pipeline, same file
layout, same reproducibility guarantees.

Like `examples/im1`, this is self-contained. Every input is committed -
geometry, survey and configs. The observations are not shipped: they are
produced by the forward stage of the pipeline, so a run exercises the whole
chain.

The reference run is in: it converges on the RMS criterion in 105 L-BFGS steps
and the analysis passes 22/22 checks. Measured values and acceptance tolerances
are in `reference/`.

## Physical model

A resistive halfspace containing four blocks in a checkerboard, beneath an air
layer. Coordinates use **z negative downward** - depth is a negative z, the
same convention as `examples/fm` and `examples/im1`; `z = 0` is the air/earth
interface.

| region                | resistivity | conductivity |
|-----------------------|-------------|--------------|
| air                   | -           | 1 × 10⁻⁸ S/m |
| background (earth)    | 100 Ω·m     | 0.01 S/m     |
| conductive bodies (2) | 10 Ω·m      | 0.1 S/m      |
| resistive bodies (2)  | 1000 Ω·m    | 0.001 S/m    |

**Body geometry:** four 200 × 200 × 200 m cubes (8 × 10⁶ m³ each), all buried
at 100 m depth - top at z = −100 m, bottom at z = −300 m - with 100 m gaps
between neighbours:

| body | polarity, resistivity, centre (x, y) |
|------|--------------------------------------|
| C1   | conductor, 10 Ω·m, (+150, −150) m    |
| C2   | conductor, 10 Ω·m, (−150, +150) m    |
| R1   | resistor, 1000 Ω·m, (+150, +150) m   |
| R2   | resistor, 1000 Ω·m, (−150, −150) m   |

Each cube spans ±100 m in x and y around its centre.

The conductors occupy the (+,−)/(−,+) diagonal and the resistors the
(+,+)/(−,−) diagonal, so each quadrant of the survey area holds exactly one
body. The analysis tool uses that to attribute a smeared reconstruction to the
right target.

## Survey

| quantity    | value                                                                         |
|-------------|-------------------------------------------------------------------------------|
| source      | x-directed electric dipole at (0, -4000, 0) m, unit current and length        |
| receivers   | 441 (21 × 21 grid), x, y ∈ [-500, 500] m, 50 m spacing, z = 0 (the interface) |
| frequencies | 1, 10, 50, 100, 300, 800, 1500 Hz                                             |
| data        | real and imaginary parts of Ex                                                |
| basis order | 2 (Nédélec edge elements)                                                     |

The receiver grid is the one difference from `examples/im1`: the four targets
span x, y ∈ [-250, 250] m, so the footprint is stretched from ±200 m to
±500 m, keeping the same 441 points at 50 m instead of 20 m spacing.

## Directory layout

Identical in shape to `examples/im1`.

```
im2/
├── README.md
├── geometry/          meshes and their gmsh source
│   ├── mesh.geo           parametric geometry, source for both meshes
│   ├── mesh.msh           inversion mesh: AIR, BG, INVERT
│   └── mesh_true.msh      true model:     AIR, BG, INVERT, COND, RES
├── survey/            transmitters, receivers, frequencies, conductivities
│   ├── sources_im.txt     inversion source (sources_f<F>.txt: one per frequency)
│   ├── receivers.txt      441 receiver positions
│   ├── frequencies.txt    the 7 survey frequencies
│   └── sigmas_im.txt      starting model (sigmas_true.txt: the true model)
├── configs/           case options + solver presets
│   ├── params_im.txt      inversion case (I/O + L-BFGS; solver comes from a preset)
│   ├── solver_bddc.txt    solver preset: PCBDDC iterative (either kernel)
│   └── solver_mumps.txt   solver preset: MUMPS direct (either kernel)
├── scripts/           one driver per pipeline stage
│   ├── build_meshes.py       0. mesh.geo --> the two tagged .msh
│   ├── gen_survey.py         0. regenerate the survey files
│   ├── build_bundles.sh      1. and 4. preprocess --> solver input bundles
│   ├── run_fm.slurm          2. fm.csem on the true model
│   ├── make_observations.sh  3. responses + 1 % noise --> observed.h5
│   ├── run_im.slurm          5. im.csem from the homogeneous start
│   └── analyze_inversion.py  6. recovered model --> PASS/FAIL against reference/
└── reference/         expected results
    ├── reference_metrics.json  machine-readable, with acceptance tolerances
    ├── reference_metrics.md    human-readable
    ├── lbfgs_log.txt           L-BFGS trace of the reference run
    └── rms_history.txt         its RMS per objective-gradient evaluation
```

Everything generated lands in `outputs/`, which is git-ignored and **not present
in a fresh clone** - the scripts create it on demand. Only the *inputs* are
committed: geometry, survey, configs and the expected results.

General, reusable tools live in the PETGEM `utils/` package, not here:

- `utils/preprocess.py` - build a solver input bundle from a mesh + survey.
- `utils/make_observed.py` - assemble forward responses and add noise.

## The pipeline

```
true model --> forward modelling --> synthetic observations --> noise --> inversion --> recovered model
```

There is one path through this benchmark and it runs end to end. No observed
dataset is shipped: the observations the inversion targets are the ones **your**
forward run produced, so the whole chain is exercised and reproducible.

| stage | driver --> product                                                    |
|-------|-----------------------------------------------------------------------|
| 0     | `build_meshes.py`, `gen_survey.py` --> the two `.msh`, `survey/*.txt` |
| 1     | `build_bundles.sh fm` --> `outputs/input_fm_f<F>.h5` (7)              |
| 2     | `run_fm.slurm` --> `outputs/responses_fm_f<F>_p2.h5` (7)              |
| 3     | `make_observations.sh` --> `outputs/observed.h5`                      |
| 4     | `build_bundles.sh im` --> `outputs/input_im.h5`                       |
| 5     | `run_im.slurm` --> `outputs/responses_im_p2.h5`, `outputs/snapshots/` |
| 6     | `analyze_inversion.py` --> PASS/FAIL against `reference/`             |

Each stage consumes the products of the stages before it. The stage 0 files
are committed; regenerate them only if you change the model.

Stages 1, 3 and 4 are light Python preprocessing and run anywhere the `petgem`
package is installed. Stages 2 and 5 are MPI solves, normally on a cluster.

## Execution instructions

Run the commands from the repository root; the SLURM jobs are submitted from
`examples/im2`.

```bash
make

# 1-2. Forward modelling of the true model: 7 bundles, then 7 solves
bash examples/im2/scripts/build_bundles.sh fm
(cd examples/im2 && sbatch scripts/run_fm.slurm)

# 3. Synthetic observations: assemble the responses and add 1 % noise
bash examples/im2/scripts/make_observations.sh

# 4-5. Inversion from the homogeneous starting model
bash examples/im2/scripts/build_bundles.sh im
(cd examples/im2 && sbatch scripts/run_im.slurm)

# 6. Evaluate the recovered model
python3 examples/im2/scripts/analyze_inversion.py \
    -run_dir   examples/im2/outputs \
    -reference examples/im2/reference/reference_metrics.json
```

Each stage checks that its inputs exist and tells you which earlier stage
produces them, so running them out of order fails immediately rather than
silently using stale data.

If the `petgem` package is only available in the `petgem-env` container, wrap
the Python stages:

```bash
docker run --rm -u $(id -u):$(id -g) -v "$PWD":/workspace -w /workspace \
    petgem-env:latest bash examples/im2/scripts/build_bundles.sh fm
```

The SLURM scripts carry portable resource requests; set your site's
`--account` and `--qos`/`--partition`, and load the PETGEM runtime environment,
before submitting. Both default to the PCBDDC iterative solver; add
`SOLVER=mumps` for the direct one.

### Regenerating the model itself

Only needed if you change the geometry or the survey:

```bash
python3 examples/im2/scripts/build_meshes.py --verify   # check the shipped tags
python3 examples/im2/scripts/build_meshes.py --force    # re-mesh (needs gmsh)
python3 examples/im2/scripts/gen_survey.py
```

Re-meshing changes the discretisation, so the reference metrics must be
regenerated by running the pipeline again.

## Expected results

The inversion reaches the noise floor and returns four distinct anomalies, each
with the right polarity and position. Full values, with acceptance tolerances,
are in `reference/reference_metrics.md` and `reference/reference_metrics.json`.

| quantity               | value                                                         |
|------------------------|---------------------------------------------------------------|
| RMS of the true model  | 0.9864                                                        |
| initial RMS            | 7.0604                                                        |
| **final RMS**          | **1.0491** (termination `CONVERGED_RMSTOL`, 105 L-BFGS steps) |
| background resistivity | 100 Ω·m (held fixed)                                          |
| cross-contamination    | **0** wrong-sign cells of the 6330 inside the true bodies     |
| conductors C1 / C2     | mean 28.3 / 25.7 Ω·m (true 10), offsets 8.3 / 23.4 m          |
| resistors R1 / R2      | mean 178.5 / 178.2 Ω·m (true 1000), offsets 73.7 / 7.8 m      |

`analyze_inversion.py` prints these and reports **PASS** when they fall
within tolerance.  The metrics come from a run on **112 MPI tasks**.

Three things make this case harder than `examples/im1`, and the analysis reports
each separately - **polarity** (each quadrant must come back with the right
sign), **cross-contamination** (a conductor must not leak into a neighbouring
resistor's quadrant) and **separation** (the four bodies must stay distinct
across their 100 m gaps). All three hold.

### Convergence record

`reference/` also ships the optimisation history of that run, so a new run can
be compared iteration by iteration and not only on its final metrics:

| file              | content                                                         |
|-------------------|-----------------------------------------------------------------|
| `lbfgs_log.txt`   | L-BFGS trace, one row per accepted step (iterations 0 to 105)   |
| `rms_history.txt` | `/rms_history`, one row per objective-gradient evaluation (112) |

Each row of `lbfgs_log.txt` gives the cumulative objective-gradient evaluations,
RMS, regularization term `λΦm`, relative gradient norm `|g|/|x|`, step length,
KSP iteration range and wall time, and the file ends with the exit reason, the
stopping test and the iteration and evaluation counts. `rms_history.txt`
includes the 6 rejected line-search trials.

Both are extracted from a finished run rather than written by the kernel:

```bash
sed -n '/^   Iter Evals/,/^   Evaluations/p' \
    examples/im2/outputs/im_<jobid>.out > examples/im2/reference/lbfgs_log.txt

python3 - <<'PY'
import h5py, numpy as np
with h5py.File('examples/im2/outputs/responses_im_p2.h5') as f:
    r = np.asarray(f['rms_history'])          # (N, 2) on a complex PETSc build
    r = r[:, 0] if r.ndim == 2 else r
np.savetxt('examples/im2/reference/rms_history.txt', np.c_[np.arange(1, len(r) + 1), r],
           fmt='%7d%13.6f', header='   Eval          RMS', comments='')
PY
```

Regenerate both whenever the reference metrics are regenerated: the step count
in `lbfgs_log.txt` must match `iterations` in `reference_metrics.json`, and the
row count in `rms_history.txt` must match `objgrad_evaluations`.

## Outputs

All generated files are written under `outputs/`, which is **git-ignored** and
absent from a fresh clone: the scripts create it on demand and every workflow
above recreates what it needs. Nothing in it is required to run the benchmark -
the committed inputs are `geometry/`, `survey/`, `configs/` and `reference/`.
Treat the directory as disposable; what is worth keeping from a run is the
convergence record, which is extracted from it and committed under `reference/`.

> **Re-running overwrites.** `run_fm.slurm` and `run_im.slurm` write
> to fixed filenames (`responses_fm_f<F>_p2.h5`, `responses_im_p2.h5`), so
> launching them again replaces whatever is already there. Because the directory
> is git-ignored, an overwritten run is not recoverable and can only be
> reproduced by rerunning the pipeline - the forward stage alone is 7 cluster
> jobs.
>
> Clear `snapshots/` before each run. A run on N ranks writes
> `iterNNNN_r0000..r{N-1}`, so pieces written by a run on more ranks are not
> overwritten and remain alongside the new ones; the analyzer globs
> `iter*_r*.vtu` and would read both models as one. The `.pvtu` manifest gives
> the piece count a snapshot should have.

| file                              | description                                                  |
|-----------------------------------|--------------------------------------------------------------|
| `input_fm_f<F>.h5`, `input_im.h5` | preprocessed solver input bundles                            |
| `responses_fm_f<F>_p2.h5`         | forward responses (per frequency)                            |
| `observed.h5`                     | synthetic noisy observations                                 |
| `responses_im_p2.h5`              | recovered model: conductivity, log-perturbation, rms_history |
| `snapshots/`                      | model per accepted step: `iterNNNN.pvtu` + per-rank `.vtu`   |
| `im_*.out`, `fm_*.out`            | solver logs                                                  |

## Notes

- **Solvers.** Both kernels support both solver families, and the choice is
  yours through the parameter file. The case options and the solver live in
  separate `-options_file`s, so one preset drives either kernel:
  - `configs/solver_bddc.txt` - PCBDDC iterative (FGMRES on the MATIS operator
    with the exact Nédélec discrete-gradient coarse space).
  - `configs/solver_mumps.txt` - MUMPS direct (LU).

  ```bash
  ../../build/fm.csem -input_filename outputs/input_fm_f1.h5 -options_file configs/solver_bddc.txt
  ../../build/im.csem -options_file configs/params_im.txt    -options_file configs/solver_mumps.txt
  ```

  What actually selects the family is the operator type, which each preset sets:
  `-dm_mat_type is` builds the MATIS operator PCBDDC needs, `-dm_mat_type aij`
  the AIJ operator MUMPS can factorize. Omitting the preset leaves `im.csem` on
  its MATIS default.

  For `im.csem` the preset governs **both** solves: the adjoint system
  `A_f·nx = nB` uses the same operator as the forward solve, so it reuses the
  same KSP - one preconditioner setup (or one factorization) serves both. Each
  objective-gradient evaluation therefore issues `2 × n_freq` linear solves.

  Run iteratively, keep `-ksp_rtol` tight: the adjoint solution enters the
  objective gradient, so a loose solve degrades the L-BFGS search directions,
  not just the field. `configs/params_im.txt` sets `1.0e-10`, the value at
  which the two presets agree to every printed digit in `examples/im1`.
  Loosening it is a legitimate speed/accuracy trade - re-check ‖g‖ against
  `solver_mumps.txt` before adopting a looser value.

  `run_fm.slurm` takes a `SOLVER` variable (`SOLVER=bddc` / `SOLVER=mumps`)
  and defaults to `bddc`. `SOLVER=mumps` remains a useful exact cross-check when
  regenerating reference data.
- **Reproducibility.** The noise seed is fixed (20260922) by
  `scripts/make_observations.sh` and recorded in the `observed.h5` it writes, so
  the same responses always yield the same observations. It differs from the
  seed `examples/im1` uses, so the two cases carry independent noise
  realisations. Every input is regenerable from this directory:
  `build_meshes.py` rebuilds both meshes from `mesh.geo`, `gen_survey.py` the
  survey, and `build_bundles.sh` the bundles. Note that the forward responses
  themselves carry the iterative solver's tolerance (`-ksp_rtol`, 1e-5 by
  default), which is ~1000x below the 1 % noise level, so a repeated pipeline
  run reproduces the reference metrics within their tolerances rather than
  bit-for-bit.
- **Mesh tagging.** `INVERT`, `COND` and `RES` are assigned by re-tagging cells
  after meshing, not by embedding volumes. Re-tagging changes only material
  labels, so the topology - and with it the discrete gradient PCBDDC relies on -
  is untouched; four embedded bodies would add edges and faces and can trip the
  preconditioner. See `scripts/build_meshes.py`.
- **Regularization.** Tikhonov weight λ = 0.1. Only the `INVERT` region is
  updated; air and background are held fixed (4th column of `sigmas_im.txt`).

