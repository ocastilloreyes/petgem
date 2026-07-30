# im - CSEM inversion benchmark

The reference controlled-source electromagnetic (CSEM) inversion benchmark for
PETGEM. It demonstrates the complete inverse-modelling workflow end to end:

```
true model --> forward modelling --> synthetic observations --> noise --> inversion --> recovered model
```

A known resistivity model is forward-modelled with `fm.csem`, contaminated with
1 % Gaussian noise, and inverted with `im.csem` from a homogeneous starting
model. The inversion recovers the buried conductor, fitting the data down to the
noise level.

## Overview

This is a self-contained, reproducible example suitable for learning the
workflow, validating a PETGEM installation, and regression testing. Every input
is provided; the synthetic observations are shipped so the inversion can be run
without repeating the forward stage.

## Physical model

A resistive halfspace containing a single conductive block, beneath an air
layer. Coordinates use **z negative downward** - depth is a negative z, the
same convention as `examples/fm`; `z = 0` is the air/earth
interface.

| region | resistivity | conductivity |
|---|---|---|
| air | - | 1 × 10⁻⁸ S/m |
| background (earth) | 100 Ω·m | 0.01 S/m |
| anomaly | 10 Ω·m | 0.1 S/m |

**Anomaly geometry:** a 200 × 200 × 200 m cube (volume 8 × 10⁶ m³), centred at
(0, 0, -200) m, i.e. its top is 100 m below the surface.

## Survey

| quantity | value |
|---|---|
| source | x-directed electric dipole at (0, -4000, 0) m, unit current and length |
| receivers | 441 (21 × 21 grid), x, y ∈ [-200, 200] m, 20 m spacing, z = 0 (the interface) |
| frequencies | 1, 10, 50, 100, 300, 800, 1500 Hz |
| basis order | 2 (Nédélec edge elements) |

## Directory layout

```
im/
├── README.md
├── geometry/          meshes and their gmsh source
│   ├── mesh.geo           parametric geometry, source for both meshes
│   ├── mesh.msh           inversion mesh: AIR, BG, INVERT
│   └── mesh_true.msh      true model:     AIR, BG, INVERT, ANOMALY
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
│   ├── run_forward.slurm     2. fm.csem on the true model
│   ├── make_observations.sh  3. responses + 1 % noise --> observed.h5
│   └── run_inversion.slurm   5. im.csem from the homogeneous start
└── reference/         expected results
    ├── reference_metrics.json  machine-readable, with acceptance tolerances
    └── reference_metrics.md    human-readable
```

Everything generated lands in `outputs/`, which is git-ignored and **not present
in a fresh clone** - the scripts create it on demand. Only the *inputs* are
committed: geometry, survey, configs and the expected results.

General, reusable tools live in the PETGEM `utils/` package, not here:

- `utils/preprocess.py` - build a solver input bundle from a mesh + survey.
- `utils/make_observed.py` - assemble forward responses and add noise.
- `utils/analyze_inversion.py` - evaluate a recovered model against a reference.

## The pipeline

```
true model --> forward modelling --> synthetic observations --> noise --> inversion --> recovered model
```

There is one path through this benchmark and it runs end to end. No observed
dataset is shipped: the observations the inversion targets are the ones **your**
forward run produced, so the whole chain is exercised and reproducible.

| stage | driver | consumes | produces |
|---|---|---|---|
| 0 | `build_meshes.py`, `gen_survey.py` | `geometry/mesh.geo` | the two `.msh`, `survey/*.txt` (both committed; regenerate only if you change the model) |
| 1 | `build_bundles.sh fm` | true mesh + survey | `outputs/input_fm_f<F>.h5` (7) |
| 2 | `run_forward.slurm` | those bundles | `outputs/responses_fm_f<F>_p2.h5` (7) |
| 3 | `make_observations.sh` | those responses | `outputs/observed.h5` |
| 4 | `build_bundles.sh im` | starting mesh + observations | `outputs/input_im.h5` |
| 5 | `run_inversion.slurm` | that bundle | `outputs/responses_im_p2.h5`, `outputs/snapshots/` |
| 6 | `analyze_inversion.py` | the recovered model | PASS/FAIL against `reference/` |

Stages 1, 3 and 4 are light Python preprocessing and run anywhere the `petgem`
package is installed. Stages 2 and 5 are MPI solves, normally on a cluster.

## Execution instructions

Run all commands **from the repository root**.

```bash
make

# 1-2. Forward modelling of the true model: 7 bundles, then 7 solves
bash   examples/im/scripts/build_bundles.sh fm
sbatch examples/im/scripts/run_forward.slurm

# 3. Synthetic observations: assemble the responses and add 1 % noise
bash examples/im/scripts/make_observations.sh

# 4-5. Inversion from the homogeneous starting model
bash   examples/im/scripts/build_bundles.sh im
sbatch examples/im/scripts/run_inversion.slurm

# 6. Evaluate the recovered model
python3 utils/analyze_inversion.py \
    -run_dir   examples/im/outputs \
    -reference examples/im/reference/reference_metrics.json
```

Each stage checks that its inputs exist and tells you which earlier stage
produces them, so running them out of order fails immediately rather than
silently using stale data.

If the `petgem` package is only available in the `petgem-env` container, wrap
the Python stages:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace petgem-env:latest \
    bash examples/im/scripts/build_bundles.sh fm
```

The SLURM scripts carry portable resource requests; set your site's
`--account` and `--qos`/`--partition`, and load the PETGEM runtime environment,
before submitting. Both default to the PCBDDC iterative solver; add
`SOLVER=mumps` for the direct one.

### Regenerating the model itself

Only needed if you change the geometry or the survey:

```bash
python3 examples/im/scripts/build_meshes.py --verify   # check the shipped tags
python3 examples/im/scripts/build_meshes.py --force    # re-mesh (needs gmsh)
python3 examples/im/scripts/gen_survey.py
```

Re-meshing changes the discretisation, so the reference metrics must be
regenerated by running the pipeline again.

## Expected results

The inversion converges to the noise floor and recovers the conductor. Full
values, with acceptance tolerances, are in `reference/reference_metrics.md` and
`reference/reference_metrics.json`.

| quantity | value |
|---|---|
| RMS of the true model | 0.994 |
| initial RMS | 11.795 |
| **final RMS** | **1.0497** (termination `CONVERGED_RMSTOL`, 103 L-BFGS steps) |
| background resistivity | 100 Ω·m |
| peak recovered resistivity | 9.11 Ω·m (true 10) |
| conductor lateral offset | 3.0 m |
| conductor volume (ρ < 30 Ω·m) | 4.03 × 10⁶ m³ |

`analyze_inversion.py` prints these and reports **PASS** when they fall within
tolerance. RMS decreases monotonically over the accepted L-BFGS steps; the
`/rms_history` dataset also records rejected line-search trials, so it is not
monotone. The 103 accepted steps took 112 objective-gradient evaluations, and it
is that evaluation count that indexes `/rms_history`. `responses_im_p2.h5` stores
both counters (`num_iterations`, `num_objgrad_evaluations`) and
`analyze_inversion.py` prints them separately.

The recovered conductor is biased shallow (+36.3 m, shallower than true) and recovers about half the
true volume. That is inherent to the acquisition rather than to the solver:
source and receivers both lie on the air/earth interface at a 4 km offset, so
the airwave dominates the recorded field and `|Ex|` is nearly flat above 10 Hz.
The seven frequencies consequently carry less independent depth information than
their range suggests.

## Outputs

All generated files are written under `outputs/`, which is **git-ignored**: a
fresh clone starts empty there, and every workflow above recreates what it
needs. Nothing in `outputs/` is required to run the benchmark - the committed
inputs are `geometry/`, `survey/`, `configs/` and `reference/`.

> **Re-running overwrites.** `run_forward.slurm` and `run_inversion.slurm` write
> to fixed filenames (`responses_fm_f<F>_p2.h5`, `responses_im_p2.h5`), so
> launching them again replaces whatever is already there. Because the directory
> is git-ignored, an overwritten run is not recoverable and can only be
> reproduced by rerunning the pipeline - the forward stage is 7 cluster jobs,
> and the reference inversion took 01:09 on 224 MPI tasks. 

| file | description |
|---|---|
| `input_fm_f<F>.h5`, `input_im.h5` | preprocessed solver input bundles |
| `responses_fm_f<F>_p2.h5` | forward responses (per frequency) |
| `observed.h5` | synthetic noisy observations |
| `responses_im_p2.h5` | recovered model: conductivity, log-perturbation, rms_history |
| `snapshots/` | recovered-model snapshots (`iterNNNN.pvtu` + per-rank `iterNNNN_rRRRR.vtu`) for ParaView and for `analyze_inversion.py` |
| `im_*.out`, `fm_*.out` | solver logs |

## Notes

- **Solvers.** Both kernels support both solver families, and the choice is
  yours through the parameter file. The case options and the solver live in
  separate `-options_file`s, so one preset drives either kernel:
  - `configs/solver_bddc.txt` - PCBDDC iterative (FGMRES on the MATIS operator
    with the exact Nédélec discrete-gradient coarse space).
  - `configs/solver_mumps.txt` - MUMPS direct (LU).

  ```bash
  build/fm.csem -input_filename outputs/input_fm_f1.h5 -options_file configs/solver_bddc.txt
  build/im.csem -options_file configs/params_im.txt    -options_file configs/solver_mumps.txt
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
  not just the field. At `-ksp_rtol 1.0e-10` the two presets agree to every
  printed digit of RMS, objective and ‖g‖ over the first L-BFGS iterations, so
  the iterative path reproduces the exact one at this tolerance. Loosening it is
  a legitimate speed/accuracy trade - re-check ‖g‖ against `solver_mumps.txt`
  before adopting a looser value.

  `run_forward.slurm` takes a `SOLVER` variable (`SOLVER=bddc` / `SOLVER=mumps`)
  and defaults to `bddc`. `SOLVER=mumps` remains a useful exact cross-check when
  regenerating reference data.
- **Reproducibility.** The noise seed is fixed (20260720) by
  `scripts/make_observations.sh` and recorded in the `observed.h5` it writes, so
  the same responses always yield the same observations. Every input is
  regenerable from this directory: `build_meshes.py` rebuilds both meshes from
  `mesh.geo`, `gen_survey.py` the survey, and `build_bundles.sh` the bundles.
  Note that the forward responses themselves carry the iterative solver's
  tolerance (`-ksp_rtol`, 1e-5 by default), which is ~1000x below the 1 % noise
  level, so a repeated pipeline run reproduces the reference metrics within
  their tolerances rather than bit-for-bit.
- **Mesh tagging.** `INVERT` and `ANOMALY` are assigned by re-tagging cells
  after meshing, not by embedding volumes. Re-tagging changes only material
  labels, so the topology - and with it the discrete gradient PCBDDC relies on -
  is untouched; an embedded body would add edges and faces and can trip the
  preconditioner. See `scripts/build_meshes.py`.
- **Regularization.** Tikhonov weight λ = 0.1. Only the `INVERT` region is
  updated; air and background are held fixed (4th column of `sigmas_im.txt`).
