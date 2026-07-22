# im_model - CSEM inversion benchmark

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
layer. Coordinates use **z positive downward**; `z = 0` is the air/earth
interface.

| region | resistivity | conductivity |
|---|---|---|
| air | - | 1 × 10⁻⁸ S/m |
| background (earth) | 100 Ω·m | 0.01 S/m |
| anomaly | 10 Ω·m | 0.1 S/m |

**Anomaly geometry:** a 200 × 200 × 200 m cube (volume 8 × 10⁶ m³), centred at
(0, 0, 200) m, i.e. its top is 100 m below the surface.

## Survey

| quantity | value |
|---|---|
| source | x-directed electric dipole at (0, −4000, 0) m, unit current and length |
| receivers | 441 (21 × 21 grid), x, y ∈ [−200, 200] m, 20 m spacing, z = 0 |
| frequencies | 1, 10, 50, 100, 300, 800, 1500 Hz |
| basis order | 2 (Nédélec edge elements) |

## Directory layout

```
im_model/
├── README.md
├── geometry/          meshes and their gmsh source
│   ├── im_model.geo      homogeneous inversion mesh (source)
│   ├── im_model.msh      inversion mesh:  AIR, BG, INVERT
│   └── im_true.msh       true model:      AIR, BG, INVERT, ANOMALY
├── survey/            transmitters, receivers, frequencies, conductivities
│   ├── receivers.txt      sources_im.txt      sources_f<F>.txt
│   ├── frequencies.txt     sigmas_im.txt      sigmas_true.txt
├── configs/           solver / inversion options
│   ├── params_fm_f<F>.txt  forward, one per frequency (PCBDDC)
│   └── params_im.txt       inversion (MUMPS)
├── scripts/           benchmark-specific drivers
│   ├── build_meshes.py     mesh im_model.geo + tag INVERT/ANOMALY → the .msh
│   ├── gen_survey.py       regenerate the survey files
│   ├── build_bundles.sh    preprocess meshes+survey → solver input bundles
│   ├── run_forward.slurm   run fm.csem
│   └── run_inversion.slurm run im.csem
├── reference/         provided data and expected results
│   ├── observed.h5         synthetic noisy observations
│   ├── reference_metrics.json   machine-readable expected results
│   └── reference_metrics.md      human-readable expected results
└── outputs/           generated files (input bundles, responses, logs, VTU)
```

General, reusable tools live in the PETGEM `utils/` package, not here:

- `utils/preprocess.py` - build a solver input bundle from a mesh + survey.
- `utils/make_observed.py` - assemble forward responses and add noise.
- `utils/analyze_inversion.py` - evaluate a recovered model against a reference.

## Workflow

| step | action | tool | outputs |
|---|---|---|---|
| 0 | meshes (shipped) | `build_meshes.py` | `geometry/im_model.msh`, `geometry/im_true.msh` |
| 1 | true model | `geometry/im_true.msh`, `survey/` | - |
| 2 | forward modelling | `build_bundles.sh fm` → `run_forward.slurm` | `outputs/responses_fm_f<F>_p2.h5` |
| 3 | add noise | `utils/make_observed.py` | `outputs/observed.h5` |
| 4 | observed data | (output of step 3) | `outputs/observed.h5` |
| 5 | inversion | `build_bundles.sh im` → `run_inversion.slurm` | `outputs/responses_im_p2.h5` |
| 6 | evaluate | `utils/analyze_inversion.py` | metrics report |

Preprocessing (`build_bundles.sh`, `make_observed.py`) runs anywhere the
`petgem` Python package is installed. The solvers (`fm.csem`, `im.csem`) run
under MPI, typically on a cluster.

## Execution instructions

Run all commands **from the repository root**.

### Quick start (inversion only)

Uses the shipped `reference/observed.h5`, so no forward run is required.

```bash
# 1. Build the inverse input bundle (needs the petgem package; e.g. via docker)
bash examples/im_model/scripts/build_bundles.sh im

# 2. Run the inversion
sbatch examples/im_model/scripts/run_inversion.slurm

# 3. Evaluate the recovered model
python3 utils/analyze_inversion.py \
    -run_dir   examples/im_model/outputs \
    -reference examples/im_model/reference/reference_metrics.json
```

If the `petgem` package is only available in the `petgem-env` container, wrap
the preprocessing step:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace petgem-env:latest \
    bash examples/im_model/scripts/build_bundles.sh im
```

### Full workflow (from scratch)

```bash
# (optional) check the shipped meshes carry the expected tags
python3 examples/im_model/scripts/build_meshes.py --verify

# (optional) regenerate the meshes from im_model.geo (needs gmsh; byte-identical
# under the petgem-env image, see the script header before using --force)
python3 examples/im_model/scripts/build_meshes.py --force

# (optional) regenerate the survey definition
python3 examples/im_model/scripts/gen_survey.py

# 1-2. Forward: build 7 bundles, then run fm.csem (one task per frequency)
bash   examples/im_model/scripts/build_bundles.sh fm
sbatch examples/im_model/scripts/run_forward.slurm

# 3-4. Assemble the forward responses and add 1% noise
python3 utils/make_observed.py \
    -case_dir examples/im_model \
    -pattern  "outputs/responses_fm_f{freq}_p2.h5" \
    -freqs    1,10,50,100,300,800,1500 \
    -seed     20260720 -error_level 0.01 \
    -out      outputs/observed.h5

# 5. Inversion (build the bundle from the just-made observations, then run)
OBSERVED=outputs/observed.h5 \
    bash examples/im_model/scripts/build_bundles.sh im
sbatch examples/im_model/scripts/run_inversion.slurm

# 6. Evaluate
python3 utils/analyze_inversion.py \
    -run_dir   examples/im_model/outputs \
    -reference examples/im_model/reference/reference_metrics.json
```

The SLURM scripts carry portable resource requests; set your site's
`--account` and `--qos`/`--partition`, and load the PETGEM runtime environment,
before submitting.

## Expected results

The inversion converges to the noise floor and recovers the conductor. Full
values, with acceptance tolerances, are in `reference/reference_metrics.md` and
`reference/reference_metrics.json`.

| quantity | value |
|---|---|
| RMS of the true model | ≈ 1.0 |
| initial RMS | 11.35 |
| **final RMS** | **1.05** (termination `CONVERGED_RMSTOL`, ~90 iterations) |
| background resistivity | 100 Ω·m |
| peak recovered resistivity | ≈ 11 Ω·m (true 10) |
| conductor lateral offset | 6.9 m |
| conductor volume (ρ < 30 Ω·m) | 3.5 × 10⁶ m³ |

`analyze_inversion.py` prints these and reports **PASS** when they fall within
tolerance. RMS decreases monotonically at every iteration.

## Outputs

All generated files are written under `outputs/`, which is **git-ignored**: a
fresh clone starts empty there, and every workflow above recreates what it
needs. Nothing in `outputs/` is required to run the benchmark - the committed
inputs are `geometry/`, `survey/`, `configs/` and `reference/`.

> **Re-running overwrites.** `run_forward.slurm` and `run_inversion.slurm` write
> to fixed filenames (`responses_fm_f<F>_p2.h5`, `responses_im_p2.h5`), so
> launching them again replaces whatever is already there. Because the directory
> is git-ignored, an overwritten run is not recoverable and can only be
> reproduced by rerunning the simulation - the forward stage is 7 cluster jobs,
> and the reference inversion took 01:40 on 336 MPI tasks. Copy any run you want
> to keep somewhere outside `outputs/` first.

| file | description |
|---|---|
| `input_fm_f<F>.h5`, `input_im.h5` | preprocessed solver input bundles |
| `responses_fm_f<F>_p2.h5` | forward responses (per frequency) |
| `observed.h5` | synthetic noisy observations |
| `responses_im_p2.h5` | recovered model: conductivity, log-perturbation, rms_history |
| `*_iter<N>_p<R>.vtu` | recovered-model snapshots (with geometry) for visualisation and analysis |
| `im_*.out`, `fm_*.out` | solver logs |

## Notes

- **Solvers.** The forward stage uses an iterative solver (PCBDDC on the H(curl)
  operator with the exact Nédélec discrete-gradient coarse space). The inversion
  uses the MUMPS direct solver, which gives exact forward and adjoint solves -
  and therefore an exact gradient solution.
- **Reproducibility.** The noise seed is fixed (20260720) and recorded in
  `observed.h5`; the recovered model is independent of the MPI rank count.
  Every input is regenerable from this directory: `build_meshes.py` rebuilds
  both meshes from `im_model.geo`, `gen_survey.py` the survey, and
  `build_bundles.sh` the solver bundles.
- **Mesh tagging.** `INVERT` and `ANOMALY` are assigned by re-tagging cells
  after meshing, not by embedding volumes. Re-tagging changes only material
  labels, so the topology - and with it the discrete gradient PCBDDC relies on -
  is untouched; an embedded body would add edges and faces and can trip the
  preconditioner. See `scripts/build_meshes.py`.
- **Regularization.** Tikhonov weight λ = 0.1. Only the `INVERT` region is
  updated; air and background are held fixed (4th column of `sigmas_im.txt`).
