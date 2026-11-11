# Conductive block CSEM inversion

A conductive block in a resistive half-space, recovered by `im.csem` from
synthetic data. The case runs the whole inversion workflow:

```
true model --> fm.csem --> observations + 1 % noise --> im.csem --> recovered model
```

Geometry, survey, configs and the expected results are committed. The
observations are not: the forward stage produces them.

## Model

Coordinates have **z negative downward**; `z = 0` is the air/earth interface.

| region             | resistivity | conductivity |
|--------------------|-------------|--------------|
| air                | -           | 1 × 10⁻⁸ S/m |
| background (earth) | 100 Ω·m     | 0.01 S/m     |
| anomaly            | 10 Ω·m      | 0.1 S/m      |

The anomaly is a 200 m cube (8 × 10⁶ m³) centered at (0, 0, -200) m, with its
top 100 m below the surface.

## Survey

| quantity    | value                                                                  |
|-------------|------------------------------------------------------------------------|
| source      | x-directed electric dipole at (0, -4000, 0) m, unit current and length |
| receivers   | 441 (21 × 21 grid), x, y ∈ [-200, 200] m, 20 m spacing, z = 0          |
| frequencies | 1, 10, 50, 100, 300, 800, 1500 Hz                                      |
| basis order | 2                                                                      |

## Directory layout

```
im_csem_block/
├── geometry/
│   ├── mesh.geo           source of both meshes
│   ├── mesh.msh           inversion mesh: AIR, BG, INVERT
│   └── mesh_true.msh      true model:     AIR, BG, INVERT, ANOMALY
├── survey/
│   ├── sources_im.txt     inversion sources (sources_f<F>.txt: one per frequency)
│   ├── receivers.txt      441 receiver positions
│   ├── frequencies.txt    the 7 frequencies
│   └── sigmas_im.txt      starting model (sigmas_true.txt: true model)
├── configs/
│   ├── params_im.txt      inversion options (I/O, L-BFGS)
│   ├── solver_bddc.txt    BDDC solver preset (either kernel)
│   └── solver_mumps.txt   MUMPS solver preset (either kernel)
├── scripts/
│   ├── build_meshes.py       mesh.geo --> the two tagged .msh
│   ├── gen_survey.py         writes survey/*.txt
│   ├── build_bundles.sh      preprocess --> input bundles (fm or im)
│   ├── run_fm.slurm          fm.csem on the true model
│   ├── make_observations.sh  responses + noise --> observed.h5
│   ├── run_im.slurm          im.csem from the starting model
│   └── analyze_inversion.py  recovered model --> PASS/FAIL
└── reference/
    ├── reference_metrics.json  metrics and acceptance tolerances
    ├── reference_metrics.md    the same, with definitions
    ├── lbfgs_log.txt           L-BFGS log of the reference run
    └── rms_history.txt         RMS per objective-gradient evaluation
```

The bundle, observation, mesh and analysis scripts call the code in
`examples/shared/`, which `im_csem_checkerboard` uses too.
Generated files go to `outputs/` (git-ignored, created by the scripts).

## Running

From the repository root; the SLURM jobs are submitted from `examples/im_csem_block`.

```bash
make

# Forward model of the true model: 7 bundles, then 7 solves
bash examples/im_csem_block/scripts/build_bundles.sh fm
(cd examples/im_csem_block && sbatch scripts/run_fm.slurm)

# Stack the responses and add 1 % noise
bash examples/im_csem_block/scripts/make_observations.sh

# Inversion from the homogeneous starting model
bash examples/im_csem_block/scripts/build_bundles.sh im
(cd examples/im_csem_block && sbatch scripts/run_im.slurm)

# Check the recovered model
python3 examples/im_csem_block/scripts/analyze_inversion.py \
    -run_dir   examples/im_csem_block/outputs \
    -reference examples/im_csem_block/reference/reference_metrics.json
```

| step                    | output in `outputs/`                         |
|-------------------------|----------------------------------------------|
| `build_bundles.sh fm`   | `input_fm_f<F>.h5` (7)                       |
| `run_fm.slurm`          | `responses_fm_f<F>_p2.h5` (7)                |
| `make_observations.sh`  | `observed.h5`                                |
| `build_bundles.sh im`   | `input_im.h5`                                |
| `run_im.slurm`          | `responses_im_p2.h5`, `snapshots/`           |

Each script stops if the output of the previous step is missing. The Python
steps also run in the container:

```bash
docker run --rm -u $(id -u):$(id -g) -v "$PWD":/workspace -w /workspace petgem-env:latest \
    bash examples/im_csem_block/scripts/build_bundles.sh fm
```

Set your site's `--account` and `--qos`/`--partition` in the SLURM scripts
before submitting. Without SLURM, run their `mpirun` line directly.

The solves use `configs/solver_bddc.txt`; `SOLVER=mumps` switches to
`configs/solver_mumps.txt`. The presets work with either kernel, given as a
second options file:

```bash
../../build/im.csem -options_file configs/params_im.txt -options_file configs/solver_mumps.txt
```

`configs/params_im.txt` sets `-ksp_rtol 1.0e-10` because the adjoint solution
enters the gradient. If you loosen it, compare the gradient norm with a MUMPS
run first.

Clear `outputs/snapshots/` before a new inversion. A run on fewer ranks leaves
the extra pieces of the previous run, and `analyze_inversion.py` would read
them.

### Regenerating the inputs

Only needed after changing the geometry or the survey:

```bash
python3 examples/im_csem_block/scripts/build_meshes.py -verify   # check the shipped tags
python3 examples/im_csem_block/scripts/build_meshes.py -force    # re-mesh (needs gmsh)
python3 examples/im_csem_block/scripts/gen_survey.py
```

`INVERT` and `ANOMALY` are assigned by re-tagging cells after meshing, so the
two meshes share the same topology. A new mesh needs new reference metrics.

## Expected results

From `reference/reference_metrics.json` (112 MPI tasks, BDDC):

| quantity                      | value                               |
|-------------------------------|-------------------------------------|
| RMS of the true model         | 0.994                               |
| initial RMS                   | 11.058                              |
| **final RMS**                 | **1.0498**                          |
| termination                   | `CONVERGED_RMSTOL`, 91 L-BFGS steps |
| background resistivity        | 100 Ω·m                             |
| minimum recovered resistivity | 11.00 Ω·m (true 10)                 |
| conductor lateral offset      | 3.1 m                               |
| conductor vertical offset     | +36.6 m (shallower than true)       |
| conductor volume (ρ < 30 Ω·m) | 3.71 × 10⁶ m³ (true 8.0 × 10⁶)      |

`analyze_inversion.py` reports **PASS** when the run is within the tolerances
of the JSON file. The L-BFGS path depends on the domain decomposition, so
compare at 112 tasks.

The 99 objective-gradient evaluations are the initial one, the 91 accepted
steps and 7 rejected line-search trials. `/rms_history` in `responses_im_p2.h5` has
one entry per evaluation, so it is not monotone. The file stores both counts
(`num_iterations`, `num_objgrad_evaluations`).

The conductor is shallow and smaller than the truth. Source and receivers lie
on the surface 4 km apart, so the airwave dominates the data and `|Ex|` is
almost flat above 10 Hz, which limits the depth resolution.

### Convergence record

`reference/lbfgs_log.txt` has one row per accepted step (evaluations, RMS,
`λΦm`, `|g|`, `|g|/|x|`, step length, KSP iterations, wall time).
`reference/rms_history.txt` has one row per evaluation. Both are taken from a
finished run:

```bash
sed -n '/^   Iter Evals/,/^   Evaluations/p' \
    examples/im_csem_block/outputs/im_<jobid>.out > examples/im_csem_block/reference/lbfgs_log.txt

python3 - <<'PY'
import h5py, numpy as np
with h5py.File('examples/im_csem_block/outputs/responses_im_p2.h5') as f:
    r = np.asarray(f['rms_history'])          # (N, 2) on a complex PETSc build
    r = r[:, 0] if r.ndim == 2 else r
np.savetxt('examples/im_csem_block/reference/rms_history.txt', np.c_[np.arange(1, len(r) + 1), r],
           fmt='%7d%13.6f', header='   Eval          RMS', comments='')
PY
```

Regenerate them together with the reference metrics. Their row counts must
match `iterations` and `objgrad_evaluations` in `reference_metrics.json`.

## Notes

- **Noise.** `make_observations.sh` fixes the seed (20260720) and stores it in
  `observed.h5`, so the same responses always give the same observations.
- **Reproducibility.** The forward responses carry the iterative solver
  tolerance, so a repeated run matches the reference within its tolerances,
  not bit for bit.
- **Regularization.** λ = 0.1. Only `INVERT` is updated; air and background
  are fixed by the 4th column of `sigmas_im.txt`.
