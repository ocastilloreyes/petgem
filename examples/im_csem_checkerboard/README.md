# Checkerboard CSEM inversion

Four buried cubes, two conductive and two resistive, recovered by `im.csem`
from synthetic data. The workflow, file layout, solvers and scripts are the
same as in [`examples/im_csem_block`](../im_csem_block/README.md); this file
covers only what differs.

## Model

Coordinates have **z negative downward**; `z = 0` is the air/earth interface.

| region                | resistivity | conductivity |
|-----------------------|-------------|--------------|
| air                   | -           | 1 × 10⁻⁸ S/m |
| background (earth)    | 100 Ω·m     | 0.01 S/m     |
| conductive bodies (2) | 10 Ω·m      | 0.1 S/m      |
| resistive bodies (2)  | 1000 Ω·m    | 0.001 S/m    |

Four 200 m cubes, top at z = -100 m and bottom at z = -300 m, with 100 m gaps:

| body | type      | resistivity | center (x, y)  |
|------|-----------|-------------|----------------|
| C1   | conductor | 10 Ω·m      | (+150, -150) m |
| C2   | conductor | 10 Ω·m      | (-150, +150) m |
| R1   | resistor  | 1000 Ω·m    | (+150, +150) m |
| R2   | resistor  | 1000 Ω·m    | (-150, -150) m |

Each quadrant holds one body. `analyze_inversion.py` attributes the recovered
cells of each quadrant to its body.

## Survey

As in `im_csem_block` (one x-directed dipole at (0, -4000, 0) m, 7 frequencies from 1 to
1500 Hz, order 2), except for the receivers: 441 points on a 21 × 21 grid,
x, y ∈ [-500, 500] m at 50 m spacing, to cover the four bodies.

`geometry/mesh_true.msh` has the regions `AIR`, `BG`, `INVERT`, `COND` and
`RES`.

## Running

From the repository root; the SLURM jobs are submitted from `examples/im_csem_checkerboard`.

```bash
make 

# Forward model of the true model: 7 bundles, then 7 solves
bash examples/im_csem_checkerboard/scripts/build_bundles.sh fm
(cd examples/im_csem_checkerboard && sbatch scripts/run_fm.slurm)

# Stack the responses and add 1 % noise
bash examples/im_csem_checkerboard/scripts/make_observations.sh

# Inversion from the homogeneous starting model
bash examples/im_csem_checkerboard/scripts/build_bundles.sh im
(cd examples/im_csem_checkerboard && sbatch scripts/run_im.slurm)

# Check the recovered model
python3 examples/im_csem_checkerboard/scripts/analyze_inversion.py \
    -run_dir   examples/im_csem_checkerboard/outputs \
    -reference examples/im_csem_checkerboard/reference/reference_metrics.json
```

The noise seed is 20260922, different from the one of `im_csem_block`.

## Expected results

From `reference/reference_metrics.json` (112 MPI tasks, BDDC):

| quantity               | value                                                    |
|------------------------|----------------------------------------------------------|
| RMS of the true model  | 0.9864                                                   |
| initial RMS            | 6.1086                                                   |
| **final RMS**          | **1.0494** (`CONVERGED_RMSTOL`, 90 L-BFGS steps)         |
| background resistivity | 100 Ω·m (held fixed)                                     |
| wrong-sign cells       | **0** of the 6330 inside the true bodies                 |
| conductors C1 / C2     | mean 32.1 / 29.5 Ω·m (true 10), offsets 15.2 / 22.8 m    |
| resistors R1 / R2      | mean 167.7 / 164.4 Ω·m (true 1000), offsets 20.0 / 14.2 m |

`analyze_inversion.py` reports **PASS** when the run is within the tolerances
of the JSON file. It checks polarity, wrong-sign cells and the separation of
the four bodies separately.

The 90 steps took 96 objective-gradient evaluations (5 rejected
line-search trials, plus the initial evaluation). The step count is not
reproducible to the unit: two runs on 112 tasks took 91 and 90 steps. The
tolerances allow for it.

The conductors recover more of their contrast than the resistors, as expected
for CSEM data.
