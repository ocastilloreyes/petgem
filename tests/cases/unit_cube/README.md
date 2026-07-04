# `unit_cube` — the FM-CSEM test dataset

The single, canonical dataset the whole [FM-CSEM test suite](../../README.md)
is built around. A homogeneous unit cube with one electric dipole and three
receivers — small enough to run the full assemble → solve → interpolate
pipeline for several polynomial orders in CI, yet exercising every DOF entity
class (edge / face / interior) at orders 1–3.

## Physical setup

| Property     | Value                                                             |
|--------------|-------------------------------------------------------------------|
| Domain       | unit cube `[0,1]³`, one material                                   |
| Conductivity | isotropic `σ = 1.0` S/m (`sigmas.txt`)                             |
| Source       | one electric dipole at the centre `(0.5, 0.5, 0.5)`, `2 Hz`, unit current/length, dip 0, azimuth 0 (`sources.txt`) |
| Receivers    | three points along `y = z = 0.25`, `x ∈ {0.25, 0.5, 0.75}` (`receivers.txt`) |
| Mesh         | 365 vertices, 2092 edges, 3264 faces, 1536 tetrahedra (order-independent) |

## Files

| File               | Role                                                                         |
|--------------------|------------------------------------------------------------------------------|
| `input.h5`         | Preprocessed input bundle the tests drive `fm.csem` with (mesh + model + source/receiver geometry). |
| `params_p1.txt`    | PETSc options file for the production **MATIS + PCBDDC** solver (used by the level-5 physics test). |
| `sources.txt`      | Source definition (provenance for `input.h5`).                               |
| `receivers.txt`    | Receiver positions (provenance for `input.h5`).                              |
| `sigmas.txt`       | Per-material conductivity table (provenance for `input.h5`).                 |
| `reference/`       | Committed exact-LU golden responses for orders 1, 2, 3 (`responses_p{1,2,3}.h5`). |

The polynomial order is selected at run time with `-order N` (which also
bypasses the bundle's stored order), so all six orders 1–6 run against this one
`input.h5` — no per-order meshes or bundles are needed.

## Regenerating the golden references

The goldens are exact serial LU solves of the current code. Regenerate them
from a trusted build whenever the forward result legitimately changes — run
`fm.csem` on this bundle for orders 1, 2, 3 with a direct LU solve, writing
into `reference/`:

```bash
for N in 1 2 3; do
  build/fm.csem \
    -input_filename tests/cases/unit_cube/input.h5 -order $N \
    -output_dir tests/cases/unit_cube/reference -output_filename responses_p$N \
    -dm_mat_type aij -ksp_type preonly -pc_type lu \
    -pc_factor_mat_ordering_type nd -ksp_error_if_not_converged
done
```
