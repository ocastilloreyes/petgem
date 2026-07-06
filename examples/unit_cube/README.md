# `unit_cube` — the FM-CSEM test dataset

The single, canonical dataset the whole [FM-CSEM test suite](../../tests/README.md)
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
| Mesh         | structured (transfinite) unit cube: 343 vertices, 1854 edges, 2808 faces, 1296 tetrahedra (order-independent) |

## Files

| File               | Role                                                                         |
|--------------------|------------------------------------------------------------------------------|
| `mesh.geo`         | Gmsh geometry (structured unit cube, one material). Committed source for `input.h5`. |
| `sources.txt`      | Source definition — dipole (input to preprocess).                            |
| `receivers.txt`    | Receiver positions (input to preprocess).                                    |
| `sigmas.txt`       | Per-material conductivity table (input to preprocess).                       |
| `reference/`       | Committed exact-LU golden responses for orders 1, 2, 3 (`responses_p{1,2,3}.h5`). |
| `input.h5`         | Input bundle (mesh + model + source/receiver geometry). **Generated** by preprocess — not committed. |
| `params_p1.txt`    | PETSc options for the production **MATIS + PCBDDC** solver (level-5 test). **Generated** by preprocess — not committed. |

The polynomial order is selected at run time with `-order N` (which also
bypasses the bundle's stored order), so all six orders 1–6 run against this one
`input.h5` — no per-order meshes or bundles are needed.

## Regenerating the input bundle

`input.h5` and `params_p1.txt` are **generated artifacts** (git-ignored): they
are rebuilt from the committed `mesh.geo` + `*.txt` inputs through the same
`gmsh → preprocess` workflow as `examples/canonical_model` (this is exactly what
CI does before the e2e / Extrae jobs). The structured (transfinite) `mesh.geo`
is deterministic across gmsh versions, so the bundle is reproducible:

```bash
gmsh -3 examples/unit_cube/mesh.geo -o examples/unit_cube/mesh.msh
python3 utils/preprocess.py -mode forward -order 1 \
    -case_dir examples/unit_cube -mesh_filename mesh.msh \
    -source_filename sources.txt -receiver_filename receivers.txt \
    -sigma_file sigmas.txt -params_filename params_p1.txt
```

## Regenerating the golden references

The goldens are exact serial LU solves of the current code. Regenerate them
from a trusted build whenever the forward result legitimately changes — run
`fm.csem` on this bundle for orders 1, 2, 3 with a direct LU solve, writing
into `reference/`:

```bash
for N in 1 2 3; do
  build/fm.csem \
    -input_filename examples/unit_cube/input.h5 -order $N \
    -output_dir examples/unit_cube/reference -output_filename responses_p$N \
    -dm_mat_type aij -ksp_type preonly -pc_type lu \
    -pc_factor_mat_ordering_type nd -ksp_error_if_not_converged
done
```
