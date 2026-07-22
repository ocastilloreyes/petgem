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
| Conductivity | isotropic `σ = 1.0` S/m (`survey/sigmas.txt`)                      |
| Source       | one electric dipole at the centre `(0.5, 0.5, 0.5)`, `2 Hz`, unit current/length, dip 0, azimuth 0 (`survey/sources.txt`) |
| Receivers    | three points along `y = z = 0.25`, `x ∈ {0.25, 0.5, 0.75}` (`survey/receivers.txt`) |
| Mesh         | structured (transfinite) unit cube: 343 vertices, 1854 edges, 2808 faces, 1296 tetrahedra (order-independent) |

## Directory layout

```
unit_cube/
├── README.md
├── geometry/          mesh source
│   └── mesh.geo           structured unit cube, one material (committed source for the bundle)
├── survey/            transmitter, receivers, conductivities
│   ├── sources.txt        dipole definition (input to preprocess)
│   ├── receivers.txt      receiver positions (input to preprocess)
│   └── sigmas.txt         per-material conductivity table (input to preprocess)
├── configs/           solver options
│   └── params_p1.txt      production MATIS + PCBDDC solver (level-5 / Extrae)
├── reference/         committed exact-LU golden responses
│   └── responses_p{1,2,3}.h5
├── scripts/           fixture drivers
│   ├── build_bundles.sh   gmsh + preprocess → outputs/input.h5
│   └── regen_reference.sh regenerate the goldens in reference/
└── outputs/           generated files (bundle, mesh, responses) — git-ignored
```

The polynomial order is selected at run time with `-order N` (which also
bypasses the bundle's stored order), so all six orders 1–6 run against the one
`outputs/input.h5` — no per-order meshes or bundles are needed.

General, reusable tools live in the PETGEM `utils/` package, not here
(`utils/preprocess.py` builds the solver input bundle from a mesh + survey).

## Regenerating the input bundle

`outputs/input.h5` is a **generated artifact** (git-ignored): it is rebuilt from
the committed `geometry/mesh.geo` + `survey/*.txt` inputs through the same
`gmsh → preprocess` workflow as `examples/fm_model` (this is exactly what CI
does before the e2e / Extrae jobs). The structured (transfinite) `mesh.geo` is
deterministic across gmsh versions, so the bundle is reproducible. Run from the
repository root:

```bash
bash examples/unit_cube/scripts/build_bundles.sh 1
```

Under the `petgem-env` container:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace petgem-env:latest \
    bash examples/unit_cube/scripts/build_bundles.sh 1
```

## Regenerating the golden references

The goldens are exact serial LU solves of the current code. Regenerate them
from a trusted build whenever the forward result legitimately changes — this
runs `fm.csem` on the bundle for orders 1, 2, 3 with a direct LU solve, writing
into `reference/`:

```bash
bash examples/unit_cube/scripts/regen_reference.sh
```
