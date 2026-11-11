# Unit cube test dataset

The dataset of test levels 4 and 5 (see the [test suite](../../tests/README.md)):
a homogeneous unit cube with one electric dipole and three receivers. It is
small enough to run assembly, solve and interpolation at orders 1 to 3 in CI.

## Physical setup

| Property     | Value                                                             |
|--------------|-------------------------------------------------------------------|
| Domain       | unit cube `[0,1]³`, one material                                   |
| Conductivity | isotropic `σ = 1.0` S/m (`survey/sigmas.txt`)                      |
| Source       | one electric dipole at the center `(0.5, 0.5, 0.5)`, `2 Hz`, unit current/length, dip 0, azimuth 0 (`survey/sources.txt`) |
| Receivers    | three points along `y = z = 0.25`, `x ∈ {0.25, 0.5, 0.75}` (`survey/receivers.txt`) |
| Mesh         | structured (transfinite) unit cube: 343 vertices, 1854 edges, 2808 faces, 1296 tetrahedra (order-independent) |

## Directory layout

```
fm_csem_unit_cube/
├── README.md
├── geometry/          mesh source
│   └── mesh.geo           structured unit cube, one material (committed source for the bundle)
├── survey/            transmitter, receivers, conductivities
│   ├── sources.txt        dipole definition (input to preprocess)
│   ├── receivers.txt      receiver positions (input to preprocess)
│   └── sigmas.txt         per-material conductivity table (input to preprocess)
├── configs/           solver options
│   └── params_p1.txt      BDDC run at order 1 (manual and Extrae runs)
├── scripts/           fixture drivers
│   ├── build_bundles.sh   gmsh + preprocess --> outputs/input.h5
│   └── regen_golden.sh regenerate the goldens in reference/
└── reference/         committed exact-LU golden responses
    └── responses_p{1,2,3}.h5
```

Generated files go to `outputs/`, which is git-ignored and created by
`build_bundles.sh`. The tests run every order on the same `outputs/input.h5`
with the kernel's `-order` option.

## Regenerating the input bundle

`build_bundles.sh` meshes `geometry/mesh.geo` with gmsh and runs
`utils/preprocess.py`, as CI does before the e2e tests. The mesh is
transfinite, so every gmsh version gives the same mesh. From the repository
root:

```bash
bash examples/fm_csem_unit_cube/scripts/build_bundles.sh
```

Under the `petgem-env` container:

```bash
docker run --rm -u $(id -u):$(id -g) -v "$PWD":/workspace -w /workspace petgem-env:latest \
    bash examples/fm_csem_unit_cube/scripts/build_bundles.sh
```

## Regenerating the goldens

The goldens are serial LU solves for orders 1, 2 and 3. Regenerate them, from
a trusted build, only after an intended change of the forward result:

```bash
bash examples/fm_csem_unit_cube/scripts/regen_golden.sh
```
