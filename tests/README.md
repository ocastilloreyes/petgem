# PETGEM test suite

Verification tests for the kernels **`fm.csem`**, **`fm.mt`**, **`im.csem`** and **`fm.tdcsem`**. The
production sources are never modified by the suite: the C harnesses link
against them and the end-to-end tests drive the built binaries.

| Level | What it verifies | Orders | How |
|------:|------------------|:------:|-----|
| 1 | Basis functions | 1–6 | C harness `test_basis.c` |
| 2 | DOF ordering | 1–6 | C harness `test_dofs.c` |
| 3 | Element matrices | 1–6 | C harness `test_elements.c` |
| 4 | Global assembly | 1–3 | `fm.csem` on `examples/fm_csem_unit_cube` (assembly only) |
| 5 | Basic physics | 1–3 | `fm.csem` on `examples/fm_csem_unit_cube`, direct LU vs goldens |
| 6 | Order of accuracy (MMS) | 1, 3, 6 | `build/mms` (`make mms`) vs golden errors |
| MT | Boundary tools, 1D field, boundary RHS | 1–6 | C harnesses `test_boundary.c`, `test_mt1d.c`, `test_mt_rhs.c` |
| MT | Half-space | 2 | `fm.mt` on 1 and 3 ranks |
| TD-CSEM | Half-space step-on | 1–2 | `fm.tdcsem` vs empymod (order 2); 1 and 3 ranks (order 1) |
| IM | Reduced im_csem_block inversion | 1 | `im.csem`, 2 L-BFGS iterations vs `im/reference/im_golden.json` (rtol 1e-8) |
| IM | Data components | 1 | `im.csem -im_gradient_check` on Hy and Ex+Zxy (central differences, error ~ eps²); 1 L-BFGS iteration on Zxy |

Levels 1–3 cover every order cheaply. Levels 4–5 stop at order 3 because
orders 1, 2 and 3 already exercise every DOF class (edge, face, interior);
higher orders only add more DOFs of the same classes.

## Layout

```
tests/
├── conftest.py            fixtures: harness build, binary lookup, fm.csem runner
├── fmcsem_testlib.py      shared constants and helpers
├── csrc/                  C harnesses (petgem_test.h, box_mesh.h, test_*.c)
├── unit/test_harnesses.py runs every C harness per order and rank count
├── unit/test_layers.py    layered architecture of src/ and include/
├── e2e/                   levels 4-5
├── mms/                   level 6: MMS program (mms.c, mms.h), mesh.geo, inputs, build_bundle.sh,
│                          regen_golden.sh, reference/mms_golden.json
├── mt/                    MT half-space: mesh.geo, inputs, build_bundle.sh
├── im/                    reduced im_csem_block inversion (Ex, Hy, Zxy data): build_bundle.sh, reference/im_golden.json
├── tdcsem/                TD-CSEM half-space: mesh.geo, inputs, build_bundle.sh, make_reference.py,
│                          reference/empymod_golden.json
└── extrae/                Extrae trace configuration
```

The level 4–5 dataset lives in `examples/fm_csem_unit_cube`; its bundle is built with
`bash examples/fm_csem_unit_cube/scripts/build_bundles.sh`.

## Validation criteria

* **Level 1**: DOF-count closed forms (`p(p+2)(p+3)/2` for H(curl),
  `(p+1)(p+2)(p+3)/6` for H1); constant-vector reproduction of the Nédélec
  basis; H1 partition of unity and zero gradient sum.
* **Level 2**: per-entity DOF partition from the DOF nodes; nodes inside the
  reference tetrahedron; edge DOFs first.
* **Level 3**: finite, symmetric `Me` and `Ke`; `Me` positive definite, `Ke`
  positive semidefinite; `Ke·G = 0`; positive cell Jacobian.
* **Level 4**: no `Error element` from the gradient-kernel check; unit-cube
  topology; DOF-per-entity closed forms; `0 < global DOFs <= total DOFs`.
* **Level 5**: fields finite and non-trivial; relative difference to the LU
  goldens `< 1e-3`.
* **Level 6**: the four MMS error norms and the DOF count against
  `mms/reference/mms_golden.json` within `atol + rtol·|golden|`
  (`rtol = 1e-3`, `atol = 1e-8`); residual `< 1e-10`.
* **MT half-space**: `ρ_xy = ρ_yx = 100 Ω·m` (1 %), `φ_xy = 135°`,
  `φ_yx = -45°` (0.5°), vanishing `Zxx`, `Zyy` and tipper, same impedance on
  1 and 3 ranks.

Levels 5 and 6 use a direct LU solve (MUMPS), so they do not depend on the
convergence of the iterative solver.

## Running

```bash
pytest tests/unit                                   # C harnesses (PETSc + mpicc)

bash examples/fm_csem_unit_cube/scripts/build_bundles.sh
PETGEM_FM_CSEM=build/fm.csem PETGEM_NP=2 pytest tests/e2e

bash tests/mms/build_bundle.sh 4
PETGEM_MMS=build/mms PETGEM_NP=2 pytest tests/mms

bash tests/mt/build_bundle.sh
PETGEM_FM_MT=build/fm.mt pytest tests/mt

bash tests/im/build_bundle.sh
PETGEM_IM_CSEM=build/im.csem pytest tests/im

bash tests/tdcsem/build_bundle.sh
PETGEM_FM_TDCSEM=build/fm.tdcsem pytest tests/tdcsem

pytest                                              # everything available
```

Tests whose toolchain, binary or bundle is missing are skipped.

## Regenerating the goldens

Only after an intentional change of the forward result, from a trusted build:

```bash
bash examples/fm_csem_unit_cube/scripts/regen_golden.sh  # levels 4-5
bash tests/mms/regen_golden.sh                           # level 6
```

Review the diff of the regenerated files before committing them.

## CI

[`ci-develop.yml`](../.github/workflows/ci-develop.yml) builds `fm.csem`,
`im.csem`, `fm.mt`, `petgem` and `build/mms` once and passes them to
[`tests-fm-csem.yml`](../.github/workflows/tests-fm-csem.yml) (harnesses,
levels 4–5, level 6),
[`tests-fm-mt.yml`](../.github/workflows/tests-fm-mt.yml) (MT half-space) and
[`tests-im-csem.yml`](../.github/workflows/tests-im-csem.yml) (reduced inversion).
All jobs run in the CI image built by
[`image.yml`](../.github/workflows/image.yml).

The full inversion benchmarks in `examples/im_csem_block` and
`examples/im_csem_checkerboard` are run by hand, not in CI.
