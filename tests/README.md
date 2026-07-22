# PETGEM FM-CSEM test suite

Numerical-correctness and implementation-verification tests for the **forward
CSEM kernel (`fm.csem`)**. Levels 1–5 are built around the
[`examples/unit_cube`](../examples/unit_cube) dataset (a homogeneous unit cube,
one 2 Hz electric dipole at the centre, three receivers); level 6 adds a
self-contained **Method of Manufactured Solutions** order-of-accuracy check on
the `[0,1]³` unit cube ([`tests/mms/`](mms/)).

The production C implementation is treated as correct and is **never modified**
by the suite; the tests link against, or drive, the unchanged sources and
binary.

## Design: progressively stronger verification, fast + robust

The suite is organised into six levels of increasing scope. The key design
choice — driven by the fact that the fixed 1536-cell mesh makes the *full
pipeline* expensive at high order, while the *finite-element core* is cheap at
every order — is a **fast / slow split**:

| Level | What it verifies | Orders | Speed | How |
|------:|------------------|:------:|:-----:|-----|
| 1 | Basis functions | **1–6** | fast (ms) | C harness → `fe_nedelec.c`, `fe_nodal.c` |
| 2 | DOF ordering | **1–6** | fast (ms) | C harness → `fe_nedelec.c`, `fe_nodal.c` |
| 3 | Element matrices | **1–6** | fast (ms) | C harness → `fem.c` |
| 4 | Global assembly | 1–3 | slow | `fm.csem` (assembly only) |
| 5 | Basic physics | 1–3 | slow | `fm.csem` (direct LU / MUMPS solve) |
| 6 | Order of accuracy (MMS) | 1,3,6 | slow | `fm.csem -mms` (direct LU) vs golden errors |

**Why this is complete.** Levels 1–3 exercise the FE core for *every* order
1–6. Levels 4–5 run the full assemble → solve → interpolate pipeline for orders
1, 2, 3 — which is sufficient because those three orders activate **every DOF
entity class**: order 1 = edge DOFs only, order 2 adds face DOFs, order 3 adds
interior DOFs. Orders 4–6 only add *more* DOFs of the same classes (already
covered by levels 1–3), exercising no new pipeline code path. So every order
and every code path is validated, without paying for high-order full-pipeline
solves in CI.

**Level 6** adds an independent *order-of-accuracy* check. With a manufactured
exact solution the discretisation error is known in closed form, so the L²/
H(curl) errors `fm.csem -mms` reports pin the solver's accuracy at orders 1, 3, 6
(bracketing the range) against committed goldens on the coarse N=4 mesh — a cheap
regression guard on the whole assemble → solve → error-norm path. It is fully
self-contained under `tests/mms/` and reads nothing from `paper/`.

## Directory layout

```
tests/
├── README.md                     # this document
├── conftest.py                   # fixtures: harness build, fm.csem runner, hdf5
├── fmcsem_testlib.py             # shared constants/helpers (ORDERS, PIPELINE_ORDERS, …)
├── csrc/                         # C test harnesses (test infra, link production src)
│   ├── petgem_test.h             #   check/report macros
│   ├── test_basis.c              #   level 1
│   ├── test_dofs.c               #   level 2
│   └── test_elements.c           #   level 3
├── unit/                         # pytest wrappers for the C harnesses (levels 1-3)
│   ├── test_level1_basis.py
│   ├── test_level2_dofs.py
│   └── test_level3_elements.py
├── e2e/                          # end-to-end fm.csem tests (levels 4-5)
│   ├── test_level4_assembly.py
│   └── test_level5_physics.py
├── mms/                          # self-contained MMS order-of-accuracy check (level 6)
│   ├── test_level6_mms.py        #   the level-6 test
│   ├── mesh.geo                  #   [0,1]^3 domain (the MMS mesh)
│   ├── sigmas.txt / sources.txt / receivers.txt   #   minimal preprocess stubs
│   ├── make_bundle.sh            #   mesh.geo -> input.h5 (gmsh + preprocess)
│   ├── update_golden.sh          #   regenerate reference/mms_golden.json
│   ├── mms_reference.py          #   sympy self-test of E*/curl E*/f* + exact norms
│   └── reference/
│       └── mms_golden.json       #   committed golden errors (p=1,3,6 @ N=4)
└── extrae/                       # Extrae trace config for the CI extrae-smoke job
    ├── extrae.xml
    ├── petgem_functions.cfg
    └── petgem_labels.txt
```

The dataset the e2e / extrae jobs drive lives under `examples/unit_cube` (a tiny
homogeneous cube; see its README) — the `tests/` tree otherwise holds only test
code. Its `outputs/input.h5` bundle is regenerated from the committed
`geometry/mesh.geo` + `survey/*.txt` via the same `gmsh → preprocess` pipeline
as the other cases in `examples/` (`bash examples/unit_cube/scripts/build_bundles.sh`).
Level 6's
`tests/mms/` package is the one exception: it carries its own `[0,1]³` mesh and
preprocess stubs so the MMS check runs with **no `paper/` (or `examples/`)
dependence** — its N=4 bundle is built on the fly by `make_bundle.sh`.

## Test list & validation criteria

### Level 1 — Basis functions (`test_basis.c`, orders 1–6)
* Nédélec / H1 **DOF-count closed forms** (`order·(order+2)·(order+3)/2`,
  `(order+1)(order+2)(order+3)/6`) — expected polynomial-order behaviour.
* **H(curl) constant-vector reproduction** (Nédélec partition of unity):
  `Σ_m (c·tangent_m) φ_m(x) = c` for every constant field `c`, at 5 points.
* **H1 partition of unity** `Σ_j φ_j = 1` and **gradient consistency**
  `Σ_j ∇φ_j = 0`.

### Level 2 — DOF ordering (`test_dofs.c`, orders 1–6)
* Per-entity DOF partition (edge / face / interior) matches the closed forms,
  classified geometrically from each DOF node's barycentric zeros.
* Nodes lie inside the reference tetrahedron; tangent indices are valid.
* Edge-DOF block comes first (native ordering) — a regression guard on the
  enumeration the assembly's local-to-global mapping relies on.

### Level 3 — Element matrices (`test_elements.c`, orders 1–6, 2 cells)
* Finiteness, **symmetry** of `Me` and `Ke`.
* **`Me` positive definite**, **`Ke` positive semidefinite** (sampled).
* **De Rham identity `Ke · G = 0`** (discrete gradient in the curl kernel).
* Positively oriented, non-degenerate cell Jacobian.

### Level 4 — Global assembly (`test_level4_assembly.py`, orders 1–3)
* No `Error element` during assembly (`checkGradientKernel`: `K_e·G_e = 0` on
  every cell of the real mesh).
* Loaded mesh topology equals the fixed unit-cube topology.
* Per-order Nédélec DOF-per-entity counts follow the closed forms.
* Global operator is square, matches the RHS vector size, and free DOFs ≤ total
  DOFs (boundary DOFs are constrained out).

### Level 5 — Basic physics (`test_level5_physics.py`, orders 1–3)
* Fields present, correctly shaped `(3 receivers,)`, finite, non-trivial.
* Solved fields (**direct LU / MUMPS**, valid serial and under MPI) reproduce the
  committed exact-LU **golden references** to `rel < 1e-3` — a robust,
  deterministic regression guard on the whole pipeline. The check is decoupled
  from the production PCBDDC iterative solver (its convergence is order/rank
  sensitive — a solver property, not `fm.csem` forward behaviour).

### Level 6 — Order of accuracy / MMS (`mms/test_level6_mms.py`, orders 1, 3, 6)
* Runs `fm.csem -mms` on the `[0,1]³` unit cube at **N=4** with a manufactured
  exact solution `E*`; the direct LU solve makes the reported relative L² and
  H(curl) errors (of both the Galerkin solve and the L²-projection) reproducible.
* The four error metrics and the DOF count are checked against
  `mms/reference/mms_golden.json`, and the backward residual `‖Ax−b‖/‖b‖` must be
  `< 1e-10` (solve is exact). Tolerance is combined `atol + rtol·|golden|`
  (`rtol=1e-3`, `atol=1e-8`): rtol guards the well-resolved orders, atol floors
  the tiny high-order errors that wobble at round-off under MPI/BLAS reordering.
* Orders 1, 3, 6 bracket the supported range 1–6; N=4 keeps each solve to seconds
  and well under a node's memory. Self-contained — reads nothing from `paper/`.

## Running

```bash
# Levels 1-3 - all orders 1..6, fast (needs PETSc + mpicc; no binary):
pytest tests/unit

# Levels 4-5 - orders 1..3 (needs the fm.csem binary):
PETGEM_FM_CSEM=build/fm.csem pytest tests/e2e

# up to 4 MPI tasks for the e2e solves:
PETGEM_FM_CSEM=build/fm.csem FM_CSEM_NP=4 pytest tests/e2e

# Level 6 - MMS (needs the binary + gmsh; build the N=4 bundle first):
bash tests/mms/make_bundle.sh 4
PETGEM_FM_CSEM=build/fm.csem FM_CSEM_NP=2 pytest tests/mms

# everything:
PETGEM_FM_CSEM=build/fm.csem pytest
```

Fixtures `pytest.skip` cleanly when the PETSc toolchain (levels 1-3) or the
`fm.csem` binary (levels 4-5) is unavailable, so partial environments still get
whatever coverage they can run.

### Determinism & tolerances
* Levels 1-3 assert exact analytic invariants (tol ≈ `1e-9`).
* Level 5 compares a **direct LU (MUMPS) solve** against the **exact LU golden**
  with a loose relative tolerance (`1e-3`). A direct solve is deterministic and
  always succeeds, so the check is stable across MPI rank counts and never
  depends on iterative-solver convergence.
* Level 6 compares the reported MMS error norms against the golden with a
  combined `atol + rtol·|golden|` tolerance (`rtol=1e-3`, `atol=1e-8`) — same
  MPI/BLAS-robustness rationale as level 5, with `atol` covering the tiny
  high-order errors where a round-off absolute change is a large relative one.

### Regenerating the golden references
The goldens are exact serial LU solves of the current code. Regenerate them
from a trusted build whenever the forward result legitimately changes — run
`fm.csem` on the unit cube for orders 1, 2, 3 with a direct LU solve, writing
into `reference/`. The `examples/unit_cube/scripts/regen_reference.sh` wrapper
does exactly this:

```bash
bash examples/unit_cube/scripts/regen_reference.sh

# equivalently, explicit:
for N in 1 2 3; do
  build/fm.csem \
    -input_filename examples/unit_cube/outputs/input.h5 -order $N \
    -output_dir examples/unit_cube/reference -output_filename responses_p$N \
    -dm_mat_type aij -ksp_type preonly -pc_type lu \
    -pc_factor_mat_ordering_type nd -ksp_error_if_not_converged
done
```

The **level-6 MMS golden** (`mms/reference/mms_golden.json`) has its own
self-contained regenerator — run it in `petgem-env`/the CI image only after an
intentional discretisation change, then code-review the diff:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace petgem-env \
    bash tests/mms/update_golden.sh
```

## GitHub Actions execution strategy

Workflow: [`.github/workflows/tests-fm-csem.yml`](../.github/workflows/tests-fm-csem.yml)
— a **reusable** workflow called by
[`ci-develop.yml`](../.github/workflows/ci-develop.yml) after it has compiled the
`fm.csem` binary. It consumes that artifact rather than rebuilding it. The CI
image itself is built separately by
[`image.yml`](../.github/workflows/image.yml) (only when `docker/**` changes) and
merely pulled here. All jobs run inside that prebuilt image (PETSc, mpicc, gmsh,
python, h5py, numpy).

| Job | Runs | Needs |
|-----|------|-------|
| `fe-core-tests` | `pytest tests/unit` — levels 1-3, all orders 1-6, ~10 s | CI image (no binary) |
| `e2e-tests` | `pytest tests/e2e` — levels 4-5, orders 1-3, `FM_CSEM_NP=2` | the `fm.csem` artifact |
| `mms-tests` | `make_bundle.sh 4` then `pytest tests/mms` — level 6, orders 1/3/6, `FM_CSEM_NP=2`, ~3 min | the `fm.csem` artifact |

The order override is passed as `-order N`, which also bypasses the bundle's
order dataset, so all six orders run against the single committed `input.h5`
(and, for level 6, against the on-the-fly N=4 MMS bundle).

> Scope: FM-CSEM only. IM-CSEM (inverse kernel) is intentionally out of scope.
