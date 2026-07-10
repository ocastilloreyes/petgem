"""LEVEL 6 - MMS order-of-accuracy regression (self-contained; no paper/ dependence).

Runs fm.csem in Method-of-Manufactured-Solutions mode (``-mms``) on the [0,1]^3
unit cube at N=4 for a representative span of orders (p = 1, 3, 6), solved with a
DIRECT LU factorization (MUMPS) - exact and deterministic, serially and under
MPI. It checks the relative solution and L2-projection errors fm.csem reports
against committed golden values (``reference/mms_golden.json``).

The LU solve is exact, but MUMPS/BLAS reorder sums across MPI ranks and
platforms; for the high-order/coarse-mesh cases the error norm is a small
difference of near-equal fields, so that reordering surfaces as a large
*relative* wobble on a round-off-level *absolute* change. The compare therefore
uses a combined ``atol + rtol*|golden|`` tolerance (as level 5 uses a loose
tol): rtol guards the well-resolved cases, atol floors the tiny ones. Any real
discretisation regression (basis, DOF layout, assembly, quadrature, error-norm
reconstruction) moves the errors far beyond that and fails CI. The three orders
bracket the supported range 1..6 at N=4 (384 cells): seconds each, well under a
node's memory, no over-decomposition.

All inputs live under ``tests/mms/`` (mesh.geo + the sigma/source/receiver
stubs); the N=4 bundle ``tests/mms/input.h5`` is produced by ``make_bundle.sh``
(the CI job runs it before pytest; run it by hand for a local checkout).
Nothing here reads ``paper/``.
"""
import math
import os
import subprocess

import pytest

from fmcsem_testlib import REPO_ROOT

# Same "slow" bucket as the other binary-driven levels.
pytestmark = pytest.mark.slow

MMS_DIR = REPO_ROOT / "tests" / "mms"
GOLDEN_FILE = MMS_DIR / "reference" / "mms_golden.json"
BUNDLE = MMS_DIR / "input.h5"
METRICS = ("solve_L2", "solve_Hcurl", "proj_L2", "proj_Hcurl")

# Direct LU via MUMPS: exact, deterministic, rank-independent (see level 5).
SOLVER_OPTS = ["-dm_mat_type", "aij", "-ksp_type", "preonly", "-pc_type", "lu",
               "-pc_factor_mat_solver_type", "mumps", "-ksp_error_if_not_converged"]


@pytest.fixture(scope="session")
def golden():
    import json
    if not GOLDEN_FILE.exists():
        pytest.skip(f"golden {GOLDEN_FILE} not present")
    return json.loads(GOLDEN_FILE.read_text())


@pytest.fixture(scope="session")
def mms_bundle():
    """The N=4 MMS input bundle (built by tests/mms/make_bundle.sh)."""
    if not BUNDLE.exists():
        pytest.skip(f"{BUNDLE} not present - run `bash tests/mms/make_bundle.sh 4` first")
    return BUNDLE


@pytest.fixture(scope="session")
def mms_run(fm_csem_binary, mms_bundle, tmp_path_factory):
    """Run fm.csem -mms once per order on the N=4 bundle; cache + return getter."""
    outdir = tmp_path_factory.mktemp("mms_runs")
    npr = os.environ.get("FM_CSEM_NP")
    cache = {}

    def _run(order):
        if order in cache:
            return cache[order]
        stem = f"mms_o{order}_N4"
        launch = (["mpirun", "-n", npr] if npr else []) + [str(fm_csem_binary)]
        cmd = launch + ["-mms", "-order", str(order),
                        "-input_filename", str(mms_bundle),
                        "-output_dir", str(outdir), "-output_filename", stem] + SOLVER_OPTS
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                              cwd=str(REPO_ROOT))
        cache[order] = (proc, outdir / f"{stem}.h5")
        return cache[order]

    return _run


@pytest.mark.e2e
@pytest.mark.parametrize("order", [1, 3, 6])
def test_mms_errors_match_golden(mms_run, h5py_mod, golden, order):
    key = str(order)
    if key not in golden["orders"]:
        pytest.skip(f"no golden entry for order {order}")
    exp = golden["orders"][key]
    rtol = float(golden["rtol"])
    atol = float(golden["atol"])
    res_max = float(golden["residual_max"])

    proc, out = mms_run(order)
    assert proc.returncode == 0, f"fm.csem -mms failed at order {order}:\n{proc.stdout[-3000:]}"
    assert out.exists(), f"no MMS output produced at order {order}"

    with h5py_mod.File(str(out), "r") as f:
        a = f.attrs
        got = {k: float(a[k]) for k in METRICS}
        dofs = int(a["dofs"])
        residual = float(a["residual"])

    # (a) sanity: exact solve, expected problem size, finite non-trivial errors
    assert dofs == int(exp["dofs"]), f"order {order}: dofs {dofs} != golden {exp['dofs']}"
    assert residual < res_max, \
        f"order {order}: residual {residual:.2e} not < {res_max:.0e} (solve not exact?)"
    for k in METRICS:
        assert math.isfinite(got[k]) and got[k] > 0, f"order {order}: {k}={got[k]} not positive-finite"

    # (b) reproduce the committed golden within a combined abs+rel tolerance
    for k in METRICS:
        assert abs(got[k] - exp[k]) <= atol + rtol * abs(exp[k]), (
            f"order {order}: {k} deviates from golden by {abs(got[k] - exp[k]):.2e} "
            f"> atol+rtol*|golden| ({atol:.0e}+{rtol:.0e}*{abs(exp[k]):.2e}); "
            f"got {got[k]:.6e}, golden {exp[k]:.6e}")
