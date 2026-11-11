"""LEVEL 6 - MMS order-of-accuracy regression.

Runs the MMS program ``build/mms`` (tests/mms/mms.c, ``make mms``) on the [0,1000]^3 cube at N=4 for orders 1, 3 and 6 with
a direct LU solve, and compares the reported solution and L2-projection errors
with ``reference/mms_golden.json`` under ``atol + rtol*|golden|``. The atol term
covers the round-off wobble of the tiny high-order errors under MPI.

Inputs live in tests/mms/ (L = 1000 m, f = 10 Hz, sigma = (0.5, 1.0, 2.0) S/m);
``build_bundle.sh`` builds tests/mms/input.h5.
"""
import math
import os

import pytest

from fmcsem_testlib import DIRECT_SOLVER, REPO_ROOT, run_kernel

# Same "slow" bucket as the other binary-driven levels.
pytestmark = pytest.mark.slow

MMS_DIR = REPO_ROOT / "tests" / "mms"
GOLDEN_FILE = MMS_DIR / "reference" / "mms_golden.json"
BUNDLE = MMS_DIR / "input.h5"
METRICS = ("solve_L2", "solve_energy", "proj_L2", "proj_energy")


@pytest.fixture(scope="session")
def golden():
    import json
    if not GOLDEN_FILE.exists():
        pytest.skip(f"golden {GOLDEN_FILE} not present")
    return json.loads(GOLDEN_FILE.read_text())


@pytest.fixture(scope="session")
def mms_bundle():
    """The N=4 MMS input bundle (built by tests/mms/build_bundle.sh)."""
    if not BUNDLE.exists():
        pytest.skip(f"{BUNDLE} not present - run `bash tests/mms/build_bundle.sh 4` first")
    return BUNDLE


@pytest.fixture(scope="session")
def mms_run(mms_binary, mms_bundle, tmp_path_factory):
    """Run mms once per order on the N=4 bundle; cache + return getter."""
    outdir = tmp_path_factory.mktemp("mms_runs")
    nprocs = os.environ.get("PETGEM_NP")
    cache = {}

    def _run(order):
        if order not in cache:
            stem = f"mms_o{order}_N4"
            args = ["-order", str(order), "-input_filename", str(mms_bundle),
                    "-output_dir", str(outdir), "-output_filename", stem, *DIRECT_SOLVER]
            cache[order] = (run_kernel(mms_binary, args, nprocs), outdir / f"{stem}.h5")
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
    assert proc.returncode == 0, f"mms failed at order {order}:\n{proc.stdout[-3000:]}"
    assert out.exists(), f"no MMS output produced at order {order}"

    with h5py_mod.File(str(out), "r") as f:
        a = f.attrs
        got = {metric: float(a[metric]) for metric in METRICS}
        dofs = int(a["dofs"])
        residual = float(a["residual"])

    # (a) sanity: exact solve, expected problem size, finite non-trivial errors
    assert dofs == int(exp["dofs"]), f"order {order}: dofs {dofs} != golden {exp['dofs']}"
    assert residual < res_max, \
        f"order {order}: residual {residual:.2e} not < {res_max:.0e} (solve not exact?)"
    for metric in METRICS:
        assert math.isfinite(got[metric]) and got[metric] > 0, f"order {order}: {metric}={got[metric]} not positive-finite"

    # (b) reproduce the committed golden within a combined abs+rel tolerance
    for metric in METRICS:
        assert abs(got[metric] - exp[metric]) <= atol + rtol * abs(exp[metric]), (
            f"order {order}: {metric} deviates from golden by {abs(got[metric] - exp[metric]):.2e} "
            f"> atol+rtol*|golden| ({atol:.0e}+{rtol:.0e}*{abs(exp[metric]):.2e}); "
            f"got {got[metric]:.6e}, golden {exp[metric]:.6e}")
