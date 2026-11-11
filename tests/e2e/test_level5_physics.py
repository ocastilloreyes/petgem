"""LEVEL 5 - Basic FM-CSEM physics (unit cube, orders 1..3).

Forward run on the homogeneous unit cube (one 2 Hz dipole, three receivers)
with a direct LU solve. Checks that every field component is present, finite
and non-trivial, and that it matches the committed LU golden in
examples/fm_csem_unit_cube/reference/ to a relative 1e-3.
"""
import numpy as np
import pytest

from fmcsem_testlib import PIPELINE_ORDERS, FIELD_COMPONENTS, REFERENCE_DIR, load_fields

# The whole e2e family is the "slow" bucket (full pipeline on the fixed mesh).
pytestmark = pytest.mark.slow

N_RECV = 3
REL_TOL = 1e-3


@pytest.mark.e2e
@pytest.mark.parametrize("order", PIPELINE_ORDERS)
def test_fields_valid_and_match_reference(fm_run, h5py_mod, order):
    run = fm_run(order, "solve")
    assert run.returncode == 0, f"fm.csem (LU) failed at order {order}:\n{run.stdout[-3000:]}"
    assert run.responses.exists(), f"no responses file produced at order {order}"
    assert "Error element" not in run.stdout, f"kernel violation reported at order {order}"

    got = load_fields(h5py_mod, run.responses)

    # (a) structural validity
    for comp in FIELD_COMPONENTS:
        a = got[comp]
        assert a.shape == (N_RECV,), f"{comp} shape {a.shape} != ({N_RECV},) at order {order}"
        assert np.all(np.isfinite(a)), f"{comp} has non-finite values at order {order}"
    assert sum(np.linalg.norm(got[comp]) for comp in ("Ex", "Ey", "Ez")) > 0, f"E all-zero at order {order}"
    assert sum(np.linalg.norm(got[comp]) for comp in ("Hx", "Hy", "Hz")) > 0, f"H all-zero at order {order}"

    # (b) reproduce the committed golden reference
    ref = REFERENCE_DIR / f"responses_p{order}.h5"
    if not ref.exists():
        pytest.skip(f"golden reference {ref.name} not present (see tests README to regenerate)")
    exp = load_fields(h5py_mod, ref)
    for comp in FIELD_COMPONENTS:
        denom = max(np.linalg.norm(exp[comp]), 1e-30)
        rel = np.linalg.norm(got[comp] - exp[comp]) / denom
        assert rel < REL_TOL, f"order {order}: {comp} deviates from reference by rel={rel:.2e} (tol {REL_TOL})"
