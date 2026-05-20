"""End-to-end smoke test for the forward kernel, parametrized over nord.

Pipeline (skipped if fm.csem isn't built):

  1. Run utils/preprocess.py against tests/cases/csem_model/ to produce a
     per-nord input bundle + params file inside a tmp_path/case copy.
  2. Run fm.csem on those params.
  3. Read the responses HDF5 through petgem.readResponses and assert the
     output is structurally well-formed (Ex/Ey/Ez/Hx/Hy/Hz present, all
     non-empty, dimensions agree with receivers).

Structural check only - kernel runs and produces a finite, correctly-
sized response across the supported polynomial orders.  Numerical
validation against the reference solution lives in
test_fm_csem_csem_model_reference.py.
"""
import numpy as np
import pytest

import petgem

from _fm_pipeline import (
    copy_case_workspace,
    run_pipeline_for_nord,
    skip_if_no_binary,
)


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def case_workspace(tmp_path_factory, cases_dir):
    """Shared workspace for every parametrization in this file."""
    return copy_case_workspace(tmp_path_factory, cases_dir, "csem_model")


@pytest.mark.parametrize("nord", [1, 2])
def test_fm_csem_smoke(repo_root, fm_csem_binary, case_workspace, nord):
    """Kernel runs at nord ∈ {1, 2}; field components exist, are finite,
    and sized to the receivers vector."""
    skip_if_no_binary(fm_csem_binary)

    responses_path, bundle_path = run_pipeline_for_nord(repo_root, fm_csem_binary, case_workspace, nord)

    responses = petgem.readResponses(str(responses_path))
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert name in responses
        arr = np.asarray(responses[name])
        assert arr.size > 0, f"{name} is empty at nord={nord}"
        assert np.all(np.isfinite(arr)), (
            f"{name} contains non-finite values at nord={nord}"
        )

    bundle = petgem.readBundle(str(bundle_path))
    n_recv = bundle["receivers"].shape[0]
    assert np.asarray(responses["Ex"]).size == n_recv

    prov = responses.get("provenance", {})
    prov_nord = prov.get("nord", -1)
    assert int(prov_nord) == nord, (
        f"responses provenance nord={prov_nord!r} disagrees with {nord}"
    )
