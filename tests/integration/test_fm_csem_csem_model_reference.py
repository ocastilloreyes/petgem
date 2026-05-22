"""Numerical regression test for the csem_model case.

Runs the full gmsh -> preprocess -> fm.csem -> readResponses pipeline at
nord = 1, 2, 3 and asserts that the magnitude of Ex agrees with the
precomputed ModEM reference (`reference.h5` shipped under
tests/cases/csem_model/) within a configured NRMSD tolerance.  This is the
canonical "did fm.csem produce the right numbers?" check, complementary to
the structural smoke test in test_fm_csem_smoke.py.

The same ModEM reference is used at every order - the receiver line is
fixed, so the reference Ex is order-independent and each nord's solution
is compared against it.  nord >= 4 is intentionally NOT exercised here:
those orders need a direct solver (BDDC degrades) and/or longer runs, so
they are validated manually on the cluster (see the per-nord tuned meshes
documented in the project memory).

Skipped (not failed) when:
  - the fm.csem binary is unavailable;
  - the reference.h5 file is missing from the case directory.
"""
import h5py
import numpy as np
import pytest

import petgem

from _fm_pipeline import (
    copy_case_workspace,
    run_pipeline_for_nord,
    skip_if_no_binary,)


pytestmark = pytest.mark.integration


# Orders validated numerically in CI.  The tuned per-nord meshes
# (tests/cases/csem_model/mesh_p{1,2,3}.geo) keep each run feasible:
# observed NRMSD is 1.55e-3 / 5.79e-3 / 1.51e-2 for nord 1/2/3 - all
# comfortably under the shared 0.03 threshold.
NORDS = [1, 2, 3]
NRMSD_TOLERANCE = 0.03   # 3 %, matches the legacy postprocess threshold


@pytest.fixture(scope="module")
def reference_workspace(tmp_path_factory, cases_dir):
    """Standalone workspace for the reference test, so the file can run in
    isolation or in any order relative to the smoke parametrization."""
    return copy_case_workspace(
        tmp_path_factory, cases_dir, "csem_model",
        require_files=("reference.h5",),
    )


def _load_reference(case_dir):
    """ModEM reference layout: top-level /reference_real and /reference_imag
    datasets, flattened to a 1-D complex Ex array on the receiver line."""
    with h5py.File(case_dir / "reference.h5", "r") as f:
        re = f["/reference_real"][()]
        im = f["/reference_imag"][()]
    return (re + 1j * im).ravel()


def _nrmsd(reference, simulated):
    """Normalized RMSD of |·| - matches the legacy postprocess threshold."""
    mag_ref = np.abs(reference)
    mag_sim = np.abs(simulated)
    rmsd = np.sqrt(np.mean((mag_sim - mag_ref) ** 2))
    spread = mag_ref.max() - mag_ref.min()
    return rmsd / spread


@pytest.mark.parametrize("nord", NORDS)
def test_fm_csem_csem_model_reference(repo_root, fm_csem_binary, reference_workspace, nord):
    skip_if_no_binary(fm_csem_binary)

    responses_path, _ = run_pipeline_for_nord(repo_root, fm_csem_binary, reference_workspace, nord)

    responses = petgem.readResponses(str(responses_path))
    Ex_sim    = np.asarray(responses["Ex"])
    Ex_ref    = _load_reference(reference_workspace)

    assert Ex_sim.shape == Ex_ref.shape, (f"PETGEM Ex shape {Ex_sim.shape} vs reference {Ex_ref.shape} at nord={nord}")

    nrmsd_val = _nrmsd(Ex_ref, Ex_sim)
    rel_l2    = (np.linalg.norm(np.abs(Ex_sim) - np.abs(Ex_ref))
                 / np.linalg.norm(np.abs(Ex_ref)))

    print(f"\n[reference nord={nord}] NRMSD = {nrmsd_val:.6e}  (tolerance {NRMSD_TOLERANCE})")
    print(f"[reference nord={nord}] relative L2 of |Ex| = {rel_l2:.6e}")

    assert nrmsd_val < NRMSD_TOLERANCE, (
        f"csem_model NRMSD {nrmsd_val:.6e} exceeded tolerance "
        f"{NRMSD_TOLERANCE} at nord={nord}")
