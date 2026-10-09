"""MT 1D boundary field (src/mt.c).

Drives tests/csrc/test_mt1d.c and checks:
  * solveMt1D against the exact layered solution for both 1D equations,
    with second-order convergence,
  * evalMt1DField interpolation and clamping,
  * classifyMtBoxFaces on a box and rejection of a non-box boundary,
  * buildMt1DProfile layers on a 3-layer box (same on every rank) and
    rejection of a lateral anomaly.
"""
import pytest

from fmcsem_testlib import run_harness


@pytest.mark.parametrize("nprocs", [1, 3])
def test_mt_1d(harnesses, nprocs):
    r = run_harness(harnesses["test_mt1d"], 1, nprocs)
    assert r.returncode == 0, (
        f"MT 1D harness failed on {nprocs} ranks\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    assert "0 failures" in r.stdout
