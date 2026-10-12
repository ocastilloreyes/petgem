"""Reduced im_csem_block inversion test (im.csem).

Runs im.csem for two L-BFGS iterations on the im_csem_block conductive-block model at
order 1 with two frequencies (1 and 10 Hz), direct LU (MUMPS), on 2 ranks, and
checks the RMS history, the norm of the recovered log-perturbation and the
iteration counters against reference/im_golden.json (rtol 1e-8).

``build_bundle.sh`` builds tests/im/input.h5.
"""
import json

import numpy as np
import pytest

from fmcsem_testlib import DIRECT_SOLVER, REPO_ROOT, find_binary, run_kernel

pytestmark = pytest.mark.slow

IM_DIR = REPO_ROOT / "tests" / "im"
BUNDLE = IM_DIR / "input.h5"
GOLDEN = json.loads((IM_DIR / "reference" / "im_golden.json").read_text())
RTOL = 1e-8


@pytest.fixture(scope="session")
def im_run(tmp_path_factory):
    binary = find_binary("im.csem", "PETGEM_IM_CSEM")
    if binary is None:
        pytest.skip("im.csem binary not found (set PETGEM_IM_CSEM or build build/im.csem)")
    if not BUNDLE.exists():
        pytest.skip(f"{BUNDLE} not present - run `bash tests/im/build_bundle.sh` first")
    outdir = tmp_path_factory.mktemp("im_run")
    args = ["-input_filename", str(BUNDLE), "-output_dir", str(outdir), "-output_filename", "im",
            "-im_max_iter", str(GOLDEN["max_iter"]), "-im_lbfgs_memory", "2", "-im_lambda", "0.1",
            "-im_snapshot_interval", "0", *DIRECT_SOLVER]
    return run_kernel(binary, args, 2), outdir / "im.h5"


def _text(v):
    return v.decode() if isinstance(v, bytes) else str(v)


@pytest.mark.e2e
def test_im_reduced(im_run, h5py_mod):
    proc, out = im_run
    assert proc.returncode == 0, f"im.csem failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    assert out.exists(), "no im.csem output produced"

    with h5py_mod.File(str(out), "r") as f:
        rms = np.asarray(f["rms_history"])[:, 0]
        dx = np.asarray(f["log_perturbation"])[:, 0]
        assert "conductivity" in f
        attrs = {key: f.attrs[key] for key in ("convergence_reason", "num_iterations", "num_objgrad_evaluations")}

    np.testing.assert_allclose(rms, GOLDEN["rms_history"], rtol=RTOL)
    assert np.linalg.norm(dx) == pytest.approx(GOLDEN["log_perturbation_norm"], rel=RTOL)
    assert _text(attrs["convergence_reason"]) == GOLDEN["convergence_reason"]
    assert int(np.ravel(attrs["num_iterations"])[0]) == GOLDEN["num_iterations"]
    assert int(np.ravel(attrs["num_objgrad_evaluations"])[0]) == GOLDEN["num_objgrad_evaluations"]
