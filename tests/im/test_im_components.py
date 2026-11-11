"""Data components of im.csem on the reduced im_csem_block bundle.

The gradient check (-im_gradient_check) compares the adjoint directional
derivative of the data misfit with central differences at two step sizes, for
a field component (Hy) and for Ex jointly with the ratio Zxy = Ex/Hy; the
error must be small and fall as the square of the step. One L-BFGS iteration
on Zxy alone must lower the RMS, and the predicted Zxy must reproduce the
reported RMS against /observed/Zxy.

Same run settings as test_im_reduced.py: order 1, 1 and 10 Hz, MUMPS, 2 ranks.
``build_bundle.sh`` builds tests/im/input.h5 with /observed/Ex, Hy and Zxy.
"""
import numpy as np
import pytest

from fmcsem_testlib import DIRECT_SOLVER, REPO_ROOT, find_binary, run_kernel

pytestmark = pytest.mark.slow

BUNDLE = REPO_ROOT / "tests" / "im" / "input.h5"
STEPS = (1e-2, 1e-3)
REL_ERROR_MAX = 1e-6


def _run(tmp_path, name, args):
    binary = find_binary("im.csem", "PETGEM_IM_CSEM")
    if binary is None:
        pytest.skip("im.csem binary not found (set PETGEM_IM_CSEM or build build/im.csem)")
    if not BUNDLE.exists():
        pytest.skip(f"{BUNDLE} not present - run `bash tests/im/build_bundle.sh` first")
    proc = run_kernel(binary, ["-input_filename", str(BUNDLE), "-output_dir", str(tmp_path), "-output_filename", name,
                               *args, *DIRECT_SOLVER], 2)
    assert proc.returncode == 0, f"im.csem failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    out = tmp_path / f"{name}.h5"
    assert out.exists(), "no im.csem output produced"
    return out


@pytest.mark.e2e
@pytest.mark.parametrize("components", ["Hy", "Ex,Zxy"])
def test_gradient_check(components, tmp_path, h5py_mod):
    out = _run(tmp_path, "gc", ["-im_gradient_check", "-im_data_components", components,
                                "-im_gradient_check_eps", ",".join(str(eps) for eps in STEPS)])
    with h5py_mod.File(str(out), "r") as f:
        eps = np.asarray(f["gradient_check_eps"])[:, 0]
        fd = np.asarray(f["gradient_check_fd"])[:, 0]
        adjoint = np.asarray(f["gradient_check_adjoint"])[:, 0]

    np.testing.assert_allclose(eps, STEPS)
    error = np.abs(fd - adjoint) / np.abs(adjoint)
    assert error[-1] < REL_ERROR_MAX, f"{components}: relative error {error}"
    # central differences: error ~ eps^2
    assert error[0] / error[-1] > 50.0, f"{components}: relative error {error} does not fall as eps^2"


@pytest.mark.e2e
def test_zxy_inversion(tmp_path, h5py_mod):
    out = _run(tmp_path, "zxy", ["-im_data_components", "Zxy", "-im_max_iter", "1", "-im_lbfgs_memory", "2",
                                 "-im_lambda", "0.1", "-im_snapshot_interval", "0"])
    with h5py_mod.File(str(out), "r") as f:
        rms = np.asarray(f["rms_history"])[:, 0]
        pre = np.asarray(f["predicted_Zxy"])
        rms_pre = float(np.ravel(f.attrs["rms_predicted"])[0])
        assert "predicted_Ex" not in f
    with h5py_mod.File(str(BUNDLE), "r") as f:
        obs = np.asarray(f["observed/Zxy"])
        err = float(f["observed"].attrs["error_level"])

    assert rms[-1] < rms[0]
    assert pre.shape == obs.shape + (2,)
    pre = pre[..., 0] + 1j * pre[..., 1]
    rms = np.sqrt(np.mean(np.abs((obs - pre) / (err * np.abs(obs)))**2) / 2)
    assert rms == pytest.approx(rms_pre, rel=1e-10)
