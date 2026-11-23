"""im.csem secondary-field formulation on the reduced im_csem_block inversion.

Order 1, 1 and 10 Hz, MUMPS, 2 ranks (tests/secondary/im.h5,
``build_bundle.sh``). The model background differs from the starting model in
the invertible material; the layered background equals it:

  * the gradient check (-im_gradient_check) on Ex, Hy and Zxy passes with both
    backgrounds: small error that falls as the square of the step;
  * one L-BFGS iteration with the model background gives the RMS history,
    predicted data and model of the total-field formulation;
  * one L-BFGS iteration with the layered background lowers the RMS;
  * the layered background is rejected when a transmitter lies in a free
    cell.
"""
import numpy as np
import pytest

from fmcsem_testlib import DIRECT_SOLVER, REPO_ROOT, find_binary, run_kernel

pytestmark = pytest.mark.slow

BUNDLE = REPO_ROOT / "tests" / "secondary" / "im.h5"
NPROCS = 2
STEPS = (1e-2, 1e-3)
REL_ERROR_MAX = 1e-6
IDENTITY_RTOL = 1e-8
COMPONENTS = "Ex,Hy,Zxy"
LBFGS = ["-im_data_components", "Ex", "-im_max_iter", "1", "-im_lbfgs_memory", "2", "-im_lambda", "0.1",
         "-im_snapshot_interval", "0"]


@pytest.fixture(scope="module")
def im(tmp_path_factory):
    """Cached im.csem runs on im.h5: run(name, args, expect_ok) -> (proc, output path)."""
    binary = find_binary("im.csem", "PETGEM_IM_CSEM")
    if binary is None:
        pytest.skip("im.csem binary not found (set PETGEM_IM_CSEM or build build/im.csem)")
    if not BUNDLE.exists():
        pytest.skip(f"{BUNDLE} not present - run `bash tests/secondary/build_bundle.sh` first")
    outdir = tmp_path_factory.mktemp("secondary_im")
    cache = {}

    def run(name, args=(), expect_ok=True):
        if name not in cache:
            proc = run_kernel(binary, ["-input_filename", str(BUNDLE), "-output_dir", str(outdir), "-output_filename", name,
                                       *args, *DIRECT_SOLVER], NPROCS)
            if expect_ok:
                assert proc.returncode == 0, f"im.csem {name} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
            cache[name] = (proc, outdir / f"{name}.h5")
        return cache[name]

    return run


@pytest.mark.e2e
@pytest.mark.parametrize("primary, report", [("model", "model, 3 material(s)"), ("layered", "layered, 2 layer(s)")])
def test_gradient_check(im, h5py_mod, primary, report):
    proc, out = im(f"gc_{primary}", ["-csem_formulation", "secondary", "-csem_primary", primary, "-im_gradient_check",
                                     "-im_data_components", COMPONENTS, "-im_gradient_check_eps", ",".join(str(e) for e in STEPS)])
    assert report in proc.stdout
    with h5py_mod.File(str(out), "r") as f:
        fd = np.asarray(f["gradient_check_fd"])[:, 0]
        adjoint = np.asarray(f["gradient_check_adjoint"])[:, 0]
    error = np.abs(fd - adjoint) / np.abs(adjoint)
    assert error[-1] < REL_ERROR_MAX, f"{primary}: relative error {error}"
    # central differences: error ~ eps^2
    assert error[0] / error[-1] > 50.0, f"{primary}: relative error {error} does not fall as eps^2"


@pytest.mark.e2e
def test_model_background_reproduces_total(im, h5py_mod):
    _, total = im("lbfgs_total", LBFGS)
    _, model = im("lbfgs_model", ["-csem_formulation", "secondary", "-csem_primary", "model", *LBFGS])
    with h5py_mod.File(str(total), "r") as ft, h5py_mod.File(str(model), "r") as fm:
        for name in ("rms_history", "predicted_Ex", "conductivity"):
            t, m = np.asarray(ft[name]), np.asarray(fm[name])
            err = np.abs(m - t).max() / np.abs(t).max()
            assert err < IDENTITY_RTOL, f"{name}: model background against the total field, relative difference {err:.3e}"


@pytest.mark.e2e
def test_layered_inversion(im, h5py_mod):
    _, out = im("lbfgs_layered", ["-csem_formulation", "secondary", "-csem_primary", "layered", *LBFGS])
    with h5py_mod.File(str(out), "r") as f:
        rms = np.asarray(f["rms_history"])[:, 0]
    assert rms[-1] < rms[0], f"RMS history {rms}"


@pytest.mark.e2e
def test_layered_source_in_free_cell_rejected(im):
    proc, _ = im("free_source", ["-csem_formulation", "secondary", "-csem_primary", "layered", "-im_fixed_materials", "2",
                                 "-im_max_iter", "1"], expect_ok=False)
    assert proc.returncode != 0 and "lies in a free cell" in proc.stdout + proc.stderr


@pytest.mark.e2e
@pytest.mark.parametrize("kernel", ["fm.csem", "im.csem"])
def test_help_lists_formulation_options(kernel):
    binary = find_binary(kernel, "PETGEM_FM_CSEM" if kernel == "fm.csem" else "PETGEM_IM_CSEM")
    if binary is None:
        pytest.skip(f"{kernel} binary not found")
    proc = run_kernel(binary, ["-help"])
    assert all(opt in proc.stdout for opt in ("FIELD FORMULATION", "-csem_formulation", "-csem_primary", "-csem_background_file"))
