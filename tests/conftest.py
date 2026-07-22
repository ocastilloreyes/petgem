"""Shared pytest fixtures for the FM-CSEM test suite (unit cube, orders 1..6).

Two families share this infrastructure:

  * levels 1-3 (unit): small C harnesses under tests/csrc/ are compiled against
    the UNCHANGED production sources and run per order. They verify the real
    reference-element bases, DOF enumeration and elemental matrices directly.
  * levels 4-5 (e2e): the built `fm.csem` binary is run once per order on the
    single examples/unit_cube/outputs/input.h5 bundle (order forced with `-order N`,
    which also bypasses the bundle's order dataset).

Fixtures needing a toolchain/binary that may be absent ``pytest.skip`` so the
suite degrades gracefully (Python-only checkout, or a CI stage without the
kernel built).
"""
import sys
from pathlib import Path

import pytest

# Make the plain helper module importable from every test file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmcsem_testlib as lib  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return lib.REPO_ROOT


@pytest.fixture(scope="session")
def unit_cube() -> Path:
    """The single dataset the suite is allowed to use."""
    if not lib.INPUT_BUNDLE.exists():
        pytest.skip("unit_cube/outputs/input.h5 dataset not present")
    return lib.UNIT_CUBE


# --------------------------------------------------------------------------- #
# C harnesses (levels 1-3)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def harnesses(tmp_path_factory):
    """Compile every C harness once; return {name: exe_path}.

    Skips the whole level 1-3 family when PETSc / mpicc is unavailable.
    """
    if lib.petsc_paths() is None:
        pytest.skip("PETSc toolchain (PETSC_DIR + mpicc) not available")
    outdir = tmp_path_factory.mktemp("harness")
    built = {}
    for name in lib.HARNESS_SOURCES:
        exe, err = lib.build_harness(name, outdir)
        if exe is None:
            pytest.fail(f"failed to build harness {name}:\n{err}")
        built[name] = exe
    return built


# --------------------------------------------------------------------------- #
# fm.csem binary + per-order runner (levels 4-5)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def fm_csem_binary() -> Path:
    """Resolve the fm.csem binary; skip e2e tests when it is not built."""
    import os
    import shutil
    env = os.environ.get("PETGEM_FM_CSEM")
    candidates = ([Path(env)] if env else []) + [
        lib.REPO_ROOT / "build" / "fm.csem", lib.REPO_ROOT / "fm.csem"]
    which = shutil.which("fm.csem")
    if which:
        candidates.append(Path(which))
    for c in candidates:
        if c.exists():
            return c
    pytest.skip("fm.csem binary not found (set PETGEM_FM_CSEM or build build/fm.csem)")


class FmRun:
    """Captured result of one fm.csem run on the unit cube."""
    def __init__(self, order, returncode, stdout, responses):
        self.order = order
        self.returncode = returncode
        self.stdout = stdout
        self.responses = responses   # Path to responses_p{order}.h5 (may be absent on failure)


# Solver profiles selected purely through PETSc runtime options (no code change).
#   "solve"    - direct LU via MUMPS: exact fields, deterministic, and valid both
#                serially and under MPI (level 5 golden compare). MUMPS is in the
#                CI image; a direct solve always "converges", so the regression
#                check never depends on iterative-solver behaviour. The committed
#                goldens are exact LU solves, so this reproduces them to round-off.
#   "assemble" - trivial preonly/jacobi: exercises assembly + the per-cell
#                checkGradientKernel only; the "solution" is ignored (level 4).
SOLVER_OPTS = {
    "solve":    ["-dm_mat_type", "aij", "-ksp_type", "preonly", "-pc_type", "lu",
                 "-pc_factor_mat_solver_type", "mumps", "-ksp_error_if_not_converged"],
    "assemble": ["-dm_mat_type", "aij", "-ksp_type", "preonly", "-pc_type", "jacobi"],
}


@pytest.fixture(scope="session")
def fm_run(fm_csem_binary, unit_cube, tmp_path_factory):
    """Run fm.csem once per (order, solver) on the unit cube; cache + return getter.

    Order is forced with `-order N` (also bypassing the bundle's order dataset);
    the solver is chosen via runtime PETSc options (see SOLVER_OPTS). Output goes
    to a temp dir so the repo stays clean. Serial by default; set FM_CSEM_NP to
    run the (direct) solve under mpirun.
    """
    import os
    import subprocess
    outdir = tmp_path_factory.mktemp("fm_runs")
    npr = os.environ.get("FM_CSEM_NP")
    cache = {}

    def _run(order, solver="solve"):
        key = (order, solver)
        if key in cache:
            return cache[key]
        stem = f"responses_p{order}_{solver}"
        launch = (["mpirun", "-n", npr] if npr else []) + [str(fm_csem_binary)]
        base = ["-input_filename", str(lib.INPUT_BUNDLE),
                "-order", str(order),
                "-output_dir", str(outdir), "-output_filename", stem]
        cmd = launch + base + SOLVER_OPTS[solver]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                              cwd=str(lib.REPO_ROOT))
        res = FmRun(order, proc.returncode, proc.stdout + proc.stderr, outdir / f"{stem}.h5")
        cache[key] = res
        return res

    return _run


# --------------------------------------------------------------------------- #
# HDF5
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def h5py_mod():
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("numpy")
    return h5py
