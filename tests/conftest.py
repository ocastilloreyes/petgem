"""Shared pytest fixtures for the PETGEM test suite.

  * tests/unit: C harnesses under tests/csrc/, compiled once against the
    production sources and run per order.
  * tests/e2e, tests/mt, tests/im: the built fm.csem / fm.mt / im.csem binaries.
  * tests/mms: the MMS verification program build/mms (`make mms`).

Fixtures whose toolchain or binary is missing call ``pytest.skip``.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmcsem_testlib as lib  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return lib.REPO_ROOT


@pytest.fixture(scope="session")
def unit_cube() -> Path:
    if not lib.INPUT_BUNDLE.exists():
        pytest.skip("examples/fm_csem_unit_cube/outputs/input.h5 not present "
                    "(bash examples/fm_csem_unit_cube/scripts/build_bundles.sh)")
    return lib.UNIT_CUBE


@pytest.fixture(scope="session")
def harnesses(tmp_path_factory):
    """Compile every C harness once; return {name: exe_path}."""
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


@pytest.fixture(scope="session")
def fm_csem_binary() -> Path:
    path = lib.find_binary("fm.csem", "PETGEM_FM_CSEM")
    if path is None:
        pytest.skip("fm.csem binary not found (set PETGEM_FM_CSEM or build build/fm.csem)")
    return path


@pytest.fixture(scope="session")
def mms_binary() -> Path:
    path = lib.find_binary("mms", "PETGEM_MMS")
    if path is None:
        pytest.skip("mms binary not found (set PETGEM_MMS or run `make mms`)")
    return path


@pytest.fixture(scope="session")
def fm_mt_binary() -> Path:
    path = lib.find_binary("fm.mt", "PETGEM_FM_MT")
    if path is None:
        pytest.skip("fm.mt binary not found (set PETGEM_FM_MT or build build/fm.mt)")
    return path


@pytest.fixture(scope="session")
def fm_tdcsem_binary() -> Path:
    path = lib.find_binary("fm.tdcsem", "PETGEM_FM_TDCSEM")
    if path is None:
        pytest.skip("fm.tdcsem binary not found (set PETGEM_FM_TDCSEM or build build/fm.tdcsem)")
    return path


class FmRun:
    """Result of one fm.csem run on the unit cube."""
    def __init__(self, order, returncode, stdout, responses):
        self.order = order
        self.returncode = returncode
        self.stdout = stdout
        self.responses = responses   # responses file (absent on failure)


# Solver profiles, selected through PETSc options only.
#   "solve"    - direct LU, compared against the LU goldens (level 5).
#   "assemble" - preonly/jacobi: exercises assembly only (level 4).
SOLVER_OPTS = {
    "solve":    lib.DIRECT_SOLVER,
    "assemble": ["-dm_mat_type", "aij", "-ksp_type", "preonly", "-pc_type", "jacobi"],
}


@pytest.fixture(scope="session")
def fm_run(fm_csem_binary, unit_cube, tmp_path_factory):
    """Run fm.csem once per (order, solver) on the unit cube; cached getter.

    The order is forced with -order N. Serial unless PETGEM_NP is set.
    """
    outdir = tmp_path_factory.mktemp("fm_runs")
    nprocs = os.environ.get("PETGEM_NP")
    cache = {}

    def _run(order, solver="solve"):
        key = (order, solver)
        if key not in cache:
            stem = f"responses_p{order}_{solver}"
            args = ["-input_filename", str(lib.INPUT_BUNDLE), "-order", str(order),
                    "-output_dir", str(outdir), "-output_filename", stem, *SOLVER_OPTS[solver]]
            proc = lib.run_kernel(fm_csem_binary, args, nprocs)
            cache[key] = FmRun(order, proc.returncode, proc.stdout + proc.stderr, outdir / f"{stem}.h5")
        return cache[key]

    return _run


@pytest.fixture(scope="session")
def h5py_mod():
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("numpy")
    return h5py
