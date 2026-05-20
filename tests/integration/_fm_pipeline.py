"""Shared helpers for the fm.csem integration tests.

Not collected by pytest (no `test_` prefix). Provides the workspace
fixture builder, the subprocess wrapper, and the preprocess→kernel
driver used by both the smoke (parametrized over nord) and the
numerical-regression tests.
"""
import os
import shutil
import subprocess
import sys

import pytest


# Per-nord runtime budget for the kernel.  Higher orders need more time
# (3D quadrature + denser local matrices).
FM_CSEM_TIMEOUT_BY_NORD = {
    1:  120,
    2: 4800,
    3: 4800,
    4: 4800,
    5: 4800,
    6: 4800,
}
PREPROCESS_TIMEOUT = 120

# Per-nord runtime budget for gmsh.  The mesh sizes scale down with nord
# (fewer elements as the basis order increases), so higher-order generation
# is cheaper.  Generous upper bounds: even nord=1 generation typically runs
# in under a minute, but slow CI runners need headroom.
GMSH_TIMEOUT_BY_NORD = {
    1: 300,
    2: 240,
    3: 180,
    4: 120,
    5: 120,
    6: 120,
}


def skip_if_no_binary(fm_csem_binary):
    if not fm_csem_binary.exists():
        pytest.skip(
            f"fm.csem binary not found at {fm_csem_binary}. "
            f"Build the kernel (`make`) or set $PETGEM_FM_CSEM to skip explicitly."
        )


def copy_case_workspace(tmp_path_factory, cases_dir, case_name,
                        require_files=()):
    """Copy a case directory into a fresh tmp_path so preprocess can write
    artifacts without polluting the checked-in case.  `require_files` lets
    a caller refuse to start when a needed input (e.g. reference.h5) is
    missing - the test is skipped in that case rather than failing.
    """
    src = cases_dir / case_name
    if not src.exists():
        pytest.skip(f"{case_name} case directory not present at {src}")
    for f in require_files:
        if not (src / f).exists():
            pytest.skip(f"{f} missing from {src} - cannot run this test")
    dst = tmp_path_factory.mktemp(f"{case_name}_workspace")
    print(f"\n[fixture] copying {case_name} → {dst}", flush=True)
    for entry in os.listdir(src):
        full = src / entry
        if full.is_file():
            shutil.copy2(full, dst / entry)
    print(f"[fixture] copy done ({len(list(dst.iterdir()))} files)", flush=True)
    return dst


def _run_with_timeout(cmd, *, cwd, timeout, label):
    """Run a subprocess with a hard timeout. Output streams to the test's
    stdout/stderr (visible under `pytest -s`) so the user can watch live
    progress instead of waiting for the whole run to complete before any
    output appears."""
    print(f"\n[{label}] running ({timeout}s budget): {' '.join(cmd)}", flush=True)
    try:
        result = subprocess.run(cmd, cwd=str(cwd), timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"{label} timed out after {timeout}s. Command: {' '.join(cmd)}"
        )
    if result.returncode != 0:
        pytest.fail(
            f"{label} exited with rc={result.returncode}. "
            f"See preceding output. Command: {' '.join(cmd)}"
        )
    print(f"[{label}] done (rc=0)", flush=True)
    return result


def ensure_mesh_for_nord(case_workspace, nord):
    """Generate `mesh_p{nord}.msh` inside the workspace if it doesn't exist.

    The case directory ships per-nord `.geo` files (tuned for each FEM
    order so higher-nord meshes are appropriately coarser).  The .msh
    files are NOT shipped — they're large, regenerable, and order-
    specific.  This helper runs `gmsh mesh_p{nord}.geo -3` once per
    workspace, writing the .msh next to the .geo in the tmp workspace.

    Skips silently if the .msh is already present (so pre-generated
    meshes from local development still work) or if the .geo is absent
    (test will fail later with a clearer error).  Skips the test if the
    gmsh binary is unavailable.
    """
    msh = case_workspace / f"mesh_p{nord}.msh"
    if msh.exists():
        print(f"[gmsh] reusing existing {msh.name}", flush=True)
        return
    geo = case_workspace / f"mesh_p{nord}.geo"
    if not geo.exists():
        pytest.skip(
            f"mesh_p{nord}.geo not present in {case_workspace} "
            f"and {msh.name} not pre-generated — cannot run at nord={nord}"
        )
    gmsh = shutil.which("gmsh")
    if gmsh is None:
        pytest.skip(
            f"gmsh binary not found on PATH; cannot generate {msh.name} "
            f"for nord={nord}"
        )
    timeout = GMSH_TIMEOUT_BY_NORD.get(nord, max(GMSH_TIMEOUT_BY_NORD.values()))
    _run_with_timeout([gmsh, geo.name, "-3", "-o", msh.name],
                      cwd=case_workspace, timeout=timeout,
                      label=f"gmsh[nord={nord}]")
    assert msh.exists(), f"gmsh ran but did not produce {msh}"


def run_pipeline_for_nord(repo_root, fm_csem_binary, case_workspace, nord,
                          mpi_nproc=4):
    """Driver shared by the smoke test and the reference-regression test.

    Runs gmsh → preprocess → fm.csem for the given polynomial order
    using per-nord mesh, bundle and params filenames so multiple
    invocations within the same workspace do not clobber each other.

    Returns (responses_path, bundle_path).
    """
    bundle_name    = f"input_p{nord}.h5"
    params_name    = f"params_p{nord}.txt"
    responses_base = f"responses_p{nord}"
    kernel_timeout = FM_CSEM_TIMEOUT_BY_NORD.get(nord,
                                                  max(FM_CSEM_TIMEOUT_BY_NORD.values()))

    ensure_mesh_for_nord(case_workspace, nord)

    preprocess = repo_root / "utils" / "preprocess.py"
    cmd = [
        sys.executable, str(preprocess),
        "-mode", "forward",
        "-nord", str(nord),
        "-case_dir", str(case_workspace),
        "-mesh_filename", f"mesh_p{nord}.msh",
        "-source_filename", "sources.txt",
        "-receiver_filename", "receivers.txt",
        "-sigma_file", "sigmas.csv",
        "-input_filename", bundle_name,
        "-params_filename", params_name,
    ]
    _run_with_timeout(cmd, cwd=repo_root, timeout=PREPROCESS_TIMEOUT,
                      label=f"preprocess[nord={nord}]")
    assert (case_workspace / bundle_name).exists()
    assert (case_workspace / params_name).exists()

    cmd = [
        "mpirun", "-n", str(mpi_nproc), str(fm_csem_binary),
        "-options_file", str(case_workspace / params_name),
    ]
    _run_with_timeout(cmd, cwd=case_workspace, timeout=kernel_timeout,
                      label=f"fm.csem[nord={nord}]")

    responses_path = case_workspace / f"{responses_base}_src1.h5"
    assert responses_path.exists(), f"expected responses at {responses_path}"
    return responses_path, case_workspace / bundle_name
