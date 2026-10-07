"""Plain helpers shared by the FM-CSEM test suite (no pytest dependency).

Kept separate from conftest.py so both the fixtures and the individual test
modules can import the same constants/functions (``from fmcsem_testlib import
ORDERS, ...``). conftest.py puts this directory on sys.path.
"""
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIT_CUBE = REPO_ROOT / "examples" / "unit_cube"
#: Generated fm.csem input bundle (built into the case's outputs/ by preprocess).
INPUT_BUNDLE = UNIT_CUBE / "outputs" / "input.h5"
CSRC = REPO_ROOT / "tests" / "csrc"

#: All orders validated at the FE-core level (levels 1-3, fast C harnesses).
ORDERS = [1, 2, 3, 4, 5, 6]

#: Orders validated end-to-end through the fm.csem pipeline (levels 4-5).
#: 1/2/3 exercise every DOF entity class - edge-only (p1), +face (p2),
#: +interior (p3) - so the assembly / L2G / BC / solve / interpolation code
#: paths are all covered; higher orders only add more DOFs of the same classes
#: (validated by levels 1-3) at a runtime the fixed 1536-cell mesh makes slow.
PIPELINE_ORDERS = [1, 2, 3]

#: Committed golden references (exact LU solves of the current code).
REFERENCE_DIR = UNIT_CUBE / "reference"

#: C harness -> production sources it links against (all unchanged).
HARNESS_SOURCES = {
    "test_basis":    ["fe_nedelec.c", "fe_nodal.c"],
    "test_dofs":     ["fe_nedelec.c", "fe_nodal.c"],
    "test_elements": ["fem.c", "fe_nedelec.c", "fe_nodal.c"],
}

#: Field components in a responses file.
FIELD_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def petsc_paths():
    """Return (mpicc, petsc_dir, petsc_arch) or None if the toolchain is absent."""
    petsc_dir = os.environ.get("PETSC_DIR")
    petsc_arch = os.environ.get("PETSC_ARCH", "")
    if not petsc_dir:
        return None
    cc = Path(petsc_dir) / petsc_arch / "bin" / "mpicc"
    if not cc.exists():
        found = shutil.which("mpicc")
        if not found:
            return None
        cc = Path(found)
    return str(cc), petsc_dir, petsc_arch


def build_harness(name, outdir):
    """Compile one C harness against the unchanged production sources.

    Returns (exe_path, "") on success or (None, stderr) on failure.
    """
    paths = petsc_paths()
    if paths is None:
        return None, "no PETSc toolchain"
    cc, petsc_dir, petsc_arch = paths
    exe = Path(outdir) / name
    inc = ["-I", str(REPO_ROOT / "include"), "-I", str(CSRC),
           "-I", f"{petsc_dir}/include", "-I", f"{petsc_dir}/{petsc_arch}/include"]
    lib = [f"-L{petsc_dir}/{petsc_arch}/lib",
           f"-Wl,-rpath,{petsc_dir}/{petsc_arch}/lib", "-lpetsc", "-lm"]
    cmd = [cc, "-Wall", "-O2", *inc, str(CSRC / f"{name}.c"),
           *[str(REPO_ROOT / "src" / s) for s in HARNESS_SOURCES[name]], *lib, "-o", str(exe)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, proc.stderr
    return exe, ""


def run_harness(exe, order):
    """Run a compiled harness for one order; return the CompletedProcess."""
    return subprocess.run([str(exe), str(order)], capture_output=True, text=True, timeout=120)


def load_fields(h5py, path):
    """Return {'Ex': complex ndarray, ...} for src1 of a responses file.

    Each dataset is (n_recv, 2) = (real, imag) from a PETSc complex Vec.
    """
    import numpy as np
    out = {}
    with h5py.File(str(path), "r") as f:
        g = f["/sources/src1/fields"]
        for comp in FIELD_COMPONENTS:
            a = np.asarray(g[comp])
            out[comp] = a[:, 0] + 1j * a[:, 1]
    return out


def parse_grid_stats(stdout):
    """Extract mesh and FE-space counts from the fm.csem run report.

    Parses the 'Mesh', 'FE space' and 'DOFs/entity' lines (grouped thousands
    ok). Missing keys are simply absent so callers can assert on what they need.
    """
    import re

    def num(txt):
        return int(re.sub(r"[ ,]", "", txt))

    grp = r"(\d+(?:[ ,]\d{3})*)"
    stats = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        label = lhs.strip()
        if label == "Mesh":
            for key in ("cells", "faces", "edges", "vertices"):
                m = re.search(grp + r" " + key, rhs)
                if m:
                    stats[key] = num(m.group(1))
        elif label == "FE space":
            m = re.search(r"p=(\d+), " + grp + r" DOFs", rhs)
            if m:
                stats["order"] = int(m.group(1))
                stats["dofs"] = num(m.group(2))
        elif label == "DOFs/entity":
            for key in ("vertex", "edge", "face", "volume"):
                m = re.search(key + r" (\d+)", rhs)
                if m:
                    stats["dof_" + key] = int(m.group(1))
            m = re.search(r"\((\d+) per cell\)", rhs)
            if m:
                stats["dof_cell"] = int(m.group(1))
    return stats
