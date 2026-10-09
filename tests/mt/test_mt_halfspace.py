"""MT half-space end-to-end test (fm.mt).

Runs fm.mt on a 100 ohm-m half-space under air at 10 Hz (tests/mt/mesh.geo),
order 2, direct LU (MUMPS), with the 'h' 1D equation, on 1 and 3 ranks, and
checks at the three surface receivers:
  * apparent resistivity rho_xy = rho_yx = 100 ohm-m (1 %),
  * phase phi_xy = 135 deg and phi_yx = -45 deg (0.5 deg),
  * Zxx, Zyy and the tipper vanish relative to Zxy,
  * the output layout and the identical result on 1 and 3 ranks.

The bundle tests/mt/input.h5 is produced by ``make_bundle.sh``.
"""
import os
import subprocess

import numpy as np
import pytest

from fmcsem_testlib import REPO_ROOT

pytestmark = pytest.mark.slow

MT_DIR = REPO_ROOT / "tests" / "mt"
BUNDLE = MT_DIR / "input.h5"
RHO = 100.0
SOLVER_OPTS = ["-dm_mat_type", "aij", "-ksp_type", "preonly", "-pc_type", "lu",
               "-pc_factor_mat_solver_type", "mumps", "-ksp_error_if_not_converged"]


@pytest.fixture(scope="session")
def mt_bundle():
    if not BUNDLE.exists():
        pytest.skip(f"{BUNDLE} not present - run `bash tests/mt/make_bundle.sh` first")
    return BUNDLE


@pytest.fixture(scope="session")
def mt_run(fm_mt_binary, mt_bundle, tmp_path_factory):
    """Run fm.mt once per rank count; cache + return getter."""
    outdir = tmp_path_factory.mktemp("mt_runs")
    cache = {}

    def _run(nprocs):
        if nprocs in cache:
            return cache[nprocs]
        stem = f"halfspace_np{nprocs}"
        launch = (["mpirun", "-n", str(nprocs)] if nprocs > 1 else []) + [str(fm_mt_binary)]
        cmd = launch + ["-order", "2", "-mt_1d_equation", "h",
                        "-input_filename", str(mt_bundle),
                        "-output_dir", str(outdir), "-output_filename", stem] + SOLVER_OPTS
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT))
        cache[nprocs] = (proc, outdir / f"{stem}.h5")
        return cache[nprocs]

    return _run


def _read(h5py, path):
    out = {}
    with h5py.File(str(path), "r") as f:
        for group in ("impedance", "apparent_resistivity", "phase"):
            for name in ("xx", "xy", "yx", "yy"):
                a = np.asarray(f[group][name])
                out[f"{group}/{name}"] = a[:, 0] + 1j * a[:, 1]
        for name in ("x", "y"):
            a = np.asarray(f["tipper"][name])
            out[f"tipper/{name}"] = a[:, 0] + 1j * a[:, 1]
        for pol in ("x", "y"):
            for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                assert comp in f[f"polarizations/{pol}/fields"], f"missing {pol}/{comp}"
        out["simulation_type"] = f.attrs["simulation_type"]
        out["frequency"] = float(np.asarray(f.attrs["frequency"]).ravel()[0])
    return out


@pytest.mark.e2e
@pytest.mark.parametrize("nprocs", [1, 3])
def test_mt_halfspace(mt_run, h5py_mod, nprocs):
    proc, out = mt_run(nprocs)
    assert proc.returncode == 0, f"fm.mt failed on {nprocs} ranks:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    assert out.exists(), f"no MT output produced on {nprocs} ranks"

    r = _read(h5py_mod, out)
    sim = r["simulation_type"]
    assert (sim.decode() if isinstance(sim, bytes) else str(sim)) == "fm.mt"
    assert r["frequency"] == pytest.approx(10.0)

    for c in ("xy", "yx"):
        rho = r[f"apparent_resistivity/{c}"].real
        assert np.all(np.abs(rho - RHO) / RHO < 1e-2), f"rho_{c} = {rho} (expected {RHO})"
    assert np.all(np.abs(r["phase/xy"].real - 135.0) < 0.5), f"phi_xy = {r['phase/xy'].real}"
    assert np.all(np.abs(r["phase/yx"].real + 45.0) < 0.5), f"phi_yx = {r['phase/yx'].real}"

    zxy = np.abs(r["impedance/xy"])
    for c in ("xx", "yy"):
        assert np.all(np.abs(r[f"impedance/{c}"]) / zxy < 1e-2), f"|Z{c}/Zxy| too large"
    for c in ("x", "y"):
        assert np.all(np.abs(r[f"tipper/{c}"]) < 1e-2), f"|T{c}| too large"


@pytest.mark.e2e
def test_mt_halfspace_rank_invariant(mt_run, h5py_mod):
    p1, o1 = mt_run(1)
    p3, o3 = mt_run(3)
    assert p1.returncode == 0 and p3.returncode == 0
    a, b = _read(h5py_mod, o1), _read(h5py_mod, o3)
    for c in ("xx", "xy", "yx", "yy"):
        za, zb = a[f"impedance/{c}"], b[f"impedance/{c}"]
        scale = np.max(np.abs(a["impedance/xy"]))
        assert np.max(np.abs(za - zb)) / scale < 1e-8, f"Z{c} differs between 1 and 3 ranks"
