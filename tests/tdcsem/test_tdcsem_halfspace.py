"""Time-domain CSEM half-space end-to-end test (fm.tdcsem).

Runs fm.tdcsem on a 10 ohm-m half-space under air (tests/tdcsem/mesh.geo) for
an x-directed 1 A.m dipole switched on by a 1 us ramp, direct LU (MUMPS), and
checks at the three inline receivers (200, 300 and 400 m):
  * Ex (3 %) and dBy/dt (3.5 %) at order 2 against empymod
    (reference/empymod_golden.json) for t in [1e-4, 1e-2] s,
  * the output layout and a real-valued solution,
  * the identical result on 1 and 3 ranks (order 1).

``build_bundle.sh`` builds tests/tdcsem/input.h5; ``make_reference.py``
regenerates the golden.
"""
import json

import numpy as np
import pytest

from fmcsem_testlib import DIRECT_SOLVER, REPO_ROOT, run_kernel

pytestmark = pytest.mark.slow

TD_DIR = REPO_ROOT / "tests" / "tdcsem"
BUNDLE = TD_DIR / "input.h5"
GOLDEN = TD_DIR / "reference" / "empymod_golden.json"
FIELDS = ("Ex", "Ey", "Ez", "dBxdt", "dBydt", "dBzdt")
STEPPING = ["-tdcsem_t_end", "1.0e-2", "-tdcsem_atsd_steps", "25", "-tdcsem_atsd_rtol", "0.01"]
T_MIN = 1.0e-4
RTOL = {"Ex": 0.03, "dBydt": 0.035}


@pytest.fixture(scope="session")
def td_bundle():
    if not BUNDLE.exists():
        pytest.skip(f"{BUNDLE} not present - run `bash tests/tdcsem/build_bundle.sh` first")
    return BUNDLE


@pytest.fixture(scope="session")
def td_run(fm_tdcsem_binary, td_bundle, tmp_path_factory):
    """Run fm.tdcsem once per (order, rank count); cache + return getter."""
    outdir = tmp_path_factory.mktemp("tdcsem_runs")
    cache = {}

    def _run(order, nprocs):
        if (order, nprocs) not in cache:
            stem = f"halfspace_p{order}_np{nprocs}"
            args = ["-order", str(order), "-input_filename", str(td_bundle), "-output_dir", str(outdir),
                    "-output_filename", stem, *STEPPING, *DIRECT_SOLVER]
            cache[(order, nprocs)] = (run_kernel(fm_tdcsem_binary, args, nprocs), outdir / f"{stem}.h5")
        return cache[(order, nprocs)]

    return _run


def _read(h5py, path):
    out = {}
    with h5py.File(str(path), "r") as f:
        t = np.asarray(f["times"])
        out["times"] = t[:, 0] if t.ndim == 2 else t
        for name in FIELDS:
            a = np.asarray(f[f"sources/src1/fields/{name}"])
            out[name] = a[..., 0] + 1j * a[..., 1] if a.ndim == 3 else a
        out["simulation_type"] = f.attrs["simulation_type"]
        out["num_times"] = int(np.asarray(f.attrs["num_times"]).ravel()[0])
    return out


def _check(proc, out, label):
    assert proc.returncode == 0, f"fm.tdcsem failed ({label}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    assert out.exists(), f"no fm.tdcsem output produced ({label})"


@pytest.mark.e2e
def test_tdcsem_halfspace(td_run, h5py_mod):
    proc, out = td_run(2, 1)
    _check(proc, out, "order 2")
    r = _read(h5py_mod, out)
    golden = json.loads(GOLDEN.read_text())

    sim = r["simulation_type"]
    assert (sim.decode() if isinstance(sim, bytes) else str(sim)) == "fm.tdcsem"
    assert r["times"].shape == (r["num_times"],)
    for name in FIELDS:
        assert r[name].shape == (len(golden["offsets"]), r["num_times"]), f"{name} has shape {r[name].shape}"
        assert np.all(r[name].imag == 0.0), f"{name} is not real"

    t = np.asarray(golden["times"])
    sel = t >= T_MIN
    for name, rtol in RTOL.items():
        for j, x in enumerate(golden["offsets"]):
            ref = np.asarray(golden[name][j])[sel]
            fe = np.interp(np.log(t[sel]), np.log(r["times"][1:]), r[name][j, 1:].real)
            err = np.max(np.abs(fe - ref) / np.abs(ref))
            assert err < rtol, f"{name} at x = {x} m: max relative error {err:.4f} >= {rtol}"


@pytest.mark.e2e
def test_tdcsem_halfspace_rank_invariant(td_run, h5py_mod):
    p1, o1 = td_run(1, 1)
    p3, o3 = td_run(1, 3)
    _check(p1, o1, "order 1, 1 rank")
    _check(p3, o3, "order 1, 3 ranks")
    a, b = _read(h5py_mod, o1), _read(h5py_mod, o3)
    assert np.array_equal(a["times"], b["times"])
    for name in FIELDS:
        scale = np.max(np.abs(a[name]))
        assert np.max(np.abs(a[name] - b[name])) / scale < 1e-8, f"{name} differs between 1 and 3 ranks"
