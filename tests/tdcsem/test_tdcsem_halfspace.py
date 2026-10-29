"""Time-domain CSEM half-space end-to-end test (fm.tdcsem).

Runs fm.tdcsem on a 10 ohm-m half-space under air (tests/tdcsem/mesh.geo) for
an x-directed 1 A.m dipole switched on by a 1 us ramp, direct LU (MUMPS), and
checks at the three inline receivers (200, 300 and 400 m):
  * Ex (3 %) and dBy/dt (3.5 %) at order 2 against empymod
    (reference/empymod_golden.json) for t in [1e-4, 1e-2] s,
  * the output layout and a real-valued solution,
  * the identical result on 1 and 3 ranks (order 1),
  * the step-off (DC initial field) against empymod and the superposition
    step-on + step-off = DC field (order 2).

``build_bundle.sh`` builds tests/tdcsem/input.h5; ``make_reference.py``
regenerates the golden.
"""
import json
import shutil

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
RTOL_OFF = {"Ex_off": 0.03}
RTOL_SUM = 1.0e-6


@pytest.fixture(scope="session")
def td_bundle():
    if not BUNDLE.exists():
        pytest.skip(f"{BUNDLE} not present - run `bash tests/tdcsem/build_bundle.sh` first")
    return BUNDLE


@pytest.fixture(scope="session")
def td_off_bundle(td_bundle, h5py_mod, tmp_path_factory):
    """Copy of the bundle with the step-off waveform 1 - s(t)."""
    path = tmp_path_factory.mktemp("tdcsem_off") / "input_off.h5"
    shutil.copy(td_bundle, path)
    with h5py_mod.File(str(path), "r+") as f:
        w = f["tdcsem/waveform"]
        s = np.asarray(w)
        s[:, 0] = 1.0 - s[:, 0]
        w[...] = s
    return path


@pytest.fixture(scope="session")
def td_run(fm_tdcsem_binary, td_bundle, td_off_bundle, tmp_path_factory):
    """Run fm.tdcsem once per (order, rank count, case); cache + return getter."""
    outdir = tmp_path_factory.mktemp("tdcsem_runs")
    cases = {"on": (td_bundle, []), "off": (td_off_bundle, [])}
    cache = {}

    def _run(order, nprocs, case="on"):
        if (order, nprocs, case) not in cache:
            bundle, extra = cases[case]
            stem = f"halfspace_{case}_p{order}_np{nprocs}"
            args = ["-order", str(order), "-input_filename", str(bundle), "-output_dir", str(outdir),
                    "-output_filename", stem, *STEPPING, *DIRECT_SOLVER, *extra]
            cache[(order, nprocs, case)] = (run_kernel(fm_tdcsem_binary, args, nprocs), outdir / f"{stem}.h5")
        return cache[(order, nprocs, case)]

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


def _check_golden(r, golden, rtols, label):
    t = np.asarray(golden["times"])
    sel = t >= T_MIN
    for name, rtol in rtols.items():
        field = name.split("_")[0]
        for j, x in enumerate(golden["offsets"]):
            ref = np.asarray(golden[name][j])[sel]
            fe = np.interp(np.log(t[sel]), np.log(r["times"][1:]), r[field][j, 1:].real)
            err = np.max(np.abs(fe - ref) / np.abs(ref))
            assert err < rtol, f"{name} ({label}) at x = {x} m: max relative error {err:.4f} >= {rtol}"


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
    _check_golden(r, golden, RTOL, "step-on")


@pytest.mark.e2e
def test_tdcsem_halfspace_step_off(td_run, h5py_mod):
    p_on, o_on = td_run(2, 1)
    p_off, o_off = td_run(2, 1, "off")
    _check(p_on, o_on, "order 2, step-on")
    _check(p_off, o_off, "order 2, step-off")
    assert "Initial field            = DC field" in p_off.stdout
    on, off = _read(h5py_mod, o_on), _read(h5py_mod, o_off)
    golden = json.loads(GOLDEN.read_text())
    _check_golden(off, golden, RTOL_OFF, "step-off")

    assert np.array_equal(on["times"], off["times"])
    dc = off["Ex"][:, :1].real
    err = np.max(np.abs(on["Ex"].real + off["Ex"].real - dc) / np.abs(dc))
    assert err < RTOL_SUM, f"Ex step-on + step-off differs from the DC field by {err:.2e} >= {RTOL_SUM}"
    scale = np.max(np.abs(on["dBydt"]))
    err = np.max(np.abs(on["dBydt"].real + off["dBydt"].real)) / scale
    assert err < RTOL_SUM, f"dBy/dt step-on + step-off is not zero: {err:.2e} >= {RTOL_SUM}"


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
