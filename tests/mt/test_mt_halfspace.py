"""MT half-space end-to-end test (fm.mt).

Runs fm.mt on a 100 ohm-m half-space under air at 10 Hz (tests/mt/mesh.geo),
order 2, direct LU (MUMPS), with the 'h' 1D equation, on 1 and 3 ranks, and
checks at the three surface receivers:
  * apparent resistivity rho_xy = rho_yx = 100 ohm-m (1 %),
  * phase phi_xy = 135 deg and phi_yx = -45 deg (0.5 deg),
  * Zxx, Zyy and the tipper vanish relative to Zxy,
  * the output layout and the identical result on 1 and 3 ranks.

``build_bundle.sh`` builds tests/mt/input.h5.
"""
import numpy as np
import pytest

from fmcsem_testlib import DIRECT_SOLVER, FIELD_COMPONENTS, REPO_ROOT, run_kernel

pytestmark = pytest.mark.slow

MT_DIR = REPO_ROOT / "tests" / "mt"
BUNDLE = MT_DIR / "input.h5"
RHO = 100.0


@pytest.fixture(scope="session")
def mt_bundle():
    if not BUNDLE.exists():
        pytest.skip(f"{BUNDLE} not present - run `bash tests/mt/build_bundle.sh` first")
    return BUNDLE


@pytest.fixture(scope="session")
def mt_run(fm_mt_binary, mt_bundle, tmp_path_factory):
    """Run fm.mt once per rank count; cache + return getter."""
    outdir = tmp_path_factory.mktemp("mt_runs")
    cache = {}

    def _run(nprocs):
        if nprocs not in cache:
            stem = f"halfspace_np{nprocs}"
            args = ["-order", "2", "-mt_1d_equation", "h", "-input_filename", str(mt_bundle),
                    "-output_dir", str(outdir), "-output_filename", stem, *DIRECT_SOLVER]
            cache[nprocs] = (run_kernel(fm_mt_binary, args, nprocs), outdir / f"{stem}.h5")
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
            for comp in FIELD_COMPONENTS:
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

    for comp in ("xy", "yx"):
        rho = r[f"apparent_resistivity/{comp}"].real
        assert np.all(np.abs(rho - RHO) / RHO < 1e-2), f"rho_{comp} = {rho} (expected {RHO})"
    assert np.all(np.abs(r["phase/xy"].real - 135.0) < 0.5), f"phi_xy = {r['phase/xy'].real}"
    assert np.all(np.abs(r["phase/yx"].real + 45.0) < 0.5), f"phi_yx = {r['phase/yx'].real}"

    zxy = np.abs(r["impedance/xy"])
    for comp in ("xx", "yy"):
        assert np.all(np.abs(r[f"impedance/{comp}"]) / zxy < 1e-2), f"|Z{comp}/Zxy| too large"
    for comp in ("x", "y"):
        assert np.all(np.abs(r[f"tipper/{comp}"]) < 1e-2), f"|T{comp}| too large"


@pytest.mark.e2e
def test_mt_halfspace_rank_invariant(mt_run, h5py_mod):
    p1, o1 = mt_run(1)
    p3, o3 = mt_run(3)
    assert p1.returncode == 0 and p3.returncode == 0
    a, b = _read(h5py_mod, o1), _read(h5py_mod, o3)
    for comp in ("xx", "xy", "yx", "yy"):
        za, zb = a[f"impedance/{comp}"], b[f"impedance/{comp}"]
        scale = np.max(np.abs(a["impedance/xy"]))
        assert np.max(np.abs(za - zb)) / scale < 1e-8, f"Z{comp} differs between 1 and 3 ranks"
