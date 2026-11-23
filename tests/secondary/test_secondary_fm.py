"""fm.csem secondary-field formulation on the im_csem_block true model.

Order 1, 1 Hz, MUMPS, 2 ranks, background without the anomaly
(tests/secondary/fm.h5, ``build_bundle.sh``):

  * model background: the total fields equal those of the total-field
    formulation; a background equal to the model, in a separate file
    (-csem_background_file), gives zero secondary fields;
  * layered background: the secondary fields agree with those of the model
    background, the total fields with those of the total-field formulation
    within its order-1 error, and the result does not depend on the number
    of ranks;
  * a background without the selected model, or a file without
    /background, is rejected.
"""
import sys

import numpy as np
import pytest

from fmcsem_testlib import DIRECT_SOLVER, FIELD_COMPONENTS, REPO_ROOT, find_binary, load_fields, run_kernel

sys.path.insert(0, str(REPO_ROOT / "utils"))

pytestmark = pytest.mark.slow

BUNDLE = REPO_ROOT / "tests" / "secondary" / "fm.h5"
TRUE_SIGMA = REPO_ROOT / "examples" / "im_csem_block" / "survey" / "sigmas_true.txt"
NPROCS = 2
IDENTITY_RTOL = 1e-8
ANOMALY_CELLS = "1 616"
# Layered against model background: secondary fields (E_p,h analytic against the order-1 background solution)
SECONDARY_RTOL = 5e-2
# Layered against total-field formulation: total fields (order-1 error of the total field at the receivers)
TOTAL_RTOL = {"Ex": 5e-2, "Ey": 2e-1, "Ez": 2e-1, "Hx": 3e-1, "Hy": 5e-2, "Hz": 5e-2}


def _max_rel(a, b):
    return {c: float(np.abs(a[c] - b[c]).max() / np.abs(b[c]).max()) for c in FIELD_COMPONENTS}


@pytest.fixture(scope="module")
def fm(tmp_path_factory, h5py_mod):
    """Cached fm.csem runs on fm.h5: run(name, args, nprocs) -> (proc, responses path)."""
    binary = find_binary("fm.csem", "PETGEM_FM_CSEM")
    if binary is None:
        pytest.skip("fm.csem binary not found (set PETGEM_FM_CSEM or build build/fm.csem)")
    if not BUNDLE.exists():
        pytest.skip(f"{BUNDLE} not present - run `bash tests/secondary/build_bundle.sh` first")
    outdir = tmp_path_factory.mktemp("secondary_fm")
    cache = {}

    def run(name, args=(), nprocs=NPROCS, expect_ok=True):
        if name not in cache:
            proc = run_kernel(binary, ["-input_filename", str(BUNDLE), "-output_dir", str(outdir), "-output_filename", name,
                                       *args, *DIRECT_SOLVER], nprocs)
            if expect_ok:
                assert proc.returncode == 0, f"fm.csem {name} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
            cache[name] = (proc, outdir / f"{name}.h5")
        return cache[name]

    run.outdir = outdir
    return run


@pytest.mark.e2e
def test_model_background_reproduces_total(fm, h5py_mod):
    _, total = fm("total")
    proc, model = fm("model", ["-csem_formulation", "secondary", "-csem_primary", "model"])
    t = load_fields(h5py_mod, total)
    err = _max_rel(load_fields(h5py_mod, model), t)
    secondary = load_fields(h5py_mod, model, "secondary_fields")
    assert max(err.values()) < IDENTITY_RTOL, f"total from the secondary formulation against the total field: {err}"
    assert all(np.abs(secondary[c]).max() > 1e-3 * np.abs(t[c]).max() for c in FIELD_COMPONENTS)
    assert "model, 4 material(s)" in proc.stdout and f"Anomalous cells          = {ANOMALY_CELLS}" in proc.stdout
    with h5py_mod.File(str(model), "r") as f:
        assert f.attrs["formulation"] in ("secondary", b"secondary")
    with h5py_mod.File(str(total), "r") as f:
        assert "formulation" not in f.attrs and "secondary_fields" not in f["/sources/src1"]


@pytest.mark.e2e
def test_background_equal_to_model_gives_zero_secondary(fm, h5py_mod):
    import functions as petgem
    background = fm.outdir / "background_true.h5"
    petgem.writeBackgroundPayload(str(background), material_sigma=np.column_stack(petgem.readSigmaTable(str(TRUE_SIGMA))[:3]))
    _, total = fm("total")
    proc, equal = fm("equal", ["-csem_formulation", "secondary", "-csem_background_file", str(background)])
    secondary = load_fields(h5py_mod, equal, "secondary_fields")
    err = _max_rel(load_fields(h5py_mod, equal), load_fields(h5py_mod, total))
    assert "Anomalous cells          = 0" in proc.stdout
    assert all(np.abs(secondary[c]).max() == 0.0 for c in FIELD_COMPONENTS)
    assert max(err.values()) < IDENTITY_RTOL, err


@pytest.mark.e2e
def test_layered_background(fm, h5py_mod):
    _, total = fm("total")
    _, model = fm("model", ["-csem_formulation", "secondary", "-csem_primary", "model"])
    proc, layered = fm("layered", ["-csem_formulation", "secondary"])
    assert "layered, 2 layer(s)" in proc.stdout and "WARNING" not in proc.stdout
    assert f"Anomalous cells          = {ANOMALY_CELLS}" in proc.stdout

    sec = _max_rel(load_fields(h5py_mod, layered, "secondary_fields"), load_fields(h5py_mod, model, "secondary_fields"))
    assert max(sec.values()) < SECONDARY_RTOL, f"secondary fields, layered against model background: {sec}"
    tot = _max_rel(load_fields(h5py_mod, layered), load_fields(h5py_mod, total))
    assert all(tot[c] < TOTAL_RTOL[c] for c in FIELD_COMPONENTS), f"total fields, layered against total field: {tot}"


@pytest.mark.e2e
def test_layered_rank_independence(fm, h5py_mod):
    _, two = fm("layered", ["-csem_formulation", "secondary"])
    _, three = fm("layered_np3", ["-csem_formulation", "secondary"], nprocs=3)
    err = _max_rel(load_fields(h5py_mod, three), load_fields(h5py_mod, two))
    assert max(err.values()) < IDENTITY_RTOL, f"3 against 2 ranks: {err}"


@pytest.mark.e2e
def test_rejections(fm):
    import functions as petgem
    materials = fm.outdir / "background_materials.h5"
    petgem.writeBackgroundPayload(str(materials), material_sigma=np.ones((4, 3)))
    proc, _ = fm("bad_layered", ["-csem_formulation", "secondary", "-csem_primary", "layered", "-csem_background_file", str(materials)],
                 nprocs=1, expect_ok=False)
    assert proc.returncode != 0 and "needs /background/layer_sigma" in proc.stdout + proc.stderr
    _, total = fm("total")
    proc, _ = fm("no_background", ["-csem_formulation", "secondary", "-csem_background_file", str(total)], nprocs=1, expect_ok=False)
    assert proc.returncode != 0 and "has no /background/layer_sigma or /background/material_sigma" in proc.stdout + proc.stderr
