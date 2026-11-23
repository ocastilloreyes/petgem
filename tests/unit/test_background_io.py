"""Background model input of preprocess.py (secondary-field formulation)."""
import os
import sys

import h5py
import meshio
import numpy as np
import pytest

UTILS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils")
sys.path.insert(0, UTILS)
import functions as petgem  # noqa: E402

LAYERS = """# z_top sigma_h [sigma_v]
inf    1e-8          # air
0      3.3333
-1000  1.0    0.5    # VTI
"""


def write(path, text):
    path.write_text(text)
    return str(path)


def unit_cube_case(case_dir):
    """Six-tetrahedron unit cube (materials 10 and 20), survey and sigma tables."""
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    tets = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
                     [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])
    meshio.write(str(case_dir / "mesh.vtu"),
                 meshio.Mesh(points, [("tetra", tets)], cell_data={"cell_scalars": [np.array([10, 10, 10, 20, 20, 20])]}))
    write(case_dir / "receivers.txt", "0.5 0.5 0.9\n")
    write(case_dir / "sources.txt", "2.0 0.5 0.5 0.1 1.0 1.0 0.0 0.0\n")
    write(case_dir / "sigmas.txt", "1.0 1.0 1.0\n2.0 2.0 2.0\n")
    write(case_dir / "layers.txt", LAYERS)
    write(case_dir / "background_sigmas.txt", "1.0 1.0 1.0\n1.0 1.0 0.5\n")


def preprocess(case_dir, mode="fm.csem", **kwargs):
    sx, sy, sz, _ = petgem.readSigmaTable(str(case_dir / "sigmas.txt"))
    inputs = {"source_filename": "sources.txt"} if mode == "fm.csem" else {"mt_frequency_filename": "freq.txt"}
    petgem.runPreprocessing(mode=mode, order=1, case_dir=str(case_dir), mesh_filename="mesh.vtu",
                            receiver_filename="receivers.txt", sigma_x=sx, sigma_y=sy, sigma_z=sz,
                            **inputs, **kwargs)


def test_reads_layers_with_default_vertical_conductivity(tmp_path):
    ztop, sigma_h, sigma_v = petgem.readBackgroundLayers(write(tmp_path / "layers.txt", LAYERS))
    assert ztop[0] == np.inf and np.array_equal(ztop[1:], [0.0, -1000.0])
    assert np.array_equal(sigma_h, [1e-8, 3.3333, 1.0])
    assert np.array_equal(sigma_v, [1e-8, 3.3333, 0.5])


@pytest.mark.parametrize("text, match", [
    ("", "no layer rows"),
    ("inf 1.0 1.0 1.0\n", "2 or 3 columns"),
    ("0 1.0\n-10 1.0\n", "first z_top must be 'inf'"),
    ("inf 1.0\n-10 1.0\n0 1.0\n", "strictly decreasing"),
    ("inf 1.0\n0 1.0\n0 2.0\n", "strictly decreasing"),
    ("inf 1.0\n0 0.0\n", "must be positive"),
])
def test_rejects_invalid_layers(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        petgem.readBackgroundLayers(write(tmp_path / "layers.txt", text))


def test_writes_background_payload(tmp_path):
    path = str(tmp_path / "bundle.h5")
    layers = petgem.readBackgroundLayers(write(tmp_path / "layers.txt", LAYERS))
    material_sigma = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    petgem.writeBackgroundPayload(path, layers=layers, material_sigma=material_sigma)
    ztop = np.real(petgem.readVectorH5(path, "layer_ztop", group="/background"))
    sigma = np.real(petgem.readVectorH5(path, "layer_sigma", group="/background"))
    materials = np.real(petgem.readVectorH5(path, "material_sigma", group="/background"))
    assert ztop[0] == np.inf and np.array_equal(ztop[1:], [0.0, -1000.0])
    assert np.array_equal(sigma, [1e-8, 1e-8, 3.3333, 3.3333, 1.0, 0.5])
    assert np.array_equal(materials, material_sigma.reshape(-1))


def test_params_file_selects_formulation(tmp_path):
    petgem.writeParamsFile("fm.csem", str(tmp_path), "out", "input.h5", "total.txt")
    petgem.writeParamsFile("im.csem", str(tmp_path), "out", "input.h5", "secondary.txt", formulation="secondary")
    assert "-csem_formulation" not in (tmp_path / "total.txt").read_text()
    assert "-csem_formulation secondary\n" in (tmp_path / "secondary.txt").read_text()


@pytest.mark.parametrize("mode, inputs", [
    ("fm.mt", {"mt_frequency_filename": "freq.txt"}),
    ("fm.tdcsem", {"source_filename": "sources.txt", "tdcsem_waveform_filename": "w.txt", "tdcsem_t_end": 1.0}),
])
def test_background_rejected_outside_csem(tmp_path, mode, inputs):
    with pytest.raises(ValueError, match="-background_layers_file is not valid"):
        petgem.runPreprocessing(mode=mode, order=1, case_dir=str(tmp_path), mesh_filename="mesh.vtu",
                                receiver_filename="receivers.txt", sigma_x=[1.0], sigma_y=[1.0], sigma_z=[1.0],
                                background_layers_file="layers.txt", **inputs)


def test_preprocessing_embeds_background(tmp_path):
    unit_cube_case(tmp_path)
    preprocess(tmp_path, background_layers_file="layers.txt", background_sigma_file="background_sigmas.txt")
    bundle = str(tmp_path / "input.h5")
    sigma = np.real(petgem.readVectorH5(bundle, "layer_sigma", group="/background"))
    materials = np.real(petgem.readVectorH5(bundle, "material_sigma", group="/background"))
    assert np.array_equal(sigma, [1e-8, 1e-8, 3.3333, 3.3333, 1.0, 0.5])
    assert np.array_equal(materials, [1.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    assert "-csem_formulation secondary\n" in (tmp_path / "params.txt").read_text()


def test_preprocessing_without_background_is_total(tmp_path):
    unit_cube_case(tmp_path)
    preprocess(tmp_path)
    assert "-csem_formulation" not in (tmp_path / "params.txt").read_text()
    with h5py.File(tmp_path / "input.h5", "r") as f:
        assert "background" not in f


def test_rejects_background_sigma_with_other_material_count(tmp_path):
    unit_cube_case(tmp_path)
    write(tmp_path / "background_sigmas.txt", "1.0 1.0 1.0\n")
    with pytest.raises(ValueError, match="1 row\\(s\\), the conductivity table has 2"):
        preprocess(tmp_path, background_sigma_file="background_sigmas.txt")
    assert not (tmp_path / "input.h5").exists()
