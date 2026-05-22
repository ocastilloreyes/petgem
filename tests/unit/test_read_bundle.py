"""Round-trip test for the PETGEM input bundle.

Writes a tiny bundle via writePetgemInputFile, reads it back through
readBundle, and asserts every field round-trips exactly.

The mesh is a single tetrahedron - minimum viable DMPlex - which is enough
to exercise the HDF5 plumbing without requiring a real Gmsh file.
"""
import numpy as np
import pytest

import petgem


@pytest.fixture
def tiny_case(tmp_path):
    """Build a 1-tet test case (4 vertices, 1 cell) with synthetic receivers
    and sources, write it as a bundle, return the resulting path + inputs."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    cells = np.array([[0, 1, 2, 3]], dtype=np.int32)

    num_cells = 1
    conductivity = np.array([[0.1, 0.2, 0.3]], dtype=float)  # one cell
    materials_id = np.array([0], dtype=int)

    receivers = np.array([
        [100.0, 0.0, 0.0],
        [200.0, 0.0, 0.0],
        [300.0, 0.0, 0.0],
    ], dtype=float)

    freq = 1.25
    sources = np.array([
        [10.0, 20.0, -30.0, 1.0, 100.0, 0.0,  0.0],
        [11.0, 21.0, -31.0, 2.0, 200.0, 5.0, 90.0],
    ], dtype=float)

    nord = 2

    plex = petgem.createDM(3, cells, coords, dm_view=False)
    bundle_path = tmp_path / "bundle.h5"
    petgem.writePetgemInputFile(
        plex, conductivity, materials_id,
        receivers, freq, sources, nord,
        str(bundle_path),
        cells=cells, coords=coords,
        output_vtk=None,
    )
    return {
        'path': str(bundle_path),
        'receivers': receivers,
        'sources': sources,
        'freq': freq,
        'nord': nord,
    }


def test_bundle_roundtrip(tiny_case):
    bundle = petgem.readBundle(tiny_case['path'])
    np.testing.assert_allclose(bundle['receivers'], tiny_case['receivers'])
    np.testing.assert_allclose(bundle['sources'],   tiny_case['sources'])
    assert bundle['frequency'] == pytest.approx(tiny_case['freq'])
    assert bundle['nord'] == tiny_case['nord']


def test_bundle_returns_dict_shape(tiny_case):
    bundle = petgem.readBundle(tiny_case['path'])
    assert isinstance(bundle, dict)
    assert set(bundle.keys()) == {'receivers', 'nord', 'frequency', 'sources'}
    assert bundle['receivers'].shape[1] == 3
    assert bundle['sources'].shape[1] == 7
    assert isinstance(bundle['nord'], int)
    assert isinstance(bundle['frequency'], float)


def test_bundle_receivers_count_matches(tiny_case):
    bundle = petgem.readBundle(tiny_case['path'])
    assert bundle['receivers'].shape[0] == tiny_case['receivers'].shape[0]
    assert bundle['sources'].shape[0]   == tiny_case['sources'].shape[0]
