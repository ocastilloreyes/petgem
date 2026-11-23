"""fm.csem response readers: total and secondary fields."""
import os
import sys

import h5py
import numpy as np
import pytest
from petsc4py import PETSc

UTILS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils")
sys.path.insert(0, UTILS)
import functions as petgem  # noqa: E402

FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def write_responses(path, num_sources, secondary):
    """Responses file in the fm.csem layout; secondary_fields = -fields when secondary."""
    viewer = PETSc.ViewerHDF5().create(str(path), "w", comm=PETSc.COMM_SELF)
    for k in range(1, num_sources + 1):
        groups = [("fields", 1.0)] + ([("secondary_fields", -1.0)] if secondary else [])
        for group, scale in groups:
            viewer.pushGroup(f"/sources/src{k}/{group}")
            for i, name in enumerate(FIELDS):
                vec = PETSc.Vec().createWithArray(scale * (10 * k + i + np.arange(3.0)), comm=PETSc.COMM_SELF)
                vec.setName(name)
                vec.view(viewer)
                vec.destroy()
            viewer.popGroup()
    viewer.destroy()
    with h5py.File(path, "a") as f:
        f.attrs["num_sources"] = num_sources
        if secondary:
            f.attrs["formulation"] = "secondary"
        for k in range(1, num_sources + 1):
            f[f"/sources/src{k}"].attrs["frequency"] = 2.0
    return str(path)


def test_reads_total_and_secondary_fields(tmp_path):
    path = write_responses(tmp_path / "secondary.h5", 2, secondary=True)
    total = petgem.readResponses(path, source=2)
    secondary = petgem.readResponses(path, source=2, group="secondary_fields")
    for i, name in enumerate(FIELDS):
        assert np.array_equal(np.real(total[name]), 20 + i + np.arange(3.0))
        assert np.array_equal(secondary[name], -total[name])
    assert secondary["provenance"]["formulation"] == "secondary"
    assert secondary["source"]["frequency"] == 2.0


def test_reads_all_secondary_fields(tmp_path):
    path = write_responses(tmp_path / "secondary.h5", 2, secondary=True)
    responses = petgem.readAllResponses(path, group="secondary_fields")
    assert responses["num_sources"] == 2
    assert np.array_equal(np.real(responses["sources"][1]["Ex"]), -(10 + np.arange(3.0)))


def test_total_file_has_no_secondary_fields(tmp_path):
    path = write_responses(tmp_path / "total.h5", 1, secondary=False)
    assert np.array_equal(np.real(petgem.readResponses(path)["Hz"]), 15 + np.arange(3.0))
    with pytest.raises(KeyError, match="secondary_fields.*formulation 'total'"):
        petgem.readResponses(path, group="secondary_fields")
