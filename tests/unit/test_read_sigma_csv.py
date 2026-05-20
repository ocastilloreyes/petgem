"""Unit tests for petgem.readSigmaCSV."""
import numpy as np
import pytest

import petgem


def test_basic_three_column(tmp_path):
    csv = tmp_path / "sigmas.csv"
    csv.write_text(
        "# sigma_x, sigma_y, sigma_z\n"
        "0.1, 0.2, 0.3\n"
        "1.0, 1.5, 2.0\n"
    )
    sx, sy, sz = petgem.readSigmaCSV(str(csv))
    np.testing.assert_allclose(sx, [0.1, 1.0])
    np.testing.assert_allclose(sy, [0.2, 1.5])
    np.testing.assert_allclose(sz, [0.3, 2.0])


def test_blank_lines_and_comments_ignored(tmp_path):
    csv = tmp_path / "sigmas.csv"
    csv.write_text(
        "# leading comment\n"
        "\n"
        "0.1, 0.1, 0.1\n"
        "# mid-stream comment\n"
        "1.0, 1.0, 1.0\n"
    )
    sx, sy, sz = petgem.readSigmaCSV(str(csv))
    assert len(sx) == 2
    np.testing.assert_allclose(sx, [0.1, 1.0])


def test_single_row(tmp_path):
    csv = tmp_path / "sigmas.csv"
    csv.write_text("0.5, 0.5, 0.5\n")
    sx, sy, sz = petgem.readSigmaCSV(str(csv))
    np.testing.assert_allclose(sx, [0.5])
    np.testing.assert_allclose(sy, [0.5])
    np.testing.assert_allclose(sz, [0.5])


def test_wrong_column_count_raises(tmp_path):
    csv = tmp_path / "sigmas.csv"
    csv.write_text("0.1, 0.2\n0.3, 0.4\n")
    with pytest.raises(ValueError, match="3 columns"):
        petgem.readSigmaCSV(str(csv))


def test_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        petgem.readSigmaCSV(str(tmp_path / "does_not_exist.csv"))


def test_returns_float64(tmp_path):
    csv = tmp_path / "sigmas.csv"
    csv.write_text("1, 2, 3\n")  # integers
    sx, sy, sz = petgem.readSigmaCSV(str(csv))
    assert sx.dtype == np.float64
    assert sy.dtype == np.float64
    assert sz.dtype == np.float64
