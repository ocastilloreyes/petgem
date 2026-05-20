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
    sx, sy, sz, fixed = petgem.readSigmaCSV(str(csv))
    np.testing.assert_allclose(sx, [0.1, 1.0])
    np.testing.assert_allclose(sy, [0.2, 1.5])
    np.testing.assert_allclose(sz, [0.3, 2.0])
    assert fixed == []


def test_blank_lines_and_comments_ignored(tmp_path):
    csv = tmp_path / "sigmas.csv"
    csv.write_text(
        "# leading comment\n"
        "\n"
        "0.1, 0.1, 0.1\n"
        "# mid-stream comment\n"
        "1.0, 1.0, 1.0\n"
    )
    sx, _sy, _sz, fixed = petgem.readSigmaCSV(str(csv))
    assert len(sx) == 2
    np.testing.assert_allclose(sx, [0.1, 1.0])
    assert fixed == []


def test_single_row(tmp_path):
    csv = tmp_path / "sigmas.csv"
    csv.write_text("0.5, 0.5, 0.5\n")
    sx, sy, sz, fixed = petgem.readSigmaCSV(str(csv))
    np.testing.assert_allclose(sx, [0.5])
    np.testing.assert_allclose(sy, [0.5])
    np.testing.assert_allclose(sz, [0.5])
    assert fixed == []


def test_optional_fixed_column(tmp_path):
    """4th column (`fixed`) marks per-material rows excluded from inversion.

    A non-zero entry flags the material as fixed; the returned list holds
    the 0-based row indices in sorted order.  Forward mode ignores the
    list but it must still parse without error.
    """
    csv = tmp_path / "sigmas.csv"
    csv.write_text(
        "# sigma_x, sigma_y, sigma_z, fixed\n"
        "1e-8, 1e-8, 1e-8, 1\n"     # air
        "3.3,  3.3,  3.3,  1\n"     # ocean
        "0.01, 0.01, 0.01, 0\n"     # sediments (invertable)
        "0.1,  0.1,  0.1,  0\n"     # basement (invertable)
    )
    sx, sy, sz, fixed = petgem.readSigmaCSV(str(csv))
    assert len(sx) == 4
    assert fixed == [0, 1]


def test_wrong_column_count_raises(tmp_path):
    csv = tmp_path / "sigmas.csv"
    csv.write_text("0.1, 0.2\n0.3, 0.4\n")
    with pytest.raises(ValueError, match="3 or 4 columns"):
        petgem.readSigmaCSV(str(csv))


def test_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        petgem.readSigmaCSV(str(tmp_path / "does_not_exist.csv"))


def test_returns_float64(tmp_path):
    csv = tmp_path / "sigmas.csv"
    csv.write_text("1, 2, 3\n")  # integers
    sx, sy, sz, _fixed = petgem.readSigmaCSV(str(csv))
    assert sx.dtype == np.float64
    assert sy.dtype == np.float64
    assert sz.dtype == np.float64
