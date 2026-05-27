"""Unit tests for petgem.readSigmaTable (per-material conductivity table).

The canonical format is whitespace-delimited ``sigmas.txt``; commas are also
accepted so legacy comma-separated tables still parse.
"""
import numpy as np
import pytest

import petgem


def test_basic_three_column(tmp_path):
    f = tmp_path / "sigmas.txt"
    f.write_text(
        "# sigma_x sigma_y sigma_z\n"
        "0.1 0.2 0.3\n"
        "1.0 1.5 2.0\n"
    )
    sx, sy, sz, fixed = petgem.readSigmaTable(str(f))
    np.testing.assert_allclose(sx, [0.1, 1.0])
    np.testing.assert_allclose(sy, [0.2, 1.5])
    np.testing.assert_allclose(sz, [0.3, 2.0])
    assert fixed == []


def test_blank_lines_and_comments_ignored(tmp_path):
    f = tmp_path / "sigmas.txt"
    f.write_text(
        "# leading comment\n"
        "\n"
        "0.1 0.1 0.1\n"
        "1.0 1.0 1.0   # inline comment\n"
    )
    sx, _sy, _sz, fixed = petgem.readSigmaTable(str(f))
    assert len(sx) == 2
    np.testing.assert_allclose(sx, [0.1, 1.0])
    assert fixed == []


def test_optional_fixed_column(tmp_path):
    """4th column (`fixed`) marks per-material rows excluded from inversion."""
    f = tmp_path / "sigmas.txt"
    f.write_text(
        "# sigma_x sigma_y sigma_z fixed\n"
        "1e-8 1e-8 1e-8 1\n"     # air
        "3.3  3.3  3.3  1\n"     # ocean
        "0.01 0.01 0.01 0\n"     # sediments (invertable)
        "0.1  0.1  0.1  0\n"     # basement (invertable)
    )
    sx, sy, sz, fixed = petgem.readSigmaTable(str(f))
    assert len(sx) == 4
    assert fixed == [0, 1]


def test_wrong_column_count_raises(tmp_path):
    f = tmp_path / "sigmas.txt"
    f.write_text("0.1 0.2\n0.3 0.4\n")
    with pytest.raises(ValueError, match="3 or 4 columns"):
        petgem.readSigmaTable(str(f))


def test_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        petgem.readSigmaTable(str(tmp_path / "does_not_exist.txt"))


def test_returns_float64(tmp_path):
    f = tmp_path / "sigmas.txt"
    f.write_text("1 2 3\n")  # integers
    sx, sy, sz, _fixed = petgem.readSigmaTable(str(f))
    assert sx.dtype == np.float64
