"""Unit tests for petgem.readSourcesText (unified transmitter parser).

It is the single point that interprets the transmitter file format for both
forward and inverse modeling. The canonical layout is 8 fields per row
(freq x y z current length dip azimuth); the legacy forward layout (a lone
frequency line followed by 7-field rows) is accepted and converted to the
same (N, 8) form.
"""
import numpy as np
import pytest

from petgem import readSourcesText


def test_eight_field_single_row(tmp_path):
    src = tmp_path / "sources.txt"
    src.write_text(
        "# freq x y z current length dip azimuth\n"
        "1.5 100.0 200.0 -50.0 1.0 100.0 0.0 90.0\n"
    )
    arr = readSourcesText(str(src))
    assert arr.shape == (1, 8)
    np.testing.assert_allclose(arr[0], [1.5, 100.0, 200.0, -50.0, 1.0, 100.0, 0.0, 90.0])


def test_eight_field_multi_frequency(tmp_path):
    src = tmp_path / "sources.txt"
    src.write_text(
        "50   0 0 0 1 1 0 0\n"
        "\n"
        "300  0 0 0 1 1 0 0\n"
        "# trailing comment\n"
        "800  0 0 0 1 1 0 0\n"
    )
    arr = readSourcesText(str(src))
    assert arr.shape == (3, 8)
    np.testing.assert_allclose(arr[:, 0], [50, 300, 800])


def test_legacy_forward_format(tmp_path):
    """Lone frequency line + 7-field rows -> (N, 8) with freq prepended."""
    src = tmp_path / "sources.txt"
    src.write_text(
        "# frequency (Hz)\n"
        "0.25\n"
        "\n"
        "1 2 3 4 5 6 7\n"
        "8 9 10 11 12 13 14\n"
    )
    arr = readSourcesText(str(src))
    assert arr.shape == (2, 8)
    np.testing.assert_allclose(arr[0], [0.25, 1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_allclose(arr[1], [0.25, 8, 9, 10, 11, 12, 13, 14])


def test_legacy_wrong_field_count_raises(tmp_path):
    src = tmp_path / "sources.txt"
    src.write_text("1.0\n1 2 3 4 5 6\n")  # 6 fields after freq, expected 7
    with pytest.raises(ValueError, match="7 fields"):
        readSourcesText(str(src))


def test_empty_raises(tmp_path):
    src = tmp_path / "sources.txt"
    src.write_text("# only comments here\n")
    with pytest.raises(ValueError, match="no source rows"):
        readSourcesText(str(src))


def test_unrecognized_width_raises(tmp_path):
    src = tmp_path / "sources.txt"
    src.write_text("1 2 3 4 5\n")  # 5 fields: neither 8 nor legacy
    with pytest.raises(ValueError, match="unrecognized source format"):
        readSourcesText(str(src))
