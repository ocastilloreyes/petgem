"""Unit tests for petgem.readSourcesText (forward-source text parser).

It is the single point that interprets the forward source file format
(frequency line + per-source 7-field rows), so it warrants direct coverage.
"""
import numpy as np
import pytest

from petgem import readSourcesText


def test_single_source(tmp_path):
    src = tmp_path / "sources.txt"
    src.write_text(
        "# frequency (Hz)\n"
        "1.5\n"
        "# x y z current length dip azimuth\n"
        "100.0 200.0 -50.0 1.0 100.0 0.0 90.0\n"
    )
    freq, arr = readSourcesText(str(src))
    assert freq == pytest.approx(1.5)
    assert arr.shape == (1, 7)
    np.testing.assert_allclose(arr[0], [100.0, 200.0, -50.0, 1.0, 100.0, 0.0, 90.0])


def test_multiple_sources_and_blank_lines(tmp_path):
    src = tmp_path / "sources.txt"
    src.write_text(
        "0.25\n"
        "\n"
        "1 2 3 4 5 6 7\n"
        "\n"
        "8 9 10 11 12 13 14\n"
        "# trailing comment\n"
    )
    freq, arr = readSourcesText(str(src))
    assert freq == pytest.approx(0.25)
    assert arr.shape == (2, 7)
    np.testing.assert_allclose(arr[0], [1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_allclose(arr[1], [8, 9, 10, 11, 12, 13, 14])


def test_wrong_field_count_raises(tmp_path):
    src = tmp_path / "sources.txt"
    src.write_text("1.0\n1 2 3 4 5 6\n")  # 6 fields, expected 7
    with pytest.raises(ValueError, match="expected 7 fields"):
        readSourcesText(str(src))


def test_missing_frequency_raises(tmp_path):
    src = tmp_path / "sources.txt"
    src.write_text("# only comments here\n")
    with pytest.raises(ValueError, match="missing frequency"):
        readSourcesText(str(src))


def test_missing_sources_raises(tmp_path):
    src = tmp_path / "sources.txt"
    src.write_text("1.0\n")  # frequency but no source rows
    with pytest.raises(ValueError, match="missing frequency or source rows"):
        readSourcesText(str(src))
