"""Observed-data input of preprocess.py and the text converter."""
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

UTILS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils")
sys.path.insert(0, UTILS)
import functions as petgem  # noqa: E402

FREQS = np.array([1.0, 10.0])
EX = np.array([[1 + 2j, 3 + 4j, np.nan + 1j * np.nan], [5 + 6j, 7 + 8j, 9 + 1j]])


def write_h5(path, **datasets):
    with h5py.File(path, "w") as f:
        for name, values in datasets.items():
            f.create_dataset(name, data=values)
    return str(path)


def write_text(path, freqs, values):
    rows = np.empty((values.shape[0], 1 + 2 * values.shape[1]))
    rows[:, 0], rows[:, 1::2], rows[:, 2::2] = freqs, values.real, values.imag
    np.savetxt(path, rows, fmt="%.17e", header="freq Re Im ...")
    return str(path)


def convert(*args):
    return subprocess.run([sys.executable, os.path.join(UTILS, "convert_observed.py"), *args],
                          capture_output=True, text=True, check=False)


def test_reads_components_frequencies_and_error_level(tmp_path):
    path = write_h5(tmp_path / "obs.h5", Ex=EX, Zxy=2 * EX, frequencies=FREQS)
    with h5py.File(path, "a") as f:
        f.attrs["error_level"] = 0.05
    data, freqs, level = petgem.readObservedData(path)
    assert set(data) == {"Ex", "Zxy"}
    assert np.array_equal(data["Ex"], EX, equal_nan=True)
    assert np.array_equal(freqs, FREQS)
    assert level == 0.05
    assert set(petgem.readObservedData(path, "Zxy")[0]) == {"Zxy"}


def test_rejects_text_file(tmp_path):
    path = write_text(tmp_path / "obs.dat", FREQS, EX)
    with pytest.raises(ValueError, match="not an HDF5 file"):
        petgem.readObservedData(path)


def test_rejects_real_dataset(tmp_path):
    path = write_h5(tmp_path / "obs.h5", Ex=EX.real, frequencies=FREQS)
    with pytest.raises(ValueError, match="must be complex"):
        petgem.readObservedData(path)


def test_rejects_missing_frequencies(tmp_path):
    path = write_h5(tmp_path / "obs.h5", Ex=EX)
    with pytest.raises(ValueError, match="no /frequencies"):
        petgem.readObservedData(path)


def test_rejects_frequencies_of_wrong_length(tmp_path):
    path = write_h5(tmp_path / "obs.h5", Ex=EX, frequencies=np.array([1.0]))
    with pytest.raises(ValueError, match="does not match"):
        petgem.readObservedData(path)


def test_frequencies_match_sources_within_tolerance():
    petgem.checkObservedFrequencies("obs.h5", FREQS, FREQS * (1 + 1e-8))


def test_rejects_reordered_frequencies():
    with pytest.raises(ValueError, match="row 0 holds 10.0 Hz"):
        petgem.checkObservedFrequencies("obs.h5", FREQS[::-1], FREQS)


def test_rejects_frequency_count_mismatch():
    with pytest.raises(ValueError, match="2 frequencies, 3 source rows"):
        petgem.checkObservedFrequencies("obs.h5", FREQS, np.array([1.0, 10.0, 100.0]))


def test_converter_round_trip(tmp_path):
    ex = write_text(tmp_path / "ex.dat", FREQS, EX)
    zxy = write_text(tmp_path / "zxy.dat", FREQS, -EX)
    out = str(tmp_path / "obs.h5")
    run = convert("-input", f"ex:{ex}", f"zxy:{zxy}", "-output_filename", out, "-error_level", "0.05")
    assert run.returncode == 0, run.stderr
    data, freqs, level = petgem.readObservedData(out)
    assert np.array_equal(data["Ex"], EX, equal_nan=True)
    assert np.array_equal(data["Zxy"], -EX, equal_nan=True)
    assert np.array_equal(freqs, FREQS)
    assert level == 0.05


@pytest.mark.parametrize("prefix, other_freqs, message", [
    ("", FREQS, "expected component:file"),
    ("ex:", FREQS * 2, "frequencies differ"),
])
def test_converter_rejects_bad_input(tmp_path, prefix, other_freqs, message):
    first = write_text(tmp_path / "ex.dat", FREQS, EX)
    second = write_text(tmp_path / "zxy.dat", other_freqs, EX)
    specs = [f"{prefix}{first}"] + ([f"zxy:{second}"] if prefix else [])
    run = convert("-input", *specs, "-output_filename", str(tmp_path / "obs.h5"))
    assert run.returncode != 0
    assert message in run.stderr
