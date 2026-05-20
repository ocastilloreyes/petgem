#!/usr/bin/env python3
"""
Convert invEx.dat (plain text) to HDF5 format for the PETGEM inverse kernel.

Reads the MATLAB-style invEx.dat file and writes an HDF5 file with:
    /Ex             [numFreqs x numReceivers] complex128
    /frequencies    [numFreqs] float64

h5py stores complex128 as HDF5 compound type {r: float64, i: float64},
which the C loader reads directly into PetscScalar buffers.

Frequencies are read from the inversion sources.txt file (8-field format:
    freq  x  y  z  current  length  dip  azimuth

invEx.dat format (one row per frequency):
    freq_index  Re(Ex_1) Im(Ex_1)  Re(Ex_2) Im(Ex_2)  ...  Re(Ex_N) Im(Ex_N)

Usage:
    python convert_invex_to_hdf5.py \\
        --input petgem_inv_matlab/petgem_inv1/Test_cases/fastm1/Measuredata/invEx.dat \\
        --sources tests/inverse/sources.txt \\
        --output tests/inverse/observed_data.h5
"""

import argparse

import h5py
import numpy as np


def parse_inversion_sources(filename):
    """Read frequencies from inversion sources file (8-field format)."""
    frequencies = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) >= 8:
                frequencies.append(float(fields[0]))
    return np.array(frequencies)


def main():
    parser = argparse.ArgumentParser(
        description="Convert invEx.dat to HDF5 for PETGEM inverse kernel")
    parser.add_argument("--input", required=True,
                        help="Path to invEx.dat")
    parser.add_argument("--sources", required=True,
                        help="Path to inversion sources.txt (8-field format)")
    parser.add_argument("--output", default="tests/inverse/observed_data.h5",
                        help="Output HDF5 file path")
    args = parser.parse_args()

    # Read frequencies from sources file
    frequencies = parse_inversion_sources(args.sources)
    num_freqs = len(frequencies)
    print(f"Read {num_freqs} frequencies from {args.sources}: {frequencies}")

    # Read invEx.dat
    data = np.loadtxt(args.input)
    print(f"Read {args.input}: shape = {data.shape}")

    if data.shape[0] != num_freqs:
        raise ValueError(
            f"Row count {data.shape[0]} != number of frequencies {num_freqs}")

    # Column 0 is the frequency index label — skip it
    # Remaining columns are Re/Im pairs
    values = data[:, 1:]
    num_receivers = values.shape[1] // 2

    if values.shape[1] != 2 * num_receivers:
        raise ValueError(
            f"Data columns {values.shape[1]} is not even "
            f"(expected 2 * numReceivers)")

    # Build complex array [numFreqs x numReceivers]
    ex = values[:, 0::2] + 1j * values[:, 1::2]

    print(f"Frequencies:  {num_freqs} ({frequencies})")
    print(f"Receivers:    {num_receivers}")
    print(f"Data shape:   {ex.shape}")

    # Write HDF5 — h5py stores complex128 as compound type {r, i}
    with h5py.File(args.output, "w") as f:
        f.create_dataset("Ex", data=ex)
        f.create_dataset("frequencies", data=frequencies)
        f.attrs["num_frequencies"] = num_freqs
        f.attrs["num_receivers"] = num_receivers

    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
