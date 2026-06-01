#!/usr/bin/env python3
"""
Generate observed data (with noise) for the PETGEM inverse kernel (im.csem).

Reads fm.csem forward solver HDF5 output files, adds amplitude-relative
Gaussian noise, and saves the result as an HDF5 file readable by
loadObservedData() in src/inversion.c.

Output HDF5 structure:
    /Ex             [numFreqs x numReceivers] complex128
    /frequencies    [numFreqs] float64
    Attributes: error_level, num_frequencies, num_receivers, noise_seed

h5py stores complex128 as HDF5 compound type {r: float64, i: float64},
which the C loader reads directly into PetscScalar buffers.

Usage:
    python generate_observed_data.py \\
        --input responses_f0.25.h5 responses_f1.0.h5 ... \\
        --error-level 0.01 \\
        --output observed_data.h5

    Or with a glob pattern:
    python generate_observed_data.py \\
        --input-pattern "output/responses_*.h5" \\
        --error-level 0.01 \\
        --output observed_data.h5

Notes:
    - Each input file is a fm.csem unified HDF5 output. The script pulls
      the Ex dataset from --ex-dataset (default ``sources/src1/fields/Ex``)
      and the source frequency from a robust fallback list (see
      resolve_frequency).
    - Files are sorted by frequency in ascending order.
    - Noise model: std_i = |Ex_i| * error_level  (amplitude-relative,
      matches petgem_inv_new/add_noise_ari.m).

Author: Octavio Castillo Reyes (UPC/BSC)
"""

import argparse
import glob
import sys

import h5py
import numpy as np


def read_petsc_complex_vec(dataset):
    """Read a PETSc complex Vec stored in HDF5.

    PETSc stores complex Vecs as compound type {r, i} or as a flat
    array of doubles with alternating real/imag values.  h5py may
    return a structured array or a plain float array.
    """
    raw = dataset[:]

    # Case 1: structured/compound type with named fields
    if raw.dtype.names is not None:
        for re_name, im_name in [("r", "i"), ("re", "im"),
                                  ("real", "imag")]:
            if re_name in raw.dtype.names and im_name in raw.dtype.names:
                return raw[re_name] + 1j * raw[im_name]
        names = raw.dtype.names
        return raw[names[0]] + 1j * raw[names[1]]

    # Case 2: already complex
    if np.issubdtype(raw.dtype, np.complexfloating):
        return raw

    # Case 3: flat real array with interleaved re/im
    if raw.ndim == 1 and raw.size % 2 == 0:
        return raw[0::2] + 1j * raw[1::2]

    # Case 4: shape (N, 2)
    if raw.ndim == 2 and raw.shape[1] == 2:
        return raw[:, 0] + 1j * raw[:, 1]

    raise ValueError(f"Cannot interpret dataset as complex: "
                     f"shape={raw.shape}, dtype={raw.dtype}")


def resolve_frequency(f, fpath):
    """Read the source frequency from an fm.csem output file, format-robust.

    Tries, in order:
      1. new unified-file root attribute  frequency  (single-file fm.csem
         postprocessing.c writes this once at the root)
      2. new per-source attribute  /sources/src1@frequency  (the same value
         is mirrored on the source group)
      3. legacy/root attribute  Source_frequency   (older fm.csem / MATLAB)
      4. legacy per-file group   /source@frequency  (one-file-per-source layout)
      5. a /frequencies dataset (first entry) as a last resort
    """
    if "frequency" in f.attrs:
        return float(f.attrs["frequency"])
    if "/sources/src1" in f and "frequency" in f["/sources/src1"].attrs:
        return float(f["/sources/src1"].attrs["frequency"])
    if "Source_frequency" in f.attrs:
        return float(f.attrs["Source_frequency"])
    if "source" in f and "frequency" in f["source"].attrs:
        return float(f["source"].attrs["frequency"])
    if "frequencies" in f:
        return float(np.asarray(f["frequencies"]).ravel()[0])
    raise ValueError(
        f"Cannot find a source frequency in {fpath}. Looked for root attrs "
        f"'frequency' / 'Source_frequency', group attrs "
        f"'/sources/src1@frequency' / '/source@frequency', and dataset "
        f"'/frequencies'. Available attrs: {list(f.attrs.keys())}, "
        f"datasets: {list(f.keys())}")


def resolve_ex_dataset(f, requested, fpath):
    """Return the Ex dataset, trying the requested name then known layouts.

    Current fm.csem stores receiver Ex under /sources/src{k}/fields/Ex in a
    single unified file. Older fm.csem output used /fields/Ex (one file per
    source). h5py resolves path-like names directly.
    """
    candidates = (
        requested,
        "sources/src1/fields/Ex",
        "fields/Ex",
        "Ex",
        f"fields/{requested}",
    )
    for name in candidates:
        if name and name in f:
            return f[name]
    raise ValueError(
        f"Dataset '{requested}' (and fallbacks {candidates[1:]}) not found "
        f"in {fpath}. Available top-level keys: {list(f.keys())}")


def add_noise(data, error_level, rng):
    """Add amplitude-relative Gaussian noise to complex data.

    Matches petgem_inv_new/add_noise_ari.m:
        std_i = |data_i| * error_level
        noisy = data + std_i * (randn + 1j*randn)
    """
    amplitude = np.abs(data)
    std = amplitude * error_level
    noise_re = rng.normal(0.0, 1.0, data.shape)
    noise_im = rng.normal(0.0, 1.0, data.shape)
    return data + std * (noise_re + 1j * noise_im)


def main():
    parser = argparse.ArgumentParser(
        description="Generate observed data HDF5 for PETGEM inverse kernel")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", nargs="+",
                       help="List of fm.csem HDF5 output files "
                            "(one per frequency)")
    group.add_argument("--input-pattern",
                       help="Glob pattern for fm.csem HDF5 output files")
    parser.add_argument("--error-level", type=float, default=0.01,
                        help="Relative noise level (default: 0.01 = 1%%)")
    parser.add_argument("--output", default="observed_data.h5",
                        help="Output HDF5 file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-noise", action="store_true",
                        help="Skip noise addition (use exact forward data)")
    parser.add_argument("--ex-dataset", default="sources/src1/fields/Ex",
                        help="HDF5 path to the Ex dataset in each input file "
                             "(default: sources/src1/fields/Ex)")
    args = parser.parse_args()

    # Resolve input files
    if args.input_pattern:
        input_files = sorted(glob.glob(args.input_pattern))
    else:
        input_files = args.input

    if not input_files:
        print("Error: no input files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(input_files)} input file(s)")

    # Read frequencies and Ex data from each file
    freq_data = []
    for fpath in input_files:
        print(f"  Reading: {fpath}")
        with h5py.File(fpath, "r") as f:
            freq = resolve_frequency(f, fpath)
            ex = read_petsc_complex_vec(resolve_ex_dataset(f, args.ex_dataset, fpath))
            freq_data.append((freq, ex))

    # Sort by frequency
    freq_data.sort(key=lambda x: x[0])

    frequencies = np.array([fd[0] for fd in freq_data])
    num_freqs = len(frequencies)
    num_receivers = len(freq_data[0][1])

    for i, (freq, ex) in enumerate(freq_data):
        if len(ex) != num_receivers:
            raise ValueError(
                f"Inconsistent receiver count: file for freq={freq} has "
                f"{len(ex)} receivers, expected {num_receivers}")

    # Build complex data matrix [numFreqs x numReceivers]
    ex_matrix = np.zeros((num_freqs, num_receivers), dtype=np.complex128)
    for i, (freq, ex) in enumerate(freq_data):
        ex_matrix[i, :] = ex

    # Add noise
    if args.no_noise:
        ex_noisy = ex_matrix
        print(f"\nNo noise added (exact forward data)")
    else:
        rng = np.random.default_rng(args.seed)
        ex_noisy = add_noise(ex_matrix, args.error_level, rng)
        print(f"\nNoise added: error_level={args.error_level}, seed={args.seed}")

    # Save to HDF5 - complex128 stored as compound type {r, i}
    print(f"Writing: {args.output}")
    print(f"  Frequencies:  {num_freqs} ({frequencies})")
    print(f"  Receivers:    {num_receivers}")
    print(f"  Data shape:   {ex_noisy.shape}")

    with h5py.File(args.output, "w") as f:
        f.create_dataset("Ex", data=ex_noisy)
        f.create_dataset("frequencies", data=frequencies)
        f.attrs["error_level"] = args.error_level
        f.attrs["num_frequencies"] = num_freqs
        f.attrs["num_receivers"] = num_receivers
        f.attrs["noise_seed"] = args.seed
        f.attrs["no_noise"] = args.no_noise

    print("Done.")


if __name__ == "__main__":
    main()
