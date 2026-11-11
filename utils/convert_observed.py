#!/usr/bin/env python3
"""
PETGEM observed-data converter - text to HDF5.

Writes the observed-data HDF5 taken by preprocess.py -observed_filename from
text files, one per component, with one row per frequency:

    freq  Re(d_1) Im(d_1)  Re(d_2) Im(d_2)  ...  Re(d_N) Im(d_N)

freq is in Hz; receivers follow the order of the receiver list. Lines starting
with '#' are comments. NaN marks a missing datum.

Usage:
    python3 utils/convert_observed.py \\
        -input ex:observed_ex.dat zxy:observed_zxy.dat \\
        -output_filename observed.h5 \\
        [-error_level 0.05]
"""
import argparse
import os
import sys

import h5py
import numpy as np

import functions as petgem  # the copy next to this script


def readObservedText(path):
    """Return (frequencies, values [N_freq, N_recv] complex128) of one text file."""
    table = np.loadtxt(path, comments="#", ndmin=2)
    columns = table.shape[1] - 1
    if columns < 2 or columns % 2 != 0:
        raise ValueError(f"{path}: expected freq and Re/Im pairs, got {table.shape[1]} columns")
    return table[:, 0], table[:, 1::2] + 1j * table[:, 2::2]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-input", nargs="+", required=True,
                        help="Text files, each 'component:file', component one of "
                             f"{', '.join(petgem.DATA_COMPONENTS)}.")
    parser.add_argument("-output_filename", required=True, help="Output HDF5.")
    parser.add_argument("-error_level", type=float, default=None,
                        help="Relative noise level stored as the error_level attribute. "
                             "Omit to store none.")
    args = parser.parse_args()

    frequencies, components, sources = None, {}, []
    for spec in args.input:
        name, path = petgem.parseObservedSpec(spec)
        if name is None:
            raise ValueError(f"{spec}: expected component:file")
        if name in components:
            raise ValueError(f"Component {name} given twice")
        freqs, values = readObservedText(path)
        if frequencies is None:
            frequencies = freqs
        elif freqs.shape != frequencies.shape or not np.allclose(freqs, frequencies, rtol=1e-6, atol=0.0):
            raise ValueError(f"{path}: frequencies differ from {sources[0]}")
        components[name] = values
        sources.append(path)

    receivers = {values.shape[1] for values in components.values()}
    if len(receivers) != 1:
        raise ValueError(f"Receiver count differs across components: {sorted(receivers)}")

    os.makedirs(os.path.dirname(args.output_filename) or ".", exist_ok=True)
    with h5py.File(args.output_filename, "w") as f:
        for name, values in components.items():
            f.create_dataset(name, data=values)
        f.create_dataset("frequencies", data=frequencies)
        f.attrs["num_frequencies"] = frequencies.size
        f.attrs["num_receivers"] = receivers.pop()
        f.attrs["components"] = ",".join(components)
        f.attrs["source_files"] = ";".join(os.path.basename(path) for path in sources)
        if args.error_level is not None:
            f.attrs["error_level"] = args.error_level

    for name, values in components.items():
        print(f"  {name:<3} shape : {values.shape}  (N_freq, N_recv)")
    print(f"  wrote     : {args.output_filename}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as err:
        sys.exit(f"ERROR: {err}")
