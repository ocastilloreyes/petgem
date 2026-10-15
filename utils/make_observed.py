#!/usr/bin/env python3
"""
PETGEM synthetic-observations builder - general CLI utility.

Assembles a set of monochromatic fm.csem forward responses into a single
multi-frequency dataset and adds Gaussian measurement noise, producing the
observed-data HDF5 that the inverse kernel (im.csem) consumes.

This tool is problem-independent: the model, mesh, survey and frequency list
are supplied on the command line, so it can build the observed data for any
CSEM inversion example. See examples/im_csem_block for a complete benchmark.

COMPONENTS
    -components selects the data written (default Ex): any of Ex Ey Ez Hx Hy
    Hz, and the ratios Zxx Zxy Zyx Zyy = Ei / Hj of the same responses. The
    noise is drawn component by component in the given order.

NOISE MODEL
    For every datum, independently on the real and imaginary parts,

        noise ~ N(0, error_level * |d_true|)

    This matches how im.csem weights the misfit
    (W = 1 / (|d_obs| * error_level)), so the RMS of the true model against
    the noisy data is ~1.0 - i.e. "fit to the noise level".

REPRODUCIBILITY
    The RNG seed is fixed via -seed and recorded in the output file as the
    /@noise_seed attribute, so the dataset is reproducible bit-for-bit
    (numpy PCG64 is stable across platforms).

Usage:
    python3 utils/make_observed.py \\
        -case_dir examples/im_csem_block \\
        -pattern  "outputs/responses_fm_f{freq}_p2.h5" \\
        -freqs    1,10,50,100,300,800,1500 \\
        -seed     20260720 \\
        -error_level 0.01 \\
        [-components Ex,Zxy] \\
        -output_filename outputs/observed.h5
"""
import argparse
import os
import sys

import h5py
import numpy as np

import functions as petgem  # the copy next to this script


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-case_dir", required=True,
                        help="Base directory; -pattern and -output_filename are relative to it.")
    parser.add_argument("-pattern", default="outputs/responses_fm_f{freq}_p2.h5",
                        help="fm.csem responses filename pattern ('{freq}' placeholder).")
    parser.add_argument("-freqs", required=True,
                        help="Comma-separated frequency list, e.g. 1,10,50,100,300,800,1500.")
    parser.add_argument("-seed", type=int, required=True,
                        help="RNG seed for the noise draw.")
    parser.add_argument("-error_level", type=float, default=0.01,
                        help="Relative noise level (0.01 = 1%%). Default 0.01.")
    parser.add_argument("-components", default="Ex",
                        help="Comma-separated data components, from "
                             f"{', '.join(petgem.DATA_COMPONENTS)}. Default Ex.")
    parser.add_argument("-output_filename", default="outputs/observed.h5",
                        help="Output HDF5 (relative to -case_dir).")
    args = parser.parse_args()

    freqs = [float(value) for value in args.freqs.split(",")]
    lower = {name.lower(): name for name in petgem.DATA_COMPONENTS}
    components = []
    for value in args.components.split(","):
        name = lower.get(value.strip().lower())
        if name is None:
            sys.exit(f"ERROR: unknown component {value!r} (one of {', '.join(petgem.DATA_COMPONENTS)})")
        if name in components:
            sys.exit(f"ERROR: component {name} given twice")
        components.append(name)

    clean_rows, nrec = {name: [] for name in components}, set()
    for fr in freqs:
        fn = os.path.join(args.case_dir, args.pattern.format(freq=_fmt(fr)))
        if not os.path.isfile(fn):
            sys.exit(f"ERROR: missing {fn}. Run the fm.csem forward stage first.")
        r = petgem.readResponses(fn, source=1)
        fields = {name: np.asarray(r[name], dtype=np.complex128).reshape(-1)
                  for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        got = float(r["source"]["frequency"])
        # Guard against stacking a file generated for the wrong frequency:
        # the row order here IS the frequency axis, so a silent mismatch would
        # misassign every datum.
        if not np.isclose(got, fr, rtol=1e-9):
            sys.exit(f"ERROR: {fn} holds frequency {got} Hz, expected {fr} Hz.")
        for name in components:
            if name.startswith("Z"):
                # Zij = Ei / Hj
                values = fields["E" + name[1]] / fields["H" + name[2]]
            else:
                values = fields[name]
            clean_rows[name].append(values)
        nrec.add(fields["Ex"].shape[0])

    if len(nrec) != 1:
        sys.exit(f"ERROR: receiver count differs across frequencies: {nrec}")
    clean = {name: np.vstack(rows) for name, rows in clean_rows.items()}   # [N_freq, N_recv]

    # --- noise: sigma = error_level * |d_true|, drawn independently for Re/Im
    rng = np.random.default_rng(args.seed)
    noisy = {}
    for name in components:
        sd = args.error_level * np.abs(clean[name])
        noisy[name] = (clean[name].real + rng.normal(0.0, sd)
                       + 1j * (clean[name].imag + rng.normal(0.0, sd)))

    # RMS that im.csem would report for the TRUE model against this dataset.
    misfit = sum(np.sum(np.abs((noisy[name] - clean[name])
                               / (np.abs(noisy[name]) * args.error_level)) ** 2)
                 for name in components)
    rms_true = float(np.sqrt(misfit / sum(noisy[name].size * 2 for name in components)))

    out = os.path.join(args.case_dir, args.output_filename)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as f:
        for name in components:
            f.create_dataset(name, data=noisy[name])
        f.create_dataset("frequencies", data=np.array(freqs, dtype=float))
        f.attrs["num_frequencies"] = len(freqs)
        f.attrs["num_receivers"] = nrec.pop()
        f.attrs["components"] = ",".join(components)
        f.attrs["error_level"] = args.error_level
        f.attrs["noise_seed"] = args.seed
        f.attrs["noise_model"] = ("gaussian, sigma = error_level*|d_true|, "
                                  "independent on Re and Im")
        f.attrs["rms_true_model"] = rms_true
        f.attrs["source_files"] = ";".join(
            args.pattern.format(freq=_fmt(fr)) for fr in freqs)

    print(f"  frequencies      : {freqs}")
    print(f"  seed             : {args.seed}")
    print(f"  error level      : {args.error_level}")
    for name in components:
        resid = np.abs(noisy[name] - clean[name])
        rel = resid / np.where(np.abs(clean[name]) > 0, np.abs(clean[name]), 1)
        print(f"  {name:<3} shape        : {clean[name].shape}  (N_freq, N_recv)")
        print(f"  |noise|/|{name}| mean: {rel.mean():.5f}  (expect ~{args.error_level*1.25:.5f})")
    print(f"  RMS(true model)  : {rms_true:.4f}  (must be ~1.0)")
    print(f"  wrote            : {out}")


def _fmt(fr):
    """Render a frequency as it appears in filenames: 10.0 -> '10'."""
    return f"{fr:g}"


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as err:
        sys.exit(f"ERROR: {err}")
