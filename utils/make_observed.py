#!/usr/bin/env python3
"""
PETGEM synthetic-observations builder - general CLI utility.

Assembles a set of monochromatic fm.csem forward responses into a single
multi-frequency dataset and adds Gaussian measurement noise, producing the
observed-data HDF5 that the inverse kernel (im.csem) consumes.

This tool is problem-independent: the model, mesh, survey and frequency list
are supplied on the command line, so it can build the observed data for any
CSEM inversion example. See examples/im_model for a complete benchmark.

NOISE MODEL
    For every datum, independently on the real and imaginary parts,

        noise ~ N(0, error_level * |Ex_true|)

    This matches how im.csem weights the misfit
    (W = 1 / (|d_obs| * error_level)), so the RMS of the true model against
    the noisy data is ~1.0 - i.e. "fit to the noise level".

REPRODUCIBILITY
    The RNG seed is fixed via -seed and recorded in the output file as the
    /@noise_seed attribute, so the dataset is reproducible bit-for-bit
    (numpy PCG64 is stable across platforms).

Usage:
    python3 utils/make_observed.py \\
        -case_dir examples/im_model \\
        -pattern  "outputs/responses_fm_f{freq}_p2.h5" \\
        -freqs    1,10,50,100,300,800,1500 \\
        -seed     20260720 \\
        -error_level 0.01 \\
        -out      reference/observed.h5
"""
import argparse
import os
import sys

import h5py
import numpy as np

try:
    import petgem
except ModuleNotFoundError:
    # Allow running directly from a fresh clone, without `pip install -e .`:
    # expose the in-tree `utils` package under its distribution name `petgem`.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import utils as petgem  # noqa: E402
    sys.modules["petgem"] = petgem


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-case_dir", required=True,
                    help="Base directory; -pattern and -out are relative to it.")
    ap.add_argument("-pattern", default="outputs/responses_fm_f{freq}_p2.h5",
                    help="fm.csem responses filename pattern ('{freq}' placeholder).")
    ap.add_argument("-freqs", required=True,
                    help="Comma-separated frequency list, e.g. 1,10,50,100,300,800,1500.")
    ap.add_argument("-seed", type=int, required=True,
                    help="RNG seed for the noise draw.")
    ap.add_argument("-error_level", type=float, default=0.01,
                    help="Relative noise level (0.01 = 1%%). Default 0.01.")
    ap.add_argument("-out", default="outputs/observed.h5",
                    help="Output HDF5 (relative to -case_dir).")
    args = ap.parse_args()

    freqs = [float(x) for x in args.freqs.split(",")]

    clean_rows, nrec = [], set()
    for fr in freqs:
        fn = os.path.join(args.case_dir, args.pattern.format(freq=_fmt(fr)))
        if not os.path.isfile(fn):
            sys.exit(f"ERROR: missing {fn}. Run the fm.csem forward stage first.")
        r = petgem.readResponses(fn, source=1)
        ex = np.asarray(r["Ex"], dtype=np.complex128).reshape(-1)
        got = float(r["source"]["frequency"])
        # Guard against stacking a file generated for the wrong frequency:
        # the row order here IS the frequency axis, so a silent mismatch would
        # misassign every datum.
        if not np.isclose(got, fr, rtol=1e-9):
            sys.exit(f"ERROR: {fn} holds frequency {got} Hz, expected {fr} Hz.")
        clean_rows.append(ex)
        nrec.add(ex.shape[0])

    if len(nrec) != 1:
        sys.exit(f"ERROR: receiver count differs across frequencies: {nrec}")
    clean = np.vstack(clean_rows)                     # [N_freq, N_recv]

    # --- noise: sigma = error_level * |d_true|, drawn independently for Re/Im
    rng = np.random.default_rng(args.seed)
    sd = args.error_level * np.abs(clean)
    noisy = (clean.real + rng.normal(0.0, sd)
             + 1j * (clean.imag + rng.normal(0.0, sd)))

    # RMS that im.csem would report for the TRUE model against this dataset.
    w = 1.0 / (np.abs(noisy) * args.error_level)
    rms_true = float(np.sqrt(np.sum(np.abs(w * (noisy - clean)) ** 2)
                             / (noisy.size * 2)))

    out = os.path.join(args.case_dir, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("Ex", data=noisy)
        f.create_dataset("frequencies", data=np.array(freqs, dtype=float))
        f.attrs["num_frequencies"] = len(freqs)
        f.attrs["num_receivers"] = clean.shape[1]
        f.attrs["error_level"] = args.error_level
        f.attrs["noise_seed"] = args.seed
        f.attrs["noise_model"] = ("gaussian, sigma = error_level*|Ex_true|, "
                                  "independent on Re and Im")
        f.attrs["rms_true_model"] = rms_true
        f.attrs["source_files"] = ";".join(
            args.pattern.format(freq=_fmt(fr)) for fr in freqs)

    resid = np.abs(noisy - clean)
    rel = resid / np.where(np.abs(clean) > 0, np.abs(clean), 1)
    print(f"  frequencies      : {freqs}")
    print(f"  Ex shape         : {clean.shape}  (N_freq, N_recv)")
    print(f"  seed             : {args.seed}")
    print(f"  error level      : {args.error_level}")
    print(f"  |noise|/|Ex| mean: {rel.mean():.5f}  (expect ~{args.error_level*1.25:.5f})")
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
