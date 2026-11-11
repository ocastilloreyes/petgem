#!/usr/bin/env python3
"""
fm_tdcsem_halfspace - time-domain CSEM half-space example validation script.

Compares the fm.tdcsem inline field Ex at the two surface receivers (offsets
1000 and 2000 m) with the analytical impulse response of Cai et al. (2017),
Eq. (A.8), for t > 0:

    Ex(t) = (I ds / 8) (mu^3 sigma / (pi^3 t^5))^(1/2) exp(-mu sigma rho^2 / (4 t))

The comparison runs from the peak of the analytical response,
t_peak = mu sigma rho^2 / 10, to -t_max.

Prints, per receiver and decade, the median relative misfit (%), writes a
figure, and exits 0 when the median relative misfit of every receiver over
[t_peak, -t_max] is within -tolerance, 1 otherwise.

Usage:
    python3 examples/fm_tdcsem_halfspace/scripts/postprocess.py \\
        [-order 2] [-case_dir DIR] \\
        [-figure_filename outputs/figure_p<order>.png] \\
        [-t_max 0.3] [-tolerance 3.0]
"""
import argparse
import os
import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

CASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MU = 4.0e-7 * np.pi


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the time-domain CSEM half-space example.")
    parser.add_argument("-order", type=int, default=2, help="Nedelec order of the run. Default: 2")
    parser.add_argument("-case_dir", type=str, default=CASE_DIR, help="Example root directory")
    parser.add_argument("-figure_filename", type=str, default=None,
                        help="Figure path. Default: outputs/figure_p<order>.png")
    parser.add_argument("-t_max", type=float, default=0.3,
                        help="Last time (s) included in the pass/fail check. Default: 0.3")
    parser.add_argument("-tolerance", type=float, default=3.0,
                        help="Median relative misfit (%%) allowed per receiver. Default: 3.0")
    return parser.parse_args()


def read_fm_tdcsem(path):
    """Return times (N_t,) and Ex (N_recv, N_t) from an fm.tdcsem responses file."""
    with h5py.File(path, "r") as f:
        t = np.asarray(f["times"])
        ex = np.asarray(f["sources/src1/fields/Ex"])
    return (t[:, 0] if t.ndim == 2 else t), (ex[..., 0] if ex.ndim == 3 else ex)


def read_table(path):
    """Return the rows of a whitespace text table, '#' comments skipped."""
    return np.atleast_2d(np.loadtxt(path, comments="#"))


def analytic_ex(t, rho, sigma, moment):
    """Eq. (A.8) of Cai et al. (2017) for t > 0."""
    return moment / 8.0 * np.sqrt(MU ** 3 * sigma / (np.pi ** 3 * t ** 5)) * np.exp(-MU * sigma * rho ** 2 / (4.0 * t))


def main():
    args = parse_args()
    outputs = os.path.join(args.case_dir, "outputs")
    path = os.path.join(outputs, f"responses_p{args.order}.h5")
    if not os.path.exists(path):
        sys.exit(f"error: {path} not found")

    survey = os.path.join(args.case_dir, "survey")
    sigma = read_table(os.path.join(survey, "sigmas.txt"))[0, 0]
    src = read_table(os.path.join(survey, "sources.txt"))[0]
    receivers = read_table(os.path.join(survey, "receivers.txt"))
    moment = src[3] * src[4]
    offsets = np.hypot(receivers[:, 0] - src[0], receivers[:, 1] - src[1])

    t, ex = read_fm_tdcsem(path)
    t, ex = t[1:], ex[:, 1:]

    print(f"Half-space, p = {args.order}, {1.0 / sigma:g} ohm-m: median relative misfit of Ex (%)")
    print(f"{'offset':>7}  {'t_peak (s)':>10}  " + "  ".join(f"{f'1e{d}':>7}" for d in range(-5, 1)))
    passed = True
    fig, ax = plt.subplots(figsize=(7, 5))
    for j, rho in enumerate(offsets):
        ref = analytic_ex(t, rho, sigma, moment)
        t_peak = MU * sigma * rho ** 2 / 10.0
        after = t >= t_peak
        misfit = np.full(t.shape, np.nan)
        misfit[after] = np.abs(ex[j, after] / ref[after] - 1.0) * 100.0
        cells = []
        for d in range(-5, 1):
            sel = after & (t >= 10.0 ** d) & (t < 10.0 ** (d + 1))
            cells.append(f"{np.median(misfit[sel]):7.2f}" if sel.any() else f"{'-':>7}")
        print(f"{rho:7.0f}  {t_peak:10.2e}  " + "  ".join(cells))
        sel = after & (t <= args.t_max)
        passed = passed and bool(np.median(misfit[sel]) <= args.tolerance)

        ax.loglog(t, np.abs(ex[j]), lw=1.5, label=f"fm.tdcsem, offset {rho:.0f} m")
        ax.loglog(t, ref, "k--", lw=1)
    ax.loglog([], [], "k--", lw=1, label="Analytical (Eq. A.8)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|Ex| (V/m)")
    ax.set_ylim(1.0e-13, 10.0 * np.abs(ex).max())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(f"Half-space, inline Ex at z = 0, p = {args.order}")
    figure = args.figure_filename or os.path.join(outputs, f"figure_p{args.order}.png")
    fig.tight_layout()
    fig.savefig(figure, dpi=150)
    print(f"Figure: {figure}")

    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
