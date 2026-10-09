#!/usr/bin/env python3
"""
mt1 - trapezoidal hill MT example validation script.

Compares the fm.mt apparent resistivities and phases (modes xy and yx) along
the 41-station survey line against two references at 2 Hz:

  * reference/emmi3d/       independent 3D solution (EMMI3D), one row per
                            frequency in each of Rxy, Ryx, Pxy, Pyx.dat
  * reference/petgem_2022.h5  responses published in Castillo-Reyes et al.
                            (2022), Section 3.1.1, p = 1, 2

Phases are compared in the first-quadrant convention of both references,
phi = mod(-phi_fm.mt, 180) (fm.mt uses exp(-iwt) with z up).

Prints, per number of skin depths, the median relative misfit in rho (%) and
the median absolute misfit in phase (deg), writes a profile figure, and exits 0
when every run with nskin >= -check_nskin is within -tolerance_rho and
-tolerance_phase of EMMI3D, 1 otherwise.

Usage:
    python3 examples/mt1/scripts/postprocess.py \\
        [-order 2] \\
        [-nskin 1 2 4 6 8 10] \\
        [-case_dir DIR] \\
        [-figure_filename outputs/figure_p<order>.png] \\
        [-check_nskin 4] [-tolerance_rho 2.0] [-tolerance_phase 1.0]
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
REF_FREQ = 2.0
MODES = ("xy", "yx")


def parseArgs():
    parser = argparse.ArgumentParser(description="Validate the trapezoidal hill MT example.")
    parser.add_argument("-order", type=int, default=2, help="Nedelec order of the runs. Default: 2")
    parser.add_argument("-nskin", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10],
                        help="Skin depths to read (missing runs are skipped). Default: 1 2 4 6 8 10")
    parser.add_argument("-case_dir", type=str, default=CASE_DIR, help="Example root directory")
    parser.add_argument("-figure_filename", type=str, default=None,
                        help="Figure path. Default: outputs/figure_p<order>.png")
    parser.add_argument("-check_nskin", type=int, default=4,
                        help="Smallest nskin included in the pass/fail check. Default: 4")
    parser.add_argument("-tolerance_rho", type=float, default=2.0,
                        help="Median relative misfit in rho (%%) allowed against EMMI3D. Default: 2.0")
    parser.add_argument("-tolerance_phase", type=float, default=1.0,
                        help="Median absolute misfit in phase (deg) allowed against EMMI3D. Default: 1.0")
    return parser.parse_args()


def readFmMt(path):
    """Return {('rho'|'phase', mode): (41,) array} from an fm.mt responses file."""
    out = {}
    with h5py.File(path, "r") as f:
        for mode in MODES:
            out[("rho", mode)] = np.asarray(f["apparent_resistivity"][mode])[:, 0]
            out[("phase", mode)] = np.mod(-np.asarray(f["phase"][mode])[:, 0], 180.0)
    return out


def readEmmi3d(dirpath):
    """Return x, z and {('rho'|'phase', mode): (41,) array} at REF_FREQ."""
    coords = np.loadtxt(os.path.join(dirpath, "Coordinates.dat"))
    files = {("rho", "xy"): "Rxy.dat", ("rho", "yx"): "Ryx.dat",
             ("phase", "xy"): "Pxy.dat", ("phase", "yx"): "Pyx.dat"}
    out = {}
    for key, name in files.items():
        rows = np.loadtxt(os.path.join(dirpath, name))
        hit = np.flatnonzero(np.isclose(rows[:, 0], REF_FREQ))
        if hit.size == 0:
            sys.exit(f"error: no {REF_FREQ} Hz row in {name}")
        out[key] = rows[hit[0], 1:]
    return coords[0], -coords[2], out


def readPublished(path, order, nskin):
    """Return {('rho'|'phase', mode): (41,) array} from petgem_2022.h5, or None."""
    groups = {"rho": "apparent_res", "phase": "phase"}
    out = {}
    with h5py.File(path, "r") as f:
        for q, g in groups.items():
            for mode in MODES:
                key = f"{g}_{mode}_p{order}/{nskin}_skin"
                if key not in f:
                    return None
                out[(q, mode)] = np.asarray(f[key]).ravel()
    return out


def misfits(resp, ref):
    """Median relative misfit in rho (%) and median absolute misfit in phase (deg), per mode."""
    m = {}
    for mode in MODES:
        m[("rho", mode)] = float(np.median(np.abs(resp[("rho", mode)] / ref[("rho", mode)] - 1.0)) * 100.0)
        m[("phase", mode)] = float(np.median(np.abs(resp[("phase", mode)] - ref[("phase", mode)])))
    return m


def main():
    args = parseArgs()
    outputs = os.path.join(args.case_dir, "outputs")
    x, z, emmi = readEmmi3d(os.path.join(args.case_dir, "reference", "emmi3d"))
    published_file = os.path.join(args.case_dir, "reference", "petgem_2022.h5")

    runs = {}
    for n in args.nskin:
        path = os.path.join(outputs, f"responses_p{args.order}_n{n}.h5")
        if os.path.exists(path):
            runs[n] = readFmMt(path)
    if not runs:
        sys.exit(f"error: no responses_p{args.order}_n*.h5 found in {outputs}")

    print(f"Trapezoidal hill, p = {args.order}, {REF_FREQ} Hz: median misfit "
          f"(rho in %, phase in deg)")
    print(f"{'nskin':>5}  {'vs':<12} {'rho_xy':>7} {'rho_yx':>7} {'phi_xy':>7} {'phi_yx':>7}")
    passed = True
    for n, resp in runs.items():
        rows = [("EMMI3D", emmi)]
        published = readPublished(published_file, args.order, n)
        if published is not None:
            rows.append(("PETGEM 2022", published))
        for label, ref in rows:
            m = misfits(resp, ref)
            print(f"{n:>5}  {label:<12} {m[('rho', 'xy')]:7.2f} {m[('rho', 'yx')]:7.2f} "
                  f"{m[('phase', 'xy')]:7.2f} {m[('phase', 'yx')]:7.2f}")
            if label == "EMMI3D" and n >= args.check_nskin:
                ok = all(m[("rho", mode)] <= args.tolerance_rho and m[("phase", mode)] <= args.tolerance_phase
                         for mode in MODES)
                passed = passed and ok

    # Profiles along the survey line
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [(("rho", "xy"), r"$\rho_{xy}$ ($\Omega\cdot$m)"), (("rho", "yx"), r"$\rho_{yx}$ ($\Omega\cdot$m)"),
              (("phase", "xy"), r"$\phi_{xy}$ (deg)"), (("phase", "yx"), r"$\phi_{yx}$ (deg)")]
    for ax, (key, label) in zip(axes.ravel(), panels):
        ax.plot(x, emmi[key], "k-", lw=2, label="EMMI3D")
        for n, resp in runs.items():
            ax.plot(x, resp[key], "o-", ms=3, lw=1, label=f"fm.mt, {n} skin")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("x (m)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Trapezoidal hill, {REF_FREQ} Hz, p = {args.order}")
    fig.tight_layout()
    figure = args.figure_filename or os.path.join(outputs, f"figure_p{args.order}.png")
    fig.savefig(figure, dpi=150)
    print(f"\nFigure: {figure}")

    checked = [n for n in runs if n >= args.check_nskin]
    if not checked:
        print(f"No run with nskin >= {args.check_nskin}: nothing to check.")
        return 0
    print(f"Check (nskin >= {args.check_nskin}, rho <= {args.tolerance_rho} %, "
          f"phase <= {args.tolerance_phase} deg vs EMMI3D): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
