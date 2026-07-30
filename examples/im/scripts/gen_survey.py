#!/usr/bin/env python3
"""Generate the survey definition for the im CSEM inversion benchmark.

Writes, into ../survey/, the transmitter, receiver, frequency and
conductivity files that drive the forward and inverse runs. All values follow
the benchmark specification (see ../README.md):

    frequencies  1, 10, 50, 100, 300, 800, 1500 Hz
    source       x-directed electric dipole at (0, -4000, 0) m, unit I and L
    receivers    441 (21x21) points, x,y in [-200, 200] m, z = 0
    background   100 ohm-m  (0.01 S/m)
    anomaly       10 ohm-m  (0.1  S/m)
    air                     (1e-8 S/m)

Run from anywhere:  python3 scripts/gen_survey.py

Outputs (../survey/):
    receivers.txt      441 receiver locations
    sources_im.txt     7-frequency transmitter table -> im.csem
    sources_f<F>.txt   one single-frequency transmitter -> fm.csem (monochromatic)
    frequencies.txt    the frequency list
    sigmas_true.txt    conductivities for mesh_true.msh (AIR, BG, INVERT, ANOMALY)
    sigmas_im.txt     starting model for mesh.msh (AIR, BG, INVERT), with
                       the 4th column flagging fixed (1) vs invertable (0) cells
"""
import os

import numpy as np

FREQS = [1, 10, 50, 100, 300, 800, 1500]              # Hz
SRC = (0.0, -4000.0, 0.0)                             # x, y, z  (z negative down)
CURRENT, LENGTH, DIP, AZIMUTH = 1.0, 1.0, 0.0, 0.0   # az=dip=0 -> x-directed
RECV_HALF, RECV_N, RECV_Z = 200.0, 21, 0.0           # 21x21 = 441 on [-200,200], z=0
SIGMA_AIR, SIGMA_BG, SIGMA_ANOM = 1e-8, 0.01, 0.1

SURVEY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "survey")


def main():
    os.makedirs(SURVEY, exist_ok=True)

    # ---- receivers: 21 x 21 grid, z = 0 (air/earth interface) --------------
    ax = np.linspace(-RECV_HALF, RECV_HALF, RECV_N)
    pts = [(x, y, RECV_Z) for x in ax for y in ax]
    assert len(pts) == 441, len(pts)
    with open(_p("receivers.txt"), "w") as f:
        f.write("# 441 (21x21) receivers, x,y in [-200,200] m, 20 m spacing, "
                "z = 0.\n# z is negative down: z=0 is the air/earth interface.\n"
                "# x y z\n")
        for x, y, z in pts:
            f.write(f"{x:12.4f} {y:12.4f} {z:12.4f}\n")

    # ---- frequencies -------------------------------------------------------
    with open(_p("frequencies.txt"), "w") as f:
        f.write("\n".join(str(fr) for fr in FREQS) + "\n")

    # ---- transmitters ------------------------------------------------------
    hdr = ("# x-directed electric dipole at (0, -4000, 0) m, unit current and "
           "length,\n# dip = azimuth = 0.\n# freq  x  y  z  current  length  "
           "dip  azimuth\n")
    with open(_p("sources_im.txt"), "w") as f:                 # multi-freq -> im.csem
        f.write(hdr)
        for fr in FREQS:
            f.write(_srcrow(fr))
    for fr in FREQS:                                           # one file -> each fm.csem run
        with open(_p(f"sources_f{fr}.txt"), "w") as f:
            f.write(hdr)
            f.write(_srcrow(fr))

    # ---- conductivity tables (row index = gmsh physical tag - 1) -----------
    # True model, on mesh_true.msh -> tags AIR(1) BG(2) INVERT(3) ANOMALY(4).
    with open(_p("sigmas_true.txt"), "w") as f:
        f.write("# TRUE model conductivities for mesh_true.msh --> fm.csem.\n"
                "# Row = 0-based material id (gmsh physical tag - 1).\n"
                "# sigma_x sigma_y sigma_z\n")
        for s, c in [(SIGMA_AIR,  "AIR      1e8 ohm-m"),
                     (SIGMA_BG,   "BG       100 ohm-m"),
                     (SIGMA_BG,   "INVERT   100 ohm-m (background here)"),
                     (SIGMA_ANOM, "ANOMALY   10 ohm-m  <- target")]:
            f.write(f"{s:<8g} {s:<8g} {s:<8g}   # {c}\n")

    # Inversion starting model, on mesh.msh -> tags AIR(1) BG(2) INVERT(3).
    # Uniform 100 ohm-m halfspace; only the INVERT region is free.
    with open(_p("sigmas_im.txt"), "w") as f:
        f.write("# INVERSION starting model for mesh.msh --> im.csem.\n"
                "# Uniform 100 ohm-m halfspace (no anomaly - recovering it is\n"
                "# the objective). Row = 0-based material id (tag - 1).\n"
                "# 4th column: 1 = held fixed, 0 = invertable.\n"
                "# sigma_x sigma_y sigma_z fixed\n")
        for s, fx, c in [(SIGMA_AIR, 1, "AIR     (fixed)"),
                         (SIGMA_BG,  1, "BG      (fixed background)"),
                         (SIGMA_BG,  0, "INVERT  (invertable region)")]:
            f.write(f"{s:<8g} {s:<8g} {s:<8g} {fx}   # {c}\n")

    print(f"wrote survey files to {os.path.normpath(SURVEY)}:")
    print("  receivers.txt, sources_im.txt, sources_f*.txt, frequencies.txt,")
    print("  sigmas_true.txt, sigmas_im.txt")


def _p(name):
    return os.path.join(SURVEY, name)


def _srcrow(fr):
    return (f"{fr:<6g} {SRC[0]:8.1f} {SRC[1]:10.1f} {SRC[2]:6.1f} "
            f"{CURRENT:6.1f} {LENGTH:6.1f} {DIP:5.1f} {AZIMUTH:5.1f}\n")


if __name__ == "__main__":
    main()
