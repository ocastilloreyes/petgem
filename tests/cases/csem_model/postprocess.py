#!/usr/bin/env python3
"""
csem_model validation script.

Compares the PETGEM Ex response against a precomputed ModEM reference and
emits a magnitude-vs-offset figure. Exits 0 on success, 1 if NRMSD exceeds
the configured tolerance.

This script is case-specific: csem_model ships a reference.h5; other cases
will have different validation strategies (no reference, H-component checks,
inverse recovery, etc.) and will provide their own postprocess.py.

Generic loading (bundle + responses) is delegated to petgem.readBundle /
petgem.readResponses; only the comparison + plot live here.

Usage:
    python3 tests/cases/csem_model/postprocess.py \\
        -responses_filename responses_p1.h5 \\
        [-case_dir tests/cases/csem_model]   \\  # default: directory of this script
        [-input_filename input.h5] \\
        [-reference_filename reference.h5] \\
        [-figure_filename figure.png] \\
        [-tolerance 0.03]
"""
import argparse
import os
import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import petgem


# Default case dir = the directory this script lives in.  The script ships
# inside its own case (tests/cases/csem_model/), so callers running it
# directly never need to repeat that path with -case_dir.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-case_dir",           default=SCRIPT_DIR,
                   help="Directory holding the bundle / responses / reference "
                        "files. Default: directory of this script.")
    p.add_argument("-input_filename",     default="input.h5")
    p.add_argument("-responses_filename", required=True)
    p.add_argument("-reference_filename", default="reference.h5")
    p.add_argument("-figure_filename",    default=None)
    p.add_argument("-tolerance",          type=float, default=0.03)
    return p.parse_args()


def main():
    args = parse_args()

    bundle_path    = os.path.join(args.case_dir, args.input_filename)
    responses_path = os.path.join(args.case_dir, args.responses_filename)
    reference_path = os.path.join(args.case_dir, args.reference_filename)

    bundle    = petgem.readBundle(bundle_path)
    responses = petgem.readResponses(responses_path)

    nord      = bundle['nord']
    x_coords  = bundle['receivers'][:, 0]
    Ex        = responses['Ex']

    with h5py.File(reference_path, 'r') as f:
        ref = (f['/reference_real'][()] + 1j * f['/reference_imag'][()]).ravel()

    mag_ref    = np.abs(ref)
    mag_petgem = np.abs(Ex)
    nrmsd  = np.sqrt(np.mean((mag_petgem - mag_ref) ** 2)) / (mag_ref.max() - mag_ref.min())
    rel_l2 = np.linalg.norm(mag_petgem - mag_ref) / np.linalg.norm(mag_ref)
    mape   = np.mean(np.abs((mag_petgem - mag_ref) / mag_ref)) * 100

    print(f"  NRMSD (magnitude) : {nrmsd:.6e}")
    print(f"  Relative L2 error : {rel_l2:.6e}")
    print(f"  MAPE (%)          : {mape:.6e}")

    fig_name = args.figure_filename or f"figure_p{nord}.png"
    fig_path = os.path.join(args.case_dir, fig_name)
    matplotlib.rcParams['font.family'] = 'Serif'
    plt.figure(figsize=(8, 4))
    plt.semilogy(x_coords, mag_ref,    label='Reference', linewidth=2)
    plt.semilogy(x_coords, mag_petgem, label='PETGEM',    linewidth=2,
                 marker='o', linestyle='', markersize=6)
    plt.xlabel('Offset (m)')
    plt.ylabel('|Ex| (V/m)')
    plt.title(f'csem_model - nord={nord}')
    plt.legend()
    plt.savefig(fig_path, dpi=300)

    if nrmsd < args.tolerance:
        print(f"\n  csem_model PASSED (NRMSD < {args.tolerance})")
        return 0
    print(f"\n  csem_model FAILED (NRMSD >= {args.tolerance})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
