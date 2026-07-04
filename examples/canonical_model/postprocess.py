#!/usr/bin/env python3
"""
canonical_model - CSEM example validation script.

Compares the PETGEM Ex response for this canonical marine CSEM model
(see mesh.geo / README.md) against a precomputed reference solution and
emits a magnitude-vs-offset figure. Prints the
NRMSD, relative L2 and MAPE error metrics, and exits 0 when the magnitude
NRMSD is below the tolerance (default 0.03), 1 otherwise.

The reference (reference.h5) is stored in the native PETSc complex Vec
layout - a single '/reference' dataset with real/imag columns - and is
read with petgem.readVectorH5, the same helper used for the input bundle
and the responses, so it comes back already combined into a complex array.

Generic loading (bundle + responses + reference vector) is delegated to
petgem.readBundle / petgem.readResponses / petgem.readVectorH5; only the
comparison and plot live here.

Usage:
    python3 examples/canonical_model/postprocess.py \\
        -responses_filename responses_p1.h5 \\
        [-case_dir DIR]    \\  # default: directory of this script
        [-input_filename input.h5] \\
        [-reference_filename reference.h5] \\
        [-figure_filename figure.png] \\
        [-tolerance 0.03]
"""
import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import petgem


# Default case dir = the directory this script lives in.  The script ships
# inside its own case (wham/), so callers running it directly never need
# to repeat that path with -case_dir.
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

    order      = bundle['order']
    x_coords  = bundle['receivers'][:, 0]
    Ex        = responses['Ex']

    ref = np.array(petgem.readVectorH5(reference_path, 'reference')).ravel()

    mag_ref    = np.abs(ref)
    mag_petgem = np.abs(Ex)
    nrmsd  = np.sqrt(np.mean((mag_petgem - mag_ref) ** 2)) / (mag_ref.max() - mag_ref.min())
    rel_l2 = np.linalg.norm(mag_petgem - mag_ref) / np.linalg.norm(mag_ref)
    mape   = np.mean(np.abs((mag_petgem - mag_ref) / mag_ref)) * 100

    print(f"  NRMSD (magnitude) : {nrmsd:.6e}")
    print(f"  Relative L2 error : {rel_l2:.6e}")
    print(f"  MAPE (%)          : {mape:.6e}")

    fig_name = args.figure_filename or f"figure_p{order}.png"
    fig_path = os.path.join(args.case_dir, fig_name)
    matplotlib.rcParams['font.family'] = 'Serif'
    plt.figure(figsize=(8, 4))
    plt.semilogy(x_coords, mag_ref,    label='Reference', linewidth=2)
    plt.semilogy(x_coords, mag_petgem, label='PETGEM',    linewidth=2,
                 marker='o', linestyle='', markersize=6)
    plt.xlabel('Offset (m)')
    plt.ylabel('|Ex| (V/m)')
    plt.title(f'wham - order={order}')
    plt.legend()
    plt.savefig(fig_path, dpi=300)

    if nrmsd < args.tolerance:
        print(f"\n  wham PASSED (NRMSD < {args.tolerance})")
        return 0
    print(f"\n  wham FAILED (NRMSD >= {args.tolerance})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
