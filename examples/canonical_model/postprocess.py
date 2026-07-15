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
        [-responses_filename responses_p<order>.h5] \\  # default from bundle order
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


# Default case dir = the directory this script lives in, so callers running
# it directly never need to repeat that path with -case_dir.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_NAME = os.path.basename(SCRIPT_DIR)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-case_dir",           default=SCRIPT_DIR,
                   help="Directory holding the bundle / responses / reference "
                        "files. Default: directory of this script.")
    p.add_argument("-input_filename",     default="input.h5")
    p.add_argument("-responses_filename", default=None,
                   help="Responses HDF5. Default: responses_p<order>.h5, "
                        "the name the preprocess/kernel produce for the "
                        "bundle's order.")
    p.add_argument("-reference_filename", default="reference.h5")
    p.add_argument("-figure_filename",    default=None)
    p.add_argument("-tolerance",          type=float, default=0.03)
    return p.parse_args()


def main():
    args = parse_args()

    bundle_path    = os.path.join(args.case_dir, args.input_filename)
    reference_path = os.path.join(args.case_dir, args.reference_filename)

    bundle = petgem.readBundle(bundle_path)
    order      = bundle['order']
    x_coords  = bundle['receivers'][:, 0]

    # Responses default to the fixed responses_p<order>.h5 convention, so a
    # bare run needs no -responses_filename; the order comes from the bundle.
    responses_name = args.responses_filename or f"responses_p{order}.h5"
    responses_path = os.path.join(args.case_dir, responses_name)
    responses = petgem.readResponses(responses_path)
    Ex        = responses['Ex']

    ref = np.array(petgem.readVectorH5(reference_path, 'reference')).ravel()

    # Shared metric helper - same NRMSD / rel-L2 / MAPE every case reports.
    metrics = petgem.compareMagnitude(Ex, ref)
    nrmsd = metrics['nrmsd']

    print(f"  NRMSD (magnitude) : {nrmsd:.6e}")
    print(f"  Relative L2 error : {metrics['rel_l2']:.6e}")
    print(f"  MAPE (%)          : {metrics['mape']:.6e}")

    fig_name = args.figure_filename or f"figure_p{order}.png"
    fig_path = os.path.join(args.case_dir, fig_name)
    matplotlib.rcParams['font.family'] = 'Serif'
    plt.figure(figsize=(8, 4))
    plt.semilogy(x_coords, np.abs(ref), label='Reference', linewidth=2)
    plt.semilogy(x_coords, np.abs(Ex),  label='PETGEM',    linewidth=2,
                 marker='o', linestyle='', markersize=6)
    plt.xlabel('Offset (m)')
    plt.ylabel('|Ex| (V/m)')
    plt.title(f'{CASE_NAME} - order={order}')
    plt.legend()
    plt.savefig(fig_path, dpi=300)

    if nrmsd < args.tolerance:
        print(f"\n  {CASE_NAME} PASSED (NRMSD < {args.tolerance})")
        return 0
    print(f"\n  {CASE_NAME} FAILED (NRMSD >= {args.tolerance})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
