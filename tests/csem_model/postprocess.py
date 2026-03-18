#!/usr/bin/env python3
"""
===============================================================================
PETGEM postprocessing 
===============================================================================

Postprocessing script to:

  1) Read reference data (ModEM)
  2) Read PETGEM responses (Ex field)
  3) Plot responses (semilogarithmic comparison)

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
python3 postprocess.py \
    -nord <order> \
    -case_dir <directory> \
    -receiver_filename <receivers_file> \
    -responses_filename <responses_file>
    

-------------------------------------------------------------------------------
REQUIRED ARGUMENTS
-------------------------------------------------------------------------------
-nord <int>
    Polynomial interpolation order (e.g., 1, 2, 3)

-case_dir <str>
    Directory containing the case input data

-receiver_filename <str>
    Receivers filename

-responses_filename <str>
    PETGEM responses filename

-------------------------------------------------------------------------------
OUTPUT FILES
-------------------------------------------------------------------------------
- Comparison figure saved as "figure_p{nord}.png"

-------------------------------------------------------------------------------
Author:
    Octavio Castillo-Reyes (UPC/BSC)
    octavio.castillo@upc.edu
    octavio.castillo@bsc.es

Latest update:
    March 04th, 2026
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import os
import sys
import petgem

# =============================================================================
# CONSTANTS
# =============================================================================
NUM_DIMENSIONS = 3

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():

    print("====================================================")
    print(" PETGEM POSTPROCESSING ")
    print("====================================================")

    # -------------------------------------------------------------------------
    # 1. Parse user arguments
    # -------------------------------------------------------------------------
    print("\nParsing input arguments")

    args = petgem.parsePostprocessingArgs()

    print(f"  Polynomial order (nord): {args.nord}")
    print(f"  Case directory         : {args.case_dir}")
    print(f"  PETGEM responses file  : {args.responses_filename}")
    print(f"  Receivers file         : {args.receiver_filename}")

    # -------------------------------------------------------------------------
    # 2. Load reference data
    # -------------------------------------------------------------------------
    print("\nLoading reference responses (ModEM code)")
    reference_filename = os.path.join(args.case_dir, f"reference.h5") 

    # Load reference data
    with h5py.File(reference_filename, "r") as f:
        ref_real = f["/reference_real"][()]
        ref_imag = f["/reference_imag"][()]
        reference = ref_real + 1j * ref_imag
        reference = reference.ravel()

    # -------------------------------------------------------------------------
    # 3. Load PETGEM responses
    # -------------------------------------------------------------------------
    print("\nLoading PETGEM responses")
    petgem_filename = os.path.join(args.case_dir, args.responses_filename) 
    Ex = petgem.readVectorH5(petgem_filename, 'Ex')

    # -------------------------------------------------------------------------
    # 4. Load receivers data
    # -------------------------------------------------------------------------
    print("\nLoading receivers data")
    receivers_filename = os.path.join(args.case_dir, args.receiver_filename) 
    receivers = petgem.readVectorH5(receivers_filename, 'receivers')
    
    # Reshape to a 2D array (num_receivers, xyz)
    receivers = np.array(receivers).reshape(-1, 3)
    
    # Cast: complex to real
    receivers = np.real(receivers)

    # Get only x-coordinates
    x_coordinates = receivers[:, 0]

    # ------------------------------------------------------------------------------
    # 5. Set environment for plots
    # ------------------------------------------------------------------------------
    size_marker = 8
    size_font = 11
    line_width = 2
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['font.size'] = size_font
    colors_fields = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628']

    # ------------------------------------------------------------------------------
    # 6. Plot electromagnetic responses
    # ------------------------------------------------------------------------------
    print("\nPlotting electromagnetic fields")
    plt.figure(figsize=(8, 4))
    plt.semilogy(x_coordinates,np.abs(reference),label=r'\texttt{Reference}',c=colors_fields[0],markersize=size_marker,markeredgecolor='k',linewidth=line_width)
    plt.semilogy(x_coordinates,np.abs(Ex),label=r'\texttt{PETGEM}',c=colors_fields[1],markersize=size_marker,markeredgecolor='k',linewidth=line_width, marker='o', linestyle='')
    plt.title('Ex')
    plt.ylabel(r'Amplitude (V/m)')
    plt.xlabel(r'Offset (m)')
    plt.legend()
    plt.savefig(os.path.join(args.case_dir, f"figure_p{args.nord}.png") , format='png', dpi=1200)

    # ------------------------------------------------------------------------------
    # 7. Compare electromagnetic fields
    # ------------------------------------------------------------------------------
    print("\nComputing errors:")
    mag_ref = np.abs(reference)
    mag_petgem = np.abs(Ex)

    # Normalized RMSD
    nrmsd = np.sqrt(np.mean((mag_petgem - mag_ref)**2)) / (mag_ref.max() - mag_ref.min())
    print("  NRMSD (magnitude):", nrmsd)

    # Relative L2 error
    rel_l2 = np.linalg.norm(mag_petgem - mag_ref) / np.linalg.norm(mag_ref)
    print("  Relative L2 error:", rel_l2)

    # Mean absolute percentage error
    mape = np.mean(np.abs((mag_petgem - mag_ref) / mag_ref)) * 100
    print("  MAPE (%):", mape)

    threshold = 0.03    # 3% of error

    if nrmsd < threshold:
        print("  CSEM test passed")
    else:
        print("  CSEM test failed")
        sys.exit(1)  # failure

    # -------------------------------------------------------------------------
    print("\n====================================================")
    print(" Preprocessing completed successfully")
    print("====================================================\n")
    
# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()















