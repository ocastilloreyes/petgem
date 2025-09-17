#!/usr/bin/env python3
"""
This script compares reference electromagnetic field data 
with PETGEM simulation results for a given basis order. 
It performs the following tasks:

1. Reads reference field data (real and imaginary parts) 
   from an HDF5 file.
2. Reads PETGEM-computed Ex field data from an HDF5 file.
3. Imports receiver coordinates from an HDF5 file.
4. Produces a semilogarithmic comparison plot of the 
   reference vs PETGEM fields along receiver positions.
5. Saves the resulting plot as a PNG file.

Inputs (command line arguments):
    1. basis_order : int
        Basis order of the PETGEM simulation.
    2. test_dir : str
        Path to the directory containing the HDF5 files.

Outputs:
    - Comparison figure saved as "figure_p{basis_order}.png"
      in the given test directory.

Author: Octavio Castillo-Reyes (UPC/BSC) (octavio.castillo@upc.edu; octavio.castillo@bsc.es)
Latest update: September 10th, 2025
"""

import numpy as np
from pathlib import Path
import petsc4py
import h5py
import matplotlib
import matplotlib.pyplot as plt
import sys

petsc4py.init(sys.argv)
from petsc4py import PETSc


def read_h5_vector(filename, dataset_name):
    """Read a PETSc Vec from an HDF5 file."""
    tmp = PETSc.Vec().create(comm=PETSc.COMM_SELF)
    tmp.setName(dataset_name)
    viewer = PETSc.Viewer().createHDF5(str(filename), mode='r', comm=PETSc.COMM_SELF)
    tmp.load(viewer)
    viewer.destroy()
    vector = tmp.getArray()
    return vector


# Main function
if __name__ == "__main__":
    # Read input parameters
    nord = sys.argv[1]
    
    # Setup file paths using pathlib
    reference_filename = f"tests/csem_model/reference.h5"
    petgem_filename = f"tests/csem_model/responses_p{nord}_src1.h5"
    receivers_filename = f"tests/csem_model/receivers.h5"
    figure_out_filename = f"tests/csem_model/figure_p{nord}.png"

    # Load reference data
    with h5py.File(reference_filename, "r") as f:
        ref_real = f["/reference_real"][()]
        ref_imag = f["/reference_imag"][()]
        reference = ref_real + 1j * ref_imag
        reference = reference.ravel()

    # Load PETGEM output
    Ex = read_h5_vector(petgem_filename, 'Ex')

    # ------------------------------------------------------------------------------
    # IMPORT RECEIVERS
    # ------------------------------------------------------------------------------
    receivers = read_h5_vector(receivers_filename, 'receivers')
    receivers = np.array(receivers).reshape(-1, 3)
    receivers = np.real(receivers)
    x_coordinates = receivers[:, 0]

    # ------------------------------------------------------------------------------
    # SET ENVIRONMENT FOR PLOTS
    # ------------------------------------------------------------------------------
    size_marker = 8
    size_font = 11
    line_width = 2
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'Serif'
    matplotlib.rcParams['font.size'] = size_font
    colors_fields = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628']

    # ------------------------------------------------------------------------------
    # PLOT FIELDS
    # ------------------------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.semilogy(x_coordinates,np.abs(reference),label=r'\texttt{Reference}',c=colors_fields[0],markersize=size_marker,markeredgecolor='k',linewidth=line_width)
    plt.semilogy(x_coordinates,np.abs(Ex),label=r'\texttt{PETGEM}',c=colors_fields[1],markersize=size_marker,markeredgecolor='k',linewidth=line_width)
    plt.title('Ex')
    plt.ylabel(r'Amplitude (V/m)')
    plt.xlabel(r'Offset (m)')
    plt.legend()

    # -----------------------------
    # Save figure
    # -----------------------------
    plt.savefig(figure_out_filename, format='png', dpi=1200)

    # ------------------------------------------------------------------------------
    # Compare electromagnetic fields
    # # ------------------------------------------------------------------------------    
    mag_ref = np.abs(reference)
    mag_petgem = np.abs(Ex)

    # Normalized RMSD
    nrmsd = np.sqrt(np.mean((mag_petgem - mag_ref)**2)) / (mag_ref.max() - mag_ref.min())
    print("NRMSD (magnitude):", nrmsd)

    # Relative L2 error
    rel_l2 = np.linalg.norm(mag_petgem - mag_ref) / np.linalg.norm(mag_ref)
    print("Relative L2 error:", rel_l2)

    # Mean absolute percentage error
    mape = np.mean(np.abs((mag_petgem - mag_ref) / mag_ref)) * 100
    print("MAPE (%):", mape)

    threshold = 0.03    # 3% of error

    if nrmsd < threshold:
        print("CSEM test passed")
        sys.exit(0)  # success
    else:
        print("CSEM test failed")
        sys.exit(1)  # failure