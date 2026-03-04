#!/usr/bin/env python3
"""
===============================================================================
PETGEM Input Generator
===============================================================================

Preprocessing script to generate:

  1) PETSc DM object (mesh + resistivity model)
  2) PETGEM parameter file
  3) Receiver file in PETGEM-compatible format

This script prepares all required input data for PETGEM electromagnetic
simulations.

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
python3 preprocess.py \
    -nord <order> \
    -case_dir <directory> \
    -mesh_filename <mesh_file> \
    -source_filename <sources_file> \
    -receiver_filename <receivers_file> \
    [optional arguments]

-------------------------------------------------------------------------------
REQUIRED ARGUMENTS
-------------------------------------------------------------------------------
-nord <int>
    Polynomial interpolation order (e.g., 1, 2, 3)

-case_dir <str>
    Directory containing the case input data

-mesh_filename <str>
    Mesh filename (Gmsh format supported by meshio)

-source_filename <str>
    Sources filename (currently not processed here, but required by PETGEM)

-receiver_filename <str>
    Receivers filename

-------------------------------------------------------------------------------
OPTIONAL ARGUMENTS
-------------------------------------------------------------------------------
-resistivity_view <vtkfile>
    Export resistivity distribution (e.g., vtk:model.vtu)

-sigma_file <csvfile>
    CSV file containing sigma_x, sigma_y, sigma_z per material

-------------------------------------------------------------------------------
OUTPUT FILES
-------------------------------------------------------------------------------
- model_p<nord>.h5
- params_nord<nord>.txt
- receivers.h5

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
import os
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
    print(" PETGEM INPUT PREPROCESSING ")
    print("====================================================")

    # -------------------------------------------------------------------------
    # 1. Parse user arguments
    # -------------------------------------------------------------------------
    print("\nParsing input arguments")

    args = petgem.parsePreprocessingArgs()

    input_mesh_filename = os.path.join(args.case_dir, args.mesh_filename)
    input_sources_filename = os.path.join(args.case_dir, args.source_filename)
    input_receivers_filename = os.path.join(args.case_dir, args.receiver_filename)

    output_mesh_filename = os.path.join(args.case_dir, f"model_p{args.nord}.h5")
    output_receivers_filename = os.path.join(args.case_dir, "receivers.h5")
    output_petgem_filename = f"responses_p{args.nord}"

    print(f"  Polynomial order (nord): {args.nord}")
    print(f"  Case directory         : {args.case_dir}")
    print(f"  Mesh file              : {input_mesh_filename}")
    print(f"  Receivers file         : {input_receivers_filename}")

    # -------------------------------------------------------------------------
    # 2. Define resistivity model
    # -------------------------------------------------------------------------
    print("\nDefining resistivity model")

    sigma_x = np.array([0.1, 1.0], dtype=float)
    sigma_y = np.array([0.1, 1.0], dtype=float)
    sigma_z = np.array([0.1, 1.0], dtype=float)

    print(f"  Number of materials     : {len(sigma_x)}")

    # -------------------------------------------------------------------------
    # 3. Import mesh
    # -------------------------------------------------------------------------
    print("\nReading mesh")

    mesh = petgem.meshio.read(input_mesh_filename)

    num_material_blocks = len(mesh.cells)
    
    # Count total number of tetrahedral elements
    num_cells = 0
    for block in mesh.cells:
        num_cells += block.data.shape[0]

    coords = mesh.points
    num_coords = coords.shape[0]

    print(f"  Total elements          : {num_cells}")
    print(f"  Total vertices          : {num_coords}")

    # Use last cell block (tetrahedral elements)
    cells = mesh.cells[-1].data

    # -------------------------------------------------------------------------
    # 4. Create PETSc DM object
    # -------------------------------------------------------------------------
    print("\n Creating PETSc DM (DMPlex) object")

    plex = petgem.createDM(NUM_DIMENSIONS, cells, coords)

    print("  DM object successfully created")

    # -------------------------------------------------------------------------
    # 5. Assign resistivity per cell
    # -------------------------------------------------------------------------
    print("\nAssigning resistivity per cell")

    resistivity = np.zeros((num_cells, NUM_DIMENSIONS), dtype=float)

    elemsS = np.copy(mesh.cell_data_dict["gmsh:physical"]["tetra"])
    elemsS -= 1  # Convert from 1-based to 0-based indexing

    for i in range(num_cells):
        resistivity[i, 0] = sigma_x[elemsS[i]]
        resistivity[i, 1] = sigma_y[elemsS[i]]
        resistivity[i, 2] = sigma_z[elemsS[i]]

    print("  Resistivity assignment completed")

    # -------------------------------------------------------------------------
    # 6. Store DM object
    # -------------------------------------------------------------------------
    print("\nWriting PETGEM model to HDF5")

    petgem.writeDM(plex, resistivity, output_mesh_filename)

    print(f"  Output mesh file: {output_mesh_filename}")

    # -------------------------------------------------------------------------
    # 7. Generate PETGEM parameter file
    # -------------------------------------------------------------------------
    print("\nGenerating PETGEM parameter file")

    petgem.writeParamsFile(args.nord, args.case_dir, output_petgem_filename)

    # -------------------------------------------------------------------------
    # 8. Store receivers
    # -------------------------------------------------------------------------
    print("\nWriting receivers file")

    # NOTE:
    # PETSc in PETGEM is configured with complex scalars.
    # Receivers are stored as complex to avoid type mismatch issues.
    petgem.writeReceivers(input_receivers_filename, output_receivers_filename)

    print(f"  Output receivers file: {output_receivers_filename}")

    # -------------------------------------------------------------------------
    print("\n====================================================")
    print(" Preprocessing completed successfully")
    print("====================================================\n")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()