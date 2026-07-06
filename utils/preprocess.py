#!/usr/bin/env python3
"""
PETGEM preprocessing - centralized CLI entry point.

Reads CLI arguments, loads the per-material conductivity table from
`-sigma_file <txt>`, and calls petgem.runPreprocessing to produce the
unified PETGEM input HDF5 plus the matching params file.

Mode-specific arguments (validated by runPreprocessing):
    -mode forward    requires -source_filename
                     forbids  -inv_source_filename, -observed_filename
    -mode inverse    requires -inv_source_filename, -observed_filename
                     forbids  -source_filename
                     (the inverse kernel reads multi-freq sources from
                      /sources/* in the bundle)
                     -observed_filename accepts either an HDF5 (.h5) or a raw
                     MATLAB-style invEx.dat text file (parsed inline, so no
                     separate convert-to-HDF5 step is needed); -error_level
                     optionally records the noise level.

Shared arguments (both modes):
    -order, -case_dir, -mesh_filename, -receiver_filename, -sigma_file
    -input_filename, -params_filename, -output_vtk        (all optional with defaults)

Usage examples:
    # Forward modeling
    python3 utils/preprocess.py \\
        -mode forward \\
        -order 1 \\
        -case_dir examples/<case> \\
        -mesh_filename mesh_p1.msh \\
        -receiver_filename receivers.txt \\
        -source_filename sources.txt \\
        -sigma_file sigmas.txt \\
        [-output_vtk model.vtu]

    # Inverse modeling
    python3 utils/preprocess.py \\
        -mode inverse \\
        -order 1 \\
        -case_dir examples/<case> \\
        -mesh_filename mesh_p1.msh \\
        -receiver_filename receivers.txt \\
        -inv_source_filename sources.txt \\
        -observed_filename observed_data.h5 \\
        -sigma_file sigmas.txt \\
        [-output_vtk model.vtu]
"""
import os
import sys

try:
    import petgem
except ModuleNotFoundError:
    # Allow running directly from a fresh clone, without `pip install -e .`:
    # expose the in-tree `utils` package under its distribution name
    # `petgem` (same shim as tests/conftest.py).
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import utils as petgem
    sys.modules["petgem"] = petgem


def main():
    args = petgem.parsePreprocessingArgs()
    sigma_x, sigma_y, sigma_z, fixed_materials = petgem.readSigmaTable(
        os.path.join(args.case_dir, args.sigma_file)
    )
    petgem.runPreprocessing(
        mode=args.mode,
        order=args.order,
        case_dir=args.case_dir,
        mesh_filename=args.mesh_filename,
        receiver_filename=args.receiver_filename,
        source_filename=args.source_filename,
        sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z,
        fixed_materials=fixed_materials,
        input_filename=args.input_filename,
        params_filename=args.params_filename,
        inv_source_filename=args.inv_source_filename,
        observed_filename=args.observed_filename,
        error_level=args.error_level,
        output_vtk=args.output_vtk,
        dm_view=args.dm_view is not None,
    )


if __name__ == "__main__":
    main()
