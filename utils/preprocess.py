#!/usr/bin/env python3
"""
PETGEM preprocessing - centralized CLI entry point.

Reads CLI arguments, loads the per-material conductivity table from
`-sigma_file <csv>`, and calls petgem.runPreprocessing to produce the
unified PETGEM input HDF5 plus the matching params file.

Mode-specific arguments (validated by runPreprocessing):
    -mode forward    requires -source_filename
                     forbids  -inv_source_filename, -observed_filename
    -mode inverse    requires -inv_source_filename, -observed_filename
                     forbids  -source_filename
                     (the inverse kernel reads multi-freq sources from
                      /inv_sources/* in the bundle, not /sources)

Shared arguments (both modes):
    -nord, -case_dir, -mesh_filename, -receiver_filename, -sigma_file
    -input_filename, -params_filename, -output_vtk        (all optional with defaults)

Usage examples:
    # Forward modeling
    python3 utils/preprocess.py \\
        -mode forward \\
        -nord 1 \\
        -case_dir tests/cases/csem_model \\
        -mesh_filename mesh_p1.msh \\
        -receiver_filename receivers.txt \\
        -source_filename sources.txt \\
        -sigma_file sigmas.csv \\
        [-output_vtk model.vtu]

    # Inverse modeling
    python3 utils/preprocess.py \\
        -mode inverse \\
        -nord 1 \\
        -case_dir tests/cases/inverse \\
        -mesh_filename mesh_p1.msh \\
        -receiver_filename receivers.txt \\
        -inv_source_filename sources.txt \\
        -observed_filename observed_data.h5 \\
        -sigma_file sigmas.csv \\
        [-output_vtk model.vtu]
"""
import os
import petgem


def main():
    args = petgem.parsePreprocessingArgs()
    sigma_x, sigma_y, sigma_z, fixed_materials = petgem.readSigmaCSV(
        os.path.join(args.case_dir, args.sigma_file)
    )
    petgem.runPreprocessing(
        mode=args.mode,
        nord=args.nord,
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
        output_vtk=args.output_vtk,
        dm_view=args.dm_view is not None,
    )


if __name__ == "__main__":
    main()
