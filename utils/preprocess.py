#!/usr/bin/env python3
"""
PETGEM preprocessing - centralized CLI entry point.

Reads CLI arguments, loads the per-material conductivity table from
`-sigma_file <csv>`, and calls petgem.runPreprocessing to produce the
unified PETGEM input HDF5 plus the matching params file.

Usage examples:
    # Forward modeling
    python3 utils/preprocess.py \\
        -mode forward \\
        -nord 1 \\
        -case_dir tests/cases/csem_model \\
        -mesh_filename mesh_p1.msh \\
        -source_filename sources.txt \\
        -receiver_filename receivers.txt \\
        -sigma_file sigmas.csv \\
        -input_filename input.h5 \\
        -params_filename params.txt \\
        [-output_vtk model.vtu]

    # Inverse modeling (multi-freq sources + observed Ex embedded in bundle)
    python3 utils/preprocess.py \\
        -mode inverse \\
        -nord 1 \\
        -case_dir tests/cases/inverse \\
        -mesh_filename mesh_p1.msh \\
        -source_filename sources_fwd.txt \\
        -inv_source_filename sources.txt \\
        -observed_filename observed_data.h5 \\
        -receiver_filename receivers.txt \\
        -sigma_file sigmas.csv
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
