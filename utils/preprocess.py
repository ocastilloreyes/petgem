#!/usr/bin/env python3
"""
PETGEM preprocessing - centralized CLI entry point.

Reads CLI arguments, loads the per-material conductivity table from
`-sigma_file <csv>`, and calls petgem.runPreprocessing to produce the
unified PETGEM input HDF5 plus the matching params file.

Usage example:
    python3 utils/preprocess.py \\
        -mode forward \\
        -nord 1 \\
        -case_dir tests/csem_model \\
        -mesh_filename mesh_p1.msh \\
        -source_filename sources.txt \\
        -receiver_filename receivers.txt \\
        -sigma_file sigmas.csv \\
        -input_filename input.h5 \\
        -params_filename params.txt \\
        [-output_vtk model.vtu]
"""
import os
import petgem


def main():
    args = petgem.parsePreprocessingArgs()
    sigma_x, sigma_y, sigma_z = petgem.readSigmaCSV(
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
        input_filename=args.input_filename,
        params_filename=args.params_filename,
        output_vtk=args.output_vtk,
        dm_view=args.dm_view is not None,
    )


if __name__ == "__main__":
    main()
