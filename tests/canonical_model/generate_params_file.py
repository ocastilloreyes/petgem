# *********************************************************************
# Script to generate parameter files for CSEM simulations with PETGEM.
# Creates two params.txt files with different polynomial orders (-nord 1 and -nord 2).
# Output directories and filenames are adjusted accordingly.
#
# Author: Octavio Castillo-Reyes (UPC/BSC) (octavio.castillo@upc.edu; octavio.castillo@bsc.es)
# Latest update: September 10th, 2025
#********************************************************************* 
import textwrap


def write_params(nord, output_dir, output_filename):
    content = textwrap.dedent(f"""\
        -mesh_filename {output_dir}/canonical_model.h5
        -receivers_filename {output_dir}/receivers.h5
        -source_filename {output_dir}/sources.txt
        -nord {nord}
        -pc_type lu 
        -pc_factor_mat_solver_type mumps
        -output_dir {output_dir}/
        -output_filename {output_filename}
        -malloc_dump
    """)

    filename = f"{output_dir}/params_nord{nord}.txt"
    with open(filename, "w") as f:
        f.write(content)
    print(f"Created {filename}")


def main():
    # Parameters common to both files
    base_output_dir = "tests/canonical_model"
    base_output_filename = "responses_p"

    # Create for nord=1
    write_params(
        nord=1,
        output_dir=f"{base_output_dir}",
        output_filename=f"{base_output_filename}1"
    )

    # Create for nord=2
    write_params(
        nord=2,
        output_dir=f"{base_output_dir}",
        output_filename=f"{base_output_filename}2"
    )


if __name__ == "__main__":
    main()
