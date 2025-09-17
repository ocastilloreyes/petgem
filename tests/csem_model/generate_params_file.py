#!/usr/bin/env python3
"""
*********************************************************************
 Script to generate parameter files for CSEM simulations with PETGEM.
 Creates two params.txt files with different polynomial orders (-nord 1 and -nord 2).
 Output directories and filenames are adjusted accordingly.

 Author: Octavio Castillo-Reyes (UPC/BSC) (octavio.castillo@upc.edu; octavio.castillo@bsc.es)
 Latest update: September 10th, 2025
********************************************************************* 
"""

import textwrap
import sys

def write_params(nord, output_dir, output_filename):
    content = textwrap.dedent(f"""\
        -mesh_filename {output_dir}/resistivity_model_p{nord}.h5
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

    nord = sys.argv[1]
    
    # Common parameters
    base_output_dir = f"tests/csem_model"
    base_output_filename = f"responses_p{nord}"

    # params file
    write_params(nord=nord,output_dir=base_output_dir,output_filename=base_output_filename)

if __name__ == "__main__":
    main()
