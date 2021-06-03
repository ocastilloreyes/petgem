#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
"""**PETGEM** kernel for 3D CSEM/MT forward modelling using high order vector elements."""

if __name__ == '__main__':
    # ---------------------------------------------------------------
    # Load python modules
    # ---------------------------------------------------------------
    import sys
    import petsc4py
    # ---------------------------------------------------------------
    # PETSc init
    # ---------------------------------------------------------------
    petsc4py.init(sys.argv)
    # ---------------------------------------------------------------
    # Load python modules
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # Load petgem modules (BSC)
    # ---------------------------------------------------------------
    from petgem.common import Print, InputParameters, Timers
    from petgem.parallel import MPIEnvironment
    from petgem.preprocessing import Preprocessing
    from petgem.solver import Solver
    from petgem.postprocessing import Postprocessing

    # ---------------------------------------------------------------
    # Load system setup (both parameters and dataset configuration)
    # ---------------------------------------------------------------
    # Obtain the MPI environment
    par_env = MPIEnvironment()

    # Import parameters file
    input_setup = InputParameters(sys.argv[3], par_env)

    # Initialize timers
    Timers(input_setup.output['directory'])

    # ---------------------------------------------------------------
    # Print header
    # ---------------------------------------------------------------
    Print.header()

    # ---------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------
    Print.master(' ')
    Print.master('  Data preprocessing')

    # Create a preprocessing instance and output directory
    preprocessing = Preprocessing()

    # Run preprocessing
    preprocessing.run(input_setup)

    # ---------------------------------------------------------------
    # Solver
    # ---------------------------------------------------------------
    Print.master(' ')
    Print.master('  Run modelling')

    # Create a solver instance
    solver = Solver()

    # Setup solver (import files from preprocessing stage)
    solver.setup(input_setup)

    # Assembly linear system
    solver.assembly(input_setup)

    # Run solver (Ax=b)
    solver.run(input_setup)

    # Destroy solver
    del solver

    # ---------------------------------------------------------------
    # Postprocessing
    # ---------------------------------------------------------------
    Print.master(' ')
    Print.master('  Data postprocessing')

    # Create a postprocessing instance
    postprocessing = Postprocessing()

    # Run postprocessing
    postprocessing.run(input_setup)

    # ---------------------------------------------------------------
    # End of PETGEM kernel
    # ---------------------------------------------------------------
