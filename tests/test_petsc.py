#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es

import pytest
import sys
from petgem.parallel import readPetscMatrix, readPetscVector
from petgem.parallel import MPIEnvironment
import petsc4py

# Initialize petsc
petsc4py.init(sys.argv)

def test_petsc_environment():
    """Verify petsc enviroment by solving a test linear system."""
    from petsc4py import PETSc

    # Obtain the MPI environment
    parEnv = MPIEnvironment()

    # Import sparse matrix (LHS)
    A = readPetscMatrix('tests/data/matrix-A.dat', parEnv.comm)

    # Import vector (RHS)
    b = readPetscVector('tests/data/vector-b.dat', parEnv.comm)

    # Import solution
    x = readPetscVector('tests/data/vector-x.dat', parEnv.comm)

    # Create KSP object
    ksp = PETSc.KSP().create(comm=parEnv.comm)
    ksp.setInitialGuessNonzero(True)
    ksp.setOperators(A, A)
    ksp.setFromOptions()
    ksp.solve(b, x)

    # Destroy ksp object and petsc objects
    ksp.destroy()
    A.destroy()
    b.destroy()
    x.destroy()
