#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define functions to find the solution of a sparse linear system of the
format Ax = b, in Edge Finite Element Method (EFEM) of lowest order in
tetrahedral meshes.
'''


def setBoundaryConditions(A, b, bEdges, Istart_bEdges, Iend_bEdges, rank):
    ''' Given a parallel matrix and a parallel vector, set Dirichlet boundary
    conditions.

    :param petsc matrix A: sparse and complex coefficients matrix in
                           petsc format
    :param petsc vector b: parallel right hand side
    :param petsc vector bEdges: array of boundary indexes
    :param int Istart_bEdges: init range for boundaries
    :param int Iend_bEdges: last range for boundaries
    :para int rank: MPI rank
    :return: equation system after applied Dirichlet boundary conditions
             and elapsed time
    :rtype: petsc matrix, petsc vector and float.
    '''

    PETSc.Sys.syncPrint('  Rank: ', rank, ' is setting boundary conditions')
    PETSc.Sys.syncFlush()

    # Start timer
    Init_boundaries = getTime()

    # Boundaries for LHS
    A.zeroRowsColumns(np.real(bEdges).astype(PETSc.IntType))
    # Boundaries for RHS
    numLocalBoundaries = Iend_bEdges - Istart_bEdges
    b.setValues(np.real(bEdges).astype(PETSc.IntType),
                np.zeros(numLocalBoundaries, dtype=np.complex),
                addv=PETSc.InsertMode.INSERT_VALUES)

    # Start global system assembly
    A.assemblyBegin()
    b.assemblyBegin()
    # End global system assembly
    A.assemblyEnd()
    b.assemblyEnd()

    # End timer
    End_boundaries = getTime()

    # Elapsed time in assembly
    elapsedTimeBoundaries = End_boundaries-Init_boundaries

    return A, b, elapsedTimeBoundaries


def solveSystem(A, b, x, rank):
    ''' Solve a matrix system of the form Ax = b in parallel.

    :param petsc matrix A: sparse and complex coefficients matrix in
                           petsc format
    :param petsc vector b: parallel right hand side
    :param petsc vector x: parallel solution vector
    :param int rank: MPI rank
    :return: solution of the equation system, iteration number
             of solver and elapsed time
    :rtype: petsc vector, int and float
    '''

    PETSc.Sys.syncPrint('  Rank: ', rank, ' is solving system')
    PETSc.Sys.syncFlush()

    # Start timer
    Init_solver = getTime()

    # Create KSP: linear equation solver
    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.solve(b, x)
    iterationNumber = ksp.getIterationNumber()
    ksp.destroy()

    # End timer
    End_solver = getTime()

    # Elapsed time in assembly
    elapsedTimeSolver = End_solver-Init_solver

    return x, iterationNumber, elapsedTimeSolver


def unitary_test():
    ''' Unitary test for solver.py script.
    '''

if __name__ == '__main__':
    # Standard module import
    unitary_test()
else:
    # Standard module import
    import petsc4py
    from petsc4py import PETSc
    import numpy as np
    # PETGEM module import
    from petgem.monitoring.monitoring import getTime
