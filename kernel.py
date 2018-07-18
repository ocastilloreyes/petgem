#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' **PETGEM** kernel. It solve a CSEM forward modelling according to
parameters into a **PETGEM** parameters file format.
'''

if __name__ == '__main__':
    # ###############################################
    # ---------- Load modules and PETSc init --------
    import sys
    import petsc4py
    # ###############################################
    # ------------------ PETSc init -----------------
    petsc4py.init(sys.argv)
    # ###############################################
    # -----------------------------------------------
    import numpy as np
    from petsc4py import PETSc
    # ###############################################
    # ------------- Load PETGEM modules -------------
    from petgem.base.styles import printPetgemHeader
    from petgem.base.styles import printPetgemFooter
    from petgem.base.base import readUserParams
    from petgem.mesh.mesh import getMeshInfo
    from petgem.efem.efem import defineEFEMConstants
    from petgem.monitoring.monitoring import printElapsedTimes
    from petgem.monitoring.monitoring import startLogEvent
    from petgem.parallel.parallel import printMessage
    from petgem.parallel.parallel import getRankSize
    from petgem.parallel.parallel import getRanges
    from petgem.parallel.parallel import readPetscMatrix
    from petgem.parallel.parallel import readPetscVector
    from petgem.parallel.parallel import createParallelMatrix
    from petgem.parallel.parallel import createParallelVector
    from petgem.parallel.parallel import parallelAssembler
    from petgem.solver.solver import setBoundaryConditions
    from petgem.solver.solver import solveSystem
    from petgem.postprocessing.postprocessing import postProcessingFields

    # ###############################################
    # -- Get rank, size and MASTER of the MPI pool --
    MASTER, rank, size = getRankSize()

    # ###############################################
    # ----- Define geometry and EFEM constants  -----
    edgeOrder, nodalOrder, numDimensions = defineEFEMConstants()

    # ###############################################
    # ----------- Print header (Master) -------------
    printPetgemHeader(rank)

    # ###############################################
    # ----------------- User input ------------------
    printMessage('\nInit', rank)
    printMessage('='*75, rank)
    input_params = sys.argv

    # ###############################################
    # ------- Check and read parameters file --------
    modelling = readUserParams(input_params, rank)

    # ###############################################
    # ---------------- Read mesh --------------------
    printMessage('\nImport files', rank)
    printMessage('='*75, rank)
    # Create and start log event for importing task
    importLog = startLogEvent("Import_files")
    importLog.begin()
    # Read nodes coordinates
    printMessage('  Nodes coordinates', rank)
    nodes = readPetscMatrix(modelling['NODES_FILE'], communicator=None)
    # elements-nodes connectivity
    printMessage('  Elements-nodes connectivity', rank)
    elemsN = readPetscMatrix(modelling['MESH_CONNECTIVITY_FILE'],
                             communicator=None)
    # elements-edges connectivity
    printMessage('  Elements-edges connectivity', rank)
    elemsE = readPetscMatrix(modelling['DOFS_CONNECTIVITY_FILE'],
                             communicator=None)
    # edgesN connectivity
    printMessage('  Edges-nodes connectivity', rank)
    edgesN = readPetscMatrix(modelling['DOFS_NODES_FILE'], communicator=None)
    # Boundary-Edges
    printMessage('  Boundary-Edges', rank)
    bEdges = readPetscVector(modelling['BOUNDARIES_FILE'], communicator=None)
    # Sparsity pattern (NNZ) for matrix allocation
    printMessage('  Vector for matrix allocation', rank)
    Q = readPetscVector(modelling['NNZ_FILE'], communicator=None)
    nnz = (Q.getArray().real).astype(PETSc.IntType)
    # Conductivity model
    printMessage('  Conductivity model', rank)
    elemsSigma = readPetscVector(modelling['CONDUCTIVITY_MODEL_FILE'],
                                 communicator=None)
    # Receivers data
    printMessage('  Receivers data', rank)
    receivers = readPetscMatrix(modelling['RECEIVERS_FILE'], communicator=None)
    # End log event for importing task
    importLog.end()

    # ###############################################
    # -------------- Mesh information ---------------
    [nElems, nEdges, nBoundaries, ndofs] = getMeshInfo(elemsN, edgesN,
                                                       bEdges, rank)

    # ###############################################
    # --------- Information of parallel pool --------
    printMessage('\nParallel information', rank)
    printMessage('='*75, rank)
    [Istart_elemsE, Iend_elemsE,
     Istart_bEdges, Iend_bEdges,
     Istart_receivers, Iend_receivers] = getRanges(elemsE, bEdges, receivers,
                                                   size, rank)

    # ###############################################
    # ----- Create and setup parallel structures ----
    # Left-hand side
    A = createParallelMatrix(nEdges, nEdges, nnz, communicator=None)
    # Right-hand side
    b = createParallelVector(nEdges, communicator=None)
    # X vector
    x = createParallelVector(nEdges, communicator=None)

    # ###############################################
    # -------------- Parallel assembly --------------
    printMessage('\nParallel assembly', rank)
    printMessage('='*75, rank)
    # Create and start log event for assembly task
    assemblerLog = startLogEvent("Assembler")
    assemblerLog.begin()
    # System assembly
    [A, b, elapsedTimeAssembly] = parallelAssembler(modelling, A, b, nodes,
                                                    elemsE, elemsN, elemsSigma,
                                                    Istart_elemsE, Iend_elemsE,
                                                    rank)
    # End log event for assembly task
    assemblerLog.end()

    # ###############################################
    # ----------- Set boundary conditions -----------
    printMessage('\nBoundary conditions', rank)
    printMessage('='*75, rank)
    # Create and start log event for setting boundary conditions task
    boundariesLog = startLogEvent("Boundaries")
    boundariesLog.begin()
    # Set boundary conditions
    [A, b, elapsedTimeBoundaries] = setBoundaryConditions(A, b, bEdges,
                                                          Istart_bEdges,
                                                          Iend_bEdges, rank)
    # End log event for setting boundary conditions task
    boundariesLog.end()

    # ###############################################
    # ------------------- Solver --------------------
    printMessage('\nSolver information', rank)
    printMessage('='*75, rank)
    # Create and start log event for assembly task
    solverLog = startLogEvent("Solver")
    solverLog.begin()
    # Solve system
    [x, iterationNumber, elapsedTimeSolver] = solveSystem(A, b, x, rank)
    # End log event for setting boundary conditions task
    solverLog.end()

    # ###############################################
    # --------------- Post-processing ---------------
    printMessage('\nPost-processing', rank)
    printMessage('='*75, rank)
    # Create and start log event for assembly task
    postProcessingLog = startLogEvent("Postprocessing")
    postProcessingLog.begin()
    elapsedTimepostProcessing = postProcessingFields(receivers, modelling, x,
                                                     Iend_receivers,
                                                     Istart_receivers,
                                                     edgeOrder, nodalOrder,
                                                     numDimensions,  rank)
    postProcessingLog.end()

    # ###############################################
    # -------------- Print elapsed times-------------
    printMessage('\nElapsed times (seconds)', rank)
    printMessage('='*75, rank)
    printElapsedTimes(elapsedTimeAssembly, elapsedTimeSolver,
                      elapsedTimepostProcessing, iterationNumber, rank)

    # ###############################################
    # ----------- Print footer (Master) -------------
    printPetgemFooter(rank)
else:
    pass
