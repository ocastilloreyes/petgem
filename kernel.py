#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' **PETGEM** kernel for 3D CSEM forward modelling using higg order
vector elements.
'''

if __name__ == '__main__':
    # ---------------------------------------------------------------
    # Load python modules
    # ---------------------------------------------------------------
    import sys
    import petsc4py
    import numpy as np
    # ---------------------------------------------------------------
    # PETSc init
    # ---------------------------------------------------------------
    petsc4py.init(sys.argv)
    # ---------------------------------------------------------------
    # Load python modules
    # ---------------------------------------------------------------
    from petsc4py import PETSc
    # ---------------------------------------------------------------
    # Load petgem modules (BSC)
    # ---------------------------------------------------------------
    from petgem.common import Print, InputParameters, Timers
    from petgem.parallel import MPIEnvironment
    from petgem.preprocessing import Preprocessing
    from petgem.solver import Solver

    # ---------------------------------------------------------------
    # Load system setup (both parameters and dataset configuration)
    # ---------------------------------------------------------------
    # Obtain the MPI environment
    parEnv = MPIEnvironment()

    # Import parameters file
    inputSetup = InputParameters(sys.argv[3], parEnv)

    # Initialize timers
    Timers(inputSetup.output.directory)

    # ---------------------------------------------------------------
    # Print header
    # ---------------------------------------------------------------
    Print.header()

    # ---------------------------------------------------------------
    # Initialize preprocessing and timers
    # ---------------------------------------------------------------
    Print.master(' ')
    Print.master('  Data preprocessing')

    # Create a preprocessing instance and output directory
    preprocessing = Preprocessing()

    # Run preprocessing
    preprocessing.run(inputSetup)

    # ---------------------------------------------------------------
    # Initialize and execute the solver
    # ---------------------------------------------------------------
    Print.master(' ')
    Print.master('  Run modelling')

    # Create a solver instance
    csem_solver = Solver()

    # Setup solver (import files from preprocessing stage)
    csem_solver.setup(inputSetup)

    # Assembly linear system
    csem_solver.assembly(inputSetup)

    # Set dirichlet boundary conditions
    csem_solver.solve()

    # Compute electromagnetic responses
    csem_solver.postprocess(inputSetup)

    # ---------------------------------------------------------------
    # End of PETGEM kernel
    # ---------------------------------------------------------------


















    #     # ###############################################
#     # ------------- Load PETGEM modules -------------
#     from petgem.base.styles import printPetgemHeader
#     from petgem.base.styles import printPetgemFooter
#     from petgem.base.base import readUserParams
#     from petgem.mesh.mesh import getMeshInfo
#     from petgem.efem.efem import defineEFEMConstants
#     from petgem.monitoring.monitoring import printElapsedTimes
#     from petgem.monitoring.monitoring import startLogEvent
#     from petgem.parallel.parallel import printMessage
#     from petgem.parallel.parallel import getRankSize
#     from petgem.parallel.parallel import getRanges
#     from petgem.parallel.parallel import readPetscMatrix
#     from petgem.parallel.parallel import readPetscVector
#     from petgem.parallel.parallel import createParallelMatrix
#     from petgem.parallel.parallel import createParallelVector
#     from petgem.parallel.parallel import parallelAssembler
#     from petgem.solver.solver import setBoundaryConditions
#     from petgem.solver.solver import solveSystem
#     from petgem.postprocessing.postprocessing import postProcessingFields
#
#     # ###############################################
#     # -- Get rank, size and MASTER of the MPI pool --
#     MASTER, rank, size = getRankSize()
#
#     # ###############################################
#     # ----------- Print header (Master) -------------
#     printPetgemHeader(rank)
#
#     # ###############################################
#     # ----------------- User input ------------------
#     printMessage('\nInit', rank)
#     printMessage('='*75, rank)
#     input_params = sys.argv
#
#     # ###############################################
#     # ------- Check and read parameters file --------
#     modelling = readUserParams(input_params, rank)
#
#     # ###############################################
#     # ----- Define geometry and EFEM constants  -----
#     [edgeOrder, nodalOrder,
#      numDimensions] = defineEFEMConstants(modelling['NEDELEC_ORDER'])
#
#     # ###############################################
#     # ---------------- Read mesh --------------------
#     printMessage('\nImport files', rank)
#     printMessage('='*75, rank)
#     # Create and start log event for importing task
#     importLog = startLogEvent("Import_files")
#     importLog.begin()
#     # Read nodes coordinates
#     printMessage('  Nodes coordinates', rank)
#     nodes = readPetscMatrix(modelling['NODES_FILE'], communicator=None)
#     # elements-nodes connectivity
#     printMessage('  Elements-nodes connectivity', rank)
#     elemsN = readPetscMatrix(modelling['MESH_CONNECTIVITY_FILE'],
#                              communicator=None)
#     # elements-faces connectivity
#     printMessage('  Elements-faces connectivity', rank)
#     elemsF = readPetscMatrix(modelling['FACES_CONNECTIVITY_FILE'],
#                              communicator=None)
#     # facesN connectivity
#     printMessage('  Faces-nodes connectivity', rank)
#     facesN = readPetscMatrix(modelling['FACES_NODES_FILE'],
#                              communicator=None)
#     # elements-edges connectivity
#     printMessage('  Elements-edges connectivity', rank)
#     elemsE = readPetscMatrix(modelling['EDGES_CONNECTIVITY_FILE'],
#                              communicator=None)
#     # edgesN connectivity
#     printMessage('  Edges-nodes connectivity', rank)
#     edgesN = readPetscMatrix(modelling['EDGES_NODES_FILE'], communicator=None)
#     # Boundaries
#     printMessage('  Boundary-Edges', rank)
#     boundaries = readPetscVector(modelling['BOUNDARIES_FILE'],
#                                  communicator=None)
#     # Sparsity pattern (NNZ) for matrix allocation
#     printMessage('  Vector for matrix allocation', rank)
#     Q = readPetscVector(modelling['NNZ_FILE'], communicator=None)
#     nnz = (Q.getArray().real).astype(PETSc.IntType)
#     # Conductivity model
#     printMessage('  Conductivity model', rank)
#     elemsSigma = readPetscVector(modelling['CONDUCTIVITY_MODEL_FILE'],
#                                  communicator=None)
#     # Receivers data
#     printMessage('  Receivers data', rank)
#     receivers = readPetscMatrix(modelling['RECEIVERS_FILE'], communicator=None)
#     # End log event for importing task
#     importLog.end()
#
#     # ###############################################
#     # -------------- Mesh information ---------------
#     [nElems, nFaces, nEdges, ndofs,
#      nBoundaries] = getMeshInfo(modelling['NEDELEC_ORDER'], elemsN, elemsF,
#                                 facesN, edgesN, boundaries, rank)
#
#     # ###############################################
#     # --------- Information of parallel pool --------
#     printMessage('\nParallel information', rank)
#     printMessage('='*75, rank)
#     [Istart_elemsE, Iend_elemsE,
#      Istart_boundaries, Iend_boundaries,
#      Istart_receivers, Iend_receivers] = getRanges(elemsE, boundaries,
#                                                    receivers, size, rank)
#
#     # ###############################################
#     # ----- Create and setup parallel structures ----
#     # Left-hand side
#     A = createParallelMatrix(ndofs, ndofs, nnz, modelling['CUDA'],
#                              communicator=None)
#     # Right-hand side
#     b = createParallelVector(ndofs, modelling['CUDA'], communicator=None)
#     # X vector
#     x = createParallelVector(ndofs, modelling['CUDA'], communicator=None)
#
#     # ###############################################
#     # -------------- Parallel assembly --------------
#     printMessage('\nParallel assembly', rank)
#     printMessage('='*75, rank)
#     # Create and start log event for assembly task
#     assemblerLog = startLogEvent("Assembler")
#     assemblerLog.begin()
#     # System assembly
#     [A, b, elapsedTimeAssembly] = parallelAssembler(modelling, A, b, nodes,
#                                                     elemsE, elemsN, elemsF,
#                                                     facesN,  elemsSigma,
#                                                     Istart_elemsE, Iend_elemsE,
#                                                     nEdges, nFaces, rank)
#     # End log event for assembly task
#     assemblerLog.end()
#
#     # ###############################################
#     # ----------- Set boundary conditions -----------
#     printMessage('\nBoundary conditions', rank)
#     printMessage('='*75, rank)
#     # Create and start log event for setting boundary conditions task
#     boundariesLog = startLogEvent("Boundaries")
#     boundariesLog.begin()
#     # Set boundary conditions
#     [A, b, elapsedTimeBoundaries] = setBoundaryConditions(A, b, boundaries,
#                                                           Istart_boundaries,
#                                                           Iend_boundaries,
#                                                           rank)
#     # End log event for setting boundary conditions task
#     boundariesLog.end()
#
#     # ###############################################
#     # ------------------- Solver --------------------
#     printMessage('\nSolver information', rank)
#     printMessage('='*75, rank)
#     # Create and start log event for assembly task
#     solverLog = startLogEvent("Solver")
#     solverLog.begin()
#     # Solve system
#     [x, iterationNumber, elapsedTimeSolver] = solveSystem(A, b, x, rank)
#     # End log event for setting boundary conditions task
#     solverLog.end()
#
#     # ###############################################
#     # --------------- Post-processing ---------------
#     printMessage('\nPost-processing', rank)
#     printMessage('='*75, rank)
#     # # Create and start log event for assembly task
#     postProcessingLog = startLogEvent("Postprocessing")
#     postProcessingLog.begin()
#     elapsedTimepostProcess = postProcessingFields(receivers, modelling, x,
#                                                   Iend_receivers,
#                                                   Istart_receivers,
#                                                   modelling['NEDELEC_ORDER'],
#                                                   modelling['CUDA'],
#                                                   nodalOrder,
#                                                   numDimensions, rank)
#     postProcessingLog.end()
#
#     # ###############################################
#     # -------------- Print elapsed times-------------
#     printMessage('\nElapsed times (seconds)', rank)
#     printMessage('='*75, rank)
#     printElapsedTimes(elapsedTimeAssembly, elapsedTimeSolver,
#                       elapsedTimepostProcess, iterationNumber, rank)
#
#     # ###############################################
#     # ----------- Print footer (Master) -------------
#     printPetgemFooter(rank)
# else:
#     pass
