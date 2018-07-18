#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define parallel functions for Edge Finite Element Method (EFEM) of
lowest order in tetrahedral meshes, namely, Nedelec elements.
'''


def getRankSize():
    ''' Get information of parallel pool, namely Rank and size. Furthermore,
    here is defined the Rank==0 as MASTER whithin the parallel pool.

    :param: None.
    :return: MASTER, rank, size.
    :rtype: integer.
    '''
    # Define rank for MPI master
    MASTER = 0
    # -------- Determine rank and size ---------
    rank = PETSc.COMM_WORLD.getRank()
    size = PETSc.COMM_WORLD.getSize()

    return MASTER, rank, size


def printMessage(msg, rank):
    ''' Master prints a message in a parallel pool.

    :param str msg: message to be printed.
    :param int rank: MPI rank.
    :return: None.
    '''
    if rank == 0:
        PETSc.Sys.Print(msg)

    return


def createParallelMatrix(dimension1, dimension2, nnz, communicator=None):
    ''' Create a parallel sparse matrix in petsc format.

    :param int dimension1: matrix dimension (rows)
    :param int dimension2: matrix dimension (columns)
    :param int nnz: not zero pattern for allocation
    :param str communicator: mpi communicator
    :return: parallel matrix
    :rtype: petsc AIJ parallel matrix
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    parallel_matrix = PETSc.Mat().createAIJ([dimension1, dimension2],
                                            comm=communicator)
    parallel_matrix.setPreallocationNNZ((nnz, nnz))
    parallel_matrix.setFromOptions()
    parallel_matrix.setUp()

    return parallel_matrix


def createParallelVector(size, communicator=None):
    ''' Create a parallel vector in petsc format.

    :param int size: vector size
    :param str communicator: mpi communicator
    :return: parallel vector
    :rtype: petsc parallel vector
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    parallel_vector = PETSc.Vec().createMPI(size, comm=communicator)
    parallel_vector.setUp()

    return parallel_vector


def createParallelDenseMatrix(dimension1, dimension2, communicator=None):
    ''' Create a parallel dense matrix in petsc format.

    :param int dimension1: matrix dimension (rows)
    :param int dimension2: matrix dimension (columns)
    :param str communicator: mpi communicator
    :return: parallel matrix
    :rtype: petsc parallel and dense matrix
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    parallel_matrix = PETSc.Mat().createDense([dimension1, dimension2],
                                              comm=communicator)
    parallel_matrix.setFromOptions()
    parallel_matrix.setUp()

    return parallel_matrix


def createSequentialDenseMatrixWithArray(dimension1, dimension2, data):
    ''' Given an input array, create a sequential dense matrix in petsc format.

    :param int dimension1: matrix dimension (rows)
    :param int dimension2: matrix dimension (columns)
    :param ndarray data: data to be exported
    :return: parallel matrix
    :rtype: petsc parallel and dense matrix
    '''

    parallel_matrix = PETSc.Mat().createDense([dimension1, dimension2],
                                              array=data, comm=PETSc.COMM_SELF)

    parallel_matrix.setFromOptions()
    parallel_matrix.setUp()

    return parallel_matrix


def createSequentialVectorWithArray(data):
    ''' Given an input array, create a sequential vector in petsc format.

    :param ndarray data: data to be exported
    :return: parallel matrix
    :rtype: petsc parallel and dense matrix
    '''

    parallel_vector = PETSc.Vec().createWithArray(data, comm=PETSc.COMM_SELF)

    parallel_vector.setUp()

    return parallel_vector


def createSequentialVector(size, communicator=None):
    ''' Create a sequential vector in petsc format.

    :param int size: vector size
    :param str communicator: mpi communicator
    :return: sequential vector
    :rtype: petsc sequential vector
    '''
    if communicator is None:
        communicator = PETSc.COMM_SELF

    sequential_vector = PETSc.Vec().createSeq(size, comm=communicator)
    sequential_vector.setUp()

    return sequential_vector


def readPetscMatrix(input_file, communicator=None):
    ''' Read a Petsc matrix which format is defined by two files:
    input_file.dat and input_file.info

    :param str input_file: file name to be readed
    :param str communicator: mpi communicator
    :return: petsc_matrix
    :rtype: petsc sparse matrix
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    viewer = PETSc.Viewer().createBinary(input_file, mode='r',
                                         comm=communicator)
    petsc_matrix = PETSc.Mat().load(viewer)

    return petsc_matrix


def readPetscVector(input_file, communicator=None):
    ''' Read a Petsc vector which format is defined by two files:
    input_file.dat and input_file.info.

    :param str input_file: file name to be readed
    :param str communicator: mpi communicator
    :return: petsc_vector
    :rtype: petsc vector
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    viewer = PETSc.Viewer().createBinary(input_file, mode='r',
                                         comm=communicator)
    petsc_vector = PETSc.Vec().load(viewer)

    return petsc_vector


def writePetscVector(output_file, data, communicator=None):
    ''' Write a Petsc vector which format is defined by two files:
    output_file.dat and output_file.info.

    :param str output_file: file name to be saved
    :param petsc vector data: array to be saved
    :param str communicator: mpi communicator
    :return: None
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    viewer = PETSc.Viewer().createBinary(output_file, mode='w',
                                         comm=communicator)
    viewer(data)

    return


def writeDenseMatrix(output_file, data, communicator=None):
    ''' Write a Petsc dense matrix which format is defined by two files:
    output_file.dat and output_file.info.

    :param str output_file: file name to be saved
    :param petsc matrix data: dense matrix to be saved
    :param str communicator: mpi communicator
    :return: None
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    viewer = PETSc.Viewer().createBinary(output_file, mode='w',
                                         comm=communicator)
    viewer.pushFormat(viewer.Format.NATIVE)

    viewer(data)

    return


def writeParallelDenseMatrix(output_file, data, communicator=None):
    ''' Write a Petsc parallel dense matrix which format is defined by two
    files: output_file.dat and output_file.info.

    :param str output_file: file name to be saved
    :param petsc matrix data: dense matrix to be saved
    :param str communicator: mpi communicator
    :return: None
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    viewer = PETSc.Viewer().createBinary(output_file, mode='w',
                                         comm=communicator)
    viewer(data)

    return


def getRanges(elemsE, bEdges, receivers, size, rank):
    ''' Get owner ship ranges of matrices and vectors for parallel computations.

    :param petsc matrix elemsE: elements-edges connectivity
    :param petsc vector bEdges: boundary-edges indexes
    :param petsc matrix receivers: receivers matrix
    :param int size: size of the parallel pool
    :para int rank: MPI rank
    :return: Istart_elemsE, Iend_elemsE, Istart_bEdges,
             Iend_bEdges, Istart_bEdges, Iend_bEdges,
             Istart_receivers, Iend_receivers
    :rtype: int
    '''
    # Get global indices
    # elemsE
    Istart_elemsE, Iend_elemsE = elemsE.getOwnershipRange()
    # bEdges
    Istart_bEdges, Iend_bEdges = bEdges.getOwnershipRange()
    # receivers
    Istart_receivers, Iend_receivers = receivers.getOwnershipRange()
    if rank == 0:
        # Print size of MPI pool
        PETSc.Sys.Print('  Number of MPI tasks: ', size)
        # Print ranges
        PETSc.Sys.Print('  Global ranges over elements: ')

    PETSc.Sys.syncPrint('    Rank: ', rank, 'low/high: ',
                        Istart_elemsE, Iend_elemsE)
    PETSc.Sys.syncFlush()

    return (Istart_elemsE, Iend_elemsE, Istart_bEdges,
            Iend_bEdges, Istart_receivers, Iend_receivers)


def parallelAssembler(modelling, A, b, nodes, elemsE, elemsN, elemsSigma,
                      Istart_elemsE, Iend_elemsE, rank):
    ''' Assembly matrix A and vector b for 3D CSEM in parallel.

    :param dictionary modelling: CSEM modelling with physical parameters
    :param petsc matrix A: left-hand side
    :param petsc vector b: right-hand side
    :param petsc matrix nodes: nodal coordinates
    :param petsc matrix elemsE: elements-edges connectivity
    :param petsc matrix elemsN: elements-nodes connectivity
    :param petsc vector elemsSigma: elements-conductivity array
    :param int Istart_elemsE: init range for assembly
    :param int Iend_elemsE: last range for assembly
    :para int rank: MPI rank
    :return: matrix A, vector b assembled, elapsedTimeAssembly
    :rtype: petsc matrix, petsc vector and float
    '''
    # Print information of assembly
    PETSc.Sys.syncPrint('  Rank: ', rank, ' is assembling ',
                        Iend_elemsE-Istart_elemsE, ' elements')
    PETSc.Sys.syncFlush()

    # Start timer
    Init_assembly = getTime()

    # Compute contributions for all local elements
    for iEle in np.arange(Istart_elemsE, Iend_elemsE):
        # Get coordinates of iEle
        coordEle = nodes.getRow(iEle)[1].real
        # Get edges of iEle
        edgesEle = (elemsE.getRow(iEle)[1].real).astype(PETSc.IntType)
        # Get nodal indexes of iEle
        nodesEle = elemsN.getRow(iEle)[1].real
        # Get sigma of iEle
        sigmaEle = elemsSigma.getValue(iEle).real
        # Compute elemental contributions for iEle
        # Elemental matrix (Ae) and elemental vector (be)
        [Ae, be] = computeElementalContributionsMPI(modelling, coordEle,
                                                    nodesEle, sigmaEle)
        # Add local contributions to global matrix
        A.setValues(edgesEle, edgesEle, Ae, addv=PETSc.InsertMode.ADD_VALUES)
        # Add local contributions to global vector
        b.setValues(edgesEle, be, addv=PETSc.InsertMode.ADD_VALUES)

    # Start global system assembly
    A.assemblyBegin()
    b.assemblyBegin()
    # End global system assembly
    A.assemblyEnd()
    b.assemblyEnd()

    # End timer
    End_assembly = getTime()

    # Elapsed time in assembly
    elapsedTimeAssembly = End_assembly-Init_assembly

    return A, b, elapsedTimeAssembly


def unitary_test():
    ''' Unitary test for parallel.py script.
    '''

if __name__ == '__main__':
    # Standard module import
    unitary_test()
else:
    # Standard module import
    import numpy as np
    from petsc4py import PETSc
    # PETGEM module import
    from petgem.monitoring.monitoring import getTime
    from petgem.solver.assembler import computeElementalContributionsMPI
