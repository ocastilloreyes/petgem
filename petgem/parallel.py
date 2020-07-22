#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define classes and functions for parallel computations within **PETGEM**.
'''

# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import numpy as np
from mpi4py import MPI
from singleton_decorator import singleton
from petsc4py import PETSc

# ###############################################################
# ################     CLASSES DEFINITION      ##################
# ###############################################################

@singleton
class MPIEnvironment():
    def __init__(self):
        ''' Class for initialization of an MPI environment.

        :param: None.
        :return: class for MPI environment.
        :rtype: mpi_env class.
        '''

        # Store the MPI environment
        self.MPI = MPI

        # MPI communicator
        self.comm = MPI.COMM_WORLD
        # MPI process identifier
        self.rank = self.comm.Get_rank()
        # Size of the group associated with a communicator
        self.num_proc = self.comm.Get_size()

        return


# ###############################################################
# ################     FUNCTIONS DEFINITION     #################
# ###############################################################

def createSequentialDenseMatrixWithArray(dimension1, dimension2, data):
    ''' Given an input array, create a sequential dense matrix in petsc format.
    :param int dimension1: matrix dimension (rows).
    :param int dimension2: matrix dimension (columns).
    :param ndarray data: data to be exported.
    :return: parallel matrix.
    :rtype: petsc parallel and dense matrix.
    '''

    parallel_matrix = PETSc.Mat().createDense([dimension1, dimension2],
                                              array=data, comm=PETSc.COMM_SELF)

    parallel_matrix.setFromOptions()
    parallel_matrix.setUp()

    return parallel_matrix


def writeParallelDenseMatrix(output_file, data, communicator=None):
    ''' Write a Petsc parallel dense matrix which format is defined by two
    files: output_file.dat and output_file.info.
    :param str output_file: file name to be saved.
    :param petsc matrix data: dense matrix to be saved.
    :param str communicator: mpi communicator.
    :return: None.
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    viewer = PETSc.Viewer().createBinary(output_file, mode='w', comm=communicator)
    viewer(data)

    return


def createSequentialVectorWithArray(data):
    ''' Given an input array, create a sequential vector in petsc format.
    :param ndarray data: data to be exported.
    :return: parallel matrix.
    :rtype: petsc parallel and dense matrix.
    '''

    parallel_vector = PETSc.Vec().createWithArray(data, comm=PETSc.COMM_SELF)

    parallel_vector.setUp()

    return parallel_vector


def writePetscVector(output_file, data, communicator=None):
    ''' Write a Petsc vector which format is defined by two files:
    output_file.dat and output_file.info.
    :param str output_file: file name to be saved.
    :param petsc vector data: array to be saved.
    :param str communicator: mpi communicator.
    :return: None.
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    viewer = PETSc.Viewer().createBinary(output_file, mode='w',
                                         comm=communicator)
    viewer(data)

    return


def readPetscMatrix(input_file, communicator=None):
    ''' Read a Petsc matrix which format is defined by two files:
    input_file.dat and input_file.info
    :param str input_file: file name to be readed.
    :param str communicator: mpi communicator.
    :return: petsc_matrix.
    :rtype: petsc sparse matrix.
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
    :param str input_file: file name to be readed.
    :param str communicator: mpi communicator.
    :return: petsc_vector.
    :rtype: petsc vector.
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    viewer = PETSc.Viewer().createBinary(input_file, mode='r',
                                         comm=communicator)
    petsc_vector = PETSc.Vec().load(viewer)

    return petsc_vector


def createParallelMatrix(dimension1,dimension2,nnz,matrix_type,communicator=None):
    ''' Create a parallel sparse matrix in petsc format.
    :param int dimension1: matrix dimension (rows).
    :param int dimension2: matrix dimension (columns).
    :param int nnz: not zero pattern for allocation.
    :param int matrix_type: matrix type for parallel computations.
    :param str communicator: mpi communicator.
    :return: parallel matrix.
    :rtype: petsc AIJ parallel matrix.
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    if matrix_type:    # Cuda support
        parallel_matrix = PETSc.Mat().create(comm=communicator)
        parallel_matrix.setType('aijcusparse')
        parallel_matrix.setSizes([dimension1, dimension2])
    elif not matrix_type:    # No cuda support
        parallel_matrix = PETSc.Mat().createAIJ([dimension1, dimension2], comm=communicator)
    else:
        raise ValueError('Cuda option=', matrix_type, ' not supported.')

    parallel_matrix.setPreallocationNNZ((nnz, nnz))
    parallel_matrix.setFromOptions()
    parallel_matrix.setUp()

    return parallel_matrix


def createParallelVector(size, vector_type, communicator=None):
    ''' Create a parallel vector in petsc format.
    :param int size: vector size.
    :param int vector_type: vector type for parallel computations.
    :param str communicator: mpi communicator.
    :return: parallel vector.
    :rtype: petsc parallel vector.
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    if vector_type:    # Cuda support
        parallel_vector = PETSc.Vec().create(comm=communicator)
        parallel_vector.setType('cuda')
        parallel_vector.setSizes(size)
    elif not vector_type:    # No cuda support
        parallel_vector = PETSc.Vec().createMPI(size, comm=communicator)
    else:
        raise ValueError('Cuda option=', vector_type, ' not supported.')

    parallel_vector.setUp()

    return parallel_vector


def createSequentialVector(size, vector_type, communicator=None):
    ''' Create a sequential vector in petsc format.
    :param int size: vector size.
    :param int vector_type: vector type for parallel computations.
    :param str communicator: mpi communicator.
    :return: sequential vector.
    :rtype: petsc sequential vector.
    '''
    if communicator is None:
        communicator = PETSc.COMM_SELF

    if vector_type == 0:    # No cuda support
        sequential_vector = PETSc.Vec().createSeq(size, comm=communicator)
    elif vector_type == 1:    # Cuda support
        sequential_vector = PETSc.Vec().create(comm=communicator)
        sequential_vector.setType('cuda')
        sequential_vector.setSizes(size)

    sequential_vector.setUp()

    return sequential_vector


def createParallelDenseMatrix(dimension1, dimension2, communicator=None):
    ''' Create a parallel dense matrix in petsc format.
    :param int dimension1: matrix dimension (rows).
    :param int dimension2: matrix dimension (columns).
    :param str communicator: mpi communicator.
    :return: parallel matrix.
    :rtype: petsc parallel and dense matrix.
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    parallel_matrix = PETSc.Mat().createDense([dimension1, dimension2],
                                              comm=communicator)
    parallel_matrix.setFromOptions()
    parallel_matrix.setUp()

    return parallel_matrix


def writeDenseMatrix(output_file, data, communicator=None):
    ''' Write a Petsc dense matrix which format is defined by two files:
    output_file.dat and output_file.info.
    :param str output_file: file name to be saved.
    :param petsc matrix data: dense matrix to be saved.
    :param str communicator: mpi communicator.
    :return: None.
    '''
    if communicator is None:
        communicator = PETSc.COMM_WORLD

    viewer = PETSc.Viewer().createBinary(output_file, mode='w',
                                         comm=communicator)
    viewer.pushFormat(viewer.Format.NATIVE)

    viewer(data)

    return


def unitary_test():
    ''' Unitary test for parallel.py script.
    '''

# ###############################################################
# ################             MAIN             #################
# ###############################################################

if __name__ == '__main__':
    # ---------------------------------------------------------------
    # Run unitary test
    # ---------------------------------------------------------------
    unitary_test()
