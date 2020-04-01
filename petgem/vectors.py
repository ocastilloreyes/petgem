#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define standard vector and matrix functions.
'''

# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import sys
import numpy as np


# ###############################################################
# ################     FUNCTIONS DEFINITION     #################
# ###############################################################
def deleteDuplicateRows(matrix):
    ''' Delete duplicate rows in a matrix.
    :param ndarray matrix: input matrix to be processed.
    :return: matrix without duplicate rows
    :rtype: ndarray
    '''
    temp = np.copy(matrix)
    temp.sort(axis=1)

    [dummy, J, I] = findUniqueRows(temp, return_index=True,
                                   return_inverse=True)
    out = matrix[J, :]

    return (out, I)


def findUniqueRows(array, return_index=False, return_inverse=False):
    ''' Find unique rows of a two-dimensional numpy array.
    :param ndarray: array to be processed.
    :param bool return_index: the indices of array that result in the
     unique array.
    :param bool return_inverse: indices of the unique array that
     can be used to reconstruct array.
    :return: unique rows.
    :rtype: ndarray.
    '''
    array = np.ascontiguousarray(array)

    # View the rows as a 1D structured array.
    arv = array.view(array.shape[1] * [('', array.dtype)])
    out = np.unique(arv, return_index=return_index,
                    return_inverse=return_inverse)
    if isinstance(out, tuple):
        uarv = out[0]

    else:
        uarv = out

    # Restore the original dimensions.
    uar = uarv.view(array.dtype).reshape((-1, array.shape[1]))

    if isinstance(out, tuple):
        out = (uar,) + out[1:]

    else:
        out = uar

    return out


def is_duplicate_entry(x):
    ''' Compute number of duplicate entries in a vector.

    :param int-array x: matrix.
    :return: number of duplicate entries.
    :rtype: int.
    '''
    counts = np.bincount(x)
    duplicate_entries = np.where(counts > 1)[0]
    num_duplicate = duplicate_entries.size

    return num_duplicate


def invConnectivity(M, nP):
    '''This function computes the opposite connectivity matrix of M.

    :param ndarray M: connectivity matrix with dimensions = (nElems,eleOrder)
    :param int nP: number of nodes/edges/faces in matrix M.
    :return: connectivity matrix with dimensions = (nNodes,S), (nEdges,S) or (nFaces,S)
    :rtype: ndarray.

    .. note::
        eleOrder determines the number of entities per element in matrix M,
        therefore 4 is the nodal element order, 6 is the edge element order and
        3 is the element order of faces.
        S in the output is the maximum number of elements sharing a given node/edge/face

        If M represents a element/nodes connectivity, the function computes
        a node/elements connectivity.

        If M represents a element/edges connectivity, the function computes
        a edge/elements connectivity.

        If M represents a element/faces connectivity, the function computes
        a faces/elements connectivity.
    '''

    # Get matrix dimensions
    size = M.shape

    # Number of items
    nItems = size[0]

    # Element order
    orderEle = size[1]

    # Set maximum valence
    valence = 13

    # Allocate
    N = np.zeros((nP,valence), dtype=np.int)
    nn = np.ones(nP, dtype=np.int)

    # Build matrix
    for i in np.arange(nItems):
        iele = M[i,:]
        nn_Te = nn[iele]
        for j in np.arange(orderEle):
            N[iele[j], nn_Te[j]] = i

        nn[iele] += np.int(1)

    # Delete zeros of matrix N
    N = np.delete(N, np.arange(np.max(nn)-1, valence+1), axis=1)

    return N


def unitary_test():
    ''' Unitary test for vectors.py script.
    '''


if __name__ == '__main__':
    unitary_test()
