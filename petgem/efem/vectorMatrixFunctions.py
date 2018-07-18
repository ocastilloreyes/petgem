#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define standard vector and matrix functions.
'''


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


def unitary_test():
    ''' Unitary test for vector_matrix_functions.py script.
    '''

if __name__ == '__main__':
    # Standard module import
    import sys
    unitary_test()
else:
    # Standard module import
    import sys
    import numpy as np
