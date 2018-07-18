#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define the classes, methods and functions for Edge Finite
Element Method (EFEM) of lowest order in tetrahedral meshes, namely,
Nedelec elements.
'''


def computeDofs(elemsN, nElems):
    ''' Compute degrees of freedom (DOF) of a 3D tetrahedral mesh. Here, dofs
    are defined for edge finite element computations.

    :param ndarray elemsN: elements-nodes connectivity.
    :param int nElems: number of tetrahedral elements in the mesh.
    :return: dof connectivity and dofsNodes connectivity.
    :rtype: ndarray
    '''

    # Extracts sets of edges
    edges1 = elemsN[:, [0, 1]]
    edges2 = elemsN[:, [0, 2]]
    edges3 = elemsN[:, [0, 3]]
    edges4 = elemsN[:, [1, 2]]
    edges5 = elemsN[:, [3, 1]]
    edges6 = elemsN[:, [2, 3]]

    # Edges as sets of their nodes (vertices)
    vertices = np.zeros([nElems*6, 2])
    vertices[0::6] = edges1
    vertices[1::6] = edges2
    vertices[2::6] = edges3
    vertices[3::6] = edges4
    vertices[4::6] = edges5
    vertices[5::6] = edges6

    # Delete duplicate rows
    [dofsNodes, dofs] = deleteDuplicateRows(vertices)

    # Build dofs matrix
    dofs = np.array(np.reshape(dofs, (nElems, 6)), dtype=np.int)

    # Build dofs to nodes connectivity
    dofsNodes.sort(axis=1)
    dofsNodes = np.array(dofsNodes, dtype=np.int)

    return dofs, dofsNodes


def computeFaces(elemsN, nElems):
    ''' Compute the element\'s faces of a 3D tetrahedral mesh.

    :param ndarray matrix: elements-nodes connectivity.
    :return: element/faces connectivity.
    :rtype: ndarray

    .. note:: References:\n
       Rognes, Marie E., Robert Cndarray. Kirby, and Anders Logg. "Efficient
       assembly of H(div) and H(curl) conforming finite elements."
       SIAM Journal on Scientific Computing 31.6 (2009): 4130-4151.
    '''

    # Extracts sets of faces
    faces1 = elemsN[:, [0, 2, 1]]
    faces2 = elemsN[:, [0, 1, 3]]
    faces3 = elemsN[:, [0, 3, 2]]
    faces4 = elemsN[:, [1, 2, 3]]

    # Faces as sets of their nodes (vertices)
    vertices = np.zeros([nElems*4, 3])
    vertices[0::4] = faces1
    vertices[1::4] = faces2
    vertices[2::4] = faces3
    vertices[3::4] = faces4

    [facesN, elemsF] = deleteDuplicateRows(vertices)

    numFacesElement = 4
    elemsF = np.array(np.reshape(elemsF, (nElems, numFacesElement)),
                      dtype=np.int)
    facesN = np.array(facesN, dtype=np.int)

    return elemsF, facesN


def computeBoundaryFaces(elemsF, facesN):
    ''' Compute boundary faces of a tetrahedral mesh.

    :param ndarray elemsF: elements-face connectivity.
    :param ndarray facesN: faces-nodes connectivity.
    :return: boundary-faces connectivity.
    :rtype: ndarray
    '''

    # Sort indexes and add 1 position in order to use indexes as Matlab
    A0 = np.sort(elemsF[:, 0]) + 1
    I0 = np.argsort(elemsF[:, 0]) + 1
    A1 = np.sort(elemsF[:, 1]) + 1
    I1 = np.argsort(elemsF[:, 1]) + 1
    A2 = np.sort(elemsF[:, 2]) + 1
    I2 = np.argsort(elemsF[:, 2]) + 1
    A3 = np.sort(elemsF[:, 3]) + 1
    I3 = np.argsort(elemsF[:, 3]) + 1

    # Number of faces
    nFaces = elemsF.max()

    # As consequence, dimensions of E must be increased
    # 2 rows and 1 column
    E = np.zeros((nFaces+2, 9))

    E[A0, 1] = I0
    E[A1, 2] = I1
    E[A2, 3] = I2
    E[A3, 4] = I3

    # If the same face is listed in the same row of 'elemsF'
    # more than, once it will simply be missed! Because of this we
    # have to insert the following dummy variables in order to
    # determine the boundary faces.
    tmp = np.diff(A0) == 0
    ind0 = np.where(tmp)[False]
    tmp = np.diff(A1) == 0
    ind1 = np.where(tmp)[False]
    tmp = np.diff(A2) == 0
    ind2 = np.where(tmp)[False]
    tmp = np.diff(A3) == 0
    ind3 = np.where(tmp)[False]

    E[A0[ind0], 5] = 1
    E[A1[ind1], 6] = 1
    E[A2[ind2], 7] = 1
    E[A3[ind3], 8] = 1

    # Delete extra rows and column
    E = np.delete(E, (0), axis=0)
    E = np.delete(E, (0), axis=1)

    # Final sorting
    E.sort()
    E = np.fliplr(E)

    #  Get boundary nodes by first examining which columns in E
    # have only one nonzero element, meaning that this face is
    # related to only one single tetra, which means it is on the
    # boundary of the domain. Since faces are defined by their nodes,
    # we have the boundary nodes too.
    tmp = (E[:, 1] == 0)
    ind = np.where(tmp)[False]

    bfacesN = np.array(np.transpose(facesN[ind, :]), dtype=np.int)

    return bfacesN


def computeBoundaryDofs(edgesN, bfacesN):
    ''' Compute boundary dofs of a tetrahedral mesh.

    :param ndarray edgesN: edges-nodes connectivity.
    :param ndarray bfacesN: boundary-faces-nodes connectivity.
    :return: boundary-edges connectivity.
    :rtype: ndarray
    '''

    # Extracts sets of edges-nodes (add 1 to indexes - Matlab indexing)
    edges1 = (bfacesN[[0, 1], :] + 1).transpose()
    edges2 = (bfacesN[[1, 2], :] + 1).transpose()
    edges3 = (bfacesN[[2, 0], :] + 1).transpose()

    # Number of boundary-faces
    dim = bfacesN.shape
    nBoundaryFaces = dim[1]

    # Boudary faces as sets of their edges (vertices)
    vertices = np.zeros([nBoundaryFaces*3, 2])
    vertices[0::3] = edges1
    vertices[1::3] = edges2
    vertices[2::3] = edges3

    # Repeated setts of nodes (joint edges) are eliminated
    [temp, _] = deleteDuplicateRows(vertices)

    matrixs = np.concatenate((edgesN + 1, temp), axis=0)

    matrixs.sort(axis=1)

    tags = np.lexsort((matrixs[:, 1], matrixs[:, 0]))
    matrixs = matrixs[tags]

    ind0 = np.diff(matrixs[:, 0]) == 0
    ind1 = np.diff(matrixs[:, 1]) == 0

    # Concatenate vectors (vertical stack)
    ind = np.vstack((ind0, ind1))
    ind = ind.transpose()

    # Which ones were reps? k is a vector of indexes to matrix
    k = np.array(np.all(ind, axis=1).ravel().nonzero())

    # tags(k) is an index vector to edgesN (matrix) and denotes those edges
    # which are on boundary tags(k+1) is an index vector to matrix and
    # matrix(tags(k+a)) is the same as bedges, but in different order.
    # I could just return tags(k), but we want that the order is the same
    # as in bEdgesN
    tags2 = np.array(np.argsort(tags[k+1]))

    bEdges = np.array(tags[k[0][tags2]], dtype=np.int)

    return bEdges


def defineEFEMConstants():
    ''' Set constants for edge finite element computations, namely,
    order linear edge finite elements (edgeOrder), order linear nodal finite
    elements (nodalOrder) and number of dimensions (numDimensions).

    :param: None.
    :return: edgeOrder, nodalOrder and numDimensions.
    :rtype: integer.
    '''
    # Set order of linear edge finite elements
    edgeOrder = 6
    # Set order of linear nodal finite lements
    nodalOrder = 4
    # Set number of dimensions
    numDimensions = 3

    return edgeOrder, nodalOrder, numDimensions


def nedelecBasisIterative(eleNodes, points, eleVol, lengthEdges, edgeOrder):
    ''' Compute the basis Nedelec functions in an iterative way for a
    set of points in a given element.

    :param ndarray eleNodes: nodal spatial coordinates of the element
    :param ndarray points: spatial coordinates of the evaluation points
    :param float eleVol: element's volume
    :param ndarray lengthEdges: element's edges defined by their length
    :param int edgeOrder: order of tetrahedral edge element
    :return: values of Nedelec functions.
    :rtype: ndarray.

    .. note: References:\n
       Jin, Jian-Ming. The finite element method in electromagnetics.
       John Wiley & Sons, 2002.
    '''
    # Coefficients computation. Running in a cycling way
    a = np.zeros([4], dtype=np.float64)
    b = np.zeros([4], dtype=np.float64)
    c = np.zeros([4], dtype=np.float64)
    d = np.zeros([4], dtype=np.float64)

    tmp = np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int)
    temp_ones = np.ones([3], dtype=np.float64)

    for iCoeff in np.arange(4):
        a[iCoeff] = det([[eleNodes[tmp[iCoeff+1], 0],
                          eleNodes[tmp[iCoeff+2], 0],
                          eleNodes[tmp[iCoeff+3], 0]],
                         [eleNodes[tmp[iCoeff+1], 1],
                          eleNodes[tmp[iCoeff+2], 1],
                          eleNodes[tmp[iCoeff+3], 1]],
                         [eleNodes[tmp[iCoeff+1], 2],
                          eleNodes[tmp[iCoeff+2], 2],
                          eleNodes[tmp[iCoeff+3], 2]]])
        b[iCoeff] = det([temp_ones,
                         [eleNodes[tmp[iCoeff+1], 1],
                          eleNodes[tmp[iCoeff+2], 1],
                          eleNodes[tmp[iCoeff+3], 1]],
                         [eleNodes[tmp[iCoeff+1], 2],
                          eleNodes[tmp[iCoeff+2], 2],
                          eleNodes[tmp[iCoeff+3], 2]]])
        c[iCoeff] = det([temp_ones,
                         [eleNodes[tmp[iCoeff+1], 0],
                          eleNodes[tmp[iCoeff+2], 0],
                          eleNodes[tmp[iCoeff+3], 0]],
                         [eleNodes[tmp[iCoeff+1], 2],
                          eleNodes[tmp[iCoeff+2], 2],
                          eleNodes[tmp[iCoeff+3], 2]]])
        d[iCoeff] = det([temp_ones,
                         [eleNodes[tmp[iCoeff+1], 0],
                          eleNodes[tmp[iCoeff+2], 0],
                          eleNodes[tmp[iCoeff+3], 0]],
                         [eleNodes[tmp[iCoeff+1], 1],
                          eleNodes[tmp[iCoeff+2], 1],
                          eleNodes[tmp[iCoeff+3], 1]]])

    # Add signs
    sign = np.float64(-1.0)
    a[1] = a[1] * sign
    a[3] = a[3] * sign
    b[0] = b[0] * sign
    b[2] = b[2] * sign
    c[1] = c[1] * sign
    c[3] = c[3] * sign
    d[0] = d[0] * sign
    d[2] = d[2] * sign

    # Number of points
    if points.ndim == 1:
        nPoints = 1
    else:
        nPoints = points.shape[0]

    # Nedelec basis for all points
    if nPoints == 1:
        AA = np.float64(1.0) / ((np.float64(6.0)*eleVol)**2)
        # To reduce number of multiplications
        b1x = b[0]*points[0]
        b2x = b[1]*points[0]
        b3x = b[2]*points[0]
        b4x = b[3]*points[0]
        c1y = c[0]*points[1]
        c2y = c[1]*points[1]
        c3y = c[2]*points[1]
        c4y = c[3]*points[1]
        d1z = d[0]*points[2]
        d2z = d[1]*points[2]
        d3z = d[2]*points[2]
        d4z = d[3]*points[2]
        A1 = a[0] + b1x + c1y + d1z
        A2 = a[1] + b2x + c2y + d2z
        A3 = a[2] + b3x + c3y + d3z
        A4 = a[3] + b4x + c4y + d4z
        # Basis 1
        b1 = np.multiply([(b[1]*A1)-(b[0]*A2),
                          (c[1]*A1)-(c[0]*A2),
                          (d[1]*A1)-(d[0]*A2)], lengthEdges[0])
        # Basis 2
        b2 = np.multiply([b[2]*A1-b[0]*A3,
                          c[2]*A1-c[0]*A3,
                          d[2]*A1-d[0]*A3], lengthEdges[1])
        # Basis 3
        b3 = np.multiply([b[3]*A1-b[0]*A4,
                          c[3]*A1-c[0]*A4,
                          d[3]*A1-d[0]*A4], lengthEdges[2])
        # Basis 4
        b4 = np.multiply([b[2]*A2-b[1]*A3,
                          c[2]*A2-c[1]*A3,
                          d[2]*A2-d[1]*A3], lengthEdges[3])
        # Basis 5
        b5 = np.multiply([b[1]*A4-b[3]*A2,
                          c[1]*A4-c[3]*A2,
                          d[1]*A4-d[3]*A2], lengthEdges[4])
        # Basis 6
        b6 = np.multiply([b[3]*A3-b[2]*A4,
                          c[3]*A3-c[2]*A4,
                          d[3]*A3-d[2]*A4], lengthEdges[5])

        basis = np.array(np.vstack((b1, b2, b3, b4, b5, b6)) * AA,
                         dtype=np.float64)
    # If not
    else:
        basis = np.zeros((edgeOrder, 3, nPoints), dtype=np.float64)
        AA = np.float64(1.0) / ((np.float64(6.0)*eleVol)**2)
        # Compute basis for each point
        for iP in np.arange(nPoints):
            # To reduce number of multiplications
            b1x = b[0]*points[iP, 0]
            b2x = b[1]*points[iP, 0]
            b3x = b[2]*points[iP, 0]
            b4x = b[3]*points[iP, 0]
            c1y = c[0]*points[iP, 1]
            c2y = c[1]*points[iP, 1]
            c3y = c[2]*points[iP, 1]
            c4y = c[3]*points[iP, 1]
            d1z = d[0]*points[iP, 2]
            d2z = d[1]*points[iP, 2]
            d3z = d[2]*points[iP, 2]
            d4z = d[3]*points[iP, 2]
            A1 = a[0] + b1x + c1y + d1z
            A2 = a[1] + b2x + c2y + d2z
            A3 = a[2] + b3x + c3y + d3z
            A4 = a[3] + b4x + c4y + d4z
            # Basis 1
            b1 = np.multiply([(b[1]*A1)-(b[0]*A2),
                              (c[1]*A1)-(c[0]*A2),
                              (d[1]*A1)-(d[0]*A2)], lengthEdges[0])
            # Basis 2
            b2 = np.multiply([b[2]*A1-b[0]*A3,
                              c[2]*A1-c[0]*A3,
                              d[2]*A1-d[0]*A3], lengthEdges[1])
            # Basis 3
            b3 = np.multiply([b[3]*A1-b[0]*A4,
                              c[3]*A1-c[0]*A4,
                              d[3]*A1-d[0]*A4], lengthEdges[2])
            # Basis 4
            b4 = np.multiply([b[2]*A2-b[1]*A3,
                              c[2]*A2-c[1]*A3,
                              d[2]*A2-d[1]*A3], lengthEdges[3])
            # Basis 5
            b5 = np.multiply([b[1]*A4-b[3]*A2,
                              c[1]*A4-c[3]*A2,
                              d[1]*A4-d[3]*A2], lengthEdges[4])
            # Basis 6
            b6 = np.multiply([b[3]*A3-b[2]*A4,
                              c[3]*A3-c[2]*A4,
                              d[3]*A3-d[2]*A4], lengthEdges[5])

            basis[:, :, iP] = np.vstack((b1, b2, b3, b4, b5, b6)) * AA

    return basis


def unitary_test():
    ''' Unitary test for efem.py script.
    '''

if __name__ == '__main__':
    # Standard module import
    unitary_test()
else:
    # Standard module import
    import numpy as np
    from scipy.linalg import det
    # PETGEM module import
    from petgem.efem.vectorMatrixFunctions import deleteDuplicateRows
