#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define the classes, methods and functions for Edge Finite
Element Method (EFEM) of lowest order in tetrahedral meshes, namely,
Nedelec elements.
'''


def computeEdges(elemsN, nElems, nedelec_order):
    ''' Compute edges of a 3D tetrahedral mesh. For edge finite element of linear
    order the edges are the dofs. For edge finite element of second order
    the dofs are computed in runtime based on edges and faces on each
    tetrahedral element.

    :param ndarray elemsN: elements-nodes connectivity.
    :param int nElems: number of tetrahedral elements in the mesh.
    :param int nedelec_order: nedelec element order.
    :return: edges connectivity and edgesNodes connectivity.
    :rtype: ndarray
    '''

    # First order edge element
    if nedelec_order == 1:
        # Extracts sets of edges
        edges1 = elemsN[:, [0, 1]]
        edges2 = elemsN[:, [0, 2]]
        edges3 = elemsN[:, [0, 3]]
        edges4 = elemsN[:, [1, 2]]
        edges5 = elemsN[:, [3, 1]]
        edges6 = elemsN[:, [2, 3]]
    # Second and third order edge element
    elif nedelec_order == 2 or nedelec_order == 3:
        # Extracts sets of edges
        edges1 = elemsN[:, [0, 1]]
        edges2 = elemsN[:, [1, 2]]
        edges3 = elemsN[:, [2, 0]]
        edges4 = elemsN[:, [0, 3]]
        edges5 = elemsN[:, [1, 3]]
        edges6 = elemsN[:, [2, 3]]
    else:
        raise ValueError('Edge element order=',
                         nedelec_order, ' not supported.')

    # Edges as sets of their nodes (vertices)
    vertices = np.zeros([nElems*6, 2])
    vertices[0::6] = edges1
    vertices[1::6] = edges2
    vertices[2::6] = edges3
    vertices[3::6] = edges4
    vertices[4::6] = edges5
    vertices[5::6] = edges6

    # Delete duplicate rows
    [edgesNodes, edges] = deleteDuplicateRows(vertices)

    # Build dofs matrix
    edges = np.array(np.reshape(edges, (nElems, 6)), dtype=np.int)

    # Build dofs to nodes connectivity
    edgesNodes.sort(axis=1)
    edgesNodes = np.array(edgesNodes, dtype=np.int)

    return edges, edgesNodes


def computeFaces(elemsN, nElems, nedelec_order):
    ''' Compute the element\'s faces of a 3D tetrahedral mesh.

    :param ndarray matrix: elements-nodes connectivity.
    :param int nElems: number of elements in the mesh.
    :param int nedelec_order: nedelec element order.
    :return: element/faces connectivity.
    :rtype: ndarray

    .. note:: References:\n
       Rognes, Marie E., Robert Cndarray. Kirby, and Anders Logg. "Efficient
       assembly of H(div) and H(curl) conforming finite elements."
       SIAM Journal on Scientific Computing 31.6 (2009): 4130-4151.
    '''

    # Extracts sets of faces for each nedelec element order
    if nedelec_order == 1:  # First order edge element
        faces1 = elemsN[:, [0, 2, 1]]
        faces2 = elemsN[:, [0, 1, 3]]
        faces3 = elemsN[:, [0, 3, 2]]
        faces4 = elemsN[:, [1, 2, 3]]
    elif nedelec_order == 2 or nedelec_order == 3:  # Second and third order
                                                    # edge element
        faces1 = elemsN[:, [0, 1, 2]]
        faces2 = elemsN[:, [0, 2, 3]]
        faces3 = elemsN[:, [0, 3, 1]]
        faces4 = elemsN[:, [3, 1, 2]]
    else:
        raise ValueError('Edge element order=',
                         nedelec_order, ' not supported.')

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
    :return: nodal-connectivity and indexes of boundary-faces.
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
    # Get boundary faces to nodes
    ind = (E[:, 1] == 0)
    bfacesN = np.array(np.transpose(facesN[ind, :]), dtype=np.int)

    # Get indexes of boundary faces
    ind = np.where(ind == True)
    bFaces = np.array(np.transpose(ind), dtype=np.int)
    size = bFaces.shape
    nBoundaryFaces = size[0]
    bFaces = bFaces.reshape((nBoundaryFaces))

    return bfacesN, bFaces


def computeBoundaryEdges(edgesN, bfacesN):
    ''' Compute boundary edges of a tetrahedral mesh.

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


def computeBoundaries(elemsN, nElems, edgesNodes, nedelec_order):
    ''' Compute boundaries of the domain for edge finite element computations.

    :param ndarray elemsN: elements-nodes connectivity.
    :param int nElems: number of elements in the mesh.
    :param ndarray edgesN: edges-nodes connectivity.
    :param int nedelec_order: nedelec element order.
    :return: boundaries, ndofs
    :rtype: ndarray
    '''

    # Compute faces
    elemsF, facesN = computeFaces(elemsN, nElems, nedelec_order)

    # Compute boundary faces
    boundaryFacesN, boundaryFaces = computeBoundaryFaces(elemsF, facesN)

    # Compute boundary edges
    boundaryEdges = computeBoundaryEdges(edgesNodes, boundaryFacesN)

    # Number of edges
    size = edgesNodes.shape
    nEdges = size[0]

    # Number of faces
    size = facesN.shape
    nFaces = size[0]

    # Number of boundary edges
    size = boundaryEdges.shape
    nBoundaryEdges = size[1]

    # Number of boundary faces
    size = boundaryFaces.shape
    nBoundaryFaces = size[0]

    # Compute boundaries
    if nedelec_order == 1:      # First order edge element
        # Boundaries correspond to boundary edges
        boundaries = boundaryEdges
        # Number of DOFs correspond to edges in the mesh
        size = edgesNodes.shape
        ndofs = size[0]

    elif nedelec_order == 2:    # Second order edge element
        # Number of DOFS
        ndofs = nEdges*2 + nFaces*2

        # Total number of boundaries
        nBoundaries = nBoundaryEdges*2 + nBoundaryFaces*2

        # Allocate
        boundaries = np.zeros((nBoundaries), dtype=np.int)
        newEdgesNumb = np.zeros((nBoundaryEdges, 2), dtype=np.int)
        newFacesNumb = np.zeros((nBoundaryFaces, 2), dtype=np.int)

        # Compute boundaries on edges
        # Use 1-based indexing in boundaryEdges and boundaryFaces
        boundaryEdges += np.int(1)
        boundaryFaces += np.int(1)

        # Get boundaries on edges
        newEdgesNumb[:, 0] = (boundaryEdges*2)-1
        newEdgesNumb[:, 1] = boundaryEdges*2
        newEdgesNumb = newEdgesNumb.reshape((nBoundaryEdges*2), order='F')

        # Insert edge boundary indexes in boundary array
        boundaries[0:nBoundaryEdges*2] = newEdgesNumb

        # Get boundaries on faces
        newFacesNumb[:, 0] = (boundaryFaces*2+nEdges*2)-1
        newFacesNumb[:, 1] = boundaryFaces*2+nEdges*2
        newFacesNumb = newFacesNumb.reshape((nBoundaryFaces*2), order='F')

        # Insert face boundary indexes in boundary array
        boundaries[nBoundaryEdges*2:] = newFacesNumb

        # Use 0-based indexing in boundaries
        boundaries -= np.int(1)

    elif nedelec_order == 3:    # Third order edge element
        # Number of DOFS
        ndofs = nEdges*3 + nFaces*6 + nElems*3

        # Total number of boundaries
        nBoundaries = nBoundaryEdges*3 + nBoundaryFaces*6

        # Allocate
        boundaries = np.zeros((nBoundaries), dtype=np.int)
        newEdgesNumb = np.zeros((nBoundaryEdges, 3), dtype=np.int)
        newFacesNumb = np.zeros((nBoundaryFaces, 6), dtype=np.int)

        # Compute boundaries on edges
        # Use 1-based indexing in boundaryEdges and boundaryFaces
        boundaryEdges += np.int(1)
        boundaryFaces += np.int(1)

        # Get boundaries on edges
        newEdgesNumb[:, 0] = (boundaryEdges*3)-2
        newEdgesNumb[:, 1] = (boundaryEdges*3)-1
        newEdgesNumb[:, 2] = boundaryEdges*3
        newEdgesNumb = newEdgesNumb.reshape((nBoundaryEdges*3), order='F')

        # Insert edge boundary indexes in boundary array
        boundaries[0:nBoundaryEdges*3] = newEdgesNumb

        # Get boundaries on faces
        newFacesNumb[:, 0] = (boundaryFaces*6+nEdges*3)-5
        newFacesNumb[:, 1] = (boundaryFaces*6+nEdges*3)-4
        newFacesNumb[:, 2] = (boundaryFaces*6+nEdges*3)-3
        newFacesNumb[:, 3] = (boundaryFaces*6+nEdges*3)-2
        newFacesNumb[:, 4] = (boundaryFaces*6+nEdges*3)-1
        newFacesNumb[:, 5] = boundaryFaces*6+nEdges*3
        newFacesNumb = newFacesNumb.reshape((nBoundaryFaces*6), order='F')

        # Insert face boundary indexes in boundary array
        boundaries[nBoundaryEdges*3:] = newFacesNumb

        # Use 0-based indexing in boundaries
        boundaries -= np.int(1)

    else:
        raise ValueError('Edge element order=',
                         nedelec_order, ' not supported.')

    return boundaries, ndofs


def defineEFEMConstants(nedelec_order):
    ''' Set constants for edge finite element computations, namely,
    order of edge finite elements (edgeOrder), order of nodal finite
    elements (nodalOrder) and number of dimensions (numDimensions).

    :param int nedelec_order: nedelec element order.
    :return: edgeOrder, nodalOrder and numDimensions.
    :rtype: integer.
    '''
    # Set order of edge finite elements
    if nedelec_order == 1:      # First order edge element
        # Nedelec order
        edgeOrder = 6
    elif nedelec_order == 2:     # Second order edge element
        # Nedelec order
        edgeOrder = 20
    elif nedelec_order == 3:    # Third order edge element
        # Nedelec order
        edgeOrder = 45
    else:
        raise ValueError('Edge element order=',
                         nedelec_order, ' not supported.')

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
    a = np.zeros([4], dtype=np.float)
    b = np.zeros([4], dtype=np.float)
    c = np.zeros([4], dtype=np.float)
    d = np.zeros([4], dtype=np.float)

    tmp = np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int)
    temp_ones = np.ones([3], dtype=np.float)

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
    sign = np.float(-1.0)
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
        AA = np.float(1.0) / ((np.float(6.0)*eleVol)**2)
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
                         dtype=np.float)
    # If not
    else:
        basis = np.zeros((edgeOrder, 3, nPoints), dtype=np.float)
        AA = np.float(1.0) / ((np.float(6.0)*eleVol)**2)
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


def computeElementDOFs(iEle, nodesEle, edgesEle, facesEle, nodesFace,
                       nEdges, nFaces, nedelec_order):
    ''' Compute global DOFs for tetrahedral element.

    :param int iEle: id of the tetrahedral element-th.
    :param ndarray nodesEle: array with nodal indexes of element.
    :param ndarray edgesEle: array with edge indexes of element.
    :param ndarray facesEle: array with face indexes of element.
    :param ndarray nodesFace: array with node indexes of faces.
    :param ndarray nEdges: total number of edges in the mesh.
    :param ndarray nFaces: total number of faces in the mesh.
    :param int nedelec_order: nedelec element order.
    :return: DOFs indexes.
    :rtype: ndarray
    '''
    if nedelec_order == 1:      # First order edge element
        # DOFs correspond to element's edges in the mesh (6 DOFs)
        dofsEle = edgesEle

    elif nedelec_order == 2:    # Second order edge element
        # First order ele
        firstOrderEdgeElement = 6

        # Second order ele
        secondOrderEdgeElement = 20

        # Number of faces per tetrahedral element
        nFacesEle = 4

        # Edge's signs computation
        idx_signs1 = np.array([1, 2, 0, 3, 3, 3], dtype=np.int)
        idx_signs2 = np.array([0, 1, 2, 0, 1, 2], dtype=np.int)
        tmp = nodesEle
        tmp = tmp[idx_signs1] - tmp[idx_signs2]
        signsEle = tmp / np.abs(tmp)

        # Allocate
        dofsEle = np.zeros((secondOrderEdgeElement), dtype=np.int)
        newEdgesNumb = np.zeros((firstOrderEdgeElement, 2), dtype=np.int)
        newFacesNumb = np.zeros((nFacesEle, 2), dtype=np.int)

        # Use 1-based indexing in edgesEle and facesEle
        edgesEle += np.int(1)
        facesEle += np.int(1)

        # Get dofs on edges
        newEdgesNumb[:, 0] = (edgesEle*2)-1
        newEdgesNumb[:, 1] = edgesEle*2

        # Reverse dofs in case they are negative
        for iEdge in np.arange(firstOrderEdgeElement):
            if (signsEle[iEdge] < 0):
                newEdgesNumb[iEdge, [0, 1]] = newEdgesNumb[iEdge, [1, 0]]

        newEdgesNumb = newEdgesNumb.reshape((firstOrderEdgeElement * 2))

        # Get dofs on faces
        newFacesNumb[:, 0] = (facesEle*2+nEdges*2)-1
        newFacesNumb[:, 1] = facesEle*2+nEdges*2
        newFacesNumb = newFacesNumb.reshape((nFacesEle*2))

        # Insert dofs on edges in dofsEle array
        dofsEle[0:firstOrderEdgeElement*2] = newEdgesNumb

        # Insert dofs on faces in dofsEle array
        dofsEle[firstOrderEdgeElement*2:] = newFacesNumb

        # Use 0-based indexing in boundaries
        dofsEle -= np.int(1)

    elif nedelec_order == 3:    # Third order edge element
        # First order ele
        firstOrderEdgeElement = 6

        # Third order ele
        thirdOrderEdgeElement = 45

        # Number of faces per tetrahedral element
        nFacesEle = 4

        # Edge's signs computation
        idx_signs1 = np.array([1, 2, 0, 3, 3, 3], dtype=np.int)
        idx_signs2 = np.array([0, 1, 2, 0, 1, 2], dtype=np.int)
        tmp = nodesEle
        tmp = tmp[idx_signs1] - tmp[idx_signs2]
        signsEle = tmp / np.abs(tmp)

        # Allocate
        dofsEle = np.zeros((thirdOrderEdgeElement), dtype=np.int)
        newEdgesNumb = np.zeros((firstOrderEdgeElement, 3), dtype=np.int)
        newFacesNumb = np.zeros((nFacesEle, 6), dtype=np.int)
        newVolumeNumb = np.zeros([3], dtype=np.int)

        # Use 1-based indexing in edgesEle and facesEle
        edgesEle += np.int(1)
        facesEle += np.int(1)
        iEle += np.int(1)

        # Get dofs on edges
        newEdgesNumb[:, 0] = (edgesEle*3)-2
        newEdgesNumb[:, 1] = (edgesEle*3)-1
        newEdgesNumb[:, 2] = edgesEle*3

        # Reverse dofs in case they are negative
        for iEdge in np.arange(firstOrderEdgeElement):
            if (signsEle[iEdge] < 0):
                newEdgesNumb[iEdge, [0, 1, 2]] = newEdgesNumb[iEdge, [2, 1, 0]]

        newEdgesNumb = newEdgesNumb.reshape((firstOrderEdgeElement*3))

        # Get dofs on faces
        # Definition of faces (global ordering)
        global_faces = np.array([[0, 1, 2], [0, 2, 3],
                                 [0, 3, 1], [3, 1, 2]], dtype=np.int)

        tmp_dofs = np.zeros((3, 2), dtype=np.int)

        for iFace in np.arange(nFacesEle):
            tmp_dofs[0, :] = [(facesEle[iFace]*6+nEdges*3)-5,
                              (facesEle[iFace]*6+nEdges*3)-4]
            tmp_dofs[1, :] = [(facesEle[iFace]*6+nEdges*3)-3,
                              (facesEle[iFace]*6+nEdges*3)-2]
            tmp_dofs[2, :] = [(facesEle[iFace]*6+nEdges*3)-1,
                              (facesEle[iFace]*6+nEdges*3)]

            if not np.all(nodesFace[iFace, :] ==
                          nodesEle[global_faces[iFace, :]]):
                a = nodesFace[iFace, :]
                b = nodesEle[global_faces[iFace, :]]
                idx = np.where(a[:, None] == b[None, :])[1]
                tmp_dofs_idx = tmp_dofs[idx, :]
                newFacesNumb[iFace, :] = tmp_dofs_idx.reshape(1, 6)
            else:
                newFacesNumb[iFace, :] = tmp_dofs.reshape(1, 6)

        newFacesNumb = newFacesNumb.reshape((nFacesEle*6))

        # Get dofs on element volume
        newVolumeNumb[0] = (((nEdges*3) + (nFaces*6)) +
                            (iEle-1) * 3 + np.int(1))
        newVolumeNumb[1] = (((nEdges*3) + (nFaces*6)) +
                            (iEle-1)*3 + np.int(2))
        newVolumeNumb[2] = (((nEdges*3) + (nFaces*6)) +
                            (iEle-1) * 3 + np.int(3))

        # Insert dofs on edges in dofsEle array
        dofsEle[0:firstOrderEdgeElement*3] = newEdgesNumb

        # Insert dofs on faces in dofsEle array
        dofsEle[firstOrderEdgeElement*3:
                firstOrderEdgeElement*3+nFacesEle*6] = newFacesNumb

        # Insert dofs on volume in dofsEle array
        dofsEle[firstOrderEdgeElement*3+nFacesEle*6:] = newVolumeNumb

        # Use 0-based indexing in boundaries
        dofsEle -= np.int(1)
        iEle -= np.int(1)
        edgesEle -= np.int(1)
        facesEle -= np.int(1)

    else:
        raise ValueError('Edge element order=',
                         nedelec_order, ' not supported.')

    return dofsEle


def edgeFaceVerticesInit():
    ''' Initialization of arrays that define edges and faces of tetrahedral
    elements. Vector basis functions are based on these arrays.

    :param: None.
    :return: edge_vertices and face_vertices.
    :rtype: ndarray

    .. note: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    '''

    # Edges definition
    edge_vertices = np.array([[0, 1],
                              [1, 2],
                              [2, 0],
                              [0, 3],
                              [1, 3],
                              [2, 3]], dtype=np.int)

    # Faces definition
    face_vertices = np.array([[0, 1, 2],
                              [1, 3, 2],
                              [2, 3, 0],
                              [3, 1, 0]], dtype=np.int)

    # Reference element
    ref_ele = np.array([[0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.int)

    return edge_vertices, face_vertices, ref_ele


def isigno(xx, yy, zz):
    ''' Compute signs for edges in a given tetrahedral element.

    :param float xx: x-coordinates of the nodes.
    :param float yy: x-coordinates of the nodes.
    :param float zz: x-coordinates of the nodes.
    :return: signs of edges.
    :rtype: ndarray
    '''
    isign = -1

    if ((xx == 0) and (yy == 0) and (zz == 0)):
        print('Warning: Null sign in edge')
        isign = 0
    elif ((xx > 0) or ((xx == 0) and (yy > 0)) or ((xx == 0)
          and (yy == 0) and (zz > 0))):
        isign = 1

    return isign


def computeSignsJacobLegth(eleNodes, edge_vertices, face_vertices,
                           nedelec_order):
    ''' Compute signs, edges length and jacobian for mapping.

    :param ndarray nodesEle: array with nodal indexes of element.
    :param ndarray edge_vertices: edges connectivity based on vertices.
    :param ndarray face_vertices: .
    :param int nedelec_order: nedelec element order.
    :return: tangential unitary vectors for each edge, edges length,
             dofs signs, normal face vectors, jacobiano, determinant
             of jacobian, faces area.
    :rtype: ndarray

    .. note: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    '''
    # First order ele
    firstOrderEdgeElement = 6

    # Second order ele
    secondOrderEdgeElement = 20

    # Third order ele
    thirdOrderEdgeElement = 45

    if nedelec_order == 2:
        orderEle = secondOrderEdgeElement
    elif nedelec_order == 3:
        orderEle = thirdOrderEdgeElement
    else:
        raise ValueError('Edge element order=',
                         nedelec_order, ' not supported.')

    # Absolute values for edges and faces numbering
    edge_vertices_abs = np.abs(edge_vertices)
    face_vertices_abs = np.abs(face_vertices)

    # Jacobian computation
    jacob = np.zeros((3, 3), dtype=np.float)
    jacob[0, :] = eleNodes[:, 1] - eleNodes[:, 0]
    jacob[1, :] = eleNodes[:, 2] - eleNodes[:, 0]
    jacob[2, :] = eleNodes[:, 3] - eleNodes[:, 0]

    # Compute jacobian determinant
    detJacob = (jacob[0, 0]*jacob[1, 1]*jacob[2, 2] +
                jacob[0, 1]*jacob[1, 2]*jacob[2, 0] +
                jacob[0, 2]*jacob[1, 0]*jacob[2, 1] -
                jacob[0, 2]*jacob[1, 1]*jacob[2, 0] -
                jacob[0, 1]*jacob[1, 0]*jacob[2, 2] -
                jacob[0, 0]*jacob[1, 2]*jacob[2, 1])

    # Definition of tangetial unitary vectors for each edge
    taui = np.zeros((3, firstOrderEdgeElement), dtype=np.float)
    taui[0:3, :] = (eleNodes[:, edge_vertices_abs[:, 1]] -
                    eleNodes[:, edge_vertices_abs[:, 0]])

    # Computation of normal face vectors, signs and edge length computation
    nreal = np.zeros((3, 4), dtype=np.float)
    area_faces = np.zeros(4, dtype=np.float)
    length = np.zeros(firstOrderEdgeElement, dtype=np.float)
    signs = np.zeros(orderEle, dtype=np.float)

    if nedelec_order == 2:
        for iedge in np.arange(firstOrderEdgeElement):
            length[iedge] = norm(taui[:, iedge])
            taui[:, iedge] = taui[:, iedge]/length[iedge]

            if np.sum(np.sign(edge_vertices[iedge, :]+np.int(1))) == -2:
                taui[:, iedge] = -taui[:, iedge]
            elif np.sum(np.sign(edge_vertices[iedge, :]+np.int(1))) == 0:
                raise ValueError('The signs of ', iedge,
                                 ' are incorrect.')

            dof = np.linspace(2*iedge-1, 2*iedge, 2,
                              dtype=np.int) + np.int(1)

            signs[dof] = isigno(taui[0, iedge], taui[1, iedge],
                                taui[2, iedge])

        for facenumber in np.arange(4):
            aux1 = (eleNodes[:, face_vertices_abs[facenumber, 1]] -
                    eleNodes[:, face_vertices_abs[facenumber, 0]])
            aux2 = (eleNodes[:, face_vertices_abs[facenumber, 2]] -
                    eleNodes[:, face_vertices_abs[facenumber, 0]])

            nreal[:, facenumber] = -np.cross(aux1, aux2)

            area_faces[facenumber] = norm(nreal[:, facenumber])
            nreal[:, facenumber] = nreal[:, facenumber]/area_faces[facenumber]

            dof = np.linspace(2*facenumber-1, 2*facenumber, 2,
                              dtype=np.int) + np.int(13)

            signs[dof] = isigno(nreal[0, facenumber], nreal[1, facenumber],
                                nreal[2, facenumber])

    elif nedelec_order == 3:
        # Compute components for xx, yy, zz for the dofs associated
        # to edges (18 dofs, 3 per edge)
        # Allocate
        xx = np.zeros(18, dtype=np.float)
        yy = np.zeros(18, dtype=np.float)
        zz = np.zeros(18, dtype=np.float)

        xyz21 = eleNodes[:, 1] - eleNodes[:, 0]
        xyz31 = eleNodes[:, 2] - eleNodes[:, 0]
        xyz41 = eleNodes[:, 3] - eleNodes[:, 0]

        # Node 1,2,3
        xx[[0, 1, 2]] = xyz21[0]
        yy[[0, 1, 2]] = xyz21[1]
        zz[[0, 1, 2]] = xyz21[2]

        # Node 4,5,6
        xx[[3, 4, 5]] = eleNodes[0, 2]-eleNodes[0, 1]
        yy[[3, 4, 5]] = eleNodes[1, 2]-eleNodes[1, 1]
        zz[[3, 4, 5]] = eleNodes[2, 2]-eleNodes[2, 1]

        # Node 7,8,9
        xx[[6, 7, 8]] = -xyz31[0]
        yy[[6, 7, 8]] = -xyz31[1]
        zz[[6, 7, 8]] = -xyz31[2]

        # Node 10,11,12
        xx[[9, 10, 11]] = xyz41[0]
        yy[[9, 10, 11]] = xyz41[1]
        zz[[9, 10, 11]] = xyz41[2]

        # Node 13,14,15
        xx[[12, 13, 14]] = eleNodes[0, 3]-eleNodes[0, 1]
        yy[[12, 13, 14]] = eleNodes[1, 3]-eleNodes[1, 1]
        zz[[12, 13, 14]] = eleNodes[2, 3]-eleNodes[2, 1]

        # Node 16,17,18
        xx[[15, 16, 17]] = eleNodes[0, 3]-eleNodes[0, 2]
        yy[[15, 16, 17]] = eleNodes[1, 3]-eleNodes[1, 2]
        zz[[15, 16, 17]] = eleNodes[2, 3]-eleNodes[2, 2]

        nreal[:, 0] = np.cross(np.hstack((xx[0], yy[0], zz[0])),
                               np.hstack((xx[6], yy[6], zz[6])))
        nreal[:, 1] = np.cross(np.hstack((xx[6], yy[6], zz[6])),
                               np.hstack((xx[9], yy[9], zz[9])))
        nreal[:, 2] = np.cross(np.hstack((xx[0], yy[0], zz[0])),
                               np.hstack((xx[9], yy[9], zz[9])))
        nreal[:, 3] = np.cross(np.hstack((xx[3], yy[3], zz[3])),
                               np.hstack((xx[12], yy[12], zz[12])))

        for facenumber in np.arange(4):
            nreal[:, facenumber] = (nreal[:, facenumber] /
                                    norm(nreal[:, facenumber]))

        # Sign for dofs on edges
        for idof in np.arange(18):
            signs[idof] = isigno(xx[idof], yy[idof], zz[idof])

        # Sign for dofs on faces
        signs[18:24] = isigno(nreal[0, 0], nreal[1, 0], nreal[2, 0])
        signs[24:30] = isigno(nreal[0, 1], nreal[1, 1], nreal[2, 1])
        signs[30:36] = isigno(nreal[0, 2], nreal[1, 2], nreal[2, 2])
        signs[36:42] = isigno(nreal[0, 3], nreal[1, 3], nreal[2, 3])

        # Sign for dofs on volume
        signs[42:45] = 1

    else:
        raise ValueError('Edge element order=',
                         nedelec_order, ' not supported.')

    area_faces = area_faces/2.

    return taui, length, signs, nreal, jacob, detJacob, area_faces


def definitionHighOrderTet(nreal, signs, jacob, nedelec_order):
    ''' Compute "q" vectors on faces -dof definition of edge tetrahedral
    element of second and third order.

    :param float-array nreal: normal face vectors.
    :param float-array signs: dofs signs.
    :param float jacob: jacobian.
    :param int nedelec_order: nedelec element order.
    :return: q vectors on faces, inverse jacobian, 3x3 tensor.
    :rtype: ndarray.

    .. note: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    '''
    # q vectors general
    q_genera = np.eye(3, dtype=np.float)

    # Add sign to face
    n_faces = np.zeros((3, 4), dtype=np.float)
    if nedelec_order == 2:
        n_faces[:, 0] = nreal[:, 0]*signs[12]
        n_faces[:, 1] = nreal[:, 1]*signs[14]
        n_faces[:, 2] = nreal[:, 2]*signs[16]
        n_faces[:, 3] = nreal[:, 3]*signs[18]
    elif nedelec_order == 3:
        n_faces[:, 0] = nreal[:, 0]*signs[18]
        n_faces[:, 1] = nreal[:, 1]*signs[24]
        n_faces[:, 2] = nreal[:, 2]*signs[30]
        n_faces[:, 3] = nreal[:, 3]*signs[36]
    else:
        raise ValueError('Edge element order=',
                         nedelec_order, ' not supported.')

    qface = np.zeros((3, 8), dtype=np.float)

    for nface in np.arange(4):
        aux1 = crossprod(n_faces[:, nface], q_genera[:, 0])
        aux2 = crossprod(n_faces[:, nface], q_genera[:, 1])
        aux3 = crossprod(n_faces[:, nface], q_genera[:, 2])

        if ((norm(aux1) >= norm(aux2)) and (norm(aux1) >= norm(aux3))):
            qface[:, nface*2] = aux1[:, 0]
        elif ((norm(aux2) >= norm(aux1)) and (norm(aux2) >= norm(aux3))):
            qface[:, nface*2] = aux2[:, 0]
        elif ((norm(aux3) >= norm(aux1)) and (norm(aux3) >= norm(aux2))):
            qface[:, nface*2] = aux3[:, 0]

        qface[:, nface*2] = qface[:, nface*2]/norm(qface[:, nface*2])

        qface[:, (nface*2)+1] = crossprod(n_faces[:, nface],
                                          qface[:, nface*2])[:, 0]

        qface[:, (nface*2)+1] = (qface[:, (nface*2)+1] /
                                 norm(qface[:, (nface*2)+1]))

    # Computation of jacob^-1
    gr = np.eye(3, dtype=np.float)
    invjj = inv(jacob)
    GR = np.matmul(invjj.transpose(), np.matmul(gr, invjj))

    qface1_ref = np.matmul(jacob, qface[:, [0, 1]])
    qface2_ref = np.matmul(jacob, qface[:, [2, 3]])
    qface3_ref = np.matmul(jacob, qface[:, [4, 5]])
    qface4_ref = np.matmul(jacob, qface[:, [6, 7]])

    return qface1_ref, qface2_ref, qface3_ref, qface4_ref, invjj, GR


def aristaMappingTetrahedral(edge_vertices_old, edge_vertices_new):
    ''' Mapping edges of a real element defined by edge_vertices to a reference
    element por edge_vertices_ref. Furthermore, compute relations between
    "tau" tangential vectors to edges.

    :param int-array edge_vertices_old: edges definition based on
     vertices (old).
    :param int-array edge_vertices_new: edges definition based on
     vertices (new).
    :return: signs of unitary tangential vectors for each edge and
     edges mapping.

    .. note: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.

       Example:
       tau_sign = [-1 1 1 1 1 1]  sign of tau of (old) edge 1 is on opposite
                                  direction. Note that old edge 1 will be
                                  new edge 3 (see arista_mapping below)


       arista_mapping=[-3 2 1 4 5 6]  new edge 1 is old edge 3 The dof of
                                      edge are interchanged, i.e., the dof
                                      of the old edge 3 are placed
                                      interchanged in new edge 1.
    '''
    # Number of edges
    size = edge_vertices_old.shape
    numaristas = size[0]
    size = edge_vertices_new.shape
    numaristas2 = size[0]

    # Use 1-based indexes on edge_vertices_new
    edge_vertices_new += np.int(1)

    # Allocate
    tau_sign = np.ones(numaristas, dtype=np.int)
    arista_mapping = np.zeros(numaristas, dtype=np.int)

    edge_vertices_old_abs = np.abs(edge_vertices_old)
    edge_vertices_new_abs = np.abs(edge_vertices_new)

    # Previous validations
    # Dimensional coherence
    if numaristas != numaristas2:
        raise ValueError('Dimensions of edge_vertices_old and ' +
                         'edge_vertices_new are incoherence')

    # Signs coherence
    if not (np.array_equal(np.sign(edge_vertices_old[:, 0]),
                           np.sign(edge_vertices_old[:, 1]))):
        raise ValueError('Signs of edge_vertices_old are incoherence')

    if not (np.array_equal(np.sign(edge_vertices_new[:, 0]),
                           np.sign(edge_vertices_new[:, 1]))):
        raise ValueError('Signs of edge_vertices_new are incoherence')

    for new_iedge in np.arange(numaristas):
        new = edge_vertices_new[new_iedge, :]
        new_abs = edge_vertices_new_abs[new_iedge, :]
        new_sign = np.sign(new[0])

        for old_iedge in np.arange(numaristas):
            old = edge_vertices_old[old_iedge, :]
            old_abs = edge_vertices_old_abs[old_iedge, :]
            old_sign = np.sign(old[0])

            # Check vertices coincidence (regardless of order or sign)
            if (np.array_equal(new_abs, old_abs) or
               np.array_equal(new_abs, np.flip(old_abs, axis=0))):
                if (new_sign != old_sign):
                    tau_sign[old_iedge] = -tau_sign[old_iedge]

                if (np.array_equal(new_abs, old_abs)):
                    # Coincidence in order too
                    arista_mapping[new_iedge] = old_iedge
                else:
                    # vertices ordering changed => sign '-' in arista_mapping
                    #                           => change sign in tau of
                    #                              old edge
                    arista_mapping[new_iedge] = -old_iedge
                    tau_sign[old_iedge] = -tau_sign[old_iedge]

    # Subsequent validations
    # In arista_mapping must be all edge numbers and no repetitions
    for iedge in np.arange(numaristas):
        if np.where(np.abs(arista_mapping) == iedge)[0].size == 0:
            raise ValueError('arista_mapping does not contain arista ' +
                             iedge)

    if is_duplicate_entry(np.abs(arista_mapping)) != 0:
        raise ValueError('arista_mapping has duplicated values')

    return tau_sign, arista_mapping


def faceMappingTetrahedral(face_vertices_old, face_vertices_new, stage):
    ''' Compute mapping between faces of a tetrahedron with face_vertices to a
    tetrahedron with faces_vertices_ref.

    :param int-array face_vertices_old: faces definition based on
     vertices (old).
    :param int-array face_vertices_new: faces definition based on
     vertices (new).
    :param int face: number of stage of mapping (1 or 2).
    :return: signs of unitary tangential vectors for each face and
     faces mapping.

    .. note: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    '''
    # Number of faces
    size = face_vertices_old.shape
    numfaces = size[0]
    size = face_vertices_new.shape
    numfaces2 = size[0]

    # Use 1-based indexes on face_vertices_old or face_vertices_new
    if stage == 1:  # Mapping of first stage
        face_vertices_old += np.int(1)
    elif stage == 2:    # Mapping of second stage
        pass
    else:
        raise ValueError('Mapping stage = ', stage, ' not supported.')

    # Allocate
    face_mapping = np.zeros(numfaces, dtype=np.int)
    vertex_shift = np.zeros(numfaces, dtype=np.int)

    face_vertices_old_abs = np.abs(face_vertices_old)
    face_vertices_new_abs = np.abs(face_vertices_new)

    # Previous validations
    # Dimensional coherence
    if numfaces != numfaces2:
        raise ValueError('Dimensions of face_vertices_old and ' +
                         'face_vertices_new are incoherence')

    # Mapping
    nvertices_face = 3

    for new_facenumber in np.arange(numfaces):
        new = face_vertices_new[new_facenumber, :]
        new_abs = face_vertices_new_abs[new_facenumber, :]

        for old_facenumber in np.arange(numfaces):
            old = face_vertices_old[old_facenumber, :]
            old_abs = face_vertices_old_abs[old_facenumber, :]

            old_abs_reversed = np.roll(np.flip(old_abs, axis=0), 1)

            # Ciclic shift until match
            for cont_vertex_shift in np.arange(nvertices_face):
                if (np.array_equal(new_abs, old_abs)):
                    face_mapping[new_facenumber] = old_facenumber
                    vertex_shift[new_facenumber] = cont_vertex_shift
                    break
                if (np.array_equal(new_abs, old_abs_reversed)):
                    face_mapping[new_facenumber] = -old_facenumber
                    vertex_shift[new_facenumber] = cont_vertex_shift
                    break

                old_abs = np.roll(old_abs, 1)
                old_abs_reversed = np.roll(old_abs_reversed, -1)

    # Subsequent validations
    # In face_mapping must be all face numbers and no repetitions
    for facenumber in np.arange(numfaces):
        if np.where(np.abs(face_mapping) == facenumber)[0].size == 0:
            raise ValueError('face_mapping does not contain arista ' +
                             facenumber)

    if is_duplicate_entry(np.abs(face_mapping)) != 0:
        raise ValueError('face_mapping has duplicated values')

    return vertex_shift, face_mapping


def computeCoefficientsSecondOrder(qface1, qface2, qface3, qface4,
                                   r_vertices_ref, edge_vertices,
                                   face_vertices):
    ''' Compute the coefficients for edge basis functions of second order
    on reference element.

    :param float-array qface1: vector q on face 1.
    :param float-array qface2: vector q on face 2.
    :param float-array qface3: vector q on face 3.
    :param float-array qface4: vector q on face 4.
    :param float-array r_vertices: vertices of reference element.
    :param int-array edge_vertices: initialization of edges connectivity.
    :param int-array face_vertices: initialization of faces connectivity.
    :return: coefficients.
    :rtype: float.

    .. note: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    '''
    # First order ele
    firstOrderEdgeElement = 6

    # Second order ele
    secondOrderEdgeElement = 20

    # Absolute values for edges numbering
    edge_vertices_abs = np.abs(edge_vertices)

    # Definition of tangetial unitary vectors for each edge (2 dofs per edge)
    taui = np.zeros((3, firstOrderEdgeElement*2), dtype=np.float)

    for iedge in np.arange(firstOrderEdgeElement):
        taui[:, iedge*2] = (r_vertices_ref[:, edge_vertices_abs[iedge, 1]] -
                            r_vertices_ref[:, edge_vertices_abs[iedge, 0]])
        taui[:, iedge*2+1] = taui[:, 2*iedge]

    # Normalization of taui by using edges length
    length = np.zeros(firstOrderEdgeElement, dtype=np.float)

    for iedge in np.arange(firstOrderEdgeElement):
        length[iedge] = norm(taui[:, iedge*2+1])
        taui[:, iedge*2: 2*iedge+2] = taui[:, iedge*2: 2*iedge+2]/length[iedge]
        if (np.sum(np.sign(edge_vertices[iedge, :]+1)) == -2):
            taui[:, iedge*2: 2*iedge+2] = -taui[:, iedge*2: 2*iedge+2]
        elif (np.sum(np.sign(edge_vertices[iedge, :]+1)) == 0):
            raise ValueError('The signs of ', iedge, ' are incorrect.')

    # Definition of normal vectors to faces
    n1 = np.array([0, 0, -1], dtype=np.float)
    n1 = n1/norm(n1)
    n2 = np.array([-1, 0, 0], dtype=np.float)
    n2 = n2/norm(n2)
    n3 = np.array([0, -1, 0], dtype=np.float)
    n3 = n3/norm(n3)
    n4 = np.array([1, 1, 1], dtype=np.float)
    n4 = n4/norm(n4)

    # Allocate
    matrix = np.zeros((secondOrderEdgeElement, secondOrderEdgeElement),
                      dtype=np.float)

    # ----- Auxiliar definition of edges -----
    # We define the edges and their dof according to a predefined direction
    # given by edge_vertices_aux. Then at the end the necessary changes are
    # made so that the dof of the edges adhere to the definition given by
    #  edge_vertices. Here we use a 1-based indexes.
    edge_vertices_aux = np.array([[1, 2],
                                  [2, 3],
                                  [-3, -1],
                                  [1, 4],
                                  [2, 4],
                                  [3, 4]], dtype=np.int)
    # Edges length
    LengthAris = length

    # Integrals
    # Int edge 1 (node 1 --> 2)
    aux_x_L1 = LengthAris[0]*np.array([1./2., 1./6., 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      dtype=np.float)
    aux_x_L2 = LengthAris[0]*np.array([1./2., 1./3., 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      dtype=np.float)
    matrix[0, :] = aux_x_L1    # Dof 1 associated to L1
    matrix[1, :] = aux_x_L2    # Dof 2 associated to L2

    del aux_x_L1, aux_x_L2

    aux_x_L2 = LengthAris[1]*np.array([1./2., 1./3., 1./6., 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 1./12., 0, -1./12., 0, 0, 0,
                                      0, 0], dtype=np.float)
    aux_y_L2 = LengthAris[1]*np.array([0, 0, 0, 0, 1./2., 1./3., 1./6., 0, 0,
                                      0, 0, 0, -1./12., 0, 1./4., 0, 0, 0,
                                      0, 0], dtype=np.float)
    aux_x_L3 = LengthAris[1]*np.array([1./2., 1./6., 1./3., 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 1./4., 0, -1./12., 0, 0, 0,
                                      0, 0], dtype=np.float)
    aux_y_L3 = LengthAris[1]*np.array([0, 0, 0, 0, 1./2., 1./6., 1./3., 0, 0,
                                      0, 0, 0, -1./12., 0, 1./12., 0, 0, 0,
                                      0, 0], dtype=np.float)

    # Dof 3 associated to L2
    matrix[2, :] = (1./LengthAris[1])*(-aux_x_L2+aux_y_L2)
    # Dof 4 associated to L3
    matrix[3, :] = (1./LengthAris[1])*(-aux_x_L3+aux_y_L3)

    del aux_x_L2, aux_y_L2, aux_x_L3, aux_y_L3

    # Int edge 3 (node 1 --> 3)
    aux_y_L1 = LengthAris[2]*np.array([0, 0, 0, 0, 1./2., 0, 1./6., 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      dtype=np.float)
    aux_y_L3 = LengthAris[2]*np.array([0, 0, 0, 0, 1./2., 0, 1./3., 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      dtype=np.float)
    matrix[4, :] = aux_y_L3   # Dof 5 associated to L3
    matrix[5, :] = aux_y_L1   # Dof 6 associated to L1

    del aux_y_L1, aux_y_L3

    # Int edge 4 (node 1 --> 4)
    aux_z_L1 = LengthAris[3]*np.array([0, 0, 0, 0, 0, 0, 0, 0, 1./2., 0, 0,
                                      1./6., 0, 0, 0, 0, 0, 0, 0, 0],
                                      dtype=np.float)
    aux_z_L4 = LengthAris[3]*np.array([0, 0, 0, 0, 0, 0, 0, 0, 1./2., 0, 0,
                                      1./3., 0, 0, 0, 0, 0, 0, 0, 0],
                                      dtype=np.float)
    matrix[6, :] = aux_z_L1   # Dof 7 associated to L1
    matrix[7, :] = aux_z_L4   # Dof 8 associated to L4

    del aux_z_L1, aux_z_L4

    # Int edge 5 (vertices 2 --> 4)
    aux_x_L2 = LengthAris[4]*np.array([1./2., 1./3., 0, 1./6., 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, -1./12., 1./12., 0,
                                      0, 0], dtype=np.float)
    aux_z_L2 = LengthAris[4]*np.array([0, 0, 0, 0, 0, 0, 0, 0, 1./2., 1./3.,
                                      0, 1./6., 0, 0, 0, 1./4., -1./12., 0,
                                      0, 0], dtype=np.float)
    aux_x_L4 = LengthAris[4]*np.array([1./2., 1./6., 0, 1./3., 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, -1./12., 1./4., 0, 0,
                                      0], dtype=np.float)
    aux_z_L4 = LengthAris[4]*np.array([0, 0, 0, 0, 0, 0, 0, 0, 1./2., 1./6.,
                                      0, 1./3., 0, 0, 0, 1./12., -1./12., 0,
                                      0, 0], dtype=np.float)

    # Dof 9 associated to L2
    matrix[8, :] = (1./np.sqrt(2.))*(-aux_x_L2+aux_z_L2)
    # Dof 10 associated to L4
    matrix[9, :] = (1./np.sqrt(2.))*(-aux_x_L4+aux_z_L4)

    del aux_x_L2, aux_z_L2, aux_x_L4, aux_z_L4

    # Int edge 6 (nodes 3 --> 4)
    aux_y_L3 = LengthAris[5]*np.array([0, 0, 0, 0, 1./2., 0, 1./3., 1./6., 0,
                                      0, 0, 0, 0, -1./12., 0, 0, 0, 1./12.,
                                      0, 0], dtype=np.float)
    aux_z_L3 = LengthAris[5]*np.array([0, 0, 0, 0, 0, 0, 0, 0, 1./2., 0, 1./3.,
                                      1./6., 0., 1./4., 0, 0, 0, -1./12., 0,
                                      0], dtype=np.float)
    aux_y_L4 = LengthAris[5]*np.array([0, 0, 0, 0, 1./2., 0, 1./6., 1./3., 0,
                                      0, 0, 0, 0, -1./12., 0, 0, 0, 1./4.,
                                      0, 0], dtype=np.float)
    aux_z_L4 = LengthAris[5]*np.array([0, 0, 0, 0, 0, 0, 0, 0, 1./2., 0, 1./6.,
                                      1./3., 0, 1./12., 0, 0, 0, -1./12., 0,
                                      0], dtype=np.float)

    # Dof 11 associated to L3
    matrix[10, :] = (1./np.sqrt(2))*(-aux_y_L3+aux_z_L3)
    # Dof 12 associated to L4
    matrix[11, :] = (1./np.sqrt(2))*(-aux_y_L4+aux_z_L4)

    del aux_y_L3, aux_z_L3, aux_y_L4, aux_z_L4

    # Mapping DOF order
    # A change of order of the dof (functions) of the edge is made to
    # fit the given in edge_vertices
    tau_sign, arista_mapping = aristaMappingTetrahedral(edge_vertices_aux,
                                                        edge_vertices)

    # Change sign 'tau' of edges
    nodes_sign = np.reshape(np.ones((2, 1))*tau_sign, (12, 1), order='F')
    tmp1 = nodes_sign*np.ones((1, secondOrderEdgeElement), dtype=np.float)
    matrix[0:12, :] = np.multiply(tmp1, matrix[0:12, :])

    # Change sign 'tau' of faces
    nodes_mapping = np.reshape(np.array([[2*np.abs(arista_mapping)],
                               [2*np.abs(arista_mapping)+1]]), (12), order='F')
    matrix[0:12, :] = matrix[nodes_mapping, :]

    # Once ordered by edges, sort the dof of each edge according to the
    # sign of arista_mapping
    for iedge in np.arange(firstOrderEdgeElement):
        if np.sign(arista_mapping[iedge]) == -1:
            matrix[[2*iedge+1, 2*iedge], :] = matrix[[2*iedge, 2*iedge+1], :]

    del edge_vertices_aux, arista_mapping, tau_sign, nodes_sign, nodes_mapping

    # Conditions on faces int {n x Ni) .q}
    # We define the faces and their dof according to a predefined sign given
    # by face_vertices_aux. Then, the necessary changes are made so that the
    # dof of the faces adhere to the definition given by edge_vertices. In
    # this case, the dof are all associated to the barycenter of the face
    # with what there is no such numbering sense as such and the only
    # thing we will do is map the dof of some faces in the dof of other faces.
    face_vertices_aux = np.array([[1, 2, 3],
                                  [1, 3, 4],
                                  [1, 4, 2],
                                  [2, 4, 3]], dtype=np.int)

    # The qfacei that enter as a parameter of entry to the routine are
    # referred to face numbering given by face_vertices. You have to map
    # them to the number given by face_verttices_aux, which is what the
    # calculations are programmed to do.

    # For second order, vertex_shift can be ignored
    stage = 1
    _, face_mapping = faceMappingTetrahedral(face_vertices, face_vertices_aux,
                                             stage)

    tmp1 = np.array([[2*np.abs(face_mapping)], [2*np.abs(face_mapping)+1]])
    q_mapping_aux = np.reshape(tmp1, (8), order='F')

    qface_all = np.hstack((qface1, qface2, qface3, qface4))
    qface_all = qface_all[:, q_mapping_aux]
    qface1 = qface_all[:, [0, 1]]
    qface2 = qface_all[:, [2, 3]]
    qface3 = qface_all[:, [4, 5]]
    qface4 = qface_all[:, [6, 7]]

    # The variables aux_x (i, :), aux_y (i, :), aux_z (i, :) are the
    # integrals of Nx, Ny and Nz on the i side of the tetrahedron. The
    # integral is a function of the coefficients in such a way that
    # aux_x (ii,:)*coeff is the integral coefficient Nix on face ii.

    # Int face1
    aux_x = np.array([1./2., 1./6., 1./6., 0, 0, 0, 0, 0, 0, 0, 0, 0, 1./12.,
                      0, -1./24., 0, 0, 0, 0, 0], dtype=np.float)
    aux_y = np.array([0, 0, 0, 0, 1./2., 1./6., 1./6., 0, 0, 0, 0, 0, -1./24.,
                      0, 1./12., 0, 0, 0, 0, 0], dtype=np.float)
    aux_z = np.zeros(secondOrderEdgeElement)  # No considered

    q1 = qface1[:, 0]
    q2 = qface1[:, 1]
    matrix[12, :] = ((q1[1]*n1[2]-q1[2]*n1[1])*aux_x +
                     (q1[2]*n1[0]-q1[0]*n1[2])*aux_y +
                     (q1[0]*n1[1]-q1[1]*n1[0])*aux_z)
    matrix[13, :] = ((q2[1]*n1[2]-q2[2]*n1[1])*aux_x +
                     (q2[2]*n1[0]-q2[0]*n1[2])*aux_y +
                     (q2[0]*n1[1]-q2[1]*n1[0])*aux_z)

    del aux_x, aux_y, aux_z

    # Int face2
    aux_x = np.zeros(secondOrderEdgeElement)  # No considered
    aux_y = np.array([0, 0, 0, 0, 1./2., 0, 1./6., 1./6., 0, 0, 0, 0, 0,
                     -1./24., 0, 0, 0, 1./12., 0, 0], dtype=np.float)
    aux_z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1./2., 0, 1./6., 1./6., 0,
                      1./12., 0, 0, 0, -1./24., 0, 0], dtype=np.float)

    q1 = qface2[:, 0]
    q2 = qface2[:, 1]
    matrix[14, :] = ((q1[1]*n2[2]-q1[2]*n2[1])*aux_x +
                     (q1[2]*n2[0]-q1[0]*n2[2])*aux_y +
                     (q1[0]*n2[1]-q1[1]*n2[0])*aux_z)
    matrix[15, :] = ((q2[1]*n2[2]-q2[2]*n2[1])*aux_x +
                     (q2[2]*n2[0]-q2[0]*n2[2])*aux_y +
                     (q2[0]*n2[1]-q2[1]*n2[0])*aux_z)

    del aux_x, aux_y, aux_z

    # Int face3
    aux_x = np.array([1./2., 1./6., 0, 1./6., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, -1./24., 1./12., 0, 0, 0], dtype=np.float)
    aux_y = np.zeros(secondOrderEdgeElement)  # No considered
    aux_z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1./2., 1./6., 0, 1./6., 0, 0, 0,
                     1./12., -1./24., 0, 0, 0], dtype=np.float)

    q1 = qface3[:, 0]
    q2 = qface3[:, 1]
    matrix[16, :] = ((q1[1]*n3[2]-q1[2]*n3[1])*aux_x +
                     (q1[2]*n3[0]-q1[0]*n3[2])*aux_y +
                     (q1[0]*n3[1]-q1[1]*n3[0])*aux_z)
    matrix[17, :] = ((q2[1]*n3[2]-q2[2]*n3[1])*aux_x +
                     (q2[2]*n3[0]-q2[0]*n3[2])*aux_y +
                     (q2[0]*n3[1]-q2[1]*n3[0])*aux_z)

    del aux_x, aux_y, aux_z

    # Int face4
    # Here (x,y,z) are L1, L2, L3
    area_face4 = np.sqrt(3.)/2.

    aux_x = 2.*area_face4*np.array([1./2., 1./6., 1./6., 1./6., 0, 0, 0, 0,
                                    0, 0, 0, 0, 1./12., 0, -1./24., -1./24.,
                                    1./12., 0, 1./24., 0], dtype=np.float)
    aux_y = 2.*area_face4*np.array([0, 0, 0, 0, 1./2., 1./6., 1./6., 1./6., 0,
                                    0, 0, 0, -1./24., -1./24., 1./12., 0, 0,
                                    1./12., -1./24., 1./24.], dtype=np.float)
    aux_z = 2.*area_face4*np.array([0, 0, 0, 0, 0, 0, 0, 0, 1./2., 1./6.,
                                    1./6., 1./6., 0, 1./12., 0, 1./12.,
                                   -1./24., -1./24., 0, -1./24],
                                   dtype=np.float)

    q1 = qface4[:, 0]
    q2 = qface4[:, 1]
    matrix[18, :] = ((q1[1]*n4[2]-q1[2]*n4[1])*aux_x +
                     (q1[2]*n4[0]-q1[0]*n4[2])*aux_y +
                     (q1[0]*n4[1]-q1[1]*n4[0])*aux_z)
    matrix[19, :] = ((q2[1]*n4[2]-q2[2]*n4[1])*aux_x +
                     (q2[2]*n4[0]-q2[0]*n4[2])*aux_y +
                     (q2[0]*n4[1]-q2[1]*n4[0])*aux_z)

    del aux_x, aux_y, aux_z

    # vertex_shift can be ignored for the case of second order since the
    # dof are associated with barycenter of the face and not a certain
    # face vertice
    stage = 2
    _, face_mapping = faceMappingTetrahedral(face_vertices_aux, face_vertices,
                                             stage)

    tmp1 = np.array([[2*np.abs(face_mapping)], [2*np.abs(face_mapping)+1]])
    nodes_mapping = np.int(12) + np.reshape(tmp1, (8), order='F')

    matrix[12:20, :] = matrix[nodes_mapping, :]

    del face_vertices_aux, face_mapping, nodes_mapping, qface_all

    # computation of the second member: one column for each Ni
    secm = np.zeros((secondOrderEdgeElement, secondOrderEdgeElement),
                    dtype=np.float)
    for ii in np.arange(secondOrderEdgeElement):
        if ii <= 11:
            secm[ii, ii] = 1.
        elif ii > 11:
            secm[ii, ii] = 1.

    # Dual basis computation. Obtaining polynomial coefficients by resolution
    # of the system: matrix * {coef} = {secm}
    coef = lstsq(matrix, secm, rcond=None)[0]

    # Coefficients EPS --> 0
    EPS = 1.e-12
    aux1, aux2 = np.where(np.abs(coef) < EPS)
    size = aux1.shape
    n = size[0]

    for kk in np.arange(n):
        coef[aux1[kk], aux2[kk]] = 0.

    a1 = coef[0, :]
    a2 = coef[1, :]
    a3 = coef[2, :]
    a4 = coef[3, :]
    b1 = coef[4, :]
    b2 = coef[5, :]
    b3 = coef[6, :]
    b4 = coef[7, :]
    c1 = coef[8, :]
    c2 = coef[9, :]
    c3 = coef[10, :]
    c4 = coef[11, :]
    D = coef[12, :]
    E = coef[13, :]
    F = coef[14, :]
    G = coef[15, :]
    H = coef[16, :]
    II = coef[17, :]
    JJ = coef[18, :]
    K = coef[19, :]

    return (a1, a2, a3, a4, b1, b2, b3, b4, c1,
            c2, c3, c4, D, E, F, G, H, II, JJ, K)


def NiTetrahedralSecondOrder(a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4,
                             D, E, F, G, H, II, JJ, K, x, y, z, r):
    ''' Computation of Ni (Nedelec basis functions of second order) in a
    tetrahedral element with vertices (x,y,z) for point r.

    :param float coefficients: a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4,D,E,
     F,G,H,II,JJ,K.
    :param float-array x: x-coordinates of reference element.
    :param float-array y: y-coordinates of reference element.
    :param float-array z: z-coordinates of reference element.
    :param float-array r: xyz coordinates of the evaluation point.
    :return: basis nedelec funcions of second order.
    :rtype: ndarray.

    .. note: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    '''
    # Number of dimensions
    nDimensions = 3

    # Second order ele
    secondOrderEdgeElement = 20

    # Initialization
    coef = np.zeros((secondOrderEdgeElement, secondOrderEdgeElement),
                    dtype=np.float)

    coef[0, :] = a1
    coef[1, :] = a2
    coef[2, :] = a3
    coef[3, :] = a4
    coef[4, :] = b1
    coef[5, :] = b2
    coef[6, :] = b3
    coef[7, :] = b4
    coef[8, :] = c1
    coef[9, :] = c2
    coef[10, :] = c3
    coef[11, :] = c4
    coef[12, :] = D
    coef[13, :] = E
    coef[14, :] = F
    coef[15, :] = G
    coef[16, :] = H
    coef[17, :] = II
    coef[18, :] = JJ
    coef[19, :] = K

    # Computation on reference element
    L = cartesianToVolumetricCoordinates(x, y, z, r)

    xref = np.array([0, 1, 0, 0], dtype=np.float)
    yref = np.array([0, 0, 1, 0], dtype=np.float)
    zref = np.array([0, 0, 0, 1], dtype=np.float)

    rref = np.zeros(3, dtype=np.float)
    rref[0] = np.dot(L, xref)
    rref[1] = np.dot(L, yref)
    rref[2] = np.dot(L, zref)

    aux_x = np.array([1,  rref[0], rref[1], rref[2], 0, 0, 0, 0, 0, 0, 0,
                     0, rref[1]**2, 0,  -rref[0]*rref[1], -rref[0]*rref[2],
                     rref[2]**2, 0, rref[1]*rref[2], 0], dtype=np.float)

    aux_y = np.array([0, 0, 0, 0, 1, rref[0], rref[1], rref[2], 0, 0, 0, 0,
                     -rref[0]*rref[1], -rref[1]*rref[2], rref[0]**2, 0, 0,
                     rref[2]**2, -rref[0]*rref[2], rref[0]*rref[2]],
                     dtype=np.float)

    aux_z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, rref[0], rref[1], rref[2],
                     0, rref[1]**2, 0, rref[0]**2, -rref[0]*rref[2],
                     -rref[1]*rref[2], 0, -rref[0]*rref[1]], dtype=np.float)

    # Allocate
    Niref = np.zeros((nDimensions, secondOrderEdgeElement), dtype=np.float)

    Niref[0, :] = np.matmul(aux_x, coef)
    Niref[1, :] = np.matmul(aux_y, coef)
    Niref[2, :] = np.matmul(aux_z, coef)

    del aux_x, aux_y, aux_z

    # Vector element reference to the real element through the Jacobian
    # Ni_real=([J]^-1)*Niref

    x21 = x[1]-x[0]
    y21 = y[1]-y[0]
    z21 = z[1]-z[0]
    x31 = x[2]-x[0]
    y31 = y[2]-y[0]
    z31 = z[2]-z[0]
    x41 = x[3]-x[0]
    y41 = y[3]-y[0]
    z41 = z[3]-z[0]

    jacob = np.array([[x21, y21, z21], [x31, y31, z31],
                      [x41, y41, z41]], dtype=np.float)

    Ni = lstsq(jacob, Niref, rcond=None)[0]

    return Ni


def nedelecBasisSecondOrder(a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, D,
                            E, F, G, H, II, JJ, K, ref_ele, points):
    ''' This function computes the basis nedelec functions of second order
    for a set of points in a  given tetrahedral element.

    :param float coefficients: a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4,
     D,E,F,G,H,II,JJ,K.
    :param float-array ref_ele: nodal coordinates of reference element.
    :param float-array points: spatial coordinates of the evaluation points.
    :return: basis nedelec functions of second order.
    :rtype: ndarray.

    .. note: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    '''
    # Number of points
    size = points.shape
    if len(size) == 2:  # More than one point
        nPoints = size[1]
    elif len(size) == 1:    # One point
        nPoints = 1

    # Number of dimensions
    nDimensions = 3

    # Second order ele
    secondOrderEdgeElement = 20

    # Allocate
    basis = np.zeros((nDimensions, secondOrderEdgeElement, nPoints),
                     dtype=np.float)

    # Get reference element coordinates
    xref = ref_ele[0, :]
    yref = ref_ele[1, :]
    zref = ref_ele[2, :]

    # Compute nedelec basis functions of second order for all points
    if nPoints == 1:    # One point
        for iPoint in np.arange(nPoints):
            r = points

            # Basis funtions for iPoint|
            basis[:, :, iPoint] = NiTetrahedralSecondOrder(a1, a2, a3, a4,
                                                           b1, b2, b3, b4,
                                                           c1, c2, c3, c4,
                                                           D, E, F, G, H,
                                                           II, JJ, K, xref,
                                                           yref, zref, r)
    else:    # More than one point
        for iPoint in np.arange(nPoints):
            r = points[:, iPoint]

            # Basis funtions for iPoint|
            basis[:, :, iPoint] = NiTetrahedralSecondOrder(a1, a2, a3, a4,
                                                           b1, b2, b3, b4,
                                                           c1, c2, c3, c4,
                                                           D, E, F, G, H,
                                                           II, JJ, K, xref,
                                                           yref, zref, r)

    return basis


def computeMassMatrixSecondOrder(a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3,
                                 c4, D, E, F, G, H, II, JJ, K, ref_ele, GR,
                                 signs, DetJacob, Wi, rx, ry, rz, ngaussP):
    ''' Compute mass matrix for tetrahedral edge elements of second order.

    :param float coefficients: a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4,D,E,F,G,
     H,II,JJ,K.
    :param float-array ref_ele: nodal coordinates of reference element.
    :param float-array GR: tensor.
    :param int-array signs: dofs signs.
    :param float DetJacob: determinant of the jacobian.
    :param float Wi: gauss weigths.
    :param float-array rx: x-coordinates of gauss points.
    :param float-array ry: y-coordinates of gauss points.
    :param float-array rz: z-coordinates of gauss points.
    :param int ngaussP: number of gauss points.
    :return: mass matrix for edge element of second order.
    :rtype: ndarray.

    .. note: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    '''
    # Second order ele
    secondOrderEdgeElement = 20

    # Get reference element coordinates
    xref = ref_ele[0, :]
    yref = ref_ele[1, :]
    zref = ref_ele[2, :]

    # Allocate
    Me = np.zeros((secondOrderEdgeElement, secondOrderEdgeElement),
                  dtype=np.float)

    # Mass matrix computation
    r = np.zeros(3, dtype=np.float)
    for iPoint in np.arange(ngaussP):
        r[0] = rx[iPoint]
        r[1] = ry[iPoint]
        r[2] = rz[iPoint]

        Ni = NiTetrahedralSecondOrder(a1, a2, a3, a4, b1, b2, b3, b4,
                                      c1, c2, c3, c4, D, E, F, G, H,
                                      II, JJ, K, xref, yref, zref, r)
        nix = Ni[0, :]
        niy = Ni[1, :]
        niz = Ni[2, :]

        for ii in np.arange(secondOrderEdgeElement):
            for jj in np.arange(secondOrderEdgeElement):
                Me[ii, jj] += Wi[iPoint]*((GR[0, 0] * nix[ii]*nix[jj] +
                                           GR[0, 1] * (nix[ii]*niy[jj] +
                                                       niy[ii]*nix[jj]) +
                                           GR[1, 1] * niy[ii]*niy[jj] +
                                           GR[0, 2] * (nix[ii]*niz[jj] +
                                                       niz[ii]*nix[jj]) +
                                           GR[1, 2] * (niy[ii]*niz[jj] +
                                                       niz[ii]*niy[jj]) +
                                           GR[2, 2] * niz[ii]*niz[jj]) *
                                          signs[ii]*signs[jj])

    Me = Me*DetJacob*(1./6.)

    return Me


def computeDerivativesSecondOrder(a2, a3, a4, b2, b3, b4, c2, c3, c4,
                                  D, E, F, G, H, II, JJ, K, point):
    ''' Compute partial derivatives of basis functions for tetrahedral edge
    elements of second order (reference element)

    :param float coefficients: a2,a3,a4,b2,b3,b4,c2,c3,c4,D,E,F,G,H,II,JJ,K.
    :param float-array point: coordinates of the gaussian point.
    :return: partial derivatives for edge element of second order.
    :rtype: ndarray.

    .. note: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    '''
    # Point coordinates
    x = point[0]
    y = point[1]
    z = point[2]

    # dxNix
    dxNix = a2 - F*y - G*z

    # dxNiy
    dxNiy = b2 + 2.*F*x - D*y - JJ*z + K*z

    # dxNiz
    dxNiz = c2 + 2.*G*x - K*y - H*z

    # dyNix
    dyNix = a3 - F*x + 2.*D*y + JJ*z

    # dyNiy
    dyNiy = b3 - D*x - E*z

    # dyNiz
    dyNiz = c3 - K*x + 2.*E*y - II*z

    # dzNix
    dzNix = a4 - G*x + JJ*y + 2.*H*z

    # dzNiy
    dzNiy = b4 - JJ*x + K*x - E*y + 2.*II*z

    # dzNiz
    dzNiz = c4 - H*x - II*y

    return dxNix, dxNiy, dxNiz, dyNix, dyNiy, dyNiz, dzNix, dzNiy, dzNiz


def computeStiffnessMatrixSecondOrder(a2, a3, a4, b2, b3, b4, c2, c3, c4,
                                      D, E, F, G, H, II, JJ, K, invjj, signs,
                                      DetJacob, Wi, rx, ry, rz, ngaussP):
    ''' Compute stiffness matrix for tetrahedral edge elements of second order.

    :param float coefficients: a2,a3,a4,b2,b3,b4,c2,c3,c4,D,E,F,G,H,II,JJ,K.
    :param float-array invjj: inverse jacobian.
    :param int-array signs: dofs signs.
    :param float DetJacob: determinant of the jacobian.
    :param float Wi: gauss weigths.
    :param float-array rx: x-coordinates of gauss points.
    :param float-array ry: y-coordinates of gauss points.
    :param float-array rz: z-coordinates of gauss points.
    :param int ngaussP: number of gauss points.
    :return: stiffness matrix for edge element of second order.
    :rtype: ndarray.

    .. note: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    '''
    # Second order element
    secondOrderEdgeElement = 20

    # Allocate
    Ke = np.zeros((secondOrderEdgeElement, secondOrderEdgeElement),
                  dtype=np.float)

    # Tensor computation
    fr = np.eye(3, dtype=np.float)
    invfr = inv(fr)

    tmp1 = np.array([[0, 0, 0], [0, 0, 1], [0, -1,  0]], dtype=np.float)
    A = np.matmul(invjj.transpose(), np.matmul(tmp1, invjj))

    tmp1 = np.array([[0, 0, -1], [0, 0, 0], [1, 0,  0]], dtype=np.float)
    B = np.matmul(invjj.transpose(), np.matmul(tmp1, invjj))

    tmp1 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0,  0]], dtype=np.float)
    C = np.matmul(invjj.transpose(), np.matmul(tmp1, invjj))

    # Stiffness matrix computation
    r = np.zeros(3, dtype=np.float)
    for iPoint in np.arange(ngaussP):
        r[0] = rx[iPoint]
        r[1] = ry[iPoint]
        r[2] = rz[iPoint]

        [_, dxNiy, dxNiz,
         dyNix, _, dyNiz,
         dzNix, dzNiy, _] = computeDerivativesSecondOrder(a2, a3, a4, b2, b3,
                                                          b4, c2, c3, c4, D,
                                                          E, F, G, H,
                                                          II, JJ, K, r)

        rotNix = (A[0, 1]*dxNiy + A[0, 2]*dxNiz -
                  A[0, 1]*dyNix + A[1, 2]*dyNiz -
                  A[0, 2]*dzNix - A[1, 2]*dzNiy)
        rotNiy = (B[0, 1]*dxNiy + B[0, 2]*dxNiz -
                  B[0, 1]*dyNix + B[1, 2]*dyNiz -
                  B[0, 2]*dzNix - B[1, 2]*dzNiy)
        rotNiz = (C[0, 1]*dxNiy + C[0, 2]*dxNiz -
                  C[0, 1]*dyNix + C[1, 2]*dyNiz -
                  C[0, 2]*dzNix - C[1, 2]*dzNiy)

        for ii in np.arange(secondOrderEdgeElement):
            for jj in np.arange(secondOrderEdgeElement):
                Ke[ii, jj] += Wi[iPoint]*((invfr[0, 0]*rotNix[ii]*rotNix[jj] +
                                           invfr[0, 1]*(rotNix[ii] *
                                                        rotNiy[jj] +
                                                        rotNiy[ii] *
                                                        rotNix[jj]) +
                                           invfr[1, 1]*rotNiy[ii]*rotNiy[jj] +
                                           invfr[0, 2]*(rotNix[ii] *
                                                        rotNiz[jj] +
                                                        rotNiz[ii] *
                                                        rotNix[jj]) +
                                           invfr[1, 2]*(rotNiy[ii] *
                                                        rotNiz[jj] +
                                                        rotNiz[ii] *
                                                        rotNiy[jj]) +
                                           invfr[2, 2]*rotNiz[ii]*rotNiz[jj]) *
                                          signs[ii]*signs[jj])

    Ke = Ke*DetJacob*(1./6.)

    return Ke


def polynomialProduct(poly1, poly2):
    ''' Compute the product of two polynomials according to convention for
    tetrahedral edge element of third order.

    :param float-array poly1: first polynomial to be computed.
    :param float-array poly2: second polynomial to be computed.
    :return: polynomial product.
    :rtype: ndarray
    '''

    # Allocate
    polProd = np.zeros((5, 5), dtype=np.float)

    for abs1 in np.arange(5):
        for ord1 in np.arange(5-abs1):
            for abs2 in np.arange(5):
                for ord2 in np.arange(5-abs2):
                    if (poly1[abs1, ord1] != 0):
                        if (poly2[abs2, ord2] != 0):
                            indx1 = abs1+abs2
                            indx2 = ord1+ord2
                            polProd[indx1, indx2] = (polProd[indx1, indx2] +
                                                     poly1[abs1, ord1] *
                                                     poly2[abs2, ord2])
    return polProd


def componentProductPolynomial(nxNi_term, function_n, component_q, face):
    ''' Compute the product for a given component of nxNi.

    :param float-array nxNi_term: matrix of coefficients.
    :param float-array function_n: matrix function.
    :param float-array component_q: id of the q component.
    :param float-array face: face of the aforementioned parameters.
    :return: product of nxNi, face and function_q.
    :rtype: ndarray
    '''

    # Third order element
    thirdOrderEdgeElement = 45

    # Copy data as float
    data_temp = nxNi_term.astype(np.float)

    if (component_q != 0):
        for coefficient in np.arange(1, thirdOrderEdgeElement+1):
            indx1 = (coefficient-1)*5+1
            indx2 = coefficient*5
            coeff_Ni = data_temp[:, indx1-1:indx2]

            coeff_aux_indep = np.zeros((5, 5), dtype=np.float)
            if (function_n[0, 0] != 0):
                output = 0
                for abs in np.arange(5):
                    for ord in np.arange(5-abs):
                        if (coeff_Ni[abs, ord] != 0):
                            coeff_aux_indep[abs, ord] = (function_n[0, 0] *
                                                         coeff_Ni[abs, ord] *
                                                         component_q)
                            if (face != 4):
                                output = 1
                                break
                    if (output == 1):
                        break

            coeff_aux_abs = np.zeros((5, 5), dtype=np.float)
            if (function_n[1, 0] != 0):
                output = 0
                for abs in np.arange(5):
                    for ord in np.arange(5-abs):
                        if (coeff_Ni[abs, ord] != 0):
                            coeff_aux_abs[abs+1, ord] = (function_n[1, 0] *
                                                         coeff_Ni[abs, ord] *
                                                         component_q)
                            if (face != 4):
                                output = 1
                                break
                    if (output == 1):
                        break

            coeff_aux_ord = np.zeros((5, 5), dtype=np.float)
            if (function_n[0, 1] != 0):
                output = 0
                for abs in np.arange(5):
                    for ord in np.arange(5-abs):
                        if (coeff_Ni[abs, ord] != 0):
                            coeff_aux_ord[abs, ord+1] = (function_n[0, 1] *
                                                         coeff_Ni[abs, ord] *
                                                         component_q)
                            if (face != 4):
                                output = 1
                                break
                    if (output == 1):
                        break

            tmp = coeff_aux_indep + coeff_aux_abs + coeff_aux_ord
            indx1_1 = ((coefficient-1)*5+1)-1
            indx2_1 = coefficient*5
            data_temp[:, indx1_1:indx2_1] = tmp

        product_result = data_temp
    else:
        product_result = np.zeros((5, 225), dtype=np.float)

    return product_result


def analyticIntegral(input_functions):
    ''' Compute the integrals of the product of monomials in the reference
    triangle.

    :param float-array input_functions: matrix of functions with coefficients
     of Ni before integration process.
    :return: integrals of the product of monomials in the reference triangle.
    :rtype: ndarray
    '''

    # Third order element
    thirdOrderEdgeElement = 45

    integral_tab = np.array([[1/2, 1/6, 1/12, 1/20, 1/30],
                             [1/6, 1/24, 1/60, 1/120, 0],
                             [1/12, 1/60, 1/180, 0, 0],
                             [1/20, 1/120, 0, 0, 0],
                             [1/30, 0, 0, 0, 0]], dtype=np.float)
    # Allocate
    integral_res = np.zeros((thirdOrderEdgeElement), dtype=np.float)

    for coefficient in np.arange(1, thirdOrderEdgeElement+1):
        indx1 = (coefficient-1)*5+1
        indx2 = coefficient*5
        aux = input_functions[:, indx1-1:indx2]

        for abs in np.arange(5):
            for ord in np.arange(5-abs):
                aux[abs, ord] = aux[abs, ord]*integral_tab[abs, ord]

        integral_res[coefficient-1] = 0

        for abs in np.arange(5):
            for ord in np.arange(5-abs):
                integral_res[coefficient-1] = (integral_res[coefficient-1] +
                                               aux[abs, ord])
    return integral_res


def computeCoefficientsThirdOrder(qface1, qface2, qface3, qface4):
    ''' Compute the coefficients for edge basis functions of third order
    on reference element.

    :param float-array qface1: vector q on face 1.
    :param float-array qface2: vector q on face 2.
    :param float-array qface3: vector q on face 3.
    :param float-array qface4: vector q on face 4.
    :return: coefficients.
    :rtype: float.

    .. note: References:\n
       Garcia-Castillo, L. E., Ruiz-Genovés, A. J., Gómez-Revuelto, I.,
       Salazar-Palma, M., & Sarkar, T. K. (2002). Third-order Nédélec
       curl-conforming finite element. IEEE transactions on magnetics,
       38(5), 2370-2372.
    '''
    # Third order element
    thirdOrderEdgeElement = 45

    # Definition of q functions
    q_functions = np.array([[1, -1, 0, 0, 0, 1],
                            [-1, 0, 1, 0, 0, 0]], dtype=np.float)

    # Definition of q functions on faces
    q_faces_ref = np.hstack((qface1, qface2, qface3, qface4))

    # Definition of q functions on volume
    q_volumen_ref = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)

    # Tangential unitary vectors for each edge (18 dofs)
    Taui = np.array([[1, 1, 1, -1, -1, -1, 0, 0, 0,
                      0, 0, 0, -1, -1, -1, 0, 0, 0],
                     [0, 0, 0, 1, 1, 1, -1, -1, -1,
                      0, 0, 0, 0, 0, 0, -1, -1, -1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                      1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float)

    # Normalization to unitary module
    for idof in np.arange(18):
        Taui[:, idof] = Taui[:, idof]/norm(Taui[:, idof])

    # Edge length on reference element
    length = np.array([1, np.sqrt(2), 1, 1, np.sqrt(2), np.sqrt(2)],
                      dtype=np.float)

    # Definition of the position of the 18 nodes of the edges. "a" is the
    # distance to the center of the edge of the integration points for the
    # interval [-1,1]. For our length edge Li => points located at
    # "(Li / 2) * a" from the center of the edge. In the case that concerns
    # us, tetrahedron of order 3, the third point of integration is the center
    # of the edge. a = sqrt (3/5) in the case of third-order Gauss integration.
    a = 0.7745966692

    # Auxiliar varaible for spatial coordinates of the nodes
    aux1 = (1.-a)/2.
    aux2 = 1./2.
    aux3 = (1.+a)/2.

    # Nodei contains coordinates of the 18 nodes on edges
    Nodei = np.array([[aux1, aux2, aux3, aux3, aux2, aux1,
                       0, 0, 0, 0, 0, 0, aux3, aux2, aux1, 0, 0, 0],
                      [0, 0, 0, aux1, aux2, aux3, aux3, aux2, aux1,
                      0, 0, 0, 0, 0, 0, aux3, aux2, aux1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, aux1, aux2, aux3,
                      aux1, aux2, aux3, aux1, aux2, aux3]], dtype=np.float)

    # Clear variables
    del aux1, aux2

    # Definition of the first 18 rows of the coefficient matrix of the system
    # of equations to be presented. These equations are those that come from
    # the imposition of the dofs of the edges.
    matrix = np.zeros((thirdOrderEdgeElement, thirdOrderEdgeElement),
                      dtype=np.float)

    # Integrals over edges
    for idof in np.arange(18):
        tmp = np.hstack((Taui[0, idof], Nodei[0, idof]*Taui[0, idof],
                         Nodei[1, idof]*Taui[0, idof],
                         Nodei[2, idof]*Taui[0, idof],
                         (Nodei[0, idof]**2)*Taui[0, idof],
                         (Nodei[1, idof]**2)*Taui[0, idof],
                         (Nodei[2, idof]**2)*Taui[0, idof],
                         Nodei[0, idof]*Nodei[1, idof]*Taui[0, idof],
                         Nodei[0, idof]*Nodei[2, idof]*Taui[0, idof],
                         Nodei[1, idof]*Nodei[2, idof]*Taui[0, idof],
                         Taui[1, idof], Nodei[0, idof]*Taui[1, idof],
                         Nodei[1, idof]*Taui[1, idof],
                         Nodei[2, idof]*Taui[1, idof],
                         (Nodei[0, idof]**2)*Taui[1, idof],
                         (Nodei[1, idof]**2)*Taui[1, idof],
                         (Nodei[2, idof]**2)*Taui[1, idof],
                         Nodei[0, idof]*Nodei[1, idof]*Taui[1, idof],
                         Nodei[0, idof]*Nodei[2, idof]*Taui[1, idof],
                         Nodei[1, idof]*Nodei[2, idof]*Taui[1, idof],
                         Taui[2, idof], Nodei[0, idof]*Taui[2, idof],
                         Nodei[1, idof]*Taui[2, idof],
                         Nodei[2, idof]*Taui[2, idof],
                         (Nodei[0, idof]**2)*Taui[2, idof],
                         (Nodei[1, idof]**2)*Taui[2, idof],
                         (Nodei[2, idof]**2)*Taui[2, idof],
                         Nodei[0, idof]*Nodei[1, idof]*Taui[2, idof],
                         Nodei[0, idof]*Nodei[2, idof]*Taui[2, idof],
                         Nodei[1, idof]*Nodei[2, idof]*Taui[2, idof],
                         (Nodei[0, idof]**2)*Nodei[1, idof]*Taui[0, idof] -
                         (Nodei[0, idof]**3)*Taui[1, idof],
                         (Nodei[1, idof]**2)*Nodei[0, idof]*Taui[1, idof] -
                         (Nodei[1, idof]**3)*Taui[0, idof],
                         (Nodei[2, idof]**2)*Nodei[0, idof]*Taui[2, idof] -
                         (Nodei[2, idof]**3)*Taui[0, idof],
                         (Nodei[0, idof]**2)*Nodei[2, idof]*Taui[0, idof] -
                         (Nodei[0, idof]**3)*Taui[2, idof],
                         (Nodei[2, idof]**2)*Nodei[1, idof]*Taui[2, idof] -
                         (Nodei[2, idof]**3)*Taui[1, idof],
                         (Nodei[1, idof]**2)*Nodei[2, idof]*Taui[1, idof] -
                         (Nodei[1, idof]**3)*Taui[2, idof],
                         (Nodei[1, idof]**2)*Nodei[0, idof]*Taui[0, idof] -
                         (Nodei[0, idof]**2)*Nodei[1, idof]*Taui[1, idof],
                         (Nodei[2, idof]**2)*Nodei[0, idof]*Taui[0, idof] -
                         (Nodei[0, idof]**2)*Nodei[2, idof]*Taui[2, idof],
                         (Nodei[2, idof]**2)*Nodei[1, idof]*Taui[1, idof] -
                         (Nodei[1, idof]**2)*Nodei[2, idof]*Taui[2, idof],
                         Nodei[0, idof]*Nodei[1, idof] *
                         Nodei[2, idof]*Taui[0, idof] -
                         (Nodei[0, idof]**2)*Nodei[2, idof]*Taui[1, idof],
                         (Nodei[0, idof]**2)*Nodei[2, idof]*Taui[1, idof] -
                         (Nodei[0, idof]**2)*Nodei[1, idof]*Taui[2, idof],
                         Nodei[0, idof]*Nodei[1, idof] *
                         Nodei[2, idof]*Taui[1, idof] -
                         (Nodei[1, idof]**2)*Nodei[2, idof]*Taui[0, idof],
                         (Nodei[1, idof]**2)*Nodei[2, idof]*Taui[0, idof] -
                         (Nodei[1, idof]**2)*Nodei[0, idof]*Taui[2, idof],
                         Nodei[0, idof]*Nodei[1, idof] *
                         Nodei[2, idof]*Taui[2, idof] -
                         (Nodei[2, idof]**2)*Nodei[1, idof]*Taui[0, idof],
                         (Nodei[2, idof]**2)*Nodei[1, idof]*Taui[0, idof] -
                         (Nodei[2, idof]**2)*Nodei[0, idof]*Taui[1, idof]))

        matrix[idof, :] = tmp

    # Integrals over faces
    # Definition of a matrix zeros (5,5) for the coefficients that do not
    # appear in the definition of Ni particularized in each one of the faces
    # and that can not be ignored so that when stacking matrices all result
    # with the same size.
    null_values = np.zeros((5, 5), dtype=np.float)

    # Definition of x-component for Ni on face 1
    Ni_x_face1_a1 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face1_a1[0, 0] = 1
    Ni_x_face1_a2 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face1_a2[1, 0] = 1
    Ni_x_face1_a3 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face1_a3[0, 1] = 1
    Ni_x_face1_a5 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face1_a5[2, 0] = 1
    Ni_x_face1_a6 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face1_a6[0, 2] = 1
    Ni_x_face1_a8 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face1_a8[1, 1] = 1

    Ni_x_face1_D = np.zeros((5, 5), dtype=np.float)
    Ni_x_face1_D[2, 1] = 1
    Ni_x_face1_E = np.zeros((5, 5), dtype=np.float)
    Ni_x_face1_E[0, 3] = -1
    Ni_x_face1_J = np.zeros((5, 5), dtype=np.float)
    Ni_x_face1_J[1, 2] = 1

    Ni_x_face1 = np.hstack((Ni_x_face1_a1, Ni_x_face1_a2, Ni_x_face1_a3,
                            null_values, Ni_x_face1_a5, Ni_x_face1_a6,
                            null_values, Ni_x_face1_a8, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, Ni_x_face1_D, Ni_x_face1_E,
                            null_values, null_values, null_values, null_values,
                            Ni_x_face1_J, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values))

    # Definition of y-component for Ni on face 1
    Ni_y_face1_b1 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face1_b1[0, 0] = 1
    Ni_y_face1_b2 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face1_b2[1, 0] = 1
    Ni_y_face1_b3 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face1_b3[0, 1] = 1
    Ni_y_face1_b5 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face1_b5[2, 0] = 1
    Ni_y_face1_b6 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face1_b6[0, 2] = 1
    Ni_y_face1_b8 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face1_b8[1, 1] = 1

    Ni_y_face1_D = np.zeros((5, 5), dtype=np.float)
    Ni_y_face1_D[3, 0] = -1
    Ni_y_face1_E = np.zeros((5, 5), dtype=np.float)
    Ni_y_face1_E[1, 2] = 1
    Ni_y_face1_J = np.zeros((5, 5), dtype=np.float)
    Ni_y_face1_J[2, 1] = -1

    Ni_y_face1 = np.hstack((null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, Ni_y_face1_b1,
                            Ni_y_face1_b2, Ni_y_face1_b3, null_values,
                            Ni_y_face1_b5, Ni_y_face1_b6, null_values,
                            Ni_y_face1_b8, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, Ni_y_face1_D,
                            Ni_y_face1_E, null_values, null_values,
                            null_values, null_values, Ni_y_face1_J,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values,
                            null_values))

    # Definition of y-component for Ni on face 2
    Ni_y_face2_b1 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face2_b1[0, 0] = 1
    Ni_y_face2_b3 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face2_b3[1, 0] = 1
    Ni_y_face2_b4 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face2_b4[0, 1] = 1
    Ni_y_face2_b6 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face2_b6[2, 0] = 1
    Ni_y_face2_b7 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face2_b7[0, 2] = 1
    Ni_y_face2_b10 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face2_b10[1, 1] = 1

    Ni_y_face2_H = np.zeros((5, 5), dtype=np.float)
    Ni_y_face2_H[0, 3] = -1
    Ni_y_face2_I = np.zeros((5, 5), dtype=np.float)
    Ni_y_face2_I[2, 1] = 1
    Ni_y_face2_L = np.zeros((5, 5), dtype=np.float)
    Ni_y_face2_L[1, 2] = 1

    Ni_y_face2 = np.hstack((null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, Ni_y_face2_b1,
                            null_values, Ni_y_face2_b3, Ni_y_face2_b4,
                            null_values, Ni_y_face2_b6, Ni_y_face2_b7,
                            null_values, null_values, Ni_y_face2_b10,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, Ni_y_face2_H,
                            Ni_y_face2_I, null_values, null_values,
                            Ni_y_face2_L, null_values, null_values,
                            null_values, null_values, null_values,
                            null_values))

    # Definition of z-component for Ni on face 2
    Ni_z_face2_c1 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face2_c1[0, 0] = 1
    Ni_z_face2_c3 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face2_c3[1, 0] = 1
    Ni_z_face2_c4 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face2_c4[0, 1] = 1
    Ni_z_face2_c6 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face2_c6[2, 0] = 1
    Ni_z_face2_c7 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face2_c7[0, 2] = 1
    Ni_z_face2_c10 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face2_c10[1, 1] = 1

    Ni_z_face2_H = np.zeros((5, 5), dtype=np.float)
    Ni_z_face2_H[1, 2] = 1
    Ni_z_face2_I = np.zeros((5, 5), dtype=np.float)
    Ni_z_face2_I[3, 0] = -1
    Ni_z_face2_L = np.zeros((5, 5), dtype=np.float)
    Ni_z_face2_L[2, 1] = -1

    Ni_z_face2 = np.hstack((null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            Ni_z_face2_c1, null_values, Ni_z_face2_c3,
                            Ni_z_face2_c4, null_values, Ni_z_face2_c6,
                            Ni_z_face2_c7, null_values, null_values,
                            Ni_z_face2_c10, null_values, null_values,
                            null_values, null_values, Ni_z_face2_H,
                            Ni_z_face2_I, null_values, null_values,
                            Ni_z_face2_L, null_values, null_values,
                            null_values, null_values, null_values,
                            null_values))

    # Definition of x-component for Ni on face 3
    Ni_x_face3_a1 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face3_a1[0, 0] = 1
    Ni_x_face3_a2 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face3_a2[0, 1] = 1
    Ni_x_face3_a4 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face3_a4[1, 0] = 1
    Ni_x_face3_a5 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face3_a5[0, 2] = 1
    Ni_x_face3_a7 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face3_a7[2, 0] = 1
    Ni_x_face3_a9 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face3_a9[1, 1] = 1

    Ni_x_face3_F = np.zeros((5, 5), dtype=np.float)
    Ni_x_face3_F[3, 0] = -1
    Ni_x_face3_G = np.zeros((5, 5), dtype=np.float)
    Ni_x_face3_G[1, 2] = 1
    Ni_x_face3_K = np.zeros((5, 5), dtype=np.float)
    Ni_x_face3_K[2, 1] = 1

    Ni_x_face3 = np.hstack((Ni_x_face3_a1, Ni_x_face3_a2, null_values,
                            Ni_x_face3_a4, Ni_x_face3_a5, null_values,
                            Ni_x_face3_a7, null_values, Ni_x_face3_a9,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values,
                            Ni_x_face3_F, Ni_x_face3_G, null_values,
                            null_values, null_values, Ni_x_face3_K,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values))

    # Definition of z-component for Ni on face 3
    Ni_z_face3_c1 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face3_c1[0, 0] = 1
    Ni_z_face3_c2 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face3_c2[0, 1] = 1
    Ni_z_face3_c4 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face3_c4[1, 0] = 1
    Ni_z_face3_c5 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face3_c5[0, 2] = 1
    Ni_z_face3_c7 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face3_c7[2, 0] = 1
    Ni_z_face3_c9 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face3_c9[1, 1] = 1

    Ni_z_face3_F = np.zeros((5, 5), dtype=np.float)
    Ni_z_face3_F[2, 1] = 1
    Ni_z_face3_G = np.zeros((5, 5), dtype=np.float)
    Ni_z_face3_G[0, 3] = -1
    Ni_z_face3_K = np.zeros((5, 5), dtype=np.float)
    Ni_z_face3_K[1, 2] = -1

    Ni_z_face3 = np.hstack((null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            Ni_z_face3_c1, Ni_z_face3_c2, null_values,
                            Ni_z_face3_c4, Ni_z_face3_c5, null_values,
                            Ni_z_face3_c7, null_values, Ni_z_face3_c9,
                            null_values, null_values, null_values,
                            Ni_z_face3_F, Ni_z_face3_G, null_values,
                            null_values, null_values, Ni_z_face3_K,
                            null_values, null_values, null_values,
                            null_values, null_values,
                            null_values, null_values))

    # For the computation of the surface integral in the face 4, we are
    # going to do it translating it to the calculation of the integral on the
    # domain of its projection in the XY plane, and substituting z for the
    # value in the surface, i.e., z = 1-xy, taking into account that the
    # surface differential ds, is transformed into dxdy/cos (gamma), where
    # gamma is the director cosine with the z axis. For z = 1-x-y
    # cos(gamma)= 1/sqrt(3); With the help of the routine polynomialProduct
    # we will develop all the terms. We will define first the equivalent
    # terms of z(z), z^2 (z_2) and z^3 (z_3) that we will need.
    z = np.zeros((5, 5), dtype=np.float)
    z[0, 0] = 1
    z[1, 0] = -1
    z[0, 1] = -1
    z_2 = polynomialProduct(z, z)
    z_3 = polynomialProduct(z_2, z)

    # Definition of x-component for Ni on face 4
    Ni_x_face4_a1 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_a1[0, 0] = 1
    Ni_x_face4_a2 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_a2[1, 0] = 1
    Ni_x_face4_a3 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_a3[0, 1] = 1
    Ni_x_face4_a4 = z
    Ni_x_face4_a5 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_a5[2, 0] = 1
    Ni_x_face4_a6 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_a6[0, 2] = 1
    Ni_x_face4_a7 = z_2
    Ni_x_face4_a8 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_a8[1, 1] = 1

    Ni_x_face4_a9 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_a9[1, 0] = 1
    Ni_x_face4_a9 = polynomialProduct(Ni_x_face4_a9, z)

    Ni_x_face4_a10 = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_a10[0, 1] = 1
    Ni_x_face4_a10 = polynomialProduct(Ni_x_face4_a10, z)

    Ni_x_face4_D = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_D[2, 1] = 1
    Ni_x_face4_E = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_E[0, 3] = -1
    Ni_x_face4_F = -z_3

    Ni_x_face4_G = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_G[2, 0] = 1
    Ni_x_face4_G = polynomialProduct(Ni_x_face4_G, z)

    Ni_x_face4_J = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_J[1, 2] = 1

    Ni_x_face4_K = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_K[1, 0] = 1
    Ni_x_face4_K = polynomialProduct(Ni_x_face4_K, z_2)

    Ni_x_face4_M = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_M[1, 1] = 1
    Ni_x_face4_M = polynomialProduct(Ni_x_face4_M, z)

    Ni_x_face4_O = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_O[0, 2] = 1
    Ni_x_face4_O = -polynomialProduct(Ni_x_face4_O, z)

    Ni_x_face4_P = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_P[0, 2] = 1
    Ni_x_face4_P = polynomialProduct(Ni_x_face4_P, z)

    Ni_x_face4_Q = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_Q[0, 1] = 1
    Ni_x_face4_Q = -polynomialProduct(Ni_x_face4_Q, z_2)

    Ni_x_face4_R = np.zeros((5, 5), dtype=np.float)
    Ni_x_face4_R[0, 1] = 1
    Ni_x_face4_R = polynomialProduct(Ni_x_face4_R, z_2)

    Ni_x_face4 = np.hstack((Ni_x_face4_a1, Ni_x_face4_a2, Ni_x_face4_a3,
                            Ni_x_face4_a4, Ni_x_face4_a5, Ni_x_face4_a6,
                            Ni_x_face4_a7, Ni_x_face4_a8, Ni_x_face4_a9,
                            Ni_x_face4_a10, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, Ni_x_face4_D,
                            Ni_x_face4_E, Ni_x_face4_F, Ni_x_face4_G,
                            null_values, null_values, Ni_x_face4_J,
                            Ni_x_face4_K, null_values, Ni_x_face4_M,
                            null_values, Ni_x_face4_O, Ni_x_face4_P,
                            Ni_x_face4_Q, Ni_x_face4_R))

    # Definition of y-component for Ni on face 4
    Ni_y_face4_b1 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_b1[0, 0] = 1
    Ni_y_face4_b2 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_b2[1, 0] = 1
    Ni_y_face4_b3 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_b3[0, 1] = 1
    Ni_y_face4_b4 = z
    Ni_y_face4_b5 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_b5[2, 0] = 1
    Ni_y_face4_b6 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_b6[0, 2] = 1
    Ni_y_face4_b7 = z_2
    Ni_y_face4_b8 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_b8[1, 1] = 1

    Ni_y_face4_b9 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_b9[1, 0] = 1
    Ni_y_face4_b9 = polynomialProduct(Ni_y_face4_b9, z)

    Ni_y_face4_b10 = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_b10[0, 1] = 1
    Ni_y_face4_b10 = polynomialProduct(Ni_y_face4_b10, z)

    Ni_y_face4_D = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_D[3, 0] = -1
    Ni_y_face4_E = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_E[1, 2] = 1
    Ni_y_face4_H = -z_3

    Ni_y_face4_I = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_I[0, 2] = 1
    Ni_y_face4_I = polynomialProduct(Ni_y_face4_I, z)

    Ni_y_face4_J = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_J[2, 1] = -1

    Ni_y_face4_L = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_L[0, 1] = 1
    Ni_y_face4_L = polynomialProduct(Ni_y_face4_L, z_2)

    Ni_y_face4_M = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_M[2, 0] = 1
    Ni_y_face4_M = -polynomialProduct(Ni_y_face4_M, z)

    Ni_y_face4_N = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_N[2, 0] = 1
    Ni_y_face4_N = polynomialProduct(Ni_y_face4_N, z)

    Ni_y_face4_O = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_O[1, 1] = 1
    Ni_y_face4_O = polynomialProduct(Ni_y_face4_O, z)

    Ni_y_face4_R = np.zeros((5, 5), dtype=np.float)
    Ni_y_face4_R[1, 0] = 1
    Ni_y_face4_R = -polynomialProduct(Ni_y_face4_R, z_2)

    Ni_y_face4 = np.hstack((null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, Ni_y_face4_b1,
                            Ni_y_face4_b2, Ni_y_face4_b3, Ni_y_face4_b4,
                            Ni_y_face4_b5, Ni_y_face4_b6, Ni_y_face4_b7,
                            Ni_y_face4_b8, Ni_y_face4_b9, Ni_y_face4_b10,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, Ni_y_face4_D,
                            Ni_y_face4_E, null_values, null_values,
                            Ni_y_face4_H, Ni_y_face4_I, Ni_y_face4_J,
                            null_values, Ni_y_face4_L, Ni_y_face4_M,
                            Ni_y_face4_N, Ni_y_face4_O, null_values,
                            null_values, Ni_y_face4_R))

    # Definition of z-component for Ni on face 4
    Ni_z_face4_c1 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_c1[0, 0] = 1
    Ni_z_face4_c2 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_c2[1, 0] = 1
    Ni_z_face4_c3 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_c3[0, 1] = 1
    Ni_z_face4_c4 = z
    Ni_z_face4_c5 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_c5[2, 0] = 1
    Ni_z_face4_c6 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_c6[0, 2] = 1
    Ni_z_face4_c7 = z_2
    Ni_z_face4_c8 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_c8[1, 1] = 1

    Ni_z_face4_c9 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_c9[1, 0] = 1
    Ni_z_face4_c9 = polynomialProduct(Ni_z_face4_c9, z)

    Ni_z_face4_c10 = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_c10[0, 1] = 1
    Ni_z_face4_c10 = polynomialProduct(Ni_z_face4_c10, z)

    Ni_z_face4_F = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_F[1, 0] = 1
    Ni_z_face4_F = polynomialProduct(Ni_z_face4_F, z_2)

    Ni_z_face4_G = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_G[3, 0] = -1

    Ni_z_face4_H = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_H[0, 1] = 1
    Ni_z_face4_H = polynomialProduct(Ni_z_face4_H, z_2)

    Ni_z_face4_I = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_I[0, 3] = -1

    Ni_z_face4_K = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_K[2, 0] = 1
    Ni_z_face4_K = -polynomialProduct(Ni_z_face4_K, z)

    Ni_z_face4_L = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_L[0, 2] = 1
    Ni_z_face4_L = -polynomialProduct(Ni_z_face4_L, z)

    Ni_z_face4_N = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_N[2, 1] = -1
    Ni_z_face4_P = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_P[1, 2] = -1

    Ni_z_face4_Q = np.zeros((5, 5), dtype=np.float)
    Ni_z_face4_Q[1, 1] = 1
    Ni_z_face4_Q = polynomialProduct(Ni_z_face4_Q, z)

    Ni_z_face4 = np.hstack((null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            null_values, null_values, null_values, null_values,
                            Ni_z_face4_c1, Ni_z_face4_c2, Ni_z_face4_c3,
                            Ni_z_face4_c4, Ni_z_face4_c5, Ni_z_face4_c6,
                            Ni_z_face4_c7, Ni_z_face4_c8, Ni_z_face4_c9,
                            Ni_z_face4_c10, null_values, null_values,
                            Ni_z_face4_F, Ni_z_face4_G, Ni_z_face4_H,
                            Ni_z_face4_I, null_values, Ni_z_face4_K,
                            Ni_z_face4_L, null_values, Ni_z_face4_N,
                            null_values, Ni_z_face4_P, Ni_z_face4_Q,
                            null_values))

    # As a previous step to the integration, we must obtain the matrices
    # of: n(normal to the face) x Ni(particularized in the face). Then, we
    # can make the scalar product with the corresponding q's.
    null_values2 = np.zeros((5, 225), dtype=np.float)

    n_face1xNi_face1 = np.vstack((Ni_y_face1, -Ni_x_face1, null_values2))
    n_face2xNi_face2 = np.vstack((null_values2, Ni_z_face2, -Ni_y_face2))
    n_face3xNi_face3 = np.vstack((-Ni_z_face3, null_values2, Ni_x_face3))
    n_face4xNi_face4 = np.vstack((Ni_z_face4 - Ni_y_face4,
                                  Ni_x_face4 - Ni_z_face4,
                                  Ni_y_face4 - Ni_x_face4))

    n_facesxNi = np.vstack((n_face1xNi_face1, n_face2xNi_face2,
                            n_face3xNi_face3, n_face4xNi_face4))

    for face in np.arange(1, 5):
        for q in np.arange(1, 3):
            for function_N in np.arange(1, 4):
                # Three components of n_face(face)xNi particularized to face
                indx1 = (face-1)*15+1
                indx2 = (face*15)
                n_facexNi_face = n_facesxNi[indx1-1:indx2, :]

                # Auxiliar for integral
                integral_aux = np.zeros((5, 225), dtype=np.float)

                for icomponent in np.arange(1, 4):
                    # Development of what I have to integrate in each face
                    # for each row of the matrix of the system of equations.
                    # n_facexNi_x*q_x+n_facexNi_y*q_y+n_facexNi_z*q_z

                    # Each of the parameters and counters involved in the
                    # call to componentProductPolynomail are:
                    # n_facexNi (((icomponent-1)*5+1):(5*icomponent),:): One
                    # of the 3 components of n_facexNi in the face that
                    # concerns us.
                    # Functions_q(:,2*function-1:2*function): The function
                    # (2x2 matrix) that at that moment is multiplying to q.
                    # q_faces_ref(icomponent,2*(face-1)+q): One of the 3
                    # components of the vector q.
                    v1 = n_facexNi_face[((icomponent-1)*5+1)-1:
                                        (5*icomponent), :]
                    v2 = q_functions[:, (2*function_N-1)-1:2*function_N]
                    v3 = q_faces_ref[icomponent-1, (2*(face-1)+q)-1]
                    integral_aux += componentProductPolynomial(v1, v2,
                                                               v3, face)

                # Get row of the coefficient matrix of the system of equations.
                # These equations are those that come from the imposition of
                # the dofs of the edges.
                row_matrix = 19+(face-1)*6+(function_N-1)*2+(q-1)
                matrix[row_matrix-1, :] = analyticIntegral(integral_aux)

    # Integrals over volume
    # Rows 43,44, and 45 of the coefficient matrix of the system of equations
    # are linked to the definition of the dofs associated with the volume.
    # We need again a matrix zeros(5,5) for the coefficients that do not
    # appear in the definition of Ni and that can not be ignored so that when
    # stacking matrices all result in the same size. We use "null_values"
    # already defined for face integrals.
    # For the computation of the volume integral, we perform the integral of
    # Ni = f(x,y,z) between z(x,y)=0 and z(x,y)=1-xy, face 4 in the
    # reference element. For this, we must integrate in z, which we do
    # manually and then with the help of "polynomialProduct"
    # develop all the terms. We will define first the term z^4(z_4)
    # that we will need. The terms z(z), z^2(z_2) and z^3(z_3) have been
    # defined previously for the integrals on face 4.
    z_4 = polynomialProduct(z_3, z)

    # Definition of x-component of z-integral in Ni
    Int_en_z_Ni_x_a1 = z

    Int_en_z_Ni_x_a2 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_a2[1, 0] = 1
    Int_en_z_Ni_x_a2 = polynomialProduct(Int_en_z_Ni_x_a2, z)

    Int_en_z_Ni_x_a3 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_a3[0, 1] = 1
    Int_en_z_Ni_x_a3 = polynomialProduct(Int_en_z_Ni_x_a3, z)

    Int_en_z_Ni_x_a4 = 1/2*z_2

    Int_en_z_Ni_x_a5 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_a5[2, 0] = 1
    Int_en_z_Ni_x_a5 = polynomialProduct(Int_en_z_Ni_x_a5, z)

    Int_en_z_Ni_x_a6 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_a6[0, 2] = 1
    Int_en_z_Ni_x_a6 = polynomialProduct(Int_en_z_Ni_x_a6, z)

    Int_en_z_Ni_x_a7 = 1/3*z_3

    Int_en_z_Ni_x_a8 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_a8[1, 1] = 1
    Int_en_z_Ni_x_a8 = polynomialProduct(Int_en_z_Ni_x_a8, z)

    Int_en_z_Ni_x_a9 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_a9[1, 0] = 1
    Int_en_z_Ni_x_a9 = 1/2*polynomialProduct(Int_en_z_Ni_x_a9, z_2)

    Int_en_z_Ni_x_a10 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_a10[0, 1] = 1
    Int_en_z_Ni_x_a10 = 1/2*polynomialProduct(Int_en_z_Ni_x_a10, z_2)

    Int_en_z_Ni_x_D = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_D[2, 1] = 1
    Int_en_z_Ni_x_D = polynomialProduct(Int_en_z_Ni_x_D, z)

    Int_en_z_Ni_x_E = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_E[0, 3] = 1
    Int_en_z_Ni_x_E = -polynomialProduct(Int_en_z_Ni_x_E, z)

    Int_en_z_Ni_x_F = -1/4*z_4

    Int_en_z_Ni_x_G = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_G[2, 0] = 1
    Int_en_z_Ni_x_G = 1/2*polynomialProduct(Int_en_z_Ni_x_G, z_2)

    Int_en_z_Ni_x_J = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_J[1, 2] = 1
    Int_en_z_Ni_x_J = polynomialProduct(Int_en_z_Ni_x_J, z)

    Int_en_z_Ni_x_K = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_K[1, 0] = 1
    Int_en_z_Ni_x_K = 1/3*polynomialProduct(Int_en_z_Ni_x_K, z_3)

    Int_en_z_Ni_x_M = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_M[1, 1] = 1
    Int_en_z_Ni_x_M = 1/2*polynomialProduct(Int_en_z_Ni_x_M, z_2)

    Int_en_z_Ni_x_O = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_O[0, 2] = 1
    Int_en_z_Ni_x_O = -1/2*polynomialProduct(Int_en_z_Ni_x_O, z_2)

    Int_en_z_Ni_x_P = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_P[0, 2] = 1
    Int_en_z_Ni_x_P = 1/2*polynomialProduct(Int_en_z_Ni_x_P, z_2)

    Int_en_z_Ni_x_Q = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_Q[0, 1] = 1
    Int_en_z_Ni_x_Q = -1/3*polynomialProduct(Int_en_z_Ni_x_Q, z_3)

    Int_en_z_Ni_x_R = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_x_R[0, 1] = 1
    Int_en_z_Ni_x_R = 1/3*polynomialProduct(Int_en_z_Ni_x_R, z_3)

    Int_en_z_Ni_x = np.hstack((Int_en_z_Ni_x_a1, Int_en_z_Ni_x_a2,
                               Int_en_z_Ni_x_a3, Int_en_z_Ni_x_a4,
                               Int_en_z_Ni_x_a5, Int_en_z_Ni_x_a6,
                               Int_en_z_Ni_x_a7, Int_en_z_Ni_x_a8,
                               Int_en_z_Ni_x_a9, Int_en_z_Ni_x_a10,
                               null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, null_values, Int_en_z_Ni_x_D,
                               Int_en_z_Ni_x_E, Int_en_z_Ni_x_F,
                               Int_en_z_Ni_x_G, null_values, null_values,
                               Int_en_z_Ni_x_J, Int_en_z_Ni_x_K, null_values,
                               Int_en_z_Ni_x_M, null_values, Int_en_z_Ni_x_O,
                               Int_en_z_Ni_x_P, Int_en_z_Ni_x_Q,
                               Int_en_z_Ni_x_R))

    # Definition of y-component of z-integral in Ni
    Int_en_z_Ni_y_b1 = z

    Int_en_z_Ni_y_b2 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_b2[1, 0] = 1
    Int_en_z_Ni_y_b2 = polynomialProduct(Int_en_z_Ni_y_b2, z)

    Int_en_z_Ni_y_b3 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_b3[0, 1] = 1
    Int_en_z_Ni_y_b3 = polynomialProduct(Int_en_z_Ni_y_b3, z)

    Int_en_z_Ni_y_b4 = 1/2*z_2

    Int_en_z_Ni_y_b5 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_b5[2, 0] = 1
    Int_en_z_Ni_y_b5 = polynomialProduct(Int_en_z_Ni_y_b5, z)

    Int_en_z_Ni_y_b6 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_b6[0, 2] = 1
    Int_en_z_Ni_y_b6 = polynomialProduct(Int_en_z_Ni_y_b6, z)

    Int_en_z_Ni_y_b7 = 1/3*z_3

    Int_en_z_Ni_y_b8 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_b8[1, 1] = 1
    Int_en_z_Ni_y_b8 = polynomialProduct(Int_en_z_Ni_y_b8, z)

    Int_en_z_Ni_y_b9 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_b9[1, 0] = 1
    Int_en_z_Ni_y_b9 = 1/2*polynomialProduct(Int_en_z_Ni_y_b9, z_2)

    Int_en_z_Ni_y_b10 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_b10[0, 1] = 1
    Int_en_z_Ni_y_b10 = 1/2*polynomialProduct(Int_en_z_Ni_y_b10, z_2)

    Int_en_z_Ni_y_D = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_D[3, 0] = 1
    Int_en_z_Ni_y_D = -polynomialProduct(Int_en_z_Ni_y_D, z)

    Int_en_z_Ni_y_E = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_E[1, 2] = 1
    Int_en_z_Ni_y_E = polynomialProduct(Int_en_z_Ni_y_E, z)

    Int_en_z_Ni_y_H = -1/4*z_4

    Int_en_z_Ni_y_I = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_I[0, 2] = 1
    Int_en_z_Ni_y_I = 1/2*polynomialProduct(Int_en_z_Ni_y_I, z_2)

    Int_en_z_Ni_y_J = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_J[2, 1] = 1
    Int_en_z_Ni_y_J = -polynomialProduct(Int_en_z_Ni_y_J, z)

    Int_en_z_Ni_y_L = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_L[0, 1] = 1
    Int_en_z_Ni_y_L = 1/3*polynomialProduct(Int_en_z_Ni_y_L, z_3)

    Int_en_z_Ni_y_M = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_M[2, 0] = 1
    Int_en_z_Ni_y_M = -1/2*polynomialProduct(Int_en_z_Ni_y_M, z_2)

    Int_en_z_Ni_y_N = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_N[2, 0] = 1
    Int_en_z_Ni_y_N = 1/2*polynomialProduct(Int_en_z_Ni_y_N, z_2)

    Int_en_z_Ni_y_O = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_O[1, 1] = 1
    Int_en_z_Ni_y_O = 1/2*polynomialProduct(Int_en_z_Ni_y_O, z_2)

    Int_en_z_Ni_y_R = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_y_R[1, 0] = 1
    Int_en_z_Ni_y_R = -1/3*polynomialProduct(Int_en_z_Ni_y_R, z_3)

    Int_en_z_Ni_y = np.hstack((null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, Int_en_z_Ni_y_b1, Int_en_z_Ni_y_b2,
                               Int_en_z_Ni_y_b3, Int_en_z_Ni_y_b4,
                               Int_en_z_Ni_y_b5, Int_en_z_Ni_y_b6,
                               Int_en_z_Ni_y_b7, Int_en_z_Ni_y_b8,
                               Int_en_z_Ni_y_b9, Int_en_z_Ni_y_b10,
                               null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, Int_en_z_Ni_y_D, Int_en_z_Ni_y_E,
                               null_values, null_values, Int_en_z_Ni_y_H,
                               Int_en_z_Ni_y_I, Int_en_z_Ni_y_J, null_values,
                               Int_en_z_Ni_y_L, Int_en_z_Ni_y_M,
                               Int_en_z_Ni_y_N, Int_en_z_Ni_y_O, null_values,
                               null_values, Int_en_z_Ni_y_R))

    # Definition of z-component of z-integral in Ni
    Int_en_z_Ni_z_c1 = z

    Int_en_z_Ni_z_c2 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_c2[1, 0] = 1
    Int_en_z_Ni_z_c2 = polynomialProduct(Int_en_z_Ni_z_c2, z)

    Int_en_z_Ni_z_c3 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_c3[0, 1] = 1
    Int_en_z_Ni_z_c3 = polynomialProduct(Int_en_z_Ni_z_c3, z)

    Int_en_z_Ni_z_c4 = 1/2*z_2

    Int_en_z_Ni_z_c5 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_c5[2, 0] = 1
    Int_en_z_Ni_z_c5 = polynomialProduct(Int_en_z_Ni_z_c5, z)

    Int_en_z_Ni_z_c6 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_c6[0, 2] = 1
    Int_en_z_Ni_z_c6 = polynomialProduct(Int_en_z_Ni_z_c6, z)

    Int_en_z_Ni_z_c7 = 1/3*z_3

    Int_en_z_Ni_z_c8 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_c8[1, 1] = 1
    Int_en_z_Ni_z_c8 = polynomialProduct(Int_en_z_Ni_z_c8, z)

    Int_en_z_Ni_z_c9 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_c9[1, 0] = 1
    Int_en_z_Ni_z_c9 = 1/2*polynomialProduct(Int_en_z_Ni_z_c9, z_2)

    Int_en_z_Ni_z_c10 = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_c10[0, 1] = 1
    Int_en_z_Ni_z_c10 = 1/2*polynomialProduct(Int_en_z_Ni_z_c10, z_2)

    Int_en_z_Ni_z_F = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_F[1, 0] = 1
    Int_en_z_Ni_z_F = 1/3*polynomialProduct(Int_en_z_Ni_z_F, z_3)

    Int_en_z_Ni_z_G = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_G[3, 0] = 1
    Int_en_z_Ni_z_G = -polynomialProduct(Int_en_z_Ni_z_G, z)

    Int_en_z_Ni_z_H = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_H[0, 1] = 1
    Int_en_z_Ni_z_H = 1/3*polynomialProduct(Int_en_z_Ni_z_H, z_3)

    Int_en_z_Ni_z_I = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_I[0, 3] = 1
    Int_en_z_Ni_z_I = -polynomialProduct(Int_en_z_Ni_z_I, z)

    Int_en_z_Ni_z_K = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_K[2, 0] = 1
    Int_en_z_Ni_z_K = -1/2*polynomialProduct(Int_en_z_Ni_z_K, z_2)

    Int_en_z_Ni_z_L = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_L[0, 2] = 1
    Int_en_z_Ni_z_L = -1/2*polynomialProduct(Int_en_z_Ni_z_L, z_2)

    Int_en_z_Ni_z_N = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_N[2, 1] = 1
    Int_en_z_Ni_z_N = -polynomialProduct(Int_en_z_Ni_z_N, z)

    Int_en_z_Ni_z_P = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_P[1, 2] = 1
    Int_en_z_Ni_z_P = -polynomialProduct(Int_en_z_Ni_z_P, z)

    Int_en_z_Ni_z_Q = np.zeros((5, 5), dtype=np.float)
    Int_en_z_Ni_z_Q[1, 1] = 1
    Int_en_z_Ni_z_Q = 1/2*polynomialProduct(Int_en_z_Ni_z_Q, z_2)

    Int_en_z_Ni_z = np.hstack((null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, null_values, null_values,
                               null_values, null_values, Int_en_z_Ni_z_c1,
                               Int_en_z_Ni_z_c2, Int_en_z_Ni_z_c3,
                               Int_en_z_Ni_z_c4, Int_en_z_Ni_z_c5,
                               Int_en_z_Ni_z_c6, Int_en_z_Ni_z_c7,
                               Int_en_z_Ni_z_c8, Int_en_z_Ni_z_c9,
                               Int_en_z_Ni_z_c10, null_values, null_values,
                               Int_en_z_Ni_z_F, Int_en_z_Ni_z_G,
                               Int_en_z_Ni_z_H, Int_en_z_Ni_z_I, null_values,
                               Int_en_z_Ni_z_K, Int_en_z_Ni_z_L, null_values,
                               Int_en_z_Ni_z_N, null_values, Int_en_z_Ni_z_P,
                               Int_en_z_Ni_z_Q, null_values))

    # Build matrix of volume
    Int_en_z_Ni = np.vstack((Int_en_z_Ni_x, Int_en_z_Ni_y, Int_en_z_Ni_z))

    # Volumetric integral
    for q in np.arange(1, 4):
        integral_aux = np.zeros((5, 225), dtype=np.float)
        for idof in np.arange(1, 4):
            idx1 = ((idof-1)*5+1)
            idx2 = (5*idof)
            integral_aux = (integral_aux+Int_en_z_Ni[idx1-1:idx2, :] *
                            q_volumen_ref[idof-1, q-1])

        # Get row of the coefficient matrix of the system of equations.
        row_matrix = 43+(q-1)
        matrix[row_matrix-1, :] = analyticIntegral(integral_aux)

    # Computation of the second member of the total equation system. Each
    # column represents the independent terms in the second member of the
    # system of equations proposed for each Ni function.
    rhs = np.zeros((thirdOrderEdgeElement, thirdOrderEdgeElement),
                   dtype=np.float)

    for idof in np.arange(1, thirdOrderEdgeElement+1):
        if (idof <= 18):
            idx1 = np.int(np.round((idof+1)/3)) - np.int(1)
            rhs[idof-1, idof-1] = 1/length[idx1]
        elif (idof > 18):
            rhs[idof-1, idof-1] = 1

    # Obtaining the coefficients of the Ni by solving the system of equations:
    # matrix * {Coefficients} = rhs
    # We call "Coefficients" the matrix that hosts each of the 45 coefficients
    # a1, ... R of the Ni functions. Each of the 45 columns of said matrix
    # therefore represents each of the functions.
    coef = lstsq(matrix, rhs, rcond=None)[0]

    # Coefficients EPS --> 0
    EPS = 1.e-14
    aux1, aux2 = np.where(np.abs(coef) < EPS)
    size = aux1.shape
    n = size[0]

    for kk in np.arange(n):
        coef[aux1[kk], aux2[kk]] = 0.

    a1 = coef[0, :]
    a2 = coef[1, :]
    a3 = coef[2, :]
    a4 = coef[3, :]
    a5 = coef[4, :]
    a6 = coef[5, :]
    a7 = coef[6, :]
    a8 = coef[7, :]
    a9 = coef[8, :]
    a10 = coef[9, :]

    b1 = coef[10, :]
    b2 = coef[11, :]
    b3 = coef[12, :]
    b4 = coef[13, :]
    b5 = coef[14, :]
    b6 = coef[15, :]
    b7 = coef[16, :]
    b8 = coef[17, :]
    b9 = coef[18, :]
    b10 = coef[19, :]

    c1 = coef[20, :]
    c2 = coef[21, :]
    c3 = coef[22, :]
    c4 = coef[23, :]
    c5 = coef[24, :]
    c6 = coef[25, :]
    c7 = coef[26, :]
    c8 = coef[27, :]
    c9 = coef[28, :]
    c10 = coef[29, :]

    D = coef[30, :]
    E = coef[31, :]
    F = coef[32, :]
    G = coef[33, :]
    H = coef[34, :]
    II = coef[35, :]
    JJ = coef[36, :]
    K = coef[37, :]
    L = coef[38, :]
    M = coef[39, :]
    N = coef[40, :]
    OO = coef[41, :]
    P = coef[42, :]
    Q = coef[43, :]
    R = coef[44, :]

    return (a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6,
            b7, b8, b9, b10, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, D, E,
            F, G, H, II, JJ, K, L, M, N, OO, P, Q, R)


def NiTetrahedralThirdOrder(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                            b1, b2, b3, b4, b5, b6, b7, b8, b9, b10,
                            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10,
                            D, E, F, G, H, II, JJ, K, L, M, N, OO, P, Q,
                            R, x, y, z, r):
    ''' Computation of Ni (Nedelec basis functions of second order) in a
    tetrahedral element with vertices (x,y,z) for point r.

    :param float coefficients: a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4,D,E,F,
     G,H,II,JJ,K.
    :param float-array x: x-coordinates of reference element.
    :param float-array y: y-coordinates of reference element.
    :param float-array z: z-coordinates of reference element.
    :param float-array r: xyz coordinates of the evaluation point.
    :return: basis nedelec funcions of second order.
    :rtype: ndarray.

    .. note: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    '''
    # Number of dimensions
    nDimensions = 3

    # Third order ele
    thirdOrderEdgeElement = 45

    # Initialization
    coef = np.zeros((thirdOrderEdgeElement, thirdOrderEdgeElement),
                    dtype=np.float)

    coef[0, :] = a1
    coef[1, :] = a2
    coef[2, :] = a3
    coef[3, :] = a4
    coef[4, :] = a5
    coef[5, :] = a6
    coef[6, :] = a7
    coef[7, :] = a8
    coef[8, :] = a9
    coef[9, :] = a10
    coef[10, :] = b1
    coef[11, :] = b2
    coef[12, :] = b3
    coef[13, :] = b4
    coef[14, :] = b5
    coef[15, :] = b6
    coef[16, :] = b7
    coef[17, :] = b8
    coef[18, :] = b9
    coef[19, :] = b10
    coef[20, :] = c1
    coef[21, :] = c2
    coef[22, :] = c3
    coef[23, :] = c4
    coef[24, :] = c5
    coef[25, :] = c6
    coef[26, :] = c7
    coef[27, :] = c8
    coef[28, :] = c9
    coef[29, :] = c10
    coef[30, :] = D
    coef[31, :] = E
    coef[32, :] = F
    coef[33, :] = G
    coef[34, :] = H
    coef[35, :] = II
    coef[36, :] = JJ
    coef[37, :] = K
    coef[38, :] = L
    coef[39, :] = M
    coef[40, :] = N
    coef[41, :] = OO
    coef[42, :] = P
    coef[43, :] = Q
    coef[44, :] = R

    # Computation on reference element
    L = cartesianToVolumetricCoordinates(x, y, z, r)

    xref = np.array([0, 1, 0, 0], dtype=np.float)
    yref = np.array([0, 0, 1, 0], dtype=np.float)
    zref = np.array([0, 0, 0, 1], dtype=np.float)

    rref = np.zeros(3, dtype=np.float)
    rref[0] = np.dot(L, xref)
    rref[1] = np.dot(L, yref)
    rref[2] = np.dot(L, zref)

    aux_x = np.array([1, rref[0], rref[1], rref[2],  rref[0]**2, rref[1]**2,
                     rref[2]**2, rref[0]*rref[1], rref[0]*rref[2],
                     rref[1]*rref[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, (rref[0]**2)*rref[1], -rref[1]**3,
                     -rref[2]**3, (rref[0]**2)*rref[2], 0, 0,
                     (rref[1]**2)*rref[0],  (rref[2]**2)*rref[0], 0,
                     rref[0]*rref[1]*rref[2], 0, -(rref[1]**2)*rref[2],
                     (rref[1]**2)*rref[2], -(rref[2]**2)*rref[1],
                     (rref[2]**2)*rref[1]], dtype=np.float)

    aux_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, rref[0], rref[1],
                     rref[2], rref[0]**2, rref[1]**2, rref[2]**2,
                     rref[0]*rref[1], rref[0]*rref[2], rref[1]*rref[2],
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -rref[0]**3,
                     (rref[1]**2)*rref[0], 0, 0, -rref[2]**3,
                     (rref[1]**2)*rref[2], -(rref[0]**2)*rref[1], 0,
                     (rref[2]**2)*rref[1], -(rref[0]**2)*rref[2],
                     (rref[0]**2)*rref[2], rref[0]*rref[1]*rref[2], 0, 0,
                     -(rref[2]**2)*rref[0]], dtype=np.float)

    aux_z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 1, rref[0], rref[1], rref[2], rref[0]**2,
                      rref[1]**2, rref[2]**2, rref[0]*rref[1], rref[0]*rref[2],
                      rref[1]*rref[2], 0, 0, (rref[2]**2)*rref[0],
                      -rref[0]**3, (rref[2]**2)*rref[1], -rref[1]**3, 0,
                      -(rref[0]**2)*rref[2], -(rref[1]**2)*rref[2], 0,
                      -(rref[0]**2)*rref[1], 0, -(rref[1]**2)*rref[0],
                      rref[0]*rref[1]*rref[2], 0], dtype=np.float)

    # Allocate
    Niref = np.zeros((nDimensions, thirdOrderEdgeElement), dtype=np.float)

    Niref[0, :] = np.matmul(aux_x, coef)
    Niref[1, :] = np.matmul(aux_y, coef)
    Niref[2, :] = np.matmul(aux_z, coef)

    del aux_x, aux_y, aux_z

    # Vector element reference to the real element through the Jacobian
    # Ni_real=([J]^-1)*Niref
    x21 = x[1]-x[0]
    y21 = y[1]-y[0]
    z21 = z[1]-z[0]
    x31 = x[2]-x[0]
    y31 = y[2]-y[0]
    z31 = z[2]-z[0]
    x41 = x[3]-x[0]
    y41 = y[3]-y[0]
    z41 = z[3]-z[0]

    jacob = np.array([[x21, y21, z21], [x31, y31, z31],
                      [x41, y41, z41]], dtype=np.float)

    Ni = lstsq(jacob, Niref, rcond=None)[0]

    return Ni


def nedelecBasisThirdOrder(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                           b1, b2, b3, b4, b5, b6, b7, b8, b9, b10,
                           c1, c2, c3, c4, c5, c6, c7, c8, c9, c10,
                           D, E, F, G, H, II, JJ, K, L, M, N,
                           O, P, Q, R, ref_ele, points):
    ''' This function computes the basis nedelec functions of third order
    for a set of points in a  given tetrahedral element.

    :param float coefficients: a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,b1,b2,b3,b4,b5,
     b6,b7,b8,b9,b10,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,D,E,F,G,H,II,JJ,K,L,M,N,
     O,P,Q,R.
    :param float-array ref_ele: nodal coordinates of reference element.
    :param float-array points: spatial coordinates of the evaluation points.
    :return: basis nedelec functions of second order.
    :rtype: ndarray.

    .. note: References:\n
       Garcia-Castillo, L. E., Ruiz-Genovés, A. J., Gómez-Revuelto, I.,
       Salazar-Palma, M., & Sarkar, T. K. (2002). Third-order Nédélec
       curl-conforming finite element. IEEE transactions on magnetics,
       38(5), 2370-2372.
    '''
    # Number of points
    size = points.shape
    # More than one point
    if len(size) == 2:
        nPoints = size[1]
    # One point
    elif len(size) == 1:
        nPoints = 1

    # Number of dimensions
    nDimensions = 3

    # Third order ele
    thirdOrderEdgeElement = 45

    # Allocate
    basis = np.zeros((nDimensions, thirdOrderEdgeElement, nPoints),
                     dtype=np.float)

    # Get reference element coordinates
    xref = ref_ele[0, :]
    yref = ref_ele[1, :]
    zref = ref_ele[2, :]

    # Compute nedelec basis functions of second order for all points
    if nPoints == 1:    # One point
        for iPoint in np.arange(nPoints):
            r = points

            # Basis funtions for iPoint|
            basis[:, :, iPoint] = NiTetrahedralThirdOrder(a1, a2, a3, a4, a5,
                                                          a6, a7, a8, a9, a10,
                                                          b1, b2, b3, b4, b5,
                                                          b6, b7, b8, b9, b10,
                                                          c1, c2, c3, c4, c5,
                                                          c6, c7, c8, c9, c10,
                                                          D, E, F, G, H, II,
                                                          JJ, K, L, M, N, O,
                                                          P, Q, R, xref, yref,
                                                          zref, r)
    else:    # More than one point
        for iPoint in np.arange(nPoints):
            r = points[:, iPoint]

            # Basis funtions for iPoint|
            basis[:, :, iPoint] = NiTetrahedralThirdOrder(a1, a2, a3, a4, a5,
                                                          a6, a7, a8, a9, a10,
                                                          b1, b2, b3, b4, b5,
                                                          b6, b7, b8, b9, b10,
                                                          c1, c2, c3, c4, c5,
                                                          c6, c7, c8, c9, c10,
                                                          D, E, F, G, H, II,
                                                          JJ, K, L, M, N, O,
                                                          P, Q, R, xref, yref,
                                                          zref, r)

    return basis


def computeMassMatrixThirdOrder(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                                b1, b2, b3, b4, b5, b6, b7, b8, b9, b10,
                                c1, c2, c3, c4, c5, c6, c7, c8, c9, c10,
                                D, E, F, G, H, II, JJ, K, L, M, N, O, P,
                                Q, R, ref_ele, GR, signs, DetJacob):
    ''' Compute mass matrix for tetrahedral edge elements of third order.

    :param float coefficients: a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,b1,b2,b3,b4,b5,
     b6,b7,b8,b9,b10,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,D,E,F,G,H,II,JJ,K,L,M,N,
     O,P,Q,R
    :param float-array ref_ele: nodal coordinates of reference element.
    :param float-array GR: tensor.
    :param int-array signs: dofs signs.
    :param float DetJacob: determinant of the jacobian.
    :return: mass matrix for edge element of third order.
    :rtype: ndarray.

    .. note: References:\n
       Garcia-Castillo, L. E., Ruiz-Genovés, A. J., Gómez-Revuelto, I.,
       Salazar-Palma, M., & Sarkar, T. K. (2002). Third-order Nédélec
       curl-conforming finite element. IEEE transactions on magnetics,
       38(5), 2370-2372.
    '''
    # Third order ele
    nedelecOrder = 3
    thirdOrderEdgeElement = 45

    # Gaussian points for the unit reference tetrahedron
    ngaussP = 24
    [Wi, rx, ry, rz] = gauss_points_reference_tetrahedron(ngaussP,
                                                          nedelecOrder)

    # Get reference element coordinates
    xref = ref_ele[0, :]
    yref = ref_ele[1, :]
    zref = ref_ele[2, :]

    # Allocate
    Me = np.zeros((thirdOrderEdgeElement, thirdOrderEdgeElement),
                  dtype=np.float)

    # Mass matrix computation
    r = np.zeros(3, dtype=np.float)
    for iPoint in np.arange(ngaussP):
        r[0] = rx[iPoint]
        r[1] = ry[iPoint]
        r[2] = rz[iPoint]

        Ni = NiTetrahedralThirdOrder(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                                     b1, b2, b3, b4, b5, b6, b7, b8, b9, b10,
                                     c1, c2, c3, c4, c5, c6, c7, c8, c9, c10,
                                     D, E, F, G, H, II, JJ, K, L, M, N, O,
                                     P, Q, R, xref, yref, zref, r)
        nix = Ni[0, :]
        niy = Ni[1, :]
        niz = Ni[2, :]

        for ii in np.arange(thirdOrderEdgeElement):
            for jj in np.arange(thirdOrderEdgeElement):
                Me[ii, jj] += Wi[iPoint]*((GR[0, 0]*nix[ii]*nix[jj] +
                                           GR[0, 1]*(nix[ii]*niy[jj] +
                                                     niy[ii]*nix[jj]) +
                                           GR[1, 1]*niy[ii]*niy[jj] +
                                           GR[0, 2]*(nix[ii]*niz[jj] +
                                                     niz[ii]*nix[jj]) +
                                           GR[1, 2]*(niy[ii]*niz[jj] +
                                                     niz[ii]*niy[jj]) +
                                           GR[2, 2]*niz[ii]*niz[jj]) *
                                          signs[ii]*signs[jj])

    Me = Me*DetJacob

    return Me


def computeDerivativesThirdOrder(a2, a3, a4, a5, a6, a7, a8, a9, a10, b2, b3,
                                 b4, b5, b6, b7, b8, b9, b10, c2, c3, c4, c5,
                                 c6, c7, c8, c9, c10, D, E, F, G, H, II, JJ,
                                 K, L, M, N, O, P, Q, R, point):
    ''' Compute partial derivatives of basis functions for tetrahedral edge
    elements of third order (reference element)

    :param float coefficients: a2,a3,a4,a5,a6,a7,a8,a9,a10,b2,b3,b4,b5,b6,
     b7,b8,b9,b10,c2,c3,c4,c5,c6,c7,c8,c9,c10,D,E,F,G,H,II,JJ,K,L,M,N,O,P,Q,R.
    :param float-array point: coordinates of the gaussian point.
    :return: partial derivatives for edge element of third order.
    :rtype: ndarray.

    .. note: References:\n
       Garcia-Castillo, L. E., Ruiz-Genovés, A. J., Gómez-Revuelto, I.,
       Salazar-Palma, M., & Sarkar, T. K. (2002). Third-order Nédélec
       curl-conforming finite element. IEEE transactions on magnetics,
       38(5), 2370-2372.
    '''
    # Point coordinates
    x = point[0]
    y = point[1]
    z = point[2]

    # dxNix
    dxNix = (a2 + 2*a5*x + a8*y + a9*z + 2*D*x*y +
             2*G*x*z + JJ*y**2 + K*z**2 + M*y*z)

    # dxNiy
    dxNiy = (b2 + 2*b5*x + b8*y + b9*z - 2*JJ*x*y - 2*M*x*z +
             2*N*x*z + E*y**2 - R*z**2 + O*y*z - 3*D*x**2)

    # dxNiz
    dxNiz = (c2 + 2*c5*x + c8*y + c9*z - 2*N*x*y -
             2*K*x*z - P*y**2 + F*z**2 + Q*y*z - 3*G*x**2)

    # dyNix
    dyNix = (a3 + 2*a6*y + a8*x + a10*z + D*x**2 + 2*JJ*y*x -
             2*O*y*z + 2*P*y*z - Q*z**2 + R*z**2 + M*x*z - 3*E*y**2)

    # dyNiy
    dyNiy = (b3 + 2*b6*y + b8*x + b10*z - JJ*x**2 +
             2*E*y*x + 2*II*y*z + L*z**2 + O*x*z)

    # dyNiz
    dyNiz = (c3 + 2*c6*y + c8*x + c10*z - N*x**2 -
             2*P*y*x - 2*L*y*z + H*z**2 + Q*x*z - 3*II*y**2)

    # dzNix
    dzNix = (a4 + 2*a7*z + a9*x + a10*y + G*x**2 - O*y**2 +
             P*y**2 + 2*K*z*x - 2*Q*z*y + 2*R*z*y + M*x*y - 3*F*z**2)

    # dzNiy
    dzNiy = (b4 + 2*b7*z + b9*x + b10*y - M*x**2 +
             N*x**2 + II*y**2 - 2*R*z*x + 2*L*z*y + O*x*y - 3*H*z**2)

    # dzNiz
    dzNiz = (c4 + 2*c7*z + c9*x + c10*y - K*x**2 -
             L*y**2 + 2*F*z*x + 2*H*z*y + Q*x*y)

    return dxNix, dxNiy, dxNiz, dyNix, dyNiy, dyNiz, dzNix, dzNiy, dzNiz


def computeStiffnessMatrixThirdOrder(a2, a3, a4, a5, a6, a7, a8, a9, a10,
                                     b2, b3, b4, b5, b6, b7, b8, b9, b10,
                                     c2, c3, c4, c5, c6, c7, c8, c9, c10,
                                     D, E, F, G, H, II, JJ, K, L, M, N,
                                     O, P, Q, R, invjj, signs, DetJacob):
    ''' Compute stiffness matrix for tetrahedral edge elements of third order.

    :param float coefficients: a2,a3,a4,a5,a6,a7,a8,a9,a10,b2,b3,b4,b5,b6,b7,
     b8,b9,b10,c2,c3,c4,c5,c6,c7,c8,c9,c10,D,E,F,G,H,II,JJ,K,L,M,N,O,P,Q,R.
    :param float-array invjj: inverse jacobian.
    :param int-array signs: dofs signs.
    :param float DetJacob: determinant of the jacobian.
    :return: stiffness matrix for edge element of third order.
    :rtype: ndarray.

    .. note: References:\n
       Garcia-Castillo, L. E., Ruiz-Genovés, A. J., Gómez-Revuelto, I.,
       Salazar-Palma, M., & Sarkar, T. K. (2002). Third-order Nédélec
       curl-conforming finite element. IEEE transactions on magnetics,
       38(5), 2370-2372.
    '''
    # Third order ele
    nedelecOrder = 3
    thirdOrderEdgeElement = 45

    # Gaussian points for the unit reference tetrahedron
    ngaussP = 15
    [Wi, rx, ry, rz] = gauss_points_reference_tetrahedron(ngaussP,
                                                          nedelecOrder)

    # Allocate
    Ke = np.zeros((thirdOrderEdgeElement, thirdOrderEdgeElement),
                  dtype=np.float)

    # Tensor computation
    fr = np.eye(3, dtype=np.float)
    invfr = inv(fr)

    tmp1 = np.array([[0, 0, 0], [0, 0, 1], [0, -1,  0]], dtype=np.float)
    tmp2 = np.matmul(tmp1, invjj)
    A = np.matmul(invjj.transpose(), tmp2)

    tmp1 = np.array([[0, 0, -1], [0, 0, 0], [1, 0,  0]], dtype=np.float)
    tmp2 = np.matmul(tmp1, invjj)
    B = np.matmul(invjj.transpose(), tmp2)

    tmp1 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0,  0]], dtype=np.float)
    tmp2 = np.matmul(tmp1, invjj)
    C = np.matmul(invjj.transpose(), tmp2)

    # Stiffness matrix computation
    r = np.zeros(3, dtype=np.float)
    for iPoint in np.arange(ngaussP):
        r[0] = rx[iPoint]
        r[1] = ry[iPoint]
        r[2] = rz[iPoint]

        [_, dxNiy, dxNiz,
         dyNix, _, dyNiz,
         dzNix, dzNiy, _] = computeDerivativesThirdOrder(a2, a3, a4, a5, a6,
                                                         a7, a8, a9, a10, b2,
                                                         b3, b4, b5, b6, b7,
                                                         b8, b9, b10, c2, c3,
                                                         c4, c5, c6, c7, c8,
                                                         c9, c10, D, E, F, G,
                                                         H, II, JJ, K, L, M, N,
                                                         O, P, Q, R, r)

        rotNix = (A[0, 1]*dxNiy + A[0, 2]*dxNiz -
                  A[0, 1]*dyNix + A[1, 2]*dyNiz -
                  A[0, 2]*dzNix - A[1, 2]*dzNiy)
        rotNiy = (B[0, 1]*dxNiy + B[0, 2]*dxNiz -
                  B[0, 1]*dyNix + B[1, 2]*dyNiz -
                  B[0, 2]*dzNix - B[1, 2]*dzNiy)
        rotNiz = (C[0, 1]*dxNiy + C[0, 2]*dxNiz -
                  C[0, 1]*dyNix + C[1, 2]*dyNiz -
                  C[0, 2]*dzNix - C[1, 2]*dzNiy)

        for ii in np.arange(thirdOrderEdgeElement):
            for jj in np.arange(thirdOrderEdgeElement):
                Ke[ii, jj] += Wi[iPoint]*((invfr[0, 0]*rotNix[ii]*rotNix[jj] +
                                           invfr[0, 1]*(rotNix[ii] *
                                                        rotNiy[jj] +
                                                        rotNiy[ii] *
                                                        rotNix[jj]) +
                                           invfr[1, 1]*rotNiy[ii]*rotNiy[jj] +
                                           invfr[0, 2]*(rotNix[ii] *
                                                        rotNiz[jj] +
                                                        rotNiz[ii] *
                                                        rotNix[jj]) +
                                           invfr[1, 2]*(rotNiy[ii] *
                                                        rotNiz[jj] +
                                                        rotNiz[ii] *
                                                        rotNiy[jj]) +
                                           invfr[2, 2]*rotNiz[ii] *
                                           rotNiz[jj])*signs[ii]*signs[jj])

    Ke = Ke*DetJacob

    return Ke


def unitary_test():
    ''' Unitary test for efem.py script.
    '''


if __name__ == '__main__':
    # Standard module import
    unitary_test()
else:
    # Standard module import
    import numpy as np
    from numpy.linalg import lstsq
    from scipy.linalg import det
    from scipy.linalg import norm
    from scipy.linalg import inv
    # PETGEM module import
    from petgem.efem.vectorMatrixFunctions import deleteDuplicateRows
    from petgem.efem.vectorMatrixFunctions import crossprod
    from petgem.efem.vectorMatrixFunctions import is_duplicate_entry
    from petgem.efem.fem import cartesianToVolumetricCoordinates
    from petgem.efem.fem import gauss_points_reference_tetrahedron
