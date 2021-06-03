#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
"""Define functions for mesh handling."""

# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import numpy as np

# ---------------------------------------------------------------
# Load petgem modules (BSC)
# ---------------------------------------------------------------
from .vectors import deleteDuplicateRows
from .vectors import invConnectivity


def computeEdges(elemsN, nElems):
    """Compute edges of a 3D tetrahedral mesh.

    :param ndarray elemsN: elements-nodes connectivity.
    :param int nElems: number of tetrahedral elements in the mesh.
    :return: edges connectivity and edgesNodes connectivity.
    :rtype: ndarray
    """
    # Extracts sets of edges
    edges1 = elemsN[:, [0, 1]]
    edges2 = elemsN[:, [1, 2]]
    edges3 = elemsN[:, [0, 2]]
    edges4 = elemsN[:, [0, 3]]
    edges5 = elemsN[:, [1, 3]]
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
    [edgesNodes, edges] = deleteDuplicateRows(vertices)

    # Build dofs matrix
    edges = np.array(np.reshape(edges, (nElems, 6)), dtype=np.int)

    # Build dofs to nodes connectivity
    edgesNodes.sort(axis=1)
    edgesNodes = np.array(edgesNodes, dtype=np.int)

    return edges, edgesNodes


def computeFaces(elemsN, nElems):
    r"""Compute the element\'s faces of a 3D tetrahedral mesh.

    :param ndarray matrix: elements-faces connectivity.
    :param int nElems: number of elements in the mesh.
    :return: element/faces connectivity.
    :rtype: ndarray

    .. note:: References:\n
       Rognes, Marie E., Robert Cndarray. Kirby, and Anders Logg. "Efficient
       assembly of H(div) and H(curl) conforming finite elements."
       SIAM Journal on Scientific Computing 31.6 (2009): 4130-4151.
    """
    # Extracts sets of faces for each nedelec element order
    faces1 = elemsN[:, [0, 1, 2]]
    faces2 = elemsN[:, [0, 1, 3]]
    faces3 = elemsN[:, [1, 2, 3]]
    faces4 = elemsN[:, [0, 2, 3]]

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


def computeFacesEdges(elemsF, elemsE, nFaces, nElems):
    r"""Compute the edge\'s faces connectivity

    :param ndarray elemsF: elements-faces connectivity.
    :param ndarray elemsE: elements-edges connectivity.
    :param int nFaces: number of faces in the mesh.
    :param int nElems: number of elements in the mesh.
    :return: edges/faces connectivity.
    :rtype: ndarray.
    """
    N = invConnectivity(elemsF, nFaces)

    if nElems != 1:
        N = np.delete(N, 0, axis=1)

    # Allocate
    facesE = np.zeros((nFaces, 3), dtype=np.int)

    # Compute edges list for each face
    for i in np.arange(nFaces):
        iEle = N[i, 0]
        edgesEle = elemsE[iEle,:]
        facesEle = elemsF[iEle,:]
        kFace = np.where(facesEle == i)[0]
        if kFace == 0:  # Face 1
            facesE[facesEle[kFace],:] = [edgesEle[0], edgesEle[1], edgesEle[2]]
        elif kFace == 1:  # Face 2
            facesE[facesEle[kFace],:] = [edgesEle[0], edgesEle[4], edgesEle[3]]
        elif kFace == 2:  # Face 3
            facesE[facesEle[kFace],:] = [edgesEle[1], edgesEle[5], edgesEle[4]]
        elif kFace == 3:  # Face 4
            facesE[facesEle[kFace],:] = [edgesEle[2], edgesEle[5], edgesEle[3]]

    return facesE


def computeBoundaryElements(elemsF, bFaces, nFaces):
    r"""Compute boundary elements.

    :param ndarray elemsF: elements-faces connectivity.
    :param ndarray bfaces: indexes of boundary faces.
    :param int nFaces: number of faces in the mesh.
    :return: indexes of boundary elements.
    :rtype: ndarray
    """
    # Get to what element the boundary face belongs
    N = invConnectivity(elemsF, nFaces)
    bndElems = N[bFaces, :]
    bndElems = np.delete(bndElems, 0, axis=1)
    bndElems = np.reshape(bndElems, bndElems.size)

    # Number of boundary elements
    nBoundaryElems = bndElems.shape[0]

    return bndElems, nBoundaryElems


def computeBoundaryFaces(elemsF, facesN):
    """Compute boundary faces of a tetrahedral mesh.

    :param ndarray elemsF: elements-face connectivity.
    :param ndarray facesN: faces-nodes connectivity.
    :return: nodal-connectivity and indexes of boundary-faces, number of boundary faces.
    :rtype: ndarray
    """
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

    return bfacesN, bFaces, nBoundaryFaces


def computeBoundaryEdges(edgesN, bfacesN):
    """Compute boundary edges of a tetrahedral mesh.

    :param ndarray edgesN: edges-nodes connectivity.
    :param ndarray bfacesN: boundary-faces-nodes connectivity.
    :return: boundary-edges connectivity.
    :rtype: ndarray
    """
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
    bEdges = bEdges[0,:]

    return bEdges


def computeBoundaries(dof_connectivity, dof_edges, dof_faces, bEdges, bFaces, Nord):
    """Compute the indexes of dofs boundaries and internal dofs.

    :param ndarray dof_connectivity: local/global dofs list for elements
    :param ndarray dof_edges: dofs index on edges
    :param ndarray dof_faces: dofs index on faces
    :param ndarray bEdges: boundary-edges connectivity with dimensions = (number_boundary_edges,1)
    :param ndarray bfaces: indexes of boundary-faces = (number_boundary_faces, 1)
    :param int Nord: polynomial order of nedelec basis functions
    :return: indexes of internal dofs and indexes of boundary dofs
    :rtype: ndarray
    """
    # Number of boundaries on edges
    nBoundaryEdges = len(bEdges)
    num_dof_in_edge = Nord

    # Number of boundaries on faces
    nBoundaryFaces = len(bFaces)
    num_dof_in_face = Nord*(Nord-1)

    # Get boundary dofs for edges
    indx_boundary_edges = dof_edges[bEdges,:]

    # Get boundary dofs for faces
    if dof_faces.size == 0:
        # No dofs on faces (first order, Nord==1)
        indx_boundary_faces = np.zeros((1,0), dtype=np.int)
    else:
        indx_boundary_faces = dof_faces[bFaces,:]

    # Get indexes of boundary dofs
    tmp1 = np.reshape(indx_boundary_edges, (nBoundaryEdges*num_dof_in_edge))
    tmp2 = np.reshape(indx_boundary_faces, (nBoundaryFaces*num_dof_in_face))
    indx_boundary_dofs = np.hstack((tmp1, tmp2))

    # Get total number of dofs in the mesh
    total_num_dofs = np.max(dof_connectivity) + 1

    # Get indexes of inner dofs
    indx_inner_dofs = np.setdiff1d(np.arange(0,total_num_dofs), indx_boundary_dofs)

    return indx_inner_dofs, indx_boundary_dofs


def computeFacePlane(nodes, bFaces, bFacesN):
    r"""Compute the plane id to which the boundary belongs.

    :param ndarray nodes: nodes coordinates.
    :param ndarray bFaces: indexes of boundary-faces.
    :param ndarray bFacesN: boundary-faces-nodes connectivity.

    :return: plane id to which the boundary belongs.
    :rtype: ndarray

    """
    # Number of boundary faces
    nBndFaces = np.size(bFaces)

    # Get computational domain limits
    min_x = np.min(nodes[:,0])
    max_x = np.max(nodes[:,0])
    min_y = np.min(nodes[:,1])
    max_y = np.max(nodes[:,1])
    min_z = np.min(nodes[:,2])
    max_z = np.max(nodes[:,2])

    # Set plane equation for each side
    # We consider following configuration:

    #         g ------------- h
    #        /|              /|     where:
    #       / |             / |     a = [min_x, min_y, min_z]
    #      /  |            /  |     b = [max_x, min_y, min_z]
    #     /   |           /   |     c = [min_x, max_y, min_z]
    #    e --------------f    |     d = [max_x, max_y, min_z]
    #    |    |          |    |     e = [min_x, min_y, max_z]
    #    |    |c --------|----|d    f = [max_x, min_y, max_z]
    #    |   /           |   /      g = [min_x, max_y, max_z]
    #    |  /            |  /       h = [max_x, max_y, max_z]
    #    | /             | /
    #    a ------------- b

    # With xyz-axis:
    #         Z+   Y+
    #          |  /
    #          | /
    #          |/
    #           ------X+

    #  Therefore, each plane of the cube is defined by two vectors:
    #  Bottom: ab, ac   ---> flag: 0
    #  Left: ac, ae     ---> flag: 1
    #  Front: ab, ae    ---> flag: 2
    #  Rigth: bd, bf    ---> flag: 3
    #  Back: dc, dh     ---> flag: 4
    #  Top: ef, eg      ---> flag: 5

    # Initialize main points
    a = np.array([min_x, min_y, min_z], np.float)
    b = np.array([max_x, min_y, min_z], np.float)
    c = np.array([min_x, max_y, min_z], np.float)
    d = np.array([max_x, max_y, min_z], np.float)
    e = np.array([min_x, min_y, max_z], np.float)
    f = np.array([max_x, min_y, max_z], np.float)
    g = np.array([min_x, max_y, max_z], np.float)
    h = np.array([max_x, max_y, max_z], np.float)

    # Compute normal vectors to planes
    normal_bottom = np.cross(a-b, a-c)
    normal_left   = np.cross(a-c, a-e)
    normal_front  = np.cross(a-b, a-e)
    normal_right  = np.cross(b-d, b-f)
    normal_back   = np.cross(d-c, d-h)
    normal_top    = np.cross(e-f, e-g)

    # Allocate space for tag plane
    planeFace  = np.zeros(nBndFaces, dtype=np.int)
    plane_list = np.zeros(6, dtype=np.float)

    # Solve plane equation for each boundary face
    for i in np.arange(nBndFaces):
        # Get nodes of face
        faceNodes = nodes[bFacesN[:,i],:]

        # Compute face centroid
        centroid = np.sum(faceNodes, 0)/3.

        # Solve equation for bottom plane
        plane_list[0] = np.dot(normal_bottom, centroid-a)
        # Solve equation for left plane
        plane_list[1] = np.dot(normal_left, centroid-a)
        # Solve equation for front plane
        plane_list[2] = np.dot(normal_front, centroid-a)
        # Solve equation for right plane
        plane_list[3] = np.dot(normal_right, centroid-b)
        # Solve equation for back plane
        plane_list[4] = np.dot(normal_back, centroid-d)
        # Solve equation for top plane
        plane_list[5] = np.dot(normal_top, centroid-e)
        # Get to what plane the face belongs
        # Flags for faces:
        # Bottom ---> flag: 0
        # Left   ---> flag: 1
        # Front  ---> flag: 2
        # Rigth  ---> flag: 3
        # Back   ---> flag: 4
        # Top    ---> flag: 5
        planeFace[i] = np.where(np.abs(plane_list)<1.e-13)[0][0]

    return planeFace


def unitary_test():
    """Unitary test for mesh.py script."""


if __name__ == '__main__':
    # Standard module import
    unitary_test()
