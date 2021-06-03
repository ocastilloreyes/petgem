#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
"""Define functions for high-order vector finite element method."""

# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import numpy as np
from .common import Print

# ###############################################################
# ################     FUNCTIONS DEFINITION     #################
# ###############################################################
def computeConnectivityDOFS(elemsE, elemsF, Nord):
    r"""Compute the degrees of freedom connectivity for a given list of edges, faces and elements.

    :param ndarray elemsE: elements-edge connectivity with dimensions = (number_elements, 6)
    :param ndarray elemsF: element/faces connectivity with dimensions = (number_elements, 4)
    :param int Nord: polynomial order of nedelec basis functions
    :return: local/global dofs list for elements, dofs index on edges, dofs index on faces, dofs index on volumes, total number of dofs
    :rtype: ndarray and int

    .. note:: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    """
    # Get number of elements
    nElems = elemsE.shape[0]

    # Get number of edges
    nEdges = np.max(elemsE) + 1

    # Get number of faces
    nFaces = np.max(elemsF) + 1

    # Define number of edges and faces per element
    num_edges_in_element = 6
    num_faces_in_element = 4

    # Define number of dofs per entity
    num_dof_in_edge = Nord
    num_dof_in_face = Nord*(Nord-1)
    num_dof_in_volume = np.int(Nord*(Nord-1)*(Nord-2)/2)
    # Compute number of dofs per element
    num_dof_in_element = np.int(Nord*(Nord+2)*(Nord+3)/2)

    # Compute dofs list for edges, faces and volume
    # Initialize dof counter
    current_dof = 0

    # Compute connectivity for dofs on edges
    dof_edges = np.zeros((nEdges, num_dof_in_edge), dtype=np.int)

    for index_edge in np.arange(nEdges):
        dof_edges[index_edge,:] = np.arange(current_dof, current_dof+num_dof_in_edge)
        current_dof += num_dof_in_edge

    # Compute connectivity for dofs on faces
    dof_faces = np.zeros((nFaces, num_dof_in_face), dtype=np.int)
    for index_face in np.arange(nFaces):
        dof_faces[index_face,:] = np.arange(current_dof, current_dof+num_dof_in_face)
        current_dof += num_dof_in_face

    # Compute connectivity for dofs on volume
    dof_volume = np.zeros((nElems, num_dof_in_volume), dtype=np.int)
    for index_ele in np.arange(nElems):
        dof_volume[index_ele,:] = np.arange(current_dof, current_dof+num_dof_in_volume)
        current_dof += num_dof_in_volume

    # Compute local/global dofs connectivity list
    # Allocate
    dof_connectivity = np.zeros((nElems, num_dof_in_element), dtype=np.int)

    # Compute local/global dofs connectivity
    for i in np.arange(nElems):
        # Initialize dof counter
        dof_counter = 0

        # Get dofs on edges
        for j in np.arange(num_edges_in_element):
            dof_connectivity[i, np.arange(dof_counter, dof_counter+num_dof_in_edge)] = dof_edges[elemsE[i,j],:]
            dof_counter += num_dof_in_edge

        # Get dofs on faces
        for j in np.arange(num_faces_in_element):
            dof_connectivity[i, np.arange(dof_counter, dof_counter+num_dof_in_face)] = dof_faces[elemsF[i,j],:]
            dof_counter += num_dof_in_face

        # Get dofs in volume
        dof_connectivity[i, np.arange(dof_counter, dof_counter+num_dof_in_volume)] = dof_volume[i,:]

    # Get total number of dofs
    total_num_dofs = np.max(dof_connectivity) + 1

    return dof_connectivity, dof_edges, dof_faces, dof_volume, total_num_dofs


def computeJacobian(eleNodes):
    """Compute the jacobian and its inverse.

    :param ndarray eleNodes: spatial coordinates of the nodes with dimensions = (4,3)
    :return: jacobian matrix and its inverse.
    :rtype: ndarray
    """
    # Allocate
    jacobian = np.zeros((3, 3), dtype=np.float)

    # Jacobian computation
    jacobian[0, :] = eleNodes[1, :] - eleNodes[0, :]
    jacobian[1, :] = eleNodes[2, :] - eleNodes[0, :]
    jacobian[2, :] = eleNodes[3, :] - eleNodes[0, :]

    # Inverse of jacobian
    invjacobian = np.linalg.inv(jacobian)

    return jacobian, invjacobian


def computeElementOrientation(edgesEle,nodesEle,edgesNodesEle,globalEdgesInFace):
    r"""Compute the orientation for the computation of hierarchical basis functions of high-order (High-order nédélec basis functions).

    :param ndarray edgesEle:list of element's edges
    :param ndarray nodesEle: list of element's nodes
    :param ndarray edgesNodesEle: list of nodes for each edge in edgesEle
    :param ndarray globalEdgesInFace: list of edges for each face
    :return: orientation for edges and orientation for faces
    :rtype: ndarray.

    .. note:: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Allocation (10 = 6 edges per element + 4 faces per element)
    orientation = np.zeros(10, dtype=np.int)

    # ---------------------------------------------------------------
    # Compute orientation for edges
    # ---------------------------------------------------------------
    NumEdgesInElement = 6

    for i in np.arange(NumEdgesInElement):
        if i == 0:
            localNodes = [0, 1]
        elif i == 1:
            localNodes = [1, 2]
        elif i == 2:
            localNodes = [0, 2]
        elif i == 3:
            localNodes = [0, 3]
        elif i == 4:
            localNodes = [1, 3]
        elif i == 5:
            localNodes = [2, 3]

        nodesEleForThisEdge = nodesEle[localNodes]
        globalNodesInEdge = edgesNodesEle[i, :]

        orientationForThisEdge = 0
        if ((nodesEleForThisEdge[0] == globalNodesInEdge[1]) and (nodesEleForThisEdge[1] == globalNodesInEdge[0])):
            orientationForThisEdge = 1

        orientation[i] = orientationForThisEdge

    # ---------------------------------------------------------------
    # Compute orientation for faces
    # ---------------------------------------------------------------
    NumFacesInElement = 4
    for i in np.arange(NumFacesInElement):
        if i == 0:
            localEdges = [0, 1, 2]
        elif i == 1:
            localEdges = [0, 4, 3]
        elif i == 2:
            localEdges = [1, 5, 4]
        elif i == 3:
            localEdges = [2, 5, 3]

        globalEdgesInElementForThisFace = edgesEle[localEdges]

        orientationForThisFace = 0

        k1 = 0
        k2 = 0

        for k in np.arange(3):
            if (globalEdgesInElementForThisFace[0] == globalEdgesInFace[i, k]):
                k1 = k + 1
            if (globalEdgesInElementForThisFace[1] == globalEdgesInFace[i, k]):
                k2 = k + 1

        selection = k1*10 + k2
        if selection == 12:
            orientationForThisFace = 0
        elif selection == 31:
            orientationForThisFace = 1
        elif selection == 23:
            orientationForThisFace = 2
        elif selection == 32:
            orientationForThisFace = 3
        elif selection == 13:
            orientationForThisFace = 4
        elif selection == 21:
            orientationForThisFace = 5

        orientation[6+i] = orientationForThisFace

    # Edge orientation (6 edges)
    edges_orientation = orientation[0:6]
    # Face orientation (4 faces)
    faces_orientation = orientation[6:11]

    return edges_orientation, faces_orientation


def computeElementalMatrices(edge_orientation, face_orientation, jacobian, invjacob, Nord, sigmaEle):
    r"""Compute the elemental mass matrix and stiffness matrix based ons high-order vector finite element.

    :param ndarray edges_orientation: orientation for edges
    :param ndarray faces_orientation: orientation for faces
    :param ndarray jacobian: jacobian matrix
    :param ndarray invjacob: inverse of jacobian matrix
    :param int Nord: polynomial order of vector basis functions
    :param ndarray sigmaEle: element conductivity with dimensions (1, 2), (horizontal and vertical)
    :return: elemental mass matrix and elemental stiffness matrix
    :rtype: ndarray

    .. note:: References:\n
       Fuentes, F., Keith, B., Demkowicz, L., & Nagaraj, S. (2015). Orientation
       embedded high order shape functions for the exact sequence elements of
       all shapes. Computers & Mathematics with applications, 70(4), 353-458.
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Compute number of dofs per element
    num_dof_in_element = np.int(Nord*(Nord+2)*(Nord+3)/2)

    # Allocate elemental mass matrix and stiffness  matrix
    Me = np.zeros((num_dof_in_element, num_dof_in_element), dtype=np.float)
    Ke = np.zeros((num_dof_in_element, num_dof_in_element), dtype=np.float)

    # Compute gauss points
    gauss_order = 2*Nord
    gaussPoints, Wi = compute3DGaussPoints(gauss_order)
    ngaussP = gaussPoints.shape[0]

    # Tensor for integration (Vertical transverse electric permittivity)
    # sigmaX = sigmaY = sigmaEle[0]
    # sigmaZ = sigmaEle[1]
    e_r = np.diag((sigmaEle[0], sigmaEle[0], sigmaEle[1]))

    # Tensor for integration (Constant magnetic permittivity)
    mu_r = np.ones(3, dtype=np.float)
    inv_mu_r = np.diag(mu_r**-1.)

    # Compute determinant of jacobian
    det_jacob = np.linalg.det(jacobian)

    # ---------------------------------------------------------------
    # Assembly matrices
    # ---------------------------------------------------------------
    for i in np.arange(ngaussP):
        # Get gauss point coordinates
        X = gaussPoints[i, :]
        # Polynomial order (6 edges, 4 faces, 1 volume)
        Nord_vector = np.ones(11, dtype=np.int)*Nord
        # Edge orientation (6 edges)
        NoriE = edge_orientation
        # Face orientation (4 faces)
        NoriF = face_orientation

        # Compute basis for iPoint
        NrdofE, ShapE, CurlE = shape3DETet(X, Nord_vector, NoriE, NoriF)

        # Verify consistency of number of dofs for this point
        if (NrdofE != num_dof_in_element):
            Print.master('        Number of DOFs is not consistent')
            exit(-1)

        # Niref=Ni in reference element
        Niref = ShapE[0:3, 0:NrdofE]

        # Ni=Ni in real element
        Ni_real = np.matmul(invjacob, Niref)

        # Perform mass matrix integration
        for j in np.arange(num_dof_in_element):
            for k in np.arange(num_dof_in_element):
                Me[j,k] += Wi[i] * np.dot(np.matmul(Ni_real[:,j], e_r), Ni_real[:,k])

        # Transform curl on reference element to real element
        curl_reference_element = CurlE[0:3, 0:NrdofE]
        curl_real_element = np.zeros((num_dof_in_element,3), dtype=np.float)

        for j in np.arange(num_dof_in_element):
            curl_real_element[j,:] = np.matmul(curl_reference_element[:,j],jacobian)/det_jacob

        # Perform stiffness matrix integration

        for j in np.arange(num_dof_in_element):
            for k in np.arange(num_dof_in_element):
                Ke[j,k] += Wi[i] * np.dot(np.matmul(curl_real_element[j,:], inv_mu_r), curl_real_element[k,:])

    # Matrix normalization
    Me = Me*det_jacob
    Ke = Ke*det_jacob

    return Me, Ke


def shape3DETet(X, Nord, NoriE, NoriF):
    r"""Compute values of 3D tetrahedron element H(curl) shape functions and their derivatives.

    :param ndarray X: master tetrahedron coordinates from (0,1)^3
    :param int Nord: polynomial order
    :param ndarray NoriE: edge orientation
    :param ndarray NoriF: face orientation
    :return: number of dof, values of the shape functions at the point, curl of the shape functions
    :rtype: ndarray.

    .. note:: References:\n
       Fuentes, F., Keith, B., Demkowicz, L., & Nagaraj, S. (2015). Orientation
       embedded high order shape functions for the exact sequence elements of
       all shapes. Computers & Mathematics with applications, 70(4), 353-458.
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    MAXP = 6
    MAXtetraE = np.int(MAXP*(MAXP+2)*(MAXP+3)/2)
    IdecB = np.array([False, False])

    # Spatial dimension
    N = 3

    # Allocate matrices for shape functions
    ShapE = np.zeros((3, MAXtetraE), dtype=np.float)
    CurlE = np.zeros((3, MAXtetraE), dtype=np.float)

    # Local parameters
    minI = 0
    minJ = 1
    minK = 1
    minIJ = minI+minJ
    minIJK = minIJ+minK
    NumEdgesInElement = 6
    NumFacesInElement = 4

    # Initialize counter for shape functions
    m = 0

    # ---------------------------------------------------------------
    # Basis computation
    # ---------------------------------------------------------------
    # Define affine coordinates and gradients
    Lam, DLam = AffineTetrahedron(X)

    # Edge shape functions
    LampE, DLampE, IdecE = ProjectTetE(Lam, DLam)

    # Loop over edges
    for i in np.arange(NumEdgesInElement):
        # Local parameters
        nordE = Nord[i]
        ndofE = nordE
        if (ndofE > 0):
            # Local parameters
            maxI = nordE-1
            # Orient
            D = DLampE[i, 0:N, 0:2]
            GLampE, GDLampE = OrientE(LampE[i, 0:2], D, NoriE[i], N)
            # Construct the shape functions
            EE, CurlEE = AncEE(GLampE,GDLampE,nordE,IdecE,N)
            for i in np.arange(minI, maxI+1):
                ShapE[0:N, m] = EE[0:N, i]
                CurlE[0:N, m] = CurlEE[0:N, i]
                m += 1

    # Face shape functions
    LampF, DLampF, IdecF = ProjectTetF(Lam,DLam)

    # Loop over faces
    for i in np.arange(NumFacesInElement):
        # Local parameters
        nordF = Nord[6+i]
        ndofF = np.int(nordF*(nordF-1)/2)
        if (ndofF > 0):
            # Local parameters (again)
            maxIJ = nordF-1;
            # Orient
            D = DLampF[i, 0:N, 0:3]
            GLampF, GDLampF = OrientTri(LampF[i, 0:3], D, NoriF[i], N)
            # Loop over families
            famctr = m
            for j in np.arange(2):
                m = famctr+j-1
                abc = np.roll([0,1,2], -j)
                # Construct the shape functions
                ETri, CurlETri = AncETri(GLampF[abc], GDLampF[0:N,abc], nordF, IdecF, N)
                for k in np.arange(minIJ, maxIJ+1):
                    for r in np.arange(minI, k-minJ+1):
                        p = k-r
                        m += 2
                        ShapE[0:N, m-1] = ETri[0:N, r, p-1]
                        CurlE[0:N, m-1] = CurlETri[0:N, r, p-1]

    # BUBBLE FUNCTIONS
    # Local parameters
    nordB = Nord[10]
    ndofB = np.int(nordB*(nordB-1)*(nordB-2)/6)

    # If necessary, create bubbles
    if (ndofB > 0):
        # Local parameters (again)
        IdecB[0] = IdecF
        IdecB[1] = True
        minbeta = 2*minIJ
        maxIJK = nordB-1
        maxK = maxIJK-minIJ
        # Loop over families
        famctr = m
        for i in np.arange(3):
            m = famctr+i-2
            abcd = np.roll([0,1,2,3], -i)

            abc = abcd[[0,1,2]]
            d = abcd[3]

            # Now construct the shape functions (no need to orient)
            ETri,CurlETri = AncETri(Lam[abc],DLam[0:N,abc],nordB-minK,IdecB[0],N)

            tmp1 = np.array([1-Lam[d],Lam[d]], dtype=np.float)
            tmp2 = np.array([-DLam[0:N,d],DLam[0:N,d]], dtype=np.float)
            homLbet, DhomLbet = HomIJacobi(tmp1,tmp2,maxK,minbeta,IdecB[1],N)

            for j in np.arange(minIJK, maxIJK+1):
                for k in np.arange(minIJ, j-minK+1):
                    for r in np.arange(minI, k-minJ+1):
                        p = k-r
                        q = j-k
                        m += 3

                        ShapE[0:N, m-1] = ETri[0:N,r,p-1]*homLbet[k-1,q-1]
                        DhomLbetxETri = np.cross(DhomLbet[0:N,k-1,q-1], ETri[0:N,r,p-1])
                        CurlE[0:N, m-1] = homLbet[k-1,q-1]*CurlETri[0:N,r,p-1] + DhomLbetxETri

    # Get total degrees of freedom
    # First order  = 6
    # Second order = 20
    # Third order  = 45
    # Fourth order = 84
    # Fifth order  = 140
    # Sixth order  = 216
    NrdofE = m

    return NrdofE, ShapE, CurlE


def AncEE(S, DS, Nord, Idec, N):
    r"""Compute edge Hcurl ancillary functions and their curls.

    :param ndarray S: affine coordinates associated to edge
    :param ndarray DS: derivatives of S in R^N
    :param int Nord: polynomial order
    :param bool Idec: Binary flag
    :param int N: spatial dimension
    :return: edge Hcurl ancillary functions, curls of edge Hcurl ancillary functions
    :rtype: ndarray

    .. note:: References:\n
       Idec: = FALSE  s0+s1 != 1
             = TRUE   s0+s1  = 1
     """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Allocate
    EE = np.zeros((N, Nord), dtype=np.float)
    CurlEE = np.zeros((2*N-3, Nord), dtype=np.float)

    # Local parameters
    minI = 1
    maxI = Nord
    Ncurl = 2*N-3

    # Extract homogenized Legendre polyomials first
    homP = HomLegendre(S,maxI)

    # Simplified case
    if (Idec):
        for i in np.arange(minI, maxI+1):
            EE[0:N, i-1] = homP[i-1]*DS[0:N, 1]
        # No need to compute Whitney function or curl
        CurlEE[0:Ncurl, minI-1: maxI-1] = 0.
    # In general
    else:
        # Lowest order Whitney function and its curl
        whiE = S[0]*DS[0:N, 1] - S[1]*DS[0:N, 0]
        curlwhiE = np.cross(DS[0:N, 0], DS[0:N, 1])
        # Now construct the higher order elements
        for i in np.arange(minI, maxI+1):
            EE[0:N, i-1] = homP[i-1]*whiE
            CurlEE[0:Ncurl, i-1] = (i+1)*homP[i-1]*curlwhiE

    return EE, CurlEE


def AncETri(S, DS, Nord, Idec, N):
    """Compute triangle face Hcurl ancillary functions and their curls.

    :param ndarray S: (s0,s1,s2) affine coordinates associated to triangle face
    :param ndarray DS: derivatives of S0,S1,S2
    :param int Nord: polynomial order
    :param boll Idec: Binary flag:
    :param int N: spatial dimension
    :return: triangle Hcurl ancillary functions and curls of triangle Hcurl ancillary functions
    :rtype: ndarray
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Allocate
    ETri = np.zeros((N, Nord-1, Nord-1), dtype=np.float)
    CurlETri = np.zeros((2*N-3, Nord-1, Nord-1), dtype=np.float)
    DsL = np.zeros((3,2), dtype=np.float)
    sL = np.zeros(2, dtype=np.float)

    # Local parameters
    minI = 0
    #maxI = Nord-2
    minJ = 1
    maxJ = Nord-1
    #minIJ = minI+minJ
    maxIJ = Nord-1
    minalpha = 2*minI+1
    Ncurl = 2*N-3
    IdecE = False

    # get EE - this is never a simplified case (IdecE=0)
    EE, curlEE = AncEE(S[0:2], DS[0:N, 0:2], Nord-minJ, IdecE, N)

    #  get homogenized Integrated Jacobi polynomials, homLal, and gradients
    sL[0] = S[0]+S[1]
    sL[1] = S[2]
    DsL[0:N, 0] = DS[0:N, 0] + DS[0:N, 1]
    DsL[0:N, 1] = DS[0:N, 2]

    homLal, DhomLal = HomIJacobi(sL, DsL, maxJ, minalpha, Idec, N)

    # Simply complete the required information
    for i in np.arange(maxIJ+1):
        for j in np.arange(minI, i-minJ+1):
            k = i-j
            ETri[0:N, j, k-1] = EE[0:N,j]*homLal[j,k-1]
            DhomLalxEE = np.cross(DhomLal[0:N, j, k-1], EE[0:N,j])
            CurlETri[0:Ncurl, j, k-1] = homLal[j,k-1]*curlEE[0:Ncurl,j] + DhomLalxEE

    return ETri, CurlETri


def HomLegendre(S, Nord):
    """Compute values of homogenized Legendre polynomials.

    :param ndarray S: affine(like) coordinates
    :param int Nord: polynomial order
    :return: polynomial values
    :rtype: ndarray
    """
    # Simply the definition of homogenized polynomials
    HomP = PolyLegendre(S[1], S[0]+S[1], Nord)

    return HomP


def HomIJacobi(S, DS, Nord, Minalpha, Idec, N):
    """Compute values of integrated homogenized Jacobi polynomials and their gradients.
    Result is half of a  matrix with each row  associated to a fixed alpha.
    Alpha grows by 2 in each row.

    :param ndarray S: (s0,s1) affine(like) coordinates
    :param ndarray DS: gradients of S in R(N)
    :param int Nord: max polynomial order
    :param int Minalpha: first row value of alpha (integer)
    :param bool Idec: decision flag to compute
    :return: polynomial values and derivatives in x (Jacobi polynomials)
    :rtype: ndarray
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Allocate
    DHomL = np.zeros((N, Nord, Nord), dtype=np.float)

    # clearly (minI,maxI)=(1,Nord), but the syntax is written as it is
    # because it reflects how the indexing is called from outside
    minI = 1
    maxI = minI+Nord-1

    if (Idec):
        HomL, homP, _ = PolyIJacobi(S[1],1,Nord,Minalpha,Idec)
        ni = -1
        for i in np.arange(minI, maxI+1):
            ni += 1
            for j in np.arange(1, Nord-ni+1):
                DHomL[0:N, i-1, j-1] = homP[i-1, j-1]*DS[1, 0:N]
    else:
        # If sum of S different from 1 -> Idec=.FALSE.
        HomL, homP, homR = PolyIJacobi(S[1], S[0]+S[1], Nord, Minalpha, Idec)

        ni = -1
        DS01 = DS[0:N, 0] + DS[0:N, 1]

        for i in np.arange(minI, maxI+1):
            ni += 1
            for j in np.arange(1, Nord-ni+1):
                DHomL[0:N, i-1, j-1] = homP[i-1, j-1]*DS[0:N, 1] + homR[i-1, j-1]*DS01

    return HomL, DHomL


def PolyLegendre(X, T, Nord):
    """Compute values of shifted scaled Legendre polynomials.

    :param ndarray X: coordinate from [0,1]
    :param float T: scaling parameter
    :param int Nord: polynomial order
    :return: polynomial values
    :rtype: ndarray
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Allocate
    P = np.zeros(Nord, dtype=np.float)

    # i stands for the order of the polynomial, stored in P(i)
    # lowest order case (order 0)
    P[0] = 1.

    # First order case (order 1) if necessary
    if (Nord == 1):
        y = 2.*X - T
        P = np.hstack((P, y))

    # Higher order if necessary - use recurrence formula
    if (Nord >= 2):
        # Compute second point
        y = 2.*X - T
        P[1] = y
        # Remaining points
        tt = T**2
        for i in np.arange(1, Nord-1):
            P[i+1] = (2.*i+1)*y*P[i] - (i)*tt*P[i-1]
            P[i+1] = P[i+1]/np.float(i+1)

    return P


def PolyIJacobi(X, T, Nord, Minalpha, Idec):
    """Compute values of integrated shifted scaled Jacobi polynomials and their derivatives starting with p=1.

    Result is 'half' of a  matrix with each row  associated to a fixed alpha.
    Alpha grows by 2 in each row.

    :param ndarray X: coordinate from [0,1]
    :param ndarray T: scaling parameter
    :param int Nord: max polynomial order
    :param int Minalpha: first row value of alpha
    :param bool Idec: decision flag to compute (= FALSE polynomials with x and t derivatives, = TRUE  polynomials with x derivatives only)
    :return: polynomial values, derivatives in x (Jacobi polynomials), derivatives in t
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Allocate
    alpha = np.zeros(Nord, dtype=np.float)
    L = np.zeros((Nord, Nord), dtype=np.float)
    R = np.zeros((Nord, Nord), dtype=np.float)

    # clearly (minI,maxI)=(1,Nord), but the syntax is written as it is
    # because it reflects how the indexing is called from outside
    minI = 0
    maxI = minI+Nord

    # Calling Jacobi for required information
    ptemp = PolyJacobi(X,T,Nord,Minalpha)

    # Define P. Note that even though P is defined at all entries,
    # because of the way Jacobi computes ptemp, only the necessary entries,
    # and those on the first subdiagonal (which are never used later)
    # are actually accurate.
    P = ptemp[minI:maxI, 0:Nord]

    # Create vector alpha first
    for i in np.arange(maxI):
        alpha[i] = Minalpha+2*(i-minI)

    # Initiate first column (order 1 in L)
    L[minI:maxI, 0] = X

    # General case; compute R
    R[minI:maxI, ] = 0

    #  Fill the last columns if necessary
    if (Nord>=2):
        tt = T**2
        ni = -1
        for i in np.arange(maxI-1):
            al = alpha[i]
            ni += 1
            for j in np.arange(2, Nord-ni+1):
                tia = j+j+al
                tiam1 = tia-1
                tiam2 = tia-2
                ai = (j+al)/(tiam1*tia)
                bi = (al)/(tiam2*tia)
                ci = (j-1)/(tiam2*tiam1)
                L[i, j-1] = ai*ptemp[i, j]+bi*T*ptemp[i,j-1]-ci*tt*ptemp[i,j-2]
                R[i, j-1] = -(j-1)*(ptemp[i,j-1]+T*ptemp[i,j-2])
                R[i, j-1] = R[i, j-1]/tiam2

    return L, P, R


def PolyJacobi(X, T, Nord, Minalpha):
    """Compute values of shifted scaled Jacobi polynomials P**alpha-i.

    Result is a half of a matrix with each row associated to a fixed alpha.
    Alpha grows by 2 in each row.

    :param ndarray X: coordinate from [0,1]
    :param float T: scaling parameter
    :param int Nord: max polynomial order
    :param int Minalpha: first row value of alpha (integer)
    :return: polynomial values
    :rtype: ndarray
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Allocate
    alpha = np.zeros(Nord+1, dtype=np.float)
    P = np.zeros((Nord+1, Nord+1), dtype=np.float)

    # Clearly (minI,maxI)=(0,Nord), but the syntax is written as it is
    # because it reflects how the indexing is called from outside
    minI = 0
    maxI = minI+Nord

    for i in np.arange(maxI+1):
        alpha[i] = Minalpha+2*(i-minI)

    # Initiate first column (order 0)
    P[minI:maxI+1, 0] = 1.

    # Initiate second column (order 1) if necessary
    if (Nord>=1):
        y = 2*X - T
        indx = np.arange(minI, maxI)
        P[indx, 1] = y+alpha[indx]*X

    # Fill the last columns if necessary
    if (Nord >= 2):
        tt = T**2
        ni = -1
        for i in np.arange(maxI-1):
            al = alpha[i]
            aa = al**2
            ni += 1
            # Use recursion in order, i, to compute P^alpha_i for i>=2
            for j in np.arange(2, Nord-ni+1):
                ai = 2*j*(j+al)*(2*j+al-2)
                bi = 2*j+al-1
                ci = (2*j+al)*(2*j+al-2)
                di = 2*(j+al-1)*(j-1)*(2*j+al)

                P[i,j] = bi*(ci*y+aa*T)*P[i,j-1]-di*tt*P[i,j-2]
                P[i,j] = P[i, j]/ai

    return P


def OrientE(S, DS, Nori, N):
    """Compute the local to global transformations of edges.

    :param ndarray S: projection of affine coordinates on edges
    :param ndarray DS: projection of gradients of affine coordinates on edges
    :param ndarray Nori: edge orientation
    :param int N: number of dimensions
    :return: global transformation of edges and global transformation of gradients of edges
    :rtype: ndarray
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Allocate
    Or =  np.zeros((2,2), dtype=np.float)
    GS =  np.zeros(2, dtype=np.float)
    GDS = np.zeros((3,2), dtype=np.float)

    # Nori=0 => (s0,s1)->(s0,s1)
    Or[0, 0] = 0
    Or[0, 1] = 1
    # Nori=1 => (s0,s1)->(s1,s0)
    Or[1, 0] = 1
    Or[1, 1] = 0

    # Local-to-global transformation
    GS[0] = S[np.int(Or[Nori, 0])]
    GS[1] = S[np.int(Or[Nori, 1])]
    GDS[0:N, 0] = DS[0:N, np.int(Or[Nori,0])]
    GDS[0:N, 1] = DS[0:N, np.int(Or[Nori,1])]

    return GS, GDS


def OrientTri(S, DS, Nori, N):
    """Compute the local to global transformations of edges.

    :param ndarray S: projection of affine coordinates on faces
    :param ndarray DS: projection of gradients of affine coordinates on faces
    :param ndarray Nori: face orientation
    :param int N: number of dimensions
    :return: global transformation of faces and global transformation of gradients of faces
    :rtype: ndarray
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Allocate
    Or = np.zeros((6,3), np.int)
    GS = np.zeros(3, np.float)
    GDS = np.zeros((3,3), np.float)

    # GS(0:2)=S(Or(Nori,0:2))
    # Nori=0 => (s0,s1,s2)->(s0,s1,s2)
    Or[0, 0] = 0
    Or[0, 1] = 1
    Or[0, 2] = 2
    # Nori=1 => (s0,s1,s2)->(s1,s2,s0)
    Or[1, 0] = 1
    Or[1, 1] = 2
    Or[1, 2] = 0
    # Nori=2 => (s0,s1,s2)->(s2,s0,s1)
    Or[2, 0] = 2
    Or[2, 1] = 0
    Or[2, 2] = 1
    # Nori=3 => (s0,s1,s2)->(s0,s2,s1)
    Or[3, 0] = 0
    Or[3, 1] = 2
    Or[3, 2] = 1
    # Nori=4 => (s0,s1,s2)->(s1,s0,s2)
    Or[4, 0] = 1
    Or[4, 1] = 0
    Or[4, 2] = 2
    # Nori=5 => (s0,s1,s2)->(s2,s1,s0)
    Or[5, 0] = 2
    Or[5, 1] = 1
    Or[5, 2] = 0

    # Local-to-global transformation
    GS[0] = S[Or[Nori,0]]
    GS[1] = S[Or[Nori,1]]
    GS[2] = S[Or[Nori,2]]

    GDS[0:N, 0] = DS[0:N, Or[Nori, 0]]
    GDS[0:N, 1] = DS[0:N, Or[Nori, 1]]
    GDS[0:N, 2] = DS[0:N, Or[Nori, 2]]

    return GS, GDS


def ProjectTetE(Lam, DLam):
    r"""Projection of tetrahedral edges in concordance with numbering of topological entities (vertices, edges, faces).

    :param ndarray Lam: affine coordinates
    :param ndarray DLam: gradients of affine coordinates
    :return: projection of affine coordinates on edges, projection of gradients of affine coordinates on edges
    :rtype: ndarray

    .. note:: References:\n
       Fuentes, F., Keith, B., Demkowicz, L., & Nagaraj, S. (2015). Orientation
       embedded high order shape functions for the exact sequence elements of
       all shapes. Computers & Mathematics with applications, 70(4), 353-458.
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # 6 edges, each with a locally oriented pair representing a projection
    N = 3
    num_edges_in_element = 6
    LampE = np.zeros((num_edges_in_element, 2), dtype=np.float)
    DLampE = np.zeros((num_edges_in_element, N, 2), dtype=np.float)

    # ---------------------------------------------------------------
    # Compute projection
    # ---------------------------------------------------------------
    # e=1 --> edge01 with local orientation v0->v1
    e = 0
    LampE[e, 0] = Lam[0]
    LampE[e, 1] = Lam[1]
    DLampE[e, 0:N, 0] = DLam[0:N, 0]
    DLampE[e, 0:N, 1] = DLam[0:N, 1]

    # e=2 --> edge12 with local orientation v1->v2
    e = 1
    LampE[e, 0] = Lam[1]
    LampE[e, 1] = Lam[2]
    DLampE[e, 0:N, 0] = DLam[0:N, 1]
    DLampE[e, 0:N, 1] = DLam[0:N, 2]

    # e=3 --> edge20 with local orientation v0->v2
    e = 2
    LampE[e, 0] = Lam[0]
    LampE[e, 1] = Lam[2]
    DLampE[e, 0:N, 0] = DLam[0:N, 0]
    DLampE[e, 0:N, 1] = DLam[0:N, 2]

    # e=4 --> edge03 with local orientation v0->v3
    e = 3
    LampE[e, 0] = Lam[0]
    LampE[e, 1] = Lam[3]
    DLampE[e, 0:N, 0] = DLam[0:N, 0]
    DLampE[e, 0:N, 1] = DLam[0:N, 3]

    # e=5 --> edge13 with local orientation v1->v3
    e = 4
    LampE[e, 0] = Lam[1]
    LampE[e, 1] = Lam[3]
    DLampE[e, 0:N, 0] = DLam[0:N, 1]
    DLampE[e, 0:N, 1] = DLam[0:N, 3]

    # e=6 --> edge23 with local orientation v2->v3
    e = 5
    LampE[e, 0] = Lam[2]
    LampE[e, 1] = Lam[3]
    DLampE[e, 0:N, 0] = DLam[0:N, 2]
    DLampE[e, 0:N, 1] = DLam[0:N, 3]

    # Projected coordinates are Lam, so IdecE=false for all edges
    IdecE = False

    return LampE, DLampE, IdecE


def ProjectTetF(Lam, DLam):
    """Projection of tetrahedral faces in concordance with numbering of topological entities (vertices, edges, faces).

    :param ndarray Lam: affine coordinates
    :param ndarray DLam: gradients of affine coordinates
    :return: projection of affine coordinates on faces, projection of gradients of affine coordinates on faces
    :rtype: ndarray
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # 4 faces, each with a locally oriented triplet representing
    # a projection
    # Allocate
    N = 3
    num_faces_in_element = 4
    LampF = np.zeros((num_faces_in_element, N), dtype=np.float)
    DLampF = np.zeros((num_faces_in_element, N, N), dtype=np.float)

    # ---------------------------------------------------------------
    # Compute projection
    # ---------------------------------------------------------------
    # f=1 --> face012 with local orientation v0->v1->v2
    f = 0
    LampF[f, 0] = Lam[0]
    LampF[f, 1] = Lam[1]
    LampF[f, 2] = Lam[2]
    DLampF[f, 0:N, 0] = DLam[0:N, 0]
    DLampF[f, 0:N, 1] = DLam[0:N, 1]
    DLampF[f, 0:N, 2] = DLam[0:N, 2]


    # f=2 --> face013 with local orientation v0->v1->v3
    f = 1
    LampF[f, 0] = Lam[0]
    LampF[f, 1] = Lam[1]
    LampF[f, 2] = Lam[3]
    DLampF[f, 0:N, 0] = DLam[0:N, 0]
    DLampF[f, 0:N, 1] = DLam[0:N, 1]
    DLampF[f, 0:N, 2] = DLam[0:N, 3]

    # f=3 --> face123 with local orientation v1->v2->v3
    f = 2
    LampF[f, 0] = Lam[1]
    LampF[f, 1] = Lam[2]
    LampF[f, 2] = Lam[3]
    DLampF[f, 0:N, 0] = DLam[0:N, 1]
    DLampF[f, 0:N, 1] = DLam[0:N, 2]
    DLampF[f, 0:N, 2] = DLam[0:N, 3]

    # f=4 --> face023 with local orientation v0->v2->v3
    f = 3
    LampF[f, 0] = Lam[0]
    LampF[f, 1] = Lam[2]
    LampF[f, 2] = Lam[3]
    DLampF[f, 0:N, 0] = DLam[0:N, 0]
    DLampF[f, 0:N, 1] = DLam[0:N, 2]
    DLampF[f, 0:N, 2] = DLam[0:N, 3]

    # Pprojected coordinates are Lam, so IdecF=false for all faces
    IdecF = False

    return LampF, DLampF, IdecF


def AffineTetrahedron(X):
    r"""Compute affine coordinates and their gradients.

    :param ndarray X: point coordinates
    :return: affine coordinates and gradients of affine coordinates
    :rtype: ndarray

    .. note:: References:\n
       Fuentes, F., Keith, B., Demkowicz, L., & Nagaraj, S. (2015). Orientation
       embedded high order shape functions for the exact sequence elements of
       all shapes. Computers & Mathematics with applications, 70(4), 353-458.
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Allocate
    Lam = np.zeros(4, dtype=np.float)
    DLam = np.zeros((3,4), dtype=np.float)

    # Define affine coordinates
    Lam[0] = 1.-X[0]-X[1]-X[2]
    Lam[1] = X[0]
    Lam[2] = X[1]
    Lam[3] = X[2]

    # and their gradients
    DLam[0, 0] = -1
    DLam[0, 1] =  1
    DLam[1, 0] = -1
    DLam[1, 2] =  1
    DLam[2, 0] = -1
    DLam[2, 3] =  1

    return Lam, DLam


def compute3DGaussPoints(Nord):
    """Compute 3D gauss points for high-order nédélec elements.

    :param int Nord: polynomial order of nedelec basis functions
    :return: coordinates of gauss points and its weights
    :rtype: ndarray.
    """
    # Select gauss points
    if Nord == 1:
        table = np.array([-0.500000000000000, -0.500000000000000, -0.500000000000000, 1.333333333333333], dtype=np.float)
    elif Nord == 2:
        table = np.array([[-0.723606797749979, -0.723606797749979, -0.723606797749979, 0.333333333333333],
                          [ 0.170820393249937, -0.723606797749979, -0.723606797749979, 0.333333333333333],
                          [-0.723606797749979,  0.170820393249937, -0.723606797749979, 0.333333333333333],
                          [-0.723606797749979, -0.723606797749979,  0.170820393249937, 0.333333333333333]], dtype=np.float)
    elif Nord == 3:
        table = np.array([[-0.500000000000000, -0.500000000000000, -0.500000000000000, -1.066666666666667],
                          [-0.666666666666667, -0.666666666666667, -0.666666666666667,  0.600000000000000],
                          [-0.666666666666667, -0.666666666666667,  0.000000000000000,  0.600000000000000],
                          [-0.666666666666667,  0.000000000000000, -0.666666666666667,  0.600000000000000],
                          [ 0.000000000000000, -0.666666666666667, -0.666666666666667,  0.600000000000000]], dtype=np.float)
    elif Nord == 4:
        table = np.array([[-0.500000000000000, -0.500000000000000, -0.500000000000000, -0.105244444444444],
                          [-0.857142857142857, -0.857142857142857, -0.857142857142857,  0.060977777777778],
                          [-0.857142857142857, -0.857142857142857,  0.571428571428571,  0.060977777777778],
                          [-0.857142857142857,  0.571428571428571, -0.857142857142857,  0.060977777777778],
                          [ 0.571428571428571, -0.857142857142857, -0.857142857142857,  0.060977777777778],
                          [-0.201192847666402, -0.201192847666402, -0.798807152333598,  0.199111111111111],
                          [-0.201192847666402, -0.798807152333598, -0.201192847666402,  0.199111111111111],
                          [-0.798807152333598, -0.201192847666402, -0.201192847666402,  0.199111111111111],
                          [-0.201192847666402, -0.798807152333598, -0.798807152333598,  0.199111111111111],
                          [-0.798807152333598, -0.201192847666402, -0.798807152333598,  0.199111111111111],
                          [-0.798807152333598, -0.798807152333598, -0.201192847666402,  0.199111111111111]], dtype=np.float)
    elif Nord == 5:
        table = np.array([[-0.814529499378218, -0.814529499378218, -0.814529499378218,  0.097990724155149],
                          [ 0.443588498134653, -0.814529499378218, -0.814529499378218,  0.097990724155149],
                          [-0.814529499378218,  0.443588498134653, -0.814529499378218,  0.097990724155149],
                          [-0.814529499378218, -0.814529499378218,  0.443588498134653,  0.097990724155149],
                          [-0.378228161473399, -0.378228161473399, -0.378228161473399,  0.150250567624021],
                          [-0.865315515579804, -0.378228161473399, -0.378228161473399,  0.150250567624021],
                          [-0.378228161473399, -0.865315515579804, -0.378228161473399,  0.150250567624021],
                          [-0.378228161473399, -0.378228161473399, -0.865315515579804,  0.150250567624021],
                          [-0.091007408251299, -0.091007408251299, -0.908992591748701,  0.056728027702775],
                          [-0.091007408251299, -0.908992591748701, -0.091007408251299,  0.056728027702775],
                          [-0.908992591748701, -0.091007408251299, -0.091007408251299,  0.056728027702775],
                          [-0.091007408251299, -0.908992591748701, -0.908992591748701,  0.056728027702775],
                          [-0.908992591748701, -0.091007408251299, -0.908992591748701,  0.056728027702775],
                          [-0.908992591748701, -0.908992591748701, -0.091007408251299,  0.056728027702775]], dtype=np.float)
    elif Nord == 6:
        table = np.array([[-0.570794257481696, -0.570794257481696, -0.570794257481696, 0.053230333677557],
                          [-0.287617227554912, -0.570794257481696, -0.570794257481696, 0.053230333677557],
                          [-0.570794257481696, -0.287617227554912, -0.570794257481696, 0.053230333677557],
                          [-0.570794257481696, -0.570794257481696, -0.287617227554912, 0.053230333677557],
                          [-0.918652082930777, -0.918652082930777, -0.918652082930777, 0.013436281407094],
                          [ 0.755956248792332, -0.918652082930777, -0.918652082930777, 0.013436281407094],
                          [-0.918652082930777,  0.755956248792332, -0.918652082930777 ,0.013436281407094],
                          [-0.918652082930777, -0.918652082930777,  0.755956248792332 ,0.013436281407094],
                          [-0.355324219715449, -0.355324219715449, -0.355324219715449, 0.073809575391540],
                          [-0.934027340853653, -0.355324219715449, -0.355324219715449, 0.073809575391540],
                          [-0.355324219715449, -0.934027340853653, -0.355324219715449, 0.073809575391540],
                          [-0.355324219715449, -0.355324219715449, -0.934027340853653, 0.073809575391540],
                          [-0.872677996249965, -0.872677996249965, -0.460655337083368, 0.064285714285714],
                          [-0.872677996249965, -0.460655337083368, -0.872677996249965, 0.064285714285714],
                          [-0.872677996249965, -0.872677996249965,  0.206011329583298, 0.064285714285714],
                          [-0.872677996249965,  0.206011329583298, -0.872677996249965, 0.064285714285714],
                          [-0.872677996249965, -0.460655337083368,  0.206011329583298, 0.064285714285714],
                          [-0.872677996249965,  0.206011329583298, -0.460655337083368, 0.064285714285714],
                          [-0.460655337083368, -0.872677996249965, -0.872677996249965, 0.064285714285714],
                          [-0.460655337083368, -0.872677996249965,  0.206011329583298, 0.064285714285714],
                          [-0.460655337083368,  0.206011329583298, -0.872677996249965, 0.064285714285714],
                          [ 0.206011329583298, -0.872677996249965, -0.460655337083368, 0.064285714285714],
                          [ 0.206011329583298, -0.872677996249965, -0.872677996249965, 0.064285714285714],
                          [ 0.206011329583298, -0.460655337083368, -0.872677996249965, 0.064285714285714]], dtype=np.float)

    elif Nord == 7:
        table = np.array([[0.000000000000000,   0.000000000000000,  -1.000000000000000,   0.007760141093474],
                         [ 0.000000000000000,  -1.000000000000000,   0.000000000000000,   0.007760141093474],
                         [-1.000000000000000,   0.000000000000000,   0.000000000000000,   0.007760141093474],
                         [-1.000000000000000,  -1.000000000000000,   0.000000000000000,   0.007760141093474],
                         [-1.000000000000000,   0.000000000000000,  -1.000000000000000,   0.007760141093474],
                         [ 0.000000000000000,  -1.000000000000000,  -1.000000000000000,   0.007760141093474],
                         [-0.500000000000000,  -0.500000000000000,  -0.500000000000000,   0.146113787728871],
                         [-0.843573615339364,  -0.843573615339364,  -0.843573615339364,   0.084799532195309],
                         [-0.843573615339364,  -0.843573615339364,   0.530720846018092,   0.084799532195309],
                         [-0.843573615339364,   0.530720846018092,  -0.843573615339364,   0.084799532195309],
                         [ 0.530720846018092,  -0.843573615339364,  -0.843573615339364,   0.084799532195309],
                         [-0.756313566672190,  -0.756313566672190,  -0.756313566672190,  -0.500141920914655],
                         [-0.756313566672190,  -0.756313566672190,   0.268940700016569,  -0.500141920914655],
                         [-0.756313566672190,   0.268940700016569,  -0.756313566672190,  -0.500141920914655],
                         [ 0.268940700016569,  -0.756313566672190,  -0.756313566672190,  -0.500141920914655],
                         [-0.334921671107159,  -0.334921671107159,  -0.334921671107159,   0.039131402104588],
                         [-0.334921671107159,  -0.334921671107159,  -0.995234986678524,   0.039131402104588],
                         [-0.334921671107159,  -0.995234986678524,  -0.334921671107159,   0.039131402104588],
                         [-0.995234986678524,  -0.334921671107159,  -0.334921671107159,   0.039131402104588],
                         [-0.800000000000000,  -0.800000000000000,  -0.600000000000000,   0.220458553791887],
                         [-0.800000000000000,  -0.600000000000000,  -0.800000000000000,   0.220458553791887],
                         [-0.800000000000000,  -0.800000000000000,   0.200000000000000,   0.220458553791887],
                         [-0.800000000000000,   0.200000000000000,  -0.800000000000000,   0.220458553791887],
                         [-0.800000000000000,  -0.600000000000000,   0.200000000000000,   0.220458553791887],
                         [-0.800000000000000,   0.200000000000000,  -0.600000000000000,   0.220458553791887],
                         [-0.600000000000000,  -0.800000000000000,  -0.800000000000000,   0.220458553791887],
                         [-0.600000000000000,  -0.800000000000000,   0.200000000000000,   0.220458553791887],
                         [-0.600000000000000,   0.200000000000000,  -0.800000000000000,   0.220458553791887],
                         [ 0.200000000000000,  -0.800000000000000,  -0.600000000000000,   0.220458553791887],
                         [ 0.200000000000000,  -0.800000000000000,  -0.800000000000000,   0.220458553791887],
                         [ 0.200000000000000,  -0.600000000000000,  -0.800000000000000,   0.220458553791887]], dtype=np.float)
    elif Nord == 8:
        table = np.array([[-0.500000000000000,  -0.500000000000000,  -0.500000000000000, -0.164001509269119],
                          [-0.586340136778654,  -0.586340136778654,  -0.586340136778654,  0.114002446582935],
                          [-0.586340136778654,  -0.586340136778654,  -0.240979589664039,  0.114002446582935],
                          [-0.586340136778654,  -0.240979589664039,  -0.586340136778654,  0.114002446582935],
                          [-0.240979589664039,  -0.586340136778654,  -0.586340136778654,  0.114002446582935],
                          [-0.835792823378907,  -0.835792823378907,  -0.835792823378907,  0.015736266505071],
                          [-0.835792823378907,  -0.835792823378907,   0.507378470136720,  0.015736266505071],
                          [-0.835792823378907,   0.507378470136720,  -0.835792823378907,  0.015736266505071],
                          [ 0.507378470136720,  -0.835792823378907,  -0.835792823378907,  0.015736266505071],
                          [-0.988436098989604,  -0.988436098989604,  -0.988436098989604,  0.001358672872743],
                          [-0.988436098989604,  -0.988436098989604,   0.965308296968812,  0.001358672872743],
                          [-0.988436098989604,   0.965308296968812,  -0.988436098989604,  0.001358672872743],
                          [ 0.965308296968812,  -0.988436098989604,  -0.988436098989604,  0.001358672872743],
                          [-0.898934519962212,  -0.898934519962212,  -0.101065480037788,  0.036637470595738],
                          [-0.898934519962212,  -0.101065480037788,  -0.898934519962212,  0.036637470595738],
                          [-0.101065480037788,  -0.898934519962212,  -0.898934519962212,  0.036637470595738],
                          [-0.898934519962212,  -0.101065480037788,  -0.101065480037788,  0.036637470595738],
                          [-0.101065480037788,  -0.898934519962212,  -0.101065480037788,  0.036637470595738],
                          [-0.101065480037788,  -0.101065480037788,  -0.898934519962212,  0.036637470595738],
                          [-0.541866927766378,  -0.541866927766378,  -0.928720834422932,  0.045635886469455],
                          [-0.541866927766378,  -0.928720834422932,  -0.541866927766378,  0.045635886469455],
                          [-0.541866927766378,  -0.541866927766378,   0.012454689955687,  0.045635886469455],
                          [-0.541866927766378,   0.012454689955687,  -0.541866927766378,  0.045635886469455],
                          [-0.541866927766378,  -0.928720834422932,   0.012454689955687,  0.045635886469455],
                          [-0.541866927766378,   0.012454689955687,  -0.928720834422932,  0.045635886469455],
                          [-0.928720834422932,  -0.541866927766378,  -0.541866927766378,  0.045635886469455],
                          [-0.928720834422932,  -0.541866927766378,   0.012454689955687,  0.045635886469455],
                          [-0.928720834422932,   0.012454689955687,  -0.541866927766378,  0.045635886469455],
                          [ 0.012454689955687,  -0.541866927766378,  -0.928720834422932,  0.045635886469455],
                          [ 0.012454689955687,  -0.541866927766378,  -0.541866927766378,  0.045635886469455],
                          [ 0.012454689955687,  -0.928720834422932,  -0.541866927766378,  0.045635886469455],
                          [-0.926784500893605,  -0.926784500893605,  -0.619027916130733,  0.017124153129297],
                          [-0.926784500893605,  -0.619027916130733,  -0.926784500893605,  0.017124153129297],
                          [-0.926784500893605,  -0.926784500893605,   0.472596917917943,  0.017124153129297],
                          [-0.926784500893605,   0.472596917917943,  -0.926784500893605,  0.017124153129297],
                          [-0.926784500893605,  -0.619027916130733,   0.472596917917943,  0.017124153129297],
                          [-0.926784500893605,   0.472596917917943,  -0.619027916130733,  0.017124153129297],
                          [-0.619027916130733,  -0.926784500893605,  -0.926784500893605,  0.017124153129297],
                          [-0.619027916130733,  -0.926784500893605,   0.472596917917943,  0.017124153129297],
                          [-0.619027916130733,   0.472596917917943,  -0.926784500893605,  0.017124153129297],
                          [ 0.472596917917943,  -0.926784500893605,  -0.619027916130733,  0.017124153129297],
                          [ 0.472596917917943,  -0.926784500893605,  -0.926784500893605,  0.017124153129297],
                          [ 0.472596917917943,  -0.619027916130733,  -0.926784500893605,  0.017124153129297]], dtype=np.float)
    elif Nord == 9:
        table = np.array([[-0.500000000000000,  -0.500000000000000,  -0.500000000000000,  -1.102392306608869],
                         [-0.903297922900526,  -0.903297922900526,  -0.903297922900526,   0.014922692552682],
                         [-0.903297922900526,  -0.903297922900526,   0.709893768701580,   0.014922692552682],
                         [-0.903297922900526,   0.709893768701580,  -0.903297922900526,   0.014922692552682],
                         [ 0.709893768701580,  -0.903297922900526,  -0.903297922900526,   0.014922692552682],
                         [-0.350841439764235,  -0.350841439764235,  -0.350841439764235,   0.034475391755947],
                         [-0.350841439764235,  -0.350841439764235,  -0.947475680707294,   0.034475391755947],
                         [-0.350841439764235,  -0.947475680707294,  -0.350841439764235,   0.034475391755947],
                         [-0.947475680707294,  -0.350841439764235,  -0.350841439764235,   0.034475391755947],
                         [-0.770766919552010,  -0.770766919552010,  -0.770766919552010,  -0.721478131849612],
                         [-0.770766919552010,  -0.770766919552010,   0.312300758656029,  -0.721478131849612],
                         [-0.770766919552010,   0.312300758656029,  -0.770766919552010,  -0.721478131849612],
                         [ 0.312300758656029,  -0.770766919552010,  -0.770766919552010,  -0.721478131849612],
                         [-0.549020096176972,  -0.549020096176972,  -0.549020096176972,   0.357380609620092],
                         [-0.549020096176972,  -0.549020096176972,  -0.352939711469084,   0.357380609620092],
                         [-0.549020096176972,  -0.352939711469084,  -0.549020096176972,   0.357380609620092],
                         [-0.352939711469084,  -0.549020096176972,  -0.549020096176972,   0.357380609620092],
                         [-0.736744381506260,  -0.736744381506260,  -0.832670596765630,   0.277603247076406],
                         [-0.736744381506260,  -0.832670596765630,  -0.736744381506260,   0.277603247076406],
                         [-0.736744381506260,  -0.736744381506260,   0.306159359778151,   0.277603247076406],
                         [-0.736744381506260,   0.306159359778151,  -0.736744381506260,   0.277603247076406],
                         [-0.736744381506260,  -0.832670596765630,   0.306159359778151,   0.277603247076406],
                         [-0.736744381506260,   0.306159359778151,  -0.832670596765630,   0.277603247076406],
                         [-0.832670596765630,  -0.736744381506260,  -0.736744381506260,   0.277603247076406],
                         [-0.832670596765630,  -0.736744381506260,   0.306159359778151,   0.277603247076406],
                         [-0.832670596765630,   0.306159359778151,  -0.736744381506260,   0.277603247076406],
                         [ 0.306159359778151,  -0.736744381506260,  -0.832670596765630,   0.277603247076406],
                         [ 0.306159359778151,  -0.736744381506260,  -0.736744381506260,   0.277603247076406],
                         [ 0.306159359778151,  -0.832670596765630,  -0.736744381506260,   0.277603247076406],
                         [-0.132097077177186,  -0.132097077177186,  -0.784460280901143,   0.026820671221285],
                         [-0.132097077177186,  -0.784460280901143,  -0.132097077177186,   0.026820671221285],
                         [-0.132097077177186,  -0.132097077177186,  -0.951345564744484,   0.026820671221285],
                         [-0.132097077177186,  -0.951345564744484,  -0.132097077177186,   0.026820671221285],
                         [-0.132097077177186,  -0.784460280901143,  -0.951345564744484,   0.026820671221285],
                         [-0.132097077177186,  -0.951345564744484,  -0.784460280901143,   0.026820671221285],
                         [-0.784460280901143,  -0.132097077177186,  -0.132097077177186,   0.026820671221285],
                         [-0.784460280901143,  -0.132097077177186,  -0.951345564744484,   0.026820671221285],
                         [-0.784460280901143,  -0.951345564744484,  -0.132097077177186,   0.026820671221285],
                         [-0.951345564744484,  -0.132097077177186,  -0.784460280901143,   0.026820671221285],
                         [-0.951345564744484,  -0.132097077177186,  -0.132097077177186,   0.026820671221285],
                         [-0.951345564744484,  -0.784460280901143,  -0.132097077177186,   0.026820671221285],
                         [-1.002752554636276,  -1.002752554636276,  -0.446893054726385,   0.003453031004456],
                         [-1.002752554636276,  -0.446893054726385,  -1.002752554636276,   0.003453031004456],
                         [-1.002752554636276,  -1.002752554636276,   0.452398163998938,   0.003453031004456],
                         [-1.002752554636276,   0.452398163998938,  -1.002752554636276,   0.003453031004456],
                         [-1.002752554636276,  -0.446893054726385,   0.452398163998938,   0.003453031004456],
                         [-1.002752554636276,   0.452398163998938,  -0.446893054726385,   0.003453031004456],
                         [-0.446893054726385,  -1.002752554636276,  -1.002752554636276,   0.003453031004456],
                         [-0.446893054726385,  -1.002752554636276,   0.452398163998938,   0.003453031004456],
                         [-0.446893054726385,   0.452398163998938,  -1.002752554636276,   0.003453031004456],
                         [ 0.452398163998938,  -1.002752554636276,  -0.446893054726385,   0.003453031004456],
                         [ 0.452398163998938,  -1.002752554636276,  -1.002752554636276,   0.003453031004456],
                         [ 0.452398163998938,  -0.446893054726385,  -1.002752554636276,   0.003453031004456]], dtype=np.float)

    elif Nord == 10 or Nord == 11:
        table = np.array([[-0.857142857142857,  -0.857142857142857,  0.571428571428571 ,   0.362902592520648],
                         [-0.857142857142857,  -0.571428571428571,   0.285714285714286,   0.362902592520648],
                         [-0.857142857142857,  -0.285714285714286,   0.000000000000000,   0.362902592520648],
                         [-0.857142857142857,   0.000000000000000,  -0.285714285714286,   0.362902592520648],
                         [-0.857142857142857,   0.285714285714286,  -0.571428571428571,   0.362902592520648],
                         [-0.857142857142857,   0.571428571428571,  -0.857142857142857,   0.362902592520648],
                         [-0.571428571428571,  -0.857142857142857,   0.285714285714286,   0.362902592520648],
                         [-0.571428571428571,  -0.571428571428571,   0.000000000000000,   0.362902592520648],
                         [-0.571428571428571,  -0.285714285714286,  -0.285714285714286,   0.362902592520648],
                         [-0.571428571428571,   0.000000000000000,  -0.571428571428571,   0.362902592520648],
                         [-0.571428571428571,   0.285714285714286,  -0.857142857142857,   0.362902592520648],
                         [-0.285714285714286,  -0.857142857142857,   0.000000000000000,   0.362902592520648],
                         [-0.285714285714286,  -0.571428571428571,  -0.285714285714286,   0.362902592520648],
                         [-0.285714285714286,  -0.285714285714286,  -0.571428571428571,   0.362902592520648],
                         [-0.285714285714286,   0.000000000000000,  -0.857142857142857,   0.362902592520648],
                         [ 0.000000000000000,  -0.857142857142857,  -0.285714285714286,   0.362902592520648],
                         [ 0.000000000000000,  -0.571428571428571,  -0.571428571428571,   0.362902592520648],
                         [ 0.000000000000000,  -0.285714285714286,  -0.857142857142857,   0.362902592520648],
                         [ 0.285714285714286,  -0.857142857142857,  -0.571428571428571,   0.362902592520648],
                         [ 0.285714285714286,  -0.571428571428571,  -0.857142857142857,   0.362902592520648],
                         [ 0.571428571428571,  -0.857142857142857,  -0.857142857142857,   0.362902592520648],
                         [-0.857142857142857,  -0.857142857142857,   0.285714285714286,   0.362902592520648],
                         [-0.857142857142857,  -0.571428571428571,   0.000000000000000,   0.362902592520648],
                         [-0.857142857142857,  -0.285714285714286,  -0.285714285714286,   0.362902592520648],
                         [-0.857142857142857,   0.000000000000000,  -0.571428571428571,   0.362902592520648],
                         [-0.857142857142857,   0.285714285714286,  -0.857142857142857,   0.362902592520648],
                         [-0.571428571428571,  -0.857142857142857,   0.000000000000000,   0.362902592520648],
                         [-0.571428571428571,  -0.571428571428571,  -0.285714285714286,   0.362902592520648],
                         [-0.571428571428571,  -0.285714285714286,  -0.571428571428571,   0.362902592520648],
                         [-0.571428571428571,   0.000000000000000,  -0.857142857142857,   0.362902592520648],
                         [-0.285714285714286,  -0.857142857142857,  -0.285714285714286,   0.362902592520648],
                         [-0.285714285714286,  -0.571428571428571,  -0.571428571428571,   0.362902592520648],
                         [-0.285714285714286,  -0.285714285714286,  -0.857142857142857,   0.362902592520648],
                         [ 0.000000000000000,  -0.857142857142857,  -0.571428571428571,   0.362902592520648],
                         [ 0.000000000000000,  -0.571428571428571,  -0.857142857142857,   0.362902592520648],
                         [ 0.285714285714286,  -0.857142857142857,  -0.857142857142857,   0.362902592520648],
                         [-0.857142857142857,  -0.857142857142857,   0.000000000000000,   0.362902592520648],
                         [-0.857142857142857,  -0.571428571428571,  -0.285714285714286,   0.362902592520648],
                         [-0.857142857142857,  -0.285714285714286,  -0.571428571428571,   0.362902592520648],
                         [-0.857142857142857,   0.000000000000000,  -0.857142857142857,   0.362902592520648],
                         [-0.571428571428571,  -0.857142857142857,  -0.285714285714286,   0.362902592520648],
                         [-0.571428571428571,  -0.571428571428571,  -0.571428571428571,   0.362902592520648],
                         [-0.571428571428571,  -0.285714285714286,  -0.857142857142857,   0.362902592520648],
                         [-0.285714285714286,  -0.857142857142857,  -0.571428571428571,   0.362902592520648],
                         [-0.285714285714286,  -0.571428571428571,  -0.857142857142857,   0.362902592520648],
                         [ 0.000000000000000,  -0.857142857142857,  -0.857142857142857,   0.362902592520648],
                         [-0.857142857142857,  -0.857142857142857,  -0.285714285714286,   0.362902592520648],
                         [-0.857142857142857,  -0.571428571428571,  -0.571428571428571,   0.362902592520648],
                         [-0.857142857142857,  -0.285714285714286,  -0.857142857142857,   0.362902592520648],
                         [-0.571428571428571,  -0.857142857142857,  -0.571428571428571,   0.362902592520648],
                         [-0.571428571428571,  -0.571428571428571,  -0.857142857142857,   0.362902592520648],
                         [-0.285714285714286,  -0.857142857142857,  -0.857142857142857,   0.362902592520648],
                         [-0.857142857142857,  -0.857142857142857,  -0.571428571428571,   0.362902592520648],
                         [-0.857142857142857,  -0.571428571428571,  -0.857142857142857,   0.362902592520648],
                         [-0.571428571428571,  -0.857142857142857,  -0.857142857142857,   0.362902592520648],
                         [-0.857142857142857,  -0.857142857142857,  -0.857142857142857,   0.362902592520648],
                         [-0.833333333333333,  -0.833333333333333,   0.500000000000000,  -0.932187812187812],
                         [-0.833333333333333,  -0.500000000000000,   0.166666666666667,  -0.932187812187812],
                         [-0.833333333333333,  -0.166666666666667,  -0.166666666666667,  -0.932187812187812],
                         [-0.833333333333333,   0.166666666666667,  -0.500000000000000,  -0.932187812187812],
                         [-0.833333333333333,   0.500000000000000,  -0.833333333333333,  -0.932187812187812],
                         [-0.500000000000000,  -0.833333333333333,   0.166666666666667,  -0.932187812187812],
                         [-0.500000000000000,  -0.500000000000000,  -0.166666666666667,  -0.932187812187812],
                         [-0.500000000000000,  -0.166666666666667,  -0.500000000000000,  -0.932187812187812],
                         [-0.500000000000000,   0.166666666666667,  -0.833333333333333,  -0.932187812187812],
                         [-0.166666666666667,  -0.833333333333333,  -0.166666666666667,  -0.932187812187812],
                         [-0.166666666666667,  -0.500000000000000,  -0.500000000000000,  -0.932187812187812],
                         [-0.166666666666667,  -0.166666666666667,  -0.833333333333333,  -0.932187812187812],
                         [ 0.166666666666667,  -0.833333333333333,  -0.500000000000000,  -0.932187812187812],
                         [ 0.166666666666667,  -0.500000000000000,  -0.833333333333333,  -0.932187812187812],
                         [ 0.500000000000000,  -0.833333333333333,  -0.833333333333333,  -0.932187812187812],
                         [-0.833333333333333,  -0.833333333333333,   0.166666666666667,  -0.932187812187812],
                         [-0.833333333333333,  -0.500000000000000,  -0.166666666666667,  -0.932187812187812],
                         [-0.833333333333333,  -0.166666666666667,  -0.500000000000000,  -0.932187812187812],
                         [-0.833333333333333,   0.166666666666667,  -0.833333333333333,  -0.932187812187812],
                         [-0.500000000000000,  -0.833333333333333,  -0.166666666666667,  -0.932187812187812],
                         [-0.500000000000000,  -0.500000000000000,  -0.500000000000000,  -0.932187812187812],
                         [-0.500000000000000,  -0.166666666666667,  -0.833333333333333,  -0.932187812187812],
                         [-0.166666666666667,  -0.833333333333333,  -0.500000000000000,  -0.932187812187812],
                         [-0.166666666666667,  -0.500000000000000,  -0.833333333333333,  -0.932187812187812],
                         [ 0.166666666666667,  -0.833333333333333,  -0.833333333333333,  -0.932187812187812],
                         [-0.833333333333333,  -0.833333333333333,  -0.166666666666667,  -0.932187812187812],
                         [-0.833333333333333,  -0.500000000000000,  -0.500000000000000,  -0.932187812187812],
                         [-0.833333333333333,  -0.166666666666667,  -0.833333333333333,  -0.932187812187812],
                         [-0.500000000000000,  -0.833333333333333,  -0.500000000000000,  -0.932187812187812],
                         [-0.500000000000000,  -0.500000000000000,  -0.833333333333333,  -0.932187812187812],
                         [-0.166666666666667,  -0.833333333333333,  -0.833333333333333,  -0.932187812187812],
                         [-0.833333333333333,  -0.833333333333333,  -0.500000000000000,  -0.932187812187812],
                         [-0.833333333333333,  -0.500000000000000,  -0.833333333333333,  -0.932187812187812],
                         [-0.500000000000000,  -0.833333333333333,  -0.833333333333333,  -0.932187812187812],
                         [-0.833333333333333,  -0.833333333333333,  -0.833333333333333,  -0.932187812187812],
                         [-0.800000000000000,  -0.800000000000000,   0.400000000000000,   0.815498319838598],
                         [-0.800000000000000,  -0.400000000000000,   0.000000000000000,   0.815498319838598],
                         [-0.800000000000000,   0.000000000000000,  -0.400000000000000,   0.815498319838598],
                         [-0.800000000000000,   0.400000000000000,  -0.800000000000000,   0.815498319838598],
                         [-0.400000000000000,  -0.800000000000000,   0.000000000000000,   0.815498319838598],
                         [-0.400000000000000,  -0.400000000000000,  -0.400000000000000,   0.815498319838598],
                         [-0.400000000000000,   0.000000000000000,  -0.800000000000000,   0.815498319838598],
                         [ 0.000000000000000,  -0.800000000000000,  -0.400000000000000,   0.815498319838598],
                         [ 0.000000000000000,  -0.400000000000000,  -0.800000000000000,   0.815498319838598],
                         [ 0.400000000000000,  -0.800000000000000,  -0.800000000000000,   0.815498319838598],
                         [-0.800000000000000,  -0.800000000000000,   0.000000000000000,   0.815498319838598],
                         [-0.800000000000000,  -0.400000000000000,  -0.400000000000000,   0.815498319838598],
                         [-0.800000000000000,   0.000000000000000,  -0.800000000000000,   0.815498319838598],
                         [-0.400000000000000,  -0.800000000000000,  -0.400000000000000,   0.815498319838598],
                         [-0.400000000000000,  -0.400000000000000,  -0.800000000000000,   0.815498319838598],
                         [ 0.000000000000000,  -0.800000000000000,  -0.800000000000000,   0.815498319838598],
                         [-0.800000000000000,  -0.800000000000000,  -0.400000000000000,   0.815498319838598],
                         [-0.800000000000000,  -0.400000000000000,  -0.800000000000000,   0.815498319838598],
                         [-0.400000000000000,  -0.800000000000000,  -0.800000000000000,   0.815498319838598],
                         [-0.800000000000000,  -0.800000000000000,  -0.800000000000000,   0.815498319838598],
                         [-0.750000000000000,  -0.750000000000000,   0.250000000000000,  -0.280203089091978],
                         [-0.750000000000000,  -0.250000000000000,  -0.250000000000000,  -0.280203089091978],
                         [-0.750000000000000,   0.250000000000000,  -0.750000000000000,  -0.280203089091978],
                         [-0.250000000000000,  -0.750000000000000,  -0.250000000000000,  -0.280203089091978],
                         [-0.250000000000000,  -0.250000000000000,  -0.750000000000000,  -0.280203089091978],
                         [ 0.250000000000000,  -0.750000000000000,  -0.750000000000000,  -0.280203089091978],
                         [-0.750000000000000,  -0.750000000000000,  -0.250000000000000,  -0.280203089091978],
                         [-0.750000000000000,  -0.250000000000000,  -0.750000000000000,  -0.280203089091978],
                         [-0.250000000000000,  -0.750000000000000,  -0.750000000000000,  -0.280203089091978],
                         [-0.750000000000000,  -0.750000000000000,  -0.750000000000000,  -0.280203089091978],
                         [-0.666666666666667,  -0.666666666666667,   0.000000000000000,   0.032544642857143],
                         [-0.666666666666667,   0.000000000000000,  -0.666666666666667,   0.032544642857143],
                         [ 0.000000000000000,  -0.666666666666667,  -0.666666666666667,   0.032544642857143],
                         [-0.666666666666667,  -0.666666666666667,  -0.666666666666667,   0.032544642857143],
                         [-0.500000000000000,  -0.500000000000000,  -0.500000000000000,  -0.000752498530276]], dtype=np.float)
    elif Nord == 12:
        table = np.array([[-0.875000000000000,  -0.875000000000000,   0.625000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.625000000000000,   0.375000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.375000000000000,   0.125000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.125000000000000,  -0.125000000000000,  0.420407272132140],
                          [-0.875000000000000,   0.125000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.875000000000000,   0.375000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.875000000000000,   0.625000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.875000000000000,   0.375000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.625000000000000,   0.125000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.375000000000000,  -0.125000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.125000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.625000000000000,   0.125000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.625000000000000,   0.375000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.875000000000000,   0.125000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.625000000000000,  -0.125000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.375000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.125000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.375000000000000,   0.125000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.125000000000000,  -0.875000000000000,  -0.125000000000000,  0.420407272132140],
                          [-0.125000000000000,  -0.625000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.125000000000000,  -0.375000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.125000000000000,  -0.125000000000000,  -0.875000000000000,  0.420407272132140],
                          [ 0.125000000000000,  -0.875000000000000,  -0.375000000000000,  0.420407272132140],
                          [ 0.125000000000000,  -0.625000000000000,  -0.625000000000000,  0.420407272132140],
                          [ 0.125000000000000,  -0.375000000000000,  -0.875000000000000,  0.420407272132140],
                          [ 0.375000000000000,  -0.875000000000000,  -0.625000000000000,  0.420407272132140],
                          [ 0.375000000000000,  -0.625000000000000,  -0.875000000000000,  0.420407272132140],
                          [ 0.625000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.875000000000000,   0.375000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.625000000000000,   0.125000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.375000000000000,  -0.125000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.125000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.875000000000000,   0.125000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.875000000000000,   0.375000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.875000000000000,   0.125000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.625000000000000,  -0.125000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.375000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.125000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.625000000000000,   0.125000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.875000000000000,  -0.125000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.625000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.375000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.125000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.125000000000000,  -0.875000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.125000000000000,  -0.625000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.125000000000000,  -0.375000000000000,  -0.875000000000000,  0.420407272132140],
                          [ 0.125000000000000,  -0.875000000000000,  -0.625000000000000,  0.420407272132140],
                          [ 0.125000000000000,  -0.625000000000000,  -0.875000000000000,  0.420407272132140],
                          [ 0.375000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.875000000000000,   0.125000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.625000000000000,  -0.125000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.375000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.125000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.875000000000000,   0.125000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.875000000000000,  -0.125000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.625000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.375000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.125000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.875000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.625000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.375000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.125000000000000,  -0.875000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.125000000000000,  -0.625000000000000,  -0.875000000000000,  0.420407272132140],
                          [ 0.125000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.875000000000000,  -0.125000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.625000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.375000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.125000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.875000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.625000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.375000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.875000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.625000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.125000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.875000000000000,  -0.375000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.625000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.375000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.875000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.625000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.375000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.875000000000000,  -0.625000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.625000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.625000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.875000000000000,  -0.875000000000000,  -0.875000000000000,  0.420407272132140],
                          [-0.857142857142857,  -0.857142857142857,   0.571428571428571, -1.185481802234117],
                          [-0.857142857142857,  -0.571428571428571,   0.285714285714286, -1.185481802234117],
                          [-0.857142857142857,  -0.285714285714286,   0.000000000000000, -1.185481802234117],
                          [-0.857142857142857,   0.000000000000000,  -0.285714285714286, -1.185481802234117],
                          [-0.857142857142857,   0.285714285714286,  -0.571428571428571, -1.185481802234117],
                          [-0.857142857142857,   0.571428571428571,  -0.857142857142857, -1.185481802234117],
                          [-0.571428571428571,  -0.857142857142857,   0.285714285714286, -1.185481802234117],
                          [-0.571428571428571,  -0.571428571428571,   0.000000000000000, -1.185481802234117],
                          [-0.571428571428571,  -0.285714285714286,  -0.285714285714286, -1.185481802234117],
                          [-0.571428571428571,   0.000000000000000,  -0.571428571428571, -1.185481802234117],
                          [-0.571428571428571,   0.285714285714286,  -0.857142857142857, -1.185481802234117],
                          [-0.285714285714286,  -0.857142857142857,   0.000000000000000, -1.185481802234117],
                          [-0.285714285714286,  -0.571428571428571,  -0.285714285714286, -1.185481802234117],
                          [-0.285714285714286,  -0.285714285714286,  -0.571428571428571, -1.185481802234117],
                          [-0.285714285714286,   0.000000000000000,  -0.857142857142857, -1.185481802234117],
                          [ 0.000000000000000,  -0.857142857142857,  -0.285714285714286, -1.185481802234117],
                          [ 0.000000000000000,  -0.571428571428571,  -0.571428571428571, -1.185481802234117],
                          [ 0.000000000000000,  -0.285714285714286,  -0.857142857142857, -1.185481802234117],
                          [ 0.285714285714286,  -0.857142857142857,  -0.571428571428571, -1.185481802234117],
                          [ 0.285714285714286,  -0.571428571428571,  -0.857142857142857, -1.185481802234117],
                          [ 0.571428571428571,  -0.857142857142857,  -0.857142857142857, -1.185481802234117],
                          [-0.857142857142857,  -0.857142857142857,   0.285714285714286, -1.185481802234117],
                          [-0.857142857142857,  -0.571428571428571,   0.000000000000000, -1.185481802234117],
                          [-0.857142857142857,  -0.285714285714286,  -0.285714285714286, -1.185481802234117],
                          [-0.857142857142857,   0.000000000000000,  -0.571428571428571, -1.185481802234117],
                          [-0.857142857142857,   0.285714285714286,  -0.857142857142857, -1.185481802234117],
                          [-0.571428571428571,  -0.857142857142857,   0.000000000000000, -1.185481802234117],
                          [-0.571428571428571,  -0.571428571428571,  -0.285714285714286, -1.185481802234117],
                          [-0.571428571428571,  -0.285714285714286,  -0.571428571428571, -1.185481802234117],
                          [-0.571428571428571,   0.000000000000000,  -0.857142857142857, -1.185481802234117],
                          [-0.285714285714286,  -0.857142857142857,  -0.285714285714286, -1.185481802234117],
                          [-0.285714285714286,  -0.571428571428571,  -0.571428571428571, -1.185481802234117],
                          [-0.285714285714286,  -0.285714285714286,  -0.857142857142857, -1.185481802234117],
                          [ 0.000000000000000,  -0.857142857142857,  -0.571428571428571, -1.185481802234117],
                          [ 0.000000000000000,  -0.571428571428571,  -0.857142857142857, -1.185481802234117],
                          [ 0.285714285714286,  -0.857142857142857,  -0.857142857142857, -1.185481802234117],
                          [-0.857142857142857,  -0.857142857142857,   0.000000000000000, -1.185481802234117],
                          [-0.857142857142857,  -0.571428571428571,  -0.285714285714286, -1.185481802234117],
                          [-0.857142857142857,  -0.285714285714286,  -0.571428571428571, -1.185481802234117],
                          [-0.857142857142857,   0.000000000000000,  -0.857142857142857, -1.185481802234117],
                          [-0.571428571428571,  -0.857142857142857,  -0.285714285714286, -1.185481802234117],
                          [-0.571428571428571,  -0.571428571428571,  -0.571428571428571, -1.185481802234117],
                          [-0.571428571428571,  -0.285714285714286,  -0.857142857142857, -1.185481802234117],
                          [-0.285714285714286,  -0.857142857142857,  -0.571428571428571, -1.185481802234117],
                          [-0.285714285714286,  -0.571428571428571,  -0.857142857142857, -1.185481802234117],
                          [ 0.000000000000000,  -0.857142857142857,  -0.857142857142857, -1.185481802234117],
                          [-0.857142857142857,  -0.857142857142857,  -0.285714285714286, -1.185481802234117],
                          [-0.857142857142857,  -0.571428571428571,  -0.571428571428571, -1.185481802234117],
                          [-0.857142857142857,  -0.285714285714286,  -0.857142857142857, -1.185481802234117],
                          [-0.571428571428571,  -0.857142857142857,  -0.571428571428571, -1.185481802234117],
                          [-0.571428571428571,  -0.571428571428571,  -0.857142857142857, -1.185481802234117],
                          [-0.285714285714286,  -0.857142857142857,  -0.857142857142857, -1.185481802234117],
                          [-0.857142857142857,  -0.857142857142857,  -0.571428571428571, -1.185481802234117],
                          [-0.857142857142857,  -0.571428571428571,  -0.857142857142857, -1.185481802234117],
                          [-0.571428571428571,  -0.857142857142857,  -0.857142857142857, -1.185481802234117],
                          [-0.857142857142857,  -0.857142857142857,  -0.857142857142857, -1.185481802234117],
                          [-0.833333333333333,  -0.833333333333333,   0.500000000000000,  1.198527187098616],
                          [-0.833333333333333,  -0.500000000000000,   0.166666666666667,  1.198527187098616],
                          [-0.833333333333333,  -0.166666666666667,  -0.166666666666667,  1.198527187098616],
                          [-0.833333333333333,   0.166666666666667,  -0.500000000000000,  1.198527187098616],
                          [-0.833333333333333,   0.500000000000000,  -0.833333333333333,  1.198527187098616],
                          [-0.500000000000000,  -0.833333333333333,   0.166666666666667,  1.198527187098616],
                          [-0.500000000000000,  -0.500000000000000,  -0.166666666666667,  1.198527187098616],
                          [-0.500000000000000,  -0.166666666666667,  -0.500000000000000,  1.198527187098616],
                          [-0.500000000000000,   0.166666666666667,  -0.833333333333333,  1.198527187098616],
                          [-0.166666666666667,  -0.833333333333333,  -0.166666666666667,  1.198527187098616],
                          [-0.166666666666667,  -0.500000000000000,  -0.500000000000000,  1.198527187098616],
                          [-0.166666666666667,  -0.166666666666667,  -0.833333333333333,  1.198527187098616],
                          [ 0.166666666666667,  -0.833333333333333,  -0.500000000000000,  1.198527187098616],
                          [ 0.166666666666667,  -0.500000000000000,  -0.833333333333333,  1.198527187098616],
                          [ 0.500000000000000,  -0.833333333333333,  -0.833333333333333,  1.198527187098616],
                          [-0.833333333333333,  -0.833333333333333,   0.166666666666667,  1.198527187098616],
                          [-0.833333333333333,  -0.500000000000000,  -0.166666666666667,  1.198527187098616],
                          [-0.833333333333333,  -0.166666666666667,  -0.500000000000000,  1.198527187098616],
                          [-0.833333333333333,   0.166666666666667,  -0.833333333333333,  1.198527187098616],
                          [-0.500000000000000,  -0.833333333333333,  -0.166666666666667,  1.198527187098616],
                          [-0.500000000000000,  -0.500000000000000,  -0.500000000000000,  1.198527187098616],
                          [-0.500000000000000,  -0.166666666666667,  -0.833333333333333,  1.198527187098616],
                          [-0.166666666666667,  -0.833333333333333,  -0.500000000000000,  1.198527187098616],
                          [-0.166666666666667,  -0.500000000000000,  -0.833333333333333,  1.198527187098616],
                          [ 0.166666666666667,  -0.833333333333333,  -0.833333333333333,  1.198527187098616],
                          [-0.833333333333333,  -0.833333333333333,  -0.166666666666667,  1.198527187098616],
                          [-0.833333333333333,  -0.500000000000000,  -0.500000000000000,  1.198527187098616],
                          [-0.833333333333333,  -0.166666666666667,  -0.833333333333333,  1.198527187098616],
                          [-0.500000000000000,  -0.833333333333333,  -0.500000000000000,  1.198527187098616],
                          [-0.500000000000000,  -0.500000000000000,  -0.833333333333333,  1.198527187098616],
                          [-0.166666666666667,  -0.833333333333333,  -0.833333333333333,  1.198527187098616],
                          [-0.833333333333333,  -0.833333333333333,  -0.500000000000000,  1.198527187098616],
                          [-0.833333333333333,  -0.500000000000000,  -0.833333333333333,  1.198527187098616],
                          [-0.500000000000000,  -0.833333333333333,  -0.833333333333333,  1.198527187098616],
                          [-0.833333333333333,  -0.833333333333333,  -0.833333333333333,  1.198527187098616],
                          [-0.800000000000000,  -0.800000000000000,   0.400000000000000, -0.522755333229870],
                          [-0.800000000000000,  -0.400000000000000,   0.000000000000000, -0.522755333229870],
                          [-0.800000000000000,   0.000000000000000,  -0.400000000000000, -0.522755333229870],
                          [-0.800000000000000,   0.400000000000000,  -0.800000000000000, -0.522755333229870],
                          [-0.400000000000000,  -0.800000000000000,   0.000000000000000, -0.522755333229870],
                          [-0.400000000000000,  -0.400000000000000,  -0.400000000000000, -0.522755333229870],
                          [-0.400000000000000,   0.000000000000000,  -0.800000000000000, -0.522755333229870],
                          [ 0.000000000000000,  -0.800000000000000,  -0.400000000000000, -0.522755333229870],
                          [ 0.000000000000000,  -0.400000000000000,  -0.800000000000000, -0.522755333229870],
                          [ 0.400000000000000,  -0.800000000000000,  -0.800000000000000, -0.522755333229870],
                          [-0.800000000000000,  -0.800000000000000,   0.000000000000000, -0.522755333229870],
                          [-0.800000000000000,  -0.400000000000000,  -0.400000000000000, -0.522755333229870],
                          [-0.800000000000000,   0.000000000000000 , -0.800000000000000, -0.522755333229870],
                          [-0.400000000000000,  -0.800000000000000,  -0.400000000000000, -0.522755333229870],
                          [-0.400000000000000,  -0.400000000000000,  -0.800000000000000, -0.522755333229870],
                          [ 0.000000000000000,  -0.800000000000000 , -0.800000000000000, -0.522755333229870],
                          [-0.800000000000000,  -0.800000000000000,  -0.400000000000000, -0.522755333229870],
                          [-0.800000000000000,  -0.400000000000000,  -0.800000000000000, -0.522755333229870],
                          [-0.400000000000000,  -0.800000000000000,  -0.800000000000000, -0.522755333229870],
                          [-0.800000000000000,  -0.800000000000000,  -0.800000000000000, -0.522755333229870],
                          [-0.750000000000000,  -0.750000000000000,   0.250000000000000,  0.093401029697326],
                          [-0.750000000000000,  -0.250000000000000,  -0.250000000000000,  0.093401029697326],
                          [-0.750000000000000,   0.250000000000000,  -0.750000000000000,  0.093401029697326],
                          [-0.250000000000000,  -0.750000000000000,  -0.250000000000000,  0.093401029697326],
                          [-0.250000000000000,  -0.250000000000000,  -0.750000000000000,  0.093401029697326],
                          [ 0.250000000000000,  -0.750000000000000 , -0.750000000000000,  0.093401029697326],
                          [-0.750000000000000,  -0.750000000000000,  -0.250000000000000,  0.093401029697326],
                          [-0.750000000000000,  -0.250000000000000,  -0.750000000000000,  0.093401029697326],
                          [-0.250000000000000,  -0.750000000000000,  -0.750000000000000,  0.093401029697326],
                          [-0.750000000000000,  -0.750000000000000,  -0.750000000000000,  0.093401029697326],
                          [-0.666666666666667,  -0.666666666666667,   0.000000000000000, -0.005325487012987],
                          [-0.666666666666667,   0.000000000000000,  -0.666666666666667, -0.005325487012987],
                          [ 0.000000000000000,  -0.666666666666667,  -0.666666666666667, -0.005325487012987],
                          [-0.666666666666667,  -0.666666666666667,  -0.666666666666667, -0.005325487012987],
                          [-0.500000000000000,  -0.500000000000000,  -0.500000000000000,  0.000050166568685]], dtype=np.float)

    else:
        Print.master('     Orders higher than 6 are not supported by PETGEM')
        exit(-1)

    # Renormalization (reference tetrahedron is defined from [-1, -1, -1] to lambda1+lambda2+lambda3=-1).
    weights = table[:, 3]/8.
    points = np.zeros((len(weights), 3), dtype=np.float)
    points[:, 0] =  (1 + table[:, 1])/2.
    points[:, 1] = -(1 + np.sum(table[:, 0:3], axis=1))/2.
    points[:, 2] =  (1 + table[:,0])/2.

    return points, weights


def compute2DGaussPoints(Nord):
    """Compute 2D gauss points for high-order nédélec elements.

    :param int Nord: polynomial order of nedelec basis functions
    :return: coordinates of gauss points and its weights
    :rtype: ndarray.
    """

    if Nord == 0 or Nord == 1:
        table = np.array([[-0.333333333333333,  -0.333333333333333,  2.000000000000000]], dtype=np.float)

    elif Nord == 2:
        table = np.array([[-0.666666666666667,  -0.666666666666667,  0.666666666666667],
                          [-0.666666666666667,   0.333333333333333,  0.666666666666667],
                          [ 0.333333333333333,  -0.666666666666667,  0.666666666666667]], dtype=np.float)

    elif Nord == 3:
        table = np.array([[-0.333333333333333,  -0.333333333333333, -1.125000000000000],
                          [-0.600000000000000,  -0.600000000000000,  1.041666666666667],
                          [-0.600000000000000,   0.200000000000000,  1.041666666666667],
                          [ 0.200000000000000,  -0.600000000000000,  1.041666666666667]], dtype=np.float)

    elif Nord == 4:
        table = np.array([[-0.108103018168070,  -0.108103018168070,  0.446763179356022],
                          [-0.108103018168070,  -0.783793963663860,  0.446763179356022],
                          [-0.783793963663860,  -0.108103018168070,  0.446763179356022],
                          [-0.816847572980458,  -0.816847572980458,  0.219903487310644],
                          [-0.816847572980458,   0.633695145960918,  0.219903487310644],
                          [ 0.633695145960918,  -0.816847572980458,  0.219903487310644]], dtype=np.float)

    elif Nord == 5:
        table = np.array([[-0.333333333333333,  -0.333333333333333,  0.450000000000000],
                          [-0.059715871789770,  -0.059715871789770,  0.264788305577012],
                          [-0.059715871789770,  -0.880568256420460,  0.264788305577012],
                          [-0.880568256420460,  -0.059715871789770,  0.264788305577012],
                          [-0.797426985353088,  -0.797426985353088,  0.251878361089654],
                          [-0.797426985353088,   0.594853970706174,  0.251878361089654],
                          [ 0.594853970706174,  -0.797426985353088,  0.251878361089654]], dtype=np.float)

    elif Nord == 6:
        table = np.array([[-0.501426509658180, -0.501426509658180, 0.233572551452758],
                          [-0.501426509658180,  0.002853019316358, 0.233572551452758],
                          [ 0.002853019316358, -0.501426509658180, 0.233572551452758],
                          [-0.873821971016996, -0.873821971016996, 0.101689812740414],
                          [-0.873821971016996,  0.747643942033992, 0.101689812740414],
                          [ 0.747643942033992, -0.873821971016996, 0.101689812740414],
                          [-0.379295097932432,  0.273004998242798, 0.165702151236748],
                          [ 0.273004998242798, -0.893709900310366, 0.165702151236748],
                          [-0.893709900310366, -0.379295097932432, 0.165702151236748],
                          [-0.379295097932432, -0.893709900310366, 0.165702151236748],
                          [ 0.273004998242798, -0.379295097932432, 0.165702151236748],
                          [-0.893709900310366,  0.273004998242798, 0.165702151236748]], dtype=np.float)

    elif Nord == 7:
        table = np.array([[-0.333333333333333, -0.333333333333333, -0.299140088935364],
                          [-0.479308067841920, -0.479308067841920,  0.351230514866416],
                          [-0.479308067841920, -0.041383864316160,  0.351230514866416],
                          [-0.041383864316160, -0.479308067841920,  0.351230514866416],
                          [-0.869739794195568, -0.869739794195568,  0.106694471217676],
                          [-0.869739794195568,  0.739479588391136,  0.106694471217676],
                          [ 0.739479588391136, -0.869739794195568,  0.106694471217676],
                          [-0.374269007990252,  0.276888377139620,  0.154227521780514],
                          [ 0.276888377139620, -0.902619369149368,  0.154227521780514],
                          [-0.902619369149368, -0.374269007990252,  0.154227521780514],
                          [-0.374269007990252, -0.902619369149368,  0.154227521780514],
                          [ 0.276888377139620, -0.374269007990252,  0.154227521780514],
                          [-0.902619369149368,  0.276888377139620,  0.154227521780514]], dtype=np.float)

    elif Nord == 8:
        table = np.array([[-0.333333333333333, -0.333333333333333,  0.288631215355574],
                          [-0.081414823414554, -0.081414823414554,  0.190183268534570],
                          [-0.081414823414554, -0.837170353170892,  0.190183268534570],
                          [-0.837170353170892, -0.081414823414554,  0.190183268534570],
                          [-0.658861384496480, -0.658861384496480,  0.206434741069436],
                          [-0.658861384496480,  0.317722768992960,  0.206434741069436],
                          [ 0.317722768992960, -0.658861384496480,  0.206434741069436],
                          [-0.898905543365938, -0.898905543365938,  0.064916995246396],
                          [-0.898905543365938,  0.797811086731876,  0.064916995246395],
                          [ 0.797811086731876, -0.898905543365938,  0.064916995246396],
                          [-0.473774340730724,  0.456984785910808,  0.054460628348870],
                          [ 0.456984785910808, -0.983210445180084,  0.054460628348870],
                          [-0.983210445180084, -0.473774340730724,  0.054460628348870],
                          [-0.473774340730724, -0.983210445180084,  0.054460628348870],
                          [ 0.456984785910808, -0.473774340730724,  0.054460628348870],
                          [-0.983210445180084,  0.456984785910808,  0.054460628348870]], dtype=np.float)

    elif Nord == 9:
        table = np.array([[-0.333333333333333,  -0.333333333333333,  0.194271592565598],
                          [-0.020634961602524,  -0.020634961602524,  0.062669400454278],
                          [-0.020634961602524,  -0.958730076794950,  0.062669400454278],
                          [-0.958730076794950,  -0.020634961602524,  0.062669400454278],
                          [-0.125820817014126,  -0.125820817014126,  0.155655082009548],
                          [-0.125820817014126,  -0.748358365971746,  0.155655082009548],
                          [-0.748358365971746,  -0.125820817014126,  0.155655082009548],
                          [-0.623592928761934,  -0.623592928761934,  0.159295477854420],
                          [-0.623592928761934,   0.247185857523870,  0.159295477854420],
                          [ 0.247185857523870,  -0.623592928761934,  0.159295477854420],
                          [-0.910540973211094,  -0.910540973211094,  0.051155351317396],
                          [-0.910540973211094,   0.821081946422190,  0.051155351317396],
                          [ 0.821081946422190,  -0.910540973211094,  0.051155351317396],
                          [-0.556074021678468,   0.482397197568996,  0.086567078754578],
                          [ 0.482397197568996,  -0.926323175890528,  0.086567078754578],
                          [-0.926323175890528,  -0.556074021678468,  0.086567078754578],
                          [-0.556074021678468,  -0.926323175890528,  0.086567078754578],
                          [ 0.482397197568996,  -0.556074021678468,  0.086567078754578],
                          [-0.926323175890528,   0.482397197568996,  0.086567078754578]], dtype=np.float)

    elif Nord == 10:
        table = np.array([[-0.333333333333333, -0.333333333333333,  0.181635980765508],
                          [-0.028844733232686, -0.028844733232686,  0.073451915512934],
                          [-0.028844733232686, -0.942310533534630,  0.073451915512934],
                          [-0.942310533534630, -0.028844733232686,  0.073451915512934],
                          [-0.781036849029926, -0.781036849029926,  0.090642118871056],
                          [-0.781036849029926,  0.562073698059852,  0.090642118871056],
                          [ 0.562073698059852, -0.781036849029926,  0.090642118871056],
                          [-0.384120322471758,  0.100705883641998,  0.145515833690840],
                          [ 0.100705883641998, -0.716585561170240,  0.145515833690840],
                          [-0.716585561170240, -0.384120322471758,  0.145515833690840],
                          [-0.384120322471758, -0.716585561170240,  0.145515833690840],
                          [ 0.100705883641998, -0.384120322471758,  0.145515833690840],
                          [-0.716585561170240,  0.100705883641998,  0.145515833690840],
                          [-0.506654878720194,  0.456647809194822,  0.056654485062114],
                          [ 0.456647809194822, -0.949992930474628,  0.056654485062114],
                          [-0.949992930474628, -0.506654878720194,  0.056654485062114],
                          [-0.506654878720194, -0.949992930474628,  0.056654485062114],
                          [ 0.456647809194822, -0.506654878720194,  0.056654485062114],
                          [-0.949992930474628,  0.456647809194822,  0.056654485062114],
                          [-0.866393497975600,  0.847311867175000,  0.018843333927466],
                          [ 0.847311867175000, -0.980918369199402,  0.018843333927466],
                          [-0.980918369199402, -0.866393497975600,  0.018843333927466],
                          [-0.866393497975600, -0.980918369199402,  0.018843333927466],
                          [ 0.847311867175000, -0.866393497975600,  0.018843333927466],
                          [-0.980918369199402,  0.847311867175000,  0.018843333927466]], dtype=np.float)

    elif Nord == 11:
        table = np.array([[0.069222096541516,   0.069222096541516, 0.001854012657922],
                         [ 0.069222096541516,  -1.138444193083034, 0.001854012657922],
                         [-1.138444193083034,   0.069222096541516, 0.001854012657922],
                         [-0.202061394068290,  -0.202061394068290, 0.154299069829626],
                         [-0.202061394068290,  -0.595877211863420, 0.154299069829626],
                         [-0.595877211863420,  -0.202061394068290, 0.154299069829626],
                         [-0.593380199137436,  -0.593380199137436, 0.118645954761548],
                         [-0.593380199137436,   0.186760398274870, 0.118645954761548],
                         [ 0.186760398274870,  -0.593380199137436, 0.118645954761548],
                         [-0.761298175434838,  -0.761298175434838, 0.072369081006836],
                         [-0.761298175434838,   0.522596350869674, 0.072369081006836],
                         [ 0.522596350869674,  -0.761298175434838, 0.072369081006836],
                         [-0.935270103777448,  -0.935270103777448, 0.027319462005356],
                         [-0.935270103777448,   0.870540207554896, 0.027319462005356],
                         [ 0.870540207554896,  -0.935270103777448, 0.027319462005356],
                         [-0.286758703477414,   0.186402426856426, 0.104674223924408],
                         [ 0.186402426856426,  -0.899643723379010, 0.104674223924408],
                         [-0.899643723379010,  -0.286758703477414, 0.104674223924408],
                         [-0.286758703477414,  -0.899643723379010, 0.104674223924408],
                         [ 0.186402426856426,  -0.286758703477414, 0.104674223924408],
                         [-0.899643723379010,   0.186402426856426, 0.104674223924408],
                         [-0.657022039391916,   0.614978006319584, 0.041415319278282],
                         [ 0.614978006319584,  -0.957955966927668, 0.041415319278282],
                         [-0.957955966927668,  -0.657022039391916, 0.041415319278282],
                         [-0.657022039391916,  -0.957955966927668, 0.041415319278282],
                         [ 0.614978006319584,  -0.657022039391916, 0.041415319278282],
                         [-0.957955966927668,   0.614978006319584, 0.041415319278282]], dtype=np.float)

    elif Nord == 12:
        table = np.array([[-0.023565220452390,  -0.023565220452390,  0.051462132880910],
                          [-0.023565220452390,  -0.952869559095220,  0.051462132880910],
                          [-0.952869559095220,  -0.023565220452390,  0.051462132880910],
                          [-0.120551215411080,  -0.120551215411080,  0.087385089076076],
                          [-0.120551215411080,  -0.758897569177842,  0.087385089076076],
                          [-0.758897569177842,  -0.120551215411080,  0.087385089076076],
                          [-0.457579229975768,  -0.457579229975768,  0.125716448435770],
                          [-0.457579229975768,  -0.084841540048464,  0.125716448435770],
                          [-0.084841540048464,  -0.457579229975768,  0.125716448435770],
                          [-0.744847708916828,  -0.744847708916828,  0.069592225861418],
                          [-0.744847708916828,   0.489695417833656,  0.069592225861418],
                          [ 0.489695417833656,  -0.744847708916828,  0.069592225861418],
                          [-0.957365299093580,  -0.957365299093580,  0.012332522103118],
                          [-0.957365299093580,   0.914730598187158,  0.012332522103118],
                          [ 0.914730598187158,  -0.957365299093580,  0.012332522103118],
                          [-0.448573460628972,   0.217886471559576,  0.080743115532762],
                          [ 0.217886471559576,  -0.769313010930604,  0.080743115532762],
                          [-0.769313010930604,  -0.448573460628972,  0.080743115532762],
                          [-0.448573460628972,  -0.769313010930604,  0.080743115532762],
                          [ 0.217886471559576,  -0.448573460628972,  0.080743115532762],
                          [-0.769313010930604,   0.217886471559576,  0.080743115532762],
                          [-0.437348838020120,   0.391672173575606,  0.044713546404606],
                          [ 0.391672173575606,  -0.954323335555486,  0.044713546404606],
                          [-0.954323335555486,  -0.437348838020120,  0.044713546404606],
                          [-0.437348838020120,  -0.954323335555486,  0.044713546404606],
                          [ 0.391672173575606,  -0.437348838020120,  0.044713546404606],
                          [-0.954323335555486,   0.391672173575606,  0.044713546404606],
                          [-0.767496168184806,   0.716028067088146,  0.034632462217318],
                          [ 0.716028067088146,  -0.948531898903340,  0.034632462217318],
                          [-0.948531898903340,  -0.767496168184806,  0.034632462217318],
                          [-0.767496168184806,  -0.948531898903340,  0.034632462217318],
                          [ 0.716028067088146,  -0.767496168184806,  0.034632462217318],
                          [-0.948531898903340,   0.716028067088146,  0.034632462217318]], dtype=np.float)

    elif Nord == 13:
        table = np.array([[-0.333333333333333,  -0.333333333333333,  0.105041846801604],
                          [-0.009903630120590,  -0.009903630120590,  0.022560290418660],
                          [-0.009903630120590,  -0.980192739758818,  0.022560290418660],
                          [-0.980192739758818,  -0.009903630120590,  0.022560290418660],
                          [-0.062566729780852,  -0.062566729780852,  0.062847036724908],
                          [-0.062566729780852,  -0.874866540438296,  0.062847036724908],
                          [-0.874866540438296,  -0.062566729780852,  0.062847036724908],
                          [-0.170957326397446,  -0.170957326397446,  0.094145005008388],
                          [-0.170957326397446,  -0.658085347205106,  0.094145005008388],
                          [-0.658085347205106,  -0.170957326397446,  0.094145005008388],
                          [-0.541200855914338,  -0.541200855914338,  0.094727173072710],
                          [-0.541200855914338,   0.082401711828674,  0.094727173072710],
                          [ 0.082401711828674,  -0.541200855914338,  0.094727173072710],
                          [-0.771151009607340,  -0.771151009607340,  0.062335058091588],
                          [-0.771151009607340,   0.542302019214680,  0.062335058091588],
                          [ 0.542302019214680,  -0.771151009607340,  0.062335058091588],
                          [-0.950377217273082,  -0.950377217273082,  0.015951542930148],
                          [-0.950377217273082,   0.900754434546164,  0.015951542930148],
                          [ 0.900754434546164,  -0.950377217273082,  0.015951542930148],
                          [-0.462410005882478,   0.272702349123320,  0.073696805457464],
                          [ 0.272702349123320,  -0.810292343240842,  0.073696805457464],
                          [-0.810292343240842,  -0.462410005882478,  0.073696805457464],
                          [-0.462410005882478,  -0.810292343240842,  0.073696805457464],
                          [ 0.272702349123320,  -0.462410005882478,  0.073696805457464],
                          [-0.810292343240842,   0.272702349123320,  0.073696805457464],
                          [-0.416539866531424,   0.380338319973810,  0.034802926607644],
                          [ 0.380338319973810,  -0.963798453442386,  0.034802926607644],
                          [-0.963798453442386,  -0.416539866531424,  0.034802926607644],
                          [-0.416539866531424,  -0.963798453442386,  0.034802926607644],
                          [ 0.380338319973810,  -0.416539866531424,  0.034802926607644],
                          [-0.963798453442386,   0.380338319973810,  0.034802926607644],
                          [-0.747285229016662,   0.702819075668482,  0.031043573678090],
                          [ 0.702819075668482,  -0.955533846651820,  0.031043573678090],
                          [-0.955533846651820,  -0.747285229016662,  0.031043573678090],
                          [-0.747285229016662,  -0.955533846651820,  0.031043573678090],
                          [ 0.702819075668482,  -0.747285229016662,  0.031043573678090],
                          [-0.955533846651820,   0.702819075668482,  0.031043573678090]], dtype=np.float)

    elif Nord == 14:
        table = np.array([[-0.022072179275642, -0.022072179275642,  0.043767162738858],
                          [-0.022072179275642, -0.955855641448714,  0.043767162738858],
                          [-0.955855641448714, -0.022072179275642,  0.043767162738858],
                          [-0.164710561319092, -0.164710561319092,  0.065576707088250],
                          [-0.164710561319092, -0.670578877361816,  0.065576707088250],
                          [-0.670578877361816, -0.164710561319092,  0.065576707088250],
                          [-0.453044943382322, -0.453044943382322,  0.103548209014584],
                          [-0.453044943382322, -0.093910113235354,  0.103548209014584],
                          [-0.093910113235354, -0.453044943382322,  0.103548209014584],
                          [-0.645588935174914, -0.645588935174914,  0.084325177473986],
                          [-0.645588935174914,  0.291177870349826,  0.084325177473986],
                          [ 0.291177870349826, -0.645588935174914,  0.084325177473986],
                          [-0.876400233818254, -0.876400233818254,  0.028867399339554],
                          [-0.876400233818254,  0.752800467636510,  0.028867399339554],
                          [ 0.752800467636510, -0.876400233818254,  0.028867399339554],
                          [-0.961218077502598, -0.961218077502598,  0.009846807204800],
                          [-0.961218077502598,  0.922436155005196,  0.009846807204800],
                          [ 0.922436155005196, -0.961218077502598,  0.009846807204800],
                          [-0.655466624357288,  0.541217109549992,  0.049331506425128],
                          [ 0.541217109549992, -0.885750485192704,  0.049331506425128],
                          [-0.885750485192704, -0.655466624357288,  0.049331506425128],
                          [-0.655466624357288, -0.885750485192704,  0.049331506425128],
                          [ 0.541217109549992, -0.655466624357288,  0.049331506425128],
                          [-0.885750485192704,  0.541217109549992,  0.049331506425128],
                          [-0.326277080407310,  0.140444581693366,  0.077143021574122],
                          [ 0.140444581693366, -0.814167501286056,  0.077143021574122],
                          [-0.814167501286056, -0.326277080407310,  0.077143021574122],
                          [-0.326277080407310, -0.814167501286056,  0.077143021574122],
                          [ 0.140444581693366, -0.326277080407310,  0.077143021574122],
                          [-0.814167501286056,  0.140444581693366,  0.077143021574122],
                          [-0.403254235727484,  0.373960335616176,  0.028872616227068],
                          [ 0.373960335616176, -0.970706099888692,  0.028872616227068],
                          [-0.970706099888692, -0.403254235727484,  0.028872616227068],
                          [-0.403254235727484, -0.970706099888692,  0.028872616227068],
                          [ 0.373960335616176, -0.403254235727484,  0.028872616227068],
                          [-0.970706099888692,  0.373960335616176,  0.028872616227068],
                          [-0.762051004606086,  0.759514342740342,  0.010020457677002],
                          [ 0.759514342740342, -0.997463338134256,  0.010020457677002],
                          [-0.997463338134256, -0.762051004606086,  0.010020457677002],
                          [-0.762051004606086, -0.997463338134256,  0.010020457677002],
                          [ 0.759514342740342, -0.762051004606086,  0.010020457677002],
                          [-0.997463338134256,  0.759514342740342,  0.010020457677002]], dtype=np.float)

    elif Nord == 15:
        table = np.array([[0.013945833716486,  0.013945833716486, 0.003833751285698],
                         [ 0.013945833716486, -1.027891667432972, 0.003833751285698],
                         [-1.027891667432972,  0.013945833716486, 0.003833751285698],
                         [-0.137187291433954, -0.137187291433954, 0.088498054542290],
                         [-0.137187291433954, -0.725625417132090, 0.088498054542290],
                         [-0.725625417132090, -0.137187291433954, 0.088498054542290],
                         [-0.444612710305712, -0.444612710305712, 0.102373097437704],
                         [-0.444612710305712, -0.110774579388578, 0.102373097437704],
                         [-0.110774579388578, -0.444612710305712, 0.102373097437704],
                         [-0.747070217917492, -0.747070217917492, 0.047375471741376],
                         [-0.747070217917492,  0.494140435834984, 0.047375471741376],
                         [ 0.494140435834984, -0.747070217917492, 0.047375471741376],
                         [-0.858383228050628, -0.858383228050628, 0.026579551380042],
                         [-0.858383228050628,  0.716766456101256, 0.026579551380042],
                         [ 0.716766456101256, -0.858383228050628, 0.026579551380042],
                         [-0.962069659517854, -0.962069659517854, 0.009497833216384],
                         [-0.962069659517854,  0.924139319035706, 0.009497833216384],
                         [ 0.924139319035706, -0.962069659517854, 0.009497833216384],
                         [-0.477377257719826,  0.209908933786582, 0.077100145199186],
                         [ 0.209908933786582, -0.732531676066758, 0.077100145199186],
                         [-0.732531676066758, -0.477377257719826, 0.077100145199186],
                         [-0.477377257719826, -0.732531676066758, 0.077100145199186],
                         [ 0.209908933786582, -0.477377257719826, 0.077100145199186],
                         [-0.732531676066758,  0.209908933786582, 0.077100145199186],
                         [-0.223906465819462,  0.151173111025628, 0.054431628641248],
                         [ 0.151173111025628, -0.927266645206166, 0.054431628641248],
                         [-0.927266645206166, -0.223906465819462, 0.054431628641248],
                         [-0.223906465819462, -0.927266645206166, 0.054431628641248],
                         [ 0.151173111025628, -0.223906465819462, 0.054431628641248],
                         [-0.927266645206166,  0.151173111025628, 0.054431628641248],
                         [-0.428575559900168,  0.448925326153310, 0.004364154733594],
                         [ 0.448925326153310, -1.020349766253142, 0.004364154733594],
                         [-1.020349766253142, -0.428575559900168, 0.004364154733594],
                         [-0.428575559900168, -1.020349766253142, 0.004364154733594],
                         [ 0.448925326153310, -0.428575559900168, 0.004364154733594],
                         [-1.020349766253142,  0.448925326153310, 0.004364154733594],
                         [-0.568800671855432,  0.495112932103676, 0.043010639695462],
                         [ 0.495112932103676, -0.926312260248244, 0.043010639695462],
                         [-0.926312260248244, -0.568800671855432, 0.043010639695462],
                         [-0.568800671855432, -0.926312260248244, 0.043010639695462],
                         [ 0.495112932103676, -0.568800671855432, 0.043010639695462],
                         [-0.926312260248244,  0.495112932103676, 0.043010639695462],
                         [-0.792848766847228,  0.767929148184832, 0.015347885262098],
                         [ 0.767929148184832, -0.975080381337602, 0.015347885262098],
                         [-0.975080381337602, -0.792848766847228, 0.015347885262098],
                         [-0.792848766847228, -0.975080381337602, 0.015347885262098],
                         [ 0.767929148184832, -0.792848766847228, 0.015347885262098],
                         [-0.975080381337602,  0.767929148184832, 0.015347885262098]], dtype=np.float)

    elif Nord == 16:
        table = np.array([[-0.333333333333333, -0.333333333333333,  0.093751394855284],
                          [-0.005238916103124, -0.005238916103124,  0.012811757157170],
                          [-0.005238916103124, -0.989522167793754,  0.012811757157170],
                          [-0.989522167793754, -0.005238916103124,  0.012811757157170],
                          [-0.173061122901296, -0.173061122901296,  0.083420593478774],
                          [-0.173061122901296, -0.653877754197410,  0.083420593478774],
                          [-0.653877754197410, -0.173061122901296,  0.083420593478774],
                          [-0.059082801866018, -0.059082801866018,  0.053782968500128],
                          [-0.059082801866018, -0.881834396267966,  0.053782968500128],
                          [-0.881834396267966, -0.059082801866018,  0.053782968500128],
                          [-0.518892500060958, -0.518892500060958,  0.084265045523300],
                          [-0.518892500060958,  0.037785000121916,  0.084265045523300],
                          [ 0.037785000121916, -0.518892500060958,  0.084265045523300],
                          [-0.704068411554854, -0.704068411554854,  0.060000533685546],
                          [-0.704068411554854,  0.408136823109708,  0.060000533685546],
                          [ 0.408136823109708, -0.704068411554854,  0.060000533685546],
                          [-0.849069624685052, -0.849069624685052,  0.028400197850048],
                          [-0.849069624685052,  0.698139249370104,  0.028400197850048],
                          [ 0.698139249370104, -0.849069624685052,  0.028400197850048],
                          [-0.966807194753950, -0.966807194753950,  0.007164924702546],
                          [-0.966807194753950,  0.933614389507900,  0.007164924702546],
                          [ 0.933614389507900, -0.966807194753950,  0.007164924702546],
                          [-0.406888806840226,  0.199737422349722,  0.065546294921254],
                          [ 0.199737422349722, -0.792848615509496,  0.065546294921254],
                          [-0.792848615509496, -0.406888806840226,  0.065546294921254],
                          [-0.406888806840226, -0.792848615509496,  0.065546294921254],
                          [ 0.199737422349722, -0.406888806840226,  0.065546294921254],
                          [-0.792848615509496,  0.199737422349722,  0.065546294921254],
                          [-0.324553873193842,  0.284387049883010,  0.030596612496882],
                          [ 0.284387049883010, -0.959833176689168,  0.030596612496882],
                          [-0.959833176689168, -0.324553873193842,  0.030596612496882],
                          [-0.324553873193842, -0.959833176689168,  0.030596612496882],
                          [ 0.284387049883010, -0.324553873193842,  0.030596612496882],
                          [-0.959833176689168,  0.284387049883010,  0.030596612496882],
                          [-0.590503436714376,  0.599185441942654,  0.004772488385678],
                          [ 0.599185441942654, -1.008682005228278,  0.004772488385678],
                          [-1.008682005228278, -0.590503436714376,  0.004772488385678],
                          [-0.590503436714376, -1.008682005228278,  0.004772488385678],
                          [ 0.599185441942654, -0.590503436714376,  0.004772488385678],
                          [-1.008682005228278,  0.599185441942654,  0.004772488385678],
                          [-0.621283015738754,  0.537399442802736,  0.038169585511798],
                          [ 0.537399442802736, -0.916116427063980,  0.038169585511798],
                          [-0.916116427063980, -0.621283015738754,  0.038169585511798],
                          [-0.621283015738754, -0.916116427063980,  0.038169585511798],
                          [ 0.537399442802736, -0.621283015738754,  0.038169585511798],
                          [-0.916116427063980,  0.537399442802736,  0.038169585511798],
                          [-0.829432768634686,  0.800798128173322,  0.013700109093084],
                          [ 0.800798128173322, -0.971365359538638,  0.013700109093084],
                          [-0.971365359538638, -0.829432768634686,  0.013700109093084],
                          [-0.829432768634686, -0.971365359538638,  0.013700109093084],
                          [ 0.800798128173322, -0.829432768634686,  0.013700109093084],
                          [-0.971365359538638,  0.800798128173322,  0.013700109093084]], dtype=np.float)

    elif Nord == 17:
        table = np.array([[-0.333333333333333,  -0.333333333333333,  0.066874398581606],
                          [-0.005658918886452,  -0.005658918886452,  0.010186830881014],
                          [-0.005658918886452,  -0.988682162227096,  0.010186830881014],
                          [-0.988682162227096,  -0.005658918886452,  0.010186830881014],
                          [-0.035647354750750,  -0.035647354750750,  0.029341729055276],
                          [-0.035647354750750,  -0.928705290498498,  0.029341729055276],
                          [-0.928705290498498,  -0.035647354750750,  0.029341729055276],
                          [-0.099520061958436,  -0.099520061958436,  0.048701756707344],
                          [-0.099520061958436,  -0.800959876083126,  0.048701756707344],
                          [-0.800959876083126,  -0.099520061958436,  0.048701756707344],
                          [-0.199467521245206,  -0.199467521245206,  0.062215101737938],
                          [-0.199467521245206,  -0.601064957509588,  0.062215101737938],
                          [-0.601064957509588,  -0.199467521245206,  0.062215101737938],
                          [-0.495717464058094,  -0.495717464058094,  0.062514222437240],
                          [-0.495717464058094,  -0.008565071883810,  0.062514222437240],
                          [-0.008565071883810,  -0.495717464058094,  0.062514222437240],
                          [-0.675905990683078,  -0.675905990683078,  0.049631308679330],
                          [-0.675905990683078,   0.351811981366154,  0.049631308679330],
                          [ 0.351811981366154,  -0.675905990683078,  0.049631308679330],
                          [-0.848248235478508,  -0.848248235478508,  0.028112146141114],
                          [-0.848248235478508,   0.696496470957016,  0.028112146141114],
                          [ 0.696496470957016,  -0.848248235478508,  0.028112146141114],
                          [-0.968690546064356,  -0.968690546064356,  0.006389352347558],
                          [-0.968690546064356,   0.937381092128712,  0.006389352347558],
                          [ 0.937381092128712,  -0.968690546064356,  0.006389352347558],
                          [-0.331360265272684,   0.310986407618846,  0.016239310637986],
                          [ 0.310986407618846,  -0.979626142346162,  0.016239310637986],
                          [-0.979626142346162,  -0.331360265272684,  0.016239310637986],
                          [-0.331360265272684,  -0.979626142346162,  0.016239310637986],
                          [ 0.310986407618846,  -0.331360265272684,  0.016239310637986],
                          [-0.979626142346162,   0.310986407618846,  0.016239310637986],
                          [-0.415556924406112,   0.144675181064040,  0.053611484566326],
                          [ 0.144675181064040,  -0.729118256657928,  0.053611484566326],
                          [-0.729118256657928,  -0.415556924406112,  0.053611484566326],
                          [-0.415556924406112,  -0.729118256657928,  0.053611484566326],
                          [ 0.144675181064040,  -0.415556924406112,  0.053611484566326],
                          [-0.729118256657928,   0.144675181064040,  0.053611484566326],
                          [-0.360850229153620,   0.252002380572456,  0.036919986421644],
                          [ 0.252002380572456,  -0.891152151418834,  0.036919986421644],
                          [-0.891152151418834,  -0.360850229153620,  0.036919986421644],
                          [-0.360850229153620,  -0.891152151418834,  0.036919986421644],
                          [ 0.252002380572456,  -0.360850229153620,  0.036919986421644],
                          [-0.891152151418834,   0.252002380572456,  0.036919986421644],
                          [-0.618591551615416,   0.592854429948142,  0.016953737068656],
                          [ 0.592854429948142,  -0.974262878332726,  0.016953737068656],
                          [-0.974262878332726,  -0.618591551615416,  0.016953737068656],
                          [-0.618591551615416,  -0.974262878332726,  0.016953737068656],
                          [ 0.592854429948142,  -0.618591551615416,  0.016953737068656],
                          [-0.974262878332726,   0.592854429948142,  0.016953737068656],
                          [-0.639033576702508,   0.504702011875458,  0.036585593540050],
                          [ 0.504702011875458,  -0.865668435172952,  0.036585593540050],
                          [-0.865668435172952,  -0.639033576702508,  0.036585593540050],
                          [-0.639033576702508,  -0.865668435172952,  0.036585593540050],
                          [ 0.504702011875458,  -0.639033576702508,  0.036585593540050],
                          [-0.865668435172952,   0.504702011875458,  0.036585593540050],
                          [-0.838577372640872,   0.809251008191216,  0.013331264008330],
                          [ 0.809251008191216,  -0.970673635550344,  0.013331264008330],
                          [-0.970673635550344,  -0.838577372640872,  0.013331264008330],
                          [-0.838577372640872,  -0.970673635550344,  0.013331264008330],
                          [ 0.809251008191216,  -0.838577372640872,  0.013331264008330],
                          [-0.970673635550344,   0.809251008191216,  0.013331264008330]], dtype=np.float)

    elif Nord == 18:
        table = np.array([[-0.333333333333333,  -0.333333333333333,   0.061619879875294],
                          [-0.013310382738158,  -0.013310382738158,   0.018144873358808],
                          [-0.013310382738158,  -0.973379234523686,   0.018144873358808],
                          [-0.973379234523686,  -0.013310382738158,   0.018144873358808],
                          [-0.061578811516086,  -0.061578811516086,   0.037522633879188],
                          [-0.061578811516086,  -0.876842376967828,   0.037522633879188],
                          [-0.876842376967828,  -0.061578811516086,   0.037522633879188],
                          [-0.127437208225988,  -0.127437208225988,   0.038882195970954],
                          [-0.127437208225988,  -0.745125583548022,   0.038882195970954],
                          [-0.745125583548022,  -0.127437208225988,   0.038882195970954],
                          [-0.210307658653168,  -0.210307658653168,   0.055507897221620],
                          [-0.210307658653168,  -0.579384682693664,   0.055507897221620],
                          [-0.579384682693664,  -0.210307658653168,   0.055507897221620],
                          [-0.500410862393686,  -0.500410862393686,   0.064512450702914],
                          [-0.500410862393686,   0.000821724787372,   0.064512450702914],
                          [ 0.000821724787372,  -0.500410862393686,   0.064512450702914],
                          [-0.677135612512314,  -0.677135612512314,   0.050148065233844],
                          [-0.677135612512314,   0.354271225024630,   0.050148065233844],
                          [ 0.354271225024630,  -0.677135612512314,   0.050148065233844],
                          [-0.846803545029258,  -0.846803545029258,   0.030543855943664],
                          [-0.846803545029258,   0.693607090058514,   0.030543855943664],
                          [ 0.693607090058514,  -0.846803545029258,   0.030543855943664],
                          [-0.951495121293100,  -0.951495121293100,   0.013587844045926],
                          [-0.951495121293100,   0.902990242586200,   0.013587844045926],
                          [ 0.902990242586200,  -0.951495121293100,   0.013587844045926],
                          [-0.913707265566070,  -0.913707265566070,  -0.004446197459840],
                          [-0.913707265566070,   0.827414531132142,  -0.004446197459840],
                          [ 0.827414531132142,  -0.913707265566070,  -0.004446197459840],
                          [-0.282177010118112,   0.265315937713272,   0.012663828152812],
                          [ 0.265315937713272,  -0.983138927595160,   0.012663828152812],
                          [-0.983138927595160,  -0.282177010118112,   0.012663828152812],
                          [-0.282177010118112,  -0.983138927595160,   0.012663828152812],
                          [ 0.265315937713272,  -0.282177010118112,   0.012663828152812],
                          [-0.983138927595160,   0.265315937713272,   0.012663828152812],
                          [-0.411195046496086,   0.148821943021710,   0.054515076098276],
                          [ 0.148821943021710,  -0.737626896525624,   0.054515076098276],
                          [-0.737626896525624,  -0.411195046496086,   0.054515076098276],
                          [-0.411195046496086,  -0.737626896525624,   0.054515076098276],
                          [ 0.148821943021710,  -0.411195046496086,   0.054515076098276],
                          [-0.737626896525624,   0.148821943021710,   0.054515076098276],
                          [-0.349964396716372,   0.249558093585024,   0.035353571298930],
                          [ 0.249558093585024,  -0.899593696868650,   0.035353571298930],
                          [-0.899593696868650,  -0.349964396716372,   0.035353571298930],
                          [-0.349964396716372,  -0.899593696868650,   0.035353571298930],
                          [ 0.249558093585024,  -0.349964396716372,   0.035353571298930],
                          [-0.899593696868650,   0.249558093585024,   0.035353571298930],
                          [-0.630524880667908,   0.497866353046074,   0.036758969276140],
                          [ 0.497866353046074,  -0.867341472378168,   0.036758969276140],
                          [-0.867341472378168,  -0.630524880667908,   0.036758969276140],
                          [-0.630524880667908,  -0.867341472378168,   0.036758969276140],
                          [ 0.497866353046074,  -0.630524880667908,   0.036758969276140],
                          [-0.867341472378168,   0.497866353046074,   0.036758969276140],
                          [-0.562406399973358,   0.538414010840886,   0.016209465616384],
                          [ 0.538414010840886,  -0.976007610867528,   0.016209465616384],
                          [-0.976007610867528,  -0.562406399973358,   0.016209465616384],
                          [-0.562406399973358,  -0.976007610867528,   0.016209465616384],
                          [ 0.538414010840886,  -0.562406399973358,   0.016209465616384],
                          [-0.976007610867528,   0.538414010840886,   0.016209465616384],
                          [-0.797640805727184,   0.767924604546934,   0.015268258141450],
                          [ 0.767924604546934,  -0.970283798819750,   0.015268258141450],
                          [-0.970283798819750,  -0.797640805727184,   0.015268258141450],
                          [-0.797640805727184,  -0.970283798819750,   0.015268258141450],
                          [ 0.767924604546934,  -0.797640805727184,   0.015268258141450],
                          [-0.970283798819750,   0.767924604546934,   0.015268258141450],
                          [-0.958250489434828,   1.028694520010726,   0.000092375321588],
                          [ 1.028694520010726,  -1.070444030575898,   0.000092375321588],
                          [-1.070444030575898,  -0.958250489434828,   0.000092375321588],
                          [-0.958250489434828,  -1.070444030575898,   0.000092375321588],
                          [ 1.028694520010726,  -0.958250489434828,   0.000092375321588],
                          [-1.070444030575898,   1.028694520010726,   0.000092375321588]], dtype=np.float)

    elif Nord == 19:
        table = np.array([[-0.333333333333333,  -0.333333333333333, 0.065812662777838],
                          [-0.020780025853988,  -0.020780025853988, 0.020661463782544],
                          [-0.020780025853988,  -0.958439948292026, 0.020661463782544],
                          [-0.958439948292026,  -0.020780025853988, 0.020661463782544],
                          [-0.090926214604214,  -0.090926214604214, 0.044774494526032],
                          [-0.090926214604214,  -0.818147570791570, 0.044774494526032],
                          [-0.818147570791570,  -0.090926214604214, 0.044774494526032],
                          [-0.197166638701138,  -0.197166638701138, 0.060532251738936],
                          [-0.197166638701138,  -0.605666722597724, 0.060532251738936],
                          [-0.605666722597724,  -0.197166638701138, 0.060532251738936],
                          [-0.488896691193804,  -0.488896691193804, 0.060981935604396],
                          [-0.488896691193804,  -0.022206617612390, 0.060981935604396],
                          [-0.022206617612390,  -0.488896691193804, 0.060981935604396],
                          [-0.645844115695740,  -0.645844115695740, 0.048318425483282],
                          [-0.645844115695740,   0.291688231391482, 0.048318425483282],
                          [ 0.291688231391482,  -0.645844115695740, 0.048318425483282],
                          [-0.779877893544096,  -0.779877893544096, 0.032101607173602],
                          [-0.779877893544096,   0.559755787088192, 0.032101607173602],
                          [ 0.559755787088192,  -0.779877893544096, 0.032101607173602],
                          [-0.888942751496320,  -0.888942751496320, 0.016169160523568],
                          [-0.888942751496320,   0.777885502992642, 0.016169160523568],
                          [ 0.777885502992642,  -0.888942751496320, 0.016169160523568],
                          [-0.974756272445542,  -0.974756272445542, 0.004158724054970],
                          [-0.974756272445542,   0.949512544891086, 0.004158724054970],
                          [ 0.949512544891086,  -0.974756272445542, 0.004158724054970],
                          [-0.208490425286114,   0.201267589589290, 0.007769753809962],
                          [ 0.201267589589290,  -0.992777164303176, 0.007769753809962],
                          [-0.992777164303176,  -0.208490425286114, 0.007769753809962],
                          [-0.208490425286114,  -0.992777164303176, 0.007769753809962],
                          [ 0.201267589589290,  -0.208490425286114, 0.007769753809962],
                          [-0.992777164303176,   0.201267589589286, 0.007769753809962],
                          [-0.384140032239128,   0.115206523177568, 0.051148321224044],
                          [ 0.115206523177568,  -0.731066490938440, 0.051148321224044],
                          [-0.731066490938440,  -0.384140032239128, 0.051148321224044],
                          [-0.384140032239128,  -0.731066490938440, 0.051148321224044],
                          [ 0.115206523177568,  -0.384140032239128, 0.051148321224044],
                          [-0.731066490938440,   0.115206523177568, 0.051148321224044],
                          [-0.470866103186960,   0.441974051634730, 0.017761807146676],
                          [ 0.441974051634730,  -0.971107948447770, 0.017761807146676],
                          [-0.971107948447770,  -0.470866103186960, 0.017761807146676],
                          [-0.470866103186960,  -0.971107948447770, 0.017761807146676],
                          [ 0.441974051634730,  -0.470866103186960, 0.017761807146676],
                          [-0.971107948447770,   0.441974051634730, 0.017761807146676],
                          [-0.282921295588098,   0.189054137911742, 0.032249093523462],
                          [ 0.189054137911742,  -0.906132842323644, 0.032249093523462],
                          [-0.906132842323644,  -0.282921295588098, 0.032249093523462],
                          [-0.282921295588098,  -0.906132842323644, 0.032249093523462],
                          [ 0.189054137911742,  -0.282921295588098, 0.032249093523462],
                          [-0.906132842323644,   0.189054137911742, 0.032249093523462],
                          [-0.684385188062810,   0.678662947361678, 0.004983883634982],
                          [ 0.678662947361678,  -0.994277759298866, 0.004983883634982],
                          [-0.994277759298866,  -0.684385188062810, 0.004983883634982],
                          [-0.684385188062810,  -0.994277759298866, 0.004983883634982],
                          [ 0.678662947361678,  -0.684385188062810, 0.004983883634982],
                          [-0.994277759298866,   0.678662947361678, 0.004983883634982],
                          [-0.849898806048178,   0.402175957852346, 0.036485680237902],
                          [ 0.402175957852346,  -0.552277151804168, 0.036485680237902],
                          [-0.552277151804168,  -0.849898806048178, 0.036485680237902],
                          [-0.849898806048178,  -0.552277151804168, 0.036485680237902],
                          [ 0.402175957852346,  -0.849898806048178, 0.036485680237902],
                          [-0.552277151804168,   0.402175957852346, 0.036485680237902],
                          [-0.715156797773234,   0.645862648139714, 0.020517127472398],
                          [ 0.645862648139714,  -0.930705850366480, 0.020517127472398],
                          [-0.930705850366480,  -0.715156797773234, 0.020517127472398],
                          [-0.715156797773234,  -0.930705850366480, 0.020517127472398],
                          [ 0.645862648139714,  -0.715156797773234, 0.020517127472398],
                          [-0.930705850366480,   0.645862648139714, 0.020517127472398],
                          [-0.869010743834124,   0.848688505241568, 0.007599857710604],
                          [ 0.848688505241568,  -0.979677761407444, 0.007599857710604],
                          [-0.979677761407444,  -0.869010743834124, 0.007599857710604],
                          [-0.869010743834124,  -0.979677761407444, 0.007599857710604],
                          [ 0.848688505241568,  -0.869010743834124, 0.007599857710604],
                          [-0.979677761407444,   0.848688505241568, 0.007599857710604]], dtype=np.float)

    elif Nord == 20:
        table = np.array([[-0.333333333333333,  -0.333333333333333,   0.066114111083248],
                          [ 0.001900928704400,   0.001900928704400,   0.001734038371326],
                          [ 0.001900928704400,  -1.003801857408800,   0.001734038371326],
                          [-1.003801857408800,   0.001900928704400,   0.001734038371326],
                          [-0.023574084130543,  -0.023574084130543,   0.023320105432896],
                          [-0.023574084130543,  -0.952851831738914,   0.023320105432896],
                          [-0.952851831738914,  -0.023574084130543,   0.023320105432896],
                          [-0.089726626099435,  -0.089726626099435,   0.045753872712842],
                          [-0.089726626099435,  -0.820546727801130,   0.045753872712842],
                          [-0.820546727801130,  -0.089726626099435,   0.045753872712842],
                          [-0.196007481363421,  -0.196007481363421,   0.060897965347876],
                          [-0.196007481363421,  -0.607985037273158,   0.060897965347876],
                          [-0.607985037273158,  -0.196007481363421,   0.060897965347876],
                          [-0.488214180481157,  -0.488214180481157,   0.061249783450710],
                          [-0.488214180481157,  -0.023571639037686,   0.061249783450710],
                          [-0.023571639037686,  -0.488214180481157,   0.061249783450710],
                          [-0.647023488009788,  -0.647023488009788,   0.048736115353600],
                          [-0.647023488009788,   0.294046976019576,   0.048736115353600],
                          [ 0.294046976019576,  -0.647023488009788,   0.048736115353600],
                          [-0.791658289326483,  -0.791658289326483,   0.031994864064048],
                          [-0.791658289326483,   0.583316578652966,   0.031994864064048],
                          [ 0.583316578652966,  -0.791658289326483,   0.031994864064048],
                          [-0.893862072318140,  -0.893862072318140,   0.015396603631204],
                          [-0.893862072318140,   0.787724144636280,   0.015396603631204],
                          [ 0.787724144636280,  -0.893862072318140,   0.015396603631204],
                          [-0.916762569607942,  -0.916762569607942,  -0.001264120994976],
                          [-0.916762569607942,   0.833525139215884,  -0.001264120994976],
                          [ 0.833525139215884,  -0.916762569607942,  -0.001264120994976],
                          [-0.976836157186356,  -0.976836157186356,   0.003502268602386],
                          [-0.976836157186356,   0.953672314372712,   0.003502268602386],
                          [ 0.953672314372712,  -0.976836157186356,   0.003502268602386],
                          [-0.310288459541998,   0.212805292212320,   0.032931678379152],
                          [ 0.212805292212320,  -0.902516832670322,   0.032931678379152],
                          [-0.902516832670322,  -0.310288459541998,   0.032931678379152],
                          [-0.310288459541998,  -0.902516832670322,   0.032931678379152],
                          [ 0.212805292212320,  -0.310288459541998,   0.032931678379152],
                          [-0.902516832670322,   0.212805292212320,   0.032931678379152],
                          [-0.244313460810292,   0.231685228913082,   0.009678067080970],
                          [ 0.231685228913082,  -0.987371768102790,   0.009678067080970],
                          [-0.987371768102790,  -0.244313460810292,   0.009678067080970],
                          [-0.244313460810292,  -0.987371768102790,   0.009678067080970],
                          [ 0.231685228913082,  -0.244313460810292,   0.009678067080970],
                          [-0.987371768102790,   0.231685228913082,   0.009678067080970],
                          [-0.386729041875286,   0.118096000780590,   0.051609813069300],
                          [ 0.118096000780590,  -0.731366958905304,   0.051609813069300],
                          [-0.731366958905304,  -0.386729041875286,   0.051609813069300],
                          [-0.386729041875286,  -0.731366958905304,   0.051609813069300],
                          [ 0.118096000780590,  -0.386729041875286,   0.051609813069300],
                          [-0.731366958905304,   0.118096000780590,   0.051609813069300],
                          [-0.501161274450516,   0.473213486525732,   0.016942182108882],
                          [ 0.473213486525732,  -0.972052212075216,   0.016942182108882],
                          [-0.972052212075216,  -0.501161274450516,   0.016942182108882],
                          [-0.501161274450516,  -0.972052212075216,   0.016942182108882],
                          [ 0.473213486525732,  -0.501161274450516,   0.016942182108882],
                          [-0.972052212075216,   0.473213486525732,   0.016942182108882],
                          [-0.574448550394396,   0.423350284574868,   0.036709828212560],
                          [ 0.423350284574868,  -0.848901734180472,   0.036709828212560],
                          [-0.848901734180472,  -0.574448550394396,   0.036709828212560],
                          [-0.574448550394396,  -0.848901734180472,   0.036709828212560],
                          [ 0.423350284574868,  -0.574448550394396,   0.036709828212560],
                          [-0.848901734180472,   0.423350284574868,   0.036709828212560],
                          [-0.706069127893522,   0.722805434309974,   0.001408809355816],
                          [ 0.722805434309974,  -1.016736306416454,   0.001408809355816],
                          [-1.016736306416454,  -0.706069127893522,   0.001408809355816],
                          [-0.706069127893522,  -1.016736306416454,   0.001408809355816],
                          [ 0.722805434309974,  -0.706069127893522,   0.001408809355816],
                          [-1.016736306416454,   0.722805434309974,   0.001408809355816],
                          [-0.724546042342154,   0.671173915824726,   0.020225369854924],
                          [ 0.671173915824726,  -0.946627873482572,   0.020225369854924],
                          [-0.946627873482572,  -0.724546042342154,   0.020225369854924],
                          [-0.724546042342154,  -0.946627873482572,   0.020225369854924],
                          [ 0.671173915824726,  -0.724546042342154,   0.020225369854924],
                          [-0.946627873482572,   0.671173915824726,   0.020225369854924],
                          [-0.880607781701986,   0.859512343113706,   0.007147818771900],
                          [ 0.859512343113706,  -0.978904561411718,   0.007147818771900],
                          [-0.978904561411718,  -0.880607781701986,   0.007147818771900],
                          [-0.880607781701986,  -0.978904561411718,   0.007147818771900],
                          [ 0.859512343113706,  -0.880607781701986,   0.007147818771900],
                          [-0.978904561411718,   0.859512343113706,   0.007147818771900]], dtype=np.float)

    else:
        Print.master('     Orders higher than 20 are not supported for triangle yet')
        exit(-1)

    # Renormalization (reference triangle)
    weights = table[:,2]/4.
    points = np.zeros((weights.size,2), dtype=np.float)

    points[:, 0] = (1.+table[:,0])/2.
    points[:, 1] = -(np.sum(table[:,0:2], axis=1))/2.

    return points, weights


def computeSourceVectorRotation(azimuth, dip):
    """Compute the weigths vector for source rotation in the xyz plane.

    :param float azimuth: degrees for x-y plane rotation
    :param float dip: degrees for x-z plane rotation
    :return: weigths for source rotation
    :rtype: ndarray.
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Define reference unit vector (x-directed pointing to East)
    base_vector = np.array([1,0,0], dtype=np.float)

    # ---------------------------------------------------------------
    # Compute vector for source rotation
    # ---------------------------------------------------------------
    # Convert degrees to radians for rotation
    alpha = azimuth*np.pi/180.    # x-y plane
    beta  = dip*np.pi/180.        # x-z plane
    tetha = 0*np.pi/180.          # y-z plane

    # Define rotation matrices for each plane
    # x-y plane
    M1 = np.array([[np.cos(alpha),  -np.sin(alpha),   0.],
                   [np.sin(alpha),   np.cos(alpha),   0.],
                   [     0.,              0.,         1.]], dtype=np.float)

    # x-z plane
    M2 = np.array([[np.cos(beta),    0.,    -np.sin(beta)],
                   [     0.,         1.,          0.     ],
                   [np.sin(beta),    0.,     np.cos(beta)]], dtype=np.float)

    # y-z plane
    M3 = np.array([[1.,          0.,                   0],
                   [0.,    np.cos(tetha),  -np.sin(tetha)],
                   [0.,    np.sin(tetha),   np.cos(tetha)]], dtype=np.float)

    # Apply rotation
    rotSourceVector = np.dot(np.matmul(np.matmul(M1,M2),M3), base_vector)

    return rotSourceVector


def tetrahedronXYZToXiEtaZeta(eleNodes, points):
    """Compute the reference tetrahedron coordinates from xyz global tetrahedron coordinates.

    :param ndarray eleNodes: spatial coordinates of the nodes with dimensions = (4,3)
    :param ndarray points: xyz points coordinates to be transformed
    :return: xietazeta points coordinates
    :rtype: ndarray
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Get number of points
    if points.ndim == 1:
        num_points = 1
        XiEtaZeta = np.zeros(3, dtype=np.float)
    else:
        num_points = points.shape[0]
        # Allocate
        XiEtaZeta = np.zeros((num_points, 3), dtype=np.float)

    # ---------------------------------------------------------------
    # Compute mapping
    # ---------------------------------------------------------------
    # Loop over points
    if num_points == 1:
        pointX = points[0]
        pointY = points[1]
        pointZ = points[2]

        J = eleNodes[0,2] * ( eleNodes[1,0] * (eleNodes[3,1] - eleNodes[2,1]) \
            + eleNodes[2,0] * (eleNodes[1,1] - eleNodes[3,1]) \
            + eleNodes[3,0] * (eleNodes[2,1] - eleNodes[1,1]) ) \
            + eleNodes[1,2] * ( eleNodes[0,0] * (eleNodes[2,1] - eleNodes[3,1]) \
            + eleNodes[2,0] * (eleNodes[3,1] - eleNodes[0,1]) \
            + eleNodes[3,0] * (eleNodes[0,1] - eleNodes[2,1]) ) \
            + eleNodes[2,2] * ( eleNodes[0,0] * (eleNodes[3,1] - eleNodes[1,1]) \
            + eleNodes[1,0] * (eleNodes[0,1] - eleNodes[3,1]) \
            + eleNodes[3,0] * (eleNodes[1,1] - eleNodes[0,1]) ) \
            + eleNodes[3,2] * ( eleNodes[0,0] * (eleNodes[1,1] - eleNodes[2,1]) \
            + eleNodes[1,0] * (eleNodes[2,1] - eleNodes[0,1]) \
            + eleNodes[2,0] * (eleNodes[0,1] - eleNodes[1,1]) )

        xi = ( eleNodes[3,2] * (eleNodes[2,1] - eleNodes[0,1]) + eleNodes[0,2] * (eleNodes[3,1] - eleNodes[2,1]) \
             + eleNodes[2,2] * (eleNodes[0,1] - eleNodes[3,1]) ) / J * pointX + \
            ( eleNodes[0,2] * (eleNodes[2,0] - eleNodes[3,0]) + eleNodes[3,2] * (eleNodes[0,0] - eleNodes[2,0]) \
            + eleNodes[2,2] * (eleNodes[3,0] - eleNodes[0,0]) ) / J * pointY + \
            ( eleNodes[0,0] * (eleNodes[2,1] - eleNodes[3,1]) + eleNodes[3,0] * (eleNodes[0,1] - eleNodes[2,1]) \
            + eleNodes[2,0] * (eleNodes[3,1] - eleNodes[0,1]) ) / J * pointZ + \
            ( eleNodes[2,2] * (eleNodes[0,0] * eleNodes[3,1] - eleNodes[3,0] * eleNodes[0,1]) \
            + eleNodes[0,2] * (eleNodes[3,0] * eleNodes[2,1] - eleNodes[2,0] * eleNodes[3,1]) \
            + eleNodes[3,2] * (eleNodes[2,0] * eleNodes[0,1] - eleNodes[0,0] * eleNodes[2,1]) ) / J

        eta = ( eleNodes[1,2] * (eleNodes[3,1] - eleNodes[0,1]) + eleNodes[3,2] * (eleNodes[0,1] - eleNodes[1,1]) \
            + eleNodes[0,2] * (eleNodes[1,1] - eleNodes[3,1]) ) / J * pointX + \
            ( eleNodes[1,2] * (eleNodes[0,0] - eleNodes[3,0]) + eleNodes[0,2] * (eleNodes[3,0] - eleNodes[1,0]) \
            + eleNodes[3,2] * (eleNodes[1,0] - eleNodes[0,0]) ) / J * pointY + \
            ( eleNodes[1,0] * (eleNodes[0,1] - eleNodes[3,1]) + eleNodes[0,0] * (eleNodes[3,1] - eleNodes[1,1]) \
            + eleNodes[3,0] * (eleNodes[1,1] - eleNodes[0,1]) ) / J * pointZ + \
            ( eleNodes[1,2] * (eleNodes[3,0] * eleNodes[0,1] - eleNodes[0,0] * eleNodes[3,1]) \
            + eleNodes[0,2] * (eleNodes[1,0] * eleNodes[3,1] - eleNodes[3,0] * eleNodes[1,1]) \
            + eleNodes[3,2] * (eleNodes[0,0] * eleNodes[1,1] - eleNodes[1,0] * eleNodes[0,1]) ) / J


        zeta = ( eleNodes[0,2] * (eleNodes[2,1] - eleNodes[1,1]) + eleNodes[2,2] * (eleNodes[1,1] - eleNodes[0,1]) \
            + eleNodes[1,2] * (eleNodes[0,1] - eleNodes[2,1]) ) / J * pointX + \
            ( eleNodes[2,2] * (eleNodes[0,0] - eleNodes[1,0]) + eleNodes[0,2] * (eleNodes[1,0] - eleNodes[2,0]) \
            + eleNodes[1,2] * (eleNodes[2,0] - eleNodes[0,0]) ) / J * pointY + \
            ( eleNodes[0,0] * (eleNodes[1,1] - eleNodes[2,1]) + eleNodes[1,0] * (eleNodes[2,1] - eleNodes[0,1]) \
            + eleNodes[2,0] * (eleNodes[0,1] - eleNodes[1,1]) ) / J * pointZ + \
            ( eleNodes[0,2] * ( eleNodes[3,0] * (eleNodes[1,1] - eleNodes[2,1]) \
            + eleNodes[3,1] * (eleNodes[2,0] - eleNodes[1,0]) ) \
            + eleNodes[2,2] * ( eleNodes[3,0] * (eleNodes[0,1] - eleNodes[1,1]) \
            + eleNodes[3,1] * (eleNodes[1,0] - eleNodes[0,0]) ) \
            + eleNodes[1,2] * ( eleNodes[3,0] * (eleNodes[2,1] - eleNodes[0,1]) \
            + eleNodes[3,1] * (eleNodes[0,0] - eleNodes[2,0]) ) \
            + eleNodes[3,2] * ( eleNodes[1,0] * (eleNodes[0,1] - eleNodes[2,1]) \
            + eleNodes[0,0] * (eleNodes[2,1] - eleNodes[1,1]) \
            + eleNodes[2,0] * (eleNodes[1,1] - eleNodes[0,1]) ) + J ) / J

        XiEtaZeta[0] = xi
        XiEtaZeta[1] = eta
        XiEtaZeta[2] = zeta

    else:
        for i in np.arange(num_points):
            pointX = points[i,0]
            pointY = points[i,1]
            pointZ = points[i,2]

            J = eleNodes[0,2] * ( eleNodes[1,0] * (eleNodes[3,1] - eleNodes[2,1]) \
                + eleNodes[2,0] * (eleNodes[1,1] - eleNodes[3,1]) \
                + eleNodes[3,0] * (eleNodes[2,1] - eleNodes[1,1]) ) \
                + eleNodes[1,2] * ( eleNodes[0,0] * (eleNodes[2,1] - eleNodes[3,1]) \
                + eleNodes[2,0] * (eleNodes[3,1] - eleNodes[0,1]) \
                + eleNodes[3,0] * (eleNodes[0,1] - eleNodes[2,1]) ) \
                + eleNodes[2,2] * ( eleNodes[0,0] * (eleNodes[3,1] - eleNodes[1,1]) \
                + eleNodes[1,0] * (eleNodes[0,1] - eleNodes[3,1]) \
                + eleNodes[3,0] * (eleNodes[1,1] - eleNodes[0,1]) ) \
                + eleNodes[3,2] * ( eleNodes[0,0] * (eleNodes[1,1] - eleNodes[2,1]) \
                + eleNodes[1,0] * (eleNodes[2,1] - eleNodes[0,1]) \
                + eleNodes[2,0] * (eleNodes[0,1] - eleNodes[1,1]) )

            xi = ( eleNodes[3,2] * (eleNodes[2,1] - eleNodes[0,1]) + eleNodes[0,2] * (eleNodes[3,1] - eleNodes[2,1]) \
                 + eleNodes[2,2] * (eleNodes[0,1] - eleNodes[3,1]) ) / J * pointX + \
                ( eleNodes[0,2] * (eleNodes[2,0] - eleNodes[3,0]) + eleNodes[3,2] * (eleNodes[0,0] - eleNodes[2,0]) \
                + eleNodes[2,2] * (eleNodes[3,0] - eleNodes[0,0]) ) / J * pointY + \
                ( eleNodes[0,0] * (eleNodes[2,1] - eleNodes[3,1]) + eleNodes[3,0] * (eleNodes[0,1] - eleNodes[2,1]) \
                + eleNodes[2,0] * (eleNodes[3,1] - eleNodes[0,1]) ) / J * pointZ + \
                ( eleNodes[2,2] * (eleNodes[0,0] * eleNodes[3,1] - eleNodes[3,0] * eleNodes[0,1]) \
                + eleNodes[0,2] * (eleNodes[3,0] * eleNodes[2,1] - eleNodes[2,0] * eleNodes[3,1]) \
                + eleNodes[3,2] * (eleNodes[2,0] * eleNodes[0,1] - eleNodes[0,0] * eleNodes[2,1]) ) / J

            eta = ( eleNodes[1,2] * (eleNodes[3,1] - eleNodes[0,1]) + eleNodes[3,2] * (eleNodes[0,1] - eleNodes[1,1]) \
                + eleNodes[0,2] * (eleNodes[1,1] - eleNodes[3,1]) ) / J * pointX + \
                ( eleNodes[1,2] * (eleNodes[0,0] - eleNodes[3,0]) + eleNodes[0,2] * (eleNodes[3,0] - eleNodes[1,0]) \
                + eleNodes[3,2] * (eleNodes[1,0] - eleNodes[0,0]) ) / J * pointY + \
                ( eleNodes[1,0] * (eleNodes[0,1] - eleNodes[3,1]) + eleNodes[0,0] * (eleNodes[3,1] - eleNodes[1,1]) \
                + eleNodes[3,0] * (eleNodes[1,1] - eleNodes[0,1]) ) / J * pointZ + \
                ( eleNodes[1,2] * (eleNodes[3,0] * eleNodes[0,1] - eleNodes[0,0] * eleNodes[3,1]) \
                + eleNodes[0,2] * (eleNodes[1,0] * eleNodes[3,1] - eleNodes[3,0] * eleNodes[1,1]) \
                + eleNodes[3,2] * (eleNodes[0,0] * eleNodes[1,1] - eleNodes[1,0] * eleNodes[0,1]) ) / J


            zeta = ( eleNodes[0,2] * (eleNodes[2,1] - eleNodes[1,1]) + eleNodes[2,2] * (eleNodes[1,1] - eleNodes[0,1]) \
                + eleNodes[1,2] * (eleNodes[0,1] - eleNodes[2,1]) ) / J * pointX + \
                ( eleNodes[2,2] * (eleNodes[0,0] - eleNodes[1,0]) + eleNodes[0,2] * (eleNodes[1,0] - eleNodes[2,0]) \
                + eleNodes[1,2] * (eleNodes[2,0] - eleNodes[0,0]) ) / J * pointY + \
                ( eleNodes[0,0] * (eleNodes[1,1] - eleNodes[2,1]) + eleNodes[1,0] * (eleNodes[2,1] - eleNodes[0,1]) \
                + eleNodes[2,0] * (eleNodes[0,1] - eleNodes[1,1]) ) / J * pointZ + \
                ( eleNodes[0,2] * ( eleNodes[3,0] * (eleNodes[1,1] - eleNodes[2,1]) \
                + eleNodes[3,1] * (eleNodes[2,0] - eleNodes[1,0]) ) \
                + eleNodes[2,2] * ( eleNodes[3,0] * (eleNodes[0,1] - eleNodes[1,1]) \
                + eleNodes[3,1] * (eleNodes[1,0] - eleNodes[0,0]) ) \
                + eleNodes[1,2] * ( eleNodes[3,0] * (eleNodes[2,1] - eleNodes[0,1]) \
                + eleNodes[3,1] * (eleNodes[0,0] - eleNodes[2,0]) ) \
                + eleNodes[3,2] * ( eleNodes[1,0] * (eleNodes[0,1] - eleNodes[2,1]) \
                + eleNodes[0,0] * (eleNodes[2,1] - eleNodes[1,1]) \
                + eleNodes[2,0] * (eleNodes[1,1] - eleNodes[0,1]) ) + J ) / J

            XiEtaZeta[i,0] = xi
            XiEtaZeta[i,1] = eta
            XiEtaZeta[i,2] = zeta

    return XiEtaZeta


def computeBasisFunctions(edge_orientation, face_orientation, jacobian, invjacob, Nord, points):
    """Compute the basis function for a given element.

    :param ndarray edges_orientation: orientation for edges
    :param ndarray faces_orientation: orientation for faces
    :param ndarray jacobian: jacobian matrix
    :param ndarray invjacob: inverse of jacobian matrix
    :param int Nord: polynomial order of nedelec basis functions
    :param ndarray points: spatial points at which basis functions will be computed
    :return: basis functions and its curl for p-order=Nord
    :rtype: ndarray
    """
    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Get number of points
    if points.ndim == 1:
        num_points = 1
        points = points.reshape((1,3))
    else:
        num_points = points.shape[0]

    # Compute number of dofs for element
    num_dof_in_element = np.int(Nord*(Nord+2)*(Nord+3)/2)

    # Allocate
    basis = np.zeros((3, num_dof_in_element, num_points), dtype=np.float)
    curl_basis = np.zeros((3, num_dof_in_element, num_points), dtype=np.float)

    for i in np.arange(num_points):
        # Get point coordinates
        X = points[i, :]
        # Polynomial order (6 edges, 4 faces, 1 volume)
        Nord_vector = np.ones(11, dtype=np.int)*Nord
        # Edge orientation (6 edges)
        NoriE = edge_orientation
        # Face orientation (4 faces)
        NoriF = face_orientation

        # Compute basis for iPoint
        NrdofE, ShapE, CurlE = shape3DETet(X, Nord_vector, NoriE, NoriF)

        # Verify consistency of number of dofs for this point
        if (NrdofE != num_dof_in_element):
            Print.master('        Number of DOFs is not consistent')
            exit(-1)

        # Niref=Ni in reference element
        Niref = ShapE[0:3, 0:NrdofE]

        # Ni=Ni in real element
        Ni_real = np.matmul(invjacob, Niref)

        # Curl_real = curl in real element
        Curl_real = np.matmul(np.transpose(jacobian), CurlE[0:3,0:NrdofE]/np.linalg.det(np.transpose(jacobian)))

        # Store basis functions and its curl for i
        basis[:,:,i] = Ni_real
        curl_basis[:,:,i] = Curl_real

    return basis, curl_basis


def getNeumannBCface(face_flag, polarization, ud):
    """Get Neumann boundary condition for boundary face.

    :param int face_flag: face flag (side on which face belongs)
    :param int polarization: polarization mode (x-mode or y-mode)
    :param ndarray ud: magnetic field vector
    :return: value of Neumann boundary condition for boundary face
    :rtype: ndarray
    """
    # Allocate
    xzero = np.zeros(ud.shape, dtype=np.complex)

    if polarization == 1:
        if face_flag == 0:
            ex = ud
            ey = xzero
            ez = xzero
        elif face_flag == 1:
            ex = xzero
            ey = xzero
            ez = -ud
        elif face_flag == 2:
            ex = xzero
            ey = xzero
            ez = xzero
        elif face_flag == 3:
            ex = xzero
            ey = xzero
            ez = ud
        elif face_flag == 4:
            ex = xzero
            ey = xzero
            ez = xzero
        elif face_flag == 5:
            ex = -ud
            ey = xzero
            ez = xzero

    elif polarization == 2:
        if face_flag == 0:
            ex = xzero
            ey = ud
            ez = xzero
        elif face_flag == 1:
            ex = xzero
            ey = xzero
            ez = xzero
        elif face_flag == 2:
            ex = xzero
            ey = xzero
            ez = -ud
        elif face_flag == 3:
            ex = xzero
            ey = xzero
            ez = xzero
        elif face_flag == 4:
            ex = xzero
            ey = xzero
            ez = ud
        elif face_flag == 5:
            ex = xzero
            ey = -ud
            ez = xzero

    else:
        Print.master('     Polarization mode not supported by PETGEM')
        exit(-1)

    return ex, ey, ez


def getNormalVector(faceNumber, invJacobMatrix):
    """This function computes the normal vector for a given tetrahedral face.

    :param int faceNumber: local face number
    :param int invJacobMatrix: inverse of jacobian matrix
    :return: face normal vector
    :rtype: ndarray
    """

    normalRefVectorArray = np.array([[0.,   0.,   -1.],
                                     [0.,  -1.,    0.],
                                     [1./np.sqrt(3.),  1./np.sqrt(3.),  1./np.sqrt(3.)],
                                     [-1.,  0.,    0.]], dtype=np.float)

    normalRefVector = normalRefVectorArray[faceNumber,:]

    normalVector = np.matmul(invJacobMatrix,normalRefVector)

    return normalVector


def get2DJacobDet(coordEle, faceNumber):
    """Compute the determinant of the jacobian for 2D integrals (when 3D basis functions are used)

    :param ndarray coordEle: coordinates of the tetrahedron
    :param int faceNumber: local face number
    :return: determinant of the 2D jacobian
    :rtype: ndarray
    """

    faceByLocalNodes = getFaceByLocalNodes(faceNumber)
    normalVector = np.cross(coordEle[faceByLocalNodes[1],:]-coordEle[faceByLocalNodes[0],:],
                            coordEle[faceByLocalNodes[2],:]-coordEle[faceByLocalNodes[0],:])
    detJacob2D = np.linalg.norm(normalVector)

    return detJacob2D


def getFaceByLocalNodes(faceNumber):
    """Get local nodes ordering for a given face

    :param int faceNumber: local face number
    :return: list of local nodes for faceNumber
    :rtype: ndarray
    """
    # Allocate
    facesByLocalNodes = np.zeros((4,3), dtype=np.int)

    # Reference ordering
    facesByLocalNodes[0,:] = [0, 1, 2]
    facesByLocalNodes[1,:] = [0, 1, 3]
    facesByLocalNodes[2,:] = [1, 2, 3]
    facesByLocalNodes[3,:] = [0, 2, 3]

    faceByLocalNodes = facesByLocalNodes[faceNumber,:]

    return faceByLocalNodes


def transform2Dto3DInReferenceElement(points2D, faceNumber):
    """Transforms 2D points defined on some face into its 3D representation.

    :param ndarray points2D: 2D points to be transformed
    :param int faceNumber: local face number
    :return: resulting 3D points
    :rtype: ndarray
    """

    # Number of points
    # Number of receivers
    if points2D.ndim == 1:
        nPoints = 1
    else:
        dim = points2D.shape
        nPoints = dim[0]

    # Define reference coordinates
    refCoordinates = np.array([[0., 0., 0.],
                               [1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 1.]], dtype=np.float)

    nodesInFace = getFaceByLocalNodes(faceNumber)
    originNode = refCoordinates[nodesInFace[0],:]
    tmp1 = refCoordinates[nodesInFace[1],:] - originNode
    tmp2 = refCoordinates[nodesInFace[2],:] - originNode
    jacobian2Dto3D = np.transpose(np.vstack((tmp1, tmp2)))

    if nPoints == 1:
        points3D = np.matmul(jacobian2Dto3D,points2D) + originNode
    else:
        # Allocate
        points3D = np.zeros((nPoints,3), dtype=np.float)

        for i in np.arange(nPoints):
            points3D[i,:] = np.matmul(jacobian2Dto3D,points2D[i,:]) + originNode

    return points3D


def getRealFromReference(rRef, verticesReal):
    """Translate a point defined in the reference element (rRef) to the real element (rReal)
    defined by verticesReal

    :param ndarray rRef: reference coordinates
    :param ndarray verticesReal: real coordinates of element (4x3) = (4 nodes x 3 coordinates)
    :return: points in real element
    :rtype: ndarray
    """

    # You need to change the order if this order is changed. (basically, the
    # interpolatory functions are the affine coordinates L1, L2, L3, L4). With
    # curved elements higher-order interpolatory functions need to be
    # introduced.
    rReal = [(1-rRef[0]-rRef[1]-rRef[2])*verticesReal[0,:] +
            rRef[0]*verticesReal[1,:] +
            rRef[1]*verticesReal[2,:] +
            rRef[2]*verticesReal[3,:]]
    rReal = np.asarray(rReal[0], dtype=np.float)

    return rReal


def computeBasisFunctionsReferenceElement(edge_orientation, face_orientation, Nord, points):
    """Compute the basis function for the reference element.

    :param ndarray edges_orientation: orientation for edges
    :param ndarray faces_orientation: orientation for faces
    :param int Nord: polynomial order of nedelec basis functions
    :param ndarray points: spatial points at which basis functions will be computed
    :return: basis functions on reference element
    :rtype: ndarray
    """
    # Get number of points
    if points.ndim == 1:
        num_points = 1
        points = points.reshape((1,3))
    else:
        num_points = points.shape[0]

    # Compute number of dofs for element
    num_dof_in_element = np.int(Nord*(Nord+2)*(Nord+3)/2)

    # Allocate
    basis = np.zeros((3, num_dof_in_element, num_points), dtype=np.float)

    for i in np.arange(num_points):
        # Get gauss point coordinates
        X = points[i,:]
        # Polynomial order (6 edges, 4 faces, 1 volume)
        Nord_vector = np.ones(11, dtype=np.int)*Nord
        # Edge orientation (6 edges)
        NoriE = edge_orientation
        # Face orientation (4 faces)
        NoriF = face_orientation

        # Compute basis for iPoint
        NrdofE, ShapE, CurlE = shape3DETet(X, Nord_vector, NoriE, NoriF)

        # Verify consistency of number of dofs for this point
        if (NrdofE != num_dof_in_element):
            Print.master('        Number of DOFs is not consistent')
            exit(-1)

        # Niref=Ni in reference element
        Niref = ShapE[0:3, 0:NrdofE]

        # Store basis functions for i
        basis[:,:,i] = Niref

    return basis


def unitary_test():
    """Unitary test for hvfem.py script."""


if __name__ == '__main__':
    unitary_test()
