#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define functions for high-order vector finite element method.
'''

# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import sys
import numpy as np
from .common import Print
from .common import Print

# ###############################################################
# ################     FUNCTIONS DEFINITION     #################
# ###############################################################
def computeConnectivityDOFS(elemsE, elemsF, Nord):
    ''' This function computes the degrees of freedom connectivity for a given
    list of edges, faces and elements.

    :param ndarray elemsE: elements-edge connectivity with dimensions = (6,number_elements)
    :param ndarray elemsF: element/faces connectivity with dimensions = (4,number_elements)
    :param int Nord: polynomial order of nedelec basis functions
    :return: local/global dofs list for elements, dofs index on edges, dofs index on faces, dofs index on volumes, total number of dofs
    :rtype: ndarray and int

    .. note:: References:\n
       Amor-Martin, A., Garcia-Castillo, L. E., & Garcia-Doñoro, D. D.
       (2016). Second-order Nédélec curl-conforming prismatic element
       for computational electromagnetics. IEEE Transactions on
       Antennas and Propagation, 64(10), 4384-4395.
    '''
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
    ''' This functin computes the jacobian and its inverse.

    :param ndarray eleNodes: spatial coordinates of the nodes with dimensions = (4,3)
    :return: jacobian matrix and its inverse.
    :rtype: ndarray
    '''

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
    ''' This function computes the orientation for the computation of
    hierarchical basis functions of high-order (High-order nédélec basis functions)

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
    '''
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
    ''' This function computes the elemental mass matrix and stiffness matrix based on
    high-order vector finite element.

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
    '''

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
    ''' This function computes values of 3D tetrahedron element H(curl) shape
    functions and their derivatives

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
    '''

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
                    for l in np.arange(minI, k-minJ+1):
                        p = k-l
                        m += 2
                        ShapE[0:N, m-1] = ETri[0:N, l, p-1]
                        CurlE[0:N, m-1] = CurlETri[0:N, l, p-1]

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
                    for l in np.arange(minI, k-minJ+1):
                        p = k-l
                        q = j-k
                        m += 3

                        ShapE[0:N, m-1] = ETri[0:N,l,p-1]*homLbet[k-1,q-1]
                        DhomLbetxETri = np.cross(DhomLbet[0:N,k-1,q-1], ETri[0:N,l,p-1])
                        CurlE[0:N, m-1] = homLbet[k-1,q-1]*CurlETri[0:N,l,p-1] + DhomLbetxETri

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
    ''' This function computes compute edge Hcurl ancillary functions and
    their curls

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
    '''

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
    ''' This function computes compute triangle face Hcurl ancillary
    functions and their curls

    :param ndarray S: (s0,s1,s2) affine coordinates associated to triangle face
    :param ndarray DS: derivatives of S0,S1,S2
    :param int Nord: polynomial order
    :param boll Idec: Binary flag:
    :param int N: spatial dimension
    :return: triangle Hcurl ancillary functions and curls of triangle Hcurl ancillary functions
    :rtype: ndarray
    '''

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
    maxI = Nord-2
    minJ = 1
    maxJ = Nord-1
    minIJ = minI+minJ
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
    ''' This function returns values of homogenized Legendre polynomials

    :param ndarray S: affine(like) coordinates
    :param int Nord: polynomial order
    :return: polynomial values
    :rtype: ndarray
    '''

    # Simply the definition of homogenized polynomials
    HomP = PolyLegendre(S[1], S[0]+S[1], Nord)

    return HomP


def HomIJacobi(S, DS, Nord, Minalpha, Idec, N):
    ''' This function returns values of integrated homogenized Jacobi polynomials and
    their gradients. Result is half of a  matrix with each row  associated
    to a fixed alpha. Alpha grows by 2 in each row.

    :param ndarray S: (s0,s1) affine(like) coordinates
    :param ndarray DS: gradients of S in R^N
    :param int Nord: max polynomial order
    :param int Minalpha: first row value of alpha (integer)
    :param bool Idec: decision flag to compute
    :return polynomial values and derivatives in x (Jacobi polynomials)
    :rtype: ndarray
    '''
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
    ''' This function returns values of shifted scaled Legendre polynomials

    :param ndarray X: coordinate from [0,1]
    :param float T: scaling parameter
    :param int Nord: polynomial order
    :return: polynomial values
    :rtype: ndarray
    '''
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
    ''' This function computes values of integrated shifted scaled Jacobi polynomials
    and their derivatives starting with p=1. Result is 'half' of a  matrix
    with each row  associated to a fixed alpha. Alpha grows by 2 in each row.

    :param ndarray X: coordinate from [0,1]
    :param ndarray T: scaling parameter
    :param int Nord: max polynomial order
    :param int Minalpha: = first row value of alpha
    :param bool Idec = decision flag to compute (= FALSE polynomials with x and t derivatives, = TRUE  polynomials with x derivatives only)
    :return: polynomial values, derivatives in x (Jacobi polynomials), derivatives in t
    '''

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
    ''' This function computes values of shifted scaled Jacobi polynomials
    P**alpha-i. Result is a half of a  matrix with each row
    associated to a fixed alpha. Alpha grows by 2 in each row.

    :param ndarray X: coordinate from [0,1]
    :param float T: scaling parameter
    :param int Nord: max polynomial order
    :param int Minalpha: first row value of alpha (integer)
    :return: polynomial values
    :rtype: ndarray
    '''

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
    ''' This function computes the local to global transformations of edges

    :param ndarray S: projection of affine coordinates on edges
    :param ndarray DS: projection of gradients of affine coordinates on edges
    :param ndarray Nori: edge orientation
    :param int N: number of dimensions
    :return: global transformation of edges and global transformation of gradients of edges
    :rtype: ndarray
    '''

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
    ''' This function computes the local to global transformations of edges

    :param ndarray S: projection of affine coordinates on faces
    :param ndarray DS: projection of gradients of affine coordinates on faces
    :param ndarray Nori: face orientation
    :param int N: number of dimensions
    :return: global transformation of faces and global transformation of gradients of faces
    :rtype: ndarray
    '''

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
    ''' This function projection of tetrahedral edges in concordance with
    numbering of topological entities (vertices, edges, faces)

    :param ndarray Lam: affine coordinates
    :param ndarray DLam: gradients of affine coordinates
    :return: projection of affine coordinates on edges, projection of gradients of affine coordinates on edges
    :rtype: ndarray

    .. note:: References:\n
       Fuentes, F., Keith, B., Demkowicz, L., & Nagaraj, S. (2015). Orientation
       embedded high order shape functions for the exact sequence elements of
       all shapes. Computers & Mathematics with applications, 70(4), 353-458.
    '''

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
    ''' This function projection of tetrahedral faces in concordance with
    numbering of topological entities (vertices, edges, faces)

    :param ndarray Lam: affine coordinates
    :param ndarray DLam: gradients of affine coordinates
    :return: projection of affine coordinates on faces, projection of gradients of affine coordinates on faces
    :rtype: ndarray
    '''

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
    ''' This function computes affine coordinates and their gradients.

    :param ndarray X: point coordinates
    :return: affine coordinates and gradients of affine coordinates
    :rtype: ndarray

    .. note:: References:\n
       Fuentes, F., Keith, B., Demkowicz, L., & Nagaraj, S. (2015). Orientation
       embedded high order shape functions for the exact sequence elements of
       all shapes. Computers & Mathematics with applications, 70(4), 353-458.
    '''

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
    ''' This function computes gauss points for high-order nédélec elements

    :param int Nord: polynomial order of nedelec basis functions
    :return: coordinates of gauss points and its weights
    :rtype: ndarray.
    '''
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


def computeSourceVectorRotation(model):
    ''' This function compute the weigths vector for source rotation in the
    xyz plane

    :param object model: object model with source data.
    :return: weigths for source rotation
    :rtype: ndarray.
    '''

    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    # Define reference unit vector (x-directed pointing to East)
    base_vector = np.array([1,0,0], dtype=np.float)

    # ---------------------------------------------------------------
    # Compute vector for source rotation
    # ---------------------------------------------------------------
    # Convert degrees to radians for rotation
    alpha = model.src_azimuth*np.pi/180.    # x-y plane
    beta  = model.src_dip*np.pi/180.        # x-z plane
    tetha = 0*np.pi/180.                    # y-z plane

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
    ''' This function computes the reference tetrahedron coordinates from
    xyz global tetrahedron coordinates.

    :param ndarray eleNodes: spatial coordinates of the nodes with dimensions = (4,3)
    :param ndarray points: xyz points coordinates to be transformed
    :return: xietazeta points coordinates
    :rtype: ndarray
    '''
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


def computeBasisFunctions(edge_orientation, face_orientation, invjacob, Nord, points):
    ''' This function computes the basis function for a given element

    :param ndarray edges_orientation: orientation for edges
    :param ndarray faces_orientation: orientation for faces
    :param ndarray jacobian: jacobian matrix
    :param ndarray invjacob: inverse of jacobian matrix
    :param int Nord: polynomial order of nedelec basis functions
    :param ndarray points: spatial points at which basis functions will be computed
    '''

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
        NrdofE, ShapE, _ = shape3DETet(X, Nord_vector, NoriE, NoriF)

        # Verify consistency of number of dofs for this point
        if (NrdofE != num_dof_in_element):
            Print.master('        Number of DOFs is not consistent')
            exit(-1)

        # Niref=Ni in reference element
        Niref = ShapE[0:3, 0:NrdofE]

        # Ni=Ni in real element
        Ni_real = np.matmul(invjacob, Niref)

        # Store basis functions for i
        basis[:,:,i] = Ni_real

    return basis


def unitary_test():
    ''' Unitary test for hvfem.py script.
    '''


if __name__ == '__main__':
    unitary_test()
