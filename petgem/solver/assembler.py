#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define functions for assembly of sparse linear systems in Edge Finite
Element Method (EFEM) of lowest order in tetrahedral meshes.
'''


def computeElementalContributionsMPI(modelling, coordEle, nodesEle, sigmaEle):
    '''Compute the elemental contributions of matrix A (LHS) and right hand
    side (RHS) in a parallel-vectorized manner for CSEM surveys by EFEM. Here,
    all necessary arrays are populated (Distributed-memory approach).

    :param dictionary modelling: CSEM modelling with physical parameters.
    :param ndarray coordEle: array with nodal coordinates of element.
    :param ndarray nodesEle: array with nodal indexes of element.
    :param float sigmaEle: element conductiviy.
    :return: Ae, be.
    :rtype: complex.
    '''
    # ----------- Read physical parameters -----------
    FREQ = np.float(modelling['FREQ'])
    SIGMA_BGROUND = np.float(modelling['CONDUCTIVITY_BACKGROUND'])
    SRC_POS = np.asarray(modelling['SRC_POS'], dtype=np.float)
    SRC_DIREC = np.int(modelling['SRC_DIREC'])
    I = np.float(modelling['SRC_CURRENT'])
    dS = np.float(modelling['SRC_LENGTH'])

    # ----------- Edge order -----------
    edgeOrder = 6
    # ----------- Nodal order -----------
    nodalOrder = 4
    # ----------- Number of dimensions -----------
    nDimensions = 3

    # ----------- Definition of constants-----------
    ZERO = np.float(0.0)
    ONE = np.float(1.0)
    TWO = np.float(2.0)
    THREE = np.float(3.0)
    FOUR = np.float(4.0)
    SIX = np.float(6.0)
    # Imaginary part for complex numbers
    IMAG_PART = np.complex128(0.0 + 1.0j)
    # Volume constants
    CONST_VOL1 = np.float(1.0)/np.float(6.0)
    CONST_VOL2 = np.float(360.0)
    CONST_VOL3 = np.float(720.0)
    # Vacuum permeability
    MU = np.float(FOUR*np.pi*np.float(1.0e-7))
    # Angular frequency
    OMEGA = np.float(FREQ*TWO*np.pi)
    # Propagation parameter
    WAVENUMBER = np.complex(np.sqrt(-IMAG_PART*MU*OMEGA*SIGMA_BGROUND))
    # Physical constants
    CONST_PHY1 = I * dS
    CONST_PHY2 = FOUR * np.pi * SIGMA_BGROUND
    CONST_PHY3 = -IMAG_PART * WAVENUMBER
    CONST_PHY4 = -WAVENUMBER**2
    CONST_PHY5 = THREE * IMAG_PART * WAVENUMBER
    CONST_PHY6 = IMAG_PART*OMEGA*MU

    # ----------- Gaussian points for the unit -----------
    # ------------ reference tetrahedron -----------------
    polyOrder = 3
    [gaussP, gaussW] = gauss_points_tetrahedron(polyOrder)

    # Number of Gaussian points
    ngaussP = gaussP.shape[0]

    # Normalization of gauss points
    SUM_GAUSS_WEIGTHS = np.sum(gaussW)

    # ----------- Edge definition for the tetrahedral elements -----------
    # ----- Table 8.2 of Jin's book. Here is consider as an 1D-array -----
    edgesN = np.array([0, 1, 0, 2, 0, 3, 1, 2, 3, 1, 2, 3], dtype=np.int)

    # ----------- Definition of arrays for vector operations -----------
    # Signs computation
    idx_signs1 = np.array([1, 2, 3, 2, 1, 3], dtype=np.int)
    idx_signs2 = np.array([0, 0, 0, 1, 3, 2], dtype=np.int)

    # ----------- Allocate arrays for vector operations -----------
    # Lagrange coefficients
    allocate = nodalOrder
    a_coeff = np.zeros(allocate, dtype=np.float)
    b_coeff = np.zeros(allocate, dtype=np.float)
    c_coeff = np.zeros(allocate, dtype=np.float)
    d_coeff = np.zeros(allocate, dtype=np.float)
    # Coordinates of evaluation points (primaty field)
    allocate = nDimensions
    X_Coeff = np.zeros(allocate, dtype=np.float)
    Y_Coeff = np.zeros(allocate, dtype=np.float)
    Z_Coeff = np.zeros(allocate, dtype=np.float)
    allocate = ngaussP
    pEvalX = np.zeros(allocate, dtype=np.float)
    pEvalY = np.zeros(allocate, dtype=np.float)
    pEvalZ = np.zeros(allocate, dtype=np.float)
    # Nedelec basis functions
    allocate = ngaussP*edgeOrder
    temp_A1 = np.zeros(allocate, dtype=np.float)
    temp_A2 = np.zeros(allocate, dtype=np.float)
    temp_b1 = np.zeros(allocate, dtype=np.float)
    temp_b2 = np.zeros(allocate, dtype=np.float)
    basis = np.zeros((nDimensions, allocate), dtype=np.float)
    # Vector operations over edges
    rep_edges1 = np.repeat((0, 1, 2, 3, 4, 5), ngaussP)
    rep_edges2 = np.repeat((0, 1, 2, 3, 4, 5), edgeOrder)
    rep_edges3 = np.tile((0, 1, 2, 3, 4, 5), edgeOrder)
    # Vector operations over gauss points
    idxbx1 = np.repeat((1, 2, 3, 2, 1, 3), ngaussP)
    idxbx2 = np.repeat((0, 0, 0, 1, 3, 2), ngaussP)
    # Mass matrix
    allocate = edgeOrder**2
    Me = np.zeros(allocate, dtype=np.float)
    # Stiffness matrix
    rep_edges_stiff1 = np.tile((0, 0, 0, 1, 3, 2), edgeOrder)
    rep_edges_stiff2 = np.tile((1, 2, 3, 2, 1, 3), edgeOrder)
    rep_edges_stiff3 = np.array([0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0,
                                 1, 1, 1, 1, 1, 1,
                                 3, 3, 3, 3, 3, 3,
                                 2, 2, 2, 2, 2, 2], dtype=np.int)
    rep_edges_stiff4 = np.repeat((1, 2, 3, 2, 1, 3), edgeOrder)
    # Stiffness matrix
    Ke = np.zeros(allocate, dtype=np.float)
    # Auxiliar vectors
    # Elemental matrix
    allocate = nodalOrder
    f1 = np.zeros(allocate, dtype=np.float)
    f2 = np.zeros(allocate, dtype=np.float)
    f3 = np.zeros(allocate, dtype=np.float)
    f4 = np.zeros(allocate, dtype=np.float)
    allocate = edgeOrder**2
    std_v1 = np.zeros(allocate, dtype=np.float)
    std_v2 = np.zeros(allocate, dtype=np.float)
    std_v3 = np.zeros(allocate, dtype=np.float)
    std_v4 = np.zeros(allocate, dtype=np.float)
    std_v5 = np.zeros(allocate, dtype=np.float)
    std_v6 = np.zeros(allocate, dtype=np.float)
    std_v7 = np.zeros(allocate, dtype=np.float)
    std_v8 = np.zeros(allocate, dtype=np.float)
    std_v9 = np.zeros(allocate, dtype=np.float)
    std_v10 = np.zeros(allocate, dtype=np.float)
    std_v11 = np.zeros(allocate, dtype=np.float)
    std_v12 = np.zeros(allocate, dtype=np.float)
    std_v13 = np.zeros(allocate, dtype=np.float)
    std_v14 = np.zeros(allocate, dtype=np.float)
    std_v15 = np.zeros(allocate, dtype=np.float)
    std_v16 = np.zeros(allocate, dtype=np.float)
    std_v17 = np.zeros(allocate, dtype=np.float)
    std_v18 = np.zeros(allocate, dtype=np.float)
    std_v19 = np.zeros(allocate, dtype=np.float)
    std_v20 = np.zeros(allocate, dtype=np.float)
    std_v21 = np.zeros(allocate, dtype=np.float)
    std_v22 = np.zeros(allocate, dtype=np.float)
    std_v23 = np.zeros(allocate, dtype=np.float)
    std_v24 = np.zeros(allocate, dtype=np.float)
    std_v25 = np.zeros(allocate, dtype=np.float)
    # Distance to source
    allocate = ngaussP
    distX = np.zeros(allocate, dtype=np.float)
    distY = np.zeros(allocate, dtype=np.float)
    distZ = np.zeros(allocate, dtype=np.float)
    distance = np.zeros(allocate, dtype=np.float)
    # Electric field
    field = np.zeros((nDimensions, allocate), dtype=np.complex)
    indx_field = np.tile((np.arange(0, ngaussP)), edgeOrder)
    allocate = edgeOrder*ngaussP
    temp_field = np.zeros((nDimensions, allocate), dtype=np.complex)
    # Elemental matrix and elemental vector
    allocate = edgeOrder**2
    Ae = np.zeros(allocate, dtype=np.complex)
    allocate = edgeOrder
    be = np.zeros(allocate, dtype=np.complex)
    # Indexes within dimensions of gauss points and basis functions
    idx_gaussP_1 = ngaussP
    idx_gaussP_2 = ngaussP*2
    idx_gaussP_3 = ngaussP*3
    idx_gaussP_4 = ngaussP*4
    idx_gaussP_5 = ngaussP*5

    # ----------- Compute element's volume -----------
    eleVol = (((coordEle[3]-coordEle[0])*(coordEle[7]-coordEle[1]) *
               (coordEle[11]-coordEle[2]) +
               (coordEle[4]-coordEle[1])*(coordEle[8]-coordEle[2]) *
               (coordEle[9]-coordEle[0]) +
               (coordEle[6]-coordEle[0])*(coordEle[10]-coordEle[1]) *
               (coordEle[5]-coordEle[2])) -
              ((coordEle[5]-coordEle[2])*(coordEle[7]-coordEle[1]) *
               (coordEle[9]-coordEle[0]) +
               (coordEle[6]-coordEle[0])*(coordEle[4]-coordEle[1]) *
               (coordEle[11]-coordEle[2]) +
               (coordEle[10]-coordEle[1])*(coordEle[8]-coordEle[2]) *
               (coordEle[3]-coordEle[0]))) * CONST_VOL1

    # ----------- Compute edges's length of element -----------
    tmp = coordEle.reshape(4, 3)
    tmp = tmp[edgesN, :]
    edges = tmp[1::2, :] - tmp[0::2, :]
    lengthEle = np.sqrt(np.sum(np.square(edges), axis=1))

    # ----------- Delta sigma of element -----------
    deltaSigma = sigmaEle - SIGMA_BGROUND

    # ----------- Edge's signs -----------
    tmp = nodesEle
    tmp = tmp[idx_signs1] - tmp[idx_signs2]
    signsEle = tmp / np.abs(tmp)

    # ----------- Compute Lagrange coefficients -----------
    # Constants
    S1 = coordEle[3]*coordEle[7]
    S2 = coordEle[5]*coordEle[6]
    S3 = coordEle[4]*coordEle[8]
    S4 = coordEle[9]*coordEle[7]
    S5 = coordEle[6]*coordEle[4]
    S6 = coordEle[10]*coordEle[8]
    S7 = coordEle[6]*coordEle[10]
    S8 = coordEle[9]*coordEle[1]
    S9 = coordEle[7]*coordEle[11]
    S10 = coordEle[1]*coordEle[11]
    S11 = coordEle[0]*coordEle[4]
    S12 = coordEle[10]*coordEle[2]
    S13 = coordEle[0]*coordEle[10]
    S14 = coordEle[4]*coordEle[2]
    S15 = coordEle[3]*coordEle[1]
    S16 = coordEle[7]*coordEle[5]
    # Coefficients a
    a_coeff[0] = ((S1*coordEle[11] + S2*coordEle[10] + S3*coordEle[9]) -
                  (S4*coordEle[5] + S5*coordEle[11] + S6*coordEle[3]))
    a_coeff[1] = (S7*coordEle[2] + S8*coordEle[8] + S9*coordEle[0] -
                  (coordEle[0]*S6 + S4*coordEle[2] + S10*coordEle[6]))*-ONE
    a_coeff[2] = (S8*coordEle[5] + S11*coordEle[11] + S12*coordEle[3] -
                  (coordEle[3]*S10 + S13*coordEle[5] + S14*coordEle[9]))
    a_coeff[3] = (coordEle[0]*S3 + S1*coordEle[2] + coordEle[1]*S2 -
                  (S5*coordEle[2] + S15*coordEle[8] +
                   S16*coordEle[0]))*-ONE
    # Coefficients b
    b_coeff[0] = ((S9 + coordEle[10]*coordEle[5] + S3) -
                  (S16 + coordEle[4]*coordEle[11] + S6))*-ONE
    b_coeff[1] = (S12 + coordEle[1]*coordEle[8] + S9 -
                  (S6 + coordEle[7]*coordEle[2] + S10))
    b_coeff[2] = (coordEle[1]*coordEle[5] + coordEle[4]*coordEle[11] +
                  S12 - (S10 + coordEle[10]*coordEle[5] + S14))*-ONE
    b_coeff[3] = (S3 + coordEle[7]*coordEle[2] + coordEle[1]*coordEle[5] -
                  (S14 + coordEle[1]*coordEle[8] + S16))
    # Coefficients c
    c_coeff[0] = (coordEle[6]*coordEle[11] + coordEle[9]*coordEle[5] +
                  coordEle[3]*coordEle[8] - (S2 +
                                             coordEle[3]*coordEle[11] +
                                             coordEle[9]*coordEle[8]))
    c_coeff[1] = (coordEle[9]*coordEle[2] +
                  coordEle[0]*coordEle[8] + coordEle[6]*coordEle[11] -
                  (coordEle[9]*coordEle[8] + coordEle[6]*coordEle[2] +
                   coordEle[0]*coordEle[11]))*-ONE
    c_coeff[2] = (coordEle[0]*coordEle[5] + coordEle[3]*coordEle[11] +
                  coordEle[9]*coordEle[2] - (coordEle[0]*coordEle[11] +
                                             coordEle[9]*coordEle[5] +
                                             coordEle[3]*coordEle[2]))
    c_coeff[3] = (coordEle[3]*coordEle[8] + coordEle[6]*coordEle[2] +
                  coordEle[0]*coordEle[5] - (coordEle[3]*coordEle[2] +
                                             coordEle[0]*coordEle[8] +
                                             S2))*-ONE
    # Coefficients d
    d_coeff[0] = (S7 + coordEle[9]*coordEle[4] + S1 -
                  (S5 + coordEle[3]*coordEle[10] + S4))*-ONE
    d_coeff[1] = (S8 + coordEle[0]*coordEle[7] + S7 -
                  (S4 + coordEle[6]*coordEle[1] + S13))
    d_coeff[2] = (S11 + coordEle[3]*coordEle[10] + S8 -
                  (S13 + coordEle[9]*coordEle[4] + S15))*-ONE
    d_coeff[3] = (S1 + coordEle[6]*coordEle[1] + S11 -
                  (S15 + coordEle[0]*coordEle[7] + S5))

    # ----------- Map XiEtaZeta Gaussian points to real element -----------
    # Compute coefficients
    X_Coeff[0] = coordEle[3]-coordEle[0]
    X_Coeff[1] = coordEle[6]-coordEle[0]
    X_Coeff[2] = coordEle[9]-coordEle[0]
    Y_Coeff[0] = coordEle[4]-coordEle[1]
    Y_Coeff[1] = coordEle[7]-coordEle[1]
    Y_Coeff[2] = coordEle[10]-coordEle[1]
    Z_Coeff[0] = coordEle[5]-coordEle[2]
    Z_Coeff[1] = coordEle[8]-coordEle[2]
    Z_Coeff[2] = coordEle[11]-coordEle[2]
    # X-coordinates
    pEvalX = coordEle[0] + np.sum(X_Coeff*gaussP, axis=1)
    # Y-coordinates
    pEvalY = coordEle[1] + np.sum(Y_Coeff*gaussP, axis=1)
    # Z-coordinates
    pEvalZ = coordEle[2] + np.sum(Z_Coeff*gaussP, axis=1)

    # ----------- Nedelec basis computation -----------
    # Reduce number of multiplications
    A1 = (a_coeff[0] + b_coeff[0]*pEvalX +
          c_coeff[0]*pEvalY + d_coeff[0]*pEvalZ)
    A2 = (a_coeff[1] + b_coeff[1]*pEvalX +
          c_coeff[1]*pEvalY + d_coeff[1]*pEvalZ)
    A3 = (a_coeff[2] + b_coeff[2]*pEvalX +
          c_coeff[2]*pEvalY + d_coeff[2]*pEvalZ)
    A4 = (a_coeff[3] + b_coeff[3]*pEvalX +
          c_coeff[3]*pEvalY + d_coeff[3]*pEvalZ)
    # Prepare data for vector multiplications
    tmp_edges = lengthEle[rep_edges1]*((SIX*eleVol)**2)**-1
    # First component of x-basis
    temp_b1 = b_coeff[idxbx1]
    temp_A1[0:idx_gaussP_1] = A1
    temp_A1[idx_gaussP_1:idx_gaussP_2] = temp_A1[0:idx_gaussP_1]
    temp_A1[idx_gaussP_2:idx_gaussP_3] = temp_A1[0:idx_gaussP_1]
    temp_A1[idx_gaussP_3:idx_gaussP_4] = A2
    temp_A1[idx_gaussP_4:idx_gaussP_5] = A4
    temp_A1[idx_gaussP_5:] = A3
    # Second component of x_basis
    temp_b2 = b_coeff[idxbx2]
    temp_A2[0:idx_gaussP_1] = A2
    temp_A2[idx_gaussP_1:idx_gaussP_2] = A3
    temp_A2[idx_gaussP_2:idx_gaussP_3] = A4
    temp_A2[idx_gaussP_3:idx_gaussP_4] = A3
    temp_A2[idx_gaussP_4:idx_gaussP_5] = A2
    temp_A2[idx_gaussP_5:] = A4
    # X-component for all points and for all basis
    basis[0, :] = (temp_b1*temp_A1 - temp_b2*temp_A2)*tmp_edges
    # First component of y-basis
    temp_b1 = c_coeff[idxbx1]
    # Second component of y_basis
    temp_b2 = c_coeff[idxbx2]
    # Y-component for all points and for all basis
    basis[1, :] = (temp_b1*temp_A1 - temp_b2*temp_A2)*tmp_edges
    # First component of z-basis
    temp_b1 = d_coeff[idxbx1]
    # Second component of z_basis
    temp_b2 = d_coeff[idxbx2]
    # Y-component for all points and for all basis
    basis[2, :] = (temp_b1*temp_A1 - temp_b2*temp_A2)*tmp_edges

    # ----------- Computation of elemental matrix -----------
    # Compute mass matrix
    f1 = b_coeff[0]*b_coeff + c_coeff[0]*c_coeff + d_coeff[0]*d_coeff
    f2 = b_coeff[1]*b_coeff + c_coeff[1]*c_coeff + d_coeff[1]*d_coeff
    f3 = b_coeff[2]*b_coeff + c_coeff[2]*c_coeff + d_coeff[2]*d_coeff
    f4 = b_coeff[3]*b_coeff + c_coeff[3]*c_coeff + d_coeff[3]*d_coeff
    AA1 = (CONST_VOL2*eleVol)**-1
    AA2 = (CONST_VOL3*eleVol)**-1
    # Integral: Eq. 8.68. Upper triangular matrix
    Me[0] = lengthEle[0]**2 * AA1 * (f2[1] - f1[1] + f1[0])
    Me[1] = lengthEle[0] * lengthEle[1] * AA2 * (2*f2[2] - f1[1] -
                                                 f1[2] + f1[0])
    Me[2] = lengthEle[0] * lengthEle[2] * AA2 * (2*f2[3] - f1[1] -
                                                 f1[3] + f1[0])
    Me[3] = lengthEle[0] * lengthEle[3] * AA2 * (f2[2]-f2[1] -
                                                 2*f1[2] + f1[1])
    Me[4] = lengthEle[0] * lengthEle[4] * AA2 * (f2[1] - f2[3] -
                                                 f1[1] + 2*f1[3])
    Me[5] = lengthEle[0] * lengthEle[5] * AA2 * (f2[3] - f2[2] -
                                                 f1[3] + f1[2])
    Me[7] = lengthEle[1]**2 * AA1 * (f3[2] - f1[2] + f1[0])
    Me[8] = lengthEle[1] * lengthEle[2] * AA2 * (2*f3[3] - f1[2] -
                                                 f1[3] + f1[0])
    Me[9] = lengthEle[1] * lengthEle[3] * AA2 * (f3[2] - f2[2] -
                                                 f1[2] + 2*f1[1])
    Me[10] = lengthEle[1] * lengthEle[4] * AA2 * (f2[2] - f3[3] -
                                                  f1[1] + f1[3])
    Me[11] = lengthEle[1] * lengthEle[5] * AA2 * (f1[2] - f3[2] -
                                                  2*f1[3] + f3[3])
    Me[14] = lengthEle[2]**2 * AA1 * (f4[3] - f1[3] + f1[0])
    Me[15] = lengthEle[2] * lengthEle[3] * AA2 * (f3[3] - f2[3] -
                                                  f1[2] + f1[1])
    Me[16] = lengthEle[2] * lengthEle[4] * AA2 * (f2[3] - f4[3] -
                                                  2*f1[1] + f1[3])
    Me[17] = lengthEle[2] * lengthEle[5] * AA2 * (f4[3] - f3[3] -
                                                  f1[3] + 2*f1[2])
    Me[21] = lengthEle[3]**2 * AA1 * (f3[2] - f2[2] + f2[1])
    Me[22] = lengthEle[3] * lengthEle[4] * AA2 * (f2[2] - 2*f3[3] -
                                                  f2[1] + f2[3])
    Me[23] = lengthEle[3] * lengthEle[5] * AA2 * (f3[3] - f3[2] -
                                                  2*f2[3] + f2[2])
    Me[28] = lengthEle[4]**2 * AA1 * (f2[1] - f2[3] + f4[3])
    Me[29] = lengthEle[4] * lengthEle[5] * AA2 * (f2[3] - 2*f2[2] -
                                                  f4[3] + f3[3])
    Me[35] = lengthEle[5]**2 * AA1 * (f4[3] - f3[3] + f3[2])
    # Copy upper triangular matrix to lower triangular matrix
    Me[[6, 12, 18, 24, 30, 13, 19, 25,
        31, 20, 26, 32, 27, 33, 34]] = Me[[1, 2, 3, 4, 5, 8, 9, 10,
                                           11, 15, 16, 17, 22, 23, 29]]
    # Mass matrix
    Me = Me*signsEle[rep_edges2]*signsEle[rep_edges3]

    # Compute stiffness matrix
    std_v1 = b_coeff[rep_edges_stiff1]
    std_v2 = b_coeff[rep_edges_stiff2]
    std_v3 = b_coeff[rep_edges_stiff3]
    std_v4 = b_coeff[rep_edges_stiff4]
    std_v5 = c_coeff[rep_edges_stiff1]
    std_v6 = c_coeff[rep_edges_stiff2]
    std_v7 = c_coeff[rep_edges_stiff3]
    std_v8 = c_coeff[rep_edges_stiff4]
    std_v9 = d_coeff[rep_edges_stiff1]
    std_v10 = d_coeff[rep_edges_stiff2]
    std_v11 = d_coeff[rep_edges_stiff3]
    std_v12 = d_coeff[rep_edges_stiff4]
    std_v13 = std_v5*std_v10
    std_v14 = std_v9*std_v6
    std_v15 = std_v7*std_v12
    std_v16 = std_v11*std_v8
    std_v17 = std_v9*std_v2
    std_v18 = std_v1*std_v10
    std_v19 = std_v11*std_v4
    std_v20 = std_v3*std_v12
    std_v21 = std_v1*std_v6
    std_v22 = std_v5*std_v2
    std_v23 = std_v3*std_v8
    std_v24 = std_v7*std_v4
    std_v25 = (lengthEle[rep_edges2] * lengthEle[rep_edges3] *
               signsEle[rep_edges2] * signsEle[rep_edges3])

    # Stiffness matrix
    Ke = ((std_v13 - std_v14) *
          (std_v15 - std_v16) +
          (std_v17 - std_v18) *
          (std_v19 - std_v20) +
          (std_v21 - std_v22) *
          (std_v23 - std_v24)) * (FOUR*eleVol)/((SIX*eleVol)**4)*std_v25

    # Compute elemental matrix
    Ae = Ke + np.multiply(CONST_PHY6*sigmaEle, Me)

    # ----------- Contributions for vector b -----------
    # Gaussian points weigths (normalization)
    weightEle = gaussW * (eleVol / SUM_GAUSS_WEIGTHS)

    # Compute distance to the source for all Gaussian points
    distX[:] = pEvalX - SRC_POS[0]     # X-component
    distY[:] = pEvalY - SRC_POS[1]     # Y-component
    distZ[:] = pEvalZ - SRC_POS[2]     # Z-component
    distance[:] = np.sqrt(distX**2 + distY**2 + distZ**2)

    # To avoid very large or small numbers
    distance[distance < ONE] = ONE
    # Compute the primary field for all Gaussian points
    # E = AA [ BB + (wavenumber^2*distance^2 -1i*wavenumber*distance-1)]
    SQUARE_DISTANCE = distance**2
    AA = CONST_PHY1 / (CONST_PHY2 * distance**3) * np.exp(CONST_PHY3 *
                                                          distance)
    BB = CONST_PHY4 * SQUARE_DISTANCE + (CONST_PHY5 * distance) + THREE
    RR = ONE/SQUARE_DISTANCE

    # Compute primary field in function of source direction
    if SRC_DIREC == 1:
        # X-directed
        field[0, :] = AA * ((distX**2 * RR)*BB +
                            (WAVENUMBER**2 * SQUARE_DISTANCE - IMAG_PART *
                             WAVENUMBER * distance - ONE))
        field[1, :] = AA * (distX*distY*RR)*BB
        field[2, :] = AA * (distX*distZ*RR)*BB
    elif SRC_DIREC == 2:
        # Y-directed
        field[0, :] = AA * (distX*distY*RR)*BB
        field[1, :] = AA * ((distY**2 * RR)*BB +
                            (WAVENUMBER**2 * SQUARE_DISTANCE - IMAG_PART *
                             WAVENUMBER * distance - ONE))
        field[2, :] = AA * (distY*distZ*RR)*BB
    else:
        # Z-directed
        field[0, :] = AA * (distX*distZ*RR)*BB
        field[1, :] = AA * (distZ*distY*RR)*BB
        field[2, :] = AA * ((distZ**2 * RR)*BB +
                            (WAVENUMBER**2 * SQUARE_DISTANCE - IMAG_PART *
                             WAVENUMBER * distance - ONE))

    # Integral over edges
    temp_field = field[:, indx_field]

    signsEle = signsEle[rep_edges1]
    weightEle = weightEle[indx_field]

    temp = np.sum(temp_field*basis, 0) * signsEle * weightEle * deltaSigma

    # Integral over edges
    for iBasis in np.arange(edgeOrder):
        for iPoint in np.arange(ngaussP):
            be[iBasis] = be[iBasis] + temp[(iBasis*ngaussP)+iPoint]

    # Scale vector by constant -1i*OMEGA*MU
    be = be*-CONST_PHY6

    return Ae, be


def unitary_test():
    ''' Unitary test for assembler.py script.
    '''


if __name__ == '__main__':
    # Standard module import
    unitary_test()
else:
    # Standard module import
    import numpy as np
    import petsc4py
    from petsc4py import PETSc
    # PETGEM module import
    from petgem.efem.fem import gauss_points_tetrahedron
