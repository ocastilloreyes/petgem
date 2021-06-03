#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
"""Define functions for magnetotelluric modeling within **PETGEM**."""

# ---------------------------------------------------------------
# Load python modules
# ---------------------------------------------------------------
import numpy as np
from scipy.sparse import coo_matrix
from scipy.linalg import lu, solve_triangular
from scipy.sparse.linalg import spsolve

# ---------------------------------------------------------------
# Load petgem modules (BSC)
# ---------------------------------------------------------------

# ###############################################################
# ################     CLASSES DEFINITION      ##################
# ###############################################################
def eval_MT1D(za, zb, za_D, zb_D, sigma0, x0, omega, mu, n_nodes_ref, degree=1, interpolate_at=None,
              return_nodes=False):
    """Solves the 1D MT problem in [za, zb] with essential boundary conditions, using continuous finite elements of
       the specified degree (note than only few degrees may be implemented in the reference space).

    :param float za: left boundary of the 1D domain.
    :param float zb: right boundary of the 1D domain.
    :param float za_D: dirichlet condition at left boundary of the 1D domain.
    :param float zb_D: dirichlet condition at right boundary of the 1D domain.
    :param ndarray sigma0: conductivity values.
    :param ndarray x0: spacial positions for conductivity values.
    :param float omega: angular frequency.
    :param float mu: medium permeability.
    :param int n_nodes_ref: tentative number of nodes used for the 1D computation (it may be adjusted to be consistent with the specified element degree).
    :param int degree: finite element interpolation degree used for the 1D computation. [Default] 1
    :param ndarray interpolate_at: if specified, solution will be linearly interpolated and returned at these points. [Default] None
    :param bool return_nodes: if True, the spatial points used for the 1D computation will be also returned. [Default] False
    :return: solution of the 1D MT problem. If not None, returned at interpolate_at points with the same shape. Returned at computational points otherwise.
    :rtype: ndarray complex.

    .. note:: References:\n
       if return_nodes=True, spatial points used in the 1D computation.
    """

    # Mesh
    n_elems = np.int(np.ceil((n_nodes_ref-1) / degree))
    n_nodes = n_elems * degree + 1
    x = np.linspace(za, zb, n_nodes)
    T = create1Dconec(n_elems, degree)
    nOfElementNodes = (degree+1) * (degree+2) / 2
    reference_element = createReferenceElement(elementType=1, nOfElementNodes=nOfElementNodes, use_cubature=False)

    # Linear interpolation of model values
    x0_unique, x0_pos = np.unique(x0, return_index=True)
    sigma = linearInterp1D(x0_unique, sigma0[x0_pos], x)

    # System
    M, K = matrices_MT1D(x, T, reference_element, sigma)
    Mat = K + 1j * omega * mu * M
    f = np.zeros(n_nodes)

    # Essential boundary conditions
    Mat_bc, f_bc = applyEssentialBC(Mat, f, [za_D, zb_D], [0, n_nodes-1])

    # Solution
    u = spsolve(Mat_bc, f_bc)

    # Interpolation at interpolate_at points
    if interpolate_at is None:
        return (u, x) if return_nodes is True else u
    else:
        points_check, points_uniquePos = np.unique(interpolate_at.flatten(), return_inverse=True)
        u_points_check = linearInterp1D(x, u, points_check)
        u_interpolate_at = u_points_check[points_uniquePos]
        sol = u_interpolate_at.reshape(interpolate_at.shape)
        return (sol, x) if return_nodes is True else sol


def create1Dconec(nOfElems, p, list_nodes=None):

    T = np.zeros((nOfElems, p+1), dtype=np.int)

    cont = 0
    if list_nodes is None:
        for i in np.arange(nOfElems):
            T[i, :] = np.arange(cont, cont+p+1)
            cont += p
    else:
        for i in np.arange(nOfElems):
            T[i, :] = list_nodes[cont:cont+p+1]
            cont += p

    return T


def linearInterp1D(x, u, xp):

    # Sort
    xps_pos = np.argsort(xp)
    xs_pos = np.argsort(x)
    xps = xp[xps_pos]
    xs = x[xs_pos]
    us = u[xs_pos]

    # Init
    ups = np.zeros_like(xp, dtype=u.dtype)
    up = np.zeros_like(ups)
    ini = 0

    # Interpolate with 1D linear shape functions in [-1,1]
    for i in np.arange(1, x.size):
        for j in np.arange(ini, xp.size):
            if xps[j] > xs[i]:
                ini = j
                break
            else:
                h = xs[i] - xs[i-1]
                xi = (2*xps[j] - xs[i] - xs[i-1]) / h
                N1 = 0.5 * (1 - xi)
                N2 = 0.5 * (1 + xi)
                ups[j] = N1 * us[i-1] + N2 * us[i]
                ini = j + 1

    # Sort back the interpolated values
    up[xps_pos] = ups

    return up

# 1D integrals
def matrices_MT1D(mesh_nodes, conn_boundary, refelem, sigma):

    # Number of elements and number of mesh nodes
    number_elements, number_element_nodes = conn_boundary.shape
    number_nodes = mesh_nodes.size

    # Allocation for assembling indices
    M_values = np.zeros(number_element_nodes ** 2 * number_elements, dtype=np.float)
    K_values = np.zeros_like(M_values)
    global_index_row = np.zeros(number_element_nodes ** 2 * number_elements, dtype=np.int)
    global_index_col = np.zeros(number_element_nodes ** 2 * number_elements, dtype=np.int)

    # Number of gauss points
    number_gauss_points = len(refelem['IPweights1d'])

    # Loop in 1D elements
    for eidx in np.arange(number_elements):

        # Elemental mesh arrays
        Te = conn_boundary[eidx, :]
        Xe = mesh_nodes[Te]
        sigmae = sigma[Te]

        # Initialize elemental matrices
        Me = np.zeros((number_element_nodes, number_element_nodes), dtype=np.float)
        Ke = np.zeros_like(Me)

        # Integration in the current element
        for gauss_point in np.arange(number_gauss_points):

            # Shape functions and derivatives at the current integration point
            N_g = refelem['N1d'][gauss_point, :]
            Nxi_g = refelem['N1dxi'][gauss_point, :]

            # Non constant coefficient
            sigma_g = N_g @ sigmae

            # Jacobian and integration weight
            xyDer_g = Nxi_g @ Xe
            dline = refelem['IPweights1d'][gauss_point] * xyDer_g

            # Contribution of the current integration point to the elemental matrix
            Me = Me + dline * sigma_g * np.tensordot(N_g, N_g, axes=0)
            Ke = Ke + dline * (1 / (xyDer_g**2)) * np.tensordot(Nxi_g, Nxi_g, axes=0)

        # Assembling
        local_index_row = Te.repeat(number_element_nodes)
        local_index_col = np.tile(Te, number_element_nodes)
        ini_index = eidx * number_element_nodes ** 2
        end_index = ini_index + number_element_nodes ** 2
        global_index_row[ini_index:end_index] = local_index_row
        global_index_col[ini_index:end_index] = local_index_col
        M_values[ini_index:end_index] = Me.flatten()
        K_values[ini_index:end_index] = Ke.flatten()

    # Build sparse matrices (it also assembles zeros!)
    M = coo_matrix((M_values, (global_index_row, global_index_col)), dtype=np.float,
                   shape=(number_nodes, number_nodes)).tocsr()
    K = coo_matrix((K_values, (global_index_row, global_index_col)), dtype=np.float,
                   shape=(number_nodes, number_nodes)).tocsr()

    return M, K


def applyEssentialBC(K, f, alpha, nodes):
    # inefficient change of sparsity pattern for large len(nodes) if K is sparse

    ftot = f - K[:, nodes] @ alpha
    Ktot = K.copy()
    for i in np.arange(len(nodes)):
        node = nodes[i]
        Ktot[node, :] = 0
        Ktot[:, node] = 0
        Ktot[node, node] = 1
        ftot[node] = alpha[i]

    return Ktot, ftot

# Function to create the reference space
def createReferenceElement(elementType=1, nOfElementNodes=3, use_cubature=True):

    # Definition of reference element
    if elementType == 1:

        if nOfElementNodes == 3:

            nDeg = 1
            faceNodes = np.array([[1 ,2], [2 ,3], [3 ,1]])
            innerNodes = np.array([])
            faceNodes1d = np.array([1 ,2])
            coord2d = np.array([[-1 ,-1], [1 ,-1], [-1 ,1]])
            coord1d = np.array([-1 ,1])

        else: raise Exception('nOfElementNodes = %i not implemented' % nOfElementNodes)

    else: raise Exception('elementType = %i not implemented' % elementType)

    # Fekete nodes for nonlinear elements
    if nDeg > 3:
        pass

    # Computation of integration points and shape functions
    N, Nxi, Neta, gw2d, gp2d = computeShapeFunctionsReferenceElement2D(nDeg, coord2d, use_cubature=use_cubature)
    N1d, Nxi1d, gw1d, gp1d = computeShapeFunctionsReferenceElement1D(nDeg, coord1d)

    # Reference element
    reference_element = {
        'IPcoordinates':    gp2d,
        'IPweights':        gw2d,
        'N':                N.transpose(),
        'Nxi':              Nxi.transpose(),
        'Neta':             Neta.transpose(),
        'IPcoordinates1d':  gp1d,
        'IPweights1d':      gw1d,
        'N1d':              N1d.transpose(),
        'N1dxi':            Nxi1d.transpose(),
        'faceNodes':        faceNodes,
        'innerNodes':       innerNodes,
        'faceNodes1d':      faceNodes1d,
        'NodesCoord':       coord2d,
        'NodesCoord1d':     coord1d,
        'degree':           nDeg,
    }

    return reference_element


# Function to create shape functions 2D
def computeShapeFunctionsReferenceElement2D(nDeg, coord, use_cubature=True):

    # Number of element nodes for 2D triangles
    nOfNodes = int((nDeg + 1) * (nDeg + 2) / 2)

    # Vandermonde matrix
    V = Vandermonde_2D(nDeg, coord)
    P, L, U = lu(V.transpose())
    P = P.transpose()

    if use_cubature:

        # Not implemented yet, need implementation of GaussLegendreCubature2D
        raise Exception('Cubature not implemented yet')

        # # Set order of cubature for integration
        # if nDeg == 1:
        #     OrderCubature = 5
        # elif nDeg == 2:
        #     OrderCubature = 10
        # elif nDeg == 3:
        #     OrderCubature = 10
        # elif nDeg == 4:
        #     OrderCubature = 15
        # else: raise Exception('Cubature for degree = %i not implemented' % nDeg)
        #
        # # Compute quadrature
        # z, w = GaussLegendreCubature2D(OrderCubature)
        # w = 2 * w
        # z = 2 * z - 1
        # nIP = w.shape[0]
        #
        # # Shape functions
        # N = np.zeros((nOfNodes, nIP))
        # N_xi = N.copy()
        # N_eta = N.copy()
        # weights = np.zeros(nIP)
        # points = np.zeros((nIP, 2))
        # for i in np.arange(nIP):
        #     x = z[i, :]
        #     p, p_xi, p_eta = orthopoly2D_deriv_xieta(x, nDeg)
        #     B = np.array([p, p_xi, p_eta]).transpose()
        #     y0 = solve_triangular(L, P @ B, lower=True)
        #     y1 = solve_triangular(U, y0)
        #     N[:, i]     = y1[:, 0]
        #     N_xi[:, i]  = y1[:, 1]
        #     N_eta[:, i] = y1[:, 2]
        #     weights[i]   = w[i]
        #     points[i, :] = x

    else:

        # Compute quadrature
        nOfGaussPoints1D = nDeg + 2
        z, w = GaussLegendre1D(nOfGaussPoints1D, -1, 1)
        nIP_1D = w.shape[0]
        nIP = nIP_1D ** 2

        # Shape functions
        N = np.zeros((nOfNodes, nIP))
        N_xi = N.copy()
        N_eta = N.copy()
        weights = np.zeros(nIP)
        points = np.zeros((nIP, 2))
        iGauss = 0
        for i in np.arange(nIP_1D):
            for j in np.arange(nIP_1D):
                x = np.array([z[i], z[j]])
                p, p_xi, p_eta = orthopoly2D_deriv_rst(x, nDeg)
                B = np.array([p, p_xi, p_eta]).transpose()
                y0 = solve_triangular(L, P @ B, lower=True)
                y1 = solve_triangular(U, y0)
                N[:, iGauss]     = y1[:, 0]
                N_xi[:, iGauss]  = y1[:, 1]
                N_eta[:, iGauss] = y1[:, 2]
                r, s = x
                xi = (1 + r) * (1 - 2) / 2 - 1
                weights[iGauss]   = w[i] * w[j] * (1 - s) / 2
                points[iGauss, :] = np.array([xi, s])
                iGauss += 1

    return N, N_xi, N_eta, weights, points


# Function to create shape functions 1D
def computeShapeFunctionsReferenceElement1D(nDeg, coord):

    # Number of element nodes for 2D triangles
    nOfNodes = nDeg + 1

    # Vandermonde matrix
    V = Vandermonde_1D(nDeg, coord)
    P, L, U = lu(V.transpose())
    P = P.transpose()

    # Compute quadrature
    nOfGaussPoints1D = nDeg + 2
    z, w = GaussLegendre1D(nOfGaussPoints1D, -1, 1)
    nIP = w.shape[0]

    # Shape functions
    N = np.zeros((nOfNodes, nIP))
    N_xi = N.copy()
    for i in np.arange(nIP):
        x = z[i]
        p, p_xi = orthopoly1D_deriv(x, nDeg)
        B = np.array([p, p_xi]).transpose()
        y0 = solve_triangular(L, P @ B, lower=True)
        y1 = solve_triangular(U, y0)
        N[:, i]    = y1[:, 0]
        N_xi[:, i] = y1[:, 1]


    return N, N_xi, w, z


# Function to create Vandermonde matrix for 2D systems
def Vandermonde_2D(nDeg, coord):

    n = coord.shape[0]
    V = np.zeros([n, n])
    for i in np.arange(n):
        x = coord[i, :]
        p = orthopoly2D_deriv_xieta(x, nDeg)
        V[i, :] = p[0]

    return V


# Function to create Vandermonde matrix for 1D systems
def Vandermonde_1D(nDeg, coord):

    n = coord.shape[0]
    V = np.zeros([n, n])
    for i in np.arange(n):
        x = coord[i]
        p = orthopoly1D(x, nDeg)
        V[i, :] = p

    return V


# Function to create 1D orthogonal polynomial
def orthopoly1D(x, n):

    p = np.zeros([ n +1, x.shape[0]]) if len(x.shape) > 0 else np.zeros([ n +1, 1])

    for i in np.arange( n +1):
        p[i, :] = jacobiP_vect(x, 0, 0, i) * np.sqrt((2 * i + 1) / 2)

    return p if len(x.shape) > 0 else p.reshape( n +1)


# Function to create orthogonal 1D polynomial and derivative
def orthopoly1D_deriv(x, n):

    p = np.zeros([ n+1, x.shape[0]]) if len(x.shape) > 0 else np.zeros([ n +1, 1])
    dp = np.zeros([ n+1, x.shape[0]]) if len(x.shape) > 0 else np.zeros([ n +1, 1])

    p[0, :] = 1 / np.sqrt(2)
    for i in np.arange(1, n+ 1):
        factor = np.sqrt((2 * i + 1) / 2)
        p[i, :] = jacobiP_vect(x, 0, 0, i) * factor
        dp[i, :] = jacobiP_vect(x, 1, 1, i - 1) * ((i + 1) / 2) * factor

    if len(x.shape) > 0:
        return p, dp
    else:
        return p.reshape(n + 1), dp.reshape(n + 1)


# Function to evaluate Jacobi polynomial at points in vector x
def jacobiP_vect(x, a, b, n):

    if n == 0: return np.ones(x.shape)
    if n == 1: return (1 / 2) * (a - b + (2 + a + b) * x)

    apb = a + b
    apb_amb = (a + b) * (a - b)
    pm2 = 1
    pm1 = (1 / 2) * (a - b + (2 + a + b) * x)
    for i in np.arange(2, n + 1):
        A = 2 * i + apb
        B = i * (i + apb) * (A - 2)
        p = ((A - 1) * (apb_amb + x * (A - 2) * A) / (2 * B)) * pm1 \
            - ((i + a - 1) * (i + b - 1) * A / B) * pm2
        pm2 = pm1
        pm1 = p

    return p


# Function to compute Gauss-Legendre quadrature in [a,b]
def GaussLegendre1D(N, a, b):

    N = N - 1
    N1 = N + 1
    N2 = N + 2

    xu = np.linspace(-1, 1, N1)

    # Initial guess
    v_tmp = np.linspace(0, N, N + 1)
    y = np.cos((2 * v_tmp + 1) * np.pi / (2 * N + 2)) + (0.27 / N1) * np.sin(np.pi * xu * N / N2)

    # Legendre-Gauss Vandermonde Matrix
    L = np.zeros([N1, N2])

    # Compute the zeros of the N+1 Legendre polynomial using recursion relation and N-R method
    y0 = 2

    # Iterate until new points are uniformly within epsilon of old points
    while np.max(np.abs(y - y0)) > np.finfo(float).eps:

        L[:, 0] = 1
        L[:, 1] = y

        for k in np.arange(2, N1+1):
            L[:, k] = ((2 * k - 1) * y * L[:, k-1] - (k - 1) * L[:, k-2]) / k

        Lp = N2 * (L[:, N1-1] - y * L[:, N2-1]) / (1 - y ** 2)

        y0 = y
        y = y0 - L[:, N2-1] / Lp

    # Linear map to [a,b]
    x = (a * (1 - y) + b * (1 + y)) / 2

    # Compute weights
    w = (b - a) / ((1 - y ** 2) * Lp ** 2) * (N2 / N1) ** 2

    return x, w


# Function to create orthogonal base of 2D polynomials and derivatives of degree less or equal to n at point (r,s) in [-1,1]**2
def orthopoly2D_deriv_rst(x, n):

    N = np.int((n + 1) * (n + 2) / 2)
    p = np.zeros(N)
    dp_dxi = np.zeros(N)
    dp_deta = np.zeros(N)

    r, s = x

    xi = (1 + r) * (1 - s) / 2 - 1
    eta = s

    dr_dxi = 2 / (1 - eta)
    dr_deta = 2 * (1 + xi) / (1 - eta) ** 2

    ncount = 0
    for nDeg in np.arange(n+1):

        for i in np.arange(nDeg+1):

            if i == 0: p_i, q_i, dp_i, dq_i = 1, 1, 0, 0
            else:
                p_i = jacobiP_vect(np.array([r]), 0, 0, i)
                dp_i = jacobiP_vect(np.array([r]), 1, 1, i-1) * (i + 1) / 2
                q_i = q_i * (1 - s) / 2
                dq_i = q_i * (-i) / (1 - s)

            j = nDeg - i
            if j == 0: p_j, dp_j = 1, 0
            else:
                p_j = jacobiP_vect(np.array([s]), 2*i+1, 0, j)
                dp_j = jacobiP_vect(np.array([s]), 2*i+2, 1, j-1) * (j + 2 * i + 2) / 2

            factor = np.sqrt((2 * i + 1) * (i + j + 1) / 2)
            p[ncount] = (p_i * q_i * p_j) * factor
            dp_dr = (dp_i * q_i * p_j) * factor
            dp_ds = (p_i * (dq_i * p_j + q_i * dp_j)) * factor
            dp_dxi[ncount] = dp_dr * dr_dxi
            dp_deta[ncount] = dp_dr * dr_deta + dp_ds

            ncount += 1

    return p, dp_dxi, dp_deta


# Function to create orthogonal base of 2D polynomials of degree less or equal to n at point (r,s) in [-1,1]**2
def orthopoly2D_rst(x, n):

    N = np.int((n + 1) * (n + 2) / 2)
    p = np.zeros(N)

    r, s = x

    ncount = 0
    for nDeg in np.arange(n + 1):

        for i in np.arange(nDeg + 1):

            if i == 0:
                p_i, q_i= 1, 1
            else:
                p_i = jacobiP_vect(np.array([r]), 0, 0, i)
                q_i = q_i * (1 - s) / 2

            j = nDeg - i
            if j == 0:
                p_j = 1
            else:
                p_j = jacobiP_vect(np.array([s]), 2 * i + 1, 0, j)

            factor = np.sqrt((2 * i + 1) * (i + j + 1) / 2)
            p[ncount] = (p_i * q_i * p_j) * factor

            ncount += 1

    return p


# Function to create orthogonal base of 2D polynomials and derivatives of degree less or equal to n at point (xi,eta) in the reference triangle
def orthopoly2D_deriv_xieta(x, n):

    xi, eta = x

    if eta == 1:
        r, s = -1, 1
        p = orthopoly2D_rst([r, s], n)
        dp_dxi = np.zeros(p.shape)
        dp_deta = np.zeros(p.shape)
    else:
        r = 2 * (1 + xi) / (1 - eta) - 1
        s = eta
        p, dp_dxi, dp_deta = orthopoly2D_deriv_rst([r, s], n)

    return p, dp_dxi, dp_deta


# ---------------------------------------------------------------
# Class Print definition
# ---------------------------------------------------------------
# ###############################################################
# ################     FUNCTIONS DEFINITION     #################
# ###############################################################

def unitary_test():
    """Unitary test for mt1d.py script."""
    # TODO


# ###############################################################
# ################             MAIN             #################
# ###############################################################
if __name__ == '__main__':
    # ---------------------------------------------------------------
    # Run unitary test
    # ---------------------------------------------------------------
    unitary_test()
