/*
 * Filename: fe_nedelec.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-06-22
 *
 * Description:
 * Arbitrary-order H(curl) Nedelec basis on the reference tetrahedron.
 *
 */

#include "fe_nedelec.h"
#include <math.h>

/* Reference tangents of the six Nedelec DOF families (edges and the derived
 * face/interior directions), indexed by dof2tk. */
static const PetscReal feNd_tk[6][3] = {
  { 1., 0., 0.}, { 0., 1., 0.}, { 0., 0., 1.},
  {-1., 1., 0.}, {-1., 0., 1.}, { 0.,-1., 1.}
};
/* Barycenter shift applied to the raw curl-conforming basis factors. */
static const PetscReal feNd_c = 0.25;

struct FeNedelec {
  PetscInt   order;    /* polynomial order p */
  PetscInt   dof;      /* number of H(curl) DOFs */
  PetscReal *Ti;       /* dof x dof, row-major: Ti[n*dof + o] = (T^{-1})_{n,o} */
  PetscReal *nodes;    /* dof x 3, row-major: defining-functional node coords */
  PetscInt  *dof2tk;   /* dof: tangent index in [0,5] of each functional */

  /* Evaluation scratch owned by the handle:
   * the 1D Chebyshev values/derivatives in x,y,z,lambda and the raw space.
   * Evaluation writes through these, so a single handle is NOT reentrant. */
  PetscReal *sx, *sy, *sz, *sl;   /* size order each */
  PetscReal *dx, *dy, *dz, *dl;   /* size order each */
  PetscReal *Uscr;                /* size dof x 3 (raw V or curl)            */
};

/**
 * @brief Evaluates the shifted Chebyshev polynomials T_0..T_p at a point.
 *
 * This function fills u[0..p] with the Chebyshev polynomials of the first
 * kind evaluated on the affine map z = 2x - 1, which sends the reference
 * interval [0, 1] onto the Chebyshev domain [-1, 1]. The values are produced
 * with the standard three-term recurrence T_{n+1} = 2 z T_n - T_{n-1} and
 * supply the 1D factors from which the raw Nedelec space is assembled.
 *
 * @param[in]  p  Highest polynomial degree to evaluate (u holds p+1 reals).
 * @param[in]  x  Reference-interval coordinate in [0, 1].
 * @param[out] u  Array receiving the p+1 polynomial values u[0..p].
 */
static void feNdChebyshev(PetscInt p, PetscReal x, PetscReal *u) {
  u[0] = 1.0;
  if (p == 0) return;
  PetscReal z = 2.0 * x - 1.0;
  u[1] = z;
  for (PetscInt i = 1; i < p; i++) u[i + 1] = 2.0 * z * u[i] - u[i - 1];
}

/**
 * @brief Evaluates the shifted Chebyshev polynomials and their derivatives.
 *
 * This function fills u[0..p] with the Chebyshev values (as in feNdChebyshev)
 * and d[0..p] with their derivatives with respect to the reference-interval
 * coordinate x. Values and derivatives are propagated together with the
 * three-term recurrence and its differentiated form, so a single pass yields
 * the 1D value/derivative factors used to assemble the raw Nedelec curls.
 *
 * @param[in]  p  Highest polynomial degree to evaluate (u, d hold p+1 reals).
 * @param[in]  x  Reference-interval coordinate in [0, 1].
 * @param[out] u  Array receiving the p+1 polynomial values u[0..p].
 * @param[out] d  Array receiving the p+1 polynomial derivatives d[0..p].
 */
static void feNdChebyshevD(PetscInt p, PetscReal x, PetscReal *u, PetscReal *d) {
  u[0] = 1.0; d[0] = 0.0;
  if (p == 0) return;
  PetscReal z = 2.0 * x - 1.0;
  u[1] = z; d[1] = 2.0;
  for (PetscInt i = 1; i < p; i++) {
    u[i + 1] = 2.0 * z * u[i] - u[i - 1];
    d[i + 1] = (i + 1) * (z * d[i] / i + 2.0 * u[i]);
  }
}


/**
 * @brief Computes the n open Gauss-Legendre nodes on the interval (0, 1).
 *
 * This function returns the n roots of the Legendre polynomial P_n, mapped
 * from the canonical interval (-1, 1) to (0, 1) and sorted in ascending
 * order. Each root is located with Newton's iteration, evaluating P_n and its
 * derivative through the Legendre three-term recurrence. These open nodes are
 * the edge, face and interior defining points of the Nedelec functionals.
 *
 * @param[in]  n    Number of quadrature nodes to compute.
 * @param[out] pts  Array receiving the n ascending nodes in (0, 1).
 */
static void feNdGaussLegendre01(PetscInt n, PetscReal *pts) {
  for (PetscInt i = 0; i < n; i++) {
    /* initial guess for the (i+1)-th root on (-1,1), descending */
    PetscReal z = PetscCosReal(PETSC_PI * (i + 0.75) / (n + 0.5));
    PetscReal z1, pp;
    do {
      PetscReal p0 = 1.0, p1 = 0.0;
      for (PetscInt j = 0; j < n; j++) {     /* Legendre recurrence -> P_n(z) */
        PetscReal p2 = p1;
        p1 = p0;
        p0 = ((2.0 * j + 1.0) * z * p1 - j * p2) / (j + 1.0);
      }
      pp = n * (z * p0 - p1) / (z * z - 1.0); /* P_n'(z) */
      z1 = z;
      z = z1 - p0 / pp;                       /* Newton step */
    } while (PetscAbsReal(z - z1) > 1.0e-15);
    /* descending z in (-1,1) -> ascending t in (0,1): reverse the index */
    pts[n - 1 - i] = 0.5 * (1.0 + z);
  }
}


/**
 * @brief Assembles the raw (pre-nodal) H(curl) space values at a point.
 *
 * This function evaluates the un-dualized curl-conforming polynomial space on
 * the reference tetrahedron and stores it as U[dof x 3], row-major. The space
 * is built in three groups, matching the standard Nedelec construction: the
 * gradient-type block (three Cartesian directions per Chebyshev product), the
 * two mixed rotational blocks and the final rotational block. The 1D factors
 * sx, sy, sz, sl are the Chebyshev values in x, y, z and lambda = 1-x-y-z
 * supplied by the caller. The resulting U is later dualized against the DOF
 * functionals to obtain the nodal basis.
 *
 * @param[in]  order  Polynomial order p.
 * @param[in]  x,y,z  Reference-cell coordinates.
 * @param[in]  sx     Chebyshev values in x (order entries).
 * @param[in]  sy     Chebyshev values in y (order entries).
 * @param[in]  sz     Chebyshev values in z (order entries).
 * @param[in]  sl     Chebyshev values in lambda = 1-x-y-z (order entries).
 * @param[out] U      Raw space values, dof x 3 row-major.
 */
static void feNdRawV(PetscInt order, PetscReal x, PetscReal y, PetscReal z, const PetscReal *sx, const PetscReal *sy, 
                     const PetscReal *sz, const PetscReal *sl, PetscReal *U) {

  const PetscInt pm1 = order - 1;
  const PetscReal cx = x - feNd_c, cy = y - feNd_c, cz = z - feNd_c;
  PetscInt n = 0;

  for (PetscInt i = 0; i <= pm1; i++)
    for (PetscInt j = 0; j + i <= pm1; j++)
      for (PetscInt k = 0; k + j + i <= pm1; k++) {
        PetscReal s = sx[k] * sy[j] * sz[i] * sl[pm1 - i - j - k];
        U[n*3+0] = s;  U[n*3+1] = 0.; U[n*3+2] = 0.; n++;
        U[n*3+0] = 0.; U[n*3+1] = s;  U[n*3+2] = 0.; n++;
        U[n*3+0] = 0.; U[n*3+1] = 0.; U[n*3+2] = s;  n++;
      }
  for (PetscInt i = 0; i <= pm1; i++)
    for (PetscInt j = 0; j + i <= pm1; j++) {
      PetscReal s = sx[pm1 - j - i] * sy[j] * sz[i];
      U[n*3+0] = s * cy;  U[n*3+1] = -s * cx; U[n*3+2] = 0.;      n++;
      U[n*3+0] = s * cz;  U[n*3+1] = 0.;      U[n*3+2] = -s * cx; n++;
    }
  for (PetscInt i = 0; i <= pm1; i++) {
    PetscReal s = sy[pm1 - i] * sz[i];
    U[n*3+0] = 0.; U[n*3+1] = s * cz; U[n*3+2] = -s * cy; n++;
  }
}


/**
 * @brief Assembles the raw (pre-nodal) H(curl) curls at a point.
 *
 * This function evaluates the curls of the raw curl-conforming polynomial
 * space produced by feNdRawV and stores them as Uc[dof x 3], row-major. It
 * mirrors feNdRawV group by group, differentiating the Chebyshev products via
 * the value/derivative factors (sx..sl and dx..dl) supplied by the caller.
 * The lambda derivative already carries the -1 from lambda = 1-x-y-z through
 * the raw-curl expressions, so dl is passed as the plain polynomial derivative.
 *
 * @param[in]  order        Polynomial order p.
 * @param[in]  x,y,z        Reference-cell coordinates.
 * @param[in]  sx,sy,sz,sl  Chebyshev values in x, y, z, lambda (order entries each).
 * @param[in]  dx,dy,dz,dl  Chebyshev derivatives in x, y, z, lambda (order entries each).
 * @param[out] Uc           Raw curl values, dof x 3 row-major.
 */
static void feNdRawCurl(PetscInt order, PetscReal x, PetscReal y, PetscReal z, const PetscReal *sx, const PetscReal *sy,
                        const PetscReal *sz, const PetscReal *sl, const PetscReal *dx, const PetscReal *dy,
                        const PetscReal *dz, const PetscReal *dl, PetscReal *Uc) {

  const PetscInt pm1 = order - 1;
  const PetscReal cx = x - feNd_c, cy = y - feNd_c, cz = z - feNd_c;
  PetscInt n = 0;
  (void)z;

  for (PetscInt i = 0; i <= pm1; i++)
    for (PetscInt j = 0; j + i <= pm1; j++)
      for (PetscInt k = 0; k + j + i <= pm1; k++) {
        PetscInt l = pm1 - i - j - k;
        PetscReal gx = (dx[k] * sl[l] - sx[k] * dl[l]) * sy[j] * sz[i];
        PetscReal gy = (dy[j] * sl[l] - sy[j] * dl[l]) * sx[k] * sz[i];
        PetscReal gz = (dz[i] * sl[l] - sz[i] * dl[l]) * sx[k] * sy[j];
        Uc[n*3+0] =  0.; Uc[n*3+1] =  gz; Uc[n*3+2] = -gy; n++;
        Uc[n*3+0] = -gz; Uc[n*3+1] =  0.; Uc[n*3+2] =  gx; n++;
        Uc[n*3+0] =  gy; Uc[n*3+1] = -gx; Uc[n*3+2] =  0.; n++;
      }
  for (PetscInt i = 0; i <= pm1; i++)
    for (PetscInt j = 0; j + i <= pm1; j++) {
      PetscInt k = pm1 - j - i;
      /* curl of  sx*sy*sz * (cy, -cx, 0) */
      Uc[n*3+0] = sx[k] * cx * sy[j] * dz[i];
      Uc[n*3+1] = sx[k] * sy[j] * cy * dz[i];
      Uc[n*3+2] = -((dx[k] * cx + sx[k]) * sy[j] * sz[i] +
                    (dy[j] * cy + sy[j]) * sx[k] * sz[i]);
      n++;
      /* curl of  sx*sy*sz * (cz, 0, -cx) */
      Uc[n*3+0] = -sx[k] * cx * dy[j] * sz[i];
      Uc[n*3+1] = sx[k] * sy[j] * (dz[i] * cz + sz[i]) +
                  (dx[k] * cx + sx[k]) * sy[j] * sz[i];
      Uc[n*3+2] = -sx[k] * dy[j] * sz[i] * cz;
      n++;
    }
  for (PetscInt i = 0; i <= pm1; i++) {
    PetscInt j = pm1 - i;
    /* curl of  sy*sz * (0, cz, -cy) */
    Uc[n*3+0] = -((dy[j] * cy + sy[j]) * sz[i] + sy[j] * (dz[i] * cz + sz[i]));
    Uc[n*3+1] = 0.;
    Uc[n*3+2] = 0.;
    n++;
  }
}


/**
 * @brief Builds the Nedelec DOF nodes and their tangent indices.
 *
 * This function fills the defining-functional node coordinates (nodes, dof x 3
 * row-major) and the per-DOF tangent index (dof2tk, into feNd_tk) in the
 * canonical MFEM order: the six edges first (order points each), then the four
 * faces (two tangent DOFs per face node), then the interior (three tangent
 * DOFs per node). Edge, face and interior nodes are placed from the open
 * Gauss-Legendre point sets eop, fop and iop; face and interior nodes are
 * barycentrically normalized onto their entity.
 *
 * @param[in]  p       Polynomial order.
 * @param[in]  eop     Open Gauss-Legendre points for edges (p entries).
 * @param[in]  fop     Open Gauss-Legendre points for faces (p-1 entries).
 * @param[in]  iop     Open Gauss-Legendre points for the interior (p-2 entries).
 * @param[out] nodes   DOF node coordinates, dof x 3 row-major.
 * @param[out] dof2tk  Per-DOF tangent index in [0, 5].
 */
static void feNdBuildNodes(PetscInt p, const PetscReal *eop, const PetscReal *fop, const PetscReal *iop, PetscReal *nodes, PetscInt *dof2tk) {
  
  const PetscInt pm1 = p - 1, pm2 = p - 2, pm3 = p - 3;
  
  PetscInt o = 0;
  
  #define FE_ND_SET(px,py,pz,tk) do { nodes[o*3+0]=(px); nodes[o*3+1]=(py); \
                                      nodes[o*3+2]=(pz); dof2tk[o]=(tk); o++; } while (0)

  /* edges: (0,1) (0,2) (0,3) (1,2) (1,3) (2,3) */
  for (PetscInt i = 0; i < p; i++) FE_ND_SET(eop[i], 0., 0., 0);
  for (PetscInt i = 0; i < p; i++) FE_ND_SET(0., eop[i], 0., 1);
  for (PetscInt i = 0; i < p; i++) FE_ND_SET(0., 0., eop[i], 2);
  for (PetscInt i = 0; i < p; i++) FE_ND_SET(eop[pm1-i], eop[i], 0., 3);
  for (PetscInt i = 0; i < p; i++) FE_ND_SET(eop[pm1-i], 0., eop[i], 4);
  for (PetscInt i = 0; i < p; i++) FE_ND_SET(0., eop[pm1-i], eop[i], 5);

  /* faces: (1,2,3) (0,3,2) (0,1,3) (0,2,1), two tangents per node */
  for (PetscInt i = 0; i <= pm2; i++)
    for (PetscInt j = 0; j + i <= pm2; j++) {
      PetscReal w = fop[i] + fop[j] + fop[pm2-i-j];
      FE_ND_SET(fop[pm2-i-j]/w, fop[j]/w, fop[i]/w, 3);
      FE_ND_SET(fop[pm2-i-j]/w, fop[j]/w, fop[i]/w, 4);
    }
  for (PetscInt i = 0; i <= pm2; i++)
    for (PetscInt j = 0; j + i <= pm2; j++) {
      PetscReal w = fop[i] + fop[j] + fop[pm2-i-j];
      FE_ND_SET(0., fop[i]/w, fop[j]/w, 2);
      FE_ND_SET(0., fop[i]/w, fop[j]/w, 1);
    }
  for (PetscInt i = 0; i <= pm2; i++)
    for (PetscInt j = 0; j + i <= pm2; j++) {
      PetscReal w = fop[i] + fop[j] + fop[pm2-i-j];
      FE_ND_SET(fop[j]/w, 0., fop[i]/w, 0);
      FE_ND_SET(fop[j]/w, 0., fop[i]/w, 2);
    }
  for (PetscInt i = 0; i <= pm2; i++)
    for (PetscInt j = 0; j + i <= pm2; j++) {
      PetscReal w = fop[i] + fop[j] + fop[pm2-i-j];
      FE_ND_SET(fop[i]/w, fop[j]/w, 0., 1);
      FE_ND_SET(fop[i]/w, fop[j]/w, 0., 0);
    }

  /* interior: three tangents per node */
  for (PetscInt i = 0; i <= pm3; i++)
    for (PetscInt j = 0; j + i <= pm3; j++)
      for (PetscInt k = 0; k + j + i <= pm3; k++) {
        PetscReal w = iop[k] + iop[j] + iop[i] + iop[pm3-i-j-k];
        FE_ND_SET(iop[k]/w, iop[j]/w, iop[i]/w, 0);
        FE_ND_SET(iop[k]/w, iop[j]/w, iop[i]/w, 1);
        FE_ND_SET(iop[k]/w, iop[j]/w, iop[i]/w, 2);
      }
  #undef FE_ND_SET
}

/**
 * @brief Inverts a dense N x N matrix by Gauss-Jordan elimination.
 *
 * This function computes Ainv = A^{-1} for a row-major N x N matrix using
 * Gauss-Jordan elimination with partial pivoting. A is left unchanged (it is
 * copied into an internal working buffer); Ainv is initialized to the identity
 * and transformed alongside the working copy. A singular column (near-zero
 * pivot) raises a PETSc error. It is used to dualize the raw Nedelec space
 * against the DOF functionals (Vandermonde inverse).
 *
 * @param[in]  N     Matrix dimension.
 * @param[in]  A     Input matrix, N x N row-major.
 * @param[out] Ainv  Output inverse, N x N row-major.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code
 *         (PETSC_ERR_MAT_LU_ZRPVT) if the matrix is singular.
 */
static PetscErrorCode feNdInvert(PetscInt N, const PetscReal *A, PetscReal *Ainv) {
  
  PetscFunctionBeginUser;
  PetscReal *M;
  
  PetscCall(PetscMalloc1(N * N, &M));
  
  for (PetscInt i = 0; i < N * N; i++) {
    M[i] = A[i];
  }
  
  for (PetscInt i = 0; i < N; i++) {
    for (PetscInt j = 0; j < N; j++) {
      Ainv[i*N+j] = (i == j) ? 1.0 : 0.0;
    }
  }

  for (PetscInt i = 0; i < N; i++) {
    PetscInt piv = i;
    PetscReal best = PetscAbsReal(M[i*N+i]);
    for (PetscInt j = i + 1; j < N; j++) {
      PetscReal v = PetscAbsReal(M[j*N+i]);
      if (v > best) { 
        best = v; piv = j; 
      }
    }
    if (best < 1.0e-300) {
      PetscCall(PetscFree(M));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "fe_nedelec: singular Vandermonde matrix at column %" PetscInt_FMT, i);
    }

    if (piv != i)
      for (PetscInt j = 0; j < N; j++) {
        PetscReal t;
        t           = M[i*N+j];    
        M[i*N+j]    = M[piv*N+j];    
        M[piv*N+j]  = t;
        t           = Ainv[i*N+j]; 
        Ainv[i*N+j] = Ainv[piv*N+j]; 
        Ainv[piv*N+j] = t;
      }

    PetscReal d = M[i*N+i];
    for (PetscInt j = 0; j < N; j++) { 
      M[i*N+j] /= d; Ainv[i*N+j] /= d; 
    }
    
    for (PetscInt j = 0; j < N; j++) {
      if (j == i) {
        continue;
      }
      
      PetscReal f = M[j*N+i];
      
      if (f == 0.0) {
        continue;
      }
      
      for (PetscInt k = 0; k < N; k++) {
        M[j*N+k]    -= f * M[i*N+k];
        Ainv[j*N+k] -= f * Ainv[i*N+k];
      }
    }
  }

  PetscCall(PetscFree(M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Number of H(curl) DOFs for a Nedelec element of the given order.
 *
 * This function returns the dimension of the arbitrary-order Nedelec space on
 * a tetrahedron, order*(order+2)*(order+3)/2, or 0 for a non-positive order.
 *
 * @param[in] order  Polynomial order p (>= 1).
 *
 * @return The number of H(curl) DOFs, or 0 when order < 1.
 */
PetscInt feNedelecDofCount(PetscInt order) {
  
  if (order < 1) {
    return 0;
  }
  
  return order * (order + 2) * (order + 3) / 2;
}

/**
 * @brief Builds a Nedelec element of the given order.
 *
 * This function allocates an opaque Nedelec handle, computes the open
 * Gauss-Legendre defining points for edges, faces and the interior, builds the
 * DOF nodes/tangents (feNdBuildNodes), and forms the Vandermonde matrix
 * T(o,m) = U_o(node_m) . tk[m] of the raw space against the DOF functionals,
 * storing its inverse Ti = T^{-1} for later value/curl evaluation. The caller
 * owns the returned handle and must release it with feNedelecDestroy.
 *
 * @param[in]  order  Polynomial order p (>= 1).
 * @param[out] out    Newly created handle.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode feNedelecCreate(PetscInt order, FeNedelec **out) {
  
  PetscFunctionBeginUser;
  PetscCheck(order >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "fe_nedelec: order must be >= 1 (got %" PetscInt_FMT ")", order);
  
  *out = NULL;

  FeNedelec *fe;
  PetscCall(PetscNew(&fe));
  fe->order = order;
  fe->dof   = feNedelecDofCount(order);
  
  const PetscInt dof = fe->dof, p = order;
  PetscCall(PetscMalloc1(dof * dof, &fe->Ti));
  PetscCall(PetscMalloc1(dof * 3, &fe->nodes));
  PetscCall(PetscMalloc1(dof, &fe->dof2tk));
  PetscCall(PetscMalloc4(p, &fe->sx, p, &fe->sy, p, &fe->sz, p, &fe->sl));
  PetscCall(PetscMalloc4(p, &fe->dx, p, &fe->dy, p, &fe->dz, p, &fe->dl));
  PetscCall(PetscMalloc1(dof * 3, &fe->Uscr));

  /* Gauss-Legendre open points: p on edges, p-1 on faces, p-2 in interior. */
  PetscReal *eop = NULL, *fop = NULL, *iop = NULL;
  PetscCall(PetscMalloc1(p, &eop));
  feNdGaussLegendre01(p, eop);
  
  if (p > 1) { 
    PetscCall(PetscMalloc1(p - 1, &fop)); 
    feNdGaussLegendre01(p - 1, fop); 
  }
  
  if (p > 2) { 
    PetscCall(PetscMalloc1(p - 2, &iop)); 
    feNdGaussLegendre01(p - 2, iop); 
  }

  feNdBuildNodes(p, eop, fop, iop, fe->nodes, fe->dof2tk);

  /* Vandermonde T(o,m) = U_o(node_m) . tk[dof2tk[m]], then Ti = T^{-1}. */
  PetscReal *T = NULL;
  PetscCall(PetscMalloc1(dof * dof, &T));
  
  for (PetscInt i = 0; i < dof; i++) {
    const PetscReal X = fe->nodes[i*3+0], Y = fe->nodes[i*3+1], Z = fe->nodes[i*3+2];
    const PetscReal *tm = feNd_tk[fe->dof2tk[i]];
  
    feNdChebyshev(p - 1, X, fe->sx);
    feNdChebyshev(p - 1, Y, fe->sy);
    feNdChebyshev(p - 1, Z, fe->sz);
    feNdChebyshev(p - 1, 1.0 - X - Y - Z, fe->sl);
    feNdRawV(p, X, Y, Z, fe->sx, fe->sy, fe->sz, fe->sl, fe->Uscr);
  
    for (PetscInt j = 0; j < dof; j++) {
      T[j*dof+i] = fe->Uscr[j*3+0]*tm[0] + fe->Uscr[j*3+1]*tm[1] + fe->Uscr[j*3+2]*tm[2];
    }
  }
  
  PetscCall(feNdInvert(dof, T, fe->Ti));

  PetscCall(PetscFree(T));
  PetscCall(PetscFree(eop));
  PetscCall(PetscFree(fop));
  PetscCall(PetscFree(iop));

  *out = fe;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Destroys a Nedelec element handle (no-op on NULL).
 *
 * This function frees every buffer owned by the handle (Vandermonde inverse,
 * nodes, tangent indices, evaluation scratch) and the handle itself, then sets
 * the caller's pointer to NULL. Passing a NULL or already-freed handle is safe.
 *
 * @param[in,out] fe  Handle to free; set to NULL on return.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode feNedelecDestroy(FeNedelec **fe) {
  
  PetscFunctionBeginUser;
  
  if (!fe || !*fe) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  
  PetscCall(PetscFree((*fe)->Ti));
  PetscCall(PetscFree((*fe)->nodes));
  PetscCall(PetscFree((*fe)->dof2tk));
  PetscCall(PetscFree4((*fe)->sx, (*fe)->sy, (*fe)->sz, (*fe)->sl));
  PetscCall(PetscFree4((*fe)->dx, (*fe)->dy, (*fe)->dz, (*fe)->dl));
  PetscCall(PetscFree((*fe)->Uscr));
  PetscCall(PetscFree(*fe));

  *fe = NULL;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Returns the number of H(curl) DOFs of a handle.
 *
 * @param[in] fe  Element handle (may be NULL).
 *
 * @return The DOF count, or 0 when fe is NULL.
 */
PetscInt feNedelecGetDof(const FeNedelec *fe)   { 
  return fe ? fe->dof : 0; 
}

/**
 * @brief Returns the polynomial order of a handle.
 *
 * @param[in] fe  Element handle (may be NULL).
 *
 * @return The polynomial order, or 0 when fe is NULL.
 */
PetscInt feNedelecGetOrder(const FeNedelec *fe) { 
  return fe ? fe->order : 0; 
}

/**
 * @brief Evaluates the reference Nedelec values at a reference-cell point.
 *
 * This function evaluates the raw curl-conforming space at (x, y, z) and
 * applies the stored Vandermonde inverse Ti to obtain the nodal basis values,
 * writing them into shape as dof x 3 row-major. The handle's 1D scratch is
 * overwritten, so a single handle is not reentrant.
 *
 * @param[in]  fe     Element handle.
 * @param[in]  x,y,z  Reference-cell coordinates.
 * @param[out] shape  dof x 3 row-major basis values.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode feNedelecCalcVShape(const FeNedelec *fe, PetscReal x, PetscReal y, PetscReal z, PetscReal *shape) {
  
  PetscFunctionBeginUser;

  const PetscInt dof = fe->dof, p = fe->order;

  feNdChebyshev(p - 1, x, fe->sx);
  feNdChebyshev(p - 1, y, fe->sy);
  feNdChebyshev(p - 1, z, fe->sz);
  feNdChebyshev(p - 1, 1.0 - x - y - z, fe->sl);
  feNdRawV(p, x, y, z, fe->sx, fe->sy, fe->sz, fe->sl, fe->Uscr);

  for (PetscInt i = 0; i < dof; i++) {
    PetscReal a0 = 0., a1 = 0., a2 = 0.;
    
    const PetscReal *ti = &fe->Ti[i*dof];
    
    for (PetscInt j = 0; j < dof; j++) {
      a0 += ti[j] * fe->Uscr[j*3+0];
      a1 += ti[j] * fe->Uscr[j*3+1];
      a2 += ti[j] * fe->Uscr[j*3+2];
    }
    
    shape[i*3+0] = a0; shape[i*3+1] = a1; shape[i*3+2] = a2;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Evaluates the reference Nedelec curls at a reference-cell point.
 *
 * This function evaluates the raw curls at (x, y, z) and applies the stored
 * Vandermonde inverse Ti to obtain the nodal basis curls, writing them into
 * curl as dof x 3 row-major. The lambda-derivative sign from
 * lambda = 1-x-y-z is already carried inside the raw-curl expressions, so the
 * plain Chebyshev derivative is passed as the lambda factor. The handle's 1D
 * scratch is overwritten, so a single handle is not reentrant.
 *
 * @param[in]  fe     Element handle.
 * @param[in]  x,y,z  Reference-cell coordinates.
 * @param[out] curl   dof x 3 row-major curl values.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode feNedelecCalcCurlShape(const FeNedelec *fe, PetscReal x, PetscReal y, PetscReal z, PetscReal *curl) {
  
  PetscFunctionBeginUser;
  const PetscInt dof = fe->dof, p = fe->order;

  feNdChebyshevD(p - 1, x, fe->sx, fe->dx);
  feNdChebyshevD(p - 1, y, fe->sy, fe->dy);
  feNdChebyshevD(p - 1, z, fe->sz, fe->dz);
  feNdChebyshevD(p - 1, 1.0 - x - y - z, fe->sl, fe->dl);
  
  /* d/d(lambda) carries a -1 from lambda = 1 - x - y - z in the chain rule; the
   * raw-curl formulas already encode that sign, so dl is the plain T' here. */
  feNdRawCurl(p, x, y, z, fe->sx, fe->sy, fe->sz, fe->sl, fe->dx, fe->dy, fe->dz, fe->dl, fe->Uscr);

  for (PetscInt i = 0; i < dof; i++) {
    PetscReal a0 = 0., a1 = 0., a2 = 0.;
    const PetscReal *ti = &fe->Ti[i*dof];
    for (PetscInt j = 0; j < dof; j++) {
      a0 += ti[j] * fe->Uscr[j*3+0];
      a1 += ti[j] * fe->Uscr[j*3+1];
      a2 += ti[j] * fe->Uscr[j*3+2];
    }
    curl[i*3+0] = a0; curl[i*3+1] = a1; curl[i*3+2] = a2;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Returns the defining-functional node and tangent of a DOF.
 *
 * This function reports the reference-cell node coordinates and the tangent
 * direction (from feNd_tk, via the DOF's tangent index) of functional m.
 * Either output may be NULL to skip it. The DOF index is bounds-checked.
 *
 * @param[in]  fe       Element handle.
 * @param[in]  m        DOF index in [0, dof).
 * @param[out] node     Reference-cell node coordinates (3); may be NULL.
 * @param[out] tangent  DOF tangent direction (3); may be NULL.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode feNedelecGetDofInfo(const FeNedelec *fe, PetscInt m, PetscReal node[3], PetscReal tangent[3]) {
  PetscFunctionBeginUser;
  
  PetscCheck(m >= 0 && m < fe->dof, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "fe_nedelec: DOF index %" PetscInt_FMT " out of range [0,%" PetscInt_FMT ")", m, fe->dof);
  
  if (node) { 
    node[0] = fe->nodes[m*3+0]; 
    node[1] = fe->nodes[m*3+1]; 
    node[2] = fe->nodes[m*3+2]; 
  }
  
  if (tangent) {
    const PetscReal *tm = feNd_tk[fe->dof2tk[m]];
    tangent[0] = tm[0]; 
    tangent[1] = tm[1];
    tangent[2] = tm[2];
  }
  
  PetscFunctionReturn(PETSC_SUCCESS);
}
