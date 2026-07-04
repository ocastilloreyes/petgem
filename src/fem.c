/*
 * Filename: fem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-06-22
 *
 * Description:
 * Finite element computations for PETGEM (fe_nedelec.c for H(curl), fe_nodal.c for H1), for
 * orders order = 1..6.
 *
 * This file provides:
 *   - generic geometry / quadrature helpers (independent of the FE family);
 *   - the reference Nedelec element mass / stiffness matrices;
 *   - point evaluation of the physical Nedelec values and curls;
 *   - the per-cell discrete gradient block used by PCBDDC;
 *   - thin struct-based adapters (Cell / FEMSpace / Quadrature3D) that the
 *     DMPlex-centric assembly, postprocessing and receiver code call.
 *
 * Orientation is geometric. The reference tetrahedron edge that occupies DMPlex
 * closure slot E joins the local cell vertices femEdgeV[femEdgeSlot[E]].
 * Its sign is +1 when those vertices are already in ascending lexicographic
 * coordinate order and -1 otherwise. Because the two cells sharing an edge see
 * the same physical vertices, they choose the same sign, so the global Nedelec
 * and H1 spaces are conforming and the discrete gradient is exact (K_e G_e = 0).
 * There are therefore no external per-DOF sign multipliers.
 */

/* PETSc libraries */
#include <petsc.h>
#include <petscsys.h>
#include <petscdt.h>

/* PETGEM functions */
#include "constants.h"
#include "fem.h"
#include "fe_nedelec.h"
#include "fe_nodal.h"
#include "grid.h"

/* PETGEM closure edge slot E -> native edge; native edge -> local vertices. */
static const PetscInt femEdgeSlot[NUM_EDGES_PER_CELL] = {0, 3, 1, 2, 4, 5};
static const PetscInt femEdgeV[NUM_EDGES_PER_CELL][2] = { {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3} };

/* PETGEM closure face slot F -> native face; native face -> local vertices. */
static const PetscInt femFaceSlot[NUM_FACES_PER_CELL] = {3, 2, 1, 0};
static const PetscInt femFaceV[NUM_FACES_PER_CELL][3] =  { {1,2,3}, {0,3,2}, {0,1,3}, {0,2,1} };

/* The six triangle face transforms applied to primal basis values (one per
   triangle orientation), used to mix the per-node tangent DOF pairs on a face. */
static const PetscReal femFaceTinv[6][2][2] = {
  {{ 1, 0},{ 0, 1}}, {{-1,-1},{ 0, 1}}, {{-1,-1},{ 1, 0}},
  {{ 1, 0},{-1,-1}}, {{ 0, 1},{-1,-1}}, {{ 0, 1},{ 1, 0}}
};

/* One lazily-built reference handle for the active order of the whole run. */
static FeNedelec *femNd = NULL;
static PetscInt   femNdOrder = 0;

/**
 * @brief Builds the reference-consistent cell Jacobian and its inverse.
 *
 * This function computes the element Jacobian used by the FE routines. Its rows
 * are (v1-v0), (v2-v0), (v3-v0), i.e. J = F^T where F = [v1-v0|v2-v0|v3-v0] is
 * the Jacobian of the reference-tet (v0 at the origin) -> physical map on
 * which fe_nedelec.c defines its basis. Under this convention femPiola's value
 * (invJ * v) and curl (J^T * c / det) pullbacks are exactly the covariant F^{-T}
 * and contravariant F/det maps. The inverse is formed from the adjugate.
 *
 * @param[in]  coords       Cell vertex coordinates (4x3, row-major).
 * @param[out] jacobian     Cell Jacobian J = F^T.
 * @param[out] invJacobian  Inverse Jacobian J^{-1}.
 */
static PetscErrorCode femComputeJacobian(const PetscReal *coords, PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS]) {
    PetscFunctionBeginUser;

    PetscReal determinant, invDeterminant;
    PetscReal coFactorMatrix[NUM_DIMENSIONS][NUM_DIMENSIONS], adjugateMatrix[NUM_DIMENSIONS][NUM_DIMENSIONS];

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        jacobian[0][i] = coords[3+i] - coords[i];
        jacobian[1][i] = coords[6+i] - coords[i];
        jacobian[2][i] = coords[9+i] - coords[i];
    }

    determinant = jacobian[0][0] * (jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1]) -
                  jacobian[0][1] * (jacobian[1][0] * jacobian[2][2] - jacobian[1][2] * jacobian[2][0]) +
                  jacobian[0][2] * (jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0]);

    coFactorMatrix[0][0] =   jacobian[1][1] * jacobian[2][2] - jacobian[1][2] * jacobian[2][1];
    coFactorMatrix[0][1] = -(jacobian[1][0] * jacobian[2][2] - jacobian[1][2] * jacobian[2][0]);
    coFactorMatrix[0][2] =   jacobian[1][0] * jacobian[2][1] - jacobian[1][1] * jacobian[2][0];
    coFactorMatrix[1][0] = -(jacobian[0][1] * jacobian[2][2] - jacobian[0][2] * jacobian[2][1]);
    coFactorMatrix[1][1] =   jacobian[0][0] * jacobian[2][2] - jacobian[0][2] * jacobian[2][0];
    coFactorMatrix[1][2] = -(jacobian[0][0] * jacobian[2][1] - jacobian[0][1] * jacobian[2][0]);
    coFactorMatrix[2][0] =   jacobian[0][1] * jacobian[1][2] - jacobian[0][2] * jacobian[1][1];
    coFactorMatrix[2][1] = -(jacobian[0][0] * jacobian[1][2] - jacobian[0][2] * jacobian[1][0]);
    coFactorMatrix[2][2] =   jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
            adjugateMatrix[i][j] = coFactorMatrix[j][i];
        }
    }

    invDeterminant = 1.0 / determinant;
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
            invJacobian[i][j] = invDeterminant * adjugateMatrix[i][j];
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Builds a Stroud conical quadrature on the unit reference tetrahedron.
 *
 * This function generates a Stroud conical rule of m points per axis on the
 * biunit reference simplex [-1,1]^3 and maps it affinely to the unit tetrahedron
 * used by the element bases (x_unit = (x_biunit+1)/2, weights scaled by
 * 1/2^NUM_DIMENSIONS), giving (order+1)^3 points. It checks that the mapped
 * weights sum to the unit-tet volume 1/6. The points/weights arrays are
 * caller-owned.
 *
 * @param[in]  m        Number of points per axis (order+1).
 * @param[out] points   Quadrature point coordinates (m^3 x 3).
 * @param[out] weights  Quadrature weights (m^3).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
static PetscErrorCode femGaussPoints3D(PetscInt m, PetscReal **points, PetscReal *weights){
    PetscFunctionBeginUser;

    PetscQuadrature    quad;
    const PetscReal   *qpts, *qwts;
    PetscInt           qdim, qNc, qn;
    PetscReal          wsum = 0.0;

    /* Generate a Stroud conical rule on the biunit reference simplex [-1,1]^3 and
       map it affinely to the unit tetrahedron used by the element bases:
       x_unit = (x_biunit + 1)/2, w_unit = w_biunit / 2^NUM_DIMENSIONS. The mapped
       weights then sum to the unit-tet volume 1/6. */
    PetscCall(PetscDTStroudConicalQuadrature(NUM_DIMENSIONS, 1, m, -1.0, 1.0, &quad));
    PetscCall(PetscQuadratureGetData(quad, &qdim, &qNc, &qn, &qpts, &qwts));

    PetscCheck(qdim == NUM_DIMENSIONS && qn == m*m*m, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "femGaussPoints3D: unexpected quadrature (dim %" PetscInt_FMT ", %" PetscInt_FMT " points)", qdim, qn);

    for (PetscInt i = 0; i < qn; i++) {
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
            points[i][j] = 0.5 * (qpts[i*NUM_DIMENSIONS + j] + 1.0);
        }
        weights[i] = qwts[i] / 8.0;
        wsum += weights[i];
    }
    PetscCall(PetscQuadratureDestroy(&quad));

    /* Guard the biunit->unit convention: the rule must integrate 1 over the unit
       tetrahedron, i.e. the mapped weights must sum to its volume 1/6. */
    PetscCheck(PetscAbsReal(wsum - 1.0/6.0) < 1.0e-10, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "femGaussPoints3D: weights sum to %g, expected 1/6 (quadrature convention mismatch)", (double)wsum);

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Releases the process-wide cached Nedelec reference handle.
 *
 * This function destroys the lazily-built reference element and resets the
 * cached order. It is registered with PetscRegisterFinalize so the handle is
 * freed at PETSc shutdown.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
static PetscErrorCode femFinalize(void) {
  PetscFunctionBeginUser;

  if (femNd) {
    PetscCall(feNedelecDestroy(&femNd));
  }
  
  femNdOrder = 0;
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Ensures the cached Nedelec reference handle matches the given order.
 *
 * This function lazily (re)builds the process-wide Nedelec reference element
 * when the requested order differs from the cached one, registering the
 * finalizer on first use. Only orders order = 1..6 are supported.
 *
 * @param[in] order  Requested Nedelec order (1..6).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
static PetscErrorCode femEnsureNedelec(PetscInt order) {
  PetscFunctionBeginUser;
  PetscCheck(order >= 1 && order <= 6, PETSC_COMM_SELF, PETSC_ERR_SUP, "fem: only order = 1..6 are supported (got %" PetscInt_FMT ")", order);
  
  if (femNdOrder != order) {
    static PetscBool registered = PETSC_FALSE;
    if (!registered) { 
        PetscCall(PetscRegisterFinalize(femFinalize)); registered = PETSC_TRUE; 
    }
    
    if (femNd) { 
        PetscCall(feNedelecDestroy(&femNd)); 
    }
    
    PetscCall(feNedelecCreate(order, &femNd));
    
    femNdOrder = order;
  }
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Lexicographically compares two local cell vertices by coordinate.
 *
 * This function orders vertices vi and vj by their (x, y, z) coordinates, the
 * comparison that drives geometric edge/face orientation.
 *
 * @param[in] vc  Cell vertex coordinates (4x3, row-major).
 * @param[in] vi  First local vertex index.
 * @param[in] vj  Second local vertex index.
 *
 * @return PETSC_TRUE when vertex vi precedes vertex vj, else PETSC_FALSE.
 */
static inline PetscBool femCoordLess(const PetscReal *vc, PetscInt vi, PetscInt vj) {
  const PetscReal *a = &vc[vi*NUM_DIMENSIONS];
  const PetscReal *b = &vc[vj*NUM_DIMENSIONS];
  
  if (a[0] != b[0]) { 
    return (PetscBool)(a[0] < b[0]);
  }
  
  if (a[1] != b[1]) {
    return (PetscBool)(a[1] < b[1]);
  }
  
  return (PetscBool)(a[2] < b[2]);
}

/**
 * @brief Returns the triangle orientation taking a test order into a base.
 *
 * This function reports the triangle-orientation index (0..5) that maps the test
 * vertex ordering onto the base ordering, matching PETGEM convention for mixing
 * the per-node tangent DOF pairs on a face.
 *
 * @param[in] base  Base (canonical) triangle vertex ordering.
 * @param[in] test  Test triangle vertex ordering.
 *
 * @return The triangle orientation index in [0, 5].
 */
static inline PetscInt femTriOri(const PetscInt base[3], const PetscInt test[3]) {
  
  if (test[0] == base[0]) {
    return (test[1] == base[1]) ? 0 : 5;
  }
  
  if (test[0] == base[1]) {
    return (test[1] == base[0]) ? 1 : 2;
  }
  
  return (test[1] == base[0]) ? 4 : 3;
}

/**
 * @brief Maps a triangular-grid node (i, j) to its index within a face loop.
 *
 * This function returns the linear position of node (i, j) in a face's DOF node
 * loop, matching the enumeration used on the Nedelec and H1 sides.
 *
 * @param[in] m  Triangular-grid size parameter.
 * @param[in] i  First grid coordinate.
 * @param[in] j  Second grid coordinate.
 *
 * @return The linear index of node (i, j) within the face node loop.
 */
static inline PetscInt femTriLoopIndex(PetscInt m, PetscInt i, PetscInt j) {
  return j * (m + 1) - j * (j - 1) / 2 + i;
}

/**
 * @brief Maps a reference field to closure-slot order with geometric orientation.
 *
 * This function maps the PETGEM-native-ordered reference field `in` (dof x 3) to
 * the closure-slot ordered, oriented field `out` (dof x 3) for order p. It is
 * linear in `in`, so it serves both values and curls. Orientation is geometric:
 * every shared entity is oriented by the lexicographic order of its vertex
 * coordinates (identical for the two cells sharing it), so the global spaces are
 * conforming. Edges reverse and negate their DOFs when reversed; faces permute
 * their node grid and mix each tangent DOF pair by the triangle transform T;
 * interior DOFs are never shared and map by identity.
 *
 * The closure slot layout (interior, then faces (4 x p(p-1)), then edges
 * (6 x p)) matches the DMPlex closure; the PETGEM native layout is edges, then
 * faces, then interior.
 *
 * @param[in]  p    Polynomial order.
 * @param[in]  vc   Cell vertex coordinates (4x3, row-major) driving orientation.
 * @param[in]  in   PETGEM-native-ordered reference field (dof x 3, row-major).
 * @param[out] out  Closure-slot ordered, oriented field (dof x 3, row-major).
 */
static void femOrient(PetscInt p, const PetscReal *vc, const PetscReal *in, PetscReal *out) {
  const PetscInt dpe = p, dpf = p*(p-1), dpv = p*(p-1)*(p-2)/2;
  const PetscInt faceOff = dpv, edgeOff = dpv + NUM_FACES_PER_CELL*dpf;
  const PetscInt mEdge = 0, mFace = NUM_EDGES_PER_CELL*p, mInt = NUM_EDGES_PER_CELL*p + NUM_FACES_PER_CELL*dpf;
  const PetscInt pm2 = p - 2;

  /* edges: p DOFs; a reversed edge reverses their order and negates them. */
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; i++) {
    const PetscInt me = femEdgeSlot[i];
    const PetscBool rev = femCoordLess(vc, femEdgeV[me][1], femEdgeV[me][0]);
    const PetscInt mb = mEdge + me*p, sb = edgeOff + i*dpe;
    for (PetscInt j = 0; j < p; j++) {
      const PetscInt src = rev ? (mb + (p-1-j)) : (mb + j);
      const PetscReal s = rev ? -1.0 : 1.0;
      for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
        out[(sb+j)*3+k] = s * in[src*3+k];
      }
    }
  }

  /* faces: p(p-1)/2 nodes x 2 tangent DOFs; permute nodes + mix pairs by T(Fo). */
  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; i++) {
    const PetscInt fm = femFaceSlot[i];
    const PetscInt *fv = femFaceV[fm];
    PetscInt ord[3] = {0, 1, 2};                  /* sort face-vertex positions by coord */
    for (PetscInt j = 1; j < 3; j++) {
      for (PetscInt k = j; k > 0 && femCoordLess(vc, fv[ord[k]], fv[ord[k-1]]); k--) {
        PetscInt t = ord[k]; ord[k] = ord[k-1]; ord[k-1] = t;
      }
    }
      
    const PetscInt base[3] = {fv[ord[0]], fv[ord[1]], fv[ord[2]]};
    const PetscInt test[3] = {fv[0], fv[1], fv[2]};
    const PetscReal (*T)[2] = femFaceTinv[femTriOri(base, test)];
    const PetscInt mfb = mFace + fm*dpf, sfb = faceOff + i*dpf;

    PetscInt mnode = 0;
    for (PetscInt j = 0; j <= pm2; j++)
      for (PetscInt k = 0; k + j <= pm2; k++) {
        const PetscInt a[3] = {pm2 - k - j, k, j};
        const PetscInt cnode = femTriLoopIndex(pm2, a[ord[1]], a[ord[2]]);
        for (PetscInt l = 0; l < NUM_DIMENSIONS; l++) {
          const PetscReal m0 = in[(mfb + 2*mnode)*3 + l], m1 = in[(mfb + 2*mnode + 1)*3 + l];
          out[(sfb + 2*cnode)*3 + l]     = T[0][0]*m0 + T[0][1]*m1;
          out[(sfb + 2*cnode + 1)*3 + l] = T[1][0]*m0 + T[1][1]*m1;
        }
        mnode++;
      }
  }

  /* interior: never shared, identity (order -> closure volume slots). */
  for (PetscInt i = 0; i < dpv; i++) {
    for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
        out[i*3+j] = in[(mInt+i)*3+j];
    }
  }
}

/**
 * @brief Computes the determinant of a 3x3 matrix.
 *
 * This function returns the determinant of a 3x3 matrix by cofactor expansion
 * along the first row.
 *
 * @param[in] J  3x3 matrix.
 *
 * @return The determinant of J.
 */
static inline PetscReal femDeterminant(PetscReal J[NUM_DIMENSIONS][NUM_DIMENSIONS]) {
  return J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1])
       - J[0][1]*(J[1][0]*J[2][2] - J[1][2]*J[2][0])
       + J[0][2]*(J[1][0]*J[2][1] - J[1][1]*J[2][0]);
}

/**
 * @brief Applies the value and curl Piola pullbacks to an oriented field.
 *
 * This function maps the oriented reference field to physical space, applying
 * the covariant (value) Piola invJ * refValue and the contravariant (curl)
 * Piola J^T * refCurl / det, producing the physical values NiReal and curls
 * curlReal ([NUM_DIMENSIONS][dof]).
 *
 * @param[in]  dof          Number of DOFs.
 * @param[in]  jacobian     Cell Jacobian J = F^T.
 * @param[in]  invJacobian  Inverse Jacobian J^{-1}.
 * @param[in]  det          Determinant of the Jacobian.
 * @param[in]  orientV      Oriented reference values (dof x 3, row-major).
 * @param[in]  orientC      Oriented reference curls (dof x 3, row-major).
 * @param[out] NiReal       Physical values, [NUM_DIMENSIONS][dof].
 * @param[out] curlReal     Physical curls, [NUM_DIMENSIONS][dof].
 */
static void femPiola(PetscInt dof, PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS],
                     PetscReal invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal det,
                     const PetscReal *orientV, const PetscReal *orientC,
                     PetscReal NiReal[NUM_DIMENSIONS][FEM_MAX_ND_DOF],
                     PetscReal curlReal[NUM_DIMENSIONS][FEM_MAX_ND_DOF]) {
  
  for (PetscInt i = 0; i < dof; i++) {
    const PetscReal v0 = orientV[i*3+0], v1 = orientV[i*3+1], v2 = orientV[i*3+2];
    const PetscReal c0 = orientC[i*3+0], c1 = orientC[i*3+1], c2 = orientC[i*3+2];
    for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
      
      /* covariant (value) Piola: invJ * refValue */
      NiReal[j][i]   = invJacobian[j][0]*v0 + invJacobian[j][1]*v1 + invJacobian[j][2]*v2;
      
      /* contravariant (curl) Piola: J^T * refCurl / det */
      curlReal[j][i] = (jacobian[0][j]*c0 + jacobian[1][j]*c1 + jacobian[2][j]*c2) / det;
    }
  }
}

/**
 * @brief Inverts a dense N x N matrix by Gauss-Jordan elimination.
 *
 * This function computes Ainv = A^{-1} for a row-major N x N matrix using
 * Gauss-Jordan elimination with partial pivoting; A is overwritten in the
 * process. It is used to fold the native->closure orientation change of basis
 * into the discrete gradient. A singular column raises a PETSc error.
 *
 * @param[in]     N     Matrix dimension.
 * @param[in,out] A     Input matrix (N x N, row-major); overwritten.
 * @param[out]    Ainv  Output inverse (N x N, row-major).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
static PetscErrorCode femDenseInverse(PetscInt N, PetscReal *A, PetscReal *Ainv) {
  PetscFunctionBeginUser;

  for (PetscInt i = 0; i < N * N; i++) { 
    Ainv[i] = 0.0;
  }
  for (PetscInt i = 0; i < N; i++) {
    Ainv[i*N+i] = 1.0;
  }
  
  for (PetscInt i = 0; i < N; i++) {
    PetscInt piv = i; PetscReal best = PetscAbsReal(A[i*N+i]);
    for (PetscInt j = i + 1; j < N; j++) {
      PetscReal v = PetscAbsReal(A[j*N+i]);
      if (v > best) { 
        best = v; piv = j; 
      }
    }
    
    PetscCheck(best > 1.0e-300, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "fem: singular orientation matrix at column %" PetscInt_FMT, i);
    
    if (piv != i)
      for (PetscInt j = 0; j < N; j++) {
        PetscReal t;
        t = A[i*N+j];    A[i*N+j]    = A[piv*N+j];    A[piv*N+j]    = t;
        t = Ainv[i*N+j]; Ainv[i*N+j] = Ainv[piv*N+j]; Ainv[piv*N+j] = t;
      }
    PetscReal d = A[i*N+i];
    for (PetscInt j = 0; j < N; j++) {
        A[i*N+j] /= d; Ainv[i*N+j] /= d; 
    }
    
    for (PetscInt j = 0; j < N; j++) {
      if (j == i) {
        continue;
      }
      
      PetscReal f = A[j*N+i];
      
      if (f == 0.0) {
        continue;
      }
      
      for (PetscInt k = 0; k < N; k++) {
        A[j*N+k]    -= f * A[i*N+k];
        Ainv[j*N+k] -= f * Ainv[i*N+k];
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Accumulates the reference Nedelec mass and stiffness matrices.
 *
 * This function integrates, over the given quadrature rule, the conductivity-
 * weighted mass Me = INT (e_r . N_j) . N_k and the stiffness Ke = INT curl N_j .
 * curl N_k for one cell. At each quadrature point it evaluates the reference
 * Nedelec values/curls, orients them to closure order (femOrient) and applies
 * the Piola pullbacks (femPiola) before accumulating with the quadrature weight
 * and the Jacobian determinant.
 *
 * @param[in]  order            Nedelec order.
 * @param[in]  vertexCoords    Cell vertex coordinates (4x3) driving orientation.
 * @param[in]  jacobian        Cell Jacobian J = F^T.
 * @param[in]  invJacobian     Inverse Jacobian J^{-1}.
 * @param[in]  numGaussPoints  Number of quadrature points.
 * @param[in]  gaussPoints     Quadrature point coordinates (numGaussPoints x 3).
 * @param[in]  weights         Quadrature weights (numGaussPoints).
 * @param[in]  cellMaterial    Diagonal conductivity tensor (3).
 * @param[out] Me              Mass matrix (dof x dof).
 * @param[out] Ke              Stiffness matrix (dof x dof).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
static PetscErrorCode femElementalMatrix(PetscInt order, const PetscReal *vertexCoords,
                                         PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS],
                                         PetscReal invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS],
                                         PetscInt numGaussPoints, PetscReal **gaussPoints,
                                         PetscReal *weights, const PetscReal *cellMaterial,
                                         PetscReal **Me, PetscReal **Ke){
    PetscFunctionBeginUser;
    PetscCall(femEnsureNedelec(order));
    const PetscInt dof = feNedelecGetDof(femNd);

    const PetscReal det = femDeterminant(jacobian);
    const PetscReal e_r[NUM_DIMENSIONS] = {cellMaterial[0], cellMaterial[1], cellMaterial[2]};

    PetscReal shapeN[FEM_MAX_ND_DOF*NUM_DIMENSIONS], curlN[FEM_MAX_ND_DOF*NUM_DIMENSIONS];
    PetscReal orientV[FEM_MAX_ND_DOF*NUM_DIMENSIONS], orientC[FEM_MAX_ND_DOF*NUM_DIMENSIONS];
    PetscReal NiReal[NUM_DIMENSIONS][FEM_MAX_ND_DOF], curlReal[NUM_DIMENSIONS][FEM_MAX_ND_DOF];

    for (PetscInt i = 0; i < dof; i++) {
        for (PetscInt j = 0; j < dof; j++) { 
            Me[i][j] = 0.0; Ke[i][j] = 0.0; 
        }
    }

    for (PetscInt i = 0; i < numGaussPoints; i++) {
        
        /* Compute basis functions */
        PetscCall(feNedelecCalcVShape(femNd, gaussPoints[i][0], gaussPoints[i][1], gaussPoints[i][2], shapeN));
        
        /* Compute curl basis functions */
        PetscCall(feNedelecCalcCurlShape(femNd, gaussPoints[i][0], gaussPoints[i][1], gaussPoints[i][2], curlN));

        /* Orient basis functions */        
        femOrient(order, vertexCoords, shapeN, orientV);
        
        /* Orient curl basis functions */        
        femOrient(order, vertexCoords, curlN, orientC);
        
        /* Perform piola transformation */
        femPiola(dof, jacobian, invJacobian, det, orientV, orientC, NiReal, curlReal);

        for (PetscInt j = 0; j < dof; j++){
            for (PetscInt k = 0; k < dof; k++){
                /* mass: (e_r . Ni_j) . Ni_k  (e_r diagonal material tensor) */
                PetscReal mass = 0.0;
                
                for (PetscInt l = 0; l < NUM_DIMENSIONS; l++) {
                    mass += e_r[l]*NiReal[l][j]*NiReal[l][k];                    
                }
                
                Me[j][k] += weights[i] * mass * det;

                /* stiffness: curl_j . curl_k  (mu_r = identity) */
                PetscReal stiff = 0.0;
                
                for (PetscInt l = 0; l < NUM_DIMENSIONS; l++) {
                    stiff += curlReal[l][j]*curlReal[l][k];
                }
                
                Ke[j][k] += weights[i] * stiff * det;
            }
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Evaluates the physical Nedelec values and curls at a reference point.
 *
 * This function evaluates the reference Nedelec values/curls at a single point,
 * orients them to closure order (femOrient) and applies the Piola pullbacks
 * (femPiola), writing the physical basis values and curls as [NUM_DIMENSIONS][dof].
 *
 * @param[in]  order                Nedelec order.
 * @param[in]  vertexCoords        Cell vertex coordinates (4x3) driving orientation.
 * @param[in]  jacobian            Cell Jacobian J = F^T.
 * @param[in]  invJacobian         Inverse Jacobian J^{-1}.
 * @param[in]  point               Reference-cell evaluation point (3).
 * @param[out] basisFunctions      Physical values, [NUM_DIMENSIONS][dof].
 * @param[out] curlBasisFunctions  Physical curls, [NUM_DIMENSIONS][dof].
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
static PetscErrorCode femBasisFunctions(PetscInt order, const PetscReal *vertexCoords,
                                        PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS],
                                        PetscReal invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS],
                                        const PetscReal *point, PetscReal **basisFunctions,
                                        PetscReal **curlBasisFunctions){
    PetscFunctionBeginUser;

    PetscCall(femEnsureNedelec(order));
    const PetscInt dof = feNedelecGetDof(femNd);
    const PetscReal det = femDeterminant(jacobian);

    PetscReal shapeN[FEM_MAX_ND_DOF*NUM_DIMENSIONS], curlN[FEM_MAX_ND_DOF*NUM_DIMENSIONS];
    PetscReal orientV[FEM_MAX_ND_DOF*NUM_DIMENSIONS], orientC[FEM_MAX_ND_DOF*NUM_DIMENSIONS];
    PetscReal NiReal[NUM_DIMENSIONS][FEM_MAX_ND_DOF], curlReal[NUM_DIMENSIONS][FEM_MAX_ND_DOF];

    PetscCall(feNedelecCalcVShape(femNd, point[0], point[1], point[2], shapeN));
    PetscCall(feNedelecCalcCurlShape(femNd, point[0], point[1], point[2], curlN));
    femOrient(order, vertexCoords, shapeN, orientV);
    femOrient(order, vertexCoords, curlN, orientC);
    femPiola(dof, jacobian, invJacobian, det, orientV, orientC, NiReal, curlReal);

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++)
        for (PetscInt j = 0; j < dof; j++) {
            basisFunctions[i][j]     = NiReal[i][j];
            curlBasisFunctions[i][j] = curlReal[i][j];
        }

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Builds the per-cell discrete gradient block (Nedelec <- H1).
 *
 * This function builds the local discrete-gradient block G by direct
 * interpolation: the native coefficient of grad(phi_j) is
 * Gnat[m][j] = grad(phi_j)(node_m) . tangent_m (Nedelec DOF node/tangent from
 * feNedelecGetDofInfo, H1 reference gradient from feNodalH1Shape). The Nedelec
 * rows are then mapped to closure order via G = O^{-T} Gnat, where O is
 * femOrient's native->closure map (built by applying femOrient to the identity
 * and inverting). Entries below clampTol are set to exact zero.
 *
 * @param[in]  order            Nedelec/H1 order.
 * @param[in]  vertexCoords    Cell vertex coordinates (4x3) driving orientation.
 * @param[out] gradientMatrix  Per-cell block (dof x nH1).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
static PetscErrorCode femDiscreteGradient(PetscInt order, const PetscReal *vertexCoords, PetscReal **gradientMatrix) {
    /* Discrete gradient by direct interpolation: the native coefficient of grad(phi_j)
       is Gnat[m][j] = grad(phi_j)(node_m) . tangent_m (Nedelec dof node/tangent from
       feNedelecGetDofInfo, H1 reference gradient from feNodalH1Shape); the Nedelec rows
       are then mapped to closure order via G = O^{-T} Gnat, with O = femOrient's
       native->closure map (built by applying femOrient to the identity and inverting).
       Entries below clampTol are set to exact zero. */
    
    PetscFunctionBeginUser;
    
    PetscCall(femEnsureNedelec(order));
    const PetscInt  dof = feNedelecGetDof(femNd);
    const PetscInt  nH1 = (order + 1)*(order + 2)*(order + 3)/6;
    const PetscReal clampTol = 1.0e-12;
    
    PetscCheck(dof <= FEM_MAX_ND_DOF && nH1 <= FEM_MAX_H1_DOF, PETSC_COMM_SELF, PETSC_ERR_SUP,
               "femDiscreteGradient: order %" PetscInt_FMT " exceeds work-array cap (raise FEM_MAX_ND_DOF/FEM_MAX_H1_DOF)", order);

    PetscReal ShapH[FEM_MAX_H1_DOF], gradHbuf[NUM_DIMENSIONS][FEM_MAX_H1_DOF];
    PetscReal *GradH[NUM_DIMENSIONS] = {gradHbuf[0], gradHbuf[1], gradHbuf[2]};
    PetscReal ident[FEM_MAX_ND_DOF*NUM_DIMENSIONS], oriented[FEM_MAX_ND_DOF*NUM_DIMENSIONS];
    PetscReal *Gnat, *O, *Oinv;        /* O(dof^2) work: heap */

    PetscCall(PetscCalloc3(dof*nH1, &Gnat, dof*dof, &O, dof*dof, &Oinv));

    /* Native interpolation (reference): Gnat[m][j] = grad(phi_j)(node_m) . tangent_m */
    for (PetscInt i = 0; i < dof; i++) {
        PetscReal node[NUM_DIMENSIONS], tang[NUM_DIMENSIONS];
        PetscCall(feNedelecGetDofInfo(femNd, i, node, tang));
        PetscCall(feNodalH1Shape(order, vertexCoords, node, ShapH, GradH));
        for (PetscInt j = 0; j < nH1; j++) {
            PetscReal g = 0.0;
            for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
                g += GradH[k][j]*tang[k];
            }
            
            Gnat[i*nH1 + j] = g;
        }
    }

    /* Build O (dof x dof): O[k][m] = femOrient(e_m)[k]. femOrient acts on the dof
       index identically per spatial component, so one call yields 3 columns. */
    for (PetscInt i = 0; i < dof; i += NUM_DIMENSIONS) {
        for (PetscInt j = 0; j < dof*NUM_DIMENSIONS; j++) {
            ident[j] = 0.0;
        }
        
        for (PetscInt j = 0; j < NUM_DIMENSIONS && i + j < dof; j++) {
            ident[(i + j)*NUM_DIMENSIONS + j] = 1.0;
        }
        
        femOrient(order, vertexCoords, ident, oriented);
        
        for (PetscInt j = 0; j < NUM_DIMENSIONS && i + j < dof; j++) {
            for (PetscInt k = 0; k < dof; k++) {
                O[k*dof + (i + j)] = oriented[k*NUM_DIMENSIONS + j];
            }
        }
    }

    /* Oinv = O^{-1} (O is well conditioned -> clean inverse, no noise) */
    PetscCall(femDenseInverse(dof, O, Oinv));

    /* G = O^{-T} Gnat : G[k][j] = sum_m Oinv[m][k] Gnat[m][j], clamped */
    for (PetscInt i = 0; i < dof; i++)
        for (PetscInt j = 0; j < nH1; j++) {
            PetscReal g = 0.0;
            for (PetscInt k = 0; k < dof; k++) {
                g += Oinv[k*dof + i] * Gnat[k*nH1 + j];
            }
            gradientMatrix[i][j] = (PetscAbsReal(g) > clampTol) ? g : 0.0;
        }

    PetscCall(PetscFree3(Gnat, O, Oinv));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Maps a global point into reference-tetrahedron coordinates.
 *
 * This function applies the reference-consistent affine inverse of the cell's
 * geometric map, returning the (xi, eta, zeta) coordinates of a global point on
 * the reference tetrahedron whose vertex labelling matches the Nedelec / H1
 * reference bases. The four cell vertices are given row-major (4x3).
 *
 * @param[in]  coordinates  Cell vertex coordinates (4x3, row-major).
 * @param[in]  point        Global point to map.
 * @param[out] XiEtaZeta    Reference-cell coordinates of the point.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode tetrahedronXYZToReference(const PetscReal coordinates[NUM_VERTICES_PER_CELL * NUM_DIMENSIONS],
                                         const PetscReal point[NUM_DIMENSIONS], PetscReal XiEtaZeta[NUM_DIMENSIONS]) {
    PetscFunctionBeginUser;

    const PetscReal *c = coordinates;
    PetscReal J, xi, eta, zeta;

    J = c[5] * ( c[0] * (c[10] - c[7])
        + c[6] * (c[1] - c[10])
        + c[9] * (c[7] - c[1]) )
        + c[2] * ( c[3] * (c[7] - c[10])
        + c[6] * (c[10] - c[4])
        + c[9] * (c[4] - c[7]) )
        + c[8] * ( c[3] * (c[10] - c[1])
        + c[0] * (c[4] - c[10])
        + c[9] * (c[1] - c[4]) )
        + c[11] * ( c[3] * (c[1] - c[7])
        + c[0] * (c[7] - c[4])
        + c[6] * (c[4] - c[1]) );

    /* Compute affine transformation for xi */
    xi = ( c[11] * (c[7] - c[4]) + c[5] * (c[10] - c[7])
         + c[8] * (c[4] - c[10]) ) / J * point[0] +
         ( c[5] * (c[6] - c[9]) + c[11] * (c[3] - c[6])
         + c[8] * (c[9] - c[3]) ) / J * point[1] +
         ( c[3] * (c[7] - c[10]) + c[9] * (c[4] - c[7])
         + c[6] * (c[10] - c[4]) ) / J * point[2] +
         ( c[8] * (c[3] * c[10] - c[9] * c[4])
         + c[5] * (c[9] * c[7] - c[6] * c[10])
         + c[11] * (c[6] * c[4] - c[3] * c[7]) ) / J;

    /* Compute affine transformation for eta */
    eta = ( c[2] * (c[10] - c[4]) + c[11] * (c[4] - c[1])
          + c[5] * (c[1] - c[10]) ) / J * point[0] +
          ( c[2] * (c[3] - c[9]) + c[5] * (c[9] - c[0])
          + c[11] * (c[0] - c[3]) ) / J * point[1] +
          ( c[0] * (c[4] - c[10]) + c[3] * (c[10] - c[1])
          + c[9] * (c[1] - c[4]) ) / J * point[2] +
          ( c[2] * (c[9] * c[4] - c[3] * c[10])
          + c[5] * (c[0] * c[10] - c[9] * c[1])
          + c[11] * (c[3] * c[1] - c[0] * c[4]) ) / J;

    /* Compute affine transformation for zeta */
    zeta = ( c[5] * (c[7] - c[1]) + c[8] * (c[1] - c[4])
           + c[2] * (c[4] - c[7]) ) / J * point[0] +
           ( c[8] * (c[3] - c[0]) + c[5] * (c[0] - c[6])
           + c[2] * (c[6] - c[3]) ) / J * point[1] +
           ( c[3] * (c[1] - c[7]) + c[0] * (c[7] - c[4])
           + c[6] * (c[4] - c[1]) ) / J * point[2] +
           ( c[5] * ( c[9] * (c[1] - c[7])
           + c[10] * (c[6] - c[0]) )
           + c[8] * ( c[9] * (c[4] - c[1])
           + c[10] * (c[0] - c[3]) )
           + c[2] * ( c[9] * (c[7] - c[4])
           + c[10] * (c[3] - c[6]) )
           + c[11] * ( c[0] * (c[4] - c[7])
           + c[3] * (c[7] - c[1])
           + c[6] * (c[1] - c[4]) ) + J ) / J;

    XiEtaZeta[0] = xi;
    XiEtaZeta[1] = eta;
    XiEtaZeta[2] = zeta;

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Computes a rotated unit source-orientation vector.
 *
 * This function builds the weights vector for source rotation in the xyz frame
 * by composing the azimuth (x-y plane) and dip (x-z plane) rotation matrices and
 * applying them to the base x-directed unit vector. Angles are given in degrees.
 *
 * @param[in]  azimuth         Azimuth angle in degrees (x-y plane rotation).
 * @param[in]  dip             Dip angle in degrees (x-z plane rotation).
 * @param[out] rotationVector  Resulting rotated unit direction vector.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode computeVectorRotation(const PetscReal azimuth, const PetscReal dip, PetscReal rotationVector[NUM_DIMENSIONS]){
    
    /* Compute the weights vector for source rotation in the xyz plane. */
    PetscFunctionBeginUser;

    PetscReal base_vector[NUM_DIMENSIONS] = {1., 0., 0.};

    /* Convert degrees to radians for rotation */
    PetscReal alpha = azimuth * PETSC_PI / 180.;    // x-y plane
    PetscReal beta  = dip * PETSC_PI / 180.;        // x-z plane
    PetscReal tetha = 0.0 * PETSC_PI / 180.;        // y-z plane

    /* Define rotation matrices for each plane */
    PetscReal M1[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{PetscCosReal(alpha), -PetscSinReal(alpha),   0.},
                                                    {PetscSinReal(alpha),  PetscCosReal(alpha),   0.},
                                                    {                 0.,                   0.,   1.}};

    PetscReal M2[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{PetscCosReal(beta),  0.,  -PetscSinReal(beta)},
                                                    {                0.,  1.,                   0.},
                                                    {PetscSinReal(beta),  0.,   PetscCosReal(beta)}};

    PetscReal M3[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{1.,   0.,                                     0.},
                                                    {0.,   PetscCosReal(tetha),  -PetscSinReal(tetha)},
                                                    {0.,   PetscSinReal(tetha),   PetscCosReal(tetha)}};

    PetscReal temp1[NUM_DIMENSIONS][NUM_DIMENSIONS], temp2[NUM_DIMENSIONS][NUM_DIMENSIONS];

    /* Perform matrix multiplications */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
            PetscReal sum = 0.0;
            for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
                sum += M1[i][k] * M2[k][j];
            }
            temp1[i][j] = sum;
        }
    }

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
            PetscReal sum = 0.0;
            for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
                sum += temp1[i][k] * M3[k][j];
            }
            temp2[i][j] = sum;
        }
    }

    /* Apply rotation */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        rotationVector[i] = 0.0;
        for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
            rotationVector[i] += temp2[i][j] * base_vector[j];
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Sets the number of 3D quadrature points for a tetrahedron.
 *
 * This function selects the single quadrature path used for every order: a
 * Stroud conical rule with order+1 points per axis, exact to degree 2*order+1,
 * which covers the degree-2*order mass-matrix integrand on an affine tetrahedron.
 * It stores the total point count (order+1)^3 in the rule.
 *
 * @param[in]  order        Basis order driving the quadrature degree.
 * @param[out] quadrature  Rule whose numPoints is set.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode computeNum3DQuadraturePoints(const PetscInt order, Quadrature3D* quadrature){
    PetscFunctionBeginUser;

    PetscCheck(order >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error: order must be >= 1 (got %" PetscInt_FMT ").\n", order);

    /* One quadrature path for every order. A Stroud conical rule with order+1
       points per axis is exact to degree 2*order+1, which covers the degree-2*order
       mass-matrix integrand on an affine tetrahedron. Total points (order+1)^3. */
    const PetscInt m = order + 1;
    quadrature->numPoints = m * m * m;

    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Populates the 3D quadrature points and weights for a tetrahedron.
 *
 * This function recovers the per-axis point count m = order+1 from the stored
 * numPoints (a perfect cube) and fills the rule's points and weights via the
 * Stroud conical construction on the unit reference tetrahedron.
 *
 * @param[in,out] quadrature  Rule (numPoints set) whose points/weights are filled.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode compute3DQuadraturePoints(Quadrature3D* quadrature){
    PetscFunctionBeginUser;

    /* numPoints == (order+1)^3; recover the per-axis point count m = order+1. */
    const PetscInt m = (PetscInt)(PetscCbrtReal((PetscReal)quadrature->numPoints) + 0.5);
    PetscCheck(m*m*m == quadrature->numPoints, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "compute3DQuadraturePoints: numPoints %" PetscInt_FMT " is not a perfect cube", quadrature->numPoints);

    PetscCall(femGaussPoints3D(m, quadrature->points, quadrature->weights));

    PetscFunctionReturn(PETSC_SUCCESS);
}


/* ===========================================================================
 * Struct-based adapters consumed by the DMPlex assembly / postprocessing.
 * ========================================================================= */

/**
 * @brief Computes a cell's geometric Jacobian, inverse, and determinant.
 *
 * This function fills the cell's jacobian (rows (v1-v0), (v2-v0), (v3-v0)), its
 * inverse (from the adjugate) and its determinant. It is used for point location
 * and degeneracy checks; the FE routines build their own reference-consistent
 * Jacobian internally.
 *
 * @param[in,out] cell  Cell whose jacobian/invJacobian/detJacobian are filled.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode computeCellJacobian(Cell* cell) {
  /* Geometric Jacobian of the affine reference -> physical map, used for point
   * location / degeneracy checks. Rows are (v1-v0), (v2-v0), (v3-v0); the FE
   * routines build their own reference-consistent Jacobian internally. */
  PetscFunctionBeginUser;

  PetscReal coFactorMatrix[NUM_DIMENSIONS][NUM_DIMENSIONS], adjugateMatrix[NUM_DIMENSIONS][NUM_DIMENSIONS];
  PetscReal invDeterminant;

  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
      cell->jacobian[i][j] = 0.0;
      cell->invJacobian[i][j] = 0.0;
    }
  }

  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    cell->jacobian[0][i] = cell->coordinates[3 + i] - cell->coordinates[i]; // v1 - v0
    cell->jacobian[1][i] = cell->coordinates[6 + i] - cell->coordinates[i]; // v2 - v0
    cell->jacobian[2][i] = cell->coordinates[9 + i] - cell->coordinates[i]; // v3 - v0
  }

  cell->detJacobian = cell->jacobian[0][0] * (cell->jacobian[1][1] * cell->jacobian[2][2] - cell->jacobian[1][2] * cell->jacobian[2][1]) -
                      cell->jacobian[0][1] * (cell->jacobian[1][0] * cell->jacobian[2][2] - cell->jacobian[1][2] * cell->jacobian[2][0]) +
                      cell->jacobian[0][2] * (cell->jacobian[1][0] * cell->jacobian[2][1] - cell->jacobian[1][1] * cell->jacobian[2][0]);

  coFactorMatrix[0][0] = cell->jacobian[1][1] * cell->jacobian[2][2] - cell->jacobian[1][2] * cell->jacobian[2][1];
  coFactorMatrix[0][1] = -(cell->jacobian[1][0] * cell->jacobian[2][2] - cell->jacobian[1][2] * cell->jacobian[2][0]);
  coFactorMatrix[0][2] = cell->jacobian[1][0] * cell->jacobian[2][1] - cell->jacobian[1][1] * cell->jacobian[2][0];
  coFactorMatrix[1][0] = -(cell->jacobian[0][1] * cell->jacobian[2][2] - cell->jacobian[0][2] * cell->jacobian[2][1]);
  coFactorMatrix[1][1] = cell->jacobian[0][0] * cell->jacobian[2][2] - cell->jacobian[0][2] * cell->jacobian[2][0];
  coFactorMatrix[1][2] = -(cell->jacobian[0][0] * cell->jacobian[2][1] - cell->jacobian[0][1] * cell->jacobian[2][0]);
  coFactorMatrix[2][0] = cell->jacobian[0][1] * cell->jacobian[1][2] - cell->jacobian[0][2] * cell->jacobian[1][1];
  coFactorMatrix[2][1] = -(cell->jacobian[0][0] * cell->jacobian[1][2] - cell->jacobian[0][2] * cell->jacobian[1][0]);
  coFactorMatrix[2][2] = cell->jacobian[0][0] * cell->jacobian[1][1] - cell->jacobian[0][1] * cell->jacobian[1][0];

  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
      adjugateMatrix[i][j] = coFactorMatrix[j][i];
    }
  } 

  invDeterminant = 1.0 / (cell->detJacobian);
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
      cell->invJacobian[i][j] = invDeterminant * adjugateMatrix[i][j];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the elemental mass and stiffness matrices for a cell.
 *
 * This function builds the cell Jacobian and delegates to the reference core to
 * accumulate the conductivity-weighted mass Me and the stiffness Ke over the
 * given quadrature rule. Orientation and the Piola pullbacks are applied
 * internally, so the returned blocks are physical and in DMPlex closure order.
 *
 * @param[in]  fem         Finite-element space descriptor (order, DOF counts).
 * @param[in]  cell        Cell geometry and conductivity.
 * @param[in]  quadrature  3D quadrature rule.
 * @param[out] Me          Elemental mass matrix (numDofInCell x numDofInCell).
 * @param[out] Ke          Elemental stiffness matrix (numDofInCell x numDofInCell).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode computeElementalMatrices(const FEMSpace* fem, const Cell* cell, const Quadrature3D* quadrature,
                                        PetscReal** Me, PetscReal** Ke) {
  PetscFunctionBeginUser;
  
  PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS];
  
  /* Mass is weighted by the cell's (diagonal) conductivity tensor. */
  const PetscReal cellMaterial[NUM_DIMENSIONS] = {cell->conductivity[0], cell->conductivity[1], cell->conductivity[2]};

  PetscCall(femComputeJacobian(cell->coordinates, jacobian, invJacobian));
  PetscCall(femElementalMatrix(fem->order, cell->coordinates, jacobian, invJacobian,
                               quadrature->numPoints, quadrature->points, quadrature->weights,
                               cellMaterial, Me, Ke));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Evaluates the Nedelec basis (and optionally curls) at a reference point.
 *
 * This function builds the cell Jacobian and evaluates the physical, oriented
 * Nedelec basis values (and curls) at the given reference point. Passing
 * NiCurl = NULL skips returning the curls (a scratch buffer is used internally,
 * since the reference evaluator always produces them).
 *
 * @param[in]  fem     Finite-element space descriptor.
 * @param[in]  cell    Cell geometry.
 * @param[in]  point   Reference-cell evaluation point (3).
 * @param[out] Ni      Basis-function values: Ni[d][dof], d in [0,3).
 * @param[out] NiCurl  Basis-curl values: NiCurl[d][dof] (NULL to skip).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode evaluateNedelecBasis(const FEMSpace* fem, const Cell* cell,
                                    const PetscReal point[NUM_DIMENSIONS],
                                    PetscReal** Ni, PetscReal** NiCurl) {
  PetscFunctionBeginUser;
  PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS];
  PetscCall(femComputeJacobian(cell->coordinates, jacobian, invJacobian));

  if (NiCurl) {
    PetscCall(femBasisFunctions(fem->order, cell->coordinates, jacobian, invJacobian, point, Ni, NiCurl));
  } else {
    /* Caller wants values only; the reference evaluator always produces curls,
     * so hand it a scratch buffer for them. */
    const PetscInt dof = fem->numDofInCell;
    PetscReal *flat, *curlScratch[NUM_DIMENSIONS];
    PetscCall(PetscMalloc1(NUM_DIMENSIONS * dof, &flat));
    
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        curlScratch[i] = flat + i * dof;
    }
    
    PetscCall(femBasisFunctions(fem->order, cell->coordinates, jacobian, invJacobian, point, Ni, curlScratch));
    PetscCall(PetscFree(flat));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Builds the high-order discrete gradient block for one cell.
 *
 * This function delegates to the reference core to build the per-cell discrete
 * gradient block G_e (rows = Nedelec DOFs, columns = P_order H1 DOFs, both in
 * DMPlex closure order), the curl-kernel operator consumed by
 * PCBDDCSetDiscreteGradient.
 *
 * @param[in]  fem             FE space descriptor.
 * @param[in]  cell            Cell geometry.
 * @param[out] gradientMatrix  Block sized numDofInCell x numH1DofInCell_Porder.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode buildDiscreteGradientMatrix(const FEMSpace* fem, const Cell* cell, PetscReal** gradientMatrix) {
  PetscFunctionBeginUser;

  PetscCall(femDiscreteGradient(fem->order, cell->coordinates, gradientMatrix));
  
  PetscFunctionReturn(PETSC_SUCCESS);
}
