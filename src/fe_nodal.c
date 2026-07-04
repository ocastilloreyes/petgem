/*
 * Filename: fe_nodal.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-06-22
 *
 * Description:
 * Arbitrary-order nodal (Lagrange) H1 basis on the reference tetrahedron, the
 * column space of the discrete gradient (a De Rham pair with the Nedelec space).
 *
 * The basis is the equispaced nodal Lagrange basis written in closed form in
 * barycentric coordinates: for a node multi-index alpha (|alpha| = order) the shape
 * function is the product of 1D nodal factors
 *     phi_alpha(lam) = prod_i L_{alpha_i}(lam_i),
 *     L_k(t) = (1/k!) prod_{m=0}^{k-1} (order*t - m).
 * This reproduces standard P1/P2/P3 exactly and extends to any order with no
 * per-order formulas.
 *
 * Shape functions are emitted in DMPlex H1 closure order, by decreasing entity
 * dimension: [volume, face, edge, vertex] (volume dofs appear for order>=4, face
 * dofs for order>=3, edge dofs for order>=2). Within an entity the dofs are oriented
 * geometrically by lexicographic vertex coordinate (identical for the cells sharing
 * the entity, matching femOrient on the Nedelec side):
 *   - edges (order-1 dofs): ordered from the lower- to the higher-coordinate vertex;
 *   - faces ((order-1)(order-2)/2 dofs): the interior triangular node grid is permuted
 *     by the same femTriLoopIndex rule used on the Nedelec side;
 *   - volume: never shared, emitted in a fixed enumeration (no orientation).
 */

#include "fe_nodal.h"
#include "constants.h"

/* DMPlex closure edge slots E0..E5 -> their two local cell vertices (from the
 * reference-cell topology probe). Only the vertex pair is used; the dof order
 * along the edge is set by vertex coordinate (orientation), so the stored pair
 * order is irrelevant. */
static const PetscInt feNodalEdgeV[NUM_EDGES_PER_CELL][2] = { {0,1}, {1,2}, {2,0}, {0,3}, {3,1}, {2,3} };

/* DMPlex closure face slots F0..F3 -> their three local cell vertices (closure face
 * slot F excludes vertex 3-F, matching the Nedelec face-slot convention in fem.c). */
static const PetscInt feNodalFaceV[NUM_FACES_PER_CELL][3] = { {0,1,2}, {0,1,3}, {0,2,3}, {1,2,3} };

/**
 * @brief Lexicographically compares two local cell vertices by coordinate.
 *
 * This function orders vertices vi and vj by their (x, y, z) coordinates, the
 * comparison used to orient shared edges and faces. It mirrors femCoordLess in
 * fem.c so the H1 and Nedelec edge/face orientations agree, which keeps the two
 * global spaces conforming.
 *
 * @param[in] vc  Cell vertex coordinates (4x3, row-major).
 * @param[in] vi  First local vertex index.
 * @param[in] vj  Second local vertex index.
 *
 * @return PETSC_TRUE when vertex vi precedes vertex vj lexicographically, else
 *         PETSC_FALSE.
 */
static inline PetscBool feNodalCoordLess(const PetscReal *vc, PetscInt vi, PetscInt vj) {
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
 * @brief Maps a triangular-grid node (i, j) to its index within a face loop.
 *
 * This function returns the linear position of the node (i, j) in the interior
 * triangular grid of a face, using the same enumeration as femTriLoopIndex in
 * fem.c so the H1 face permutation matches the Nedelec one.
 *
 * @param[in] m  Triangular-grid size parameter (order - 3).
 * @param[in] i  First grid coordinate.
 * @param[in] j  Second grid coordinate.
 *
 * @return The linear index of node (i, j) within the face node loop.
 */
static inline PetscInt feNodalTriIndex(PetscInt m, PetscInt i, PetscInt j) {
  return j*(m + 1) - j*(j - 1)/2 + i;
}

/**
 * @brief Emits one nodal shape value and reference gradient at a closure slot.
 *
 * This function evaluates, for the node multi-index alpha, the shape value
 * phi = prod_i L[i][alpha_i] and its reference gradient
 * d phi/d lam_i = dL[i][alpha_i] * prod_{j!=i} L[j][alpha_j], contracted with
 * the constant barycentric gradients dlam, and writes them into closure slot s
 * of ShapH and GradH. The precomputed 1D factors L[i][k]=L_k(lam_i) and
 * dL[i][k]=L_k'(lam_i) are supplied by the caller.
 *
 * @note L and dL are read-only here but are passed non-const: implicitly adding
 *       const to a pointer-to-array type is an ISO C constraint violation
 *       flagged by -Wpedantic, so the qualifier is dropped rather than cast at
 *       every call site.
 *
 * @param[in]  L      1D nodal factors L[i][k] = L_k(lam_i).
 * @param[in]  dL     1D nodal factor derivatives dL[i][k] = L_k'(lam_i).
 * @param[in]  dlam   Constant barycentric gradients, [4][NUM_DIMENSIONS].
 * @param[in]  alpha  Node multi-index (|alpha| = order).
 * @param[in]  s      Closure slot to write.
 * @param[out] ShapH  Shape values, indexed by closure slot.
 * @param[out] GradH  Reference gradients, GradH[d][slot], d in [0,3).
 */
static void feNodalEmit(PetscReal L[][FE_NODAL_MAX_ORDER + 1], PetscReal dL[][FE_NODAL_MAX_ORDER + 1], const PetscReal dlam[][NUM_DIMENSIONS], const PetscInt alpha[4],
                        PetscInt s, PetscReal *ShapH, PetscReal **GradH) {

  ShapH[s] = L[0][alpha[0]] * L[1][alpha[1]] * L[2][alpha[2]] * L[3][alpha[3]];

  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscReal g = 0.0;
    for (PetscInt j = 0; j < 4; j++) {
      PetscReal prod = dL[j][alpha[j]];
      for (PetscInt k = 0; k < 4; k++)
        if (k != j) prod *= L[k][alpha[k]];
      g += prod * dlam[j][i];
    }
    GradH[i][s] = g;
  }
}

/**
 * @brief Reports whether the nodal H1 basis supports a given order.
 *
 * This function returns whether a matched nodal H1 basis is available for the
 * requested order, i.e. whether order lies in [1, FE_NODAL_MAX_ORDER].
 *
 * @param[in] order  Polynomial order.
 *
 * @return PETSC_TRUE for order in [1, FE_NODAL_MAX_ORDER], else PETSC_FALSE.
 */
PetscBool feNodalSupports(PetscInt order) {
  return (PetscBool)(order >= 1 && order <= FE_NODAL_MAX_ORDER);
}

/**
 * @brief Evaluates the nodal H1 shape values and reference gradients.
 *
 * This function evaluates the arbitrary-order nodal (Lagrange) H1 basis at a
 * reference-cell point and writes the shape values and reference gradients in
 * DMPlex H1 closure order [volume, face, edge, vertex]. It first builds the 1D
 * nodal factors L_k(lam_i) and their derivatives by recurrence, then emits the
 * vertex, edge, face and volume DOFs via feNodalEmit. Edge and face DOFs are
 * oriented geometrically by lexicographic vertex coordinate (using the cell
 * vertex coordinates), so the emitted ordering matches the Nedelec side and the
 * two spaces stay conforming; volume DOFs are never shared and use a fixed
 * enumeration.
 *
 * @param[in]  order          Polynomial order (1..FE_NODAL_MAX_ORDER).
 * @param[in]  vertexCoords  Cell vertex coordinates (4x3) for edge/face orientation.
 * @param[in]  X             Reference-cell evaluation point (3).
 * @param[out] ShapH         Shape values, one per H1 DOF, in closure order.
 * @param[out] GradH         Reference gradients: GradH[d][dof], d in [0,3).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode feNodalH1Shape(PetscInt order, const PetscReal *vertexCoords, const PetscReal X[3], PetscReal *ShapH, PetscReal **GradH) {
  PetscFunctionBeginUser;
  PetscCheck(order >= 1 && order <= FE_NODAL_MAX_ORDER, PETSC_COMM_SELF, PETSC_ERR_SUP, "feNodalH1Shape: only order = 1..%d implemented (got %" PetscInt_FMT ")",
             FE_NODAL_MAX_ORDER, order);

  const PetscInt  N = order;
  const PetscReal lam[4] = {1.0 - X[0] - X[1] - X[2], X[0], X[1], X[2]};
  const PetscReal dlam[4][NUM_DIMENSIONS] = {{-1,-1,-1}, {1,0,0}, {0,1,0}, {0,0,1}};

  /* 1D nodal factors L_k(lam_i) and derivatives, k = 0..N, by recurrence
     L_k = L_{k-1} (N lam_i - (k-1))/k. */
  PetscReal L[4][FE_NODAL_MAX_ORDER + 1], dL[4][FE_NODAL_MAX_ORDER + 1];
  for (PetscInt i = 0; i < 4; i++) {
    L[i][0] = 1.0; dL[i][0] = 0.0;
    for (PetscInt j = 1; j <= N; j++) {
      const PetscReal s = (PetscReal)N*lam[i] - (PetscReal)(j - 1);
      L[i][j]  = L[i][j - 1]*s/(PetscReal)j;
      dL[i][j] = (dL[i][j - 1]*s + L[i][j - 1]*(PetscReal)N)/(PetscReal)j;
    }
  }

  /* Closure offsets (decreasing entity dimension): [volume, face, edge, vertex]. */
  const PetscInt nEdgePer = N - 1;
  const PetscInt nFacePer = (N - 1)*(N - 2)/2;
  const PetscInt nVol     = (N - 1)*(N - 2)*(N - 3)/6;
  const PetscInt offFace  = nVol;
  const PetscInt offEdge  = nVol + NUM_FACES_PER_CELL*nFacePer;
  const PetscInt offVert  = offEdge + NUM_EDGES_PER_CELL*nEdgePer;

  /* vertices: alpha = N e_v */
  for (PetscInt i = 0; i < NUM_VERTICES_PER_CELL; i++) {
    PetscInt alpha[4] = {0, 0, 0, 0};
    alpha[i] = N;
    feNodalEmit(L, dL, dlam, alpha, offVert + i, ShapH, GradH);
  }

  /* edges: order-1 dofs, ordered from the lower- to the higher-coordinate vertex */
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; i++) {
    const PetscInt  va = feNodalEdgeV[i][0], vb = feNodalEdgeV[i][1];
    const PetscBool aLow = feNodalCoordLess(vertexCoords, va, vb);
    const PetscInt  lo = aLow ? va : vb, hi = aLow ? vb : va;
    for (PetscInt j = 0; j < nEdgePer; j++) {
      PetscInt alpha[4] = {0, 0, 0, 0};
      alpha[lo] = N - 1 - j;
      alpha[hi] = 1 + j;
      feNodalEmit(L, dL, dlam, alpha, offEdge + i*nEdgePer + j, ShapH, GradH);
    }
  }

  /* faces: interior triangular grid (i+j <= N-3), permuted by face orientation
     exactly as femOrient does (sort the 3 face vertices by coordinate -> ord,
     closure slot = feNodalTriIndex over the sorted grid components). */
  const PetscInt mF = N - 3;
  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; i++) {
    const PetscInt *fv = feNodalFaceV[i];
    PetscInt ord[3] = {0, 1, 2};
    for (PetscInt j = 1; j < 3; j++)
      for (PetscInt k = j; k > 0 && feNodalCoordLess(vertexCoords, fv[ord[k]], fv[ord[k - 1]]); k--) {
        PetscInt tmp = ord[k]; ord[k] = ord[k - 1]; ord[k - 1] = tmp;
      }
    for (PetscInt j = 0; j <= mF; j++)
      for (PetscInt k = 0; k + j <= mF; k++) {
        const PetscInt c[3] = {mF - k - j, k, j};
        const PetscInt cnode = feNodalTriIndex(mF, c[ord[1]], c[ord[2]]);
        PetscInt alpha[4] = {0, 0, 0, 0};
        alpha[fv[0]] = c[0] + 1;
        alpha[fv[1]] = c[1] + 1;
        alpha[fv[2]] = c[2] + 1;
        feNodalEmit(L, dL, dlam, alpha, offFace + i*nFacePer + cnode, ShapH, GradH);
      }
  }

  /* volume (interior): never shared, fixed enumeration, no orientation */
  const PetscInt mV = N - 4;
  PetscInt vs = 0;
  for (PetscInt i = 0; i <= mV; i++)
    for (PetscInt j = 0; j + i <= mV; j++)
      for (PetscInt k = 0; k + j + i <= mV; k++) {
        PetscInt alpha[4];
        alpha[0] = (mV - i - j - k) + 1;
        alpha[1] = k + 1;
        alpha[2] = j + 1;
        alpha[3] = i + 1;
        feNodalEmit(L, dL, dlam, alpha, vs, ShapH, GradH);
        vs++;
      }

  PetscFunctionReturn(PETSC_SUCCESS);
}
