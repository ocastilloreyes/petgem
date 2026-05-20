/*
 * Filename: hvfem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Functions for high-order vector finite element method (HVFEM)
 * computations.
 */

/* C libraries */

/* PETSc libraries */
#include <petsc.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>

/* PETGEM functions */
#include "constants.h"
#include "hvfem.h"
#include "hvfem_internal.h"

/* ============================================================================
 * Forward declarations of file-local (static) helpers.
 *
 * Helpers shared with the hierarchical Nédélec TU — dotProduct, crossProduct,
 * vectorNorm, invertMatrix, cartesianToVolumetricCoordinates,
 * solve3x3MatrixSystem3x6RHS, shape3DHTet — are declared in hvfem_internal.h
 * and have external linkage.
 *
 * The remaining helpers stay file-local; their definitions follow below in
 * the same logical groupings.
 * ========================================================================= */

/* Vector algebra (file-local only) */
static PetscErrorCode tripleProduct(const PetscReal a[NUM_DIMENSIONS], const PetscReal b[NUM_DIMENSIONS], const PetscReal c[NUM_DIMENSIONS], PetscReal* result);

/* Quadrature renormalization */
static PetscErrorCode renormalization2DGaussPoints(const PetscReal (*gaussPoints)[3], Quadrature2D* quadrature);
static PetscErrorCode renormalization3DGaussPoints(const PetscReal (*gaussPoints)[4], Quadrature3D* quadrature);

/* Reference-cell coordinates.
 * AffineTetrahedron / ProjectTetE / ProjectTetF and the orientation helpers
 * OrientE / OrientTri are also consumed by the hierarchical Nédélec basis
 * (src/hvfem_hierarchical.c); their declarations live in hvfem_internal.h
 * and they are no longer file-static. BlendTetV stays file-local. */
static PetscErrorCode BlendTetV(const PetscReal Lam[NUM_DIMENSIONS + 1], const PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1],
                                PetscReal LambV[NUM_VERTICES_PER_CELL], PetscReal DLambV[NUM_VERTICES_PER_CELL][NUM_DIMENSIONS]);

/* Polynomial bases. PolyLegendre is also reused by HomLegendre in the
 * hierarchical basis TU; its declaration lives in hvfem_internal.h.
 * HomIJacobi (declared in hvfem_internal.h) is consumed by AncETri there. */
static PetscErrorCode PolyILegendre(const PetscReal X, const PetscReal T, const PetscInt nord, const PetscBool Idec, PetscReal homL[],
                                    PetscReal homP[], PetscReal homR[]);
static PetscErrorCode PolyJacobi(const PetscReal X, const PetscReal T, const PetscInt nord, PetscInt Minalpha, PetscReal** P);
static PetscErrorCode PolyIJacobi(const PetscReal X, const PetscReal T, const PetscInt nord, const PetscInt Minalpha, PetscReal** L,
                                  PetscReal** P, PetscReal** R);

/* Edge/face ancillary polynomials */
static PetscErrorCode HomILegendre(const PetscReal S[2], const PetscReal DS[NUM_DIMENSIONS][2], const PetscInt nord, PetscBool const Idec,
                                   PetscReal* PhiE, PetscReal** DPhiE);
static PetscErrorCode AncPhiE(const PetscReal S[2], const PetscReal DS[NUM_DIMENSIONS][2], const PetscInt nord, const PetscBool Idec,
                              PetscReal* PhiE, PetscReal** DPhiE);
static PetscErrorCode AncPhiTri(const PetscReal S[NUM_DIMENSIONS], const PetscReal DS[NUM_DIMENSIONS][NUM_DIMENSIONS],
                                const PetscInt nordFace, const PetscBool IdecF, PetscReal** PhiTri, PetscReal*** DPhiTri);

/**
 * @brief Computes the dot product of two vectors in `NUM_DIMENSIONS`-dimensional space.
 *
 * This function calculates the standard Euclidean inner product between
 * two real-valued vectors of fixed dimension `NUM_DIMENSIONS` and stores
 * the result in the provided output variable.
 *
 * @param[in]  vector1  Array of length `NUM_DIMENSIONS` representing the first vector.
 * @param[in]  vector2  Array of length `NUM_DIMENSIONS` representing the second vector.
 * @param[out] result   Pointer to a PetscReal where the computed dot product will be stored.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 *
 * @note The function assumes both input vectors have exactly `NUM_DIMENSIONS` entries.
 */
PetscErrorCode dotProduct(const PetscReal vector1[NUM_DIMENSIONS], const PetscReal vector2[NUM_DIMENSIONS], PetscReal* result) {
  PetscFunctionBeginUser;

  *result = 0.0;
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    *result += vector1[i] * vector2[i];
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the scalar triple product of three 3D vectors.
 *
 * This function evaluates the scalar triple product of vectors `a`, `b`, and `c`,
 * i.e., `a · (b × c)`. This value corresponds to the signed volume of the
 * parallelepiped formed by the three vectors.
 *
 * @param[in]  a       Array of length 3 representing the first vector.
 * @param[in]  b       Array of length 3 representing the second vector.
 * @param[in]  c       Array of length 3 representing the third vector.
 * @param[out] result  Pointer to a PetscReal where the computed scalar triple
 *                     product will be stored.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 *
 * @note The function assumes that all input vectors are 3-dimensional.
 */
static PetscErrorCode tripleProduct(const PetscReal a[NUM_DIMENSIONS], const PetscReal b[NUM_DIMENSIONS], const PetscReal c[NUM_DIMENSIONS], PetscReal* result) {
  PetscFunctionBeginUser;

  *result = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode crossProduct(const PetscReal a[NUM_DIMENSIONS], const PetscReal b[NUM_DIMENSIONS], PetscReal result[NUM_DIMENSIONS]) {

  PetscFunctionBeginUser;

  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Renormalizes 2D Gauss quadrature points and weights for a triangular element.
 *
 * This function maps Gauss points from the reference triangle [-1,1] × [-1,1]
 * to the standard 2D triangular element used in PETGEM, and scales the
 * quadrature weights accordingly.
 *
 * @param[in]  gaussPoints  Array of size [numPoints][3], containing the original
 *                          Gauss points and weights. The first two entries are
 *                          the reference coordinates, and the third entry is the weight.
 * @param[out] quadrature   Pointer to a `Quadrature2D` struct to be populated
 *                          with mapped points and renormalized weights. Must
 *                          have `numPoints` properly set.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 *
 * @note The transformation applied is specific to PETGEM’s triangular element
 *       conventions:
 *       \f$x = (1 + ξ)/2\f$, \f$y = -(ξ + η)/2\f$, weight scaled by 1/4.
 */
static PetscErrorCode renormalization2DGaussPoints(const PetscReal (*gaussPoints)[3], Quadrature2D* quadrature) {
  PetscFunctionBeginUser;

  for (PetscInt i = 0; i < quadrature->numPoints; i++) {
    quadrature->weights[i] = gaussPoints[i][2] / 4.0;
    quadrature->points[i][0] = (1.0 + gaussPoints[i][0]) / 2.0;                // x
    quadrature->points[i][1] = -(gaussPoints[i][0] + gaussPoints[i][1]) / 2.0; // y
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Renormalizes Gauss points from a [-1, 1]-based cube to a [0, 1]-based reference tetrahedron.
 *
 * This function maps Gauss points and weights defined on a reference cube
 * [-1,1]^3 to the standard PETGEM tetrahedral element reference domain [0,1]^3.
 * The mapping also rescales the weights according to the Jacobian of the
 * transformation (division by 8).
 *
 * @param[in]  gaussPoints  Array of size [numPoints][4], where the first three entries
 *                          are the reference coordinates (ξ, η, ζ) and the fourth
 *                          entry is the original weight.
 * @param[out] quadrature   Pointer to a `Quadrature3D` struct to be populated
 *                          with mapped points (`points`) and renormalized weights (`weights`).
 *                          Must have `numPoints` properly set.
 *
 * @return PetscErrorCode   PETSC_SUCCESS always.
 *
 * @note The coordinate transformation follows PETGEM’s tetrahedral conventions:
 *       - ξ → x = (1 + η)/2
 *       - η → y = -(1 + ξ + η + ζ)/2
 *       - ζ → z = (1 + ξ)/2
 * @note The weight scaling factor accounts for the volume change from the
 *       [-1,1]^3 cube to the reference tetrahedron (weight / 8).
 */
static PetscErrorCode renormalization3DGaussPoints(const PetscReal (*gaussPoints)[4], Quadrature3D* quadrature) {
  PetscFunctionBeginUser;

  for (PetscInt i = 0; i < quadrature->numPoints; i++) {
    quadrature->weights[i] = gaussPoints[i][NUM_DIMENSIONS] / 8;
    quadrature->points[i][0] = (1 + gaussPoints[i][1]) / 2;
    quadrature->points[i][1] = -(1 + gaussPoints[i][0] + gaussPoints[i][1] + gaussPoints[i][2]) / 2;
    quadrature->points[i][2] = (1 + gaussPoints[i][0]) / 2;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the inverse of a small square matrix using Gaussian elimination with partial pivoting.
 *
 * This function inverts an N × N real matrix `A` and stores the result in `invA`.
 * It uses a simple augmented matrix approach [A | I] and performs Gaussian elimination
 * with partial pivoting. Designed for small matrices (N ≤ 6), as the algorithm
 * uses a fixed-size stack array.
 *
 * @param[in]  N     The dimension of the square matrix.
 * @param[in]  A     Array of size N×N containing the matrix to be inverted.
 * @param[out] invA  Array of size N×N where the computed inverse will be stored.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error if the matrix is singular.
 *
 * @note The function raises a PETSc error if the matrix is detected to be singular
 *       (pivot < 1e-14 during elimination).
 * @note Intended for small matrices due to the fixed-size augmented array `aug[6][12]`.
 * @note Uses PETSc macros for error handling (`SETERRQ`) and absolute values (`PetscAbsReal`).
 */
PetscErrorCode invertMatrix(const PetscInt N, const PetscReal A[], PetscReal invA[]) {
  PetscFunctionBeginUser;

  PetscReal aug[6][12]; // max N=6
  PetscReal tmp, pivot, factor, val;
  PetscInt pivot_row;

  /* Build augmented matrix [A | I] */
  for (PetscInt i = 0; i < N; i++) {
    for (PetscInt j = 0; j < N; j++) {
      aug[i][j] = A[i * N + j];
      aug[i][j + N] = (i == j) ? 1.0 : 0.0;
    }
  }

  /* Gaussian elimination with partial pivoting */
  for (PetscInt i = 0; i < N; i++) {
    /* Find pivot */
    pivot_row = i;
    PetscReal max_val = PetscAbsReal(aug[i][i]);
    for (PetscInt k = i + 1; k < N; k++) {
      val = PetscAbsReal(aug[k][i]);
      if (val > max_val) {
        max_val = val;
        pivot_row = k;
      }
    }

    if (max_val < 1e-14) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Singular matrix in invertMatrix()");
    }

    /* Swap rows if needed */
    if (pivot_row != i) {
      for (PetscInt j = 0; j < 2 * N; j++) {
        tmp = aug[i][j];
        aug[i][j] = aug[pivot_row][j];
        aug[pivot_row][j] = tmp;
      }
    }

    /* Normalize pivot row */
    pivot = aug[i][i];
    for (PetscInt j = 0; j < 2 * N; j++)
      aug[i][j] /= pivot;

    /* Eliminate other rows */
    for (PetscInt k = 0; k < N; k++) {
      if (k == i)
        continue;
      factor = aug[k][i];
      for (PetscInt j = 0; j < 2 * N; j++)
        aug[k][j] -= factor * aug[i][j];
    }
  }

  /* Extract inverse */
  for (PetscInt i = 0; i < N; i++) {
    for (PetscInt j = 0; j < N; j++) {
      invA[i * N + j] = aug[i][j + N];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Converts Cartesian coordinates to volumetric (barycentric) coordinates in a tetrahedron.
 *
 * This function computes the barycentric coordinates \(L = [L_0, L_1, L_2, L_3]\) of
 * a point `r` within the reference tetrahedral element. The transformation solves
 * the linear system `M * L = [x, y, z, 1]`, where `M` contains the reference tetrahedron
 * vertices and `L` represents the volumetric coordinates.
 *
 * @param[in]  r  Array of length NUM_DIMENSIONS containing the Cartesian coordinates [x, y, z].
 * @param[out] L  Array of length 4 where the computed barycentric coordinates will be stored.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or an error code otherwise.
 *
 * @note The function uses `invertMatrix` to invert the 4×4 reference matrix.
 * @note The input point `r` is assumed to lie inside or near the reference tetrahedron.
 * @note The reference tetrahedron vertices are taken from the `REFERENCE_CELL` global array.
 */
PetscErrorCode cartesianToVolumetricCoordinates(const PetscReal r[NUM_DIMENSIONS], PetscReal L[4]) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscReal M[16], invM[16];
  const PetscReal rhs[4] = {r[0], r[1], r[2], 1.0};

  /* Build reference matrix M (column-major) */
  for (PetscInt i = 0; i < NUM_VERTICES_PER_CELL; i++) {
    M[i + 0 * 4] = REFERENCE_CELL[0][i];
    M[i + 1 * 4] = REFERENCE_CELL[1][i];
    M[i + 2 * 4] = REFERENCE_CELL[2][i];
    M[i + 3 * 4] = 1.0;
  }

  /* Invert M */
  PetscCall(invertMatrix(4, M, invM));

  /* Compute L = invM * rhs */
  for (PetscInt i = 0; i < 4; i++) {
    L[i] = 0.0;
    for (PetscInt j = 0; j < 4; j++) {
      L[i] += invM[i * 4 + j] * rhs[j];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Solves a small linear system A * X = B with A 3×3 and B 3×6 using PETSc dense matrices.
 *
 * This function computes the solution of the linear system
 * \f$A \cdot X = B\f$, where `A` is a 3×3 matrix, `B` is a 3×6 matrix,
 * and `X` is the unknown 3×6 matrix to solve for. The computation
 * is performed using PETSc dense matrices with LU factorization of `A`.
 *
 * @param[in]  matrix1  3×3 array representing the matrix `A`.
 * @param[in]  matrix2  3×6 array representing the right-hand side matrix `B`.
 * @param[out] matrix3  3×6 array where the solution matrix `X` will be stored.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 *
 * @note The function internally converts arrays to PETSc dense matrices in column-major order,
 *       performs LU factorization on `A`, solves for `X = A^{-1} B`, and then copies the
 *       result back to `matrix3` in row-major order.
 * @note Intended for small matrices (3×3 system with 3×6 RHS). For larger systems,
 *       other PETSc solvers should be used.
 */
PetscErrorCode solve3x3MatrixSystem3x6RHS(const PetscReal matrix1[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal** matrix2,
                                                 PetscReal** matrix3) {
  PetscFunctionBeginUser;

  /* Variable declarations */
  PetscScalar matrix1_data[NUM_DIMENSIONS * NUM_DIMENSIONS];
  PetscScalar matrix2_data[NUM_DIMENSIONS * (NUM_DIMENSIONS * 2)];
  Mat A, B, C;
  IS row, col;
  PetscScalar* coef_array;

  /* Flatten input arrays (column major order for dense
   * PETSc matrices) */
  for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
      matrix1_data[j * NUM_DIMENSIONS + i] = matrix1[i][j];
    }
  }

  for (PetscInt j = 0; j < NUM_DIMENSIONS * 2; j++) {
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
      matrix2_data[j * NUM_DIMENSIONS + i] = matrix2[i][j];
    }
  }

  /* Create PETSc dense matrices corresponding to A (3x3)
   * and B (3x6) */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, NUM_DIMENSIONS, NUM_DIMENSIONS, matrix1_data, &A));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, NUM_DIMENSIONS, NUM_DIMENSIONS * 2, matrix2_data, &B));
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &C));

  /* Compute index sets for LU factorization */
  PetscCall(ISCreateStride(PETSC_COMM_SELF, NUM_DIMENSIONS, 0, 1, &row));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, NUM_DIMENSIONS, 0, 1, &col));

  /* Factorize A (LU decomposition) */
  PetscCall(MatLUFactor(A, row, col, NULL));

  /* Solve A * C = B → C = A \ B */
  PetscCall(MatMatSolve(A, B, C));

  /* Extract results from C */
  PetscCall(MatDenseGetArray(C, &coef_array));

  /* Copy results back to matrix3 in row-major order */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < NUM_DIMENSIONS * 2; j++) {
      matrix3[i][j] = coef_array[j * NUM_DIMENSIONS + i];
    }
  }

  PetscCall(MatDenseRestoreArray(C, &coef_array));

  /* Free memory */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(ISDestroy(&col));
  PetscCall(ISDestroy(&row));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes shifted and scaled Legendre polynomials P_i(y) for i = 0..nord.
 *
 * This function evaluates the Legendre polynomials using the standard three-term
 * recurrence relation, but adapted for the shifted and scaled variable
 * \f$y = 2 X - T\f$. It returns the polynomial values from order 0 up to `nord`.
 *
 * @param[in]  X    Input coordinate, typically the edge projection s1 in PETGEM.
 * @param[in]  T    Scaling parameter, typically s0 + s1 from an oriented edge projection.
 * @param[in]  nord Maximum polynomial order to compute (produces P[0] to P[nord]).
 * @param[out] P    Array of length (nord + 1) to store computed polynomial values.
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @note The polynomials satisfy the recurrence:
 *       \f[
 *       P_0 = 1, \quad
 *       P_1 = y, \quad
 *       P_{i+1} = \frac{(2i+1) y P_i - i T^2 P_{i-1}}{i+1}, \quad i \ge 1
 *       \f]
 *       where \f$y = 2 X - T\f$.
 */
PetscErrorCode PolyLegendre(const PetscReal X, const PetscReal T, const PetscInt nord, PetscReal P[]) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscReal y;

  /* i stands for the order of the polynomial, stored in P(i) lowest order case (order 0) */
  P[0] = 1.;

  /* First order case (order 1) if necessary */
  y = 2.0 * X - T;
  if (nord >= 1) {
    P[1] = y;
  }

  if (nord >= 2) {
    PetscReal tt = T * T;
    for (PetscInt i = 1; i < nord; i++) {
      P[i + 1] = (2.0 * i + 1.0) * y * P[i] - i * tt * P[i - 1];
      P[i + 1] /= (PetscReal)(i + 1.0);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes integrated shifted and scaled Legendre polynomials and related derivative terms.
 *
 * This function evaluates:
 * - Integrated Legendre polynomials \(L_i\) of order i = 1..nord,
 * - The corresponding Legendre polynomials \(P_i\) (derivative w.r.t. X),
 * - Terms \(R_i\) related to derivatives w.r.t. the scaling parameter T.
 *
 * The computation is based on the shifted variable \(y = 2 X - T\) and optionally
 * simplified when T = 1 (`Idec = PETSC_TRUE`), skipping the computation of R.
 *
 * @param[in]  X     Input coordinate, typically s1 from an oriented edge projection.
 * @param[in]  T     Scaling parameter, typically s0 + s1.
 * @param[in]  nord  Maximum polynomial order required.
 * @param[in]  Idec  Boolean flag indicating simplified case T = 1.
 * @param[out] homL  Array of length `nord+1` storing the integrated polynomial values L_i.
 * @param[out] homP  Array of length `nord+1` storing the Legendre polynomials P_i (derivative w.r.t. X).
 * @param[out] homR  Array of length `nord+1` storing terms related to the derivative w.r.t. T.
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @note Calls `PolyLegendre` internally to compute base polynomial values.
 * @note When `Idec = PETSC_TRUE`, the computation of `homR` is skipped.
 * @note Designed for use in high-order finite element edge basis computations in PETGEM.
 */
static PetscErrorCode PolyILegendre(const PetscReal X, const PetscReal T, const PetscInt nord, const PetscBool Idec, PetscReal homL[],
                                    PetscReal homP[], PetscReal homR[]) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscReal* tmp;
  PetscReal tt, ifact;

  /* Allocate array */
  PetscCall(PetscCalloc1(nord + 1, &tmp));

  /* Calling Legendre for required information */
  PetscCall(PolyLegendre(X, T, nord, tmp));

  for (PetscInt i = 1; i < nord; i++) {
    homP[i - 1] = tmp[i];
  }

  /* Integrated polynomial of order i is stored in L(i) */
  tt = T * T;

  /* Simplified case: no need to compute R */
  if (Idec) {
    for (PetscInt i = 1; i < nord; i++) {
      ifact = 4.0 * (i + 1) - 2.0;
      homL[i] = (tmp[i] - tt * tmp[i - 1]) / ifact;
    }
  } else {
    for (PetscInt i = 1; i < nord; i++) {
      ifact = 4.0 * (i + 1) - 2.0;
      homL[i - 1] = (tmp[i + 1] - tt * tmp[i - 1]) / ifact;
      homR[i - 1] = -(tmp[i] + T * tmp[i - 1]) / 2;
    }
  }

  /* Free memory */
  PetscCall(PetscFree(tmp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes Jacobi polynomials \(P_j^{\alpha,0}\) for shifted and scaled coordinates.
 *
 * Evaluates Jacobi polynomials using a recurrence relation adapted for the shifted variable
 * \(y = 2 X - T\), suitable for constructing high-order basis functions on faces and volumes.
 *
 * @param[in]  X        Coordinate, typically s1 from an oriented face projection.
 * @param[in]  T        Scaling parameter, typically s0 + s1 from an oriented face projection.
 * @param[in]  nord     Maximum polynomial order j required (computes orders 0 to nord).
 * @param[in]  Minalpha Starting alpha value; successive polynomial families increment alpha by 2.
 * @param[out] P        2D array P[family_index][order_j] to store the computed polynomial values.
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @details The first dimension of the output array @p P corresponds to different alpha
 *          values (Minalpha, Minalpha+2, …), while the second dimension corresponds to
 *          the polynomial order j (0..nord). The function initializes P_0 = 1 and P_1 using
 *          the shifted coordinate, then fills higher-order polynomials using the three-term
 *          recurrence relation.
 *
 *          Typically used in PETGEM for constructing H(curl) and H1 face/volume basis functions.
 */
static PetscErrorCode PolyJacobi(const PetscReal X, const PetscReal T, const PetscInt nord, PetscInt Minalpha, PetscReal** P) {

  PetscFunctionBeginUser;

  PetscReal *alpha, y;
  PetscInt minI = 0;
  PetscInt maxI = minI + nord;

  /* Allocate */
  PetscCall(PetscCalloc1(nord + 1, &alpha));

  /* Clearly (minI,maxI)=(0,nord), but the syntax is written as it is
     because it reflects how the indexing is called from outside */
  for (PetscInt i = 0; i < maxI + 1; i++) {
    alpha[i] = Minalpha + 2 * (i - minI);
  }

  /* Initiate first column (order 0) */
  for (PetscInt i = minI; i < maxI + 1; i++) {
    P[i][0] = 1.;
  }

  /* Initiate second column (order 1) if necessary */
  y = 2 * X - T;
  if (nord >= 1) {
    for (PetscInt i = minI; i < maxI; i++) {
      P[i][1] = y + alpha[i] * X;
    }
  }

  /* Fill the last columns if necessary */
  if (nord >= 2) {
    PetscReal tt = pow(T, 2);
    PetscInt ni = -1;
    for (PetscInt i = 0; i < maxI - 1; i++) {
      PetscReal al = alpha[i];
      PetscReal aa = pow(al, 2);
      ni += 1;
      /* Use recursion in order, i, to compute P^alpha_i for i>=2 */
      for (PetscInt j = 2; j < nord - ni + 1; j++) {
        PetscReal ai = 2 * j * (j + al) * (2 * j + al - 2);
        PetscReal bi = 2 * j + al - 1;
        PetscReal ci = (2 * j + al) * (2 * j + al - 2);
        PetscReal di = 2 * (j + al - 1) * (j - 1) * (2 * j + al);

        P[i][j] = bi * (ci * y + aa * T) * P[i][j - 1] - di * tt * P[i][j - 2];
        P[i][j] = P[i][j] / ai;
      }
    }
  }

  /* Free memory */
  PetscCall(PetscFree(alpha));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes integrated shifted scaled Jacobi polynomials \(L_j^{\alpha,0}\) and related terms P, R.
 *
 * Evaluates integrated Jacobi polynomials for shifted coordinates \(y = 2X - T\) using a recurrence
 * relation, along with the base Jacobi polynomials P (derivative w.r.t. X) and the T-derivative
 * related term R. Typically used for high-order basis function construction on faces and volumes.
 *
 * @param[in]  X        Coordinate, typically s1 or s2 depending on context (face/volume basis).
 * @param[in]  T        Scaling parameter, typically s0+s1 or 1-sd depending on context.
 * @param[in]  nord     Maximum polynomial order required (computes orders 0..nord-1).
 * @param[in]  Minalpha Starting alpha value for Jacobi families.
 * @param[out] L        2D array storing integrated polynomial values \(L_j^{\alpha,0}\).
 * @param[out] P        2D array storing Jacobi polynomials \(P_j^{\alpha,0}\) (derivative w.r.t. X).
 * @param[out] R        2D array storing terms related to the derivative w.r.t. T.
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @details First calls `PolyJacobi` to compute the base polynomials P. Then uses recurrence relations
 *          to construct the integrated polynomials L and the T-derivative related term R.
 *          The first column of L corresponds to order 1, while higher orders are computed via the
 *          recurrence. R is initialized to zero and filled where necessary.
 *
 *          Used internally in PETGEM for constructing high-order face and volume basis functions
 *          in H(curl) and H1 spaces.
 */
static PetscErrorCode PolyIJacobi(const PetscReal X, const PetscReal T, const PetscInt nord, const PetscInt Minalpha, PetscReal** L,
                                  PetscReal** P, PetscReal** R) {

  PetscFunctionBeginUser;

  /* Clearly (minI,maxI)=(1,nord), but the syntax is written as it is
     because it reflects how the indexing is called from outside */
  PetscInt minI = 0;
  PetscInt maxI = minI + nord;
  PetscReal* alpha;
  PetscReal** ptemp;

  /* Allocate */
  PetscCall(PetscCalloc1(nord, &alpha));

  PetscCall(PetscCalloc1(nord + 1, &ptemp));
  for (PetscInt i = 0; i < nord + 1; i++) {
    PetscCall(PetscCalloc1(nord + 1, &ptemp[i]));
  }

  PetscCall(PolyJacobi(X, T, nord, Minalpha, ptemp));

  /* Define P. Note that even though P is defined at all entries,
     because of the way Jacobi computes ptemp, only the necessary entries,
     and those on the first subdiagonal (which are never used later)
     are actually accurate.*/
  for (PetscInt i = minI; i < maxI; i++) {
    for (PetscInt j = 0; j < nord; j++) {
      P[i][j] = ptemp[i][j];
    }
  }

  /* Create vector alpha first */
  for (PetscInt i = 0; i < maxI; i++) {
    alpha[i] = Minalpha + 2 * (i - minI);
  }

  /* Initiate first column (order 1 in L) */
  for (PetscInt i = minI; i < maxI; i++) {
    L[i][0] = X;
  }

  /* General case; compute R */
  for (PetscInt i = minI; i < maxI; i++) {
    for (PetscInt j = 0; j < nord; j++) {
      R[i][j] = 0;
    }
  }

  /* Fill the last columns if necessary */
  if (nord >= 2) {
    PetscReal tt = pow(T, 2);
    PetscInt ni = -1;

    for (PetscInt i = 0; i < maxI - 1; i++) {
      PetscReal al = alpha[i];
      ni += 1;

      for (PetscInt j = 2; j < nord - ni + 1; j++) {
        PetscReal tia = j + j + al;
        PetscReal tiam1 = tia - 1;
        PetscReal tiam2 = tia - 2;
        PetscReal ai = (j + al) / (tiam1 * tia);
        PetscReal bi = (al) / (tiam2 * tia);
        PetscReal ci = (j - 1) / (tiam2 * tiam1);

        L[i][j - 1] = ai * ptemp[i][j] + bi * T * ptemp[i][j - 1] - ci * tt * ptemp[i][j - 2];
        R[i][j - 1] = -(j - 1) * (ptemp[i][j - 1] + T * ptemp[i][j - 2]);
        R[i][j - 1] = R[i][j - 1] / tiam2;
      }
    }
  }

  /* Free memory */
  PetscCall(PetscFree(alpha));

  for (PetscInt i = 0; i < nord + 1; i++) {
    PetscCall(PetscFree(ptemp[i]));
  }
  PetscCall(PetscFree(ptemp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the barycentric (affine) coordinates and their gradients for a reference tetrahedron.
 *
 * Given a point X = [x, y, z] in the reference tetrahedron, computes:
 * - The barycentric coordinates \(\lambda_i\) stored in Lam (i = 0..3),
 *   corresponding to the vertices of the tetrahedron.
 * - The gradients of the barycentric coordinates with respect to X,
 *   stored in DLam (3x4 matrix), where each column corresponds to a vertex.
 *
 * The reference tetrahedron is defined with vertices at:
 *   (0,0,0), (1,0,0), (0,1,0), (0,0,1)
 *
 * @param[in]  X    Array of coordinates [x, y, z] inside the reference tetrahedron.
 * @param[out] Lam  Array of size 4 storing the barycentric coordinates \(\lambda_0,\dots,\lambda_3\).
 * @param[out] DLam 3x4 array storing the gradients of Lam w.r.t X.
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
PetscErrorCode AffineTetrahedron(const PetscReal X[NUM_DIMENSIONS], PetscReal Lam[NUM_DIMENSIONS + 1],
                                        PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1]) {

  PetscFunctionBeginUser;

  /* Define affine coordinates */
  Lam[0] = 1. - X[0] - X[1] - X[2];
  Lam[1] = X[0];
  Lam[2] = X[1];
  Lam[3] = X[2];

  /* Define gradients */
  DLam[0][0] = -1;
  DLam[0][1] = 1;
  DLam[1][0] = -1;
  DLam[1][2] = 1;
  DLam[2][0] = -1;
  DLam[2][3] = 1;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the standard H1 nodal basis functions and their gradients for a tetrahedron.
 *
 * For a P1 tetrahedral element, the H1 nodal basis functions are simply the barycentric coordinates.
 * This function copies the barycentric coordinates Lam[i] to the output LambV[i], and their gradients
 * DLam[:,i] to DLambV[i][:], providing the values and gradients of the 4 nodal (vertex) basis functions.
 *
 * @param[in]  Lam     Array of 4 barycentric coordinates [L0, L1, L2, L3] at the point.
 * @param[in]  DLam    3x4 array of gradients of the barycentric coordinates w.r.t [x, y, z].
 * @param[out] LambV   Array of size 4 to store the nodal basis function values (LambV[i] = Lam[i]).
 * @param[out] DLambV  4x3 array to store the gradients of the nodal basis functions (DLambV[i] = Grad(Lam[i])).
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @note This function is specific to linear tetrahedral (P1) elements.
 *       The mapping is direct: vertex i → basis function i.
 */
static PetscErrorCode BlendTetV(const PetscReal Lam[NUM_DIMENSIONS + 1], const PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1],
                                PetscReal LambV[NUM_VERTICES_PER_CELL], PetscReal DLambV[NUM_VERTICES_PER_CELL][NUM_DIMENSIONS]) {

  PetscFunctionBeginUser;

  /* Variable declaration */
  PetscInt v;

  /* 4 vertices, each with one blending function */

  /* v=1 --> v0=(0,0,0) */
  v = 0;
  LambV[v] = Lam[0];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++)
    DLambV[v][i] = DLam[i][0];

  /* v=2 --> v1=(1,0,0) */
  v = 1;
  LambV[v] = Lam[1];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++)
    DLambV[v][i] = DLam[i][1];

  /* v=3 --> v2=(0,1,0) */
  v = 2;
  LambV[v] = Lam[2];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++)
    DLambV[v][i] = DLam[i][2];

  /* v=4 --> v3=(0,0,1) */
  v = 3;
  LambV[v] = Lam[3];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++)
    DLambV[v][i] = DLam[i][3];

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Projects barycentric coordinates and gradients onto the 6 edges of a tetrahedron.
 *
 * For each tetrahedral edge, this function selects the two barycentric coordinates corresponding
 * to the edge's vertices and copies their gradients. This projection is used for constructing
 * edge-based H(curl) basis functions.
 *
 * @param[in]  Lam     Array of 4 barycentric coordinates [L0, L1, L2, L3].
 * @param[in]  DLam    3x4 array of gradients of barycentric coordinates w.r.t. [x, y, z].
 * @param[out] LampE   6x2 array storing the projected barycentric coordinates for each edge.
 *                     LampE[edge_index][0] → first vertex, LampE[edge_index][1] → second vertex.
 * @param[out] DLampE  6x3x2 array storing the projected gradients.
 *                     DLampE[edge_index][dim][0/1] corresponds to the gradient of the first/second vertex along dim.
 * @param[out] IdecE   Boolean flag indicating if the projected coordinates sum to 1 (always PETSC_FALSE for edges).
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @note The local orientation of each edge is fixed as in PETGEM convention:
 *       e0: v1->v0, e1: v0->v2, e2: v2->v1, e3: v1->v3, e4: v3->v0, e5: v2->v3.
 */
PetscErrorCode ProjectTetE(const PetscReal Lam[NUM_DIMENSIONS + 1], const PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1],
                                  PetscReal LampE[NUM_EDGES_PER_CELL][2], PetscReal DLampE[NUM_EDGES_PER_CELL][NUM_DIMENSIONS][2],
                                  PetscBool* IdecE) {

  PetscFunctionBeginUser;

  /* Edge slot ↔ vertex pair must match the PETGEM canonical labeling in
   * EDGE_VERTICES (constants.c) — that is the convention used by the cell
   * closure traversal, the global DOF assignment, and the Block-A debug
   * dump. Any mismatch here produces Whitney functions for the wrong
   * physical edge at each slot.
   *
   * Slot layout (EDGE_VERTICES):
   *   E0: v0->v1   E1: v1->v2   E2: v2->v0
   *   E3: v0->v3   E4: v3->v1   E5: v2->v3
   *
   * For each slot e the i-th component is Lam[v_i] and DLam[*][v_i] for
   * v_i = EDGE_VERTICES[e][i]. AncEE then forms the Whitney
   *   w = Lam[v_0] · ∇Lam[v_1] − Lam[v_1] · ∇Lam[v_0]
   * which is the standard tangentially-oriented edge basis aligned with
   * the v_0 → v_1 direction.
   *
   * Historical note: an earlier convention (v1->v0, v0->v2, v2->v1, ...)
   * silently mismatched EDGE_VERTICES in 5 of 6 slots. It was invisible
   * for shape3DHTet because the CSEM solve uses H1=P1 only (no edge
   * bubbles); the hierarchical H(curl) basis exposes it on every cell. */
  for (PetscInt e = 0; e < NUM_EDGES_PER_CELL; e++) {
    const PetscInt va = EDGE_VERTICES[e][0];
    const PetscInt vb = EDGE_VERTICES[e][1];
    LampE[e][0] = Lam[va];
    LampE[e][1] = Lam[vb];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
      DLampE[e][i][0] = DLam[i][va];
      DLampE[e][i][1] = DLam[i][vb];
    }
  }

  /* Projected coordinates are Lam, so IdecE=false for all edges */
  *IdecE = PETSC_FALSE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Projects barycentric coordinates and gradients onto the 4 faces of a tetrahedron.
 *
 * For each tetrahedral face, this function selects the three barycentric coordinates corresponding
 * to the face's vertices and copies their gradients. This projection is used for constructing
 * face-based H(div) basis functions.
 *
 * @param[in]  Lam     Array of 4 barycentric coordinates [L0, L1, L2, L3].
 * @param[in]  DLam    3x4 array of gradients of barycentric coordinates w.r.t. [x, y, z].
 * @param[out] LampF   4x3 array storing the projected barycentric coordinates for each face.
 *                     LampF[face_index][0/1/2] → the three vertices of the face.
 * @param[out] DLampF  4x3x3 array storing the projected gradients.
 *                     DLampF[face_index][dim][0/1/2] corresponds to the gradient of the vertex along dim.
 * @param[out] IdecF   Boolean flag indicating if the projected coordinates sum to 1 (always PETSC_FALSE for faces).
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @note Local face orientation is fixed as:
 *       f0: v1->v0->v2, f1: v1->v3->v0, f2: v1->v2->v3, f3: v2->v0->v3.
 *       This ensures consistent face-based basis construction.
 */
PetscErrorCode ProjectTetF(const PetscReal Lam[NUM_DIMENSIONS + 1], const PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1],
                                  PetscReal LampF[NUM_FACES_PER_CELL][NUM_DIMENSIONS],
                                  PetscReal DLampF[NUM_FACES_PER_CELL][NUM_DIMENSIONS][NUM_DIMENSIONS], PetscBool* IdecF) {

  PetscFunctionBeginUser;

  /* Face slot ↔ vertex triple must match the PETGEM canonical labeling in
   * FACE_VERTICES (constants.c) — same convention used by the cell closure
   * traversal, the global face-DOF assignment, and the OrientTri permutation
   * codes 0..5 produced by computeCellOrientation.
   *
   * Slot layout (FACE_VERTICES):
   *   F0: {v0, v1, v2}    F1: {v0, v1, v3}
   *   F2: {v0, v2, v3}    F3: {v1, v2, v3}
   *
   * For each slot f the i-th component is Lam[v_i] for v_i = FACE_VERTICES[f][i].
   * The hierarchical face-bubble construction (AncETri / OrientTri) treats
   * NoriF=0 as "vertices in canonical order" — so canonical here MUST be
   * FACE_VERTICES, otherwise NoriF=0 silently means "some other permutation"
   * and adjacent cells disagree on face-DOF orientation.
   *
   * Historical note: an earlier convention (v1->v0->v2, v1->v3->v0,
   * v1->v2->v3, v2->v0->v3) silently mismatched FACE_VERTICES — slots 2 and
   * 3 referenced different physical faces, slots 0 and 1 used permuted
   * vertices. Invisible at nord=1 (no face DOFs); breaks nord >= 2. */
  for (PetscInt f = 0; f < NUM_FACES_PER_CELL; f++) {
    const PetscInt v0 = FACE_VERTICES[f][0];
    const PetscInt v1 = FACE_VERTICES[f][1];
    const PetscInt v2 = FACE_VERTICES[f][2];
    LampF[f][0] = Lam[v0];
    LampF[f][1] = Lam[v1];
    LampF[f][2] = Lam[v2];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
      DLampF[f][i][0] = DLam[i][v0];
      DLampF[f][i][1] = DLam[i][v1];
      DLampF[f][i][2] = DLam[i][v2];
    }
  }

  *IdecF = PETSC_FALSE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Orients edge-projected coordinates and gradients according to edge orientation.
 *
 * Each tetrahedron edge has a local orientation. This function ensures that the
 * projected coordinates `S` and gradients `DS` are correctly ordered along the
 * global orientation of the edge as indicated by `Nori`.
 *
 * @param[in]  S    Array of 2 projected edge coordinates [s0, s1].
 * @param[in]  DS   3x2 array of gradients corresponding to S: DS[dim][0/1] = Grad(s0/s1).
 * @param[in]  Nori Edge orientation flag: 0 = original order, 1 = swap s0 and s1.
 * @param[out] GS   Oriented coordinates [gs0, gs1] after applying Nori.
 * @param[out] GDS  Oriented gradients: GDS[dim][0/1] corresponds to Grad(gs0/gs1).
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @details If Nori == 0, GS = S and GDS = DS (no change).
 *          If Nori == 1, GS = [s1, s0] and GDS = [Grad(s1), Grad(s0)].
 *          This is used to ensure consistent local-to-global edge orientation
 *          when assembling edge-based basis functions.
 */
PetscErrorCode OrientE(const PetscReal S[2], const PetscReal DS[NUM_DIMENSIONS][2], const PetscInt Nori, PetscReal GS[2],
                              PetscReal GDS[NUM_DIMENSIONS][2]) {

  PetscFunctionBeginUser;

  PetscInt Or[2][2];

  /* Nori=0 => (s0,s1)->(s0,s1) */
  Or[0][0] = 0;
  Or[0][1] = 1;

  /* Nori=1 => (s0,s1)->(s1,s0) */
  Or[1][0] = 1;
  Or[1][1] = 0;

  /* Local-to-global transformation */
  GS[0] = S[Or[Nori][0]];
  GS[1] = S[Or[Nori][1]];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    GDS[i][0] = DS[i][Or[Nori][0]];
    GDS[i][1] = DS[i][Or[Nori][1]];
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Orients face-projected coordinates and gradients according to face orientation.
 *
 * Each triangular face of a tetrahedron can have 6 possible orientations (permutations of its
 * vertices). This function reorders the projected coordinates `S` and gradients `DS` to
 * match the global orientation indicated by `Nori`.
 *
 * @param[in]  S    Array of 3 projected face coordinates [s0, s1, s2].
 * @param[in]  DS   3x3 array of gradients corresponding to S: DS[dim][0/1/2] = Grad(s0/s1/s2).
 * @param[in]  Nori Face orientation flag (0-5) representing one of the 6 permutations of the triangle vertices.
 * @param[out] GS   Oriented coordinates [gs0, gs1, gs2] after applying Nori.
 * @param[out] GDS  Oriented gradients: GDS[dim][0/1/2] corresponds to Grad(gs0/gs1/gs2).
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @details The permutation table `Or` maps local indices to global indices according to `Nori`.
 *          For example:
 *            - Nori = 0 → original order (s0, s1, s2)
 *            - Nori = 1 → cyclic permutation (s1, s2, s0)
 *            - Nori = 2 → cyclic permutation (s2, s0, s1)
 *            - Nori = 3 → swap last two vertices (s0, s2, s1)
 *            - Nori = 4 → swap first two vertices (s1, s0, s2)
 *            - Nori = 5 → reverse order (s2, s1, s0)
 *
 *          This ensures that face-based basis functions are consistently oriented
 *          across tetrahedral elements when assembling global matrices.
 */
PetscErrorCode OrientTri(const PetscReal S[NUM_DIMENSIONS], const PetscReal DS[NUM_DIMENSIONS][NUM_DIMENSIONS], const PetscInt Nori,
                                PetscReal GS[NUM_DIMENSIONS], PetscReal GDS[NUM_DIMENSIONS][NUM_DIMENSIONS]) {

  PetscFunctionBeginUser;

  PetscInt Or[NUM_DIMENSIONS * 2][NUM_DIMENSIONS];

  /* Nori=0 => (s0,s1,s2)->(s0,s1,s2) */
  Or[0][0] = 0;
  Or[0][1] = 1;
  Or[0][2] = 2;

  /* Nori=1 => (s0,s1,s2)->(s1,s2,s0) */
  Or[1][0] = 1;
  Or[1][1] = 2;
  Or[1][2] = 0;

  /* Nori=2 => (s0,s1,s2)->(s2,s0,s1) */
  Or[2][0] = 2;
  Or[2][1] = 0;
  Or[2][2] = 1;

  /* Nori=3 => (s0,s1,s2)->(s0,s2,s1) */
  Or[3][0] = 0;
  Or[3][1] = 2;
  Or[3][2] = 1;

  /* Nori=4 => (s0,s1,s2)->(s1,s0,s2) */
  Or[4][0] = 1;
  Or[4][1] = 0;
  Or[4][2] = 2;

  /* Nori=5 => (s0,s1,s2)->(s2,s1,s0) */
  Or[5][0] = 2;
  Or[5][1] = 1;
  Or[5][2] = 0;

  /* Local-to-global transformation */
  GS[0] = S[Or[Nori][0]];
  GS[1] = S[Or[Nori][1]];
  GS[2] = S[Or[Nori][2]];

  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    GDS[i][0] = DS[i][Or[Nori][0]];
    GDS[i][1] = DS[i][Or[Nori][1]];
    GDS[i][2] = DS[i][Or[Nori][2]];
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes homogenized integrated Legendre polynomials and their gradients along a tetrahedral edge.
 *
 * This function constructs edge-based basis functions in a homogenized form:
 * it evaluates the integrated Legendre polynomials \(L_i(s_1, s_0 + s_1)\) and their derivatives
 * with respect to the global coordinates using the chain rule applied to the projected affine-like coordinates.
 *
 * @param[in]  S     Affine-like coordinates along the edge [s0, s1].
 * @param[in]  DS    Gradients of S w.r.t. global coordinates: DS[dim][0/1] = Grad(s0/s1).
 * @param[in]  nord  Maximum polynomial order required.
 * @param[in]  Idec  Boolean flag indicating a simplified case: TRUE if s0 + s1 = 1.
 * @param[out] PhiE  Array of homogenized integrated polynomial values along the edge. Size = nord - 1.
 * @param[out] DPhiE Array of gradients of PhiE: DPhiE[dim][order_idx].
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @details If Idec = TRUE, the sum s0+s1 = 1 and only derivatives w.r.t s1 are needed.
 *          Otherwise, derivatives w.r.t both s0 and s1 are included using the auxiliary term `homR`.
 *          The function internally calls `PolyILegendre` to compute L_i, P_i, and R_i and then maps
 *          them to global gradients via the chain rule:
 *            - DPhiE[j][i] = homP[i] * Grad(s1)          (if Idec = TRUE)
 *            - DPhiE[j][i] = homP[i] * Grad(s1) + homR[i] * (Grad(s0) + Grad(s1))  (otherwise)
 *
 *          `PhiE` contains the integrated polynomial values L_i(s1, s0+s1).
 *          The gradient array `DPhiE` is ready to use for finite element assembly.
 */
static PetscErrorCode HomILegendre(const PetscReal S[2], const PetscReal DS[NUM_DIMENSIONS][2], const PetscInt nord, PetscBool const Idec,
                                   PetscReal* PhiE, PetscReal** DPhiE) {
  PetscFunctionBeginUser;

  /* Variable declaration */
  PetscReal *homL, *homP, *homR;
  PetscReal DS01[NUM_DIMENSIONS];

  /* Allocate arrays */
  PetscCall(PetscCalloc1(nord - 1, &homL));
  PetscCall(PetscCalloc1(nord - 1, &homP));
  PetscCall(PetscCalloc1(nord - 1, &homR));

  /* Idec is the flag to compute x and t derivatives. If sum of S equal 1 -> Idec=TRUE */
  if (Idec) {
    PetscCall(PolyILegendre(S[1], 1.0, nord, Idec, homL, homP, homR));
    for (PetscInt i = 1; i < nord; i++) {
      for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
        DPhiE[j][i] = homP[i - 1] * DS[j][0];
      }
    }
  } else {
    PetscCall(PolyILegendre(S[1], S[0] + S[1], nord, Idec, homL, homP, homR));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
      DS01[i] = DS[i][0] + DS[i][1];
    }

    for (PetscInt i = 1; i < nord; i++) {
      for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
        DPhiE[j][i - 1] = homP[i - 1] * DS[j][1] + homR[i - 1] * DS01[j];
      }
    }
  }

  /* Prepare output for PhiE */
  for (PetscInt i = 0; i < nord - 1; i++) {
    PhiE[i] = homL[i];
  }

  /* Free memory */
  PetscCall(PetscFree(homL));
  PetscCall(PetscFree(homP));
  PetscCall(PetscFree(homR));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes homogenized integrated Jacobi polynomials and their gradients along a face or volume.
 *
 * This function constructs face- or volume-based homogenized basis functions by computing
 * integrated Jacobi polynomials \(L_j^{\alpha,0}\) and their gradients with respect to global coordinates.
 * It uses the chain rule to map derivatives from affine-like coordinates (S) to the global coordinate system.
 *
 * @param[in]  S       Affine-like coordinates [s_a, s_b], e.g., [s0+s1, s2] for faces.
 * @param[in]  DS      Gradients of S w.r.t. global coordinates: DS[dim][0/1] = Grad(s_a/s_b).
 * @param[in]  nord    Maximum polynomial order required.
 * @param[in]  Minalpha Starting value of alpha (used to generate the Jacobi polynomial family P_j^{alpha,0}).
 * @param[in]  Idec    Boolean flag indicating a simplified case: TRUE if s_a + s_b = 1 (so T=1).
 * @param[out] HomL    2D array storing the homogenized integrated Jacobi polynomial values.
 *                     Dimensions: HomL[alpha_idx][order_idx] (size ~ nord x nord).
 * @param[out] DHomL   3D array storing the gradients of HomL w.r.t. global coordinates.
 *                     Dimensions: DHomL[dim][alpha_idx][order_idx].
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @details The function internally calls `PolyIJacobi` to compute the integrated Jacobi polynomials (HomL),
 *          their derivative w.r.t. the second coordinate (homP), and the derivative w.r.t. the sum (homR).
 *          The global gradients are then computed via the chain rule:
 *            - If Idec = TRUE: DHomL = homP * Grad(s_b)
 *            - Otherwise:     DHomL = homP * Grad(s_b) + homR * Grad(s_a + s_b)
 *
 *          The arrays HomL and DHomL are ready for direct use in finite element assembly
 *          for face- or volume-based high-order basis functions.
 */
PetscErrorCode HomIJacobi(const PetscReal S[2], const PetscReal DS[NUM_DIMENSIONS][2], const PetscInt nord, const PetscInt Minalpha,
                                 const PetscBool Idec, PetscReal** HomL, PetscReal*** DHomL) {

  PetscFunctionBeginUser;

  /* Clearly (minI,maxI)=(1,nord), but the syntax is written as it is
     because it reflects how the indexing is called from outside */
  PetscInt minI = 1;
  PetscInt maxI = minI + nord - 1;

  PetscReal** homP; /* homP[nord][nord] */
  PetscReal** homR; /* homR[nord][nord] */

  /* Allocate */
  PetscCall(PetscCalloc1(nord, &homP));
  for (PetscInt i = 0; i < nord; i++) {
    PetscCall(PetscCalloc1(nord, &homP[i]));
  }

  PetscCall(PetscCalloc1(nord, &homR));
  for (PetscInt i = 0; i < nord; i++) {
    PetscCall(PetscCalloc1(nord, &homR[i]));
  }

  PetscInt ni = -1;

  if (Idec) {
    PetscCall(PolyIJacobi(S[1], 1, nord, Minalpha, HomL, homP, homR));
    for (PetscInt i = minI; i < maxI + 1; i++) {
      ni += 1;
      for (PetscInt j = 1; j < nord - ni + 1; j++) {
        for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
          DHomL[k][i - 1][j - 1] = homP[i - 1][j - 1] * DS[k][1];
        }
      }
    }
  } else {
    /* If sum of S different from 1 -> Idec=.FALSE. */
    PetscCall(PolyIJacobi(S[1], S[0] + S[1], nord, Minalpha, HomL, homP, homR));

    PetscReal DS01[NUM_DIMENSIONS];
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
      DS01[i] = DS[i][0] + DS[i][1];
    }

    for (PetscInt i = minI; i < maxI + 1; i++) {
      ni += 1;
      for (PetscInt j = 1; j < nord - ni + 1; j++) {
        for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
          DHomL[k][i - 1][j - 1] = homP[i - 1][j - 1] * DS[k][1] + homR[i - 1][j - 1] * DS01[k];
        }
      }
    }
  }

  /* Free memory */
  for (PetscInt i = 0; i < nord; i++) {
    PetscCall(PetscFree(homP[i]));
  }
  PetscCall(PetscFree(homP));

  for (PetscInt i = 0; i < nord; i++) {
    PetscCall(PetscFree(homR[i]));
  }
  PetscCall(PetscFree(homR));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes ancillary edge-based basis functions (homogenized integrated Legendre polynomials) and their gradients.
 *
 * This function is a wrapper around `HomILegendre` that constructs the edge-based
 * high-order basis functions used in H(curl) or H(div) elements along edges of a tetrahedron.
 *
 * @param[in]  S       Affine-like coordinates along the edge [s0, s1].
 * @param[in]  DS      Gradients of S w.r.t. global coordinates: DS[dim][0/1] = Grad(s0/s1).
 * @param[in]  nord    Maximum polynomial order required (defines number of basis functions along the edge).
 * @param[in]  Idec    Boolean flag indicating a simplified case: TRUE if s0 + s1 = 1.
 * @param[out] PhiE    Array to store the homogenized integrated Legendre polynomial values.
 *                     Size: nord - 1.
 * @param[out] DPhiE   2D array storing the gradients of PhiE w.r.t. global coordinates.
 *                     Dimensions: DPhiE[dim][nord - 1].
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @details The function simply calls `HomILegendre` with the same arguments.
 *          The output arrays `PhiE` and `DPhiE` are ready for use in constructing
 *          edge-based shape functions in high-order finite element assembly.
 */
static PetscErrorCode AncPhiE(const PetscReal S[2], const PetscReal DS[NUM_DIMENSIONS][2], const PetscInt nord, const PetscBool Idec,
                              PetscReal* PhiE, PetscReal** DPhiE) {
  PetscFunctionBeginUser;

  /* These are precisely the homogenized Legendre polynomials */
  PetscCall(HomILegendre(S, DS, nord, Idec, PhiE, DPhiE));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes H1 ancillary face-based basis functions (triangle) and their gradients.
 *
 * This function constructs high-order H1 basis functions associated with a triangular face
 * of a tetrahedron using a tensor-product of edge-based integrated Legendre polynomials
 * and homogenized integrated Jacobi polynomials. It also computes the gradients of each basis
 * function with respect to global coordinates.
 *
 * @param[in]  S        Oriented face coordinates [s0, s1, s2] corresponding to the three vertices.
 * @param[in]  DS       Oriented face gradients [Grad(s0), Grad(s1), Grad(s2)] with size DS[NUM_DIMENSIONS][3].
 * @param[in]  nordFace Polynomial order of the element along the face.
 * @param[in]  IdecF    Boolean flag indicating if s0 + s1 + s2 = 1 (simplified scaling case).
 * @param[out] PhiTri   2D output array storing scalar values of the face ancillary functions.
 *                      Dimensions: PhiTri[nordFace-2][nordFace-2].
 * @param[out] DPhiTri  3D output array storing gradients of each face ancillary function.
 *                      Dimensions: DPhiTri[NUM_DIMENSIONS][nordFace-2][nordFace-2].
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @details The procedure is as follows:
 * 1. Computes edge-based homogenized integrated Legendre polynomials along the first edge.
 * 2. Computes homogenized integrated Jacobi polynomials along the remaining coordinate.
 * 3. Combines the edge and Jacobi polynomials to form the full face ancillary basis functions.
 * 4. Computes gradients via the chain rule: Grad(Phi) = Grad(edge) * Jacobi + edge * Grad(Jacobi).
 * 5. Allocates temporary arrays for intermediate polynomials and their gradients and frees them before returning.
 *
 * This function is used in the assembly of H1 high-order finite element matrices on tetrahedral meshes.
 */
static PetscErrorCode AncPhiTri(const PetscReal S[NUM_DIMENSIONS], const PetscReal DS[NUM_DIMENSIONS][NUM_DIMENSIONS],
                                const PetscInt nordFace, const PetscBool IdecF, PetscReal** PhiTri, PetscReal*** DPhiTri) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt minI = 2;
  PetscInt minJ = 1;
  PetscInt maxJ = nordFace - 2;
  PetscInt minIJ = minI + minJ;
  PetscInt maxIJ = nordFace;
  PetscInt minalpha = 2 * minI;
  PetscReal GLampE[2] = {0.0};
  PetscReal GDLampE[NUM_DIMENSIONS][2] = {{0.0}};
  PetscReal *PhiE, **DPhiE;
  PetscReal DsL[NUM_DIMENSIONS][2];
  PetscReal sL[2];
  PetscReal** homLal;   /* homLal[maxJ][maxJ] */
  PetscReal*** DhomLal; /* DhomLal[NUM_DIMENSIONS][maxJ][maxJ] */

  /* Allocate arrays */
  PetscCall(PetscCalloc1(nordFace - minJ - 1, &PhiE));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DPhiE));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(nordFace - minJ - 1, &DPhiE[i]));
  }

  PetscCall(PetscCalloc1(maxJ, &homLal));
  for (PetscInt i = 0; i < maxJ; i++) {
    PetscCall(PetscCalloc1(maxJ, &homLal[i]));
  }

  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DhomLal));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(maxJ, &DhomLal[i]));
    for (PetscInt j = 0; j < maxJ; j++) {
      PetscCall(PetscCalloc1(maxJ, &DhomLal[i][j]));
    }
  }

  /* Prepare input */
  GLampE[0] = S[0];
  GLampE[1] = S[1];

  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    GDLampE[i][0] = DS[i][0];
    GDLampE[i][1] = DS[i][1];
  }

  /* Get EE - this is never a simplified case (IdecE=0) */
  PetscCall(AncPhiE((const PetscReal*)GLampE, (const PetscReal(*)[2])GDLampE, nordFace - minJ, IdecF, PhiE, DPhiE));

  /* Get homogenized Integrated Jacobi polynomials, homLal, and gradients */
  sL[0] = S[0] + S[1];
  sL[1] = S[2];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    DsL[i][0] = DS[i][0] + DS[i][1];
    DsL[i][1] = DS[i][2];
  }

  /* Compute HomIJacobi */
  PetscCall(HomIJacobi(sL, (const PetscReal(*)[2])DsL, maxJ, minalpha, IdecF, homLal, DhomLal));

  /* Simply complete the required information */
  for (PetscInt i = minIJ; i <= maxIJ; i++) {
    for (PetscInt j = minI; j <= i - minJ; j++) {
      PetscInt k = i - j;
      PhiTri[j - 2][k - 1] = PhiE[j - 2] * homLal[j - 2][k - 1];
      for (PetscInt l = 0; l < NUM_DIMENSIONS; l++) {
        DPhiTri[l][j - 2][k - 1] = homLal[j - 2][k - 1] * DPhiE[l][j - 2] + PhiE[j - 2] * DhomLal[l][j - 2][k - 1];
      }
    }
  }

  /* Free memory */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscFree(DPhiE[i]));
  }
  PetscCall(PetscFree(DPhiE));
  PetscCall(PetscFree(PhiE));

  for (PetscInt i = 0; i < maxJ; i++) {
    PetscCall(PetscFree(homLal[i]));
  }
  PetscCall(PetscFree(homLal));

  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < maxJ; j++) {
      PetscCall(PetscFree(DhomLal[i][j]));
    }
    PetscCall(PetscFree(DhomLal[i]));
  }
  PetscCall(PetscFree(DhomLal));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes H1 hierarchical high-order basis functions (shape functions) and their gradients for a tetrahedral element.
 *
 * This function constructs high-order H1 finite element shape functions for a tetrahedron of polynomial order `nord`.
 * It includes vertex, edge, face, and interior (bubble) basis functions using a hierarchical approach with
 * homogenized integrated Legendre and Jacobi polynomials. The gradients of each shape function with respect
 * to global coordinates are also computed.
 *
 * @param[in]  X               Coordinates of the point where shape functions are evaluated [x, y, z].
 * @param[in]  nord            Polynomial order of the element.
 * @param[in]  cellOrientation Array of 10 integers specifying local orientations for the 4 faces and 6 edges of the tetrahedron.
 *                             - cellOrientation[0..3]: face orientations
 *                             - cellOrientation[4..9]: edge orientations
 * @param[out] ShapH           Output array storing the scalar values of H1 shape functions at X.
 *                             Dimension: MAXtetraH = (nord+3)*(nord+2)*(nord+1)/6
 * @param[out] GradH           Output 2D array storing gradients of H1 shape functions with respect to global coordinates.
 *                             Dimensions: GradH[NUM_DIMENSIONS][MAXtetraH]
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @details The construction proceeds as follows:
 * 1. Compute barycentric (affine) coordinates of the tetrahedron and their gradients.
 * 2. Evaluate vertex-based shape functions.
 * 3. Project barycentric coordinates onto edges and compute edge-oriented shape functions using AncPhiE.
 * 4. Project barycentric coordinates onto faces and compute face-oriented shape functions using AncPhiTri.
 * 5. Compute interior (bubble) shape functions using AncPhiTri and HomIJacobi for the volume.
 * 6. Gradients are computed using the chain rule from the affine coordinates and polynomial derivatives.
 *
 * The function allocates temporary arrays for intermediate polynomial values and gradients and frees them
 * before returning. Orientation of edges and faces is applied according to `cellOrientation`.
 *
 * This function is suitable for use in assembling high-order H1 finite element matrices over tetrahedral meshes.
 */
PetscErrorCode shape3DHTet(const PetscReal X[NUM_DIMENSIONS], const PetscInt nord, const CellOrientation *cellOrientation,
                                  PetscReal* ShapH, PetscReal** GradH) {
  PetscFunctionBeginUser;

  /* Local parameters */
  PetscBool IdecB[2] = {PETSC_FALSE, PETSC_FALSE};
  PetscInt minI = 1;
  PetscInt minJ = 0;
  PetscInt minK = 0;
  PetscInt minIJ = minI + minJ;
  PetscInt minIJK = minIJ + minK;
  PetscInt m = 0;                     /* Initialize counter for shape functions */
  PetscInt NoriF[NUM_FACES_PER_CELL]; /* Orientation for faces */
  PetscInt NoriE[NUM_EDGES_PER_CELL]; /* Orientation for edges */
  PetscInt MAXtetraH;

  PetscReal Lam[NUM_DIMENSIONS + 1] = {0.0};                    /* Define affine coordinates for tetra */
  PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1] = {{0.0}}; /* Define gradients for tetra */

  PetscReal LambV[NUM_VERTICES_PER_CELL] = {0.0};                    /* Define affine coordinates for vertices */
  PetscReal DLambV[NUM_VERTICES_PER_CELL][NUM_DIMENSIONS] = {{0.0}}; /* Define gradients for vertices */

  PetscBool IdecE; /* Shape functions over edges */
  PetscBool IdecF; /* Shape functions over faces */

  /* Compute maximum number of dofs in H1 */
  MAXtetraH = ((nord + 3) * (nord + 2) * (nord + 1)) / 6;

  /* Reset matrices */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    ShapH[i] = 0.0;
    for (PetscInt j = 0; j < MAXtetraH; j++) {
      GradH[i][j] = 0.0;
    }
  }

  /* Get affine tetrahedron */
  PetscCall(AffineTetrahedron(X, Lam, DLam));

  /* Define affine coordinates and gradients */
  PetscCall(BlendTetV((const PetscReal*)Lam, (const PetscReal(*)[NUM_DIMENSIONS + 1]) DLam, LambV, DLambV));
  for (PetscInt i = 0; i < NUM_VERTICES_PER_CELL; i++) {
    ShapH[m] = LambV[i];
    for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
      GradH[j][m] = DLambV[i][j];
    }
    m += 1;
  }

  /* Arrays for basis functions projection over edges */
  PetscReal LampE[NUM_EDGES_PER_CELL][2];                  /* Shape functions over edges */
  PetscReal DLampE[NUM_EDGES_PER_CELL][NUM_DIMENSIONS][2]; /* Shape functions over edges */

  /* Compute edges projection */
  PetscCall(ProjectTetE((const PetscReal*)Lam, (const PetscReal(*)[NUM_DIMENSIONS + 1]) DLam, LampE, DLampE, &IdecE));

  /* Extract orientation for faces */
  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; ++i) {
    NoriF[i] = cellOrientation->faces[i];
  }

  /* Extract orientation for edges.
   * computeCellOrientation stores edges as sign ±1; OrientE expects
   * a {0,1} index (0 = aligned, 1 = reversed). */
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; ++i) {
    NoriE[i] = (cellOrientation->edgeSigns[i] < 0) ? 1 : 0;
  }

  /* Compute shape functions for edges */
  PetscInt nordEdge = 0;
  PetscInt numDofEdge = 0;
  PetscReal *PhiE, **DPhiE;

  /* Allocate matrices for functions for edges */
  PetscCall(PetscCalloc1(nord - 1, &PhiE));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DPhiE));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(nord - 1, &DPhiE[i]));
  }

  /* Loop over edges */
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; i++) {
    /* Local parameters */
    nordEdge = nord;
    numDofEdge = nordEdge - 1;
    if (numDofEdge > 0) {
      /* Local parameters */
      PetscInt maxI = nordEdge;
      /* Orient */
      PetscReal GLampE[2] = {0.0};
      PetscReal GDLampE[NUM_DIMENSIONS][2] = {{0.0}};
      PetscReal S[2] = {0.0};
      PetscReal D[NUM_DIMENSIONS][2] = {{0.0}};

      S[0] = LampE[i][0];
      S[1] = LampE[i][1];

      /* Extract the slice into D */
      for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
        for (PetscInt k = 0; k < 2; k++) {
          D[j][k] = DLampE[i][j][k];
        }
      }

      /* Compute edge orientation */
      PetscCall(OrientE(S, (const PetscReal(*)[2])D, NoriE[i], GLampE, GDLampE));

      /* Construct the shape functions */
      PetscCall(AncPhiE(GLampE, (const PetscReal(*)[2])GDLampE, nordEdge, IdecE, PhiE, DPhiE));

      for (PetscInt j = minI; j < maxI; j++) {
        ShapH[m] = PhiE[j - 1];
        for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
          GradH[k][m] = DPhiE[k][j - 1];
        }
        m += 1;
      }
    }
  }

  /* Arrays for basis functions projection over faces */
  PetscReal LampF[NUM_FACES_PER_CELL][NUM_DIMENSIONS];                  /* Shape functions over faces */
  PetscReal DLampF[NUM_FACES_PER_CELL][NUM_DIMENSIONS][NUM_DIMENSIONS]; /* Shape functions over faces */
  PetscReal **PhiTri, ***DPhiTri;

  /* Compute faces projection */
  PetscCall(ProjectTetF((const PetscReal*)Lam, (const PetscReal(*)[NUM_DIMENSIONS + 1]) DLam, LampF, DLampF, &IdecF));

  /* Allocate matrices for functions for faces */
  PetscCall(PetscCalloc1(nord - 2, &PhiTri));
  for (PetscInt i = 0; i < nord - 2; i++) {
    PetscCall(PetscCalloc1(nord - 2, &PhiTri[i]));
  }
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DPhiTri));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(nord - 2, &DPhiTri[i]));
    for (PetscInt j = 0; j < nord - 2; j++) {
      PetscCall(PetscCalloc1(nord - 2, &DPhiTri[i][j]));
    }
  }

  /* Compute shape functions for faces */
  PetscInt nordFace = 0;
  PetscInt numDofFace = 0;
  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; i++) {
    /* Local parameters */
    nordFace = nord;
    numDofFace = (nordFace - 1) * (nordFace - 2) / 2;
    if (numDofFace > 0) {
      /* Local parameters (again) */
      PetscInt maxIJ = nordFace;
      /* Orient */
      PetscReal GLampF[NUM_DIMENSIONS];
      PetscReal GDLampF[NUM_DIMENSIONS][NUM_DIMENSIONS];
      PetscReal tmpLampF[NUM_DIMENSIONS];
      PetscReal tempDLampF[NUM_DIMENSIONS][NUM_DIMENSIONS];

      /* Prepare input matrices */
      for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
        tmpLampF[j] = LampF[i][j];
        for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
          tempDLampF[j][k] = DLampF[i][j][k];
        }
      }

      /* Compute faces orientation */
      PetscCall(OrientTri(tmpLampF, (const PetscReal(*)[NUM_DIMENSIONS])tempDLampF, NoriF[i], GLampF, GDLampF));

      /* Construct the shape functions */
      PetscCall(AncPhiTri(GLampF, (const PetscReal(*)[NUM_DIMENSIONS])GDLampF, nordFace, IdecF, PhiTri, DPhiTri));

      for (PetscInt j = minIJ + 2; j <= maxIJ; j++) {
        for (PetscInt k = minI + 1; k < j - minJ; k++) {
          PetscInt l = j - k;
          ShapH[m] = PhiTri[k - 2][l - 1];
          for (PetscInt n = 0; n < NUM_DIMENSIONS; n++) {
            GradH[n][m] = DPhiTri[n][k - 2][l - 1];
          }
          m += 1;
        }
      }
    }
  }

  /* If necessary, create bubbles (basis functions in volume)*/
  PetscInt nordB = nord;
  PetscInt ndofB = (nordB - 1) * (nordB - 2) * (nordB - 3) / 6;

  if (ndofB > 0) {
    /* Local variables */
    PetscInt minbeta = 2 * (minIJ + 2);
    PetscInt maxIJK = nordB;
    PetscInt maxK = maxIJK - minIJ - 2;
    PetscReal **PhiTriV, ***DPhiTriV;
    PetscReal **homLbetV, ***DhomLbetV;
    PetscReal GLampV[NUM_DIMENSIONS];
    PetscReal GDLampV[NUM_DIMENSIONS][NUM_DIMENSIONS];

    IdecB[0] = IdecF;
    IdecB[1] = PETSC_TRUE;

    /* Allocate matrices for functions for volume */
    PetscCall(PetscCalloc1(nord - 3, &PhiTriV));
    PetscCall(PetscCalloc1(nord - 3, &homLbetV));
    for (PetscInt i = 0; i < nord - 3; i++) {
      PetscCall(PetscCalloc1(nord - 3, &PhiTriV[i]));
      PetscCall(PetscCalloc1(nord - 3, &homLbetV[i]));
    }
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DPhiTriV));
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &DhomLbetV));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
      PetscCall(PetscCalloc1(nord - 3, &DPhiTriV[i]));
      PetscCall(PetscCalloc1(nord - 3, &DhomLbetV[i]));
      for (PetscInt j = 0; j < nord - 3; j++) {
        PetscCall(PetscCalloc1(nord - 3, &DPhiTriV[i][j]));
        PetscCall(PetscCalloc1(nord - 3, &DhomLbetV[i][j]));
      }
    }

    /* Prepare input matrices */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
      GLampV[i] = Lam[i];
      for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
        GDLampV[i][j] = DLam[i][j];
      }
    }

    /* Call phiTri and HomIJacobi - no need to orient */
    PetscCall(AncPhiTri(GLampV, (const PetscReal(*)[NUM_DIMENSIONS])GDLampV, nordB - minK - 1, IdecB[0], PhiTriV, DPhiTriV));

    PetscReal tmp1[2] = {1 - Lam[3], Lam[3]};
    PetscReal tmp2[NUM_DIMENSIONS][2];

    /* Initialize input matrix */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
      tmp2[i][0] = -DLam[i][3];
      tmp2[i][1] = DLam[i][3];
    }

    PetscCall(HomIJacobi(tmp1, (const PetscReal(*)[2])tmp2, maxK, minbeta, IdecB[1], homLbetV, DhomLbetV));

    for (PetscInt i = minIJK + 3; i < maxIJK + 1; i++) {
      for (PetscInt j = minIJ; j < i - minK - 2; j++) {
        for (PetscInt k = minI; k < j - minJ + 1; k++) {
          ShapH[m] = PhiTriV[k - 1][j] * homLbetV[j - 1][k];
          for (PetscInt n = 0; n < NUM_DIMENSIONS; n++) {
            GradH[n][m] = homLbetV[j - 1][k] * DPhiTriV[n][k - 1][j] + PhiTriV[k - 1][j] * DhomLbetV[n][j - 1][k];
          }
          m += 1;
        }
      }
    }

    /* Free memory */
    for (PetscInt i = 0; i < nord - 3; i++) {
      PetscCall(PetscFree(PhiTriV[i]));
      PetscCall(PetscFree(homLbetV[i]));
    }
    PetscCall(PetscFree(PhiTriV));
    PetscCall(PetscFree(homLbetV));

    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
      for (PetscInt j = 0; j < nord - 3; j++) {
        PetscCall(PetscFree(DPhiTriV[i][j]));
        PetscCall(PetscFree(DhomLbetV[i][j]));
      }
      PetscCall(PetscFree(DPhiTriV[i]));
      PetscCall(PetscFree(DhomLbetV[i]));
    }
    PetscCall(PetscFree(DPhiTriV));
    PetscCall(PetscFree(DhomLbetV));
  }

  /* Free memory */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscFree(DPhiE[i]));
  }
  PetscCall(PetscFree(DPhiE));
  PetscCall(PetscFree(PhiE));

  for (PetscInt i = 0; i < nord - 2; i++) {
    PetscCall(PetscFree(PhiTri[i]));
  }
  PetscCall(PetscFree(PhiTri));

  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < nord - 2; j++) {
      PetscCall(PetscFree(DPhiTri[i][j]));
    }
    PetscCall(PetscFree(DPhiTri[i]));
  }
  PetscCall(PetscFree(DPhiTri));

  /* ---- shape3DHTet basis-natural → PETSc DMPlex closure reorder -----
   *
   * The body above fills ShapH and GradH in BASIS-NATURAL order
   *     [vertex 0..3, edge 0..6(nord-1)-1,
   *      face 0..4·(nord-1)(nord-2)/2-1,
   *      volume 0..(nord-1)(nord-2)(nord-3)/6-1].
   *
   * PETSc DMPlex closure traversal at a cell yields DOFs in the
   * REVERSE depth order
   *     [volume, face, edge, vertex]
   * — see the symmetric perm table in shape3DETet (hierarchical
   * H(curl) basis) for the same idea on the H(curl) side. Without
   * this fixup the column indices passed to MatSetValuesLocal (which
   * come from DMPlexGetClosureIndices, i.e. PETSc closure order)
   * would address natural-order positions; at nord = 1 the two
   * coincide (only vertex DOFs) and the bug is invisible, but for
   * nord ≥ 2 every G coefficient lands in the wrong column.
   *
   * The permutation is computed inline from the structural counts
   * (no static table); at most one branch per output position. */
  {
    const PetscInt e        = nord - 1;
    const PetscInt f        = (nord - 1) * (nord - 2) / 2;
    const PetscInt v        = (nord - 1) * (nord - 2) * (nord - 3) / 6;
    const PetscInt num_edge = 6 * e;
    const PetscInt num_face = 4 * f;
    const PetscInt num_vol  = v;
    const PetscInt total    = (nord + 1) * (nord + 2) * (nord + 3) / 6;

    /* Copy natural-order values into a stack scratch. */
    PetscReal tmpShapH[210];   /* >= max total (nord=6 → 84) */
    PetscReal tmpGradH[NUM_DIMENSIONS][210];
    for (PetscInt k = 0; k < total; k++) {
      tmpShapH[k] = ShapH[k];
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++)
        tmpGradH[d][k] = GradH[d][k];
    }

    /* Reorder into closure order. */
    for (PetscInt p = 0; p < total; p++) {
      PetscInt q = p;
      PetscInt nat;
      if (q < num_vol)                       nat = 4 + num_edge + num_face + q;
      else if ((q -= num_vol)      < num_face)  nat = 4 + num_edge + q;
      else if ((q -= num_face)     < num_edge)  nat = 4 + q;
      else                                   nat = q - num_edge;
      ShapH[p] = tmpShapH[nat];
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++)
        GradH[d][p] = tmpGradH[d][nat];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes a 3D unit vector resulting from sequential rotations
 *        based on azimuth and dip angles.
 *
 * @param[in] azimuth          Rotation angle in the x-y plane (degrees).
 * @param[in] dip              Rotation angle in the x-z plane (degrees).
 * @param[out] rotationVector  Output 3D vector after rotation.
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @details The function starts with the base vector [1, 0, 0] (aligned
 *          with the x-axis). It converts the input angles from degrees
 *          to radians. Rotations are applied sequentially:
 *          1. x-y plane rotation by `azimuth` (matrix M1)
 *          2. x-z plane rotation by `dip` (matrix M2)
 *          3. y-z plane rotation (matrix M3), currently hardcoded
 *             with angle `tetha = 0`.
 *
 *          The overall rotation matrix is M = M1 * M2 * M3.
 *          The final rotated vector is obtained by multiplying the base
 *          vector by this rotation matrix and stored in `rotationVector`.
 *
 * @note The y-z plane rotation is currently fixed at 0 degrees, but the
 *       code structure allows for future generalization.
 */
PetscErrorCode computeVectorRotation(const PetscReal azimuth, const PetscReal dip, PetscReal rotationVector[NUM_DIMENSIONS]) {

  PetscFunctionBeginUser;

  /* Variable declarations */
  PetscReal base_vector[NUM_DIMENSIONS] = {1., 0., 0.};

  /* Convert degrees to radians for rotation */
  PetscReal alpha = azimuth * PETSC_PI / 180.; // x-y plane
  PetscReal beta = dip * PETSC_PI / 180.;      // x-z plane
  PetscReal tetha = 0.0 * PETSC_PI / 180.;     // y-z plane

  /* Define rotation matrices for each plane */
  /* x-y plane */
  PetscReal M1[NUM_DIMENSIONS][NUM_DIMENSIONS] = {
      {PetscCosReal(alpha), -PetscSinReal(alpha), 0.}, {PetscSinReal(alpha), PetscCosReal(alpha), 0.}, {0., 0., 1.}};

  /* x-z plane */
  PetscReal M2[NUM_DIMENSIONS][NUM_DIMENSIONS] = {
      {PetscCosReal(beta), 0., -PetscSinReal(beta)}, {0., 1., 0.}, {PetscSinReal(beta), 0., PetscCosReal(beta)}};

  /* y-z plane */
  PetscReal M3[NUM_DIMENSIONS][NUM_DIMENSIONS] = {
      {1., 0., 0.}, {0., PetscCosReal(tetha), -PetscSinReal(tetha)}, {0., PetscSinReal(tetha), PetscCosReal(tetha)}};

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
 * @brief Maps global Cartesian coordinates to reference tetrahedron coordinates.
 *
 * @param[in] coordinates  Array of size 12 containing the spatial coordinates
 *                         of the tetrahedron's 4 vertices in the order
 *                         [x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3].
 * @param[in] point        The global Cartesian coordinates [x, y, z] of
 *                         the point to transform.
 * @param[out] XiEtaZeta   Output array [xi, eta, zeta] representing the
 *                         coordinates of the point in the reference tetrahedron.
 *
 * @return PetscErrorCode  PETSC_SUCCESS always.
 *
 * @details This function computes the reference coordinates of a point
 *          with respect to a physical tetrahedron. The reference tetrahedron
 *          is assumed to have vertices at (0,0,0), (1,0,0), (0,1,0), and (0,0,1).
 *
 *          The mapping is affine, so the inverse is computed using Cramer's
 *          rule via triple products (determinants of 3x3 matrices). Specifically,
 *          if v0, v1, v2 are the vectors from the first vertex to the other
 *          three vertices and vp is the vector from the first vertex to the
 *          point, the reference coordinates are given by:
 *            xi  = det(vp, v1, v2) / det(v0, v1, v2)
 *            eta = det(v0, vp, v2) / det(v0, v1, v2)
 *            zeta= det(v0, v1, vp) / det(v0, v1, v2)
 *
 *          The function checks for degenerate tetrahedra (zero or near-zero
 *          volume) and raises an error if detected.
 */
PetscErrorCode tetrahedronXYZToReference(const PetscReal coordinates[NUM_VERTICES_PER_CELL * NUM_DIMENSIONS],
                                         const PetscReal point[NUM_DIMENSIONS], PetscReal XiEtaZeta[NUM_DIMENSIONS]) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscReal x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3;
  PetscReal J, tp0, tp1, tp2;
  PetscReal v0[NUM_DIMENSIONS], v1[NUM_DIMENSIONS], v2[NUM_DIMENSIONS], vp[NUM_DIMENSIONS];

  /* Vertex coordinates */
  x0 = coordinates[0];
  y0 = coordinates[1];
  z0 = coordinates[2];
  x1 = coordinates[3];
  y1 = coordinates[4];
  z1 = coordinates[5];
  x2 = coordinates[6];
  y2 = coordinates[7];
  z2 = coordinates[8];
  x3 = coordinates[9];
  y3 = coordinates[10];
  z3 = coordinates[11];

  /* Vectors from v0 */
  v0[0] = x1 - x0;
  v0[1] = y1 - y0;
  v0[2] = z1 - z0;

  v1[0] = x2 - x0;
  v1[1] = y2 - y0;
  v1[2] = z2 - z0;

  v2[0] = x3 - x0;
  v2[1] = y3 - y0;
  v2[2] = z3 - z0;

  /* Vector from v0 to point */
  vp[0] = point[0] - x0;
  vp[1] = point[1] - y0;
  vp[2] = point[2] - z0;

  PetscCall(tripleProduct(v0, v1, v2, &J));
  PetscCheck(PetscAbsReal(J) > PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Degenerate tetrahedron: zero or near-zero volume");

  PetscCall(tripleProduct(vp, v1, v2, &tp0));
  PetscCall(tripleProduct(v0, vp, v2, &tp1));
  PetscCall(tripleProduct(v0, v1, vp, &tp2));

  XiEtaZeta[0] = tp0 / J; /* xi */
  XiEtaZeta[1] = tp1 / J; /* eta */
  XiEtaZeta[2] = tp2 / J; /* zeta */

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the Jacobian matrix, its inverse, and determinant
 *        for the affine mapping from the reference tetrahedron to
 *        a physical tetrahedron.
 *
 * @param[in,out] cell Pointer to a `Cell` structure containing:
 *                     - `coordinates`: array of size 12 with the
 *                       tetrahedron vertex coordinates in the order
 *                       [x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3].
 *                     - `jacobian`: 3x3 matrix to store the computed Jacobian.
 *                     - `invJacobian`: 3x3 matrix to store the inverse Jacobian.
 *                     - `detJacobian`: scalar to store the determinant.
 *
 * @return PetscErrorCode  PETSC_SUCCESS always.
 *
 * @details The function computes the Jacobian of the affine transformation
 *          that maps points from the reference tetrahedron
 *          (vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1))
 *          to the physical tetrahedron defined by `cell->coordinates`.
 *
 *          Steps performed:
 *            1. Constructs the Jacobian matrix using differences of vertex coordinates.
 *            2. Computes the determinant of the Jacobian.
 *            3. Computes the cofactor matrix.
 *            4. Computes the adjugate matrix (transpose of the cofactor matrix).
 *            5. Computes the inverse Jacobian by dividing the adjugate by the determinant.
 *
 *          This function assumes a non-degenerate tetrahedron (non-zero volume).
 *          The Jacobian can be used for transforming gradients and for integration
 *          over the physical tetrahedron.
 */
PetscErrorCode computeCellJacobian(Cell* cell) {
  PetscFunctionBeginUser;

  /* Variable declarations */
  PetscReal coFactorMatrix[NUM_DIMENSIONS][NUM_DIMENSIONS], adjugateMatrix[NUM_DIMENSIONS][NUM_DIMENSIONS];
  PetscReal invDeterminant;

  /* Reset matrices */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
      cell->jacobian[i][j] = 0.0;
      cell->invJacobian[i][j] = 0.0;
    }
  }

  /* Compute jacobian */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    cell->jacobian[0][i] = cell->coordinates[3 + i] - cell->coordinates[i]; // v1 - v0
    cell->jacobian[1][i] = cell->coordinates[6 + i] - cell->coordinates[i]; // v2 - v0
    cell->jacobian[2][i] = cell->coordinates[9 + i] - cell->coordinates[i]; // v3 - v0
  }

  /* Compute determinant */
  cell->detJacobian = cell->jacobian[0][0] * (cell->jacobian[1][1] * cell->jacobian[2][2] - cell->jacobian[1][2] * cell->jacobian[2][1]) -
                      cell->jacobian[0][1] * (cell->jacobian[1][0] * cell->jacobian[2][2] - cell->jacobian[1][2] * cell->jacobian[2][0]) +
                      cell->jacobian[0][2] * (cell->jacobian[1][0] * cell->jacobian[2][1] - cell->jacobian[1][1] * cell->jacobian[2][0]);

  /* Compute cofactor matrix */
  coFactorMatrix[0][0] = cell->jacobian[1][1] * cell->jacobian[2][2] - cell->jacobian[1][2] * cell->jacobian[2][1];
  coFactorMatrix[0][1] = -(cell->jacobian[1][0] * cell->jacobian[2][2] - cell->jacobian[1][2] * cell->jacobian[2][0]);
  coFactorMatrix[0][2] = cell->jacobian[1][0] * cell->jacobian[2][1] - cell->jacobian[1][1] * cell->jacobian[2][0];
  coFactorMatrix[1][0] = -(cell->jacobian[0][1] * cell->jacobian[2][2] - cell->jacobian[0][2] * cell->jacobian[2][1]);
  coFactorMatrix[1][1] = cell->jacobian[0][0] * cell->jacobian[2][2] - cell->jacobian[0][2] * cell->jacobian[2][0];
  coFactorMatrix[1][2] = -(cell->jacobian[0][0] * cell->jacobian[2][1] - cell->jacobian[0][1] * cell->jacobian[2][0]);
  coFactorMatrix[2][0] = cell->jacobian[0][1] * cell->jacobian[1][2] - cell->jacobian[0][2] * cell->jacobian[1][1];
  coFactorMatrix[2][1] = -(cell->jacobian[0][0] * cell->jacobian[1][2] - cell->jacobian[0][2] * cell->jacobian[1][0]);
  coFactorMatrix[2][2] = cell->jacobian[0][0] * cell->jacobian[1][1] - cell->jacobian[0][1] * cell->jacobian[1][0];

  /* Compute adjugate matrix */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
      adjugateMatrix[i][j] = coFactorMatrix[j][i];
    }
  }

  /* Compute inverse of jacobian */
  invDeterminant = 1.0 / (cell->detJacobian);
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
      cell->invJacobian[i][j] = invDeterminant * adjugateMatrix[i][j];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the orientation of faces and edges for a tetrahedral cell.
 *
 * @param[in,out] cell Pointer to a `Cell` structure containing:
 *                     - `closure`: array representing the cell's
 *                       transitive closure (e.g., from vertices to edges/faces).
 *                     - `orientation`: array to store the computed
 *                       face and edge orientations.
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @details The function extracts the orientation of each face and edge
 *          based on the cell's transitive closure. The convention for
 *          indexing is as follows:
 *
 *            - Faces: indices start at position 2 in `closure`, orientations
 *              at position 3, total of NUM_FACES_PER_CELL faces.
 *            - Edges: indices start after faces (2 + NUM_FACES_PER_CELL*2),
 *              orientations at the next position, total of NUM_EDGES_PER_CELL edges.
 *
 *          The extracted orientation values are then mapped to the PETGEM
 *          convention for basis functions:
 *            - Faces: [-3,-2,-1,0,1,2] → [4,3,5,0,1,2]
 *            - Edges: negative → -1, positive → 1
 *
 *          This allows the shape function construction routines to correctly
 *          handle local-to-global orientation transformations.
 */
PetscErrorCode computeCellOrientation(Cell* cell) {
  PetscFunctionBeginUser;

  /* Variable declarations */
  PetscInt currentPoint;

  /* Orden convention for transitive clousure/orientation
   * (cellTransitiveClosure) */
  /*
      - Faces indices start on position 2, orientation
     indices start on position 3
      - Edges indices start on position 2 +
     NUM_FACES_PER_CELL * 2, orientation indices start on
     position 3 + NUM_FACES_PER_CELL*2
      - Vertices indices start on position 2 +
     NUM_FACES_PER_CELL * 2 + NUM_EDGES_PER_CELL * 2

      Order convention for cellOrientation = F0, F1, F2, F3,
     E0, E1, E2, E3, E4, E5
  */

  /* Face orientation for OrientTri (PETGEM 0..5).
   *
   * IMPORTANT: we do NOT derive this from PETSc's orientation flag in the
   * cell closure. PETSc's triangle-polytope orientation convention has
   * shifted across versions and an earlier cast (raw -3..2 -> {4,3,5,0,1,2})
   * was never validated end-to-end — the legacy nord=2 path bypassed
   * OrientTri entirely (canonical face tangents from sorted vertices), and
   * shape3DHTet at H1 nord >= 2 was never exercised in production. The
   * hierarchical H(curl) basis is the first to actually consume faces[f],
   * and a wrong cast silently breaks tangential continuity on every face
   * with non-zero PETSc orientation.
   *
   * Instead, we mirror the legacy canonical-face-tangent strategy: lex-sort
   * the three face vertices by their physical coordinates. That ordering
   * is cell-invariant — both K+ and K- sharing F see the same three
   * physical points and sort them identically. The permutation that takes
   * (FACE_VERTICES[f][0], FACE_VERTICES[f][1], FACE_VERTICES[f][2]) into
   * the sorted order, expressed as one of OrientTri's 6 permutation codes
   * (0..5 — see comments around OrientTri), is what we store as faces[f].
   *
   * After this, OrientTri produces the SAME canonical (GS[0],GS[1],GS[2])
   * triple from both cells on the shared face, so AncETri / HomIJacobi
   * yield matching face-bubble basis values from each side. */
  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; i++) {
    const PetscInt v0 = FACE_VERTICES[i][0];
    const PetscInt v1 = FACE_VERTICES[i][1];
    const PetscInt v2 = FACE_VERTICES[i][2];
    const PetscReal *p[3] = {
        &cell->coordinates[v0 * NUM_DIMENSIONS],
        &cell->coordinates[v1 * NUM_DIMENSIONS],
        &cell->coordinates[v2 * NUM_DIMENSIONS]};

    /* Insertion-sort indices [0,1,2] by physical coord (x, then y, then z). */
    PetscInt ord[3] = {0, 1, 2};
    for (PetscInt a = 1; a < 3; a++) {
      for (PetscInt b = a; b > 0; b--) {
        const PetscReal *pa = p[ord[b - 1]];
        const PetscReal *pb = p[ord[b]];
        PetscBool swap = PETSC_FALSE;
        if      (pa[0] >  pb[0]) swap = PETSC_TRUE;
        else if (pa[0] == pb[0]) {
          if      (pa[1] >  pb[1]) swap = PETSC_TRUE;
          else if (pa[1] == pb[1] && pa[2] > pb[2]) swap = PETSC_TRUE;
        }
        if (swap) { PetscInt t = ord[b - 1]; ord[b - 1] = ord[b]; ord[b] = t; }
        else break;
      }
    }

    /* Map permutation tuple (ord[0],ord[1],ord[2]) to an OrientTri code.
     *   0: (0,1,2)   1: (1,2,0)   2: (2,0,1)
     *   3: (0,2,1)   4: (1,0,2)   5: (2,1,0)
     */
    PetscInt nori = -1;
    if      (ord[0] == 0 && ord[1] == 1 && ord[2] == 2) nori = 0;
    else if (ord[0] == 1 && ord[1] == 2 && ord[2] == 0) nori = 1;
    else if (ord[0] == 2 && ord[1] == 0 && ord[2] == 1) nori = 2;
    else if (ord[0] == 0 && ord[1] == 2 && ord[2] == 1) nori = 3;
    else if (ord[0] == 1 && ord[1] == 0 && ord[2] == 2) nori = 4;
    else if (ord[0] == 2 && ord[1] == 1 && ord[2] == 0) nori = 5;
    PetscCheck(nori >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "computeCellOrientation: unable to map face %" PetscInt_FMT
               " sort permutation (%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT
               ") to an OrientTri code", i, ord[0], ord[1], ord[2]);
    cell->orientation.faces[i] = nori;
  }

  /* Get orientation for edges (stored as ±1 signs) */
  currentPoint = 2 + NUM_FACES_PER_CELL * 2;
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; i++) {
    cell->orientation.edgeSigns[i] = (cell->closure[currentPoint + 1] < 0) ? -1 : 1;
    currentPoint += 2;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Determines the number of 1D Gauss-Legendre quadrature points
 *        required for a given polynomial order.
 *
 * @param[in] nord The polynomial order of the element.
 * @param[out] quadrature Pointer to a `Quadrature1D` structure where
 *                        `numPoints` will be set.
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @details Computes the minimum number of 1D Gauss-Legendre points
 *          required to integrate polynomials up to degree `2*nord`
 *          exactly. The number of quadrature points is calculated as:
 *
 *            numPoints = ceil(gaussOrder / 2) + 1
 *
 *          where `gaussOrder = 2 * nord`. For stability and predefined
 *          table limits, the number of points is saturated at 11.
 *
 *          Basic checks ensure that `nord` is non-negative and within
 *          the supported range (<=11).
 */
PetscErrorCode computeNum1DQuadraturePoints(const PetscInt nord, Quadrature1D* quadrature) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt gaussOrder, numPoints;

  /* Compute gauss order*/
  gaussOrder = 2 * nord;

  /* Basic verification */
  PetscCheck(gaussOrder >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error: Negative polynomial orders are not supported");
  PetscCheck(gaussOrder <= 12, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error: 1D polynomial orders higher than 11 are not supported");

  /* Gauss-Legendre rule:
     exact for polynomials of degree (2*numPoints - 1)
     Saturated to 11 points for high orders */
  numPoints = gaussOrder / 2 + 1;

  if (numPoints > 12) {
    numPoints = 12;
  }

  quadrature->numPoints = numPoints;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Determines the number of 2D quadrature points for a triangular element
 *        given its polynomial order.
 *
 * @param[in] nord The polynomial order of the element (0 ≤ nord ≤ 19).
 * @param[out] quadrature Pointer to a `Quadrature2D` structure where
 *                        `numPoints` will be set.
 *
 * @return PetscErrorCode PETSC_SUCCESS always.
 *
 * @details Uses a precomputed lookup table for triangular quadrature rules
 *          to determine the minimum number of points needed to exactly
 *          integrate polynomials of degree up to `nord` over a triangle.
 *          Valid orders are 0 through 19. Each table entry corresponds
 *          to the recommended number of quadrature points for that order.
 *          Basic checks ensure `nord` is within the supported range.
 */
PetscErrorCode computeNum2DQuadraturePoints(const PetscInt nord, Quadrature2D* quadrature) {
  PetscFunctionBeginUser;

  static const PetscInt numPointsTable[20] = {
      1,  // 0
      1,  // 1
      3,  // 2
      4,  // 3
      6,  // 4
      7,  // 5
      12, // 6
      13, // 7
      16, // 8
      19, // 9
      25, // 10
      27, // 11
      33, // 12
      37, // 13
      42, // 14
      48, // 15
      52, // 16
      61, // 17
      70, // 18
      73  // 19
  };

  PetscCheck(nord >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error: Negative quadrature order not supported");
  PetscCheck(nord <= 19, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error: 2D polynomial orders higher than 19 are not supported");

  quadrature->numPoints = numPointsTable[nord];

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Determines the number of Gauss quadrature points for a tetrahedron
 *        needed to exactly integrate polynomials up to a given order.
 *
 * @param[in] nord The basis polynomial order of the element.
 *                 The required integration order is 2*nord.
 * @param[out] quadrature Pointer to a `Quadrature3D` structure where
 *                        `numPoints` will be set.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success.
 *
 * @details Computes the required number of Gauss points using a precomputed
 *          lookup table for tetrahedral quadrature rules. Supports integration
 *          orders corresponding to 1 ≤ 2*nord ≤ 12. Basic checks ensure that
 *          `nord` produces a valid integration order. The table maps each
 *          Gauss order directly to the recommended number of quadrature points
 *          for exact integration over a tetrahedron.
 */
PetscErrorCode computeNum3DQuadraturePoints(const PetscInt nord, Quadrature3D* quadrature) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt gaussOrder;
  static const PetscInt numPointsTable[13] = {
      /* Lookup table: index = gaussOrder */
      0,   // 0  (unused)                      /* index 0 unused to keep direct mapping */
      1,   // 1
      4,   // 2
      5,   // 3
      11,  // 4
      14,  // 5
      24,  // 6
      31,  // 7
      43,  // 8
      53,  // 9
      126, // 10
      126, // 11
      210  // 12
  };

  /* Compute number of integration points */
  gaussOrder = 2 * nord;

  /* Basic verification */
  PetscCheck(gaussOrder >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error: Orders lower than 1 are not supported.\n");
  PetscCheck(gaussOrder <= 12, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error: Orders higher than 6 are not supported by PETGEM.\n");

  /* Setup number of integration points */
  quadrature->numPoints = numPointsTable[gaussOrder];

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Populates 1D Gauss-Legendre quadrature points and weights.
 *
 * @param[in,out] quadrature Pointer to a `Quadrature1D` structure.
 *                           The `numPoints` field must be set before calling.
 *                           After execution, `points` and `weights` are filled
 *                           with the appropriate Gauss-Legendre points
 *                           and weights.
 * @return PetscErrorCode PETSC_SUCCESS on success.
 *
 * @details The function selects the correct precomputed 1D Gauss-Legendre table
 *          based on `numPoints` and copies the values into the `quadrature`
 *          structure. Supports up to 11 points, corresponding to polynomial
 *          integration orders up to 21 (exact for polynomials of degree 2*numPoints-1).
 */
PetscErrorCode compute1DQuadraturePoints(Quadrature1D* quadrature) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  const PetscReal(*table)[2] = NULL; // pointer to the right quadrature table

  switch (quadrature->numPoints) {
  case 1:
    table = NORD1_1DGAUSSPOINTS;
    break;
  case 2:
    table = NORD2_1DGAUSSPOINTS;
    break;
  case 3:
    table = NORD3_1DGAUSSPOINTS;
    break;
  case 4:
    table = NORD4_1DGAUSSPOINTS;
    break;
  case 5:
    table = NORD5_1DGAUSSPOINTS;
    break;
  case 6:
    table = NORD6_1DGAUSSPOINTS;
    break;
  case 7:
    table = NORD7_1DGAUSSPOINTS;
    break;
  case 8:
    table = NORD8_1DGAUSSPOINTS;
    break;
  case 9:
  case 10:
  case 11:
    table = NORD11_1DGAUSSPOINTS;
    break;
  default:
    break;
  }

  for (PetscInt i = 0; i < quadrature->numPoints; i++) {
    quadrature->points[i] = table[i][0];
    quadrature->weights[i] = table[i][1];
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Populates 2D Gauss quadrature points and weights for a reference triangle.
 *
 * @param[in,out] quadrature Pointer to a `Quadrature2D` structure.
 *                           The `numPoints` field must be set before calling.
 *                           After execution, `points` and `weights` are filled
 *                           with the appropriate 2D Gauss points and weights.
 * @return PetscErrorCode PETSC_SUCCESS on success.
 *
 * @details Selects the precomputed 2D triangular quadrature table corresponding
 *          to `numPoints` and normalizes the weights. Supports standard
 *          Gauss rules with 1, 3, 4, 6, 7, 12, 13, 16, 19, 25, 27, 33, 37,
 *          42, 48, 52, 61, 70, and 73 points. Exact for polynomials of degree
 *          up to the specified order.
 */
PetscErrorCode compute2DQuadraturePoints(Quadrature2D* quadrature) {
  PetscFunctionBeginUser;

  switch (quadrature->numPoints) {
  case 1: {
    PetscCall(renormalization2DGaussPoints(NORD1_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 3: {
    PetscCall(renormalization2DGaussPoints(NORD2_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 4: {
    PetscCall(renormalization2DGaussPoints(NORD3_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 6: {
    PetscCall(renormalization2DGaussPoints(NORD4_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 7: {
    PetscCall(renormalization2DGaussPoints(NORD5_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 12: {
    PetscCall(renormalization2DGaussPoints(NORD6_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 13: {
    PetscCall(renormalization2DGaussPoints(NORD7_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 16: {
    PetscCall(renormalization2DGaussPoints(NORD8_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 19: {
    PetscCall(renormalization2DGaussPoints(NORD9_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 25: {
    PetscCall(renormalization2DGaussPoints(NORD10_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 27: {
    PetscCall(renormalization2DGaussPoints(NORD11_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 33: {
    PetscCall(renormalization2DGaussPoints(NORD12_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 37: {
    PetscCall(renormalization2DGaussPoints(NORD13_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 42: {
    PetscCall(renormalization2DGaussPoints(NORD14_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 48: {
    PetscCall(renormalization2DGaussPoints(NORD15_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 52: {
    PetscCall(renormalization2DGaussPoints(NORD16_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 61: {
    PetscCall(renormalization2DGaussPoints(NORD17_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 70: {
    PetscCall(renormalization2DGaussPoints(NORD18_2DGAUSSPOINTS, quadrature));
    break;
  }
  case 73: {
    PetscCall(renormalization2DGaussPoints(NORD19_2DGAUSSPOINTS, quadrature));
    break;
  }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Populates 3D Gauss quadrature points and weights for a reference tetrahedron.
 *
 * @param[in,out] quadrature Pointer to a `Quadrature3D` structure.
 *                           The `numPoints` field must be set before calling.
 *                           After execution, `points` and `weights` are filled
 *                           with the appropriate 3D Gauss points and weights.
 * @return PetscErrorCode PETSC_SUCCESS on success.
 *
 * @details Selects the appropriate precomputed 3D tetrahedral quadrature table
 *          based on `numPoints` and calls `renormalization3DGaussPoints` to
 *          normalize and copy the values into the `quadrature` structure.
 *          Supports the following numbers of points:
 *          1, 4, 5, 11, 14, 24, 31, 43, 53, 126, 210, which correspond to
 *          polynomial integration orders from 1 up to 12.
 */
PetscErrorCode compute3DQuadraturePoints(Quadrature3D* quadrature) {
  PetscFunctionBeginUser;

  switch (quadrature->numPoints) {
  case 1: {
    PetscCall(renormalization3DGaussPoints(NORD1_3DGAUSSPOINTS, quadrature));
    break;
  }
  case 4: {
    PetscCall(renormalization3DGaussPoints(NORD2_3DGAUSSPOINTS, quadrature));
    break;
  }
  case 5: {
    PetscCall(renormalization3DGaussPoints(NORD3_3DGAUSSPOINTS, quadrature));
    break;
  }
  case 11: {
    PetscCall(renormalization3DGaussPoints(NORD4_3DGAUSSPOINTS, quadrature));
    break;
  }
  case 14: {
    PetscCall(renormalization3DGaussPoints(NORD5_3DGAUSSPOINTS, quadrature));
    break;
  }
  case 24: {
    PetscCall(renormalization3DGaussPoints(NORD6_3DGAUSSPOINTS, quadrature));
    break;
  }
  case 31: {
    PetscCall(renormalization3DGaussPoints(NORD7_3DGAUSSPOINTS, quadrature));
    break;
  }
  case 43: {
    PetscCall(renormalization3DGaussPoints(NORD8_3DGAUSSPOINTS, quadrature));
    break;
  }
  case 53: {
    PetscCall(renormalization3DGaussPoints(NORD9_3DGAUSSPOINTS, quadrature));
    break;
  }
  case 126: {
    PetscCall(renormalization3DGaussPoints(NORD10_3DGAUSSPOINTS, quadrature));
    break;
  }
  case 210: {
    PetscCall(renormalization3DGaussPoints(NORD12_3DGAUSSPOINTS, quadrature));
    break;
  }
  default: {
    break;
  }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the elemental mass and stiffness matrices for a tetrahedral cell
 *        using first-order Nédélec edge basis functions (H(curl)-conforming).
 *
 * @param[in] nord Polynomial order of the Nédélec basis functions (currently supports 1).
 * @param[in] numDofInCell Number of degrees of freedom per cell (edges).
 * @param[in] cell Pointer to the Cell structure containing:
 *                 - Jacobian matrix and determinant
 *                 - Orientation of edges
 *                 - Conductivity tensor for material properties
 * @param[in] quadrature Pointer to the Quadrature3D structure containing:
 *                       - Gauss points
 *                       - Weights for tetrahedral integration
 * @param[out] Me Elemental mass matrix (numDofInCell x numDofInCell),
 *                representing \int_T (ε_r * N_i) · N_j dV.
 * @param[out] Ke Elemental stiffness matrix (numDofInCell x numDofInCell),
 *                representing \int_T (μ_r^{-1} * curl(N_i)) · curl(N_j) dV.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success.
 *
 * @details
 * This function performs the following steps to assemble the local
 * elemental matrices for a tetrahedral element in 3D:
 *
 * 1. **Tensor Setup**
 *    - Define the permittivity tensor `e_r` from the cell conductivity.
 *    - Define the magnetic permeability tensor `mu_r` (currently identity).
 *
 * 2. **Memory Allocation**
 *    - Allocate arrays for basis functions (Ni), their curls (NiCurl),
 *      derivatives (Dx_Ni, Dy_Ni, Dz_Ni), and Nédélec coefficients.
 *
 * 3. **Reset Elemental Matrices**
 *    - Initialize `Me` and `Ke` to zero.
 *
 * 4. **First-Order Nédélec Coefficients**
 *    - Compute the Nédélec basis function coefficients and derivatives
 *      using `computeNedelecOrder1Coefficients`.
 *
 * 5. **Loop Over Quadrature Points**
 *    - For each Gauss point:
 *      - Transform coordinates to the physical element.
 *      - Compute basis functions at the Gauss point using
 *        `computeNedelecOrder1BasisFunctions`.
 *      - Compute the mass matrix contribution:
 *          Me_ij += w_q * (N_i · (ε_r * N_j)) * sign_i * sign_j * det(J)
 *      - Compute the curls of the basis functions using
 *        `computeNedelecOrder1BasisFunctionCurls`.
 *      - Compute the stiffness matrix contribution:
 *          Ke_ij += w_q * (curl(N_i) · μ_r * curl(N_j)) * sign_i * sign_j * det(J)
 *      - Edge orientation signs from `cell->orientation` are applied
 *        to ensure global consistency.
 *
 * 6. **Cleanup**
 *    - Free all dynamically allocated arrays for Ni, NiCurl, derivatives, and coefficients.
 *
 * @note
 * - The stiffness matrix assumes isotropic magnetic permeability (μ_r = 1.0).
 */

PetscErrorCode computeElementalMatrices(const FEMSpace* fem, const Cell* cell, const Quadrature3D* quadrature,
                                        PetscReal** Me, PetscReal** Ke) {
  PetscFunctionBeginUser;

  const PetscInt numDofInCell = fem->numDofInCell;
  const NedelecOps *ops       = fem->ops;

  /* Constitutive tensors */
  PetscReal e_r[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{0.0}};
  PetscReal mu_r[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{0.0}};
  PetscReal iPoint[NUM_DIMENSIONS] = {0.0};
  PetscReal **Ni, **NiCurl, **coeffs, **Dx_Ni, **Dy_Ni, **Dz_Ni;

  e_r[0][0] = cell->conductivity[0];
  e_r[1][1] = cell->conductivity[1];
  e_r[2][2] = cell->conductivity[2];

  mu_r[0][0] = 1.0;
  mu_r[1][1] = 1.0;
  mu_r[2][2] = 1.0;

  /* Allocate arrays */
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &NiCurl));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dx_Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dy_Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dz_Ni));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(numDofInCell, &Ni[i]));
    PetscCall(PetscCalloc1(numDofInCell, &NiCurl[i]));
    PetscCall(PetscCalloc1(numDofInCell, &Dx_Ni[i]));
    PetscCall(PetscCalloc1(numDofInCell, &Dy_Ni[i]));
    PetscCall(PetscCalloc1(numDofInCell, &Dz_Ni[i]));
  }

  PetscCall(PetscCalloc1(numDofInCell, &coeffs));
  for (PetscInt i = 0; i < numDofInCell; i++) {
    PetscCall(PetscCalloc1(numDofInCell, &coeffs[i]));
  }

  const PetscReal** coeffs_const = (const PetscReal**)coeffs;
  const PetscReal** Dx_Ni_const = (const PetscReal**)Dx_Ni;
  const PetscReal** Dy_Ni_const = (const PetscReal**)Dy_Ni;
  const PetscReal** Dz_Ni_const = (const PetscReal**)Dz_Ni;

  for (PetscInt i = 0; i < numDofInCell; ++i) {
    for (PetscInt j = 0; j < numDofInCell; ++j) {
      Me[i][j] = 0.0;
      Ke[i][j] = 0.0;
    }
  }

  if (ops) {
    PetscInt signs[numDofInCell];
    PetscReal Ni_i[NUM_DIMENSIONS], Ni_j[NUM_DIMENSIONS], tmp[NUM_DIMENSIONS], value;

    PetscCall(buildDofSigns(cell, fem, signs));

    /* Cell-local coefficients (and nord=1 monomial derivatives). */
    PetscCall(ops->computeCoefficients(cell, coeffs, Dx_Ni, Dy_Ni, Dz_Ni));

    for (PetscInt q = 0; q < quadrature->numPoints; ++q) {
      iPoint[0] = quadrature->points[q][0];
      iPoint[1] = quadrature->points[q][1];
      iPoint[2] = quadrature->points[q][2];

      for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
        for (PetscInt k = 0; k < numDofInCell; k++) {
          Ni[j][k] = 0.0;
          NiCurl[j][k] = 0.0;
        }
      }

      PetscCall(ops->computeBasis(cell, iPoint, coeffs_const, Ni));

      /* Mass matrix: Me_jk += w_q * (N_j · e_r · N_k) * sign_j * sign_k * detJ */
      for (PetscInt j = 0; j < numDofInCell; j++) {
        for (PetscInt k = 0; k < numDofInCell; k++) {
          for (PetscInt l = 0; l < NUM_DIMENSIONS; l++) {
            Ni_i[l] = Ni[l][j];
            Ni_j[l] = Ni[l][k];
          }
          for (PetscInt l = 0; l < NUM_DIMENSIONS; l++) {
            tmp[l] = 0.0;
            for (PetscInt m = 0; m < NUM_DIMENSIONS; m++) {
              tmp[l] += e_r[l][m] * Ni_j[m];
            }
          }
          value = 0.0;
          for (PetscInt l = 0; l < NUM_DIMENSIONS; l++) {
            value += Ni_i[l] * tmp[l];
          }
          Me[j][k] += (quadrature->weights[q] * value * signs[j] * signs[k] * cell->detJacobian);
        }
      }

      PetscCall(ops->computeCurls(cell, iPoint, coeffs_const,
                                  Dx_Ni_const, Dy_Ni_const, Dz_Ni_const, NiCurl));

      /* Stiffness matrix: Ke_jk += w_q * (curl N_j · mu_r · curl N_k) * sign_j * sign_k * detJ */
      for (PetscInt j = 0; j < numDofInCell; j++) {
        for (PetscInt k = 0; k < numDofInCell; k++) {
          value = mu_r[0][0] * NiCurl[0][j] * NiCurl[0][k] + mu_r[0][1] * (NiCurl[0][j] * NiCurl[1][k] + NiCurl[1][j] * NiCurl[0][k]) +
                  mu_r[1][1] * NiCurl[1][j] * NiCurl[1][k] + mu_r[0][2] * (NiCurl[0][j] * NiCurl[2][k] + NiCurl[2][j] * NiCurl[0][k]) +
                  mu_r[1][2] * (NiCurl[1][j] * NiCurl[2][k] + NiCurl[2][j] * NiCurl[1][k]) + mu_r[2][2] * NiCurl[2][j] * NiCurl[2][k];

          Ke[j][k] += (quadrature->weights[q] * value * signs[j] * signs[k] * cell->detJacobian);
        }
      }
    }
  }

  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscFree(Ni[i]));
    PetscCall(PetscFree(NiCurl[i]));
    PetscCall(PetscFree(Dx_Ni[i]));
    PetscCall(PetscFree(Dy_Ni[i]));
    PetscCall(PetscFree(Dz_Ni[i]));
  }
  PetscCall(PetscFree(Ni));
  PetscCall(PetscFree(NiCurl));
  PetscCall(PetscFree(Dx_Ni));
  PetscCall(PetscFree(Dy_Ni));
  PetscCall(PetscFree(Dz_Ni));

  for (PetscInt i = 0; i < numDofInCell; i++) {
    PetscCall(PetscFree(coeffs[i]));
  }
  PetscCall(PetscFree(coeffs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal vectorNorm(const PetscReal v[NUM_DIMENSIONS]) {
  return PetscSqrtReal(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}




/* Nédélec nord=2 static data (EDGE_ROWS, FACE_NORMALS_REF, FACE*_AUX_*,
 * FACE_AUX_X/Y/Z) lives in constants.c and is declared in constants.h. */

/**
 * @brief Computes the elemental gradient matrix mapping H1 scalar basis functions
 *        (nodal) to H(curl) Nédélec edge basis functions for a tetrahedral element.
 *
 * @param[in] nord Polynomial order of the H1 basis functions (currently supports 1).
 * @param[in] numDofInCell Number of H(curl) degrees of freedom (edges) in the tetrahedral cell.
 * @param[in] numH1DofInCell Number of H1 degrees of freedom (vertices) in the tetrahedral cell.
 * @param[in] cell Pointer to the Cell structure containing:
 *                 - Edge orientations
 *                 - Jacobian mapping (from reference to real cell)
 * @param[in] quadrature Pointer to the 1D quadrature rule for edge integration:
 *                       - quadrature points along each edge
 *                       - quadrature weights
 * @param[out] gradientMatrix Elemental gradient matrix (numDofInCell x numH1DofInCell),
 *                            representing the line integral of Nédélec basis functions
 *                            projected along H1 gradients:
 *                            \f$ G_{ij} = \int_{edge} N_i \cdot \nabla \phi_j \, ds \f$
 *
 * @return PetscErrorCode PETSC_SUCCESS on success.
 *
 * @details
 * This function assembles the elemental gradient matrix by performing the following steps:
 *
 * 1. **Setup**
 *    - Allocate arrays for H1 shape functions (`ShapH`) and their gradients (`GradH`).
 *    - Reset `gradientMatrix` to zero.
 *
 * 2. **Loop over H1 basis functions**
 *    - For each H1 degree of freedom `i` (associated with tetrahedron vertices):
 *
 *      a. **Loop over edges**
 *         - Retrieve local vertex indices for the edge.
 *         - Compute the edge vector in reference coordinates (`edgeJacobian`) and its length.
 *         - Compute the unit vector along the edge.
 *         - Set the edge origin coordinates.
 *
 *      b. **Loop over quadrature points along the edge**
 *         - Map the 1D quadrature point to 3D reference coordinates along the edge.
 *         - Evaluate H1 shape function gradients at this point using `shape3DHTet`.
 *         - For first-order elements (nord = 1):
 *             - Compute the dot product of the gradient with the edge unit vector.
 *             - Multiply by quadrature weight, edge orientation sign, and edge Jacobian length.
 *             - Accumulate into `gradientMatrix`.
 *
 *      c. **Increment edge index in gradient matrix**
 *         - Only needed for first-order elements (1 dof per edge).
 *
 * 3. **Memory cleanup**
 *    - Free temporary arrays `ShapH` and `GradH`.
 *
 * @note
 * - The matrix represents a mapping from scalar H1 basis gradients to vector H(curl)
 *   edge functions, used in mixed FEM formulations (e.g., for curl-conforming discretizations).
 * - Orientation signs from `cell->orientation` are applied to ensure global assembly consistency.
 */
PetscErrorCode computeElementalGradientMatrix(const FEMSpace* fem, const Cell* cell, const Quadrature1D* quadrature1d,
                                              PetscReal** gradientMatrix) {
  PetscFunctionBeginUser;

  const PetscInt numDofInCell    = fem->numDofInCell;
  const PetscInt numH1DofInCell  = fem->numH1DofInCell;

  for (PetscInt i = 0; i < numDofInCell; ++i) {
    for (PetscInt j = 0; j < numH1DofInCell; ++j) {
      gradientMatrix[i][j] = 0.;
    }
  }

  if (fem->ops && fem->ops->buildGradientMatrix) {
    PetscCall(fem->ops->buildGradientMatrix(fem, cell, quadrature1d, gradientMatrix));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}





PetscErrorCode buildDofSigns(const Cell* cell, const FEMSpace* fem, PetscInt signs[]) {
  PetscFunctionBeginUser;
  (void)cell;

  /* Unified hierarchical Nédélec basis: orientation is encoded inside the
   * reference shape functions via OrientE / OrientTri inside shape3DETet,
   * so per-DOF sign multipliers must be +1 at every order. Multiplying by
   * cell->orientation.edgeSigns here would double-apply orientation and
   * break tangential continuity across cell boundaries. */
  const PetscInt numDofInCell = fem->numDofInCell;
  for (PetscInt j = 0; j < numDofInCell; j++) signs[j] = 1;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Evaluate Nédélec basis (and optionally curls) at a point inside `cell`.
 * Dispatches on fem->nord to the order-specific coeff/basis/curl helpers.
 * Pass NiCurl = NULL to skip curl evaluation (e.g. RHS source integrals).
 *
 * The caller owns all buffers: coeffs[numDofInCell][numDofInCell], and for
 * each of Ni, NiCurl, Dx_Ni, Dy_Ni, Dz_Ni a NUM_DIMENSIONS x numDofInCell
 * row-of-pointers layout. Dx_Ni/Dy_Ni/Dz_Ni are used only for nord=1. */
PetscErrorCode evaluateNedelecBasis(const FEMSpace* fem, const Cell* cell,
                                    const PetscReal point[NUM_DIMENSIONS],
                                    PetscReal** coeffs,
                                    PetscReal** Dx_Ni, PetscReal** Dy_Ni, PetscReal** Dz_Ni,
                                    PetscReal** Ni, PetscReal** NiCurl) {
  PetscFunctionBeginUser;

  const NedelecOps *ops = fem->ops;
  if (!ops) PetscFunctionReturn(PETSC_SUCCESS);

  const PetscReal* const* coeffs_const = (const PetscReal* const*)coeffs;

  PetscCall(ops->computeCoefficients(cell, coeffs, Dx_Ni, Dy_Ni, Dz_Ni));
  PetscCall(ops->computeBasis(cell, point, coeffs_const, Ni));

  if (NiCurl) {
    const PetscReal* const* Dx_const = (const PetscReal* const*)Dx_Ni;
    const PetscReal* const* Dy_const = (const PetscReal* const*)Dy_Ni;
    const PetscReal* const* Dz_const = (const PetscReal* const*)Dz_Ni;
    PetscCall(ops->computeCurls(cell, point, coeffs_const,
                                Dx_const, Dy_const, Dz_const, NiCurl));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}



/* ===========================================================================
 * Per-order Nédélec dispatch selector. All nedelecOps_orderN tables are
 * defined in src/hvfem_hierarchical.c (single hierarchical implementation
 * covering nord = 1..6); their extern declarations live in
 * hvfem_internal.h.
 * =========================================================================== */
const NedelecOps *nedelecOpsForOrder(PetscInt nord) {
  switch (nord) {
  case 1:  return &nedelecOps_order1;
  case 2:  return &nedelecOps_order2;
  case 3:  return &nedelecOps_order3;
  case 4:  return &nedelecOps_order4;
  case 5:  return &nedelecOps_order5;
  case 6:  return &nedelecOps_order6;
  default: return NULL;
  }
}
