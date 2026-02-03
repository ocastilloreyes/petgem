/*
 * Filename: hvfem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2025-06-12
 *
 * Description:
 * This file contains functions for high-order vector finite
 * element method (HVFEM) computations.
 *
 * Usage:
 * Include this file in your source code to utilize the grid
 * functions. For example: #include "grid.h"
 *
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

/**
 * @brief Computes the dot product of two 3D vectors.
 * @param[in] vector1 The first input vector [x1, y1,
 z1].
 * @param[in] vector2 The second input vector [x2, y2,
 z2].
 * @param[out] result Pointer to the scalar result.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Calculates *result = vector1[0]*vector2[0] +
 vector1[1]*vector2[1] + vector1[2]*vector2[2].
 */
static PetscErrorCode dotProduct(const PetscReal vector1[NUM_DIMENSIONS], const PetscReal vector2[NUM_DIMENSIONS], PetscReal* result) {
  PetscFunctionBeginUser;

  *result = 0.0;
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    *result += vector1[i] * vector2[i];
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode tripleProduct(const PetscReal a[3], const PetscReal b[3], const PetscReal c[3], PetscReal* result) {
  PetscFunctionBeginUser;

  *result = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);

  PetscFunctionReturn(PETSC_SUCCESS);
}

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
 * @brief Renormalizes Gauss points from a [-1, 1]-based
 * domain to the [0, 1]-based reference tetrahedron.
 *
 * @param[in] numPoints The number of Gauss points.
 * @param[in] gaussPoints Input array of points.
 * @param[out] points Output array (numPoints x
 * NUM_DIMENSIONS) for the renormalized coordinates [xi,
 * eta, zeta].
 * @param[out] weights Output array (numPoints) for the
 * renormalized weights.
 * @return PetscErrorCode PETSC_SUCCESS always.
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

static PetscErrorCode invertMatrix(const PetscInt N, const PetscReal A[], PetscReal invA[]) {
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

static PetscErrorCode cartesianToVolumetricCoordinates(const PetscReal r[NUM_DIMENSIONS], PetscReal L[4]) {
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

static PetscErrorCode solve3x3MatrixSystem3x6RHS(const PetscReal matrix1[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal** matrix2,
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
 * @brief Computes shifted scaled Legendre polynomials P_i(y) where y = 2*X - T.
 *
 * Computes Legendre polynomials using the standard three-term recurrence relation,
 * adapted for the scaled variable y = 2*X - T.
 *
 * @param[in] X Coordinate, typically s1 from an oriented edge projection (range depends on T).
 * @param[in] T Scaling parameter, typically s0+s1 from an oriented edge projection.
 * @param[in] nord Maximum polynomial order required (computes P_0 to P_nord).
 * @param[out] P Array to store the computed polynomial values P[0] to P[nord].
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
static PetscErrorCode PolyLegendre(const PetscReal X, const PetscReal T, const PetscInt nord, PetscReal P[]) {

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
 * @brief Computes integrated shifted scaled Legendre polynomials L_i and related terms P, R.
 *
 * @param X [in] Coordinate, typically s1.
 * @param T [in] Scaling parameter, typically s0+s1.
 * @param nord [in] Maximum polynomial order required.
 * @param Idec [in] Boolean flag indicating if T=1 (simplified case).
 * @param homL [out] Output array storing the integrated polynomial values L_i.
 * @param homP [out] Output array storing the Legendre polynomials P_i (derivative w.r.t. X).
 * @param homR [out] Output array storing terms related to the derivative w.r.t. T.
 * @return PetscErrorCode PETSC_SUCCESS always.
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
 * @brief Computes Jacobi polynomials P_j^{alpha, 0} using a recurrence relation adapted for the shifted variable y = 2*X - T.
 *
 * @param[in] X Coordinate, typically s1 from an oriented face projection (range depends on T).
 * @param[in] T Scaling parameter, typically s0+s1 from an oriented face projection.
 * @param[in] nord Maximum polynomial order j required (0 to nord).
 * @param[in] Minalpha The starting value for alpha (alpha increases by 2 for different polynomial families).
 * @param[out] P Output 2D array P[family_index][order_j] storing the polynomial values.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details The output @p P stores polynomials for different alpha values (implicitly indexed by the first dimension,
 *          corresponding to Minalpha, Minalpha+2, …). Used for constructing face and volume basis functions.
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
 * @brief Computes integrated shifted scaled Jacobi polynomials L_j^{alpha,0} and related terms P, R.
 * @param[in] X Coordinate, typically s1 or s2 depending on context (e.g., face or volume basis).
 * @param[in] T Scaling parameter, typically s0+s1 or 1-sd depending on context.
 * @param[in] nord Maximum polynomial order required.
 * @param[in] Minalpha The starting value for alpha.
 * @param[in] Idec Boolean flag indicating if T=1 (simplified case).
 * @param[out] L Output 2D array storing the integrated polynomial values L_j^{alpha,0}.
 * @param[out] P Output 2D array storing the Jacobi polynomials P_j^{alpha,0} (derivative w.r.t. X).
 * @param[out] R Output 2D array storing terms related to the derivative w.r.t. T.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details First calls `PolyJacobi` to get the base polynomials P. Then uses recurrence relations
 *          involving P to compute the integrated polynomials L and the T-derivative related term R.
 *          Used for constructing face and volume basis functions.
 */
// static PetscErrorCode PolyIJacobi(const PetscReal X, const PetscReal T, const PetscInt nord, const PetscInt Minalpha,
//                                   const PetscBool Idec, PetscReal** L, PetscReal** P, PetscReal** R) {
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

static PetscErrorCode AffineTetrahedron(const PetscReal X[NUM_DIMENSIONS], PetscReal Lam[NUM_DIMENSIONS + 1],
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
 * @brief Computes the standard H1 nodal basis functions (barycentric coordinates) and their gradients for a tetrahedron.
 * @param[in] Lam The four barycentric coordinates [L0, L1, L2, L3].
 * @param[in] DLam The gradients of the barycentric coordinates.
 * @param[out] LambV The values of the 4 nodal basis functions (LambV[i] = Lam[i]).
 * @param[out] DLambV The gradients of the 4 nodal basis functions (DLambV[i] = Grad(Lam[i])).
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details This function essentially just copies the barycentric coordinates and their gradients,
 *          as these are the standard P1 nodal basis functions on a tetrahedron.
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
 * @brief Projects tetrahedral barycentric coordinates and gradients onto the 6 edges.
 *
 * @param[in] Lam The four barycentric coordinates [L0, L1, L2, L3].
 * @param[in] DLam The gradients of the barycentric coordinates.
 * @param[out] LampE Projections onto edges. LampE[edge_index][0/1] gives the two relevant barycentric coordinates for that edge.
 * @param[out] DLampE Projections of gradients onto edges. DLampE[edge_index][dim][0/1] gives the two relevant gradients.
 * @param[out] IdecE Boolean flag, always set to PETSC_FALSE as barycentric coordinates on an edge don’t sum to 1 unless the point is
 * on the edge.
 * @return PetscErrorCode PETSC_SUCCESS always.
 */
static PetscErrorCode ProjectTetE(const PetscReal Lam[NUM_DIMENSIONS + 1], const PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1],
                                  PetscReal LampE[NUM_EDGES_PER_CELL][2], PetscReal DLampE[NUM_EDGES_PER_CELL][NUM_DIMENSIONS][2],
                                  PetscBool* IdecE) {

  PetscFunctionBeginUser;

  /* Compute projection */

  /* e=0 --> edge10 with local orientation v1->v0 */
  PetscInt e = 0;
  LampE[e][0] = Lam[1];
  LampE[e][1] = Lam[0];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    DLampE[e][i][0] = DLam[i][1];
    DLampE[e][i][1] = DLam[i][0];
  }

  /* e=1 --> edge02 with local orientation v0->v2 */
  e = 1;
  LampE[e][0] = Lam[0];
  LampE[e][1] = Lam[2];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    DLampE[e][i][0] = DLam[i][0];
    DLampE[e][i][1] = DLam[i][2];
  }

  /* e=2 --> edge21 with local orientation v2->v1 */
  e = 2;
  LampE[e][0] = Lam[2];
  LampE[e][1] = Lam[1];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    DLampE[e][i][0] = DLam[i][2];
    DLampE[e][i][1] = DLam[i][1];
  }

  /* e=3 --> edge13 with local orientation v1->v3 */
  e = 3;
  LampE[e][0] = Lam[1];
  LampE[e][1] = Lam[3];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    DLampE[e][i][0] = DLam[i][1];
    DLampE[e][i][1] = DLam[i][3];
  }

  /* e=4 --> edge30 with local orientation v3->v0 */
  e = 4;
  LampE[e][0] = Lam[3];
  LampE[e][1] = Lam[0];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    DLampE[e][i][0] = DLam[i][3];
    DLampE[e][i][1] = DLam[i][0];
  }

  /* e=5 --> edge23 with local orientation v2->v3 */
  e = 5;
  LampE[e][0] = Lam[2];
  LampE[e][1] = Lam[3];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    DLampE[e][i][0] = DLam[i][2];
    DLampE[e][i][1] = DLam[i][3];
  }

  /* Projected coordinates are Lam, so IdecE=false for all edges */
  *IdecE = PETSC_FALSE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Projects tetrahedral barycentric coordinates and gradients onto the 4 faces.
 * @param[in] Lam The four barycentric coordinates [L0, L1, L2, L3].
 * @param[in] DLam The gradients of the barycentric coordinates.
 * @param[out] LampF Projections onto faces. LampF[face_index][0/1/2] gives the three relevant barycentric coordinates.
 * @param[out] DLampF Projections of gradients onto faces. DLampF[face_index][dim][0/1/2] gives the three relevant gradients.
 * @param[out] IdecF Boolean flag, always set to PETSC_FALSE as barycentric coordinates on a face don't sum to 1 unless the point is on
 * the face.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Maps the 4 barycentric coordinates/gradients to the triplet associated with each of the 4 faces according to a fixed local
 * numbering convention (e.g., face 0 uses L1, L0, L2; face 1 uses L1, L3, L0, etc.).
 */
static PetscErrorCode ProjectTetF(const PetscReal Lam[NUM_DIMENSIONS + 1], const PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1],
                                  PetscReal LampF[NUM_FACES_PER_CELL][NUM_DIMENSIONS],
                                  PetscReal DLampF[NUM_FACES_PER_CELL][NUM_DIMENSIONS][NUM_DIMENSIONS], PetscBool* IdecF) {

  PetscFunctionBeginUser;

  /* Compute projection */

  /* f=0 --> face102 with local orientation v1->v0->v2 */
  PetscInt f = 0;
  LampF[f][0] = Lam[1];
  LampF[f][1] = Lam[0];
  LampF[f][2] = Lam[2];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    DLampF[f][i][0] = DLam[i][1];
    DLampF[f][i][1] = DLam[i][0];
    DLampF[f][i][2] = DLam[i][2];
  }

  /* f=1 --> face130 with local orientation v1->v3->v0 */
  f = 1;
  LampF[f][0] = Lam[1];
  LampF[f][1] = Lam[3];
  LampF[f][2] = Lam[0];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    DLampF[f][i][0] = DLam[i][1];
    DLampF[f][i][1] = DLam[i][3];
    DLampF[f][i][2] = DLam[i][0];
  }

  /* f=2 --> face123 with local orientation v1->v2->v3 */
  f = 2;
  LampF[f][0] = Lam[1];
  LampF[f][1] = Lam[2];
  LampF[f][2] = Lam[3];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    DLampF[f][i][0] = DLam[i][1];
    DLampF[f][i][1] = DLam[i][2];
    DLampF[f][i][2] = DLam[i][3];
  }

  /* f=3 --> face203 with local orientation v2->v0->v3 */
  f = 3;
  LampF[f][0] = Lam[2];
  LampF[f][1] = Lam[0];
  LampF[f][2] = Lam[3];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    DLampF[f][i][0] = DLam[i][2];
    DLampF[f][i][1] = DLam[i][0];
    DLampF[f][i][2] = DLam[i][3];
  }

  *IdecF = PETSC_FALSE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Orients edge-projected coordinates and gradients based on the edge orientation flag.
 * @param[in] S The pair of projected coordinates for the edge [s0, s1].
 * @param[in] DS The pair of projected gradients for the edge [Grad(s0), Grad(s1)].
 * @param[in] Nori The orientation flag (0 for original order, 1 for swapped order).
 * @param[out] GS The oriented coordinates [gs0, gs1].
 * @param[out] GDS The oriented gradients [Grad(gs0), Grad(gs1)].
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details If Nori is 0, GS = S and GDS = DS.
 *          If Nori is 1, GS = [s1, s0] and GDS = [Grad(s1), Grad(s0)].
 */

static PetscErrorCode OrientE(const PetscReal S[2], const PetscReal DS[NUM_DIMENSIONS][2], const PetscInt Nori, PetscReal GS[2],
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
 * @brief Orients face-projected coordinates and gradients based on the face orientation flag.
 * @param[in] S The triplet of projected coordinates for the face [s0, s1, s2].
 * @param[in] DS The triplet of projected gradients for the face [Grad(s0), Grad(s1), Grad(s2)].
 * @param[in] Nori The orientation flag (0-5, representing permutations of the vertices/coordinates).
 * @param[out] GS The oriented coordinates [gs0, gs1, gs2].
 * @param[out] GDS The oriented gradients [Grad(gs0), Grad(gs1), Grad(gs2)].
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Permutes the input coordinates and gradients based on the value of `Nori` (0-5), corresponding
 *          to the 6 possible orientations/permutations of the vertices of a triangle.
 */
static PetscErrorCode OrientTri(const PetscReal S[NUM_DIMENSIONS], const PetscReal DS[NUM_DIMENSIONS][NUM_DIMENSIONS], const PetscInt Nori,
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
 * @brief Computes homogenized integrated Legendre polynomials and their gradients.
 *
 * @param[in] S Affine-like coordinates [s0, s1].
 * @param[in] DS Gradients of S [Grad(s0), Grad(s1)].
 * @param[in] nord Maximum polynomial order required.
 * @param[in] Idec Boolean flag indicating if s0 + s1 = 1.
 * @param[out] PhiE Output array storing the homogenized integrated polynomial values.
 * @param[out] DPhiE Output 2D array storing the gradients of PhiE. DPhiE[dim][order_idx].
 * @return PetscErrorCode PETSC_SUCCESS always.
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
 * @brief Computes homogenized integrated Jacobi polynomials and their gradients.
 * @param[in] S Affine-like coordinates [s_a, s_b] (e.g., [s0+s1, s2] for faces).
 * @param[in] DS Gradients of S [Grad(s_a), Grad(s_b)].
 * @param[in] nord Maximum polynomial order required.
 * @param[in] Minalpha The starting value for alpha.
 * @param[in] Idec Boolean flag indicating if s_a + s_b = 1.
 * @param[out] HomL Output 2D array storing the homogenized integrated polynomial values.
 * @param[out] DHomL Output 3D array storing the gradients of HomL. DHomL[dim][alpha_idx][order_idx].
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Calls `PolyIJacobi` with appropriate arguments (X=s_b, T=s_a+s_b).
 *          Computes the gradient DHomL using the chain rule: DHomL = P * Grad(s_b) + R * Grad(s_a+s_b).
 *          Handles the simplified case where Idec=TRUE (T=1, so R term is not needed).
 */
static PetscErrorCode HomIJacobi(const PetscReal S[2], const PetscReal DS[NUM_DIMENSIONS][2], const PetscInt nord, const PetscInt Minalpha,
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
    // PetscCall(PolyIJacobi(S[1], 1, nord, Minalpha, Idec, HomL, homP, homR));
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
    // PetscCall(PolyIJacobi(S[1], S[0] + S[1], nord, Minalpha, Idec, HomL, homP, homR));
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

static PetscErrorCode AncPhiE(const PetscReal S[2], const PetscReal DS[NUM_DIMENSIONS][2], const PetscInt nord, const PetscBool Idec,
                              PetscReal* PhiE, PetscReal** DPhiE) {
  PetscFunctionBeginUser;

  /* These are precisely the homogenized Legendre polynomials */
  PetscCall(HomILegendre(S, DS, nord, Idec, PhiE, DPhiE));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes H1 ancillary basis functions associated with a triangle face.
 *
 * @param[in] S Oriented face coordinates [s0, s1, s2].
 * @param[in] DS Oriented face gradients [Grad(s0), Grad(s1), Grad(s2)].
 * @param[in] nordFace Polynomial order for the element.
 * @param[in] IdecF Boolean flag (indicating if s0+s1+s2=1).
 * @param[out] PhiTri Output 2D array (nord-2 x nord-2) storing the scalar value of each face ancillary function.
 * @param[out] DPhiTri Output 3D array (NUM_DIMENSIONS x nord-2 x nord-2) storing the gradient of each face ancillary function.
 * @return PetscErrorCode PETSC_SUCCESS always.
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
  // PetscCall(AncPhiE(GLampE, GDLampE, nordFace-minJ, IdecF, PhiE, DPhiE));

  /* Get homogenized Integrated Jacobi polynomials, homLal, and gradients */
  sL[0] = S[0] + S[1];
  sL[1] = S[2];
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    DsL[i][0] = DS[i][0] + DS[i][1];
    DsL[i][1] = DS[i][2];
  }

  /* Compute HomIJacobi */
  // PetscCall(HomIJacobi(sL, DsL, maxJ, minalpha, IdecF, homLal, DhomLal));
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

static PetscErrorCode shape3DHTet(const PetscReal X[NUM_DIMENSIONS], const PetscInt nord, const PetscInt cellOrientation[10],
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
    NoriF[i] = cellOrientation[i];
  }

  /* Extract orientation for edges */
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; ++i) {
    NoriE[i] = cellOrientation[i + NUM_FACES_PER_CELL];
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
      // PetscCall(OrientE(S, D, NoriE[i], GLampE, GDLampE));
      PetscCall(OrientE(S, (const PetscReal(*)[2])D, NoriE[i], GLampE, GDLampE));

      /* Construct the shape functions */
      // PetscCall(AncPhiE(GLampE, GDLampE, nordEdge, IdecE, PhiE, DPhiE));
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
      // PetscCall(OrientTri(tmpLampF, tempDLampF, NoriF[i], GLampF, GDLampF));
      PetscCall(OrientTri(tmpLampF, (const PetscReal(*)[NUM_DIMENSIONS])tempDLampF, NoriF[i], GLampF, GDLampF));

      /* Construct the shape functions */
      // PetscCall(AncPhiTri(GLampF, GDLampF, nordFace, IdecF, PhiTri, DPhiTri));
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
    // PetscCall(AncPhiTri(GLampV, GDLampV, nordB - minK - 1, IdecB[0], PhiTriV, DPhiTriV));
    PetscCall(AncPhiTri(GLampV, (const PetscReal(*)[NUM_DIMENSIONS])GDLampV, nordB - minK - 1, IdecB[0], PhiTriV, DPhiTriV));

    PetscReal tmp1[2] = {1 - Lam[3], Lam[3]};
    PetscReal tmp2[NUM_DIMENSIONS][2];

    /* Initialize input matrix */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
      tmp2[i][0] = -DLam[i][3];
      tmp2[i][1] = DLam[i][3];
    }

    // PetscCall(HomIJacobi(tmp1, tmp2, maxK, minbeta, IdecB[1], homLbetV, DhomLbetV));
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

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes a rotation vector based on azimuth and
 * dip angles.
 * @param[in] azimuth Rotation angle in the x-y plane
 * (degrees).
 * @param[in] dip Rotation angle in the x-z plane (degrees).
 * @param[out] rotationVector The resulting 3D unit vector
 * after rotation.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Starts with a base vector [1, 0, 0]. Converts
 * azimuth and dip to radians. Applies rotation matrices
 * sequentially (x-y plane first, then x-z plane). A y-z
 * plane rotation (tetha) is included but currently
 * hardcoded to 0 degrees. The final rotated vector is
 * stored in `rotationVector`.
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
 * @brief Transforms global XYZ coordinates to reference
 * tetrahedron coordinates (Xi, Eta, Zeta).
 * @param[in] cellCoords Spatial coordinates of the
 * tetrahedron's 4 vertices (PetscScalar array, size 12).
 * @param[in] point The global [x, y, z] coordinates of the
 * point to transform.
 * @param[out] XiEtaZeta The resulting reference coordinates
 * [xi, eta, zeta].
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Computes the inverse of the affine mapping from
 * the reference tetrahedron (vertices at (0,0,0), (1,0,0),
 * (0,1,0), (0,0,1)) to the physical tetrahedron defined by
 * `cellCoords`. Uses Cramer's rule / determinant formulas.
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
 * @brief Computes the Jacobian matrix and its inverse for
 * the affine mapping from the reference tetrahedron to the
 * physical tetrahedron.
 * @param[in] cellCoords Spatial coordinates of the
 * tetrahedron's 4 vertices (PetscScalar array, size 12).
 * @param[out] jacobian The 3x3 Jacobian matrix.
 * @param[out] invJacobian The 3x3 inverse Jacobian matrix.
 * @param[out] detJacobian The determinant of the Jacobian
 * matrix.
 * @return PetscErrorCode PETSC_SUCCESS always.
 * @details Calculates the Jacobian matrix based on the
 * differences between vertex coordinates. Computes the
 * determinant, cofactor matrix, adjugate matrix, and
 * finally the inverse Jacobian.
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

  /* Get orientation for faces */
  currentPoint = 2;
  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; i++) {
    cell->orientation[i] = cell->closure[currentPoint + 1];

    /* Cast to PETGEM basis functions orientation */
    switch (cell->orientation[i]) {
    case -3:
      cell->orientation[i] = 4;
      break;
    case -2:
      cell->orientation[i] = 3;
      break;
    case -1:
      cell->orientation[i] = 5;
      break;
    case 0:
      cell->orientation[i] = 0;
      break;
    case 1:
      cell->orientation[i] = 1;
      break;
    case 2:
      cell->orientation[i] = 2;
      break;
    default:
      break;
    }
    currentPoint += 2;
  }

  /* Get orientation for edges */
  currentPoint = 2 + NUM_FACES_PER_CELL * 2;
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; i++) {
    cell->orientation[i + NUM_FACES_PER_CELL] = cell->closure[currentPoint + 1];

    /* Cast to PETGEM basis functions orientation */
    if (cell->orientation[i + NUM_FACES_PER_CELL] < 0) {
      cell->orientation[i + NUM_FACES_PER_CELL] = -1;
    } else {
      cell->orientation[i + NUM_FACES_PER_CELL] = 1;
    }
    currentPoint += 2;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode computeNum1DQuadraturePoints(const PetscInt nord, Quadrature1D* quadrature) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt gaussOrder, numPoints;

  /* Compute gauss order*/
  gaussOrder = 2 * nord;

  /* Basic verification */
  PetscCheck(gaussOrder >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error: Negative polynomial orders are not supported");
  PetscCheck(gaussOrder <= 11, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Error: 1D polynomial orders higher than 11 are not supported");

  /* Gauss-Legendre rule:
     exact for polynomials of degree (2*numPoints - 1)
     Saturated to 11 points for high orders */
  numPoints = gaussOrder / 2 + 1;

  if (numPoints > 11) {
    numPoints = 11;
  }

  quadrature->numPoints = numPoints;

  PetscFunctionReturn(PETSC_SUCCESS);
}

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
 * @brief Determines the number of Gauss points and its
 * Weights required for integrating polynomials up to a
 * given order on a tetrahedron.
 *
 * @param[in] nord The basis order (determines the required
 * integration order 2*params.nord).
 * @param[out] numGaussPoints Pointer to store the required
 * number of Gauss points.
 * @return PetscErrorCode PETSC_SUCCESS on success. Returns
 * error code if the required Gauss order (2*nord) is out of
 * the supported range [1, 12].
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
 * @brief Provides coordinates and weights for Gauss
 * quadrature on the reference tetrahedron for various
 * orders.
 *
 * @param numPoints [in] The desired number of Gauss points.
 * @param points [out] Output array (numPoints x
 * NUM_DIMENSIONS) for the coordinates [xi, eta, zeta].
 * @param weights [out] Output array (numPoints) for the
 * weights.
 * @return PetscErrorCode PETSC_SUCCESS always.
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

PetscErrorCode computeNedelecOrder1Coefficients(const PetscInt nord, PetscReal** coeffs, PetscReal** Dx_Ni, PetscReal** Dy_Ni,
                                                PetscReal** Dz_Ni) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  Mat matrix, secm, coef;
  IS row, col;
  PetscReal EPS = 1e-14;
  PetscScalar* coef_array;
  PetscReal val;
  PetscInt numDofInCell;

  /* Compute number of dofs per cell */
  numDofInCell = nord * (nord + 2) * (nord + 3) / 2;

  /* Define data for matrix */
  PetscScalar matriz_data[36] = {1.0,  0.0,  0.0,  0.0,  0.0, 0.0,   /* Row 0 */
                                 -1.0, 1.0,  0.0,  -1.0, 0.0, 0.0,   /* Row 1 */
                                 0.0,  -1.0, 0.0,  0.0,  0.0, 0.0,   /* Row 2 */
                                 0.0,  0.0,  1.0,  0.0,  0.0, 0.0,   /* Row 3 */
                                 1.0,  0.0,  -1.0, 0.0,  1.0, 0.0,   /* Row 4 */
                                 0.0,  -1.0, 1.0,  0.0,  0.0, -1.0}; /* Row 5 */

  /* Identity matrix */
  PetscScalar secm_data[36] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};

  /* Create matrices from arrays */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, numDofInCell, numDofInCell, matriz_data, &matrix));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, numDofInCell, numDofInCell, secm_data, &secm));
  PetscCall(MatDuplicate(secm, MAT_DO_NOT_COPY_VALUES, &coef));

  /* Compute index sets */
  PetscCall(ISCreateStride(PETSC_COMM_SELF, numDofInCell, 0, 1, &row));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, numDofInCell, 0, 1, &col));

  /* Factor LU */
  PetscCall(MatLUFactor(matrix, row, col, NULL));

  /* Solve matrix * coef = secm (coef = matriz \ secm) */
  PetscCall(MatMatSolve(matrix, secm, coef));

  /* Get pointer array for coef */
  PetscCall(MatDenseGetArray(coef, &coef_array));

  /* Apply threshold and fill output matrix */
  for (PetscInt i = 0; i < numDofInCell; i++) {
    for (PetscInt j = 0; j < numDofInCell; j++) {
      val = PetscRealPart(coef_array[i * numDofInCell + j]);
      if (PetscAbsReal(val) < EPS)
        val = 0.0;
      coeffs[i][j] = val;
    }
  }

  /* Restore array */
  PetscCall(MatDenseRestoreArray(coef, &coef_array));

  /* Compute derivatives */
  for (PetscInt i = 0; i < numDofInCell; i++) {
    Dx_Ni[0][i] = 0.0;          // DxNix
    Dx_Ni[1][i] = coeffs[3][i]; // DxNiy
    Dx_Ni[2][i] = coeffs[4][i]; // DxNiz

    Dy_Ni[0][i] = -coeffs[3][i]; // DyNix
    Dy_Ni[1][i] = 0.0;           // DyNiy
    Dy_Ni[2][i] = coeffs[5][i];  // DyNiz

    Dz_Ni[0][i] = -coeffs[4][i]; // DzNix
    Dz_Ni[1][i] = -coeffs[5][i]; // DzNiy
    Dz_Ni[2][i] = 0.0;           // DzNiz
  }

  /* Free memory */
  PetscCall(MatDestroy(&matrix));
  PetscCall(MatDestroy(&secm));
  PetscCall(MatDestroy(&coef));
  PetscCall(ISDestroy(&col));
  PetscCall(ISDestroy(&row));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode computeNedelecOrder1BasisFunctions(const PetscInt nord, const PetscReal point[NUM_DIMENSIONS],
                                                  const PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], const PetscReal* const* coeffs,
                                                  PetscReal** Ni) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt numDofInCell;
  PetscReal L[4];
  PetscReal rref[NUM_DIMENSIONS];
  PetscReal jacobianReferenceCell[NUM_DIMENSIONS * NUM_DIMENSIONS], invJacobianReferenceCell[NUM_DIMENSIONS * NUM_DIMENSIONS];
  PetscReal **Ni_Reference, **Ni_ReferenceTmp;
  PetscReal dotResult;

  /* Compute number of dofs per cell */
  numDofInCell = nord * (nord + 2) * (nord + 3) / 2;

  /* Variables that depends on nord */
  PetscReal aux_x[numDofInCell], aux_y[numDofInCell], aux_z[numDofInCell];

  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Ni_Reference));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Ni_ReferenceTmp));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(numDofInCell, &Ni_Reference[i]));
    PetscCall(PetscCalloc1(numDofInCell, &Ni_ReferenceTmp[i]));
  }

  /* Compute r in reference cell */
  PetscCall(cartesianToVolumetricCoordinates(point, L));

  /* Perform dot product */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    dotResult = 0.0;
    for (PetscInt j = 0; j < 4; j++) {
      dotResult += L[j] * REFERENCE_CELL[i][j];
    }
    rref[i] = dotResult;
  }

  /* Evaluation of NiRef(rref) */
  aux_x[0] = 1.0;
  aux_x[1] = 0.0;
  aux_x[2] = 0.0;
  aux_x[3] = rref[1];
  aux_x[4] = rref[2];
  aux_x[5] = 0.0;

  aux_y[0] = 0.0;
  aux_y[1] = 1.0;
  aux_y[2] = 0.0;
  aux_y[3] = -rref[0];
  aux_y[4] = 0.0;
  aux_y[5] = rref[2];

  aux_z[0] = 0.0;
  aux_z[1] = 0.0;
  aux_z[2] = 1.0;
  aux_z[3] = 0.0;
  aux_z[4] = -rref[0];
  aux_z[5] = -rref[1];

  /* Row vector * matrix multiplication */
  for (PetscInt i = 0; i < numDofInCell; i++) {
    Ni_Reference[0][i] = 0.0;
    Ni_Reference[1][i] = 0.0;
    Ni_Reference[2][i] = 0.0;

    for (PetscInt j = 0; j < numDofInCell; j++) {
      Ni_Reference[0][i] += aux_x[j] * coeffs[j][i];
      Ni_Reference[1][i] += aux_y[j] * coeffs[j][i];
      Ni_Reference[2][i] += aux_z[j] * coeffs[j][i];
    }
  }

  /* Build Jacobian in column-major order */
  jacobianReferenceCell[0 + 0 * NUM_DIMENSIONS] = REFERENCE_CELL[0][1] - REFERENCE_CELL[0][0];
  jacobianReferenceCell[1 + 0 * NUM_DIMENSIONS] = REFERENCE_CELL[1][1] - REFERENCE_CELL[1][0];
  jacobianReferenceCell[2 + 0 * NUM_DIMENSIONS] = REFERENCE_CELL[2][1] - REFERENCE_CELL[2][0];

  jacobianReferenceCell[0 + 1 * NUM_DIMENSIONS] = REFERENCE_CELL[0][2] - REFERENCE_CELL[0][0];
  jacobianReferenceCell[1 + 1 * NUM_DIMENSIONS] = REFERENCE_CELL[1][2] - REFERENCE_CELL[1][0];
  jacobianReferenceCell[2 + 1 * NUM_DIMENSIONS] = REFERENCE_CELL[2][2] - REFERENCE_CELL[2][0];

  jacobianReferenceCell[0 + 2 * NUM_DIMENSIONS] = REFERENCE_CELL[0][3] - REFERENCE_CELL[0][0];
  jacobianReferenceCell[1 + 2 * NUM_DIMENSIONS] = REFERENCE_CELL[1][3] - REFERENCE_CELL[1][0];
  jacobianReferenceCell[2 + 2 * NUM_DIMENSIONS] = REFERENCE_CELL[2][3] - REFERENCE_CELL[2][0];

  /* Invert Jacobian */
  PetscCall(invertMatrix(NUM_DIMENSIONS, jacobianReferenceCell, invJacobianReferenceCell));

  /* Transform Ni_Reference -> Ni */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < numDofInCell; j++) {
      Ni_ReferenceTmp[i][j] = 0.0;
      for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
        Ni_ReferenceTmp[i][j] += invJacobianReferenceCell[i * NUM_DIMENSIONS + k] * Ni_Reference[k][j];
      }
    }
  }

  /* Transform basis from reference cell to real cell */
  PetscCall(solve3x3MatrixSystem3x6RHS(jacobian, Ni_ReferenceTmp, Ni));

  /* Free memory */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscFree(Ni_Reference[i]));
    PetscCall(PetscFree(Ni_ReferenceTmp[i]));
  }
  PetscCall(PetscFree(Ni_Reference));
  PetscCall(PetscFree(Ni_ReferenceTmp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode computeNedelecOrder1BasisFunctionCurls(const PetscInt nord, const PetscReal* const* Dx_Ni, const PetscReal* const* Dy_Ni,
                                                      const PetscReal* const* Dz_Ni,
                                                      const PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], const PetscReal detJacobian,
                                                      PetscReal** NiCurl) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt numDofInCell;
  PetscReal A[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, -1.0, 0.0}};

  PetscReal B[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{0.0, 0.0, -1.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};

  PetscReal C[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  /* Compute number of dofs per cell */
  numDofInCell = nord * (nord + 2) * (nord + 3) / 2;

  /* Variables that depends on nord */
  PetscReal curlReferenceCell[NUM_DIMENSIONS][numDofInCell];

  /* Initialize array */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    for (PetscInt j = 0; j < numDofInCell; j++) {
      curlReferenceCell[i][j] = 0.0;
    }
  }

  /* Compute curl in reference cell */
  for (PetscInt i = 0; i < numDofInCell; i++) {
    curlReferenceCell[0][i] += A[0][1] * Dx_Ni[1][i] + A[0][2] * Dx_Ni[2][i] - A[0][1] * Dy_Ni[0][i] + A[1][2] * Dy_Ni[2][i] -
                               A[0][2] * Dz_Ni[0][i] - A[1][2] * Dz_Ni[1][i];
    curlReferenceCell[1][i] += B[0][1] * Dx_Ni[1][i] + B[0][2] * Dx_Ni[2][i] - B[0][1] * Dy_Ni[0][i] + B[1][2] * Dy_Ni[2][i] -
                               B[0][2] * Dz_Ni[0][i] - B[1][2] * Dz_Ni[1][i];
    curlReferenceCell[2][i] += C[0][1] * Dx_Ni[1][i] + C[0][2] * Dx_Ni[2][i] - C[0][1] * Dy_Ni[0][i] + C[1][2] * Dy_Ni[2][i] -
                               C[0][2] * Dz_Ni[0][i] - C[1][2] * Dz_Ni[1][i];
  }

  /* Transform to real cell directly */
  for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
    for (PetscInt i = 0; i < numDofInCell; i++) {
      NiCurl[j][i] = 0.0;
      for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
        NiCurl[j][i] += jacobian[k][j] * curlReferenceCell[k][i];
      }
      NiCurl[j][i] /= detJacobian;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode computeElementalMatrices(const PetscInt nord, const PetscInt numDofInCell, const Cell* cell, const Quadrature3D* quadrature,
                                        PetscReal** Me, PetscReal** Ke) {
  PetscFunctionBeginUser;

  /* Variable declarations */
  PetscReal e_r[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{0.0}};
  PetscReal mu_r[NUM_DIMENSIONS][NUM_DIMENSIONS] = {{0.0}};
  PetscReal iPoint[NUM_DIMENSIONS] = {0.0};
  PetscReal **Ni, **NiCurl, **coeffs, **Dx_Ni, **Dy_Ni, **Dz_Ni;

  /* Tensor for integration (Vertical transverse electric
   * permitivity) */
  e_r[0][0] = cell->resistivity[0];
  e_r[1][1] = cell->resistivity[1];
  e_r[2][2] = cell->resistivity[2];

  /* Tensor for integration (Constant magnetic permittivity)
   */
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

  /* Create the const views for arrays */
  const PetscReal** coeffs_const = (const PetscReal**)coeffs;
  const PetscReal** Dx_Ni_const = (const PetscReal**)Dx_Ni;
  const PetscReal** Dy_Ni_const = (const PetscReal**)Dy_Ni;
  const PetscReal** Dz_Ni_const = (const PetscReal**)Dz_Ni;

  /* Reset elemental matrices */
  for (PetscInt i = 0; i < numDofInCell; ++i) {
    for (PetscInt j = 0; j < numDofInCell; ++j) {
      Me[i][j] = 0.0;
      Ke[i][j] = 0.0;
    }
  }

  /* Compute elemental matrices (mass and stifness matrix)
   */
  switch (nord) {
  case 1: {
    /* Local variables */
    PetscInt signs[NUM_EDGES_PER_CELL];
    PetscReal Ni_i[NUM_DIMENSIONS], Ni_j[NUM_DIMENSIONS], tmp[NUM_DIMENSIONS], value;

    /* Extract edges signs from cellOrientation */
    for (PetscInt i = 4; i < NUM_EDGES_PER_CELL + 4; i++) {
      signs[i - 4] = cell->orientation[i];
    }

    /* Compute nedelec coefficients and its derivatives */
    PetscCall(computeNedelecOrder1Coefficients(nord, coeffs, Dx_Ni, Dy_Ni, Dz_Ni));

    /* Compute basis functions for all gauss points */
    for (PetscInt i = 0; i < quadrature->numPoints; ++i) {
      /* Get gauss for i point */
      iPoint[0] = quadrature->points[i][0];
      iPoint[1] = quadrature->points[i][1];
      iPoint[2] = quadrature->points[i][2];

      for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
        for (PetscInt k = 0; k < numDofInCell; k++) {
          Ni[j][k] = 0.0;
          NiCurl[j][k] = 0.0;
        }
      }

      /* Compute basis functions for point i (this function
       * returns the basis function in the real cell) */
      PetscCall(computeNedelecOrder1BasisFunctions(nord, iPoint, cell->jacobian, coeffs_const, Ni));

      /* Perform mass matrix integral */
      for (PetscInt j = 0; j < numDofInCell; j++) {
        for (PetscInt k = 0; k < numDofInCell; k++) {

          /* Extract column j and k from Ni */
          for (PetscInt l = 0; l < NUM_DIMENSIONS; l++) {
            Ni_i[l] = Ni[l][j];
            Ni_j[l] = Ni[l][k];
          }

          /* Compute tmp = e_r * Ni_j   (3x3 * 3x1) */
          for (PetscInt l = 0; l < NUM_DIMENSIONS; l++) {
            tmp[l] = 0.0;
            for (PetscInt m = 0; m < NUM_DIMENSIONS; m++) {
              tmp[l] += e_r[l][m] * Ni_j[m];
            }
          }

          /* Compute scalar product Ni_i' * tmp */
          value = 0.0;
          for (PetscInt l = 0; l < NUM_DIMENSIONS; l++) {
            value += Ni_i[l] * tmp[l];
          }

          /* Apply the remaining scalar multipliers */
          Me[j][k] += (quadrature->weights[i] * value * signs[j] * signs[k] * cell->detJacobian);
        }
      }

      /* Compute curl basis functions */
      PetscCall(
          computeNedelecOrder1BasisFunctionCurls(nord, Dx_Ni_const, Dy_Ni_const, Dz_Ni_const, cell->jacobian, cell->detJacobian, NiCurl));

      /* Perform stiffness matrix integral */
      for (PetscInt j = 0; j < numDofInCell; j++) {
        for (PetscInt k = 0; k < numDofInCell; k++) {
          value = mu_r[0][0] * NiCurl[0][j] * NiCurl[0][k] + mu_r[0][1] * (NiCurl[0][j] * NiCurl[1][k] + NiCurl[1][j] * NiCurl[0][k]) +
                  mu_r[1][1] * NiCurl[1][j] * NiCurl[1][k] + mu_r[0][2] * (NiCurl[0][j] * NiCurl[2][k] + NiCurl[2][j] * NiCurl[0][k]) +
                  mu_r[1][2] * (NiCurl[1][j] * NiCurl[2][k] + NiCurl[2][j] * NiCurl[1][k]) + mu_r[2][2] * NiCurl[2][j] * NiCurl[2][k];

          Ke[j][k] += (quadrature->weights[i] * value * signs[j] * signs[k] * cell->detJacobian);
        }
      }
    }
    break;
  }
  case 2: {
    break;
  }
  default: {
    break;
  }
  }

  /* Free memory */
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

PetscErrorCode computeElementalGradientMatrix(const PetscInt nord, const PetscInt numDofInCell, const PetscInt numH1DofInCell,
                                              const Cell* cell, const Quadrature1D* quadrature, PetscReal** gradientMatrix) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt edgeVerticesLocal[NUM_VERTICES_PER_EDGE];
  PetscReal edgeJacobian[NUM_DIMENSIONS], edgeJacobianUnitVector[NUM_DIMENSIONS], originCoordinates[NUM_DIMENSIONS];
  PetscReal point3D[NUM_DIMENSIONS];
  PetscReal normJacobian, tmp, tmp_v1[NUM_DIMENSIONS];
  PetscInt MAXtetraH;
  PetscReal *ShapH, **GradH;
  PetscReal qEvaluated;
  PetscInt m;

  /* Allocate matrices for shape functions */
  MAXtetraH = ((nord + 3) * (nord + 2) * (nord + 1)) / 6;

  PetscCall(PetscCalloc1(MAXtetraH, &ShapH));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &GradH));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(MAXtetraH, &GradH[i]));
  }

  /* Reset gradient matrix */
  for (PetscInt i = 0; i < numDofInCell; ++i) {
    for (PetscInt j = 0; j < numH1DofInCell; ++j) {
      gradientMatrix[i][j] = 0.;
    }
  }

  /* Loop over H1 dofs */
  for (PetscInt i = 0; i < numH1DofInCell; i++) {

    m = 0;

    /* Loop over edges */
    for (PetscInt j = 0; j < NUM_EDGES_PER_CELL; j++) {

      /* Setup node indexes for edge j */
      for (PetscInt k = 0; k < NUM_VERTICES_PER_EDGE; k++) {
        edgeVerticesLocal[k] = EDGE_VERTICES[j][k];
      }

      /* Compute edge jacobian */
      for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
        edgeJacobian[k] = REFERENCE_CELL[k][edgeVerticesLocal[1]] - REFERENCE_CELL[k][edgeVerticesLocal[0]];
      }

      /* Compute edge jacobian unit vector */
      tmp = 0.0;
      for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
        tmp += edgeJacobian[k] * edgeJacobian[k];
      }

      normJacobian = PetscSqrtReal(tmp);

      for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
        edgeJacobianUnitVector[k] = edgeJacobian[k] / normJacobian;
      }

      /* Setup origin coordinates (first edge vertice) */
      for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
        originCoordinates[k] = REFERENCE_CELL[k][edgeVerticesLocal[0]];
      }

      /* Compute gradient matrix (loop over quadrature points) */
      for (PetscInt k = 0; k < quadrature->numPoints; k++) {

        /* Translate 1d quadrature point to 3d space */
        for (PetscInt l = 0; l < NUM_DIMENSIONS; l++) {
          point3D[l] = edgeJacobian[l] * quadrature->points[l] + originCoordinates[l];
        }

        /* Compute H1 gradient */
        PetscCall(shape3DHTet(point3D, nord, cell->orientation, ShapH, GradH));

        switch (nord) {
        case 1: {
          qEvaluated = 1.0;

          /* Init array and compute dot product */
          for (PetscInt l = 0; l < NUM_DIMENSIONS; l++) {
            tmp_v1[l] = GradH[l][i];
          }
          PetscCall(dotProduct(tmp_v1, edgeJacobianUnitVector, &tmp));

          /* Perform integral */
          gradientMatrix[m][i] += quadrature->weights[k] * tmp * cell->orientation[4 + j] * qEvaluated * normJacobian;
          break;
        }
        default: {
          break;
        }
        }
      }

      switch (nord) {
      case 1: {
        m += 1;
      }
      }
    }
  }

  /* Free memory */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscFree(GradH[i]));
  }
  PetscCall(PetscFree(GradH));
  PetscCall(PetscFree(ShapH));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printCellEntities(const DM dm, const PetscInt cell) {
  PetscFunctionBeginUser;

  /* Variable declarations */
  PetscInt cellFaces[NUM_FACES_PER_CELL];
  PetscInt cellEdges[NUM_EDGES_PER_CELL];
  PetscInt faceEdges[NUM_FACES_PER_CELL][NUM_EDGES_PER_FACE];
  PetscInt faceVertices[NUM_FACES_PER_CELL][NUM_VERTICES_PER_FACE];
  PetscInt edgeVertices[NUM_EDGES_PER_CELL][NUM_VERTICES_PER_EDGE];

  PetscInt transitiveClosureCellSize;
  PetscInt* transitiveClosureCellPoints = NULL;
  PetscInt transitiveClosureFaceSize;
  PetscInt* transitiveClosureFacePoints = NULL;
  const PetscInt* conePoints;
  PetscInt currentPoint;
  PetscInt currentFace;
  PetscBool isDG;
  PetscInt numCoords;
  const PetscScalar* arrayCoords;
  PetscScalar* cellCoords = NULL;

  PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &transitiveClosureCellSize, &transitiveClosureCellPoints));

  /* Get faces indices for cell, edges for each face, and vertices for each face */
  currentPoint = 2;
  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; i++) {
    /* Face indexes */
    cellFaces[i] = transitiveClosureCellPoints[currentPoint + i * 2];
    PetscCall(DMPlexGetCone(dm, cellFaces[i], &conePoints));

    /* Edges for each face*/
    for (PetscInt j = 0; j < NUM_EDGES_PER_FACE; j++) {
      faceEdges[i][j] = conePoints[j];
    }

    /* Vertices for each face */
    /* Orden convention:
    - Edges indices start on position 2
    - Vertices indices start on position 2 + NUM_EDGES_PER_FACE * 2
    */
    PetscCall(DMPlexGetTransitiveClosure(dm, cellFaces[i], PETSC_TRUE, &transitiveClosureFaceSize, &transitiveClosureFacePoints));
    currentFace = 8;
    for (PetscInt k = 0; k < NUM_VERTICES_PER_FACE; k++) {
      faceVertices[i][k] = transitiveClosureFacePoints[currentFace + k * 2];
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, cellFaces[i], PETSC_TRUE, &transitiveClosureFaceSize, &transitiveClosureFacePoints));
  }

  /* Get edges indices for cell */
  currentPoint = 2 + NUM_FACES_PER_CELL * 2;
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; i++) {
    cellEdges[i] = transitiveClosureCellPoints[currentPoint + i * 2];
    PetscCall(DMPlexGetCone(dm, cellEdges[i], &conePoints));
    for (PetscInt j = 0; j < NUM_VERTICES_PER_EDGE; j++) {
      edgeVertices[i][j] = conePoints[j];
    }
  }

  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\nData for cell %d:\n", cell));

  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[Cell % d] transitive closure size = % d\n ", cell, transitiveClosureCellSize));

  for (PetscInt i = 0; i < transitiveClosureCellSize; i++) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, " [Cell % d] closure entry %d = point % d(orientation % d)\n ", cell, i,
                                      transitiveClosureCellPoints[2 * i], transitiveClosureCellPoints[2 * i + 1]));
  }
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));

  /* Face --> vertices connectivity */
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Face vertices:\n"));
  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; i++) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "  [Face %d] vertices = (%d, %d, %d)\n", cellFaces[i], faceVertices[i][0],
                                      faceVertices[i][1], faceVertices[i][2]));
  }
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));

  /* Edge --> vertices connectivity */
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Edge vertices:\n"));
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; i++) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[Edge % d] vertices = (% d, % d)\n ", cellEdges[i], edgeVertices[i][0],
                                      edgeVertices[i][1]));
  }
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));

  /* Face --> edges connectivity */
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Face edges:\n"));
  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; i++) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[Face % d] edges = (% d, % d, % d)\n ", cellFaces[i], faceEdges[i][0],
                                      faceEdges[i][1], faceEdges[i][2]));
  }
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));

  /* Get/print cell coordinates */
  PetscCall(DMPlexGetCellCoordinates(dm, cell, &isDG, &numCoords, &arrayCoords, &cellCoords));

  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Vertex coordinates:\n"));
  currentPoint = 2 + NUM_FACES_PER_CELL * 2 + NUM_EDGES_PER_CELL * 2;
  for (PetscInt i = 0; i < NUM_VERTICES_PER_CELL; i++) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[Vertex %d] coordinates = (%g, %g, %g)\n",
                                      transitiveClosureCellPoints[currentPoint], PetscRealPart(cellCoords[3 * i + 0]),
                                      PetscRealPart(cellCoords[3 * i + 1]), PetscRealPart(cellCoords[3 * i + 2])));
    currentPoint += 2;
  }
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));

  /* Print edge midpoints */
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Edge midpoints:\n"));
  currentPoint = 2 + NUM_FACES_PER_CELL * 2;
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; i++) {
    PetscInt v0 = EDGE_VERTICES[i][0];
    PetscInt v1 = EDGE_VERTICES[i][1];

    PetscReal xm = 0.5 * (PetscRealPart(cellCoords[3 * v0 + 0]) + PetscRealPart(cellCoords[3 * v1 + 0]));
    PetscReal ym = 0.5 * (PetscRealPart(cellCoords[3 * v0 + 1]) + PetscRealPart(cellCoords[3 * v1 + 1]));
    PetscReal zm = 0.5 * (PetscRealPart(cellCoords[3 * v0 + 2]) + PetscRealPart(cellCoords[3 * v1 + 2]));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[Edge %d] midpoint coordinates = (%g, %g, %g)\n", transitiveClosureCellPoints[currentPoint],
                          xm, ym, zm));
    currentPoint += 2;
  }
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));

  /* Restore transitive clousure and cell coordinates */
  PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &transitiveClosureCellSize, &transitiveClosureCellPoints));
  PetscCall(DMPlexRestoreCellCoordinates(dm, cell, &isDG, &numCoords, &arrayCoords, &cellCoords));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Checks if the discrete gradient is in the kernel
 * of the mass matrix M * G == 0.
 * @param[in] M Pointer to the mass matrix data (row-major).
 * @param[in] G Pointer to the discrete gradient matrix data
 * (row-major).
 * @param[in] m Number of rows in M and G.
 * @param[in] n Number of columns in G (number of rows in H1
 * space).
 * @param[in] w Element identifier (for error reporting).
 * @return PetscErrorCode PETSC_SUCCESS on success.
 * @details This function verifies the property M * G = 0
 * for a given element's mass matrix (M) and discrete
 * gradient matrix (G). It prints an error message if the
 * product is not close to zero within PETSC_SMALL
 * tolerance. The check is currently disabled by the `#if 0`
 * block.
 */

PetscErrorCode checkDiscreteGradientKernel(const PetscReal* M, const PetscReal* G, const PetscInt m, const PetscInt n, const PetscInt cell) {
  PetscFunctionBeginUser;

  /* Activate/deactive printing */
#if 0
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Discrete gradient for cell %d:\n", cell));
  
  for (PetscInt i = 0; i < m; i++) {
    for (PetscInt j = 0; j < n; j++) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%g:\n", G[i*n + j]));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
#endif

  /* Compute discrete gradient */  
  for (PetscInt i = 0; i < m; i++) {
    for (PetscInt j = 0; j < n; j++) {
      PetscReal v = 0;
      for (PetscInt k = 0; k < m; k++) {
        /* M is m x m, G is m x n */ 
        v += M[i * m + k] * G[k * n + j];
      }
      if (!PetscIsCloseAtTol(v, 0, 0, PETSC_SMALL)) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: discrete gradient is not in the kernel of the mass matrix for cell %d (%d, %d) \n", cell, i, j));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}





