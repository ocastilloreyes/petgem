/*
 * Filename: hvfem_internal.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Private header shared by the per-order Nédélec implementation
 * files and the common hvfem.c hub. Not intended for inclusion
 * by code outside src/hvfem*.c.
 */

/*
 * Notes:
 * Exposes the geometric/polynomial helpers that live in hvfem.c plus
 * the per-order ops tables consumed by the selector.
 */

#ifndef HVFEM_INTERNAL_H
#define HVFEM_INTERNAL_H

#include "constants.h"
#include "grid.h"
#include "hvfem.h"
#include <petsc.h>

/* ---------------------------------------------------------------------------
 * Shared geometric / linear-algebra helpers (defined in hvfem.c).
 * ------------------------------------------------------------------------- */

/**
 * @brief Computes the cross product result = a × b of two 3-vectors.
 *
 * @param[in]  a       First 3-vector.
 * @param[in]  b       Second 3-vector.
 * @param[out] result  Cross product a × b.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode crossProduct(const PetscReal a[NUM_DIMENSIONS],
                            const PetscReal b[NUM_DIMENSIONS],
                            PetscReal result[NUM_DIMENSIONS]);

/**
 * @brief Inverts a dense N×N matrix.
 *
 * @param[in]  N     Matrix dimension.
 * @param[in]  A     Input matrix, row-major with N² entries.
 * @param[out] invA  Inverse of A, row-major with N² entries.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode invertMatrix(const PetscInt N, const PetscReal A[], PetscReal invA[]);

/**
 * @brief Evaluates H1 hierarchical shape functions and gradients on the
 *        reference tetrahedron.
 *
 * Used by the discrete-gradient builder (buildDiscreteGradientMatrix) for the
 * P_nord nodal values and gradients at quadrature points.
 *
 * @param[in]  X                Reference coordinates in the tetrahedron.
 * @param[in]  nord             Polynomial order.
 * @param[in]  cellOrientation  Edge/face orientation of the cell.
 * @param[out] ShapH            Shape-function values.
 * @param[out] GradH            Shape-function gradients.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode shape3DHTet(const PetscReal X[NUM_DIMENSIONS], const PetscInt nord,
                           const CellOrientation *cellOrientation,
                           PetscReal *ShapH, PetscReal **GradH);

/* ---------------------------------------------------------------------------
 * Hierarchical-basis prerequisites (shared between hvfem.c and the
 * hierarchical Nédélec TU hvfem_hierarchical.c). These were file-static in
 * hvfem.c when only shape3DHTet consumed them; the hierarchical H(curl)
 * basis (shape3DETet) needs the same machinery, so they are exposed via
 * this internal header rather than duplicated.
 * ------------------------------------------------------------------------- */
/**
 * @brief Computes the affine (barycentric) coordinates and gradients on the
 *        reference tetrahedron.
 *
 * @param[in]  X     Reference coordinates in the tetrahedron.
 * @param[out] Lam   The four barycentric coordinates.
 * @param[out] DLam  Gradients of the barycentric coordinates.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode AffineTetrahedron(const PetscReal X[NUM_DIMENSIONS],
                                 PetscReal Lam[NUM_DIMENSIONS + 1],
                                 PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1]);

/**
 * @brief Projects the affine coordinates onto the tetrahedron edges.
 *
 * @param[in]  Lam     Barycentric coordinates.
 * @param[in]  DLam    Gradients of the barycentric coordinates.
 * @param[out] LampE   Per-edge projected affine coordinate pairs.
 * @param[out] DLampE  Gradients of the per-edge projected coordinates.
 * @param[out] IdecE   Per-edge "decoupled" flag for the orientation logic.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode ProjectTetE(const PetscReal Lam[NUM_DIMENSIONS + 1],
                           const PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1],
                           PetscReal LampE[NUM_EDGES_PER_CELL][2],
                           PetscReal DLampE[NUM_EDGES_PER_CELL][NUM_DIMENSIONS][2],
                           PetscBool *IdecE);

/**
 * @brief Projects the affine coordinates onto the tetrahedron faces.
 *
 * @param[in]  Lam     Barycentric coordinates.
 * @param[in]  DLam    Gradients of the barycentric coordinates.
 * @param[out] LampF   Per-face projected affine coordinates.
 * @param[out] DLampF  Gradients of the per-face projected coordinates.
 * @param[out] IdecF   Per-face "decoupled" flag for the orientation logic.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode ProjectTetF(const PetscReal Lam[NUM_DIMENSIONS + 1],
                           const PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1],
                           PetscReal LampF[NUM_FACES_PER_CELL][NUM_DIMENSIONS],
                           PetscReal DLampF[NUM_FACES_PER_CELL][NUM_DIMENSIONS][NUM_DIMENSIONS],
                           PetscBool *IdecF);

/**
 * @brief Applies edge orientation to a 1D shape/derivative pair.
 *
 * @param[in]  S     Shape-function pair.
 * @param[in]  DS    Derivatives of the shape-function pair.
 * @param[in]  Nori  Edge orientation index.
 * @param[out] GS    Orientation-adjusted shape-function pair.
 * @param[out] GDS   Orientation-adjusted derivatives.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode OrientE(const PetscReal S[2],
                       const PetscReal DS[NUM_DIMENSIONS][2],
                       const PetscInt Nori, PetscReal GS[2],
                       PetscReal GDS[NUM_DIMENSIONS][2]);

/**
 * @brief Applies triangular-face orientation to a shape/derivative set.
 *
 * @param[in]  S     Shape-function set.
 * @param[in]  DS    Derivatives of the shape-function set.
 * @param[in]  Nori  Face orientation index.
 * @param[out] GS    Orientation-adjusted shape-function set.
 * @param[out] GDS   Orientation-adjusted derivatives.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode OrientTri(const PetscReal S[NUM_DIMENSIONS],
                         const PetscReal DS[NUM_DIMENSIONS][NUM_DIMENSIONS],
                         const PetscInt Nori, PetscReal GS[NUM_DIMENSIONS],
                         PetscReal GDS[NUM_DIMENSIONS][NUM_DIMENSIONS]);

/**
 * @brief Evaluates scaled Legendre polynomials P_0..P_nord.
 *
 * @param[in]  X     Evaluation coordinate.
 * @param[in]  T     Scaling parameter for the homogeneous coordinates.
 * @param[in]  nord  Highest polynomial order.
 * @param[out] P     Polynomial values P_0..P_nord.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode PolyLegendre(const PetscReal X, const PetscReal T,
                            const PetscInt nord, PetscReal P[]);

/**
 * @brief Evaluates homogenized integrated Jacobi polynomials and derivatives.
 *
 * @param[in]  S        Homogeneous coordinate pair.
 * @param[in]  DS       Derivatives of the coordinate pair.
 * @param[in]  nord     Highest polynomial order.
 * @param[in]  Minalpha Minimum Jacobi α parameter.
 * @param[in]  Idec     Decoupled-coordinate flag.
 * @param[out] HomL     Polynomial values.
 * @param[out] DHomL    Polynomial derivatives.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode HomIJacobi(const PetscReal S[2],
                          const PetscReal DS[NUM_DIMENSIONS][2],
                          const PetscInt nord, const PetscInt Minalpha,
                          const PetscBool Idec, PetscReal **HomL,
                          PetscReal ***DHomL);

/* ---------------------------------------------------------------------------
 * Per-order ops tables for the unified hierarchical Nédélec basis
 * (nord = 1..6). All six tables are defined in src/hvfem_hierarchical.c;
 * each captures its order at the adapter and shares the same shape3DETet
 * kernel via the order-parameterized Piola adapters. Orientation is
 * encoded inside the reference shape functions (OrientE / OrientTri), so
 * callers MUST set DOF signs to +1 - handled centrally in buildDofSigns.
 * ------------------------------------------------------------------------- */
extern const NedelecOps nedelecOps_order1;
extern const NedelecOps nedelecOps_order2;
extern const NedelecOps nedelecOps_order3;
extern const NedelecOps nedelecOps_order4;
extern const NedelecOps nedelecOps_order5;
extern const NedelecOps nedelecOps_order6;

#endif
