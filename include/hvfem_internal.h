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
 *   - 3-vector products and norm
 *   - barycentric mapping from Cartesian
 *   - dense matrix inversion + 3x3 solve with 3x6 RHS (used by the
 *     reference-to-physical pullback in nord=1)
 * ------------------------------------------------------------------------- */
PetscErrorCode dotProduct(const PetscReal vector1[NUM_DIMENSIONS],
                          const PetscReal vector2[NUM_DIMENSIONS],
                          PetscReal *result);

PetscErrorCode crossProduct(const PetscReal a[NUM_DIMENSIONS],
                            const PetscReal b[NUM_DIMENSIONS],
                            PetscReal result[NUM_DIMENSIONS]);

PetscReal vectorNorm(const PetscReal v[NUM_DIMENSIONS]);

PetscErrorCode cartesianToVolumetricCoordinates(const PetscReal r[NUM_DIMENSIONS],
                                                PetscReal L[4]);

PetscErrorCode invertMatrix(const PetscInt N, const PetscReal A[], PetscReal invA[]);

PetscErrorCode solve3x3MatrixSystem3x6RHS(const PetscReal matrix1[NUM_DIMENSIONS][NUM_DIMENSIONS],
                                          PetscReal **matrix2, PetscReal **result);

/* H1 hierarchical shape functions on the reference tetrahedron — used by the
 * order-1 gradient-matrix builder and (eventually) higher orders that need
 * P_k nodal values & gradients at quadrature points. */
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
PetscErrorCode AffineTetrahedron(const PetscReal X[NUM_DIMENSIONS],
                                 PetscReal Lam[NUM_DIMENSIONS + 1],
                                 PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1]);

PetscErrorCode ProjectTetE(const PetscReal Lam[NUM_DIMENSIONS + 1],
                           const PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1],
                           PetscReal LampE[NUM_EDGES_PER_CELL][2],
                           PetscReal DLampE[NUM_EDGES_PER_CELL][NUM_DIMENSIONS][2],
                           PetscBool *IdecE);

PetscErrorCode ProjectTetF(const PetscReal Lam[NUM_DIMENSIONS + 1],
                           const PetscReal DLam[NUM_DIMENSIONS][NUM_DIMENSIONS + 1],
                           PetscReal LampF[NUM_FACES_PER_CELL][NUM_DIMENSIONS],
                           PetscReal DLampF[NUM_FACES_PER_CELL][NUM_DIMENSIONS][NUM_DIMENSIONS],
                           PetscBool *IdecF);

PetscErrorCode OrientE(const PetscReal S[2],
                       const PetscReal DS[NUM_DIMENSIONS][2],
                       const PetscInt Nori, PetscReal GS[2],
                       PetscReal GDS[NUM_DIMENSIONS][2]);

PetscErrorCode OrientTri(const PetscReal S[NUM_DIMENSIONS],
                         const PetscReal DS[NUM_DIMENSIONS][NUM_DIMENSIONS],
                         const PetscInt Nori, PetscReal GS[NUM_DIMENSIONS],
                         PetscReal GDS[NUM_DIMENSIONS][NUM_DIMENSIONS]);

PetscErrorCode PolyLegendre(const PetscReal X, const PetscReal T,
                            const PetscInt nord, PetscReal P[]);

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
 * callers MUST set DOF signs to +1 — handled centrally in buildDofSigns.
 * ------------------------------------------------------------------------- */
extern const NedelecOps nedelecOps_order1;
extern const NedelecOps nedelecOps_order2;
extern const NedelecOps nedelecOps_order3;
extern const NedelecOps nedelecOps_order4;
extern const NedelecOps nedelecOps_order5;
extern const NedelecOps nedelecOps_order6;

#endif
