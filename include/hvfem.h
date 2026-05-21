/*
 * Filename: hvfem.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Prototypes for the high-order vector finite element functions
 * used throughout PETGEM.
 */

#ifndef HVFEM_H
#define HVFEM_H

#include "constants.h"
#include "grid.h"
#include <petsc.h>
#include <petscdmplex.h>

typedef struct {
  PetscInt numPoints;
  PetscReal* points;
  PetscReal* weights;
} Quadrature1D;

typedef struct {
  PetscInt numPoints;
  PetscReal** points;
  PetscReal* weights;
} Quadrature2D;

typedef struct {
  PetscInt numPoints;
  PetscReal** points;
  PetscReal* weights;
} Quadrature3D;

PetscErrorCode computeCellJacobian(Cell* cell);

PetscErrorCode computeCellOrientation(Cell* cell);

PetscErrorCode computeNum1DQuadraturePoints(const PetscInt nord, Quadrature1D* quadrature);

PetscErrorCode computeNum2DQuadraturePoints(const PetscInt nord, Quadrature2D* quadrature);

PetscErrorCode computeNum3DQuadraturePoints(const PetscInt nord, Quadrature3D* quadrature);

PetscErrorCode compute1DQuadraturePoints(Quadrature1D* quadrature);

PetscErrorCode compute2DQuadraturePoints(Quadrature2D* quadrature);

PetscErrorCode compute3DQuadraturePoints(Quadrature3D* quadrature);

PetscErrorCode tetrahedronXYZToReference(const PetscReal coordinates[NUM_VERTICES_PER_CELL * NUM_DIMENSIONS],
                                         const PetscReal point[NUM_DIMENSIONS], PetscReal XiEtaZeta[NUM_DIMENSIONS]);

PetscErrorCode computeVectorRotation(const PetscReal azimuth, const PetscReal dip, PetscReal rotationVector[NUM_DIMENSIONS]);

PetscErrorCode computeElementalMatrices(const FEMSpace* fem, const Cell* cell, const Quadrature3D* quadrature,
                                        PetscReal** Me, PetscReal** Ke);

PetscErrorCode printCellEntities(const DM dm, const PetscInt cell);

/* Build the per-DOF sign vector for a cell under the chosen Nédélec order.
 *   nord=1: one DOF per edge; signs[e] = cell->orientation.edgeSigns[e].
 *   nord=2: faces-first with 2 DOFs per entity. Face DOFs are +1 (canonical-
 *           geometry q-vectors agree across adjacent cells); edge-DOF signs
 *           come from cell->orientation.edgeSigns[0..5].
 * The caller provides a signs[] array of length fem->numDofInCell. */
PetscErrorCode buildDofSigns(const Cell* cell, const FEMSpace* fem, PetscInt signs[]);

/* Evaluate the Nédélec basis (and optionally curls) at a reference-cell
 * point. Dispatches on fem->nord. Pass NiCurl=NULL to skip curl evaluation. */
PetscErrorCode evaluateNedelecBasis(const FEMSpace* fem, const Cell* cell,
                                    const PetscReal point[NUM_DIMENSIONS],
                                    PetscReal** coeffs,
                                    PetscReal** Dx_Ni, PetscReal** Dy_Ni, PetscReal** Dz_Ni,
                                    PetscReal** Ni, PetscReal** NiCurl);

/* Per-order Nédélec dispatch table. Each supported basis order registers
 * one of these; hot paths call through the table instead of a switch (nord).
 * Adding a new order means: define the four per-order static helpers in
 * hvfem.c, build a NedelecOps instance for them, and register it in
 * nedelecOpsForOrder().
 *
 * Uniform signatures — some fields are unused for a given order:
 *   - computeCoefficients: nord=1 fills coeffs AND Dx/Dy/Dz; nord>=2 only
 *     fills coeffs (Dx/Dy/Dz still passed but untouched).
 *   - computeBasis      : evaluates Ni at a reference-cell point.
 *   - computeCurls      : evaluates NiCurl at a reference-cell point. nord=1
 *     uses Dx/Dy/Dz; nord>=2 uses coeffs+point. Each impl reads what it needs.
 *   - buildGradientMatrix: TOPOLOGICAL discrete gradient G against P1 H1
 *     (numDofInCell x NUM_H1_DOF_PER_CELL). Used as the BDDC Nédélec
 *     hint via PCBDDCSetDiscreteGradient. Only the lowest-order Whitney
 *     row per mesh edge is nonzero (±1 vertex incidence); all other rows
 *     are zero. K·G is NOT zero at nord >= 2 by design.
 *   - buildExactGradientMatrix: EXACT commuting discrete gradient G
 *     against the hierarchical P_nord H1 basis (numDofInCell x
 *     numH1DofInCell_Pnord). Computed as the canonical Nédélec
 *     interpolation Π^Ned(∇φ_j), evaluated via Ainsworth–Coyle DOF
 *     moments (tangential edge Legendre moments + canonical-tangent
 *     face moments against barycentric polynomials + cell-local volume
 *     moments) — see hierarchicalBuildExactGradientMatrix. Cross-cell
 *     consistency is enforced by computing every shared moment in the
 *     CANONICAL geometric frame (sorted-vertex face ordering, canonical
 *     edge direction), so the assembled global G satisfies K·G = 0 to
 *     machine precision and G·c = 0 for the constant H1 mode. Consumed
 *     by assembleCsemKandM (the unified forward/inverse LHS assembly)
 *     and surfaced to PCBDDCSetDiscreteGradient via the topological
 *     G_BDDC sibling matrix. */
typedef struct NedelecOps {
  PetscErrorCode (*computeCoefficients)(const Cell *cell, PetscReal **coeffs,
                                        PetscReal **Dx_Ni, PetscReal **Dy_Ni,
                                        PetscReal **Dz_Ni);

  PetscErrorCode (*computeBasis)(const Cell *cell,
                                 const PetscReal point[NUM_DIMENSIONS],
                                 const PetscReal *const *coeffs,
                                 PetscReal **Ni);

  PetscErrorCode (*computeCurls)(const Cell *cell,
                                 const PetscReal point[NUM_DIMENSIONS],
                                 const PetscReal *const *coeffs,
                                 const PetscReal *const *Dx_Ni,
                                 const PetscReal *const *Dy_Ni,
                                 const PetscReal *const *Dz_Ni,
                                 PetscReal **NiCurl);

  PetscErrorCode (*buildGradientMatrix)(const FEMSpace *fem, const Cell *cell,
                                        const Quadrature1D *quadrature1d,
                                        PetscReal **gradientMatrix);

  PetscErrorCode (*buildExactGradientMatrix)(const FEMSpace *fem, const Cell *cell,
                                             PetscReal **gradientMatrix);
} NedelecOps;

/* Returns a pointer to the per-order ops table (NULL if nord is unsupported). */
const NedelecOps *nedelecOpsForOrder(PetscInt nord);

#endif
