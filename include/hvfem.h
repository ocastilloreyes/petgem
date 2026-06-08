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

/** @brief 1D Gauss quadrature rule (points and weights). */
typedef struct {
  PetscInt numPoints;  /**< Number of quadrature points. */
  PetscReal* points;   /**< Quadrature point coordinates. */
  PetscReal* weights;  /**< Quadrature weights. */
} Quadrature1D;

/** @brief 2D (triangle) Gauss quadrature rule (points and weights). */
typedef struct {
  PetscInt numPoints;  /**< Number of quadrature points. */
  PetscReal** points;  /**< Quadrature point coordinates (per point). */
  PetscReal* weights;  /**< Quadrature weights. */
} Quadrature2D;

/** @brief 3D (tetrahedron) Gauss quadrature rule (points and weights). */
typedef struct {
  PetscInt numPoints;  /**< Number of quadrature points. */
  PetscReal** points;  /**< Quadrature point coordinates (per point). */
  PetscReal* weights;  /**< Quadrature weights. */
} Quadrature3D;

/**
 * @brief Computes a cell's Jacobian matrix, its inverse, and determinant.
 *
 * @param[in,out] cell  Cell whose jacobian/invJacobian/detJacobian are filled.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PetscError code otherwise.
 */
PetscErrorCode computeCellJacobian(Cell* cell);

/**
 * @brief Computes the orientation of faces and edges for a tetrahedral cell.
 *
 * @param[in,out] cell  Cell whose orientation (faces, edgeSigns) is filled.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PetscError code otherwise.
 */
PetscErrorCode computeCellOrientation(Cell* cell);

/**
 * @brief Determines the number of 1D Gauss-Legendre quadrature points.
 *
 * @param[in]  nord        Basis order driving the quadrature degree.
 * @param[out] quadrature  Rule whose numPoints is set.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PetscError code otherwise.
 */
PetscErrorCode computeNum1DQuadraturePoints(const PetscInt nord, Quadrature1D* quadrature);

/**
 * @brief Determines the number of 2D quadrature points for a triangle.
 *
 * @param[in]  nord        Basis order driving the quadrature degree.
 * @param[out] quadrature  Rule whose numPoints is set.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PetscError code otherwise.
 */
PetscErrorCode computeNum2DQuadraturePoints(const PetscInt nord, Quadrature2D* quadrature);

/**
 * @brief Determines the number of Gauss quadrature points for a tetrahedron.
 *
 * @param[in]  nord        Basis order driving the quadrature degree.
 * @param[out] quadrature  Rule whose numPoints is set.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PetscError code otherwise.
 */
PetscErrorCode computeNum3DQuadraturePoints(const PetscInt nord, Quadrature3D* quadrature);

/**
 * @brief Populates 1D Gauss-Legendre quadrature points and weights.
 *
 * @param[in,out] quadrature  Rule (numPoints set) whose points/weights fill.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PetscError code otherwise.
 */
PetscErrorCode compute1DQuadraturePoints(Quadrature1D* quadrature);

/**
 * @brief Populates 2D Gauss quadrature points and weights for a triangle.
 *
 * @param[in,out] quadrature  Rule (numPoints set) whose points/weights fill.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PetscError code otherwise.
 */
PetscErrorCode compute2DQuadraturePoints(Quadrature2D* quadrature);

/**
 * @brief Populates 3D Gauss quadrature points and weights for a tetrahedron.
 *
 * @param[in,out] quadrature  Rule (numPoints set) whose points/weights fill.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PetscError code otherwise.
 */
PetscErrorCode compute3DQuadraturePoints(Quadrature3D* quadrature);

/**
 * @brief Maps global Cartesian coordinates to reference-tetrahedron coordinates.
 *
 * @param[in]  coordinates  Cell vertex coordinates (4 vertices × 3 dims).
 * @param[in]  point        Global point to map.
 * @param[out] XiEtaZeta    Reference coordinates (ξ, η, ζ) of the point.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PetscError code otherwise.
 */
PetscErrorCode tetrahedronXYZToReference(const PetscReal coordinates[NUM_VERTICES_PER_CELL * NUM_DIMENSIONS],
                                         const PetscReal point[NUM_DIMENSIONS], PetscReal XiEtaZeta[NUM_DIMENSIONS]);

/**
 * @brief Computes a 3D unit vector from sequential azimuth/dip rotations.
 *
 * @param[in]  azimuth         Azimuth angle (degrees).
 * @param[in]  dip             Dip angle (degrees).
 * @param[out] rotationVector  Resulting unit direction vector.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PetscError code otherwise.
 */
PetscErrorCode computeVectorRotation(const PetscReal azimuth, const PetscReal dip, PetscReal rotationVector[NUM_DIMENSIONS]);

/**
 * @brief Computes the elemental mass and stiffness matrices for a cell.
 *
 * @param[in]  fem         Finite-element space descriptor (order, DOF counts, ops).
 * @param[in]  cell        Cell geometry and orientation.
 * @param[in]  quadrature  3D quadrature rule.
 * @param[out] Me          Elemental mass matrix (numDofInCell²).
 * @param[out] Ke          Elemental stiffness matrix (numDofInCell²).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PetscError code otherwise.
 */
PetscErrorCode computeElementalMatrices(const FEMSpace* fem, const Cell* cell, const Quadrature3D* quadrature,
                                        PetscReal** Me, PetscReal** Ke);

/* printCellEntities is declared in grid.h (its definition lives in
 * grid.c); hvfem.h already includes grid.h, so no re-declaration here -
 * a duplicate caused a Sphinx "Duplicate C declaration" warning. */

/**
 * @brief Builds the per-DOF sign vector for a cell at the chosen Nédélec order.
 *
 * nord=1: one DOF per edge; signs[e] = cell->orientation.edgeSigns[e].
 * nord=2: faces-first with 2 DOFs per entity. Face DOFs are +1 (canonical-
 *         geometry q-vectors agree across adjacent cells); edge-DOF signs
 *         come from cell->orientation.edgeSigns[0..5].
 *
 * @param[in]  cell   Cell with computed orientation.
 * @param[in]  fem    Finite-element space descriptor.
 * @param[out] signs  Caller-provided array of length fem->numDofInCell.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PetscError code otherwise.
 */
PetscErrorCode buildDofSigns(const Cell* cell, const FEMSpace* fem, PetscInt signs[]);

/**
 * @brief Builds the EXACT high-order discrete gradient block G_e for one cell
 *        (the curl-kernel operator PCBDDCSetDiscreteGradient expects at nord>=2).
 *
 * grad(phi_k) = sum_i G_ik N_i, with rows = Nédélec DOFs and columns = P_nord H1
 * DOFs, both in DMPlex closure order. Sparse after thresholding. See the
 * definition in hvfem_hierarchical.c for the construction and caveats.
 *
 * @param[in]  fem             FE space descriptor.
 * @param[in]  cell            Cell with computed orientation.
 * @param[out] gradientMatrix  Block sized numDofInCell x numH1DofInCell_Pnord.
 * @param[out] laResidual      Relative linear-algebra residual ||M G - B||_F /
 *                             ||B||_F of the SPD projection solve (may be NULL).
 * @param[out] condEst         Conditioning estimate of the reference Nédélec mass
 *                             matrix (max/min Cholesky pivot, an SPD lower bound
 *                             on kappa_2(M)); may be NULL.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode buildExactDiscreteGradient(const FEMSpace* fem, const Cell* cell,
                                          PetscReal** gradientMatrix,
                                          PetscReal* laResidual, PetscReal* condEst);

/**
 * @brief Per-cell diagnostic metrics for the high-order discrete gradient.
 *
 * Populated by verifyExactDiscreteGradientCell and printed by the
 * -fm_check_gradient diagnostic. Separates the two failure modes the wham run
 * exposed: a large pointwiseResidual with a small laResidual but a HUGE condEst
 * is a conditioning-bound solve (not a basis fault); a large pointwiseResidual
 * with a MODEST condEst is a genuine basis-exactness fault. crossEntityRatio and
 * the per-entity row-nnz counts answer the PCBDDC entity-locality gate
 * (edge-rows must carry exactly nord+1 nonzeros; no edge->face/volume coupling).
 */
typedef struct {
  PetscReal pointwiseResidual; /**< max |grad(phi_k) - sum_i G_ik N_i| at sample pts. */
  PetscReal laResidual;        /**< ||M G - B||_F / ||B||_F of the projection solve. */
  PetscReal condEst;           /**< max/min Cholesky pivot of M (kappa_2 lower bound). */
  PetscReal crossEntityRatio;  /**< ||G in forbidden entity blocks||_F / ||G||_F (gate 1b). */
  PetscInt  nnzTotal;          /**< structural nonzeros in the block. */
  PetscInt  maxEdgeRowNnz;     /**< max nnz over Nédélec edge rows (== nord+1 if entity-local). */
  PetscInt  maxFaceRowNnz;     /**< max nnz over Nédélec face rows. */
  PetscInt  maxVolRowNnz;      /**< max nnz over Nédélec volume rows. */
} GradientCheckResult;

/**
 * @brief Unit check for buildExactDiscreteGradient on one cell (diagnostic).
 *
 * Verifies grad(phi_k) = sum_i G_ik N_i at interior reference points and reports,
 * via @p result, the projection-solve health (residual + conditioning) and the
 * entity-local sparsity structure PCBDDC requires. Used by the -fm_check_gradient
 * diagnostic to validate the discrete gradient in isolation before it is wired
 * into PCBDDC.
 *
 * @param[in]  fem     FE space descriptor.
 * @param[in]  cell    Cell with computed orientation.
 * @param[out] result  Filled diagnostic metrics (must be non-NULL).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode verifyExactDiscreteGradientCell(const FEMSpace* fem, const Cell* cell,
                                               GradientCheckResult* result);

/**
 * @brief Evaluates the Nédélec basis (and optionally curls) at a reference point.
 *
 * Dispatches on fem->nord. Pass NiCurl = NULL to skip curl evaluation.
 *
 * @param[in]     fem     Finite-element space descriptor.
 * @param[in]     cell    Cell geometry and orientation.
 * @param[in]     point   Reference-cell evaluation point.
 * @param[in,out] coeffs  Basis-coefficient workspace (order-dependent).
 * @param[in,out] Dx_Ni   Workspace for ∂/∂x terms (used at nord=1).
 * @param[in,out] Dy_Ni   Workspace for ∂/∂y terms (used at nord=1).
 * @param[in,out] Dz_Ni   Workspace for ∂/∂z terms (used at nord=1).
 * @param[out]    Ni      Basis-function values at the point.
 * @param[out]    NiCurl  Basis-curl values (NULL to skip).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PetscError code otherwise.
 */
PetscErrorCode evaluateNedelecBasis(const FEMSpace* fem, const Cell* cell,
                                    const PetscReal point[NUM_DIMENSIONS],
                                    PetscReal** coeffs,
                                    PetscReal** Dx_Ni, PetscReal** Dy_Ni, PetscReal** Dz_Ni,
                                    PetscReal** Ni, PetscReal** NiCurl);

/**
 * @brief Per-order Nédélec dispatch table.
 *
 * Each supported basis order registers one of these; hot paths call through
 * the table instead of a switch (nord). Adding a new order means: define the
 * four per-order static helpers in hvfem.c, build a NedelecOps instance for
 * them, and register it in nedelecOpsForOrder().
 *
 * Uniform signatures - some fields are unused for a given order:
 *   - computeCoefficients: nord=1 fills coeffs AND Dx/Dy/Dz; nord>=2 only
 *     fills coeffs (Dx/Dy/Dz still passed but untouched).
 *   - computeBasis       : evaluates Ni at a reference-cell point.
 *   - computeCurls       : evaluates NiCurl at a reference-cell point. nord=1
 *     uses Dx/Dy/Dz; nord>=2 uses coeffs+point.
 *
 * The discrete gradient for PCBDDC is NOT in this table: it is the exact
 * order-p operator built directly by buildExactDiscreteGradient.
 */
typedef struct NedelecOps {
  /** Computes basis coefficients (and, at nord=1, derivative tables). */
  PetscErrorCode (*computeCoefficients)(const Cell *cell, PetscReal **coeffs,
                                        PetscReal **Dx_Ni, PetscReal **Dy_Ni,
                                        PetscReal **Dz_Ni);

  /** Evaluates basis values Ni at a reference-cell point. */
  PetscErrorCode (*computeBasis)(const Cell *cell,
                                 const PetscReal point[NUM_DIMENSIONS],
                                 const PetscReal *const *coeffs,
                                 PetscReal **Ni);

  /** Evaluates basis curls NiCurl at a reference-cell point. */
  PetscErrorCode (*computeCurls)(const Cell *cell,
                                 const PetscReal point[NUM_DIMENSIONS],
                                 const PetscReal *const *coeffs,
                                 const PetscReal *const *Dx_Ni,
                                 const PetscReal *const *Dy_Ni,
                                 const PetscReal *const *Dz_Ni,
                                 PetscReal **NiCurl);
} NedelecOps;

/**
 * @brief Returns the per-order Nédélec ops table.
 *
 * @param[in] nord  Basis order.
 *
 * @return Pointer to the ops table, or NULL if nord is unsupported.
 */
const NedelecOps *nedelecOpsForOrder(PetscInt nord);

#endif
