/*
 * Filename: fem.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-06-22
 *
 * Description:
 * Finite-element interface for PETGEM, built on the MFEM-style reference-element
 * bases (fe_nedelec.c for H(curl), fe_nodal.c for H1) for orders order = 1..6.
 *
 * Orientation is geometric and folded into the element routines (femOrient on
 * the reference field, keyed by the lexicographic order of the cell's vertex
 * coordinates), so there are NO external per-DOF sign multipliers: the values,
 * curls, mass/stiffness and discrete gradient returned here are already oriented
 * and ready to insert into the DMPlex-closure-ordered global system.
 *
 * The struct-based signatures below (Cell / FemSpace / Quadrature3D) are thin
 * adapters over the reference core so the assembly, postprocessing and
 * receiver-interpolation code can stay DMPlex-centric.
 */

#ifndef FEM_H
#define FEM_H

#include "constants.h"
#include "grid.h"
#include <petsc.h>
#include <petscdmplex.h>

/** @brief 3D (tetrahedron) Gauss quadrature rule (points and weights). */
typedef struct {
  PetscInt    numPoints; /**< Number of quadrature points. */
  PetscReal **points;    /**< Quadrature point coordinates (per point, 3 reals). */
  PetscReal  *weights;   /**< Quadrature weights. */
} Quadrature3D;

/** @brief 2D (triangle) Gauss quadrature rule (points and weights). */
typedef struct {
  PetscInt    numPoints; /**< Number of quadrature points. */
  PetscReal **points;    /**< Quadrature point coordinates (per point, 2 reals). */
  PetscReal  *weights;   /**< Quadrature weights. */
} Quadrature2D;

/**
 * @brief Computes a 3D unit vector from sequential azimuth/dip rotations.
 *
 * @param[in]  azimuth         Azimuth angle (degrees).
 * @param[in]  dip             Dip angle (degrees).
 * @param[out] rotationVector  Resulting unit direction vector.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode computeVectorRotation(const PetscReal azimuth, const PetscReal dip, PetscReal rotationVector[NUM_DIMENSIONS]);

/**
 * @brief Maps global Cartesian coordinates to reference-tetrahedron coordinates.
 *
 * Inverts x = v0 + J^T (xi, eta, zeta), with J the cell Jacobian (rows v1-v0,
 * v2-v0, v3-v0), the frame evaluateNedelecBasis evaluates the bases in.
 *
 * @param[in]  coordinates  Cell vertex coordinates (4 vertices x 3 dims).
 * @param[in]  point        Global point to map.
 * @param[out] XiEtaZeta    Reference coordinates of the point.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode tetrahedronXyzToReference(const PetscReal coordinates[NUM_VERTICES_PER_CELL * NUM_DIMENSIONS],
                                         const PetscReal point[NUM_DIMENSIONS], PetscReal XiEtaZeta[NUM_DIMENSIONS]);

/**
 * @brief Computes a cell's geometric Jacobian, its inverse, and determinant.
 *
 * Geometry helper used for point location / degeneracy checks. The FE routines
 * compute their own reference-consistent Jacobian internally from the cell
 * vertex coordinates, so callers do not need to feed this back into them.
 *
 * @param[in,out] cell  Cell whose jacobian/invJacobian/detJacobian are filled.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode computeCellJacobian(Cell *cell);

/**
 * @brief Allocates and fills the 3D quadrature rule for a basis order.
 *
 * @param[in]  order       Basis order driving the quadrature degree.
 * @param[out] quadrature  Rule; release it with destroyQuadrature3D().
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode createQuadrature3D(const PetscInt order, Quadrature3D *quadrature);

/**
 * @brief Frees a rule created by createQuadrature3D().
 */
PetscErrorCode destroyQuadrature3D(Quadrature3D *quadrature);

/**
 * @brief Allocates and fills the 2D quadrature rule for a basis order.
 *
 * @param[in]  order       Basis order driving the quadrature degree.
 * @param[out] quadrature  Rule; release it with destroyQuadrature2D().
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode createQuadrature2D(const PetscInt order, Quadrature2D *quadrature);

/**
 * @brief Frees a rule created by createQuadrature2D().
 */
PetscErrorCode destroyQuadrature2D(Quadrature2D *quadrature);

/**
 * @brief Allocates a zeroed rows x cols matrix as row pointers into one contiguous block.
 */
PetscErrorCode createDenseRows(PetscInt rows, PetscInt cols, PetscReal ***M);

/**
 * @brief Frees a matrix created by createDenseRows() (NULL-safe).
 */
PetscErrorCode destroyDenseRows(PetscReal ***M);

/**
 * @brief Allocates a zeroed basis-value buffer N[d][dof], d in [0,3).
 *
 * @param[in]  numDof  Number of DOFs per cell.
 * @param[out] N       Buffer; release it with destroyBasisValues().
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode createBasisValues(const PetscInt numDof, PetscReal ***N);

/**
 * @brief Frees a buffer created by createBasisValues().
 */
PetscErrorCode destroyBasisValues(PetscReal ***N);

/**
 * @brief Computes the elemental mass and stiffness matrices for a cell.
 *
 *   Me_jk = INT (sigma * N_j) . N_k   (mass weighted by the cell conductivity)
 *   Ke_jk = INT  curl N_j . curl N_k  (stiffness, mu_r = identity)
 *
 * Orientation and the value/curl Piola pullbacks are applied internally, so the
 * returned blocks are physical and ready to insert in DMPlex closure order.
 *
 * @param[in]  fem         Finite-element space descriptor (order, DOF counts).
 * @param[in]  cell        Cell geometry and conductivity.
 * @param[in]  quadrature  3D quadrature rule.
 * @param[out] Me          Elemental mass matrix (numDofInCell x numDofInCell).
 * @param[out] Ke          Elemental stiffness matrix (numDofInCell x numDofInCell).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode computeElementalMatrices(const FemSpace *fem, const Cell *cell, const Quadrature3D *quadrature, PetscReal **Me,
                                        PetscReal **Ke);

/**
 * @brief Builds the high-order discrete gradient block G_e for one cell (the
 *        curl-kernel operator consumed by PCBDDCSetDiscreteGradient).
 *
 * grad(phi_k) = sum_i G_ik N_i, with rows = Nedelec DOFs and columns = P_order
 * H1 DOFs, both in DMPlex closure order; sparse after thresholding, so K.G = 0.
 *
 * @param[in]  fem             FE space descriptor.
 * @param[in]  cell            Cell geometry.
 * @param[out] gradientMatrix  Block sized numDofInCell x numH1DofInCell.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode buildDiscreteGradientMatrix(const FemSpace *fem, const Cell *cell, PetscReal **gradientMatrix);

/**
 * @brief Evaluates the Nedelec basis (and optionally curls) at a reference point.
 *
 * The returned values/curls are physical and oriented. Pass NiCurl = NULL to
 * skip the curl evaluation (e.g. RHS source integrals).
 *
 * @param[in]  fem     Finite-element space descriptor.
 * @param[in]  cell    Cell geometry.
 * @param[in]  point   Reference-cell evaluation point.
 * @param[out] Ni      Basis-function values: Ni[d][dof], d in [0,3).
 * @param[out] NiCurl  Basis-curl values: NiCurl[d][dof] (NULL to skip).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode evaluateNedelecBasis(const FemSpace *fem, const Cell *cell, const PetscReal point[NUM_DIMENSIONS], PetscReal **Ni,
                                    PetscReal **NiCurl);

#endif
