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
 * The struct-based signatures below (Cell / FEMSpace / Quadrature3D) are thin
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
  PetscInt numPoints;  /**< Number of quadrature points. */
  PetscReal** points;  /**< Quadrature point coordinates (per point, 3 reals). */
  PetscReal* weights;  /**< Quadrature weights. */
} Quadrature3D;

/**
 * @brief Computes a 3D unit vector from sequential azimuth/dip rotations.
 *
 * @param[in]  azimuth         Azimuth angle (degrees).
 * @param[in]  dip             Dip angle (degrees).
 * @param[out] rotationVector  Resulting unit direction vector.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode computeVectorRotation(const PetscReal azimuth, const PetscReal dip,
                                     PetscReal rotationVector[NUM_DIMENSIONS]);

/**
 * @brief Maps global Cartesian coordinates to reference-tetrahedron coordinates.
 *
 * Produces the reference point in the same frame the Nedelec/H1 reference bases
 * are defined on, so the result feeds evaluateNedelecBasis directly.
 *
 * @param[in]  coordinates  Cell vertex coordinates (4 vertices x 3 dims).
 * @param[in]  point        Global point to map.
 * @param[out] XiEtaZeta    Reference coordinates of the point.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode tetrahedronXYZToReference(const PetscReal coordinates[NUM_VERTICES_PER_CELL * NUM_DIMENSIONS],
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
PetscErrorCode computeCellJacobian(Cell* cell);

/**
 * @brief Sets the number of 3D quadrature points for a tetrahedron.
 *
 * One quadrature path for every order: a Stroud conical rule with (order+1)
 * points per axis, exact to degree 2*order+1 (covers the degree-2*order
 * mass-matrix integrand on an affine tetrahedron). Total (order+1)^3 points.
 *
 * @param[in]  order        Basis order driving the quadrature degree.
 * @param[out] quadrature  Rule whose numPoints is set.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode computeNum3DQuadraturePoints(const PetscInt order, Quadrature3D* quadrature);

/**
 * @brief Populates 3D Gauss quadrature points and weights for a tetrahedron.
 *
 * The point/weight convention matches the reference element bases: points on
 * the unit reference tetrahedron, weights summing to its volume 1/6.
 *
 * @param[in,out] quadrature  Rule (numPoints set) whose points/weights fill.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode compute3DQuadraturePoints(Quadrature3D* quadrature);

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
PetscErrorCode computeElementalMatrices(const FEMSpace* fem, const Cell* cell, const Quadrature3D* quadrature,
                                        PetscReal** Me, PetscReal** Ke);

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
PetscErrorCode buildDiscreteGradientMatrix(const FEMSpace* fem, const Cell* cell,
                                           PetscReal** gradientMatrix);

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
PetscErrorCode evaluateNedelecBasis(const FEMSpace* fem, const Cell* cell,
                                    const PetscReal point[NUM_DIMENSIONS],
                                    PetscReal** Ni, PetscReal** NiCurl);

#endif
