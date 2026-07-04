/*
 * Filename: fe_nodal.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-06-22
 *
 * Description:
 * Public interface for the arbitrary-order nodal (Lagrange) H1 basis on the
 * reference tetrahedron - the column space of the discrete gradient (a De Rham
 * pair with the Nedelec space). Shape values and reference gradients are emitted
 * in DMPlex H1 closure order [volume, face, edge, vertex], with edge/face DOFs
 * oriented geometrically so they match the Nedelec side (femOrient).
 */

#ifndef FE_NODAL_H
#define FE_NODAL_H

#include <petsc.h>

/**
 * @brief Reports whether the nodal H1 basis supports a given order.
 *
 * @param[in] order  Polynomial order.
 * @return PETSC_TRUE for order in [1, FE_NODAL_MAX_ORDER], else PETSC_FALSE.
 */
PetscBool feNodalSupports(PetscInt order);

/**
 * @brief Evaluates the nodal H1 shape values and reference gradients.
 *
 * @param[in]  order          Polynomial order (1..6).
 * @param[in]  vertexCoords  Cell vertex coordinates (4x3) for edge/face orientation.
 * @param[in]  X             Reference-cell evaluation point (3).
 * @param[out] ShapH         Shape values, one per H1 DOF, in closure order.
 * @param[out] GradH         Reference gradients: GradH[d][dof], d in [0,3).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode feNodalH1Shape(PetscInt order, const PetscReal *vertexCoords,
                              const PetscReal X[3], PetscReal *ShapH, PetscReal **GradH);

#endif
