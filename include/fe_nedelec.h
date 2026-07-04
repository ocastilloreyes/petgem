/*
 * Filename: fe_nedelec.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-06-22
 *
 * Description:
 * Public interface for the arbitrary-order H(curl) Nedelec basis on the
 * reference tetrahedron. The handle is opaque; create one per
 * order, evaluate values/curls at reference-cell points, and query the
 * defining-functional node/tangent of each DOF for the discrete gradient.
 *
 * A single handle is NOT reentrant: evaluation writes through scratch owned
 * by the handle.
 */

#ifndef FE_NEDELEC_H
#define FE_NEDELEC_H

#include <petsc.h>

/** @brief Opaque arbitrary-order H(curl) Nedelec element on the reference tet. */
typedef struct FeNedelec FeNedelec;

/**
 * @brief Number of H(curl) DOFs for a Nedelec element of the given order.
 *
 * @param[in] order  Polynomial order p (>= 1).
 * @return order*(order+2)*(order+3)/2, or 0 when order < 1.
 */
PetscInt feNedelecDofCount(PetscInt order);

/**
 * @brief Builds a Nedelec element of the given order.
 *
 * @param[in]  order  Polynomial order p (>= 1).
 * @param[out] out    Newly created handle (caller destroys with feNedelecDestroy).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode feNedelecCreate(PetscInt order, FeNedelec **out);

/**
 * @brief Destroys a Nedelec element handle (no-op on NULL).
 *
 * @param[in,out] fe  Handle to free; set to NULL on return.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode feNedelecDestroy(FeNedelec **fe);

/** @brief Returns the number of H(curl) DOFs (0 if fe is NULL). */
PetscInt feNedelecGetDof(const FeNedelec *fe);

/** @brief Returns the polynomial order (0 if fe is NULL). */
PetscInt feNedelecGetOrder(const FeNedelec *fe);

/**
 * @brief Evaluates the reference Nedelec values at a reference-cell point.
 *
 * @param[in]  fe     Element handle.
 * @param[in]  x,y,z  Reference-cell coordinates.
 * @param[out] shape  dof x 3 row-major basis values.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode feNedelecCalcVShape(const FeNedelec *fe, PetscReal x, PetscReal y,
                                   PetscReal z, PetscReal *shape);

/**
 * @brief Evaluates the reference Nedelec curls at a reference-cell point.
 *
 * @param[in]  fe     Element handle.
 * @param[in]  x,y,z  Reference-cell coordinates.
 * @param[out] curl   dof x 3 row-major curl values.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode feNedelecCalcCurlShape(const FeNedelec *fe, PetscReal x, PetscReal y,
                                      PetscReal z, PetscReal *curl);

/**
 * @brief Returns the defining-functional node and tangent of a DOF.
 *
 * @param[in]  fe       Element handle.
 * @param[in]  m        DOF index in [0, dof).
 * @param[out] node     Reference-cell node coordinates (3); may be NULL.
 * @param[out] tangent  DOF tangent direction (3); may be NULL.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode feNedelecGetDofInfo(const FeNedelec *fe, PetscInt m,
                                   PetscReal node[3], PetscReal tangent[3]);

#endif
