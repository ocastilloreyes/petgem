/*
 * Filename: receiver_interp.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Shared receiver-interpolation operator for the CSEM kernels.
 */

/*
 * Notes:
 * Builds a set of sparse matrices Q that map an H(curl) DOF vector x
 * to the electric and magnetic field components evaluated at receiver
 * locations:
 *
 *     Ex_recv = QEx * x         Hx_recv = (QHx * x) / (i*omega*mu)
 *     Ey_recv = QEy * x         Hy_recv = (QHy * x) / (i*omega*mu)
 *     Ez_recv = QEz * x         Hz_recv = (QHz * x) / (i*omega*mu)
 *
 * Both fm.csem (postprocessing) and im.csem (inversion residual +
 * adjoint RHS) call this routine.  Building Q once and using MatMult
 * is intrinsically MPI-invariant: each receiver row is decided once
 * during assembly with global column indexing, so the same operator
 * is produced for any rank count.  Per-receiver direct evaluation
 * (the previous fm path) was not MPI-invariant when receivers fell on
 * partition boundaries.
 *
 * Conventions match fm.csem: Nédélec basis Ni[d][k] * dofSigns[k] for
 * E components, NiCurl[d][k] * dofSigns[k] for H components (caller
 * divides by constFactor = i*omega*mu).
 */

#ifndef RECEIVER_INTERP_H
#define RECEIVER_INTERP_H

#include "constants.h"
#include "grid.h"
#include <petsc.h>
#include <petscdm.h>
#include <petscmat.h>

/**
 * @brief Frequency-independent receiver-interpolation operators.
 *
 * QEx..QHz are (numReceivers x numDof) MATAIJ sparse matrices mapping an
 * H(curl) DOF vector x to field components at the receivers:
 * Ex_recv = QEx * x; Hx_recv = (QHx * x) / constFactor (constFactor = iωμ).
 */
typedef struct {
  Mat      QEx, QEy, QEz; /**< Electric-field interpolation matrices. */
  Mat      QHx, QHy, QHz; /**< Magnetic-field (curl) interpolation matrices. */
  PetscInt numReceivers;  /**< Number of receivers (rows of each Q). */
  PetscInt numDof;        /**< Number of H(curl) DOFs (columns of each Q). */
} ReceiverInterpolationMatrices;

/**
 * @brief Builds the receiver-interpolation matrices QEx..QHz.
 *
 * Assembles the Q operators on the same DM as the H(curl) solution vector.
 * Building Q once with global column indexing makes the operator
 * MPI-invariant. The routine opens no file; the caller supplies the
 * receiver coordinates.
 *
 * @param[in]  nord       Nédélec basis order (dispatched via fem->ops, 1..6).
 * @param[in]  receivers  Serial Vec (PETSC_COMM_SELF) of 3·N_recv reals,
 *                        laid out [x0 y0 z0 x1 y1 z1 ...]; produced by
 *                        loadCsemInputs from /receivers in the input bundle.
 * @param[in]  dm         H(curl) DM the solution lives on.
 * @param[in]  grid       Grid struct produced by setupCsemGrid.
 * @param[out] Q          Output struct; free with
 *                        destroyReceiverInterpolationMatrices.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode buildReceiverInterpolationMatrices(PetscInt    nord,
                                                  Vec         receivers,
                                                  const DM    dm,
                                                  const Grid *grid,
                                                  ReceiverInterpolationMatrices *Q);

/**
 * @brief Destroys the matrices held in a ReceiverInterpolationMatrices struct.
 *
 * @param[in,out] Q  Struct whose QEx..QHz matrices are destroyed.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode destroyReceiverInterpolationMatrices(ReceiverInterpolationMatrices *Q);

#endif /* RECEIVER_INTERP_H */
