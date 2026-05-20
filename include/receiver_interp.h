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

/* ------------------------------------------------------------------ */
/* Receiver interpolation matrices (frequency-independent).           */
/* QEx..QHz: (numReceivers x numDof) MATAIJ sparse                    */
/* Apply: Ex_recv = QEx * x;  Hx_recv = (QHx * x) / constFactor       */
/* ------------------------------------------------------------------ */
typedef struct {
  Mat      QEx, QEy, QEz;
  Mat      QHx, QHy, QHz;
  PetscInt numReceivers;
  PetscInt numDof;
} ReceiverInterpolationMatrices;

/* Build QEx..QHz on the same DM as the H(curl) solution vector.
 *   nord       : Nédélec basis order (dispatched via fem->ops, 1..6)
 *   receivers  : serial Vec (PETSC_COMM_SELF) of 3·N_recv reals, layout
 *                [x0 y0 z0 x1 y1 z1 ...]. Caller supplies it; this routine
 *                does not open any file. The unified PETGEM input loader
 *                (loadCsemInputs) produces it from /receivers in the
 *                bundled input HDF5.
 *   dm         : H(curl) DM the solution lives on
 *   grid       : Grid struct produced by setupCsemGrid
 *   verbose    : when PETSC_TRUE, prints DMLocatePoints + Q-matrix
 *                Frobenius / nnz diagnostics used for MPI-invariance
 *                debugging. Production runs pass PETSC_FALSE.
 *   Q          : output struct; caller invokes
 *                destroyReceiverInterpolationMatrices when done. */
PetscErrorCode buildReceiverInterpolationMatrices(PetscInt    nord,
                                                  Vec         receivers,
                                                  const DM    dm,
                                                  const Grid *grid,
                                                  PetscBool   verbose,
                                                  ReceiverInterpolationMatrices *Q);

/* Free Q matrices. */
PetscErrorCode destroyReceiverInterpolationMatrices(ReceiverInterpolationMatrices *Q);

#endif /* RECEIVER_INTERP_H */
