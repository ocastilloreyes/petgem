/*
 * Filename: inversion_internal.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Cross-file prototypes for the inversion kernel that are not part of
 * the public API exposed via include/inversion.h.
 */

/*
 * Notes:
 * Prototypes shared by the inversion translation units only
 * (inversion.c, inversion_io.c, inversion_smoother.c).
 */

#ifndef INVERSION_INTERNAL_H
#define INVERSION_INTERNAL_H

#include "inversion.h"
#include <petsc.h>

/**
 * @brief Builds the persistent 0/1 mask used to freeze fixed cells.
 *
 * Defined in src/inversion_smoother.c. The mask Vec is consumed by
 * inversionObjGrad to zero gradient entries at fixed (held-constant) cells.
 *
 * @param[in]  graph               Neighbor graph describing cell adjacency.
 * @param[in]  dmInversion         DM for the inversion (per-cell) field.
 * @param[in]  grid                Finite-element grid descriptor.
 * @param[out] maskGlobal          Global 0/1 mask Vec.
 * @param[out] maskLocal           Ghosted local counterpart of the mask.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode buildNotFixedMask(const NeighborGraph *graph, DM dmInversion, const Grid *grid, Vec *maskGlobal, Vec *maskLocal);

/**
 * @brief Clones the topology of @p dm with one scalar DOF per cell.
 *
 * Registers a 1-component PetscFV as field 0 so cell data can be written to VTU.
 *
 * @param[in]  dm     DMPlex whose topology is cloned.
 * @param[out] cellDm 1-DOF/cell DM.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode createCellScalarDm(DM dm, DM *cellDm);

/**
 * @brief Writes the final inversion results to HDF5.
 *
 * Defined in src/inversion_io.c. Dumps the recovered model, the per-iteration RMS
 * history, and provenance attributes. Called once at the end of
 * runCsemInversion.
 *
 * @param[in] imParams       Inversion parameters / provenance.
 * @param[in] dmConductivity  DM for the per-cell conductivity field.
 * @param[in] conductivity    Recovered conductivity Vec.
 * @param[in] X               Recovered log-perturbation / model Vec.
 * @param[in] allRms          Array of per-evaluation RMS values.
 * @param[in] numIters        Accepted L-BFGS steps (the "iteration" count a
 *                            user sees in the optimizer table).
 * @param[in] numEvals        Objective-gradient evaluations, i.e. the number of
 *                            entries in @p allRms. Exceeds @p numIters by the
 *                            line-search trials that were rejected, so the RMS
 *                            history it indexes is not monotone.
 * @param[in] reasonStr       Human-readable convergence/stop reason.
 * @param[in] gradNorms       ||g|| per accepted iteration (entry 0 = initial model).
 * @param[in] numGradNorms    Entries in @p gradNorms: accepted steps + 1.
 * @param[in] dPre            Predicted data of the recovered model, component-major
 *                            numComponents x numFreqs x numReceivers (read on rank 0 only).
 * @param[in] numReceivers    Number of receivers.
 * @param[in] rmsPre          Data RMS of @p dPre.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode writeInversionResults(const ImParams *imParams, DM dmConductivity, Vec conductivity, Vec X, const PetscReal *allRms,
                                     PetscInt numIters, PetscInt numEvals, const char *reasonStr, const PetscReal *gradNorms,
                                     PetscInt numGradNorms, const PetscScalar *dPre, PetscInt numReceivers, PetscReal rmsPre);

/**
 * @brief Writes the gradient check to {output_dir}/{output_filename}.h5.
 *
 * Datasets gradient_check_eps, gradient_check_fd (central difference of the
 * data misfit) and gradient_check_adjoint (adjoint directional derivative).
 *
 * @param[in] imParams  Inversion parameters (output path, components).
 * @param[in] comm      Communicator.
 * @param[in] eps       Step sizes (rank 0).
 * @param[in] fd        Central differences (rank 0).
 * @param[in] adjoint   Adjoint directional derivative, repeated per step (rank 0).
 * @param[in] n         Number of steps.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode writeGradientCheck(const ImParams *imParams, MPI_Comm comm, const PetscReal *eps, const PetscReal *fd,
                                  const PetscReal *adjoint, PetscInt n);

#endif /* INVERSION_INTERNAL_H */
