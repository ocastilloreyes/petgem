/*
 * Filename: inversion_internal.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Cross-file prototypes for the inversion kernel that are NOT part of
 * the public API exposed via include/inversion.h.
 */

/*
 * Notes:
 * After splitting src/inversion.c into multiple translation units,
 * several functions that were previously file-static
 * (writeInversionResults, buildNotFixedMask, runFdGradientCheck) need
 * linkage across the new files but should not be advertised to
 * im_csem.c or postprocessing.c.  Same pattern as
 * include/hvfem_internal.h.
 */

#ifndef INVERSION_INTERNAL_H
#define INVERSION_INTERNAL_H

#include "inversion.h"
#include <petsc.h>

/* Defined in src/inversion_smoother.c. Build the persistent 0/1 mask
 * Vec used by inversionObjGrad to zero gradient entries at fixed cells. */
PetscErrorCode buildNotFixedMask(const NeighborGraph *graph,
                                  DM                  dmInversion,
                                  const Grid         *grid,
                                  Vec                *notFixedMaskGlobal,
                                  Vec                *notFixedMaskLocal);

/* Defined in src/inversion_smoother.c. Builds the per-rank overlap=1
 * neighbor graph + ghost-aware Vec workspace used by the fully-parallel
 * block-Jacobi Gauss-Seidel path of applyGaussSeidelSmoothing. No-op on
 * a single MPI rank. Must be called AFTER buildNeighborSmoothingGraph
 * and createInversionDM, and BEFORE the first L-BFGS evaluation. */
PetscErrorCode setupParallelSmoothingGraph(NeighborGraph *graph,
                                            const DM       dm,
                                            const Grid    *grid);

/* Defined in src/io.c. Final HDF5 dump of the recovered model
 * + RMS history + provenance attributes. Called once at the end of
 * runCsemInversion. */
PetscErrorCode writeInversionResults(const invParams *iparams,
                                      DM               dmConductivity,
                                      Vec              conductivity,
                                      Vec              X,
                                      const PetscReal *allRMS,
                                      PetscInt         numIters,
                                      const char      *reasonStr);

/* Defined in src/inversion.c. Developer-only finite-difference gradient
 * check. Triggered by -inv_dev_fd_check N; bypasses the L-BFGS loop. */
PetscErrorCode runFdGradientCheck(InversionContext *ctx, Vec X,
                                   PetscInt ncells, PetscReal eps);

#endif /* INVERSION_INTERNAL_H */
