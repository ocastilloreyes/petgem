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
 * (writeInversionResults, buildNotFixedMask) need linkage across the
 * new files but should not be advertised to im_csem.c or
 * postprocessing.c.
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
 * @param[out] notFixedMaskGlobal  Global 0/1 mask Vec.
 * @param[out] notFixedMaskLocal   Ghosted local counterpart of the mask.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode buildNotFixedMask(const NeighborGraph *graph,
                                  DM                  dmInversion,
                                  const Grid         *grid,
                                  Vec                *notFixedMaskGlobal,
                                  Vec                *notFixedMaskLocal);

/**
 * @brief Builds the parallel overlap=1 smoothing graph and workspace.
 *
 * Defined in src/inversion_smoother.c. Builds the per-rank overlap=1
 * neighbor graph and ghost-aware Vec workspace used by the fully-parallel
 * block-Jacobi Gauss-Seidel path of applyGaussSeidelSmoothing. No-op on a
 * single MPI rank. Must be called AFTER buildNeighborSmoothingGraph and
 * createInversionDM, and BEFORE the first L-BFGS evaluation.
 *
 * @param[in,out] graph  Neighbor graph extended with parallel overlap data.
 * @param[in]     dm     DMPlex mesh.
 * @param[in]     grid   Finite-element grid descriptor.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode setupParallelSmoothingGraph(NeighborGraph *graph,
                                            const DM       dm,
                                            const Grid    *grid);

/**
 * @brief Writes the final inversion results to HDF5.
 *
 * Defined in src/io.c. Dumps the recovered model, the per-iteration RMS
 * history, and provenance attributes. Called once at the end of
 * runCsemInversion.
 *
 * @param[in] iparams         Inversion parameters / provenance.
 * @param[in] dmConductivity  DM for the per-cell conductivity field.
 * @param[in] conductivity    Recovered conductivity Vec.
 * @param[in] X               Recovered log-perturbation / model Vec.
 * @param[in] allRMS          Array of per-iteration RMS values.
 * @param[in] numIters        Number of entries in allRMS.
 * @param[in] reasonStr       Human-readable convergence/stop reason.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode writeInversionResults(const imParams *iparams,
                                      DM               dmConductivity,
                                      Vec              conductivity,
                                      Vec              X,
                                      const PetscReal *allRMS,
                                      PetscInt         numIters,
                                      const char      *reasonStr);

#endif /* INVERSION_INTERNAL_H */
