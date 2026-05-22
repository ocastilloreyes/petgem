/*
 * Filename: postprocessing.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Prototypes for the postprocessing functions used throughout PETGEM.
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include "grid.h"
#include "inputs.h"
#include "transmitter.h"
#include <petsc.h>
#include <petscdmplex.h>

/* Compute electric and magnetic fields at receivers for the forward kernel.
 * `receivers` is the serial Vec (PETSC_COMM_SELF, length 3·N_recv) returned
 * by loadCsemInputs - passed through so postprocessing does not re-open
 * the input HDF5. */
PetscErrorCode computeFields(const csemParams params, const CsemSourceSet sources,
                             const DM dm, const Grid grid,
                             Vec receivers, const Mat X);

#endif
