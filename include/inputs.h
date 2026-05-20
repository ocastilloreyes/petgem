/*
 * Filename: inputs.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Type definitions for the parsed user-input parameters consumed
 * by the PETGEM kernels.
 */

#ifndef INPUTS_H
#define INPUTS_H

#include <petsc.h>

typedef struct {
  /* Unified PETGEM input bundle (HDF5) — contains mesh topology, sections,
   * per-cell conductivity + materials_id, receivers (under /receivers), and
   * single-frequency forward sources (under /sources/...). Produced by
   * runPreprocessing() on the Python side. Consumed by loadCsemInputs(). */
  char inputFile[PETSC_MAX_PATH_LEN];

  char outputDirectory[PETSC_MAX_PATH_LEN];
  char outputFilename[PETSC_MAX_PATH_LEN];

  PetscInt    nord;
  PetscMPIInt numMPITasks;

  /* Suppress per-call assembly headers ("Assembly RHS:", "Vector size",
   * "Initiated", "Finished", "Assembly K + M(sigma)", etc.) emitted by
   * src/assembly.c. Default PETSC_FALSE preserves current fm.csem output;
   * the inversion kernel sets this to PETSC_TRUE to silence repeated
   * per-frequency / per-iteration headers in the L-BFGS loop. */
  PetscBool quiet;
} csemParams;

PetscErrorCode readCsemParams(const PetscMPIInt size, csemParams* params);

#endif
