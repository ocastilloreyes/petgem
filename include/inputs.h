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

/**
 * @brief Parsed user-input parameters consumed by the PETGEM kernels.
 */
typedef struct {
  /**
   * Unified PETGEM input bundle (HDF5) - contains mesh topology, sections,
   * per-cell conductivity + materials_id, receivers (under /receivers), and
   * single-frequency forward sources (under /sources/...). Produced by
   * runPreprocessing() on the Python side. Consumed by loadCsemInputs().
   */
  char inputFile[PETSC_MAX_PATH_LEN];

  char outputDirectory[PETSC_MAX_PATH_LEN]; /**< Output directory path. */
  char outputFilename[PETSC_MAX_PATH_LEN];  /**< Output filename stem for responses. */

  PetscInt    nord;        /**< Finite-element basis order (0 = take from bundle). */
  PetscMPIInt numMPITasks; /**< Number of MPI tasks in the run. */

  /**
   * Suppress per-call assembly headers ("Assembly RHS:", "Vector size",
   * "Initiated", "Finished", "Assembly K + M(sigma)", etc.) emitted by
   * src/assembly.c. Default PETSC_FALSE preserves current fm.csem output;
   * the inversion kernel sets this to PETSC_TRUE to silence repeated
   * per-frequency / per-iteration headers in the L-BFGS loop.
   */
  PetscBool quiet;
} fmParams;

/**
 * @brief Reads and validates CSEM CLI parameters from PETSc options.
 *
 * Extracts the required runtime parameters (input/output paths) from the
 * PETSc options database. The finite-element basis order -nord is optional:
 * when omitted, params->nord is set to 0 so loadCsemInputs takes the order
 * from the input bundle.
 *
 * @param[in]  size    Number of MPI tasks.
 * @param[out] params  Struct receiving the parsed CSEM parameters.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode readfmParams(const PetscMPIInt size, fmParams* params);

#endif
