/*
 * Filename: io.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Public surface for PETGEM input handling: the parsed user-input
 * parameters (fmParams / readfmParams) and the unified input loader
 * (loadCsemInputs).
 */

/*
 * Notes:
 * The loader replaces the legacy importGrid + setupCsemSource +
 * per-call receivers-file open path with a single open of the bundled
 * HDF5 file produced by the Python preprocessor
 * (utils/functions.py::writeBundle).
 */

#ifndef IO_H
#define IO_H

#include "transmitter.h"
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

  PetscInt    order;        /**< Finite-element basis order (0 = take from bundle). */
  PetscMPIInt numMPITasks; /**< Number of MPI tasks in the run. */

  /**
   * Suppress per-call assembly headers ("Assembly RHS:", "Vector size",
   * "Initiated", "Finished", "Assembly K + M(sigma)", etc.) emitted by
   * src/assembly.c. Default PETSC_FALSE preserves current fm.csem output;
   * the inversion kernel sets this to PETSC_TRUE to silence repeated
   * per-frequency / per-iteration headers in the L-BFGS loop.
   */
  PetscBool quiet;

  /**
   * Method-of-Manufactured-Solutions verification mode (forward kernel only).
   * When PETSC_TRUE, runForward dispatches to runMMSVerification (src/mms.c),
   * which runs the complete MMS verification (Galerkin solve + L2 projection +
   * optional diagnostics) and exits, skipping the normal receiver pipeline.
   * Enabled with -mms; default PETSC_FALSE. See include/mms.h and
   * paper/tests_mms/. im.csem ignores it.
   */
  PetscBool mms;
} fmParams;

/**
 * @brief Reads and validates CSEM CLI parameters from PETSc options.
 *
 * Extracts the required runtime parameters (input/output paths) from the
 * PETSc options database. The finite-element basis order -order is optional:
 * when omitted, params->order is set to 0 so loadCsemInputs takes the order
 * from the input bundle.
 *
 * @param[in]  size    Number of MPI tasks.
 * @param[out] fm_Params  Struct receiving the parsed CSEM parameters.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode readfmParams(const PetscMPIInt size, fmParams* params);

/**
 * @brief Loads all CSEM inputs from the unified PETGEM HDF5 bundle.
 *
 * Opens fm_params->inputFile on PETSC_COMM_WORLD to load the DMPlex
 * topology, sections, and the combined model-data vector (split into the
 * per-cell conductivity and materials_id local Vecs). Opens the same file
 * again on PETSC_COMM_SELF (each rank reads independently) to load:
 *   - /order                 single-element Vec, written into fm_params->order
 *   - /receivers            Vec of 3·N_recv reals
 *   - /sources/frequency    single-frequency scalar
 *   - /sources/position     Vec of 3·N_src reals
 *   - /sources/current      Vec of N_src reals
 *   - /sources/length       Vec of N_src reals
 *   - /sources/dipAngle     Vec of N_src reals
 *   - /sources/azimuthAngle Vec of N_src reals
 *
 * Replaces the legacy importGrid + setupCsemSource + per-call
 * receivers-file open path with a single open of the bundle produced by
 * the Python preprocessor (utils/functions.py::writeBundle).
 *
 * @param[in,out] fm_Params           Parameters; inputFile is read, order is
 *                                    written from the bundle's /order dataset.
 * @param[out]    odm                 Loaded DMPlex mesh.
 * @param[out]    conductivity_output Per-cell conductivity Vec.
 * @param[out]    materials_id_output Per-cell material-id Vec.
 * @param[out]    sources             Forward transmitter set; pass NULL to skip
 *                                    (im.csem pulls multi-frequency sources via
 *                                    setupInversionSources instead).
 * @param[out]    receivers_output    Serial Vec of 3·N_recv receiver reals; pass
 *                                    NULL to skip.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode loadCsemInputs(fmParams       *fm_params,
                              DM             *dm,
                              Vec            *conductivity,
                              Vec            *materialsID,
                              CsemSourceSet  *sources,
                              Vec            *receivers);

#endif
