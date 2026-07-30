/*
 * Filename: io.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Public surface for PETGEM input handling: the parsed user-input
 * parameters (petgemParams / readPetgemParams) and the unified input loader
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
} petgemParams;

/**
 * @brief Reads and validates CSEM CLI parameters from PETSc options.
 *
 * Extracts the required runtime parameters (input/output paths) from the
 * PETSc options database. The finite-element basis order -order is optional:
 * when omitted, params->order is set to 0 so loadCsemInputs takes the order
 * from the input bundle.
 *
 * @param[in]  size    Number of MPI tasks.
 * @param[out] pg_Params  Struct receiving the parsed CSEM parameters.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode readPetgemParams(const PetscMPIInt size, petgemParams* params);

/**
 * @brief Loads all CSEM inputs from the unified PETGEM HDF5 bundle.
 *
 * Opens pg_params->inputFile on PETSC_COMM_WORLD to load the DMPlex
 * topology, sections, and the combined model-data vector (split into the
 * per-cell conductivity and materials_id local Vecs). Opens the same file
 * again on PETSC_COMM_SELF (each rank reads independently) to load:
 *   - /order                 single-element Vec, written into pg_params->order
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
 * @param[in,out] pg_Params           Parameters; inputFile is read, order is
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
PetscErrorCode loadCsemInputs(petgemParams       *pg_params,
                              DM             *dm,
                              Vec            *conductivity,
                              Vec            *materialsID,
                              CsemSourceSet  *sources,
                              Vec            *receivers);

/**
 * @brief Simulation-type tags stamped into every PETGEM output product.
 *
 * Written as the `simulation_type` root attribute by writeRunProvenance so a
 * result file identifies which kernel produced it without inspecting its
 * datasets.
 */
#define PETGEM_SIM_FM "fm" /**< Forward modeling (fm.csem). */
#define PETGEM_SIM_IM "im" /**< Inverse modeling (im.csem). */

/**
 * @brief Builds the canonical output path `{output_dir}/{output_filename}{suffix}`.
 *
 * The single place both kernels compose an output path, so fm.csem and
 * im.csem name their products identically. A trailing '/' on the output
 * directory is optional (inserted when absent).
 *
 * @param[in]  params   Parameters carrying outputDirectory and outputFilename.
 * @param[in]  suffix   Extension or stem suffix, e.g. ".h5" (may be empty).
 * @param[out] out      Buffer receiving the composed path.
 * @param[in]  outSize  Size of @p out in bytes.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode buildOutputPath(const petgemParams *params, const char *suffix,
                               char *out, size_t outSize);

/**
 * @brief Writes the common root provenance attributes of an output file.
 *
 * Shared by both kernels so every PETGEM product carries the same
 * traceability block, with identical attribute names and casing:
 *
 *   petgem_version, simulation_type, input_filename, order,
 *   ksp_type, pc_type, mpi_tasks, date
 *
 * The caller adds its own product-specific attributes afterwards (fm.csem:
 * num_sources, frequency; im.csem: num_frequencies, lambda, error_level,
 * num_iterations, num_objgrad_evaluations, convergence_reason).
 *
 * The solver keys are read back from the PETSc options database so the file
 * records the configuration the run actually used; they read "default" when
 * the option was not set.
 *
 * @param[in] viewer          Open HDF5 viewer positioned at the file root.
 * @param[in] params          Shared base parameters (input path, order, tasks).
 * @param[in] simulationType  PETGEM_SIM_FM or PETGEM_SIM_IM.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode writeRunProvenance(PetscViewer viewer, const petgemParams *params,
                                  const char *simulationType);

#endif
