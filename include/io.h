/*
 * Filename: io.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Generic input and output shared by the kernels: common parameters
 * (PetgemParams), the model part of the input bundle (loadModelInputs), output
 * paths and run provenance. Physics-specific I/O lives in csem.h, mt.h and
 * inversion.h.
 */

/*
 * Notes:
 * Every kernel reads a single HDF5 bundle written by the Python
 * preprocessor (utils/functions.py::writePetgemInputFile).
 */

#ifndef IO_H
#define IO_H

#include <petsc.h>
#include <petscviewer.h>

/**
 * @brief Parsed user-input parameters consumed by the PETGEM kernels.
 */
typedef struct {
  /**
   * Unified PETGEM input bundle (HDF5) - contains mesh topology, sections,
   * per-cell conductivity + materialsId, receivers (under /receivers), and
   * single-frequency forward sources (under /sources/...). Produced by
   * runPreprocessing() on the Python side. Consumed by loadModelInputs().
   */
  char inputFile[PETSC_MAX_PATH_LEN];

  char outputDirectory[PETSC_MAX_PATH_LEN]; /**< Output directory path. */
  char outputFilename[PETSC_MAX_PATH_LEN];  /**< Output filename stem for responses. */

  PetscInt order; /**< Finite-element basis order (0 = take from bundle). */

} PetgemParams;

/**
 * @brief Reads and validates the common PETGEM CLI parameters from PETSc options.
 *
 * Extracts the required runtime parameters (input/output paths) from the
 * PETSc options database. The finite-element basis order -order is optional:
 * when omitted, params->order is set to 0 so loadModelInputs takes the order
 * from the input bundle.
 *
 * @param[out] params  Struct receiving the parsed parameters.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode readPetgemParams(PetgemParams *params);

/**
 * @brief Loads the mesh, model, order and receivers from the unified PETGEM HDF5 bundle.
 *
 * Opens params->inputFile on PETSC_COMM_WORLD to load the DMPlex
 * topology, sections, and the combined model-data vector (split into the
 * per-cell conductivity and materialsId local Vecs). Opens the same file
 * again on PETSC_COMM_SELF (each rank reads independently) to load:
 *   - /order      single-element Vec, written into params->order
 *   - /receivers  Vec of 3·N_recv reals
 *
 * @param[in,out] params     Parameters; inputFile is read, order is
 *                              written from the bundle's /order dataset.
 * @param[out]    dm            Loaded DMPlex mesh.
 * @param[out]    conductivity  Per-cell conductivity Vec.
 * @param[out]    materialsId   Per-cell material-id Vec.
 * @param[out]    receivers     Serial Vec of 3·N_recv receiver reals.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode loadModelInputs(PetgemParams *params, DM *dm, Vec *conductivity, Vec *materialsId, Vec *receivers);

/**
 * @brief Loads a named Vec from the current group of an HDF5 viewer on PETSC_COMM_SELF.
 *
 * @param[in]  viewer  HDF5 viewer positioned at the group.
 * @param[in]  name    Dataset name.
 * @param[out] out     Loaded sequential Vec.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode loadSerialVec(PetscViewer viewer, const char *name, Vec *out);

/**
 * @brief Writes `{dir}/{name}` into @p out, adding the '/' only when missing.
 *
 * @param[in]  dir      Directory.
 * @param[in]  name     File or directory name.
 * @param[out] out      Buffer receiving the path.
 * @param[in]  outSize  Size of @p out in bytes.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode joinPath(const char *dir, const char *name, char *out, size_t outSize);

/**
 * @brief Builds the canonical output path `{output_dir}/{output_filename}{suffix}`.
 *
 * The single place every kernel composes an output path, so all products are
 * named the same way. A trailing '/' on the output
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
PetscErrorCode buildOutputPath(const PetgemParams *params, const char *suffix, char *out, size_t outSize);

/**
 * @brief Writes the common root provenance attributes of an output file.
 *
 * Shared by every kernel so each PETGEM product carries the same
 * traceability block, with identical attribute names and casing:
 *
 *   petgem_version, simulation_type, input_filename, order,
 *   ksp_type, pc_type, mpi_tasks, date
 *
 * The caller adds its own product-specific attributes afterwards (fm.csem:
 * num_sources, frequency; fm.mt: frequency, num_receivers; im.csem:
 * num_frequencies, lambda, error_level, num_iterations,
 * num_objgrad_evaluations, convergence_reason).
 *
 * The solver keys are read back from the PETSc options database so the file
 * records the configuration the run actually used; they read "default" when
 * the option was not set.
 *
 * @param[in] viewer          Open HDF5 viewer positioned at the file root.
 * @param[in] params          Shared base parameters (input path, order, tasks).
 * @param[in] simulationType  Kernel tag (PETGEM_SIM_FM, PETGEM_SIM_IM, PETGEM_SIM_MT).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode writeRunProvenance(PetscViewer viewer, const PetgemParams *params, const char *simulationType);

#endif
