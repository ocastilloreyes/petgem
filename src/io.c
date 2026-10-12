/*
 * Filename: io.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Generic input and output shared by the kernels: common parameters, the
 * model part of the input bundle, output paths and run provenance.
 */

/* C libraries */
#include <stdio.h>
#include <string.h>

/* PETSc functions*/
#include <petsc.h>
#include <petscsys.h>
#include <petscviewerhdf5.h>

/* PETGEM headers */
#include "common.h"
#include "constants.h"
#include "io.h"
#include "version.h"

/**
 * @brief Loads a PETSc Vec from an HDF5 viewer on PETSC_COMM_SELF.
 *
 * This helper reads a named dataset from the current HDF5 group
 * and creates a sequential Vec containing its values. It is intended
 * for small datasets (e.g., scalars or per-process data) that are
 * loaded locally rather than in parallel.
 *
 * The Vec is assigned the same name as the HDF5 dataset for
 * debugging and traceability.
 */
PetscErrorCode loadSerialVec(PetscViewer viewer, const char *name, Vec *out) {
  PetscFunctionBeginUser;
  PetscCall(VecCreate(PETSC_COMM_SELF, out));
  PetscCall(PetscObjectSetName((PetscObject)*out, name));
  PetscCall(VecLoad(*out, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Clones @p dm with a plain section of @p numDof DOFs per cell.
 *
 * @param[in]  dm      DMPlex mesh.
 * @param[in]  numDof  DOFs per cell.
 * @param[out] cellDm  Cloned DM carrying the per-cell section.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
static PetscErrorCode createCellFieldDm(DM dm, PetscInt numDof, DM *cellDm) {
  PetscFunctionBeginUser;
  PetscSection section;
  PetscInt     pStart, pEnd, cellStart, cellEnd;
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cellStart, &cellEnd));
  PetscCall(DMClone(dm, cellDm));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section));
  PetscCall(PetscSectionSetChart(section, pStart, pEnd));
  for (PetscInt i = cellStart; i < cellEnd; i++) PetscCall(PetscSectionSetDof(section, i, numDof));
  PetscCall(PetscSectionSetUp(section));
  PetscCall(DMSetLocalSection(*cellDm, section));
  PetscCall(PetscSectionDestroy(&section));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Writes `{dir}/{name}` into @p out, adding the '/' only when missing.
 */
PetscErrorCode joinPath(const char *dir, const char *name, char *out, size_t outSize) {
  PetscFunctionBeginUser;
  PetscCall(PetscStrncpy(out, dir, outSize));
  size_t len = strlen(out);
  if (len > 0 && out[len - 1] != '/') PetscCall(PetscStrlcat(out, "/", outSize));
  PetscCall(PetscStrlcat(out, name, outSize));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Builds the canonical output path `{output_dir}/{output_filename}{suffix}`.
 *
 * See io.h. Every kernel names its output product through this function.
 */
PetscErrorCode buildOutputPath(const PetgemParams *params, const char *suffix, char *out, size_t outSize) {
  PetscFunctionBeginUser;

  PetscCall(joinPath(params->outputDirectory, params->outputFilename, out, outSize));
  if (suffix && suffix[0] != '\0') {
    PetscCall(PetscStrlcat(out, suffix, outSize));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Writes the common root provenance attributes of an output file.
 *
 * See io.h. Both kernels call this so every PETGEM product carries the same
 * traceability block under identical attribute names; each kernel then adds
 * only its product-specific attributes.
 */
PetscErrorCode writeRunProvenance(PetscViewer viewer, const PetgemParams *params, const char *simulationType) {
  PetscFunctionBeginUser;

  char      version[64];
  char      date[64];
  char      kspType[PETSC_MAX_PATH_LEN] = "";
  char      pcType[PETSC_MAX_PATH_LEN]  = "";
  PetscBool flg;

  snprintf(version, sizeof(version), "%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
  PetscCall(PetscGetDate(date, sizeof(date)));

  /* Record the solver configuration the run actually used, as resolved from
   * the options database (params file + command line). */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-ksp_type", kspType, sizeof(kspType), &flg));
  if (!flg) PetscCall(PetscStrncpy(kspType, "default", sizeof(kspType)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-pc_type", pcType, sizeof(pcType), &flg));
  if (!flg) PetscCall(PetscStrncpy(pcType, "default", sizeof(pcType)));

  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "petgem_version", PETSC_STRING, version));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "simulation_type", PETSC_STRING, simulationType));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "input_filename", PETSC_STRING, params->inputFile));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "order", PETSC_INT, &params->order));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "ksp_type", PETSC_STRING, kspType));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "pc_type", PETSC_STRING, pcType));

  PetscMPIInt size;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscInt mpiTasks = (PetscInt)size;
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "mpi_tasks", PETSC_INT, &mpiTasks));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "date", PETSC_STRING, date));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Reads and validates the common PETGEM CLI parameters from PETSc options.
 *
 * This function extracts required runtime parameters from the PETSc
 * options database, including input/output paths and the finite-element
 * basis order. All parameters are mandatory except -order, which can
 * optionally override the value stored in the input bundle.
 *
 * The -order option is mainly intended for debugging runs. If not
 * provided, params->order is set to 0, meaning the value will be
 * taken from the input bundle by loadModelInputs.
 */
PetscErrorCode readPetgemParams(PetgemParams *params) {

  PetscFunctionBeginUser;

  char      inputFilename[PETSC_MAX_PATH_LEN]  = "";
  char      outputDir[PETSC_MAX_PATH_LEN]      = "";
  char      outputFilename[PETSC_MAX_PATH_LEN] = "";
  PetscBool inputIsPresent                     = PETSC_FALSE;
  PetscBool outputDirIsPresent                 = PETSC_FALSE;
  PetscBool outputFilenameIsPresent            = PETSC_FALSE;
  PetscBool orderIsPresent                     = PETSC_FALSE;
  PetscBool helpRequested                      = PETSC_FALSE;
  PetscInt  order                              = 0;

  /* When the user runs with -help / -help intro, the option blocks below are registered (so PETSc documents them), but the mandatory-argument checks
   * are skipped so a help run prints usage cleanly instead of aborting on a missing -input_filename. The caller exits after parsing in that case. */
  PetscCall(PetscOptionsHasHelp(NULL, &helpRequested));

  /* Two PetscOptionsBegin/End groups so `-help` renders a clean "required" vs
   * "optional" split (PETSc prints groups in registration order).
   *
   * These are the options common to both kernels, so the groups are labeled
   * "PETGEM" rather than "fm.csem": im.csem registers them through this same
   * reader. */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "PETGEM: required options", "PETGEM");

  PetscCall(PetscOptionsString("-input_filename", "[REQUIRED] Unified PETGEM input bundle (mesh, sigma, materials_id, receivers, sources)",
                               "PETGEM", inputFilename, inputFilename, sizeof(inputFilename), &inputIsPresent));

  PetscCall(PetscOptionsString("-output_dir", "[REQUIRED] Output directory (created if absent)", "PETGEM", outputDir, outputDir,
                               sizeof(outputDir), &outputDirIsPresent));

  PetscCall(PetscOptionsString("-output_filename", "[REQUIRED] Output filename stem; writes <output_dir>/<stem>.h5", "PETGEM",
                               outputFilename, outputFilename, sizeof(outputFilename), &outputFilenameIsPresent));
  PetscOptionsEnd();

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "PETGEM: optional options", "PETGEM");
  PetscCall(PetscOptionsInt("-order", "Basis order override (1..6); default = take from bundle /order", "PETGEM", order, &order,
                            &orderIsPresent));

  PetscOptionsEnd();

  /* Both option groups have now been printed under -help; skip argument enforcement and model/directory setup so usage prints cleanly. The
   * caller (runForward / runInverse) exits right after parsing. */
  if (helpRequested) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCheck(inputIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Missing required -input_filename. Run with -help intro for usage.");
  PetscCheck(outputDirIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Missing required -output_dir. Run with -help intro for usage.");
  PetscCheck(outputFilenameIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Missing required -output_filename. Run with -help intro for usage.");

  PetscCall(PetscStrncpy(params->inputFile, inputFilename, sizeof(params->inputFile)));
  PetscCall(PetscStrncpy(params->outputDirectory, outputDir, sizeof(params->outputDirectory)));
  PetscCall(PetscStrncpy(params->outputFilename, outputFilename, sizeof(params->outputFilename)));

  /* 0: loadModelInputs takes the order from the bundle */
  params->order = orderIsPresent ? order : 0;

  PetscCall(createDirectory(outputDir));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Distributed mesh and per-cell conductivity / material id (parallel I/O). */
static PetscErrorCode loadMeshAndModel(const char *inputFile, DM *odm, Vec *conductivity, Vec *materialsId) {
  PetscFunctionBeginUser;
  PetscViewer  viewer;
  DM           dm, dmDist, subDmSigma, subDmMat;
  PetscSF      sfLoad, sfDist = NULL, sfXc = NULL, sfG;
  PetscSection combinedSection, localSecSigma, localSecMat;
  Vec          combinedGlobal, combinedLocal, localSigma, localMat;
  char         typeName[PETSC_MAX_PATH_LEN];
  PetscBool    flg;
  PetscInt     cellStart, cellEnd;

  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(PetscObjectSetName((PetscObject)dm, "petgem_mesh"));

  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, inputFile, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_PETSC));

  PetscCall(DMPlexTopologyLoad(dm, viewer, &sfLoad));
  PetscCall(DMPlexLabelsLoad(dm, viewer, sfLoad));
  PetscCall(DMPlexCoordinatesLoad(dm, viewer, sfLoad));
  PetscCall(DMPlexDistribute(dm, 0, &sfDist, &dmDist));
  if (dmDist) {
    PetscCall(PetscSFCompose(sfLoad, sfDist, &sfXc));
    PetscCall(DMDestroy(&dm));
    dm = dmDist;
    PetscCall(PetscObjectSetName((PetscObject)dm, "petgem_mesh"));
  } else {
    PetscCall(PetscObjectReference((PetscObject)sfLoad));
    sfXc = sfLoad;
  }
  PetscCall(DMViewFromOptions(dm, NULL, "-load_dm_view"));

  /* 2-field section: field 0 = conductivity (3 dofs/cell), field 1 = materialsId (1 dof/cell).
   * DMSetNumFields is not called (it fails without a discretization object); the layout comes from the loaded section. */
  PetscCall(DMPlexSectionLoad(dm, viewer, NULL, sfXc, &sfG, NULL));

  /* Combined global vector - 4 dofs/cell [sigma_x sigma_y sigma_z mat_id] */
  PetscCall(DMCreateGlobalVector(dm, &combinedGlobal));
  PetscCall(PetscObjectSetName((PetscObject)combinedGlobal, "model_data"));
  PetscCall(DMPlexGlobalVectorLoad(dm, viewer, NULL, sfG, combinedGlobal));
  PetscCall(VecViewFromOptions(combinedGlobal, NULL, "-load_model_view"));

  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscSFDestroy(&sfLoad));
  PetscCall(PetscSFDestroy(&sfDist));
  PetscCall(PetscSFDestroy(&sfXc));
  PetscCall(PetscSFDestroy(&sfG));

  /* Global -> local so ghosts (needed for closure walks across partition boundaries) are populated on every rank. */
  PetscCall(DMCreateLocalVector(dm, &combinedLocal));
  PetscCall(DMGlobalToLocal(dm, combinedGlobal, INSERT_VALUES, combinedLocal));
  PetscCall(VecDestroy(&combinedGlobal));

  PetscCall(DMGetLocalSection(dm, &combinedSection));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cellStart, &cellEnd));

  /* Per-cell sub-DMs: conductivity (3 dofs/cell) and materials id (1 dof/cell) */
  PetscCall(createCellFieldDm(dm, NUM_CONDUCTIVITY_COMPONENTS, &subDmSigma));
  PetscCall(createCellFieldDm(dm, NUM_MATERIALS_ID_COMPONENTS, &subDmMat));

  PetscCall(DMCreateLocalVector(subDmSigma, &localSigma));
  PetscCall(DMCreateLocalVector(subDmMat, &localMat));

  PetscCall(DMGetLocalSection(subDmSigma, &localSecSigma));
  PetscCall(DMGetLocalSection(subDmMat, &localSecMat));

  /* Split the combined per-cell block [sigma_x sigma_y sigma_z mat_id] into the two sub-Vecs */
  {
    const PetscScalar *cArr;
    PetscScalar       *sigmaArr, *matArr;
    PetscCall(VecGetArrayRead(combinedLocal, &cArr));
    PetscCall(VecGetArray(localSigma, &sigmaArr));
    PetscCall(VecGetArray(localMat, &matArr));
    for (PetscInt i = cellStart; i < cellEnd; i++) {
      PetscInt cOff, sigmaOff, matOff;
      PetscCall(PetscSectionGetOffset(combinedSection, i, &cOff));
      PetscCall(PetscSectionGetOffset(localSecSigma, i, &sigmaOff));
      PetscCall(PetscSectionGetOffset(localSecMat, i, &matOff));
      sigmaArr[sigmaOff + 0] = cArr[cOff + 0];
      sigmaArr[sigmaOff + 1] = cArr[cOff + 1];
      sigmaArr[sigmaOff + 2] = cArr[cOff + 2];
      matArr[matOff]         = cArr[cOff + 3];
    }
    PetscCall(VecRestoreArray(localMat, &matArr));
    PetscCall(VecRestoreArray(localSigma, &sigmaArr));
    PetscCall(VecRestoreArrayRead(combinedLocal, &cArr));
  }
  PetscCall(VecDestroy(&combinedLocal));
  PetscCall(DMDestroy(&subDmSigma));
  PetscCall(DMDestroy(&subDmMat));

  /* Cloned DM is the public-facing handle; setupNedelecGrid installs the H(curl) section on it without disturbing the original */
  PetscCall(DMClone(dm, odm));
  PetscCall(DMDestroy(&dm));

  *conductivity = localSigma;
  *materialsId  = localMat;

  PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_vec_type", typeName, sizeof(typeName), &flg));
  if (flg) {
    PetscCall(DMSetVecType(*odm, typeName));
  }
  PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_mat_type", typeName, sizeof(typeName), &flg));
  if (flg) {
    PetscCall(DMSetMatType(*odm, typeName));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Basis order (unless set by -order) and receivers (serial I/O on every rank). */
static PetscErrorCode loadOrderAndReceivers(PetgemParams *params, Vec *receivers) {
  PetscFunctionBeginUser;
  PetscViewer viewer;
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, params->inputFile, FILE_MODE_READ, &viewer));

  /* /order - single-element Vec. Bundle is the source of truth unless the caller explicitly set -order on the command line 
  * (params->order non-zero at entry). */
  if (params->order <= 0) {
    Vec                orderV;
    const PetscScalar *nArr;
    PetscCall(loadSerialVec(viewer, "order", &orderV));
    PetscCall(VecGetArrayRead(orderV, &nArr));
    params->order = (PetscInt)(PetscRealPart(nArr[0]) + 0.5);
    PetscCall(VecRestoreArrayRead(orderV, &nArr));
    PetscCall(VecDestroy(&orderV));
  }
  PetscCheck(params->order >= 1 && params->order <= 6, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "Basis order %" PetscInt_FMT " not in 1..6 (-order or /order of %s)", params->order, params->inputFile);

  PetscCall(loadSerialVec(viewer, "receivers", receivers));
  PetscCall(VecSetBlockSize(*receivers, NUM_DIMENSIONS));

  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Loads the mesh, model, order and receivers from a unified PETGEM HDF5 bundle.
 *
 * This routine reads the method-independent part of the simulation input
 * file (mesh, model fields, order and receivers) and reconstructs the
 * runtime PETSc objects required by the solver.
 *
 * Loading is split into two stages:
 *
 * - Phase 1 (parallel I/O): builds the DMPlex mesh, loads topology,
 *   distributes it across ranks, and reconstructs the combined per-cell
 *   model vector. This vector is then split into conductivity and
 *   material-ID fields using local sections.
 *
 * - Phase 2 (serial I/O on PETSC_COMM_SELF): reads /order and /receivers.
 *
 * The function enforces consistency between the bundled data layout
 * and the internal DM structure, and applies CLI overrides when present
 * (e.g., -order).
 */
PetscErrorCode loadModelInputs(PetgemParams *params, DM *dm, Vec *conductivity, Vec *materialsId, Vec *receivers) {
  PetscFunctionBeginUser;
  PetscCheck(params->inputFile[0] != '\0', PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "loadModelInputs: params->inputFile is empty (set -input_filename).");
  PetscCall(loadMeshAndModel(params->inputFile, dm, conductivity, materialsId));
  PetscCall(loadOrderAndReceivers(params, receivers));
  PetscFunctionReturn(PETSC_SUCCESS);
}
