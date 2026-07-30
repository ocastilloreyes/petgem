/*
 * Filename: io.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Centralized I/O and parameter parsing for PETGEM kernels.
 */

/* C libraries */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

/* PETSc functions*/
#include <petsc.h>
#include <petscsys.h>
#include <petscviewerhdf5.h>
#include <hdf5.h>

/* PETGEM headers */
#include "common.h"
#include "constants.h"
#include "grid.h"
#include "inversion.h"
#include "inversion_internal.h"
#include "io.h"
#include "transmitter.h"
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
 *
 * @param[in]  viewer  HDF5 PETSc viewer positioned at the target group.
 * @param[in]  name    Dataset name inside the HDF5 file.
 * @param[out] out     Loaded sequential Vec.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
static PetscErrorCode loadSelfVecByName(PetscViewer viewer, const char *name, Vec *out)
{
  PetscFunctionBeginUser;
  PetscCall(VecCreate(PETSC_COMM_SELF, out));
  PetscCall(PetscObjectSetName((PetscObject)*out, name));
  PetscCall(VecLoad(*out, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Writes a scalar cell-field as an ASCII VTK DataArray.
 *
 * This helper emits a single cell-centered field into a VTU file
 * using ASCII formatting. Each value corresponds to one cell.
 *
 * It is used during VTU export to serialize inversion-related fields
 * such as model parameters or gradients.
 *
 * @param[in] fp    Open file handle for VTU output.
 * @param[in] name  Field name in the VTK dataset.
 * @param[in] a     Array of scalar values (cell-wise).
 * @param[in] n     Number of cells.
 */
static void writeVtuCellField(FILE *fp, const char *name,
                              const PetscReal *a, PetscInt n)
{
  fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", name);
  for (PetscInt i = 0; i < n; i++) fprintf(fp, "%.9g\n", (double)a[i]);
  fprintf(fp, "        </DataArray>\n");
}

/**
 * @brief Shared list of VTU cell-data field names for inversion output.
 *
 * This array defines the canonical names of all cell-centered fields
 * written in VTU/PVtu inversion outputs. It is shared between per-rank
 * .vtu pieces and the master .pvtu file to ensure consistent naming
 * across the full dataset.
 *
 * Keeping the names centralized prevents drift between writers and
 * guarantees compatibility with visualization tools such as ParaView.
 */
static const char *imVtuFieldNames[IM_VTU_NUM_FIELDS] = {
  "rho"
};

/**
 * @brief Writes a per-rank VTU piece for inversion visualization.
 *
 * This function generates a standalone VTK UnstructuredGrid (.vtu)
 * file containing the portion of the mesh owned by the current MPI
 * rank. Each tetrahedral cell is "exploded" into its own set of
 * vertices, avoiding the need for global point renumbering.
 *
 * The output is designed to be combined into a parallel .pvtu dataset
 * for visualization in ParaView.
 *
 * Cell data fields are written consistently using the shared global
 * field name list (imVtuFieldNames), ensuring compatibility between
 * all ranks and the master .pvtu file.
 *
 * Even ranks with zero owned cells produce a valid empty VTU piece.
 *
 * @param[in] filename   Output VTU filename for this rank.
 * @param[in] nCells     Number of owned cells.
 * @param[in] coords     Cell vertex coordinates (flattened, 4 vertices/cell).
 * @param[in] fields     Array of pointers to cell-wise scalar fields.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
static PetscErrorCode writeVtuPiece(const char *filename, PetscInt nCells,
                                    const PetscReal *coords,
                                    const PetscReal *const fields[IM_VTU_NUM_FIELDS])
{
  PetscFunctionBeginUser;
  FILE *fp = fopen(filename, "w");
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
             "writeVtuPiece: cannot open %s", filename);
  PetscInt npts = 4 * nCells;
  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
  fprintf(fp, "  <UnstructuredGrid>\n");
  fprintf(fp, "    <Piece NumberOfPoints=\"%" PetscInt_FMT "\" NumberOfCells=\"%" PetscInt_FMT "\">\n",
          npts, nCells);
  fprintf(fp, "      <Points>\n");
  fprintf(fp, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (PetscInt c = 0; c < nCells; c++)
    for (PetscInt v = 0; v < 4; v++)
      fprintf(fp, "%.9g %.9g %.9g\n",
              (double)coords[12 * c + 3 * v + 0],
              (double)coords[12 * c + 3 * v + 1],
              (double)coords[12 * c + 3 * v + 2]);
  fprintf(fp, "        </DataArray>\n      </Points>\n");
  fprintf(fp, "      <Cells>\n");
  fprintf(fp, "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
  for (PetscInt p = 0; p < npts; p++) fprintf(fp, "%" PetscInt_FMT " ", p);
  fprintf(fp, "\n        </DataArray>\n");
  fprintf(fp, "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
  for (PetscInt c = 1; c <= nCells; c++) fprintf(fp, "%" PetscInt_FMT " ", 4 * c);
  fprintf(fp, "\n        </DataArray>\n");
  fprintf(fp, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (PetscInt c = 0; c < nCells; c++) fprintf(fp, "10 ");   /* VTK_TETRA */
  fprintf(fp, "\n        </DataArray>\n      </Cells>\n");
  fprintf(fp, "      <CellData Scalars=\"%s\">\n", imVtuFieldNames[0]);
  for (PetscInt f = 0; f < IM_VTU_NUM_FIELDS; f++)
    writeVtuCellField(fp, imVtuFieldNames[f], fields[f], nCells);
  fprintf(fp, "      </CellData>\n");
  fprintf(fp, "    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n");
  fclose(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Builds the canonical output path `{output_dir}/{output_filename}{suffix}`.
 *
 * See io.h. Single implementation shared by the forward responses writer
 * (computeFields), the inversion results writer (writeInversionResults) and
 * the VTU snapshot writer, so all three name their products identically.
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
                               char *out, size_t outSize)
{
  PetscFunctionBeginUser;

  PetscCall(PetscStrncpy(out, params->outputDirectory, outSize));
  size_t len = strlen(out);
  if (len > 0 && out[len - 1] != '/') {
    PetscCall(PetscStrlcat(out, "/", outSize));
  }
  PetscCall(PetscStrlcat(out, params->outputFilename, outSize));
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
 *
 * @param[in] viewer          Open HDF5 viewer positioned at the file root.
 * @param[in] params          Shared base parameters (input path, order, tasks).
 * @param[in] simulationType  PETGEM_SIM_FM or PETGEM_SIM_IM.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode writeRunProvenance(PetscViewer viewer, const petgemParams *params,
                                  const char *simulationType)
{
  PetscFunctionBeginUser;

  char      version[64];
  char      date[64];
  char      kspType[PETSC_MAX_PATH_LEN] = "";
  char      pcType[PETSC_MAX_PATH_LEN]  = "";
  PetscBool flg;

  snprintf(version, sizeof(version), "%d.%d.%d",
           VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
  PetscCall(PetscGetDate(date, sizeof(date)));

  /* Record the solver configuration the run actually used, as resolved from
   * the options database (params file + command line). */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-ksp_type", kspType, sizeof(kspType), &flg));
  if (!flg) PetscCall(PetscStrncpy(kspType, "default", sizeof(kspType)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-pc_type", pcType, sizeof(pcType), &flg));
  if (!flg) PetscCall(PetscStrncpy(pcType, "default", sizeof(pcType)));

  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "petgem_version",  PETSC_STRING, version));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "simulation_type", PETSC_STRING, simulationType));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "input_filename",  PETSC_STRING, params->inputFile));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "order",           PETSC_INT,    &params->order));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "ksp_type",        PETSC_STRING, kspType));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "pc_type",         PETSC_STRING, pcType));
  
  PetscInt mpiTasks = (PetscInt)params->numMPITasks;
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "mpi_tasks",       PETSC_INT,    &mpiTasks));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "date",            PETSC_STRING, date));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Reads and validates CSEM CLI parameters from PETSc options.
 *
 * This function extracts required runtime parameters from the PETSc
 * options database, including input/output paths and the finite-element
 * basis order. All parameters are mandatory except -order, which can
 * optionally override the value stored in the input bundle.
 *
 * The -order option is mainly intended for debugging runs. If not
 * provided, pg_Params->order is set to 0, meaning the value will be
 * taken from the input bundle by loadCsemInputs.
 *
 * @param[in]  size    Number of MPI tasks.
 * @param[out] pg_Params  Struct containing parsed CSEM parameters.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode readPetgemParams(const PetscMPIInt size, petgemParams* pg_Params) {

  PetscFunctionBeginUser;

  char      inputFilename[PETSC_MAX_PATH_LEN]  = "";
  char      outputDir[PETSC_MAX_PATH_LEN]      = "";
  char      outputFilename[PETSC_MAX_PATH_LEN] = "";
  PetscBool inputIsPresent           = PETSC_FALSE;
  PetscBool outputDirIsPresent       = PETSC_FALSE;
  PetscBool outputFilenameIsPresent  = PETSC_FALSE;
  PetscBool orderIsPresent            = PETSC_FALSE;
  PetscBool helpRequested            = PETSC_FALSE;
  PetscInt  order                     = 0;

  /* When the user runs with -help / -help intro, the option blocks below are registered (so PETSc documents them), but the mandatory-argument checks
   * are skipped so a help run prints usage cleanly instead of aborting on a missing -input_filename. The caller exits after parsing in that case. */
  PetscCall(PetscOptionsHasHelp(NULL, &helpRequested));

  /* Two PetscOptionsBegin/End groups so `-help` renders a clean "required" vs "optional" split (PETSc prints groups in registration order).
   *
   * These are the options COMMON to both kernels, so the groups are labelled "PETGEM", not "fm.csem": im.csem registers them through this same 
   * reader, and previously its -help presented them under an "fm.csem:" heading. */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "PETGEM: required options", "PETGEM");

  PetscCall(PetscOptionsString("-input_filename", "[REQUIRED] Unified PETGEM input bundle (mesh, sigma, materials_id, receivers, sources)",
                               "PETGEM", inputFilename, inputFilename, sizeof(inputFilename), &inputIsPresent));

  PetscCall(PetscOptionsString("-output_dir", "[REQUIRED] Output directory (created if absent)", "PETGEM", outputDir, outputDir,
                               sizeof(outputDir), &outputDirIsPresent));

  PetscCall(PetscOptionsString("-output_filename", "[REQUIRED] Output filename stem; writes <output_dir>/<stem>.h5", "PETGEM", outputFilename, outputFilename,
                               sizeof(outputFilename), &outputFilenameIsPresent));
  PetscOptionsEnd();

  PetscBool mmsMode = PETSC_FALSE;
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "PETGEM: optional options", "PETGEM");
  PetscCall(PetscOptionsInt("-order", "Basis order override (1..6); default = take from bundle /order", "PETGEM", order, &order, &orderIsPresent));
  PetscCall(PetscOptionsBool("-mms", "[fm only] Method-of-Manufactured-Solutions verification: volumetric forcing f* and L2/H(curl) error norms (unit cube [0,1]^3)",
                             "PETGEM", mmsMode, &mmsMode, NULL));

  PetscOptionsEnd();

  /* Both option groups have now been printed under -help; skip argument enforcement and model/directory setup so usage prints cleanly. The
   * caller (runForward / runInverse) exits right after parsing. */
  if (helpRequested) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCheck(inputIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Missing required -input_filename. Run ./fm.csem -help intro for usage.");
  PetscCheck(outputDirIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Missing required -output_dir. Run  ./fm.csem -help intro  for usage.");
  PetscCheck(outputFilenameIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Missing required -output_filename. Run  ./fm.csem -help intro  for usage.");

  PetscCall(PetscStrncpy(pg_Params->inputFile,        inputFilename,  sizeof(pg_Params->inputFile)));
  PetscCall(PetscStrncpy(pg_Params->outputDirectory,  outputDir,      sizeof(pg_Params->outputDirectory)));
  PetscCall(PetscStrncpy(pg_Params->outputFilename,   outputFilename, sizeof(pg_Params->outputFilename)));

  if (orderIsPresent) {
    PetscCheck(order >= 1 && order <= 6, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Exiting: -order override out of valid range (must be 1..6).");
    pg_Params->order = order;
  } else {
    /* sentinel: loadCsemInputs will fill from /order */
    pg_Params->order = 0;
  }

  pg_Params->numMPITasks = size;
  pg_Params->quiet       = PETSC_FALSE;
  pg_Params->mms         = mmsMode;

  createDirectory(outputDir);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Loads all CSEM inputs from a unified PETGEM HDF5 bundle.
 *
 * This routine reads the full simulation input file (mesh, model fields,
 * sources, receivers, and auxiliary parameters) and reconstructs the
 * runtime PETSc objects required by the solver.
 *
 * Loading is split into two stages:
 *
 * - Phase 1 (parallel I/O): builds the DMPlex mesh, loads topology,
 *   distributes it across ranks, and reconstructs the combined per-cell
 *   model vector. This vector is then split into conductivity and
 *   material-ID fields using local sections.
 *
 * - Phase 2 (serial I/O on PETSC_COMM_SELF): reads scalar or small
 *   datasets such as /order, receivers, and source definitions.
 *
 * The function enforces consistency between the bundled data layout
 * and the internal DM structure, and applies CLI overrides when present
 * (e.g., -order).
 *
 * @param[in,out] pg_Params           Runtime parameters (input file, order, etc.).
 * @param[out]    odm                 Output distributed mesh (DMPlex).
 * @param[out]    conductivity_output Cell-wise conductivity field.
 * @param[out]    materials_id_output Cell-wise material ID field.
 * @param[out]    sources             Source set (optional, allocated if non-NULL).
 * @param[out]    receivers_output    Receiver vector (optional, allocated if non-NULL).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode loadCsemInputs(petgemParams      *pg_Params,
                              DM            *odm,
                              Vec           *conductivity_output,
                              Vec           *materials_id_output,
                              CsemSourceSet *sources,
                              Vec           *receivers_output) {
  PetscFunctionBeginUser;

  PetscCheck(pg_Params->inputFile[0] != '\0', PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "loadCsemInputs: pg_Params->inputFile is empty (set -input_filename).");

  /* Phase 1: read mesh + sections + combined model vector (parallel I/O). Replicates the legacy importGrid behavior so downstream code          
   * sees the same DMPlex / sub-DM / local-Vec configuration. */
  {
    PetscViewer  viewer;
    DM           dm, dmDist, subDMSigma, subDMMat;
    PetscSF      sfLoad, sfDist = NULL, sfXC = NULL, sfG;
    PetscSection combinedSection, secSigma, secMat, localSecSigma, localSecMat;
    Vec          combinedGlobal, combinedLocal, localSigma, localMat;
    char         typeName[PETSC_MAX_PATH_LEN];
    PetscBool    flg;
    PetscInt     pStart, pEnd, cellStart, cellEnd;

    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(PetscObjectSetName((PetscObject)dm, "petgem_mesh"));

    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, pg_Params->inputFile,
                                  FILE_MODE_READ, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_PETSC));

    PetscCall(DMPlexTopologyLoad(dm, viewer, &sfLoad));
    PetscCall(DMPlexLabelsLoad(dm, viewer, sfLoad));
    PetscCall(DMPlexCoordinatesLoad(dm, viewer, sfLoad));
    PetscCall(DMPlexDistribute(dm, 0, &sfDist, &dmDist));
    if (dmDist) {
      PetscCall(PetscSFCompose(sfLoad, sfDist, &sfXC));
      PetscCall(DMDestroy(&dm));
      dm = dmDist;
      PetscCall(PetscObjectSetName((PetscObject)dm, "petgem_mesh"));
    } else {
      PetscCall(PetscObjectReference((PetscObject)sfLoad));
      sfXC = sfLoad;
    }
    PetscCall(DMViewFromOptions(dm, NULL, "-load_dm_view"));

    /* 2-field section: field 0 = conductivity (3 dofs/cell), field 1 = materials_id (1 dof/cell).
     * DMSetNumFields is intentionally NOT called - for plain section-based DMs (no PetscFE/FV) it triggers DMCreateLocalSection_Plex which fails
     * without a discretization object.  The 2-field layout is recovered from the loaded section directly. */
    PetscCall(DMPlexSectionLoad(dm, viewer, NULL, sfXC, &sfG, NULL));

    /* Combined global vector - 4 dofs/cell [sigma_x sigma_y sigma_z mat_id] */
    PetscCall(DMCreateGlobalVector(dm, &combinedGlobal));
    PetscCall(PetscObjectSetName((PetscObject)combinedGlobal, "model_data"));
    PetscCall(DMPlexGlobalVectorLoad(dm, viewer, NULL, sfG, combinedGlobal));
    PetscCall(VecViewFromOptions(combinedGlobal, NULL, "-load_model_view"));

    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscSFDestroy(&sfLoad));
    PetscCall(PetscSFDestroy(&sfDist));
    PetscCall(PetscSFDestroy(&sfXC));
    PetscCall(PetscSFDestroy(&sfG));

    /* Global -> local so ghosts (needed for closure walks across partition boundaries) are populated on every rank. */
    PetscCall(DMCreateLocalVector(dm, &combinedLocal));
    PetscCall(DMGlobalToLocal(dm, combinedGlobal, INSERT_VALUES, combinedLocal));
    PetscCall(VecDestroy(&combinedGlobal));

    PetscCall(DMGetLocalSection(dm, &combinedSection));
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cellStart, &cellEnd));

    /* Conductivity sub-DM: clone dm, install a plain section with NUM_CONDUCTIVITY_COMPONENTS dofs/cell. */
    PetscCall(DMClone(dm, &subDMSigma));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &secSigma));
    PetscCall(PetscSectionSetChart(secSigma, pStart, pEnd));
    for (PetscInt i = cellStart; i < cellEnd; i++) {
      PetscCall(PetscSectionSetDof(secSigma, i, NUM_CONDUCTIVITY_COMPONENTS));
    }
    PetscCall(PetscSectionSetUp(secSigma));
    PetscCall(DMSetLocalSection(subDMSigma, secSigma));
    PetscCall(PetscSectionDestroy(&secSigma));

    /* Materials-ID sub-DM: 1 dof/cell. */
    PetscCall(DMClone(dm, &subDMMat));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &secMat));
    PetscCall(PetscSectionSetChart(secMat, pStart, pEnd));
    for (PetscInt i = cellStart; i < cellEnd; i++) {
      PetscCall(PetscSectionSetDof(secMat, i, NUM_MATERIALS_ID_COMPONENTS));
    }
    PetscCall(PetscSectionSetUp(secMat));
    PetscCall(DMSetLocalSection(subDMMat, secMat));
    PetscCall(PetscSectionDestroy(&secMat));

    PetscCall(DMCreateLocalVector(subDMSigma, &localSigma));
    PetscCall(DMCreateLocalVector(subDMMat,   &localMat));

    PetscCall(DMGetLocalSection(subDMSigma, &localSecSigma));
    PetscCall(DMGetLocalSection(subDMMat,   &localSecMat));

    /* Split the combined per-cell block [sigma_x sigma_y sigma_z mat_id] into the two sub-Vecs */
    {
      const PetscScalar *cArr;
      PetscScalar       *sigmaArr, *matArr;
      PetscCall(VecGetArrayRead(combinedLocal, &cArr));
      PetscCall(VecGetArray(localSigma, &sigmaArr));
      PetscCall(VecGetArray(localMat,   &matArr));
      for (PetscInt i = cellStart; i < cellEnd; i++) {
        PetscInt cOff, sigmaOff, matOff;
        PetscCall(PetscSectionGetOffset(combinedSection, i, &cOff));
        PetscCall(PetscSectionGetOffset(localSecSigma,   i, &sigmaOff));
        PetscCall(PetscSectionGetOffset(localSecMat,     i, &matOff));
        sigmaArr[sigmaOff + 0] = cArr[cOff + 0];
        sigmaArr[sigmaOff + 1] = cArr[cOff + 1];
        sigmaArr[sigmaOff + 2] = cArr[cOff + 2];
        matArr[matOff]         = cArr[cOff + 3];
      }
      PetscCall(VecRestoreArray(localMat,   &matArr));
      PetscCall(VecRestoreArray(localSigma, &sigmaArr));
      PetscCall(VecRestoreArrayRead(combinedLocal, &cArr));
    }
    PetscCall(VecDestroy(&combinedLocal));
    PetscCall(DMDestroy(&subDMSigma));
    PetscCall(DMDestroy(&subDMMat));

    /* Cloned DM is the public-facing handle; setupCsemGrid installs the H(curl) section on it without disturbing the original */
    PetscCall(DMClone(dm, odm));
    PetscCall(DMDestroy(&dm));

    *conductivity_output  = localSigma;
    *materials_id_output  = localMat;

    PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_vec_type", typeName, sizeof(typeName), &flg));
    if (flg) {
      PetscCall(DMSetVecType(*odm, typeName));
    }
    PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_mat_type", typeName, sizeof(typeName), &flg));
    if (flg) {
      PetscCall(DMSetMatType(*odm, typeName));
    }
  }

  /* Phase 2: /order, receivers, and sources (rank-local serial I/O) /order is always read so the caller does not need -order in the   
   * pg_Params file. CLI -order (if set) overrides the bundle value, but normally readPetgemParams leaves it at zero and the bundle wins */
  {
    PetscViewer viewer;
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, pg_Params->inputFile, FILE_MODE_READ, &viewer));

    /* /order - single-element Vec. Bundle is the source of truth unless the caller explicitly set -order on the command line 
    * (pg_Params->order non-zero at entry). */
    if (pg_Params->order <= 0) {
      Vec orderV;
      const PetscScalar *nArr;
      PetscCall(loadSelfVecByName(viewer, "order", &orderV));
      PetscCall(VecGetArrayRead(orderV, &nArr));
      pg_Params->order = (PetscInt)(PetscRealPart(nArr[0]) + 0.5);
      PetscCall(VecRestoreArrayRead(orderV, &nArr));
      PetscCall(VecDestroy(&orderV));
    }
    PetscCheck(pg_Params->order >= 1 && pg_Params->order <= 6, PETSC_COMM_WORLD,
               PETSC_ERR_ARG_OUTOFRANGE,
               "loadCsemInputs: order %" PetscInt_FMT " not in 1..6 (bundle %s)",
               pg_Params->order, pg_Params->inputFile);

    if (receivers_output) {
      Vec recv;
      PetscCall(loadSelfVecByName(viewer, "receivers", &recv));
      PetscCall(VecSetBlockSize(recv, NUM_DIMENSIONS));
      *receivers_output = recv;
    }

    if (sources) {
      Vec freqV, posV, curV, lenV, dipV, azV;
      PetscCall(PetscViewerHDF5PushGroup(viewer, "/sources"));
      /* Unified /sources schema: per-entry frequency (one row per transmitter). Forward modeling is monochromatic - all entries
       * share the same frequency - so we use freq[0] for the whole set */
      PetscCall(loadSelfVecByName(viewer, "freq",         &freqV));
      PetscCall(loadSelfVecByName(viewer, "position",     &posV));
      PetscCall(loadSelfVecByName(viewer, "current",      &curV));
      PetscCall(loadSelfVecByName(viewer, "length",       &lenV));
      PetscCall(loadSelfVecByName(viewer, "dipAngle",     &dipV));
      PetscCall(loadSelfVecByName(viewer, "azimuthAngle", &azV));
      PetscCall(PetscViewerHDF5PopGroup(viewer));

      const PetscScalar *fArr;
      PetscCall(VecGetArrayRead(freqV, &fArr));
      sources->freq = PetscRealPart(fArr[0]);
      PetscCall(VecRestoreArrayRead(freqV, &fArr));

      PetscInt n;
      PetscCall(VecGetSize(curV, &n));
      sources->numSources = n;
      PetscCall(PetscMalloc1(n, &sources->sourceArray));

      const PetscScalar *posArr, *curArr, *lenArr, *dipArr, *azArr;
      PetscCall(VecGetArrayRead(posV, &posArr));
      PetscCall(VecGetArrayRead(curV, &curArr));
      PetscCall(VecGetArrayRead(lenV, &lenArr));
      PetscCall(VecGetArrayRead(dipV, &dipArr));
      PetscCall(VecGetArrayRead(azV,  &azArr));
      for (PetscInt i = 0; i < n; i++) {
        sources->sourceArray[i].position[0] = PetscRealPart(posArr[i * 3 + 0]);
        sources->sourceArray[i].position[1] = PetscRealPart(posArr[i * 3 + 1]);
        sources->sourceArray[i].position[2] = PetscRealPart(posArr[i * 3 + 2]);
        sources->sourceArray[i].current      = PetscRealPart(curArr[i]);
        sources->sourceArray[i].length       = PetscRealPart(lenArr[i]);
        sources->sourceArray[i].dipAngle     = PetscRealPart(dipArr[i]);
        sources->sourceArray[i].azimuthAngle = PetscRealPart(azArr[i]);
      }
      PetscCall(VecRestoreArrayRead(posV, &posArr));
      PetscCall(VecRestoreArrayRead(curV, &curArr));
      PetscCall(VecRestoreArrayRead(lenV, &lenArr));
      PetscCall(VecRestoreArrayRead(dipV, &dipArr));
      PetscCall(VecRestoreArrayRead(azV,  &azArr));

      PetscCall(VecDestroy(&freqV));
      PetscCall(VecDestroy(&posV));
      PetscCall(VecDestroy(&curV));
      PetscCall(VecDestroy(&lenV));
      PetscCall(VecDestroy(&dipV));
      PetscCall(VecDestroy(&azV));

      PetscCall(logSection(PETSC_COMM_WORLD, "CSEM sources"));
      PetscCall(logKVReal(PETSC_COMM_WORLD, "Frequency (Hz)", sources->freq));
      PetscCall(logKVInt(PETSC_COMM_WORLD, "Number of sources", sources->numSources));
      for (PetscInt i = 0; i < sources->numSources; i++) {
        const CsemSource *s = &sources->sourceArray[i];
        /* Nested block: 5 + %-22s puts the '=' in the same column as the
         * top-level 3 + %-24s, so the whole report shares one alignment. */
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Source %" PetscInt_FMT ":\n", i + 1));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     %-22s = %s\n", "Current (A)",          formatReal(s->current)));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     %-22s = %s\n", "Length (m)",           formatReal(s->length)));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     %-22s = %s\n", "Dip angle (deg)",      formatReal(s->dipAngle)));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     %-22s = %s\n", "Azimuth angle (deg)",  formatReal(s->azimuthAngle)));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     %-22s = (%s, %s, %s)\n", "Position (m)",
                              formatReal(s->position[0]), formatReal(s->position[1]),
                              formatReal(s->position[2])));
      }
    }

    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Reads inversion parameters from the PETSc options database.
 *
 * This function initializes inversion-control parameters and applies
 * optional CLI overrides for optimization, regularization, diagnostics,
 * and debugging behavior.
 *
 * The FEM basis order (-order) is optional. The bundled HDF5 input is
 * the primary source of truth, and its value is later propagated into
 * the inversion parameters after loadCsemInputs. A CLI-provided -order
 * overrides the bundled value. A value of 0 means "not overridden".
 *
 * The routine also initializes defaults for L-BFGS settings, Tikhonov
 * regularization, RMS stopping criteria, snapshot generation, and
 * optional smoothing controls.
 *
 * Transmiter/frequency metadata is not loaded here; it is populated later
 * from the unified input bundle.
 *
 * @param[out] im_Params  Structure containing inversion parameters.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode readimParams(imParams *im_Params)
{
  PetscFunctionBeginUser;

  /* Inversion control defaults (read into the struct so the option block
   * below can honour them when the corresponding -im_* flag is absent). */
  im_Params->maxIter                = 80;
  im_Params->lbfgsMemory            = 5;
  im_Params->lambda                 = 0.1;
  im_Params->errorLevel             = 0.01;
  im_Params->gtol                   = 1e-5;
  im_Params->rmsTol                 = 0.0;       /* 0 => disabled */
  im_Params->diagGradientWeight     = 0.0;
  im_Params->numFixedMaterials      = IM_MAX_FIXED_MATERIALS;
  im_Params->fixedMaterialsFromCLI  = PETSC_FALSE;
  im_Params->snapshotInterval       = 0;
  im_Params->observedMode           = OBS_EXTERNAL;
  im_Params->observedFile[0]        = '\0';
  im_Params->rmsRelTol              = 1.0e-3;    /* <0.1% improvement/iter => plateau */
  im_Params->rmsStallWindow         = 3;

  /* Group every im.csem option under one PetscOptionsBegin/End block so
   * `-help` renders a structured "im.csem: inversion options" section,
   * printed after the shared "fm.csem: required/optional" groups that
   * readPetgemParams() registers. Note: -order is parsed (and range-validated)
   * once by readPetgemParams() into im_Params->common.order - the shared base both
   * kernels use - so it is intentionally NOT re-registered here. */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "im.csem: inversion options (optional)", "PETGEM");
  PetscCall(PetscOptionsInt("-im_max_iter", "Maximum L-BFGS iterations", "im.csem", im_Params->maxIter, &im_Params->maxIter, NULL));
  PetscCall(PetscOptionsInt("-im_lbfgs_memory", "L-BFGS memory size M", "im.csem", im_Params->lbfgsMemory, &im_Params->lbfgsMemory, NULL));
  PetscCall(PetscOptionsReal("-im_lambda", "Tikhonov regularisation weight", "im.csem", im_Params->lambda, &im_Params->lambda, NULL));
  PetscCall(PetscOptionsReal("-im_error_level", "Relative data-error level (CLI overrides bundle /observed@error_level)", "im.csem",
                               im_Params->errorLevel, &im_Params->errorLevel, &im_Params->errorLevelFromCLI));
  PetscCall(PetscOptionsReal("-im_gtol", "Gradient-norm convergence tolerance", "im.csem", im_Params->gtol, &im_Params->gtol, NULL));
  PetscCall(PetscOptionsReal("-im_rms_tol", "Absolute RMS early-stop threshold (0 = disabled)", "im.csem", im_Params->rmsTol, &im_Params->rmsTol, NULL));
  PetscCall(PetscOptionsReal("-im_rms_rtol", "Relative RMS-improvement threshold for the plateau early stop (0 = disabled)", "im.csem",
                               im_Params->rmsRelTol, &im_Params->rmsRelTol, NULL));
  PetscCall(PetscOptionsInt   ("-im_rms_stall_window", "Consecutive iterations below -im_rms_rtol required to declare an RMS plateau", "im.csem",
                               im_Params->rmsStallWindow, &im_Params->rmsStallWindow, NULL));
  PetscCall(PetscOptionsReal  ("-im_diag_weight", "Self-weight in the gradient smoother", "im.csem",
                               im_Params->diagGradientWeight, &im_Params->diagGradientWeight, NULL));
  PetscCall(PetscOptionsIntArray("-im_fixed_materials", "Comma-separated list of material IDs excluded from gradient smoothing (CLI overrides bundle /im_meta/fixed_materials)",
                                 "im.csem", im_Params->fixedMaterials, &im_Params->numFixedMaterials, &im_Params->fixedMaterialsFromCLI));
  PetscCall(PetscOptionsInt   ("-im_snapshot_interval", "Write VTU snapshot every N accepted L-BFGS steps (0 = disabled)", "im.csem",
                               im_Params->snapshotInterval, &im_Params->snapshotInterval, NULL));
  {
    /* Observed-data abstraction: source schema + file. */
    const char *obsModes[] = {"external", "fm_native"};
    PetscInt    obsModeIdx = (PetscInt)im_Params->observedMode;
    PetscCall(PetscOptionsEList("-im_observed_mode", "Observed-data source: 'external' (bundle /observed/Ex) or 'fm_native' (fm.csem /sources/src*/fields/Ex)", 
                                "im.csem", obsModes, 2, obsModes[obsModeIdx], &obsModeIdx, NULL));
    im_Params->observedMode = (ObservedDataMode)obsModeIdx;
    PetscCall(PetscOptionsString("-im_observed_file", "Observed-data file (default: the input bundle for 'external'; required for 'fm_native')", "im.csem",
                                 im_Params->observedFile, im_Params->observedFile, sizeof(im_Params->observedFile), NULL));
  }
  PetscOptionsEnd();

  /* A -help run has now printed the inversion option group; skip validation and the parameter summary so usage prints cleanly. The caller
   * (runInverse) exits right after parsing. */
  {
    PetscBool helpRequested = PETSC_FALSE;
    PetscCall(PetscOptionsHasHelp(NULL, &helpRequested));
    if (helpRequested) {
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }

  /* -order is validated by readPetgemParams() (im_Params->common.order); no re-check here. */
  if (!im_Params->fixedMaterialsFromCLI) im_Params->numFixedMaterials = 0;

  /* fm-native observed data lives in a forward responses file, not the bundle, so a path is mandatory in that mode. */
  PetscCheck(im_Params->observedMode != OBS_FM_NATIVE ||
             im_Params->observedFile[0] != '\0',
             PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "-im_observed_mode fm_native requires -im_observed_file (path to the fm.csem responses HDF5).");

  /* numFreqs/imSources are populated later by setupInversionSources (from the unified bundle's /sources group). */
  im_Params->numFreqs = 0;

  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscCall(logSection(comm, "Inversion parameters"));
  /* Basis order comes from the bundle (loaded after this block, so it is not yet known here); it is reported once in the shared "FEM space" block,
   * matching fm.csem. */
  PetscCall(logKVInt(comm, "Max iterations", im_Params->maxIter));
  PetscCall(logKVInt(comm, "L-BFGS memory (M)", im_Params->lbfgsMemory));
  PetscCall(logKVReal(comm, "Tikhonov lambda", im_Params->lambda));

  /* error_level and fixed_materials are printed by loadInversionMetaFromBundle after bundle resolution (so the line
   * reflects the final value used by the kernel, including bundle overrides). */
  if (im_Params->rmsTol > 0.0) {
    PetscCall(logKVReal(comm, "RMS early-stop", im_Params->rmsTol));
  }
  else {
    PetscCall(logKVStr(comm, "RMS early-stop", "disabled"));
  }
  if (im_Params->snapshotInterval > 0) {
    PetscCall(logKVf(comm, "VTU snapshot", "every %s accepted L-BFGS step(s)", formatGroupedInt(im_Params->snapshotInterval)));
  }
  else {
    PetscCall(logKVStr(comm, "VTU snapshot", "disabled"));
  }
  PetscCall(logKVStr(comm, "Observed data mode", im_Params->observedMode == OBS_FM_NATIVE ? "fm_native" : "external"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Loads inversion sources from the unified PETGEM HDF5 bundle.
 *
 * This function reads the /sources group from the input bundle and
 * constructs the inversion source list used by the optimization layer.
 * Each entry corresponds to one frequency–dipole configuration, with
 * shared structure between forward and inverse problems.
 *
 * The data is stored as PETSc Vecs, so loading is performed using
 * PetscViewerHDF5 + VecLoad to preserve correct scalar-type handling.
 *
 * The function populates:
 *   - im_Params->imSources[] (per-frequency dipole definitions)
 *   - im_Params->numFreqs     (number of loaded entries)
 *
 * A hard limit (IM_MAX_FREQUENCIES) is enforced to prevent oversized
 * inversion problems.
 *
 * @param[in]  bundleFile  Path to unified PETGEM HDF5 input file.
 * @param[out] im_Params    Inversion parameter structure to populate.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode setupInversionSources(const char *bundleFile,
                                     imParams  *im_Params)
{
  PetscFunctionBeginUser;

  /* Variable declarations */
  PetscViewer viewer;
  Vec freqV, posV, curV, lenV, dipV, azV;
  PetscInt count;
  const PetscScalar *freqArr, *posArr, *curArr, *lenArr, *dipArr, *azArr;

  /* Initial verifications */
  PetscCheck(bundleFile && bundleFile[0] != '\0', PETSC_COMM_WORLD,
             PETSC_ERR_ARG_NULL,
             "setupInversionSources: bundleFile is empty.");

  /* Read the unified /sources group (per-entry frequency), shared with the
   * forward kernel. These datasets are PETSc Vecs written by the preprocess,
   * so they carry the complex/real marking VecLoad needs - hence the PETSc
   * HDF5 viewer + loadSelfVecByName here (mirroring loadCsemInputs) */
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, bundleFile, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/sources"));

  /* Read parameters */
  PetscCall(loadSelfVecByName(viewer, "freq",         &freqV));
  PetscCall(loadSelfVecByName(viewer, "position",     &posV));
  PetscCall(loadSelfVecByName(viewer, "current",      &curV));
  PetscCall(loadSelfVecByName(viewer, "length",       &lenV));
  PetscCall(loadSelfVecByName(viewer, "dipAngle",     &dipV));
  PetscCall(loadSelfVecByName(viewer, "azimuthAngle", &azV));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(VecGetSize(freqV, &count));
  PetscCheck(count > 0, PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "/sources/freq is empty in %s", bundleFile);
  PetscCheck(count <= IM_MAX_FREQUENCIES, PETSC_COMM_WORLD, PETSC_ERR_SUP,
             "Too many inversion sources (%" PetscInt_FMT " > max %d); "
             "bump IM_MAX_FREQUENCIES in include/constants.h to raise the cap",
             count, IM_MAX_FREQUENCIES);

  PetscCall(VecGetArrayRead(freqV, &freqArr));
  PetscCall(VecGetArrayRead(posV,  &posArr));
  PetscCall(VecGetArrayRead(curV,  &curArr));
  PetscCall(VecGetArrayRead(lenV,  &lenArr));
  PetscCall(VecGetArrayRead(dipV,  &dipArr));
  PetscCall(VecGetArrayRead(azV,   &azArr));

  for (PetscInt i = 0; i < count; i++) {
    ImCsemSource *s = &im_Params->imSources[i];
    s->freq              = PetscRealPart(freqArr[i]);
    s->position[0]       = PetscRealPart(posArr[i * 3 + 0]);
    s->position[1]       = PetscRealPart(posArr[i * 3 + 1]);
    s->position[2]       = PetscRealPart(posArr[i * 3 + 2]);
    s->current           = PetscRealPart(curArr[i]);
    s->length            = PetscRealPart(lenArr[i]);
    s->dipAngle          = PetscRealPart(dipArr[i]);
    s->azimuthAngle      = PetscRealPart(azArr[i]);
  }

  PetscCall(VecRestoreArrayRead(freqV, &freqArr));
  PetscCall(VecRestoreArrayRead(posV,  &posArr));
  PetscCall(VecRestoreArrayRead(curV,  &curArr));
  PetscCall(VecRestoreArrayRead(lenV,  &lenArr));
  PetscCall(VecRestoreArrayRead(dipV,  &dipArr));
  PetscCall(VecRestoreArrayRead(azV,   &azArr));
  PetscCall(VecDestroy(&freqV));
  PetscCall(VecDestroy(&posV));
  PetscCall(VecDestroy(&curV));
  PetscCall(VecDestroy(&lenV));
  PetscCall(VecDestroy(&dipV));
  PetscCall(VecDestroy(&azV));

  im_Params->numFreqs = count;

  /* Print parsed source data */
  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscCall(logSection(comm, "Inversion sources"));
  PetscCall(logKVStr(comm, "Bundle file", bundleFile));
  PetscCall(logKVInt(comm, "Number of entries", im_Params->numFreqs));

  /* One row per transmitter rather than fm.csem's nested per-source block: with
   * one entry per frequency a table is far easier to scan than seven blocks.
   * Column headers carry the units so the rows stay numeric. */
  PetscCall(PetscPrintf(comm, "     %-5s %10s  %-28s %8s %8s %8s %8s\n",
                        "Entry", "Freq (Hz)", "Position (m)",
                        "I (A)", "L (m)", "Dip", "Azimuth"));
  for (PetscInt i = 0; i < im_Params->numFreqs; i++) {
    ImCsemSource *s = &im_Params->imSources[i];
    char pos[40];
    PetscCall(PetscSNPrintf(pos, sizeof(pos), "(%s, %s, %s)",
                            formatReal(s->position[0]), formatReal(s->position[1]),
                            formatReal(s->position[2])));
    PetscCall(PetscPrintf(comm,
      "     %5" PetscInt_FMT " %10s  %-28s %8s %8s %8s %8s\n",
      i + 1, formatReal(s->freq), pos,
      formatReal(s->current), formatReal(s->length),
      formatReal(s->dipAngle), formatReal(s->azimuthAngle)));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Loads inversion metadata overrides from the HDF5 bundle.
 *
 * This function reads optional inversion-related metadata from the
 * unified PETGEM input file and applies it to the inversion parameter
 * structure. It complements readimParams() by filling values
 * that are defined at the case level rather than the CLI.
 *
 * The following fields may be updated from the bundle:
 *   - /observed@error_level → im_Params->errorLevel
 *   - /im_meta/fixed_materials → fixed material IDs list
 *
 * CLI values always take precedence:
 *   - If errorLevelFromCLI is set, /observed is ignored.
 *   - If fixedMaterialsFromCLI is set, bundle values are ignored.
 *
 * Missing entries are silently skipped, leaving defaults unchanged.
 *
 * @param[in]  bundleFile  Path to PETGEM HDF5 bundle.
 * @param[in,out] im_Params Inversion parameter structure to update.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode loadInversionMetaFromBundle(const char *bundleFile,
                                            imParams  *im_Params)
{
  PetscFunctionBeginUser;

  PetscCheck(bundleFile && bundleFile[0] != '\0', PETSC_COMM_WORLD,
             PETSC_ERR_ARG_NULL,
             "loadInversionMetaFromBundle: bundleFile is empty.");

  MPI_Comm    comm            = PETSC_COMM_WORLD;
  const char *errLevelOrigin  = im_Params->errorLevelFromCLI  ? "CLI" : "default";
  const char *fixedMatsOrigin = im_Params->fixedMaterialsFromCLI ? "CLI" : "default";

  /* Open the bundle through the PETSc viewer.  PetscViewerHDF5HasAttribute /
   * HasObject internally silence the underlying H5E error stack while
   * probing for optional entries, so no manual H5Eset_auto2 dance is
   * needed and missing entries return cleanly via the `has` flag. */
  PetscViewer viewer;
  PetscCall(PetscViewerHDF5Open(comm, bundleFile, FILE_MODE_READ, &viewer));

  /* /observed @error_level (optional) */
  if (!im_Params->errorLevelFromCLI) {
    PetscBool hasAttr = PETSC_FALSE;
    PetscCall(PetscViewerHDF5HasAttribute(viewer, "/observed", "error_level",
                                          &hasAttr));
    if (hasAttr) {
      PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/observed", "error_level",
                                             PETSC_REAL, NULL,
                                             &im_Params->errorLevel));
      errLevelOrigin = "bundle";
    }
  }

  /* /im_meta/fixed_materials (optional, int32 array written by the
   * Python preprocess). PetscViewerHDF5HasDataset replaces the H5Lexists
   * probe; the read itself is kept on the raw H5D path because the
   * dataset is stored as int32 while PetscInt may be 64-bit under
   * --with-64-bit-indices, requiring an explicit per-element widen. */
  if (!im_Params->fixedMaterialsFromCLI) {
    PetscBool hasObj = PETSC_FALSE;
    PetscCall(PetscViewerHDF5HasDataset(viewer, "/im_meta/fixed_materials",
                                        &hasObj));
    if (hasObj) {
      hid_t file = -1;
      PetscCall(PetscViewerHDF5GetFileId(viewer, &file));
      hid_t dset = H5Dopen2(file, "/im_meta/fixed_materials", H5P_DEFAULT);
      PetscCheck(dset >= 0, comm, PETSC_ERR_FILE_READ, "/im_meta/fixed_materials present but H5Dopen2 failed");
      hid_t   sp = H5Dget_space(dset);
      hsize_t dims[1] = {0};
      H5Sget_simple_extent_dims(sp, dims, NULL);
      PetscInt n = (PetscInt)dims[0];
      PetscCheck(n <= IM_MAX_FIXED_MATERIALS, comm, PETSC_ERR_SUP,
                 "/im_meta/fixed_materials has %" PetscInt_FMT " entries; "
                 "IM_MAX_FIXED_MATERIALS=%d. Bump the cap in include/constants.h.",
                 n, IM_MAX_FIXED_MATERIALS);
      int *buf;
      PetscCall(PetscMalloc1(n, &buf));
      H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
      for (PetscInt i = 0; i < n; i++) {
        im_Params->fixedMaterials[i] = (PetscInt)buf[i];
      }
      im_Params->numFixedMaterials = n;
      PetscCall(PetscFree(buf));
      H5Sclose(sp);
      H5Dclose(dset);
      fixedMatsOrigin = "bundle";
    }
  }

  PetscCall(PetscViewerDestroy(&viewer));

  /* Final-value banner (matches the style of readimParams) */
  PetscCall(logKVf(comm, "Error level", "%s (from %s)", formatReal(im_Params->errorLevel), errLevelOrigin));

  /* Value first, provenance after: "0 1 (2 IDs, from bundle)" reads as a value
   * with a note, where the previous "2 IDs (bundle): 0 1" led with the count and
   * buried the IDs past a colon. */
  {
    char ids[192] = "";
    for (PetscInt i = 0; i < im_Params->numFixedMaterials; i++) {
      char one[24];
      PetscCall(PetscSNPrintf(one, sizeof(one), "%s%" PetscInt_FMT,
                              (i == 0) ? "" : " ", im_Params->fixedMaterials[i]));
      PetscCall(PetscStrlcat(ids, one, sizeof(ids)));
    }
    if (im_Params->numFixedMaterials == 0) PetscCall(PetscStrncpy(ids, "none", sizeof(ids)));
    PetscCall(logKVf(comm, "Fixed materials", "%s  (%" PetscInt_FMT " IDs, from %s)",
                     ids, im_Params->numFixedMaterials, fixedMatsOrigin));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Loads observed electromagnetic data from the PETGEM bundle.
 *
 * This function reads the /observed/Ex dataset from the unified HDF5
 * input file and constructs a dense observation matrix of shape
 * [numFreqs × numReceivers].
 *
 * The dataset is stored in HDF5 as a complex compound type
 * {real, imag} (complex128). Rank 0 performs the file I/O and then
 * broadcasts the data to all MPI ranks.
 *
 * The result is stored as a PETSc dense matrix (Mat) on
 * PETSC_COMM_SELF for local use on each process.
 *
 * @param[in]  bundleFile     Path to unified PETGEM HDF5 file.
 * @param[in]  numFreqs       Number of frequencies (rows).
 * @param[in]  numReceivers   Number of receivers (columns).
 * @param[out] dObs           Dense observation matrix.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode loadObservedData(const char *bundleFile,
                                PetscInt    numFreqs,
                                PetscInt    numReceivers,
                                Mat        *dObs)
{
  PetscFunctionBeginUser;

  MPI_Comm     comm = PETSC_COMM_WORLD;
  PetscMPIInt  rank;

  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCheck(bundleFile && bundleFile[0] != '\0', comm, PETSC_ERR_ARG_NULL, "loadObservedData: bundleFile is empty.");

  /* Create sequential dense matrix (all ranks) */
  PetscCall(MatCreateDense(PETSC_COMM_SELF, numFreqs, numReceivers, numFreqs, numReceivers, NULL, dObs));
  PetscCall(MatZeroEntries(*dObs));

  /* Only rank 0 reads the HDF5 file, then broadcasts */
  if (rank == 0) {
    hid_t file_id = H5Fopen(bundleFile, H5F_ACC_RDONLY, H5P_DEFAULT);
    PetscCheck(file_id >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open bundle HDF5 file: %s", bundleFile);

    /* Open /observed/Ex dataset (complex128 compound type {r, i}) */
    hid_t dset = H5Dopen2(file_id, "/observed/Ex", H5P_DEFAULT);
    PetscCheck(dset >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Cannot find /observed/Ex dataset in %s", bundleFile);

    /* Verify dimensions [numFreqs x numReceivers] */
    hid_t   space = H5Dget_space(dset);
    int     ndims = H5Sget_simple_extent_ndims(space);
    hsize_t dims[2];
    PetscCheck(ndims == 2, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "/observed/Ex must be 2D, got %d dimensions", ndims);
    H5Sget_simple_extent_dims(space, dims, NULL);
    PetscCheck((PetscInt)dims[0] == numFreqs &&
               (PetscInt)dims[1] == numReceivers,
               PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
               "/observed/Ex shape [%llu x %llu] != expected [%" PetscInt_FMT
               " x %" PetscInt_FMT "]",
               (unsigned long long)dims[0], (unsigned long long)dims[1],
               numFreqs, numReceivers);
    H5Sclose(space);

    /* Build HDF5 compound type matching PetscScalar (complex double).
     * h5py writes complex128 as {r: float64, i: float64}. */
    hid_t h5complex = H5Tcreate(H5T_COMPOUND, sizeof(PetscScalar));
    H5Tinsert(h5complex, "r", 0,              H5T_NATIVE_DOUBLE);
    H5Tinsert(h5complex, "i", sizeof(double),  H5T_NATIVE_DOUBLE);

    /* Read directly into PetscScalar buffer */
    PetscInt     totalElems = numFreqs * numReceivers;
    PetscScalar *buf;
    PetscCall(PetscMalloc1(totalElems, &buf));
    H5Dread(dset, h5complex, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);

    H5Tclose(h5complex);
    H5Dclose(dset);
    H5Fclose(file_id);

    /* Fill Mat from row-major buffer */
    for (PetscInt ifre = 0; ifre < numFreqs; ifre++) {
      for (PetscInt irec = 0; irec < numReceivers; irec++) {
        PetscCall(MatSetValue(*dObs, ifre, irec, buf[ifre * numReceivers + irec], INSERT_VALUES));
      }
    }
    PetscCall(PetscFree(buf));
  }

  PetscCall(MatAssemblyBegin(*dObs, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*dObs, MAT_FINAL_ASSEMBLY));

  /* Broadcast from rank 0 to all ranks.  Each rank owns its own
   * PETSC_COMM_SELF dense Mat; broadcast the raw array. */
  PetscInt     totalElems = numFreqs * numReceivers;
  PetscScalar *bcast;
  PetscCall(PetscMalloc1(totalElems, &bcast));

  if (rank == 0) {
    const PetscScalar *arr;
    PetscCall(MatDenseGetArrayRead(*dObs, &arr));
    for (PetscInt i = 0; i < totalElems; i++) { 
      bcast[i] = arr[i];
    }
    PetscCall(MatDenseRestoreArrayRead(*dObs, &arr));
  }
  PetscCallMPI(MPI_Bcast(bcast, totalElems, MPIU_SCALAR, 0, comm));

  if (rank != 0) {
    PetscScalar *arr;
    PetscCall(MatDenseGetArray(*dObs, &arr));
    for (PetscInt i = 0; i < totalElems; i++) {
      arr[i] = bcast[i];
    }
    PetscCall(MatDenseRestoreArray(*dObs, &arr));
    PetscCall(MatAssemblyBegin(*dObs, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*dObs, MAT_FINAL_ASSEMBLY));
  }
  PetscCall(PetscFree(bcast));

  PetscCall(logSection(comm, "Observed data"));
  PetscCall(logKVStr(comm, "Source", "external (/observed/Ex)"));
  PetscCall(logKVStr(comm, "File", bundleFile));
  PetscCall(logKVInt(comm, "Frequencies", numFreqs));
  PetscCall(logKVInt(comm, "Receivers", numReceivers));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief OBS_FM_NATIVE backend: read per-source Ex from an fm.csem file.
 *
 * Counterpart to loadObservedData() that consumes the forward kernel's native output schema (/sources/src{k}/fields/Ex PETSc Vecs) instead of
 * the pre-reshaped /observed/Ex dataset. Source k (1-based) maps to frequency row k-1 of the returned dense Mat. Every rank opens the COMM_SELF 
 * viewer and VecLoads each row independently, so the resulting PETSC_COMM_SELF Mat is identical on all ranks (mirroring loadObservedData's 
 * post-broadcast state).
 *
 * See the inversion.h prototype for parameter semantics.
 */
PetscErrorCode loadObservedFmNative(const char *responsesFile,
                                    PetscInt    numFreqs,
                                    PetscInt    numReceivers,
                                    Mat        *dObs)
{
  PetscFunctionBeginUser;

  MPI_Comm comm = PETSC_COMM_WORLD;

  PetscCheck(responsesFile && responsesFile[0] != '\0', comm, PETSC_ERR_ARG_NULL, "loadObservedFmNative: responsesFile is empty.");

  /* Sequential dense Mat on every rank (same layout the external path yields). */
  PetscCall(MatCreateDense(PETSC_COMM_SELF, numFreqs, numReceivers, numFreqs, numReceivers, NULL, dObs));
  PetscCall(MatZeroEntries(*dObs));

  PetscViewer viewer;
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, responsesFile, FILE_MODE_READ, &viewer));

  for (PetscInt ifre = 0; ifre < numFreqs; ifre++) {
    char groupPath[PETSC_MAX_PATH_LEN];
    PetscCall(PetscSNPrintf(groupPath, sizeof(groupPath), "/sources/src%" PetscInt_FMT "/fields", ifre + 1));
    PetscCall(PetscViewerHDF5PushGroup(viewer, groupPath));

    Vec exV;
    PetscCall(loadSelfVecByName(viewer, "Ex", &exV));
    PetscCall(PetscViewerHDF5PopGroup(viewer));

    PetscInt n;
    PetscCall(VecGetSize(exV, &n));
    PetscCheck(n == numReceivers, comm, PETSC_ERR_FILE_READ,
               "%s: %s/Ex length %" PetscInt_FMT " != expected receivers %"
               PetscInt_FMT, responsesFile, groupPath, n, numReceivers);

    const PetscScalar *exArr;
    PetscCall(VecGetArrayRead(exV, &exArr));
    for (PetscInt irec = 0; irec < numReceivers; irec++) {
      PetscCall(MatSetValue(*dObs, ifre, irec, exArr[irec], INSERT_VALUES));
    }
    PetscCall(VecRestoreArrayRead(exV, &exArr));
    PetscCall(VecDestroy(&exV));
  }

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatAssemblyBegin(*dObs, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*dObs, MAT_FINAL_ASSEMBLY));

  PetscCall(logSection(comm, "Observed data"));
  PetscCall(logKVStr(comm, "Source", "fm-native (/sources/src*/fields/Ex)"));
  PetscCall(logKVStr(comm, "File", responsesFile));
  PetscCall(logKVInt(comm, "Frequencies", numFreqs));
  PetscCall(logKVInt(comm, "Receivers", numReceivers));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Observed-data abstraction entry point (see inversion.h).
 *
 * Resolves the file path (iparams->observedFile, or the unified bundle
 * iparams->common.inputFile when empty) and dispatches to the backend selected by
 * iparams->observedMode. Keeps the inverse driver agnostic to data origin.
 */
PetscErrorCode loadObservedDataset(const imParams *iparams,
                                   PetscInt        numReceivers,
                                   Mat            *dObs)
{
  PetscFunctionBeginUser;

  const char *file = (iparams->observedFile[0] != '\0')
                       ? iparams->observedFile
                       : iparams->common.inputFile;

  if (iparams->observedMode == OBS_FM_NATIVE) {
    PetscCall(loadObservedFmNative(file, iparams->numFreqs, numReceivers, dObs));
  } else {
    PetscCall(loadObservedData(file, iparams->numFreqs, numReceivers, dObs));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Writes final inversion results to an HDF5 file.
 *
 * This function exports the inverted conductivity model and the
 * log-perturbation field to an HDF5 file, along with metadata
 * describing the inversion run (convergence, regularization,
 * and configuration parameters).
 *
 * The output file is:
 *   {output_dir}/{output_filename}.h5
 *
 * The following datasets are written:
 *   - conductivity      : global conductivity model (3 components/cell)
 *   - log_perturbation  : inverted model parameter vector
 *   - rms_history       : RMS misfit per objective-gradient evaluation (optional)
 *
 * Additional attributes store provenance and inversion metadata, including
 * PETGEM version, the stopping reason, and BOTH iteration counters: attribute
 * num_iterations is the accepted-L-BFGS-step count, num_objgrad_evaluations is
 * the number of objective-gradient evaluations, which is what rms_history is
 * indexed by. They differ by the rejected line-search trials, so consumers must
 * not use one where the other is meant.
 *
 * @param[in] im_Params        Inversion parameters and metadata.
 * @param[in] dmConductivity   DM describing conductivity field.
 * @param[in] conductivity     Local conductivity vector.
 * @param[in] X                Log-perturbation model vector.
 * @param[in] allRMS           RMS history, one entry per evaluation.
 * @param[in] numIters         Accepted L-BFGS steps.
 * @param[in] numEvals         Objective-gradient evaluations (length of allRMS).
 * @param[in] reasonStr        Convergence/stopping reason string.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode writeInversionResults(const imParams *im_Params,
                                      DM               dmConductivity,
                                      Vec              conductivity,
                                      Vec              X,
                                      const PetscReal *allRMS,
                                      PetscInt         numIters,
                                      PetscInt         numEvals,
                                      const char      *reasonStr)
{
  PetscFunctionBeginUser;

  MPI_Comm comm = PetscObjectComm((PetscObject)dmConductivity);

  /* Output path: composed by the shared builder from the parameters already parsed (and validated as required) by readPetgemParams. Re-reading
   * -output_dir / -output_filename from the options database here would duplicate that parse and its required-checks. */
  char outFile[PETSC_MAX_PATH_LEN];
  PetscCall(buildOutputPath(&im_Params->common, ".h5", outFile, sizeof(outFile)));

  PetscCall(logSection(comm, "Inversion output"));
  PetscCall(logKVStr(comm, "Output file", outFile));

  /* Scatter local conductivity to global for output */
  Vec globalConductivity;
  PetscCall(DMCreateGlobalVector(dmConductivity, &globalConductivity));
  PetscCall(DMLocalToGlobal(dmConductivity, conductivity, INSERT_VALUES, globalConductivity));

  /* Create plain MPI Vec copies (no DM association) for HDF5 output. The dmConductivity section has numComp=3 but 4 vertex/cell, so
   * VecView_Plex_HDF5 would fail with a block-size mismatch. Writing through plain Vecs avoids that. */
  PetscInt localSize, globalSize;
  Vec outRes, outX;

  PetscCall(VecGetLocalSize(globalConductivity, &localSize));
  PetscCall(VecGetSize(globalConductivity, &globalSize));
  PetscCall(VecCreateMPI(comm, localSize, globalSize, &outRes));
  PetscCall(VecCopy(globalConductivity, outRes));
  PetscCall(PetscObjectSetName((PetscObject)outRes, "conductivity"));

  PetscCall(VecGetLocalSize(X, &localSize));
  PetscCall(VecGetSize(X, &globalSize));
  PetscCall(VecCreateMPI(comm, localSize, globalSize, &outX));
  PetscCall(VecCopy(X, outX));
  PetscCall(PetscObjectSetName((PetscObject)outX, "log_perturbation"));

  /* Create HDF5 viewer and write vectors */
  PetscViewer viewer;
  PetscCall(PetscViewerHDF5Open(comm, outFile, FILE_MODE_WRITE, &viewer));

  PetscCall(VecView(outRes, viewer));
  PetscCall(VecView(outX, viewer));

  /* Root provenance attributes. The common block (version, simulation type, input bundle, order, solver, tasks, date) is the SAME one fm.csem writes,
   * emitted by the shared writer under identical names; only the inversion-specific attributes below are added here. */
  PetscCall(writeRunProvenance(viewer, &im_Params->common, PETGEM_SIM_IM));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "num_frequencies",    PETSC_INT,    &im_Params->numFreqs));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "lambda",             PETSC_REAL,   &im_Params->lambda));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "error_level",        PETSC_REAL,   &im_Params->errorLevel));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "num_iterations",     PETSC_INT,    &numIters));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "num_objgrad_evaluations", PETSC_INT, &numEvals));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "convergence_reason", PETSC_STRING, reasonStr));

  /* Write RMS history as an MPI Vec so all ranks participate in the collective VecView.  Only rank 0 owns the data (size numEvals -
   * allRMS carries one entry per objective-gradient evaluation, NOT per accepted step); all other ranks contribute zero entries */
  if (numEvals > 0) {
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscInt rmsLocalSize = (rank == 0) ? numEvals : 0;

    Vec rmsVec;
    PetscCall(VecCreateMPI(comm, rmsLocalSize, numEvals, &rmsVec));
    PetscCall(PetscObjectSetName((PetscObject)rmsVec, "rms_history"));

    if (rank == 0) {
      PetscScalar *rArr;
      PetscCall(VecGetArray(rmsVec, &rArr));
      for (PetscInt i = 0; i < numEvals; i++) {
        rArr[i] = allRMS[i];
      }
      PetscCall(VecRestoreArray(rmsVec, &rArr));
    }

    PetscCall(VecView(rmsVec, viewer));
    PetscCall(VecDestroy(&rmsVec));
  }

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&outRes));
  PetscCall(VecDestroy(&outX));
  PetscCall(VecDestroy(&globalConductivity));

  PetscCall(logKVStr(comm, "Datasets written", "conductivity, log_perturbation, rms_history"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Writes a parallel VTU snapshot of the inversion state.
 *
 * This function exports the current inversion iterate as a ParaView
 * compatible parallel VTK dataset (.pvtu + per-rank .vtu pieces).
 * It is typically called every accepted L-BFGS step when snapshotting
 * is enabled.
 *
 * Each MPI rank writes only its locally owned cells (no global gather).
 * Cell data is written in exploded tetrahedral form to avoid global
 * point renumbering.
 *
 * The snapshot carries a single cell-centered scalar field:
 *   - rho            : 1 / sigma_x (conductivity-derived resistivity, Ohm.m)
 *
 * Output structure, inside a snapshot directory under {output_dir}:
 *   - snapshots/iterNNNN.pvtu       (master, rank 0)
 *   - snapshots/iterNNNN_rRRRR.vtu  (one per rank)
 *
 * The .pvtu file aggregates all per-rank pieces into a single dataset
 * for visualization in ParaView, referencing them relatively.
 *
 * @param[in] ctx            Inversion execution context (DM, fields, grid).
 * @param[in] acceptedIter   Current accepted optimization iteration.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode writeInversionSnapshotVTU(const InversionContext *ctx, PetscInt acceptedIter)
{
  PetscFunctionBeginUser;

  MPI_Comm comm   = PetscObjectComm((PetscObject)ctx->dm);
  PetscInt nOwned = ctx->grid.cellEnd - ctx->grid.cellStart;

  PetscMPIInt rank, size;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  /* Snapshots go into their own directory:
   *
   *   {output_dir}/snapshots/iterNNNN.pvtu       (master, rank 0)
   *   {output_dir}/snapshots/iterNNNN_rRRRR.vtu  (one per rank)
   *
   * At -im_snapshot_interval 1 a run emits (accepted steps x ranks) pieces - a
   * 103-step, 112-rank inversion is ~11 600 files - so keeping them out of
   * {output_dir} is what makes the results visible at all. The directory
   * carries the grouping, leaving the file names to carry only what varies:
   * iteration and rank.
   *
   * NOTE: the directory name is fixed rather than keyed on the output stem, so
   * two runs writing to the SAME -output_dir share it and the later one
   * overwrites matching iterations. Give concurrent runs distinct -output_dir.
   *
   * The parameters were already parsed by readPetgemParams; no re-read of the
   * options database here. */
  const petgemParams *pg = &ctx->iparams->common;
  char snapDir[PETSC_MAX_PATH_LEN];
  char pvtuFile[PETSC_MAX_PATH_LEN], pieceFile[PETSC_MAX_PATH_LEN];

  PetscCall(PetscStrncpy(snapDir, pg->outputDirectory, sizeof(snapDir)));
  {
    size_t len = strlen(snapDir);
    if (len > 0 && snapDir[len - 1] != '/') PetscCall(PetscStrlcat(snapDir, "/", sizeof(snapDir)));
  }
  PetscCall(PetscStrlcat(snapDir, "snapshots", sizeof(snapDir)));
  if (rank == 0) {
    PetscBool exists = PETSC_FALSE;
    PetscCall(PetscTestDirectory(snapDir, 'w', &exists));
    if (!exists) PetscCall(PetscMkdir(snapDir));
  }
  /* Every rank writes its own piece into snapDir, so none may race ahead of the
   * rank-0 mkdir. */
  PetscCallMPI(MPI_Barrier(comm));

  PetscCall(PetscSNPrintf(pvtuFile, sizeof(pvtuFile),
                          "%s/iter%04" PetscInt_FMT ".pvtu", snapDir, acceptedIter));
  PetscCall(PetscSNPrintf(pieceFile, sizeof(pieceFile),
                          "%s/iter%04" PetscInt_FMT "_r%04d.vtu", snapDir, acceptedIter, rank));

  /* Per-owned-cell field values + exploded vertex coordinates.
   * The VTU snapshot carries only the recovered model rho = 1/sigma */
  PetscReal *rho, *coords;
  PetscCall(PetscCalloc1(nOwned, &rho));
  PetscCall(PetscCalloc1(12 * nOwned, &coords));

  /* rho = 1/sigma_x (component 0 of the conductivity Vec) */
  {
    PetscSection resSec;
    PetscCall(DMGetLocalSection(ctx->dmConductivity, &resSec));
    const PetscScalar *sArr;
    PetscCall(VecGetArrayRead(ctx->conductivity, &sArr));
    for (PetscInt i = ctx->grid.cellStart; i < ctx->grid.cellEnd; i++) {
      PetscInt li = i - ctx->grid.cellStart, off;
      PetscCall(PetscSectionGetOffset(resSec, i, &off));
      PetscReal sigma = PetscRealPart(sArr[off]);
      rho[li] = (sigma > 0.0) ? 1.0 / sigma : 0.0;
    }
    PetscCall(VecRestoreArrayRead(ctx->conductivity, &sArr));
  }

  /* Exploded tet vertices: 4 vertices (12 reals) per owned cell. */
  for (PetscInt i = ctx->grid.cellStart; i < ctx->grid.cellEnd; i++) {
    PetscInt li = i - ctx->grid.cellStart;
    Cell cell;
    PetscCall(extractCellCoordinates(ctx->dm, i, &cell));
    for (PetscInt j = 0; j < 12; j++) coords[12 * li + j] = cell.coordinates[j];
  }

  /* Each rank writes its own .vtu piece */
  {
    const PetscReal *const fields[IM_VTU_NUM_FIELDS] = { rho };
    PetscCall(writeVtuPiece(pieceFile, nOwned, coords, fields));
  }

  /* Rank 0 writes the .pvtu master after all pieces exist. Holds only field/topology declarations + one <Piece Source=.../> per
     rank (O(numRanks) text, no field data). The barrier guarantees every piece file is on disk before the master that references them */
  PetscCallMPI(MPI_Barrier(comm));
  if (rank == 0) {
    FILE *fp = fopen(pvtuFile, "w");
    PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "writeInversionSnapshotVTU: cannot open %s", pvtuFile);
    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <PUnstructuredGrid GhostLevel=\"0\">\n");
    fprintf(fp, "    <PPoints>\n");
    fprintf(fp, "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>\n");
    fprintf(fp, "    </PPoints>\n");
    fprintf(fp, "    <PCells>\n");
    fprintf(fp, "      <PDataArray type=\"Int64\" Name=\"connectivity\"/>\n");
    fprintf(fp, "      <PDataArray type=\"Int64\" Name=\"offsets\"/>\n");
    fprintf(fp, "      <PDataArray type=\"UInt8\" Name=\"types\"/>\n");
    fprintf(fp, "    </PCells>\n");
    fprintf(fp, "    <PCellData Scalars=\"%s\">\n", imVtuFieldNames[0]);
    for (PetscInt f = 0; f < IM_VTU_NUM_FIELDS; f++) {
      fprintf(fp, "      <PDataArray type=\"Float64\" Name=\"%s\"/>\n", imVtuFieldNames[f]);
    }
    fprintf(fp, "    </PCellData>\n");
    /* Piece names must match the ones each rank writes above - same pattern - or
     * ParaView cannot resolve the pieces. They are plain base names: the .pvtu
     * and its pieces share the snapshot directory, so the reference is relative. */
    for (PetscMPIInt r = 0; r < size; r++) {
      char rBase[PETSC_MAX_PATH_LEN];
      PetscCall(PetscSNPrintf(rBase, sizeof(rBase),
        "iter%04" PetscInt_FMT "_r%04d.vtu", acceptedIter, r));
      fprintf(fp, "    <Piece Source=\"%s\"/>\n", rBase);
    }
    fprintf(fp, "  </PUnstructuredGrid>\n</VTKFile>\n");
    fclose(fp);
  }

  PetscCall(PetscFree(rho));
  PetscCall(PetscFree(coords));

  PetscFunctionReturn(PETSC_SUCCESS);
}
