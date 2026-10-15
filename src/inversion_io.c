/*
 * Filename: inversion_io.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * Input and output of the CSEM inversion: options, sources, observed data,
 * results and VTU snapshots.
 */

/* C libraries */
#include <stdio.h>
#include <string.h>

/* PETSc libraries */
#include <petscviewerhdf5.h>
#include <hdf5.h>

/* PETGEM functions */
#include "common.h"
#include "constants.h"
#include "csem.h"
#include "grid.h"
#include "inversion.h"
#include "inversion_internal.h"
#include "io.h"

const char *const imDataComponentNames[IM_NUM_DATA_COMPONENTS] = {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Zxx", "Zxy", "Zyx", "Zyy"};

void imDataComponentFields(ImDataComponent comp, ReceiverField *numerator, ReceiverField *denominator) {
  /* Zxx, Zxy, Zyx, Zyy = E{x,x,y,y} / H{x,y,x,y} */
  static const ReceiverField zNum[4] = {RECV_EX, RECV_EX, RECV_EY, RECV_EY};
  static const ReceiverField zDen[4] = {RECV_HX, RECV_HY, RECV_HX, RECV_HY};
  if (comp < IM_DATA_ZXX) {
    *numerator   = (ReceiverField)comp;
    *denominator = NUM_RECEIVER_FIELDS;
  } else {
    *numerator   = zNum[comp - IM_DATA_ZXX];
    *denominator = zDen[comp - IM_DATA_ZXX];
  }
}

/* Resolves the -im_data_components names into imParams->components (case-insensitive, no repeats) and frees them. */
static PetscErrorCode setDataComponents(char *names[], PetscInt n, ImParams *imParams) {
  PetscFunctionBeginUser;
  PetscCheck(n > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "-im_data_components needs at least one component");
  for (PetscInt i = 0; i < n; i++) {
    PetscInt found = -1;
    for (PetscInt c = 0; c < IM_NUM_DATA_COMPONENTS && found < 0; c++) {
      PetscBool eq;
      PetscCall(PetscStrcasecmp(names[i], imDataComponentNames[c], &eq));
      if (eq) found = c;
    }
    PetscCheck(found >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "-im_data_components: unknown component '%s' (Ex Ey Ez Hx Hy Hz Zxx Zxy Zyx Zyy)", names[i]);
    for (PetscInt j = 0; j < i; j++)
      PetscCheck(imParams->components[j] != (ImDataComponent)found, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
                 "-im_data_components: '%s' repeated", names[i]);
    imParams->components[i] = (ImDataComponent)found;
    PetscCall(PetscFree(names[i]));
  }
  imParams->numComponents = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Writes this rank's VTU piece: exploded tetrahedra with the cell field rho.
 *
 * Every cell carries its own 4 vertices, so no global point numbering is
 * needed. Ranks without cells write a valid empty piece.
 *
 * @param[in] filename  Output .vtu path.
 * @param[in] nCells    Number of owned cells.
 * @param[in] coords    Vertex coordinates, 12 reals per cell.
 * @param[in] rho       Resistivity per cell.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
static PetscErrorCode writeVtuPiece(const char *filename, PetscInt nCells, const PetscReal *coords, const PetscReal *rho) {
  PetscFunctionBeginUser;
  FILE          *fp   = fopen(filename, "w");
  const PetscInt npts = 4 * nCells;

  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "writeVtuPiece: cannot open %s", filename);
  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
  fprintf(fp, "  <UnstructuredGrid>\n");
  fprintf(fp, "    <Piece NumberOfPoints=\"%" PetscInt_FMT "\" NumberOfCells=\"%" PetscInt_FMT "\">\n", npts, nCells);
  fprintf(fp, "      <Points>\n");
  fprintf(fp, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (PetscInt p = 0; p < npts; p++) {
    fprintf(fp, "%.9g %.9g %.9g\n", (double)coords[3 * p], (double)coords[3 * p + 1], (double)coords[3 * p + 2]);
  }
  fprintf(fp, "        </DataArray>\n      </Points>\n");
  fprintf(fp, "      <Cells>\n");
  fprintf(fp, "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
  for (PetscInt p = 0; p < npts; p++) fprintf(fp, "%" PetscInt_FMT " ", p);
  fprintf(fp, "\n        </DataArray>\n");
  fprintf(fp, "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
  for (PetscInt c = 1; c <= nCells; c++) fprintf(fp, "%" PetscInt_FMT " ", 4 * c);
  fprintf(fp, "\n        </DataArray>\n");
  fprintf(fp, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (PetscInt c = 0; c < nCells; c++) fprintf(fp, "10 "); /* VTK_TETRA */
  fprintf(fp, "\n        </DataArray>\n      </Cells>\n");
  fprintf(fp, "      <CellData Scalars=\"rho\">\n");
  fprintf(fp, "        <DataArray type=\"Float64\" Name=\"rho\" format=\"ascii\">\n");
  for (PetscInt c = 0; c < nCells; c++) fprintf(fp, "%.9g\n", (double)rho[c]);
  fprintf(fp, "        </DataArray>\n");
  fprintf(fp, "      </CellData>\n");
  fprintf(fp, "    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n");
  fclose(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Name of the VTU piece of one rank, relative to the snapshot directory. */
static PetscErrorCode vtuPieceName(PetscInt iter, PetscMPIInt rank, char *name, size_t len) {
  PetscFunctionBeginUser;
  PetscCall(PetscSNPrintf(name, len, "iter%04" PetscInt_FMT "_r%04d.vtu", iter, rank));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Master .pvtu that references the pieces of all ranks. */
static PetscErrorCode writePvtuMaster(const char *filename, PetscInt iter, PetscMPIInt size) {
  PetscFunctionBeginUser;
  FILE *fp = fopen(filename, "w");
  char  piece[PETSC_MAX_PATH_LEN];

  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "writePvtuMaster: cannot open %s", filename);
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
  fprintf(fp, "    <PCellData Scalars=\"rho\">\n");
  fprintf(fp, "      <PDataArray type=\"Float64\" Name=\"rho\"/>\n");
  fprintf(fp, "    </PCellData>\n");
  for (PetscMPIInt rank = 0; rank < size; rank++) {
    PetscCall(vtuPieceName(iter, rank, piece, sizeof(piece)));
    fprintf(fp, "    <Piece Source=\"%s\"/>\n", piece);
  }
  fprintf(fp, "  </PUnstructuredGrid>\n</VTKFile>\n");
  fclose(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Copies @p src into a new named MPI Vec with no DM attached.
 *
 * @param[in]  src   Vec to copy.
 * @param[in]  name  Name of the copy (HDF5 dataset name).
 * @param[out] out   The copy.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
static PetscErrorCode createPlainCopy(Vec src, const char *name, Vec *out) {
  PetscFunctionBeginUser;
  PetscInt localSize, globalSize;
  PetscCall(VecGetLocalSize(src, &localSize));
  PetscCall(VecGetSize(src, &globalSize));
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)src), localSize, globalSize, out));
  PetscCall(VecCopy(src, *out));
  PetscCall(PetscObjectSetName((PetscObject)*out, name));
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
 * the inversion parameters after loadModelInputs. A CLI-provided -order
 * overrides the bundled value. A value of 0 means "not overridden".
 *
 * The routine also initializes defaults for L-BFGS settings, Tikhonov
 * regularization, RMS stopping criteria, snapshot generation, and
 * optional smoothing controls.
 *
 * Transmiter/frequency metadata is not loaded here; it is populated later
 * from the unified input bundle.
 */
PetscErrorCode readImParams(ImParams *imParams) {
  PetscFunctionBeginUser;

  /* Inversion control defaults (read into the struct so the option block
   * below can honor them when the corresponding -im_* flag is absent). */
  imParams->maxIter               = 80;
  imParams->lbfgsMemory           = 5;
  imParams->lambda                = 0.1;
  imParams->errorLevel            = 0.01;
  imParams->gtol                  = 1e-5;
  imParams->rmsTol                = 0.0; /* 0 => disabled */
  imParams->diagGradientWeight    = 0.0;
  imParams->numFixedMaterials     = IM_MAX_FIXED_MATERIALS;
  imParams->fixedMaterialsFromCli = PETSC_FALSE;
  imParams->snapshotInterval      = 0;
  imParams->observedMode          = OBS_EXTERNAL;
  imParams->observedFile[0]       = '\0';
  imParams->rmsRelTol             = 1.0e-3; /* <0.1% improvement/iter => plateau */
  imParams->rmsStallWindow        = 3;
  imParams->gradientCheck         = PETSC_FALSE;
  imParams->numCheckSteps         = 4;
  imParams->checkSteps[0]         = 1e-1;
  imParams->checkSteps[1]         = 1e-2;
  imParams->checkSteps[2]         = 1e-3;
  imParams->checkSteps[3]         = 1e-4;
  imParams->numComponents         = 1;
  imParams->components[0]         = IM_DATA_EX;

  /* -order is parsed by readPetgemParams() into imParams->common.order. */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "im.csem: inversion options (optional)", "PETGEM");
  PetscCall(PetscOptionsInt("-im_max_iter", "Maximum L-BFGS iterations", "im.csem", imParams->maxIter, &imParams->maxIter, NULL));
  PetscCall(PetscOptionsInt("-im_lbfgs_memory", "L-BFGS memory size M", "im.csem", imParams->lbfgsMemory, &imParams->lbfgsMemory, NULL));
  PetscCall(PetscOptionsReal("-im_lambda", "Tikhonov regularization weight", "im.csem", imParams->lambda, &imParams->lambda, NULL));
  PetscCall(PetscOptionsReal("-im_error_level", "Relative data-error level (CLI overrides bundle /observed@error_level)", "im.csem",
                             imParams->errorLevel, &imParams->errorLevel, &imParams->errorLevelFromCli));
  PetscCall(PetscOptionsReal("-im_gtol", "Gradient-norm convergence tolerance", "im.csem", imParams->gtol, &imParams->gtol, NULL));
  PetscCall(PetscOptionsReal("-im_rms_tol", "Absolute RMS early-stop threshold (0 = disabled)", "im.csem", imParams->rmsTol,
                             &imParams->rmsTol, NULL));
  PetscCall(PetscOptionsReal("-im_rms_rtol", "Relative RMS-improvement threshold for the plateau early stop (0 = disabled)", "im.csem",
                             imParams->rmsRelTol, &imParams->rmsRelTol, NULL));
  PetscCall(PetscOptionsInt("-im_rms_stall_window", "Consecutive iterations below -im_rms_rtol required to declare an RMS plateau",
                            "im.csem", imParams->rmsStallWindow, &imParams->rmsStallWindow, NULL));
  PetscCall(PetscOptionsReal("-im_diag_weight", "Self-weight in the gradient smoother", "im.csem", imParams->diagGradientWeight,
                             &imParams->diagGradientWeight, NULL));
  PetscCall(PetscOptionsIntArray(
      "-im_fixed_materials",
      "Comma-separated list of material IDs excluded from gradient smoothing (CLI overrides bundle /im_meta/fixed_materials)", "im.csem",
      imParams->fixedMaterials, &imParams->numFixedMaterials, &imParams->fixedMaterialsFromCli));
  PetscCall(PetscOptionsInt("-im_snapshot_interval", "Write VTU snapshot every N accepted L-BFGS steps, plus the last one (0 = disabled)",
                            "im.csem", imParams->snapshotInterval, &imParams->snapshotInterval, NULL));
  {
    /* Observed-data abstraction: source schema + file. */
    const char *obsModes[] = {"external", "fm_native"};
    PetscInt    obsModeIdx = (PetscInt)imParams->observedMode;
    PetscCall(
        PetscOptionsEList("-im_observed_mode",
                          "Observed-data source: 'external' (bundle /observed/<component>) or 'fm_native' (fm.csem /sources/src*/fields)",
                          "im.csem", obsModes, 2, obsModes[obsModeIdx], &obsModeIdx, NULL));
    imParams->observedMode = (ObservedDataMode)obsModeIdx;
    PetscCall(PetscOptionsString("-im_observed_file",
                                 "Observed-data file (default: the input bundle for 'external'; required for 'fm_native')", "im.csem",
                                 imParams->observedFile, imParams->observedFile, sizeof(imParams->observedFile), NULL));
  }
  {
    char     *names[IM_NUM_DATA_COMPONENTS];
    PetscInt  n   = IM_NUM_DATA_COMPONENTS;
    PetscBool set = PETSC_FALSE;
    PetscCall(PetscOptionsStringArray("-im_data_components", "Fitted data components: Ex Ey Ez Hx Hy Hz Zxx Zxy Zyx Zyy (comma-separated)",
                                      "im.csem", names, &n, &set));
    if (set) PetscCall(setDataComponents(names, n, imParams));
  }
  PetscCall(PetscOptionsBool("-im_gradient_check", "Compare the adjoint gradient with central differences at the starting model and stop",
                             "im.csem", imParams->gradientCheck, &imParams->gradientCheck, NULL));
  {
    PetscInt  n   = IM_MAX_GRADIENT_CHECK_STEPS;
    PetscBool set = PETSC_FALSE;
    PetscCall(PetscOptionsRealArray("-im_gradient_check_eps", "Step sizes in log(rho) of the gradient check (comma-separated)", "im.csem",
                                    imParams->checkSteps, &n, &set));
    if (set) imParams->numCheckSteps = n;
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

  if (!imParams->fixedMaterialsFromCli) imParams->numFixedMaterials = 0;

  /* fm-native observed data lives in a forward responses file, not the bundle, so a path is mandatory in that mode. */
  PetscCheck(imParams->observedMode != OBS_FM_NATIVE || imParams->observedFile[0] != '\0', PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "-im_observed_mode fm_native requires -im_observed_file (path to the fm.csem responses HDF5).");

  /* numFreqs/imSources are populated later by setupInversionSources (from the unified bundle's /sources group). */
  imParams->numFreqs = 0;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Loads inversion metadata overrides from the HDF5 bundle.
 *
 * This function reads optional inversion-related metadata from the
 * unified PETGEM input file and applies it to the inversion parameter
 * structure. It complements readImParams() by filling values
 * that are defined at the case level rather than the CLI.
 *
 * The following fields may be updated from the bundle:
 *   - error_level attribute of /observed → imParams->errorLevel
 *   - /im_meta/fixed_materials → fixed material IDs list
 *
 * CLI values always take precedence:
 *   - If errorLevelFromCli is set, /observed is ignored.
 *   - If fixedMaterialsFromCli is set, bundle values are ignored.
 *
 * Missing entries are silently skipped, leaving defaults unchanged.
 */
PetscErrorCode loadInversionMetaFromBundle(const char *bundleFile, ImParams *imParams) {
  PetscFunctionBeginUser;

  PetscCheck(bundleFile && bundleFile[0] != '\0', PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "loadInversionMetaFromBundle: bundleFile is empty.");

  MPI_Comm    comm            = PETSC_COMM_WORLD;
  const char *errLevelOrigin  = imParams->errorLevelFromCli ? "CLI" : "default";
  const char *fixedMatsOrigin = imParams->fixedMaterialsFromCli ? "CLI" : "default";

  /* Open the bundle through the PETSc viewer. PetscViewerHDF5HasAttribute /
   * HasObject internally silence the underlying H5E error stack while probing
   * for optional entries, so missing entries return cleanly via the `has`
   * flag without any manual H5Eset_auto2 handling. */
  PetscViewer viewer;
  PetscCall(PetscViewerHDF5Open(comm, bundleFile, FILE_MODE_READ, &viewer));

  /* /observed @error_level (optional) */
  if (!imParams->errorLevelFromCli) {
    PetscBool hasAttr = PETSC_FALSE;
    PetscCall(PetscViewerHDF5HasAttribute(viewer, "/observed", "error_level", &hasAttr));
    if (hasAttr) {
      PetscCall(PetscViewerHDF5ReadAttribute(viewer, "/observed", "error_level", PETSC_REAL, NULL, &imParams->errorLevel));
      errLevelOrigin = "bundle";
    }
  }

  /* /im_meta/fixed_materials (optional, int32 array written by the Python
   * preprocess). Presence is probed with PetscViewerHDF5HasDataset; the read
   * itself stays on the raw H5D path because the dataset is stored as int32
   * while PetscInt may be 64-bit under --with-64-bit-indices, requiring an
   * explicit per-element widen. */
  if (!imParams->fixedMaterialsFromCli) {
    PetscBool hasObj = PETSC_FALSE;
    PetscCall(PetscViewerHDF5HasDataset(viewer, "/im_meta/fixed_materials", &hasObj));
    if (hasObj) {
      hid_t file = -1;
      PetscCall(PetscViewerHDF5GetFileId(viewer, &file));
      hid_t dset = H5Dopen2(file, "/im_meta/fixed_materials", H5P_DEFAULT);
      PetscCheck(dset >= 0, comm, PETSC_ERR_FILE_READ, "/im_meta/fixed_materials present but H5Dopen2 failed");
      hid_t   sp      = H5Dget_space(dset);
      hsize_t dims[1] = {0};
      herr_t  status;
      H5Sget_simple_extent_dims(sp, dims, NULL);
      H5Sclose(sp);
      PetscInt n = (PetscInt)dims[0];
      PetscCheck(n <= IM_MAX_FIXED_MATERIALS, comm, PETSC_ERR_SUP,
                 "/im_meta/fixed_materials has %" PetscInt_FMT " entries; "
                 "IM_MAX_FIXED_MATERIALS=%d. Bump the cap in include/constants.h.",
                 n, IM_MAX_FIXED_MATERIALS);
      int *buf;
      PetscCall(PetscMalloc1(n, &buf));
      status = H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
      H5Dclose(dset);
      PetscCheck(status >= 0, comm, PETSC_ERR_FILE_READ, "Cannot read /im_meta/fixed_materials from %s", bundleFile);
      for (PetscInt i = 0; i < n; i++) {
        imParams->fixedMaterials[i] = (PetscInt)buf[i];
      }
      imParams->numFixedMaterials = n;
      PetscCall(PetscFree(buf));
      fixedMatsOrigin = "bundle";
    }
  }

  PetscCall(PetscViewerDestroy(&viewer));

  imParams->errorLevelOrigin     = errLevelOrigin;
  imParams->fixedMaterialsOrigin = fixedMatsOrigin;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Loads inversion sources from the unified PETGEM HDF5 bundle.
 *
 * This function reads the /sources group from the input bundle and
 * constructs the inversion source list used by the optimization layer.
 * Each entry corresponds to one frequency-dipole configuration, with
 * shared structure between forward and inverse problems.
 *
 * The data is stored as PETSc Vecs, so loading is performed using
 * PetscViewerHDF5 + VecLoad to preserve correct scalar-type handling.
 *
 * The function populates:
 *   - imParams->imSources[] (per-frequency dipole definitions)
 *   - imParams->numFreqs     (number of loaded entries)
 *
 * A hard limit (IM_MAX_FREQUENCIES) is enforced to prevent oversized
 * inversion problems.
 */
PetscErrorCode setupInversionSources(const char *bundleFile, ImParams *imParams) {
  PetscFunctionBeginUser;

  PetscInt   count;
  PetscReal *rows;

  PetscCheck(bundleFile && bundleFile[0] != '\0', PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "setupInversionSources: bundleFile is empty.");

  /* Same /sources group as the forward kernel, one entry per (frequency, dipole) */
  PetscCall(loadSourceRows(bundleFile, &count, &rows));
  PetscCheck(count <= IM_MAX_FREQUENCIES, PETSC_COMM_WORLD, PETSC_ERR_SUP,
             "Too many inversion sources (%" PetscInt_FMT " > max %d); "
             "bump IM_MAX_FREQUENCIES in include/constants.h to raise the cap",
             count, IM_MAX_FREQUENCIES);

  for (PetscInt i = 0; i < count; i++) {
    const PetscReal *r       = &rows[i * CSEM_SRC_NUM_COLUMNS];
    ImCsemSource    *src     = &imParams->imSources[i];
    src->freq                = r[CSEM_SRC_FREQ];
    src->dipole.position[0]  = r[CSEM_SRC_X];
    src->dipole.position[1]  = r[CSEM_SRC_Y];
    src->dipole.position[2]  = r[CSEM_SRC_Z];
    src->dipole.current      = r[CSEM_SRC_CURRENT];
    src->dipole.length       = r[CSEM_SRC_LENGTH];
    src->dipole.dipAngle     = r[CSEM_SRC_DIP];
    src->dipole.azimuthAngle = r[CSEM_SRC_AZIMUTH];
  }
  PetscCall(PetscFree(rows));

  imParams->numFreqs = count;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints the "Inversion sources" and "Inversion parameters" sections.
 *
 * When every entry shares one dipole the dipole is printed once followed by
 * the list of frequencies; otherwise one table row is printed per entry. The
 * parameters section lists the L-BFGS settings, the active stopping criteria,
 * the data error level, the fixed materials, the smoother self-weight and the
 * snapshot interval.
 */
PetscErrorCode logInversionSummary(const ImParams *imParams) {
  PetscFunctionBeginUser;

  MPI_Comm          comm   = PETSC_COMM_WORLD;
  const CsemSource *s0     = &imParams->imSources[0].dipole;
  PetscBool         shared = PETSC_TRUE;
  char              line[1024];

  for (PetscInt i = 1; i < imParams->numFreqs; i++) {
    const CsemSource *s = &imParams->imSources[i].dipole;
    if (s->position[0] != s0->position[0] || s->position[1] != s0->position[1] || s->position[2] != s0->position[2] ||
        s->current != s0->current || s->length != s0->length || s->dipAngle != s0->dipAngle || s->azimuthAngle != s0->azimuthAngle) {
      shared = PETSC_FALSE;
      break;
    }
  }

  PetscCall(logSection(comm, "Inversion sources"));
  PetscCall(logKvInt(comm, "Number of entries", imParams->numFreqs));
  if (shared) {
    line[0] = '\0';
    for (PetscInt i = 0; i < imParams->numFreqs; i++) {
      if (i > 0) PetscCall(PetscStrlcat(line, ", ", sizeof(line)));
      PetscCall(PetscStrlcat(line, formatReal(imParams->imSources[i].freq), sizeof(line)));
    }
    PetscCall(logKvStr(comm, "Frequencies (Hz)", line));
    PetscCall(logKvf(comm, "Position (m)", "(%s, %s, %s)", formatReal(s0->position[0]), formatReal(s0->position[1]),
                     formatReal(s0->position[2])));
    PetscCall(logKvReal(comm, "Current (A)", s0->current));
    PetscCall(logKvReal(comm, "Length (m)", s0->length));
    PetscCall(logKvf(comm, "Dip / azimuth (deg)", "%s / %s", formatReal(s0->dipAngle), formatReal(s0->azimuthAngle)));
  } else {
    PetscCall(PetscPrintf(comm, "     %-5s %10s  %-28s %8s %8s %8s %8s\n", "Entry", "Freq (Hz)", "Position (m)", "I (A)", "L (m)", "Dip",
                          "Azimuth"));
    for (PetscInt i = 0; i < imParams->numFreqs; i++) {
      const CsemSource *s = &imParams->imSources[i].dipole;
      char              pos[40];
      PetscCall(PetscSNPrintf(pos, sizeof(pos), "(%s, %s, %s)", formatReal(s->position[0]), formatReal(s->position[1]),
                              formatReal(s->position[2])));
      PetscCall(PetscPrintf(comm, "     %5" PetscInt_FMT " %10s  %-28s %8s %8s %8s %8s\n", i + 1, formatReal(imParams->imSources[i].freq),
                            pos, formatReal(s->current), formatReal(s->length), formatReal(s->dipAngle), formatReal(s->azimuthAngle)));
    }
  }

  PetscCall(logSection(comm, "Inversion parameters"));
  PetscCall(logKvInt(comm, "Max iterations", imParams->maxIter));
  PetscCall(logKvInt(comm, "L-BFGS memory (M)", imParams->lbfgsMemory));
  PetscCall(logKvReal(comm, "Tikhonov lambda", imParams->lambda));
  if (imParams->rmsTol > 0.0) {
    PetscCall(logKvf(comm, "Stop: RMS target", "RMS <= %s", formatCompactReal(imParams->rmsTol)));
  } else {
    PetscCall(logKvStr(comm, "Stop: RMS target", "disabled"));
  }
  if (imParams->rmsRelTol > 0.0) {
    PetscCall(logKvf(comm, "Stop: RMS plateau", "drop < %s for %" PetscInt_FMT " iterations", formatCompactReal(imParams->rmsRelTol),
                     imParams->rmsStallWindow));
  } else {
    PetscCall(logKvStr(comm, "Stop: RMS plateau", "disabled"));
  }
  PetscCall(logKvf(comm, "Stop: gradient", "|g|/|x| <= %s", formatCompactReal(imParams->gtol)));
  line[0] = '\0';
  for (PetscInt i = 0; i < imParams->numComponents; i++) {
    if (i > 0) PetscCall(PetscStrlcat(line, ", ", sizeof(line)));
    PetscCall(PetscStrlcat(line, imDataComponentNames[imParams->components[i]], sizeof(line)));
  }
  PetscCall(logKvStr(comm, "Data components", line));
  PetscCall(logKvf(comm, "Error level", "%s (from %s)", formatReal(imParams->errorLevel), imParams->errorLevelOrigin));

  line[0] = '\0';
  for (PetscInt i = 0; i < imParams->numFixedMaterials; i++) {
    char one[24];
    PetscCall(PetscSNPrintf(one, sizeof(one), "%s%" PetscInt_FMT, (i == 0) ? "" : " ", imParams->fixedMaterials[i]));
    PetscCall(PetscStrlcat(line, one, sizeof(line)));
  }
  if (imParams->numFixedMaterials == 0) PetscCall(PetscStrncpy(line, "none", sizeof(line)));
  PetscCall(logKvf(comm, "Fixed materials", "%s (from %s)", line, imParams->fixedMaterialsOrigin));
  PetscCall(logKvReal(comm, "Smoother self-weight", imParams->diagGradientWeight));
  if (imParams->snapshotInterval > 0) {
    PetscCall(logKvf(comm, "VTU snapshot", "every %s accepted L-BFGS step(s)", formatGroupedInt(imParams->snapshotInterval)));
  } else {
    PetscCall(logKvStr(comm, "VTU snapshot", "disabled"));
  }
  if (imParams->gradientCheck) PetscCall(logKvStr(comm, "Gradient check", "on (no inversion)"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* /observed/<name> (complex128 {r, i}, numFreqs x numReceivers) read on rank 0 and broadcast. */
static PetscErrorCode readObservedExternal(const char *file, const char *name, PetscInt numFreqs, PetscInt numReceivers, PetscScalar *obs) {
  PetscFunctionBeginUser;
  MPI_Comm    comm = PETSC_COMM_WORLD;
  PetscMPIInt rank;
  char        path[64];

  PetscCall(PetscSNPrintf(path, sizeof(path), "/observed/%s", name));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    hid_t   fileId, dset, space, h5complex;
    hsize_t dims[2] = {0, 0};
    herr_t  status;

    fileId = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
    PetscCheck(fileId >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open %s", file);
    PetscCheck(H5Lexists(fileId, "/observed", H5P_DEFAULT) > 0 && H5Lexists(fileId, path, H5P_DEFAULT) > 0, PETSC_COMM_SELF,
               PETSC_ERR_FILE_READ, "Cannot find %s in %s (see -im_data_components)", path, file);
    dset = H5Dopen2(fileId, path, H5P_DEFAULT);
    PetscCheck(dset >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Cannot open %s in %s", path, file);
    space = H5Dget_space(dset);
    PetscCheck(H5Sget_simple_extent_ndims(space) == 2, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "%s must be 2D", path);
    H5Sget_simple_extent_dims(space, dims, NULL);
    H5Sclose(space);
    PetscCheck((PetscInt)dims[0] == numFreqs && (PetscInt)dims[1] == numReceivers, PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
               "%s shape [%llu x %llu] != expected [%" PetscInt_FMT " x %" PetscInt_FMT "]", path, (unsigned long long)dims[0],
               (unsigned long long)dims[1], numFreqs, numReceivers);

    h5complex = H5Tcreate(H5T_COMPOUND, sizeof(PetscScalar));
    H5Tinsert(h5complex, "r", 0, H5T_NATIVE_DOUBLE);
    H5Tinsert(h5complex, "i", sizeof(double), H5T_NATIVE_DOUBLE);
    status = H5Dread(dset, h5complex, H5S_ALL, H5S_ALL, H5P_DEFAULT, obs);
    H5Tclose(h5complex);
    H5Dclose(dset);
    H5Fclose(fileId);
    PetscCheck(status >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Cannot read %s from %s", path, file);
  }
  PetscCallMPI(MPI_Bcast(obs, (PetscMPIInt)(numFreqs * numReceivers), MPIU_SCALAR, 0, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Components of an fm.csem responses file from /sources/src{k}/fields (source k -> row k-1), read on every rank. */
static PetscErrorCode readObservedFmNative(const char *file, const ImParams *imParams, PetscInt numReceivers, PetscScalar *obs) {
  PetscFunctionBeginUser;
  const char *const fieldNames[NUM_RECEIVER_FIELDS] = {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"};
  const PetscInt    numFreqs                        = imParams->numFreqs;
  PetscViewer       viewer;

  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, file, FILE_MODE_READ, &viewer));
  for (PetscInt k = 0; k < numFreqs; k++) {
    char               group[PETSC_MAX_PATH_LEN];
    Vec                v[NUM_RECEIVER_FIELDS] = {NULL};
    const PetscScalar *a[NUM_RECEIVER_FIELDS] = {NULL};

    PetscCall(PetscSNPrintf(group, sizeof(group), "/sources/src%" PetscInt_FMT "/fields", k + 1));
    PetscCall(PetscViewerHDF5PushGroup(viewer, group));
    for (PetscInt i = 0; i < imParams->numComponents; i++) {
      ReceiverField num, den;
      imDataComponentFields(imParams->components[i], &num, &den);
      for (PetscInt j = 0; j < 2; j++) {
        const ReceiverField fld = (j == 0) ? num : den;
        PetscInt            n;
        if (fld == NUM_RECEIVER_FIELDS || v[fld]) continue;
        PetscCall(loadSerialVec(viewer, fieldNames[fld], &v[fld]));
        PetscCall(VecGetSize(v[fld], &n));
        PetscCheck(n == numReceivers, PETSC_COMM_WORLD, PETSC_ERR_FILE_READ,
                   "%s: %s/%s length %" PetscInt_FMT " != expected receivers %" PetscInt_FMT, file, group, fieldNames[fld], n,
                   numReceivers);
        PetscCall(VecGetArrayRead(v[fld], &a[fld]));
      }
    }
    PetscCall(PetscViewerHDF5PopGroup(viewer));
    for (PetscInt i = 0; i < imParams->numComponents; i++) {
      ReceiverField num, den;
      PetscScalar  *row = &obs[(i * numFreqs + k) * numReceivers];
      imDataComponentFields(imParams->components[i], &num, &den);
      for (PetscInt r = 0; r < numReceivers; r++) row[r] = (den == NUM_RECEIVER_FIELDS) ? a[num][r] : a[num][r] / a[den][r];
    }
    for (PetscInt f = 0; f < NUM_RECEIVER_FIELDS; f++) {
      if (!v[f]) continue;
      PetscCall(VecRestoreArrayRead(v[f], &a[f]));
      PetscCall(VecDestroy(&v[f]));
    }
  }
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode loadObservedDataset(const ImParams *imParams, PetscInt numReceivers, PetscScalar **dObs) {
  PetscFunctionBeginUser;
  MPI_Comm        comm  = PETSC_COMM_WORLD;
  const char     *file  = (imParams->observedFile[0] != '\0') ? imParams->observedFile : imParams->common.inputFile;
  const char     *where = (strcmp(file, imParams->common.inputFile) == 0) ? "input bundle" : file;
  const PetscBool fm    = (PetscBool)(imParams->observedMode == OBS_FM_NATIVE);
  const PetscInt  block = imParams->numFreqs * numReceivers;
  char            line[256];

  PetscCall(PetscMalloc1(imParams->numComponents * block, dObs));
  line[0] = '\0';
  if (fm) {
    PetscCall(readObservedFmNative(file, imParams, numReceivers, *dObs));
    PetscCall(PetscStrncpy(line, "/sources/src*/fields", sizeof(line)));
  } else {
    for (PetscInt i = 0; i < imParams->numComponents; i++) {
      const char *name = imDataComponentNames[imParams->components[i]];
      PetscCall(readObservedExternal(file, name, imParams->numFreqs, numReceivers, &(*dObs)[i * block]));
      PetscCall(PetscStrlcat(line, (i == 0) ? "/observed/" : ", /observed/", sizeof(line)));
      PetscCall(PetscStrlcat(line, name, sizeof(line)));
    }
  }

  PetscCall(logSection(comm, "Observed data"));
  PetscCall(logKvStr(comm, "Dataset", line));
  PetscCall(logKvStr(comm, "File", where));
  PetscCall(logKvInt(comm, "Frequencies", imParams->numFreqs));
  PetscCall(logKvInt(comm, "Receivers", numReceivers));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Writes n reals held on rank 0 as an MPI Vec named name. */
static PetscErrorCode writeRealHistory(PetscViewer viewer, MPI_Comm comm, const char *name, const PetscReal *values, PetscInt n) {
  PetscFunctionBeginUser;
  PetscMPIInt rank;
  Vec         v;

  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(VecCreateMPI(comm, (rank == 0) ? n : 0, n, &v));
  PetscCall(PetscObjectSetName((PetscObject)v, name));
  if (rank == 0) {
    PetscScalar *a;
    PetscCall(VecGetArray(v, &a));
    for (PetscInt i = 0; i < n; i++) a[i] = values[i];
    PetscCall(VecRestoreArray(v, &a));
  }
  PetscCall(VecView(v, viewer));
  PetscCall(VecDestroy(&v));
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
 *   - grad_norm_history : ||g|| per accepted L-BFGS iteration, initial model first
 *   - predicted_<comp>  : each fitted component of the recovered model, numFreqs x numReceivers
 *
 * Additional attributes store provenance and inversion metadata, including
 * PETGEM version, the stopping reason, and BOTH iteration counters: attribute
 * num_iterations is the accepted-L-BFGS-step count, num_objgrad_evaluations is
 * the number of objective-gradient evaluations, which is what rms_history is
 * indexed by. They differ by the rejected line-search trials, so consumers must
 * not use one where the other is meant.
 */
PetscErrorCode writeInversionResults(const ImParams *imParams, DM dmConductivity, Vec conductivity, Vec X, const PetscReal *allRms,
                                     PetscInt numIters, PetscInt numEvals, const char *reasonStr, const PetscReal *gradNorms,
                                     PetscInt numGradNorms, const PetscScalar *dPre, PetscInt numReceivers, PetscReal rmsPre) {
  PetscFunctionBeginUser;

  MPI_Comm comm = PetscObjectComm((PetscObject)dmConductivity);

  /* Output path: composed by the shared builder from the parameters already
   * parsed by readPetgemParams. */
  char outFile[PETSC_MAX_PATH_LEN];
  PetscCall(buildOutputPath(&imParams->common, ".h5", outFile, sizeof(outFile)));

  /* Scatter local conductivity to global for output */
  Vec globalConductivity;
  PetscCall(DMCreateGlobalVector(dmConductivity, &globalConductivity));
  PetscCall(DMLocalToGlobal(dmConductivity, conductivity, INSERT_VALUES, globalConductivity));

  /* Plain MPI copies (no DM): VecView_Plex_HDF5 rejects the per-cell section's block size */
  Vec outRes, outX;
  PetscCall(createPlainCopy(globalConductivity, "conductivity", &outRes));
  PetscCall(createPlainCopy(X, "log_perturbation", &outX));

  /* Create HDF5 viewer and write vectors */
  PetscViewer viewer;
  PetscCall(PetscViewerHDF5Open(comm, outFile, FILE_MODE_WRITE, &viewer));

  PetscCall(VecView(outRes, viewer));
  PetscCall(VecView(outX, viewer));

  /* Root provenance block shared with fm.csem, then the inversion attributes. */
  PetscCall(writeRunProvenance(viewer, &imParams->common, PETGEM_SIM_IM));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "num_frequencies", PETSC_INT, &imParams->numFreqs));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "lambda", PETSC_REAL, &imParams->lambda));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "error_level", PETSC_REAL, &imParams->errorLevel));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "num_iterations", PETSC_INT, &numIters));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "num_objgrad_evaluations", PETSC_INT, &numEvals));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "convergence_reason", PETSC_STRING, reasonStr));

  /* RMS per objective-gradient evaluation, ||g|| per accepted iteration */
  if (numEvals > 0) PetscCall(writeRealHistory(viewer, comm, "rms_history", allRms, numEvals));
  PetscCall(writeRealHistory(viewer, comm, "grad_norm_history", gradNorms, numGradNorms));

  /* Predicted data of the recovered model, owned by rank 0; block size gives the (numFreqs, numReceivers) shape */
  {
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    const PetscInt n = imParams->numFreqs * numReceivers;
    char           components[256];

    components[0] = '\0';
    for (PetscInt i = 0; i < imParams->numComponents; i++) {
      const char *comp = imDataComponentNames[imParams->components[i]];
      char        name[32];
      Vec         preVec;

      PetscCall(PetscSNPrintf(name, sizeof(name), "predicted_%s", comp));
      PetscCall(VecCreate(comm, &preVec));
      PetscCall(VecSetSizes(preVec, (rank == 0) ? n : 0, n));
      PetscCall(VecSetBlockSize(preVec, numReceivers));
      PetscCall(VecSetType(preVec, VECMPI));
      PetscCall(PetscObjectSetName((PetscObject)preVec, name));
      if (rank == 0) {
        PetscScalar *p;
        PetscCall(VecGetArray(preVec, &p));
        PetscCall(PetscArraycpy(p, &dPre[i * n], n));
        PetscCall(VecRestoreArray(preVec, &p));
      }
      PetscCall(VecView(preVec, viewer));
      PetscCall(VecDestroy(&preVec));
      PetscCall(PetscStrlcat(components, (i == 0) ? "" : ",", sizeof(components)));
      PetscCall(PetscStrlcat(components, comp, sizeof(components)));
    }
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "data_components", PETSC_STRING, components));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "rms_predicted", PETSC_REAL, &rmsPre));
  }

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&outRes));
  PetscCall(VecDestroy(&outX));
  PetscCall(VecDestroy(&globalConductivity));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode writeGradientCheck(const ImParams *imParams, MPI_Comm comm, const PetscReal *eps, const PetscReal *fd,
                                  const PetscReal *adjoint, PetscInt n) {
  PetscFunctionBeginUser;
  char        outFile[PETSC_MAX_PATH_LEN], components[256];
  PetscViewer viewer;

  PetscCall(buildOutputPath(&imParams->common, ".h5", outFile, sizeof(outFile)));
  PetscCall(PetscViewerHDF5Open(comm, outFile, FILE_MODE_WRITE, &viewer));
  PetscCall(writeRunProvenance(viewer, &imParams->common, PETGEM_SIM_IM));
  components[0] = '\0';
  for (PetscInt i = 0; i < imParams->numComponents; i++) {
    PetscCall(PetscStrlcat(components, (i == 0) ? "" : ",", sizeof(components)));
    PetscCall(PetscStrlcat(components, imDataComponentNames[imParams->components[i]], sizeof(components)));
  }
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "data_components", PETSC_STRING, components));
  PetscCall(writeRealHistory(viewer, comm, "gradient_check_eps", eps, n));
  PetscCall(writeRealHistory(viewer, comm, "gradient_check_fd", fd, n));
  PetscCall(writeRealHistory(viewer, comm, "gradient_check_adjoint", adjoint, n));
  PetscCall(PetscViewerDestroy(&viewer));
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
 */
PetscErrorCode writeInversionSnapshotVtu(const InversionContext *ctx, PetscInt acceptedIter) {
  PetscFunctionBeginUser;
  MPI_Comm           comm   = PetscObjectComm((PetscObject)ctx->dm);
  const PetscInt     nOwned = ctx->grid.cellEnd - ctx->grid.cellStart;
  PetscMPIInt        rank, size;
  PetscSection       sec;
  const PetscScalar *sigma;
  PetscReal         *rho, *coords;
  char               snapDir[PETSC_MAX_PATH_LEN], file[PETSC_MAX_PATH_LEN], piece[PETSC_MAX_PATH_LEN];

  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  /* {output_dir}/snapshots, shared by every run writing to that directory */
  PetscCall(joinPath(ctx->imParams->common.outputDirectory, "snapshots", snapDir, sizeof(snapDir)));
  if (rank == 0) {
    PetscBool exists = PETSC_FALSE;
    PetscCall(PetscTestDirectory(snapDir, 'w', &exists));
    if (!exists) PetscCall(PetscMkdir(snapDir));
  }
  PetscCallMPI(MPI_Barrier(comm));

  /* rho = 1/sigma_x and the vertices of every owned cell */
  PetscCall(PetscCalloc2(nOwned, &rho, 12 * nOwned, &coords));
  PetscCall(DMGetLocalSection(ctx->dmConductivity, &sec));
  PetscCall(VecGetArrayRead(ctx->conductivity, &sigma));
  for (PetscInt i = ctx->grid.cellStart; i < ctx->grid.cellEnd; i++) {
    const PetscInt li = i - ctx->grid.cellStart;
    PetscInt       off;
    Cell           cell;
    PetscCall(PetscSectionGetOffset(sec, i, &off));
    rho[li] = (PetscRealPart(sigma[off]) > 0.0) ? 1.0 / PetscRealPart(sigma[off]) : 0.0;
    PetscCall(extractCellCoordinates(ctx->dm, i, &cell));
    for (PetscInt j = 0; j < 12; j++) coords[12 * li + j] = cell.coordinates[j];
  }
  PetscCall(VecRestoreArrayRead(ctx->conductivity, &sigma));

  PetscCall(vtuPieceName(acceptedIter, rank, piece, sizeof(piece)));
  PetscCall(joinPath(snapDir, piece, file, sizeof(file)));
  PetscCall(writeVtuPiece(file, nOwned, coords, rho));
  PetscCall(PetscFree2(rho, coords));

  /* Master after every piece exists */
  PetscCallMPI(MPI_Barrier(comm));
  if (rank == 0) {
    PetscCall(PetscSNPrintf(file, sizeof(file), "%s/iter%04" PetscInt_FMT ".pvtu", snapDir, acceptedIter));
    PetscCall(writePvtuMaster(file, acceptedIter, size));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
