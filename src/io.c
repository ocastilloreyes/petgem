/*
 * Filename: io.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Centralized I/O and parameter parsing for both PETGEM kernels.
 */

/*
 * Public surface:
 *   Forward kernel (fm.csem):
 *     readCsemParams         PETSc-options-database parser (shared params)
 *
 *   Inverse kernel (im.csem):
 *     readInversionParams        parser for -inv_* options
 *     setupInversionSources      bundle reader for /sources group (multi-freq sources)
 *     loadObservedData           bundle reader for /observed/Ex (HDF5 compound complex128)
 *     writeInversionResults      final HDF5 dump (conductivity, X, RMS history)
 *     writeInversionSnapshotVTU  per-accepted-iter ParaView VTU snapshot
 *
 * Both kernels link this single translation unit. fm.csem never calls
 * the inversion-side functions, but the linker still pulls them in
 * (small dead code; HDF5 is already linked via $(PETSC_LIB)/$(HDF5_LIB)).
 *
 * Created by merging the previous src/inputs.c and src/inversion_io.c.
 */

/* C libraries */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

/* PETSc + HDF5 libraries */
#include <petsc.h>
#include <petscsys.h>
#include <petscviewerhdf5.h>
#include <hdf5.h>

/* PETGEM headers */
#include "common.h"
#include "constants.h"
#include "grid.h"
#include "inputs.h"
#include "inversion.h"
#include "inversion_internal.h"
#include "io.h"
#include "transmitter.h"
#include "version.h"

/**
 * @brief Reads and validates CSEM CLI parameters from the PETSc options database.
 *
 * Required options:
 *   -input_filename   : unified PETGEM input HDF5 (mesh + sigma + materials_id
 *                       + receivers + forward sources). Produced by the Python
 *                       preprocessor (utils/functions.py::writeBundle).
 *   -output_dir       : directory where the kernel writes its outputs.
 *   -output_filename  : base name for output artifacts (responses_*, etc.).
 *   -nord             : finite-element basis order, integer in 1..6.
 *
 */
PetscErrorCode readCsemParams(const PetscMPIInt size, csemParams* params) {

  PetscFunctionBeginUser;

  char      inputFilename[PETSC_MAX_PATH_LEN];
  char      outputDir[PETSC_MAX_PATH_LEN];
  char      outputFilename[PETSC_MAX_PATH_LEN];
  PetscBool inputIsPresent, outputDirIsPresent, outputFilenameIsPresent;
  PetscBool nordIsPresent;
  PetscInt  nord;

  /* Unified input bundle (mesh + sigma + materials_id + receivers + forward sources) */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-input_filename",
                                  inputFilename, sizeof(inputFilename),
                                  &inputIsPresent));
  PetscCheck(inputIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Exiting: -input_filename missing. Mandatory parameter required for simulation.\n");
  PetscCall(PetscStrncpy(params->inputFile, inputFilename, sizeof(params->inputFile)));

  /* Output directory */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-output_dir",
                                  outputDir, sizeof(outputDir),
                                  &outputDirIsPresent));
  PetscCheck(outputDirIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Exiting: -output_dir missing. Mandatory parameter required for simulation.\n");
  PetscCall(PetscStrncpy(params->outputDirectory, outputDir, sizeof(params->outputDirectory)));

  /* Output filename */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-output_filename",
                                  outputFilename, sizeof(outputFilename),
                                  &outputFilenameIsPresent));
  PetscCheck(outputFilenameIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Exiting: -output_filename missing. Mandatory parameter required for simulation.\n");
  PetscCall(PetscStrncpy(params->outputFilename, outputFilename, sizeof(params->outputFilename)));

  /* Basis order is optional here - the bundle's /nord dataset is the
   * normal source of truth and is read by loadCsemInputs. -nord on the
   * command line is honored as an override (handy for debug runs that
   * want to mismatch on purpose). Leave params->nord = 0 to signal
   * "use bundle". */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nord", &nord, &nordIsPresent));
  if (nordIsPresent) {
    PetscCheck(nord >= 1 && nord <= 6, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "Exiting: -nord override out of valid range (must be 1..6).\n");
    params->nord = nord;
  } else {
    params->nord = 0;  /* sentinel: loadCsemInputs will fill from /nord */
  }

  params->numMPITasks = size;
  params->quiet       = PETSC_FALSE;

  createDirectory(outputDir);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* loadCsemInputs                                                       */
/*                                                                      */
/* Unified loader for the bundled PETGEM input HDF5.  Replaces the      */
/* legacy importGrid + setupCsemSource + receivers-file-open trio.      */
/* See include/io.h for the contract.                                   */
/* ================================================================== */

/* Internal helper: load a 1-D Vec on PETSC_COMM_SELF from the given
 * HDF5 viewer + dataset name (relative to the viewer's current group). */
static PetscErrorCode loadSelfVecByName(PetscViewer viewer, const char *name, Vec *out)
{
  PetscFunctionBeginUser;
  PetscCall(VecCreate(PETSC_COMM_SELF, out));
  PetscCall(PetscObjectSetName((PetscObject)*out, name));
  PetscCall(VecLoad(*out, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode loadCsemInputs(csemParams     *params,
                              DM             *odm,
                              Vec            *conductivity_output,
                              Vec            *materials_id_output,
                              CsemSourceSet  *sources,
                              Vec            *receivers_output)
{
  PetscFunctionBeginUser;

  PetscCheck(params->inputFile[0] != '\0', PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "loadCsemInputs: params->inputFile is empty (set -input_filename).");

  /* ---------------------------------------------------------------- */
  /* Phase 1: mesh + sections + combined model vector (parallel I/O). */
  /* Replicates the legacy importGrid behavior so downstream code     */
  /* sees the same DMPlex / sub-DM / local-Vec configuration.         */
  /* ---------------------------------------------------------------- */
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

    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, params->inputFile,
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

    /* 2-field section: field 0 = conductivity (3 dofs/cell),
     *                  field 1 = materials_id (1 dof/cell).
     * DMSetNumFields is intentionally NOT called - for plain section-based
     * DMs (no PetscFE/FV) it triggers DMCreateLocalSection_Plex which fails
     * without a discretization object.  The 2-field layout is recovered
     * from the loaded section directly. */
    PetscCall(DMPlexSectionLoad(dm, viewer, NULL, sfXC, &sfG, NULL));

    /* Combined global vector - 4 dofs/cell [sigma_x sigma_y sigma_z mat_id]. */
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

    /* Global → local so ghosts (needed for closure walks across partition
     * boundaries) are populated on every rank. */
    PetscCall(DMCreateLocalVector(dm, &combinedLocal));
    PetscCall(DMGlobalToLocal(dm, combinedGlobal, INSERT_VALUES, combinedLocal));
    PetscCall(VecDestroy(&combinedGlobal));

    PetscCall(DMGetLocalSection(dm, &combinedSection));
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cellStart, &cellEnd));

    /* Conductivity sub-DM: clone dm, install a plain section with
     * NUM_CONDUCTIVITY_COMPONENTS dofs/cell. */
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

    /* Split the combined per-cell block [σx σy σz mat] into the two sub-Vecs. */
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

    /* Cloned DM is the public-facing handle; setupCsemGrid installs the
     * H(curl) section on it without disturbing the original. */
    PetscCall(DMClone(dm, odm));
    PetscCall(DMDestroy(&dm));

    *conductivity_output  = localSigma;
    *materials_id_output  = localMat;

    PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_vec_type", typeName, sizeof(typeName), &flg));
    if (flg) PetscCall(DMSetVecType(*odm, typeName));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_mat_type", typeName, sizeof(typeName), &flg));
    if (flg) PetscCall(DMSetMatType(*odm, typeName));
  }

  /* ---------------------------------------------------------------- */
  /* Phase 2: /nord, receivers, and sources (rank-local serial I/O).  */
  /* /nord is always read so the caller does not need -nord in the    */
  /* params file. CLI -nord (if set) overrides the bundle value, but  */
  /* normally readCsemParams leaves it at zero and the bundle wins.   */
  /* ---------------------------------------------------------------- */
  {
    PetscViewer viewer;
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, params->inputFile,
                                  FILE_MODE_READ, &viewer));

    /* /nord - single-element Vec. Bundle is the source of truth unless
     * the caller explicitly set -nord on the command line (params->nord
     * non-zero at entry). */
    if (params->nord <= 0) {
      Vec nordV;
      const PetscScalar *nArr;
      PetscCall(loadSelfVecByName(viewer, "nord", &nordV));
      PetscCall(VecGetArrayRead(nordV, &nArr));
      params->nord = (PetscInt)(PetscRealPart(nArr[0]) + 0.5);
      PetscCall(VecRestoreArrayRead(nordV, &nArr));
      PetscCall(VecDestroy(&nordV));
    }
    PetscCheck(params->nord >= 1 && params->nord <= 6, PETSC_COMM_WORLD,
               PETSC_ERR_ARG_OUTOFRANGE,
               "loadCsemInputs: nord %" PetscInt_FMT " not in 1..6 (bundle %s)",
               params->nord, params->inputFile);

    if (receivers_output) {
      Vec recv;
      PetscCall(loadSelfVecByName(viewer, "receivers", &recv));
      PetscCall(VecSetBlockSize(recv, NUM_DIMENSIONS));
      *receivers_output = recv;
    }

    if (sources) {
      Vec freqV, posV, curV, lenV, dipV, azV;
      PetscCall(PetscViewerHDF5PushGroup(viewer, "/sources"));
      /* Unified /sources schema: per-entry frequency (one row per
       * transmitter). Forward modeling is monochromatic - all entries
       * share the same frequency - so we use freq[0] for the whole set. */
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

      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nCSEM source data:\n"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Freq (Hz)         = %g\n", (double)sources->freq));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Number of sources = %" PetscInt_FMT "\n", sources->numSources));
      for (PetscInt i = 0; i < sources->numSources; i++) {
        const CsemSource *s = &sources->sourceArray[i];
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Data for source %" PetscInt_FMT ":\n", i + 1));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Current         = %g\n", (double)s->current));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Length          = %g\n", (double)s->length));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Dip             = %g\n", (double)s->dipAngle));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Azimuth         = %g\n", (double)s->azimuthAngle));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Position (xyz)  = [%g, %g, %g]\n",
                              (double)s->position[0], (double)s->position[1], (double)s->position[2]));
      }
    }

    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* readInversionParams                                                 */
/* ================================================================== */
PetscErrorCode readInversionParams(invParams *iparams)
{
  /* Variables declaration */
  PetscBool flg;

  PetscFunctionBeginUser;

  /* FEM basis order: optional here. The bundle's /nord dataset is the
   * primary source (read by loadCsemInputs into csemParams.nord); im_csem
   * copies it into iparams->nord after loading.  -nord may be supplied as
   * a CLI override.  Sentinel 0 means "not overridden". */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nord",
                               &iparams->nord, &flg));
  if (!flg) iparams->nord = 0;
  if (iparams->nord != 0) {
    PetscCheck(iparams->nord >= 1 && iparams->nord <= 6,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "-nord override out of valid range (must be 1..6, got %"
               PetscInt_FMT ")", iparams->nord);
  }

  /* Inversion control */
  iparams->maxIter            = 80;
  iparams->lbfgsMemory        = 5;
  iparams->lambda             = 0.1;
  iparams->errorLevel         = 0.01;
  iparams->gtol               = 1e-5;
  iparams->rmsTol             = 0.0;   /* 0 => disabled */
  iparams->diagGradientWeight = 0.0;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-inv_max_iter",
                               &iparams->maxIter, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-inv_lbfgs_memory",
                               &iparams->lbfgsMemory, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-inv_lambda",
                                &iparams->lambda, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-inv_error_level",
                                &iparams->errorLevel,
                                &iparams->errorLevelFromCLI));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-inv_gtol",
                                &iparams->gtol, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-inv_rms_tol",
                                &iparams->rmsTol, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-inv_diag_weight",
                                &iparams->diagGradientWeight, NULL));

  /* Fixed material IDs excluded from gradient smoothing.  Default 0
   * means "use whatever the bundle says (or empty if bundle has no
   * /inv_meta/fixed_materials)". */
  iparams->numFixedMaterials       = INV_MAX_FIXED_MATERIALS;
  iparams->fixedMaterialsFromCLI   = PETSC_FALSE;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-inv_fixed_materials",
                                    iparams->fixedMaterials,
                                    &iparams->numFixedMaterials,
                                    &iparams->fixedMaterialsFromCLI));
  if (!iparams->fixedMaterialsFromCLI) iparams->numFixedMaterials = 0;

  /* VTU snapshot interval: write model every N accepted L-BFGS steps */
  iparams->snapshotInterval = 0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-inv_snapshot_interval",
                               &iparams->snapshotInterval, NULL));

  /* Developer-only FD gradient check */
  iparams->fdCheckCells = 0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-inv_dev_fd_check",
                               &iparams->fdCheckCells, NULL));

  /* Diagnostic: disable both smoothers for the whole run */
  iparams->smootherOff = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-inv_no_smoother",
                                &iparams->smootherOff, NULL));

  /* numFreqs/invSources are populated later by setupInversionSources
   * (from the unified bundle's /sources group). */
  iparams->numFreqs = 0;

  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscCall(PetscPrintf(comm, "\n Inversion parameters:\n"));
  PetscCall(PetscPrintf(comm, "   Basis order (nord)  = %" PetscInt_FMT "\n",
                        iparams->nord));
  PetscCall(PetscPrintf(comm, "   Max iterations      = %" PetscInt_FMT "\n",
                        iparams->maxIter));
  PetscCall(PetscPrintf(comm, "   L-BFGS memory (M)   = %" PetscInt_FMT "\n",
                        iparams->lbfgsMemory));
  PetscCall(PetscPrintf(comm, "   Lambda (Tikhonov)   = %g\n",
                        (double)iparams->lambda));
  /* error_level and fixed_materials are printed by
   * loadInversionMetaFromBundle after bundle resolution (so the line
   * reflects the final value used by the kernel, including bundle
   * overrides). */
  if (iparams->rmsTol > 0.0)
    PetscCall(PetscPrintf(comm, "   RMS early-stop      = %g\n",
                          (double)iparams->rmsTol));
  else
    PetscCall(PetscPrintf(comm, "   RMS early-stop      = disabled\n"));
  if (iparams->snapshotInterval > 0)
    PetscCall(PetscPrintf(comm, "   VTU snapshot        = every %" PetscInt_FMT
                          " accepted L-BFGS step(s)\n",
                          iparams->snapshotInterval));
  else
    PetscCall(PetscPrintf(comm, "   VTU snapshot        = disabled\n"));
  if (iparams->fdCheckCells > 0)
    PetscCall(PetscPrintf(comm, "   FD gradient check   = %" PetscInt_FMT
                          " cells (DEV; L-BFGS will be skipped)\n",
                          iparams->fdCheckCells));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* setupInversionSources                                               */
/*                                                                     */
/* Reads multi-frequency inversion sources from the unified bundle's   */
/* /sources group (shared with the forward kernel). Each row in        */
/* /sources/freq is one (freq, dipole) record; the sibling datasets    */
/* {position,current,length,dipAngle,azimuthAngle} hold the dipole     */
/* parameters.                                                         */
/*                                                                     */
/* Uses PetscViewerHDF5 + VecLoad (like loadCsemInputs) because the    */
/* datasets are PETSc Vecs carrying the complex/real marking VecLoad   */
/* needs on a complex-scalar build.                                    */
/*                                                                     */
/* Populates iparams->numFreqs and invSources[].                      */
/* ================================================================== */
PetscErrorCode setupInversionSources(const char *bundleFile,
                                     invParams  *iparams)
{
  PetscFunctionBeginUser;

  PetscCheck(bundleFile && bundleFile[0] != '\0', PETSC_COMM_WORLD,
             PETSC_ERR_ARG_NULL,
             "setupInversionSources: bundleFile is empty.");

  /* Read the unified /sources group (per-entry frequency), shared with the
   * forward kernel. These datasets are PETSc Vecs written by the preprocess,
   * so they carry the complex/real marking VecLoad needs - hence the PETSc
   * HDF5 viewer + loadSelfVecByName here (mirroring loadCsemInputs), rather
   * than the raw H5Dread used previously for the plain-float64 /inv_sources. */
  PetscViewer viewer;
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, bundleFile, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/sources"));

  Vec freqV, posV, curV, lenV, dipV, azV;
  PetscCall(loadSelfVecByName(viewer, "freq",         &freqV));
  PetscCall(loadSelfVecByName(viewer, "position",     &posV));
  PetscCall(loadSelfVecByName(viewer, "current",      &curV));
  PetscCall(loadSelfVecByName(viewer, "length",       &lenV));
  PetscCall(loadSelfVecByName(viewer, "dipAngle",     &dipV));
  PetscCall(loadSelfVecByName(viewer, "azimuthAngle", &azV));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscInt count;
  PetscCall(VecGetSize(freqV, &count));
  PetscCheck(count > 0, PETSC_COMM_WORLD, PETSC_ERR_FILE_READ,
             "/sources/freq is empty in %s", bundleFile);
  PetscCheck(count <= INV_MAX_FREQUENCIES, PETSC_COMM_WORLD, PETSC_ERR_SUP,
             "Too many inversion sources (%" PetscInt_FMT " > max %d); "
             "bump INV_MAX_FREQUENCIES in include/constants.h to raise the cap",
             count, INV_MAX_FREQUENCIES);

  const PetscScalar *freqArr, *posArr, *curArr, *lenArr, *dipArr, *azArr;
  PetscCall(VecGetArrayRead(freqV, &freqArr));
  PetscCall(VecGetArrayRead(posV,  &posArr));
  PetscCall(VecGetArrayRead(curV,  &curArr));
  PetscCall(VecGetArrayRead(lenV,  &lenArr));
  PetscCall(VecGetArrayRead(dipV,  &dipArr));
  PetscCall(VecGetArrayRead(azV,   &azArr));

  for (PetscInt i = 0; i < count; i++) {
    InvCsemSource *s = &iparams->invSources[i];
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

  iparams->numFreqs = count;

  /* Print parsed source data */
  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscCall(PetscPrintf(comm, "\n Inversion sources (from bundle /sources):\n"));
  PetscCall(PetscPrintf(comm, "   Bundle file         = %s\n", bundleFile));
  PetscCall(PetscPrintf(comm, "   Num entries         = %" PetscInt_FMT "\n",
                        iparams->numFreqs));
  for (PetscInt i = 0; i < iparams->numFreqs; i++) {
    InvCsemSource *s = &iparams->invSources[i];
    PetscCall(PetscPrintf(comm,
      "   [%" PetscInt_FMT "] freq=%g Hz  pos=(%g, %g, %g)"
      "  I=%g  L=%g  dip=%g  az=%g\n",
      i + 1, s->freq,
      s->position[0], s->position[1], s->position[2],
      s->current, s->length, s->dipAngle, s->azimuthAngle));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* loadInversionMetaFromBundle                                         */
/*                                                                     */
/* Pulls case-property defaults from the bundle:                       */
/*   /observed @error_level         → iparams->errorLevel              */
/*   /inv_meta/fixed_materials      → iparams->{fixedMaterials,        */
/*                                              numFixedMaterials}     */
/*                                                                     */
/* Honours CLI precedence: if iparams->errorLevelFromCLI is true,      */
/* the CLI value wins and the bundle attribute is ignored (same for    */
/* fixedMaterialsFromCLI).  Bundle entries that are absent leave the   */
/* defaults from readInversionParams alone.                            */
/* ================================================================== */
PetscErrorCode loadInversionMetaFromBundle(const char *bundleFile,
                                            invParams  *iparams)
{
  PetscFunctionBeginUser;

  PetscCheck(bundleFile && bundleFile[0] != '\0', PETSC_COMM_WORLD,
             PETSC_ERR_ARG_NULL,
             "loadInversionMetaFromBundle: bundleFile is empty.");

  MPI_Comm comm = PETSC_COMM_WORLD;
  const char *errLevelOrigin = iparams->errorLevelFromCLI ? "CLI" : "default";
  const char *fixedMatsOrigin = iparams->fixedMaterialsFromCLI ? "CLI" : "default";

  /* Suppress noisy H5 error stack when probing optional entries. */
  H5E_auto2_t old_handler;
  void       *old_client_data;
  H5Eget_auto2(H5E_DEFAULT, &old_handler, &old_client_data);
  H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

  hid_t file = H5Fopen(bundleFile, H5F_ACC_RDONLY, H5P_DEFAULT);
  PetscCheck(file >= 0, comm, PETSC_ERR_FILE_OPEN,
             "Cannot open bundle HDF5 file: %s", bundleFile);

  /* /observed @error_level (optional) */
  if (!iparams->errorLevelFromCLI && H5Lexists(file, "/observed", H5P_DEFAULT) > 0) {
    hid_t grp = H5Gopen2(file, "/observed", H5P_DEFAULT);
    if (grp >= 0 && H5Aexists(grp, "error_level") > 0) {
      hid_t  attr = H5Aopen(grp, "error_level", H5P_DEFAULT);
      double v;
      H5Aread(attr, H5T_NATIVE_DOUBLE, &v);
      H5Aclose(attr);
      iparams->errorLevel = v;
      errLevelOrigin = "bundle";
    }
    if (grp >= 0) H5Gclose(grp);
  }

  /* /inv_meta/fixed_materials (optional) */
  if (!iparams->fixedMaterialsFromCLI &&
      H5Lexists(file, "/inv_meta/fixed_materials", H5P_DEFAULT) > 0) {
    hid_t dset = H5Dopen2(file, "/inv_meta/fixed_materials", H5P_DEFAULT);
    if (dset >= 0) {
      hid_t   sp  = H5Dget_space(dset);
      hsize_t dims[1] = {0};
      H5Sget_simple_extent_dims(sp, dims, NULL);
      PetscInt n = (PetscInt)dims[0];
      PetscCheck(n <= INV_MAX_FIXED_MATERIALS, comm, PETSC_ERR_SUP,
                 "/inv_meta/fixed_materials has %" PetscInt_FMT " entries; "
                 "INV_MAX_FIXED_MATERIALS=%d. Bump the cap in include/constants.h.",
                 n, INV_MAX_FIXED_MATERIALS);
      int *buf;
      PetscCall(PetscMalloc1(n, &buf));
      H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
      for (PetscInt k = 0; k < n; k++) iparams->fixedMaterials[k] = (PetscInt)buf[k];
      iparams->numFixedMaterials = n;
      PetscCall(PetscFree(buf));
      H5Sclose(sp);
      H5Dclose(dset);
      fixedMatsOrigin = "bundle";
    }
  }

  H5Fclose(file);
  H5Eset_auto2(H5E_DEFAULT, old_handler, old_client_data);

  /* Final-value banner (matches the style of readInversionParams) */
  PetscCall(PetscPrintf(comm, "   Error level         = %g  (%s)\n",
                        (double)iparams->errorLevel, errLevelOrigin));
  PetscCall(PetscPrintf(comm, "   Fixed materials     = %" PetscInt_FMT
                        " IDs (%s):", iparams->numFixedMaterials, fixedMatsOrigin));
  for (PetscInt k = 0; k < iparams->numFixedMaterials; k++)
    PetscCall(PetscPrintf(comm, " %" PetscInt_FMT,
                          iparams->fixedMaterials[k]));
  PetscCall(PetscPrintf(comm, "\n"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* loadObservedData                                                    */
/*                                                                     */
/* Reads observed data from the unified bundle's /observed/Ex dataset, */
/* which is a 2-D HDF5 compound complex128 dataset (type {r,i} float64)*/
/* of shape [numFreqs, numReceivers].  Stores as dense Mat dObs        */
/* (PETSC_COMM_SELF) and broadcasts to all ranks.                      */
/*                                                                     */
/* Generated by utils/preprocess.py -mode inverse, which embeds the    */
/* observed-data HDF5 produced by tests/cases/inverse/                  */
/* generate_observed_data.py into the same bundle file consumed by     */
/* loadCsemInputs.                                                     */
/* ================================================================== */
PetscErrorCode loadObservedData(const char *bundleFile,
                                PetscInt    numFreqs,
                                PetscInt    numReceivers,
                                Mat        *dObs)
{
  PetscFunctionBeginUser;

  MPI_Comm     comm = PETSC_COMM_WORLD;
  PetscMPIInt  rank;

  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCheck(bundleFile && bundleFile[0] != '\0', comm, PETSC_ERR_ARG_NULL,
             "loadObservedData: bundleFile is empty.");

  /* Create sequential dense matrix (all ranks) */
  PetscCall(MatCreateDense(PETSC_COMM_SELF, numFreqs, numReceivers,
                           numFreqs, numReceivers, NULL, dObs));
  PetscCall(MatZeroEntries(*dObs));

  /* Only rank 0 reads the HDF5 file, then broadcasts */
  if (rank == 0) {
    hid_t file_id = H5Fopen(bundleFile, H5F_ACC_RDONLY, H5P_DEFAULT);
    PetscCheck(file_id >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "Cannot open bundle HDF5 file: %s", bundleFile);

    /* Open /observed/Ex dataset (complex128 compound type {r, i}) */
    hid_t dset = H5Dopen2(file_id, "/observed/Ex", H5P_DEFAULT);
    PetscCheck(dset >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
               "Cannot find /observed/Ex dataset in %s", bundleFile);

    /* Verify dimensions [numFreqs x numReceivers] */
    hid_t   space = H5Dget_space(dset);
    int     ndims = H5Sget_simple_extent_ndims(space);
    hsize_t dims[2];
    PetscCheck(ndims == 2, PETSC_COMM_SELF, PETSC_ERR_FILE_READ,
               "/observed/Ex must be 2D, got %d dimensions", ndims);
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
        PetscCall(MatSetValue(*dObs, ifre, irec,
                              buf[ifre * numReceivers + irec],
                              INSERT_VALUES));
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
    for (PetscInt k = 0; k < totalElems; k++) bcast[k] = arr[k];
    PetscCall(MatDenseRestoreArrayRead(*dObs, &arr));
  }
  PetscCallMPI(MPI_Bcast(bcast, totalElems, MPIU_SCALAR, 0, comm));

  if (rank != 0) {
    PetscScalar *arr;
    PetscCall(MatDenseGetArray(*dObs, &arr));
    for (PetscInt k = 0; k < totalElems; k++) arr[k] = bcast[k];
    PetscCall(MatDenseRestoreArray(*dObs, &arr));
    PetscCall(MatAssemblyBegin(*dObs, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*dObs, MAT_FINAL_ASSEMBLY));
  }
  PetscCall(PetscFree(bcast));

  PetscCall(PetscPrintf(comm, "\n Observed data (from bundle /observed/Ex):\n"));
  PetscCall(PetscPrintf(comm, "   Bundle file         = %s\n", bundleFile));
  PetscCall(PetscPrintf(comm, "   Frequencies         = %" PetscInt_FMT "\n",
                        numFreqs));
  PetscCall(PetscPrintf(comm, "   Receivers           = %" PetscInt_FMT "\n",
                        numReceivers));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* writeInversionResults                                               */
/*                                                                     */
/* Saves the final inverted model to HDF5.                            */
/*                                                                     */
/* Output file: {output_dir}/{output_filename}.h5                     */
/*   /conductivity       global Vec (3 DOFs/cell: sigma_x,y,z)        */
/*   /log_perturbation  global Vec (X = log(rho) - X0 perturbation)  */
/*   Attributes: provenance metadata, convergence info, RMS history   */
/* ================================================================== */
PetscErrorCode writeInversionResults(const invParams *iparams,
                                      DM               dmConductivity,
                                      Vec              conductivity,
                                      Vec              X,
                                      const PetscReal *allRMS,
                                      PetscInt         numIters,
                                      const char      *reasonStr)
{
  PetscFunctionBeginUser;

  MPI_Comm comm = PetscObjectComm((PetscObject)dmConductivity);

  /* Read output path from PETSc options (set by -output_dir, -output_filename) */
  char outputDir[PETSC_MAX_PATH_LEN]      = "";
  char outputFilename[PETSC_MAX_PATH_LEN] = "";
  PetscBool flg;

  PetscCall(PetscOptionsGetString(NULL, NULL, "-output_dir",
                                  outputDir, sizeof(outputDir), &flg));
  PetscCheck(flg, comm, PETSC_ERR_USER,
             "Option -output_dir is required for writing inversion results");
  PetscCall(PetscOptionsGetString(NULL, NULL, "-output_filename",
                                  outputFilename, sizeof(outputFilename), &flg));
  PetscCheck(flg, comm, PETSC_ERR_USER,
             "Option -output_filename is required for writing inversion results");

  /* Build output file path */
  char outFile[PETSC_MAX_PATH_LEN];
  PetscCall(PetscStrncpy(outFile, outputDir, sizeof(outFile)));
  size_t len = strlen(outFile);
  if (len > 0 && outFile[len - 1] != '/')
    PetscCall(PetscStrlcat(outFile, "/", sizeof(outFile)));
  PetscCall(PetscStrlcat(outFile, outputFilename, sizeof(outFile)));
  PetscCall(PetscStrlcat(outFile, ".h5", sizeof(outFile)));

  PetscCall(PetscPrintf(comm, "\n Writing inversion results:\n"));
  PetscCall(PetscPrintf(comm, "   Output file         = %s\n", outFile));

  /* Scatter local conductivity to global for output */
  Vec globalConductivity;
  PetscCall(DMCreateGlobalVector(dmConductivity, &globalConductivity));
  PetscCall(DMLocalToGlobal(dmConductivity, conductivity,
                            INSERT_VALUES, globalConductivity));

  /* Create plain MPI Vec copies (no DM association) for HDF5 output.
   * The dmConductivity section has numComp=3 but 4 DOFs/cell, so
   * VecView_Plex_HDF5 would fail with a block-size mismatch.
   * Writing through plain Vecs avoids that. */
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

  /* Write provenance attributes */
  char version[50];
  sprintf(version, "%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL,
             "Petgem_version", PETSC_STRING, version));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL,
             "Nord", PETSC_INT, &iparams->nord));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL,
             "Num_frequencies", PETSC_INT, &iparams->numFreqs));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL,
             "Lambda", PETSC_REAL, &iparams->lambda));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL,
             "Error_level", PETSC_REAL, &iparams->errorLevel));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL,
             "Num_iterations", PETSC_INT, &numIters));

  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL,
             "Convergence_reason", PETSC_STRING, reasonStr));

  /* Write RMS history as an MPI Vec so all ranks participate in the
   * collective VecView.  Only rank 0 owns the data (size numIters);
   * all other ranks contribute zero entries. */
  if (numIters > 0) {
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscInt rmsLocalSize = (rank == 0) ? numIters : 0;

    Vec rmsVec;
    PetscCall(VecCreateMPI(comm, rmsLocalSize, numIters, &rmsVec));
    PetscCall(PetscObjectSetName((PetscObject)rmsVec, "rms_history"));

    if (rank == 0) {
      PetscScalar *rArr;
      PetscCall(VecGetArray(rmsVec, &rArr));
      for (PetscInt i = 0; i < numIters; i++)
        rArr[i] = allRMS[i];
      PetscCall(VecRestoreArray(rmsVec, &rArr));
    }

    PetscCall(VecView(rmsVec, viewer));
    PetscCall(VecDestroy(&rmsVec));
  }

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&outRes));
  PetscCall(VecDestroy(&outX));
  PetscCall(VecDestroy(&globalConductivity));

  PetscCall(PetscPrintf(comm, "   Datasets written    = conductivity, "
                        "log_perturbation, rms_history\n"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Copy the first n real parts of a Vec into out[] (no-op if v is NULL).
 * For a 1-DOF/cell vector on dmInversion the first nOwned entries are the
 * owned cells in cellStart..cellEnd order (matches the rho fill below). */
static PetscErrorCode copyOwnedReal(Vec v, PetscInt n, PetscReal *out)
{
  PetscFunctionBeginUser;
  if (v) {
    const PetscScalar *a;
    PetscCall(VecGetArrayRead(v, &a));
    for (PetscInt i = 0; i < n; i++) out[i] = PetscRealPart(a[i]);
    PetscCall(VecRestoreArrayRead(v, &a));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Write one cell-centered Float64 scalar field as an ascii VTU DataArray
 * (rank-0 only; plain void since it just emits text). */
static void writeVtuCellField(FILE *fp, const char *name,
                              const PetscReal *a, PetscInt n)
{
  fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", name);
  for (PetscInt i = 0; i < n; i++) fprintf(fp, "%.9g\n", (double)a[i]);
  fprintf(fp, "        </DataArray>\n");
}

/* ================================================================== */
/* writeInversionSnapshotVTU                                           */
/*                                                                     */
/* Writes a multi-field VTU snapshot of the current inversion state   */
/* (MATLAB parity with Ex_inv.m: InvRho + dfdm0 + DfDM + X_guangguaqian */
/* + X_guanghuahou, bundled into a single file).  Called after every  */
/* snapshotInterval accepted L-BFGS steps when snapshotInterval > 0.  */
/*                                                                     */
/* Fields (all cell-centered, 1 scalar per cell):                     */
/*   rho_ohm_m       = 1/sigma_x                     (MATLAB InvRho) */
/*   X_pre_smooth    = X before Gauss-Seidel sweep   (X_guangguaqian) */
/*   X_post_smooth   = X after Gauss-Seidel sweep    (X_guanghuahou) */
/*   DfDm_raw        = gradient before fixed-zero and smoothing (dfdm0) */
/*   DfDm_final      = gradient after smoothing + regularization (DfDM) */
/*                                                                     */
/* Output filename: {output_dir}/inv_model_iter{N:05d}.vtu            */
/* The output_dir is read from the -output_dir PETSc option.          */
/*                                                                     */
/* Parallel output: all five per-cell fields and the cell vertex       */
/* coordinates are gathered to rank 0, which writes a SINGLE-piece     */
/* ascii VTU. This avoids PETSc's VTK viewer emitting one <Piece> per  */
/* rank into one .vtu (a non-standard multi-piece serial file that     */
/* ParaView/meshio misread as per-partition fragments - the cause of   */
/* the apparent "boundary artifacts"). Cells are written exploded (each */
/* tet carries its own 4 vertices) so no global point renumbering is   */
/* needed; cell-data colouring is unaffected by the duplicated points. */
/* ================================================================== */
PetscErrorCode writeInversionSnapshotVTU(const InversionContext *ctx,
                                          PetscInt                acceptedIter)
{
  PetscFunctionBeginUser;

  MPI_Comm comm = PetscObjectComm((PetscObject)ctx->dm);

  /* Read output directory from PETSc options */
  char outputDir[PETSC_MAX_PATH_LEN] = "";
  PetscBool flg;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-output_dir",
                                  outputDir, sizeof(outputDir), &flg));
  if (!flg) PetscCall(PetscStrncpy(outputDir, "./", sizeof(outputDir)));

  /* Ensure trailing slash */
  size_t dirLen = strlen(outputDir);
  if (dirLen > 0 && outputDir[dirLen - 1] != '/')
    PetscCall(PetscStrlcat(outputDir, "/", sizeof(outputDir)));

  /* Build output filename */
  char filename[PETSC_MAX_PATH_LEN];
  PetscCall(PetscSNPrintf(filename, sizeof(filename),
    "%sinv_model_iter%05" PetscInt_FMT ".vtu", outputDir, acceptedIter));

  PetscMPIInt rank, size;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscInt nOwned = ctx->grid.cellEnd - ctx->grid.cellStart;

  /* ---- Per-owned-cell field values + exploded vertex coordinates ---- */
  PetscReal *rho, *xpre, *xpost, *dfraw, *dffin, *coords;
  PetscCall(PetscCalloc1(nOwned, &rho));
  PetscCall(PetscCalloc1(nOwned, &xpre));
  PetscCall(PetscCalloc1(nOwned, &xpost));
  PetscCall(PetscCalloc1(nOwned, &dfraw));
  PetscCall(PetscCalloc1(nOwned, &dffin));
  PetscCall(PetscCalloc1(12 * nOwned, &coords));

  /* rho = 1/sigma_x */
  {
    PetscSection resSec;
    PetscCall(DMGetLocalSection(ctx->dmConductivity, &resSec));
    const PetscScalar *sArr;
    PetscCall(VecGetArrayRead(ctx->conductivity, &sArr));
    for (PetscInt i = ctx->grid.cellStart; i < ctx->grid.cellEnd; i++) {
      PetscInt li = i - ctx->grid.cellStart, off;
      PetscCall(PetscSectionGetOffset(resSec, i, &off));
      PetscReal sigma = PetscRealPart(sArr[off]); /* component 0 = sigma_x */
      rho[li] = (sigma > 0.0) ? 1.0 / sigma : 0.0;
    }
    PetscCall(VecRestoreArrayRead(ctx->conductivity, &sArr));
  }

  /* Diagnostic fields (NULL when snapshots disabled -> left zero). */
  PetscCall(copyOwnedReal(ctx->XPreSmooth,  nOwned, xpre));
  PetscCall(copyOwnedReal(ctx->XPostSmooth, nOwned, xpost));
  PetscCall(copyOwnedReal(ctx->DfDmRaw,     nOwned, dfraw));
  PetscCall(copyOwnedReal(ctx->DfDmFinal,   nOwned, dffin));

  /* Exploded tet vertices: 4 vertices (12 reals) per owned cell. */
  for (PetscInt i = ctx->grid.cellStart; i < ctx->grid.cellEnd; i++) {
    PetscInt li = i - ctx->grid.cellStart;
    Cell cell;
    PetscCall(extractCellCoordinates(ctx->dm, i, &cell));
    for (PetscInt k = 0; k < 12; k++) coords[12 * li + k] = cell.coordinates[k];
  }

  /* ---- Gather everything to rank 0 ---- */
  PetscMPIInt nLoc = (PetscMPIInt)nOwned;
  PetscMPIInt *cnt = NULL, *dsp = NULL, *cntC = NULL, *dspC = NULL;
  if (rank == 0) {
    PetscCall(PetscMalloc4(size, &cnt, size, &dsp, size, &cntC, size, &dspC));
  }
  PetscCallMPI(MPI_Gather(&nLoc, 1, MPI_INT, cnt, 1, MPI_INT, 0, comm));

  PetscInt   Ntot = 0;
  PetscReal *rhoA = NULL, *xpreA = NULL, *xpostA = NULL,
            *dfrawA = NULL, *dffinA = NULL, *coordsA = NULL;
  if (rank == 0) {
    dsp[0] = 0; dspC[0] = 0; cntC[0] = 12 * cnt[0];
    for (PetscMPIInt r = 1; r < size; r++) {
      cntC[r] = 12 * cnt[r];
      dsp[r]  = dsp[r - 1] + cnt[r - 1];
      dspC[r] = dspC[r - 1] + cntC[r - 1];
    }
    Ntot = dsp[size - 1] + cnt[size - 1];
    PetscCall(PetscMalloc1(Ntot, &rhoA));
    PetscCall(PetscMalloc1(Ntot, &xpreA));
    PetscCall(PetscMalloc1(Ntot, &xpostA));
    PetscCall(PetscMalloc1(Ntot, &dfrawA));
    PetscCall(PetscMalloc1(Ntot, &dffinA));
    PetscCall(PetscMalloc1(12 * Ntot, &coordsA));
  }
  PetscCallMPI(MPI_Gatherv(rho,    nLoc, MPIU_REAL, rhoA,    cnt, dsp, MPIU_REAL, 0, comm));
  PetscCallMPI(MPI_Gatherv(xpre,   nLoc, MPIU_REAL, xpreA,   cnt, dsp, MPIU_REAL, 0, comm));
  PetscCallMPI(MPI_Gatherv(xpost,  nLoc, MPIU_REAL, xpostA,  cnt, dsp, MPIU_REAL, 0, comm));
  PetscCallMPI(MPI_Gatherv(dfraw,  nLoc, MPIU_REAL, dfrawA,  cnt, dsp, MPIU_REAL, 0, comm));
  PetscCallMPI(MPI_Gatherv(dffin,  nLoc, MPIU_REAL, dffinA,  cnt, dsp, MPIU_REAL, 0, comm));
  PetscCallMPI(MPI_Gatherv(coords, 12 * nLoc, MPIU_REAL, coordsA, cntC, dspC, MPIU_REAL, 0, comm));

  /* ---- Rank 0 writes a single-piece ascii VTU ---- */
  if (rank == 0) {
    FILE *fp = fopen(filename, "w");
    PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "writeInversionSnapshotVTU: cannot open %s", filename);
    PetscInt npts = 4 * Ntot;
    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <UnstructuredGrid>\n");
    fprintf(fp, "    <Piece NumberOfPoints=\"%" PetscInt_FMT "\" NumberOfCells=\"%" PetscInt_FMT "\">\n",
            npts, Ntot);

    fprintf(fp, "      <Points>\n");
    fprintf(fp, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (PetscInt c = 0; c < Ntot; c++)
      for (PetscInt v = 0; v < 4; v++)
        fprintf(fp, "%.9g %.9g %.9g\n",
                (double)coordsA[12 * c + 3 * v + 0],
                (double)coordsA[12 * c + 3 * v + 1],
                (double)coordsA[12 * c + 3 * v + 2]);
    fprintf(fp, "        </DataArray>\n      </Points>\n");

    fprintf(fp, "      <Cells>\n");
    fprintf(fp, "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
    for (PetscInt p = 0; p < npts; p++) fprintf(fp, "%" PetscInt_FMT " ", p);
    fprintf(fp, "\n        </DataArray>\n");
    fprintf(fp, "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
    for (PetscInt c = 1; c <= Ntot; c++) fprintf(fp, "%" PetscInt_FMT " ", 4 * c);
    fprintf(fp, "\n        </DataArray>\n");
    fprintf(fp, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
    for (PetscInt c = 0; c < Ntot; c++) fprintf(fp, "10 ");   /* VTK_TETRA */
    fprintf(fp, "\n        </DataArray>\n      </Cells>\n");

    fprintf(fp, "      <CellData Scalars=\"rho_ohm_m\">\n");
    writeVtuCellField(fp, "rho_ohm_m",     rhoA,   Ntot);
    writeVtuCellField(fp, "X_pre_smooth",  xpreA,  Ntot);
    writeVtuCellField(fp, "X_post_smooth", xpostA, Ntot);
    writeVtuCellField(fp, "DfDm_raw",      dfrawA, Ntot);
    writeVtuCellField(fp, "DfDm_final",    dffinA, Ntot);
    fprintf(fp, "      </CellData>\n");

    fprintf(fp, "    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n");
    fclose(fp);

    PetscCall(PetscFree4(cnt, dsp, cntC, dspC));
    PetscCall(PetscFree(rhoA));
    PetscCall(PetscFree(xpreA));
    PetscCall(PetscFree(xpostA));
    PetscCall(PetscFree(dfrawA));
    PetscCall(PetscFree(dffinA));
    PetscCall(PetscFree(coordsA));
  }

  PetscCall(PetscFree(rho));
  PetscCall(PetscFree(xpre));
  PetscCall(PetscFree(xpost));
  PetscCall(PetscFree(dfraw));
  PetscCall(PetscFree(dffin));
  PetscCall(PetscFree(coords));

  PetscFunctionReturn(PETSC_SUCCESS);
}
