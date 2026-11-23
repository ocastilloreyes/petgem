/*
 * Filename: test_csem_background.c
 * Author: PETGEM test suite
 * Date: 2026-11-20
 *
 * Description:
 * Formulation options and background model of the secondary formulation
 * (csem_primary.c):
 *   - readCsemParams defaults and -csem_formulation, -csem_primary,
 *     -csem_background_file;
 *   - loadCsemBackground of layered, whole-space and per-material backgrounds,
 *     with the default type (layered if present) and an explicit -csem_primary;
 *   - rejection of a missing group, a missing requested model, a finite
 *     layer_ztop[0], increasing layer tops, a layer_sigma of wrong size and a
 *     non-positive material conductivity.
 * The /background files are written with VecView, as utils/functions.py does.
 *
 * Usage: test_csem_background   Exit 0 iff all checks pass.
 */
#include "csem_primary.h"
#include "petgem_test.h"
#include <petscviewerhdf5.h>
#include <string.h>
#include <unistd.h>

static char tmpFile[PETSC_MAX_PATH_LEN];

/* Writes n reals as the Vec /background/<name> of an open viewer. */
static void writeVec(PetscViewer viewer, const char *name, PetscInt n, const PetscReal values[]) {
  Vec          v;
  PetscScalar *a;
  PetscCallAbort(PETSC_COMM_SELF, VecCreateSeq(PETSC_COMM_SELF, n, &v));
  PetscCallAbort(PETSC_COMM_SELF, VecGetArray(v, &a));
  for (PetscInt i = 0; i < n; i++) a[i] = values[i];
  PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(v, &a));
  PetscCallAbort(PETSC_COMM_SELF, PetscObjectSetName((PetscObject)v, name));
  PetscCallAbort(PETSC_COMM_SELF, VecView(v, viewer));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&v));
}

/* Writes tmpFile with the given datasets; numLayers = 0 or numMaterials = 0 omits them. */
static void writeBackground(PetscInt numLayers, const PetscReal ztop[], const PetscReal layerSigma[], PetscInt numMaterials,
                            const PetscReal materialSigma[]) {
  PetscViewer viewer;
  PetscCallAbort(PETSC_COMM_SELF, PetscViewerHDF5Open(PETSC_COMM_SELF, tmpFile, FILE_MODE_WRITE, &viewer));
  PetscCallAbort(PETSC_COMM_SELF, PetscViewerHDF5PushGroup(viewer, numLayers || numMaterials ? "/background" : "/other"));
  if (numLayers) {
    writeVec(viewer, "layer_ztop", numLayers, ztop);
    writeVec(viewer, "layer_sigma", 2 * numLayers, layerSigma);
  }
  if (numMaterials) writeVec(viewer, "material_sigma", 3 * numMaterials, materialSigma);
  if (!numLayers && !numMaterials) writeVec(viewer, "dummy", 1, ztop);
  PetscCallAbort(PETSC_COMM_SELF, PetscViewerHDF5PopGroup(viewer));
  PetscCallAbort(PETSC_COMM_SELF, PetscViewerDestroy(&viewer));
}

/* CsemParams that read tmpFile, with an optional explicit -csem_primary. */
static CsemParams paramsFor(PetscBool typeSet, CsemPrimaryType type) {
  CsemParams p;
  p.formulation    = CSEM_FORMULATION_SECONDARY;
  p.primaryType    = type;
  p.primaryTypeSet = typeSet;
  PetscCallAbort(PETSC_COMM_SELF, PetscStrncpy(p.backgroundFile, tmpFile, sizeof(p.backgroundFile)));
  return p;
}

/* PETSC_TRUE if loadCsemBackground fails on tmpFile. */
static PetscBool loadFails(PetscBool typeSet, CsemPrimaryType type) {
  const CsemParams p = paramsFor(typeSet, type);
  CsemBackground   bg;
  PetscErrorCode   ierr;
  PetscCallAbort(PETSC_COMM_SELF, PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = loadCsemBackground(&p, &bg);
  PetscCallAbort(PETSC_COMM_SELF, PetscPopErrorHandler());
  PetscCallAbort(PETSC_COMM_SELF, destroyCsemBackground(&bg));
  return ierr != PETSC_SUCCESS ? PETSC_TRUE : PETSC_FALSE;
}

static const PetscReal layerZtop[5]     = {INFINITY, 0.0, -1000.0, -2000.0, -2100.0};
static const PetscReal layerSigma[10]   = {1e-8, 1e-8, 3.3, 3.3, 1.0, 0.5, 0.01, 0.01, 1.0, 0.5};
static const PetscReal materialSigma[9] = {1e-8, 1e-8, 1e-8, 3.3, 3.3, 3.3, 1.0, 1.0, 0.5};

static void check_params(void) {
  PetgemParams params;
  CsemParams   csem;

  PetscCallAbort(PETSC_COMM_SELF, PetscStrncpy(params.inputFile, "bundle.h5", sizeof(params.inputFile)));
  PetscCallAbort(PETSC_COMM_SELF, readCsemParams(&params, &csem));
  PT_CHECK(csem.formulation == CSEM_FORMULATION_TOTAL, "default formulation is not total");
  PT_CHECK(!csem.primaryTypeSet, "-csem_primary reported as set by default");
  PT_CHECK(!strcmp(csem.backgroundFile, "bundle.h5"), "default background file %s, expected the input bundle", csem.backgroundFile);

  PetscCallAbort(PETSC_COMM_SELF, PetscOptionsSetValue(NULL, "-csem_formulation", "secondary"));
  PetscCallAbort(PETSC_COMM_SELF, PetscOptionsSetValue(NULL, "-csem_primary", "model"));
  PetscCallAbort(PETSC_COMM_SELF, PetscOptionsSetValue(NULL, "-csem_background_file", "background.h5"));
  PetscCallAbort(PETSC_COMM_SELF, readCsemParams(&params, &csem));
  PT_CHECK(csem.formulation == CSEM_FORMULATION_SECONDARY, "-csem_formulation secondary not read");
  PT_CHECK(csem.primaryTypeSet && csem.primaryType == CSEM_PRIMARY_MODEL, "-csem_primary model not read");
  PT_CHECK(!strcmp(csem.backgroundFile, "background.h5"), "-csem_background_file read as %s", csem.backgroundFile);
  PetscCallAbort(PETSC_COMM_SELF, PetscOptionsClearValue(NULL, "-csem_formulation"));
  PetscCallAbort(PETSC_COMM_SELF, PetscOptionsClearValue(NULL, "-csem_primary"));
  PetscCallAbort(PETSC_COMM_SELF, PetscOptionsClearValue(NULL, "-csem_background_file"));
}

static void check_layered(void) {
  CsemParams     p;
  CsemBackground bg;

  /* Both models present: layered by default */
  writeBackground(5, layerZtop, layerSigma, 3, materialSigma);
  p = paramsFor(PETSC_FALSE, CSEM_PRIMARY_LAYERED);
  PetscCallAbort(PETSC_COMM_SELF, loadCsemBackground(&p, &bg));
  PT_CHECK(bg.type == CSEM_PRIMARY_LAYERED, "default type with both models is not layered");
  PT_CHECK(bg.layered.numLayers == 5, "numLayers = %d, expected 5", (int)bg.layered.numLayers);
  for (PetscInt k = 0; k < 5; k++) {
    if (k > 0)
      PT_CHECK(bg.layered.interfaces[k - 1] == layerZtop[k], "interface %d = %g", (int)k - 1, (double)bg.layered.interfaces[k - 1]);
    PT_CHECK(bg.layered.sigmaH[k] == layerSigma[2 * k] && bg.layered.sigmaV[k] == layerSigma[2 * k + 1], "layer %d sigma = (%g, %g)",
             (int)k, (double)bg.layered.sigmaH[k], (double)bg.layered.sigmaV[k]);
  }
  PetscCallAbort(PETSC_COMM_SELF, destroyCsemBackground(&bg));

  /* Both models present, -csem_primary model */
  p = paramsFor(PETSC_TRUE, CSEM_PRIMARY_MODEL);
  PetscCallAbort(PETSC_COMM_SELF, loadCsemBackground(&p, &bg));
  PT_CHECK(bg.type == CSEM_PRIMARY_MODEL && bg.numMaterials == 3, "-csem_primary model: type %d, %d materials", (int)bg.type,
           (int)bg.numMaterials);
  for (PetscInt i = 0; i < 9; i++)
    PT_CHECK(bg.materialSigma[i] == materialSigma[i], "materialSigma[%d] = %g", (int)i, (double)bg.materialSigma[i]);
  PetscCallAbort(PETSC_COMM_SELF, destroyCsemBackground(&bg));

  /* Whole space: one layer */
  writeBackground(1, layerZtop, &layerSigma[2], 0, NULL);
  p = paramsFor(PETSC_FALSE, CSEM_PRIMARY_LAYERED);
  PetscCallAbort(PETSC_COMM_SELF, loadCsemBackground(&p, &bg));
  PT_CHECK(bg.type == CSEM_PRIMARY_LAYERED && bg.layered.numLayers == 1 && bg.layered.sigmaH[0] == 3.3, "whole space not loaded");
  PetscCallAbort(PETSC_COMM_SELF, destroyCsemBackground(&bg));
  PT_CHECK(loadFails(PETSC_TRUE, CSEM_PRIMARY_MODEL), "-csem_primary model accepted without /background/material_sigma");

  /* Materials only: model by default */
  writeBackground(0, NULL, NULL, 3, materialSigma);
  p = paramsFor(PETSC_FALSE, CSEM_PRIMARY_LAYERED);
  PetscCallAbort(PETSC_COMM_SELF, loadCsemBackground(&p, &bg));
  PT_CHECK(bg.type == CSEM_PRIMARY_MODEL, "default type with materials only is not model");
  PetscCallAbort(PETSC_COMM_SELF, destroyCsemBackground(&bg));
  PT_CHECK(loadFails(PETSC_TRUE, CSEM_PRIMARY_LAYERED), "-csem_primary layered accepted without /background/layer_sigma");
}

static void check_rejected(void) {
  const PetscReal finiteTop[3] = {100.0, 0.0, -100.0}, increasing[3] = {INFINITY, -100.0, 0.0};
  const PetscReal badMaterial[6] = {1.0, 1.0, 1.0, 0.0, 1.0, 1.0};
  PetscViewer     viewer;

  writeBackground(0, layerZtop, NULL, 0, NULL);
  PT_CHECK(loadFails(PETSC_FALSE, CSEM_PRIMARY_LAYERED), "file without /background accepted");
  writeBackground(3, finiteTop, layerSigma, 0, NULL);
  PT_CHECK(loadFails(PETSC_FALSE, CSEM_PRIMARY_LAYERED), "finite layer_ztop[0] accepted");
  writeBackground(3, increasing, layerSigma, 0, NULL);
  PT_CHECK(loadFails(PETSC_FALSE, CSEM_PRIMARY_LAYERED), "increasing layer tops accepted");
  writeBackground(0, NULL, NULL, 2, badMaterial);
  PT_CHECK(loadFails(PETSC_FALSE, CSEM_PRIMARY_LAYERED), "zero material conductivity accepted");

  /* layer_sigma with 3 entries for 2 layers */
  PetscCallAbort(PETSC_COMM_SELF, PetscViewerHDF5Open(PETSC_COMM_SELF, tmpFile, FILE_MODE_WRITE, &viewer));
  PetscCallAbort(PETSC_COMM_SELF, PetscViewerHDF5PushGroup(viewer, "/background"));
  writeVec(viewer, "layer_ztop", 2, layerZtop);
  writeVec(viewer, "layer_sigma", 3, layerSigma);
  PetscCallAbort(PETSC_COMM_SELF, PetscViewerHDF5PopGroup(viewer));
  PetscCallAbort(PETSC_COMM_SELF, PetscViewerDestroy(&viewer));
  PT_CHECK(loadFails(PETSC_FALSE, CSEM_PRIMARY_LAYERED), "layer_sigma of wrong size accepted");
}

int main(int argc, char **argv) {
  char tmpDir[PETSC_MAX_PATH_LEN];

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(PetscGetTmp(PETSC_COMM_SELF, tmpDir, sizeof(tmpDir)));
  PetscCall(PetscSNPrintf(tmpFile, sizeof(tmpFile), "%s/petgem_test_csem_background_%d.h5", tmpDir, (int)getpid()));

  check_params();
  check_layered();
  check_rejected();

  remove(tmpFile);
  int rc = pt_report("test_csem_background");
  PetscCall(PetscFinalize());
  return rc;
}
