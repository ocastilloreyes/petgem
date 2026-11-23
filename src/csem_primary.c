/*
 * Filename: csem_primary.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-11-20
 *
 * Description:
 * Field formulation of fm.csem and im.csem (total or secondary), the
 * background model, the primary field E_p,h and the secondary right-hand side
 * (csem_primary.h).
 */

/* PETSc libraries */
#include <petscviewerhdf5.h>

/* PETGEM functions */
#include "assembly.h"
#include "common.h"
#include "constants.h"
#include "csem.h"
#include "csem_layered.h"
#include "csem_primary.h"
#include "fem.h"
#include "grid.h"
#include "io.h"
#include "receiver_interp.h"
#include "solver.h"

/* Relative conductivity difference below which a cell matches the background. */
#define CSEM_SIGMA_RTOL 1.0e-10

/* Layered model from /background/layer_ztop and /background/layer_sigma (viewer at /background). */
static PetscErrorCode loadBackgroundLayers(PetscViewer viewer, const char *file, LayeredModel *model) {
  PetscFunctionBeginUser;

  Vec                ztop, sigma;
  const PetscScalar *z, *s;
  PetscInt           n, ns;
  PetscReal         *interfaces, *sigmaH, *sigmaV;

  PetscCall(loadSerialVec(viewer, "layer_ztop", &ztop));
  PetscCall(loadSerialVec(viewer, "layer_sigma", &sigma));
  PetscCall(VecGetSize(ztop, &n));
  PetscCall(VecGetSize(sigma, &ns));
  PetscCheck(n >= 1 && ns == 2 * n, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
             "loadCsemBackground: %s: /background/layer_sigma has %" PetscInt_FMT " entries, expected 2 x %" PetscInt_FMT " layers", file,
             ns, n);

  PetscCall(VecGetArrayRead(ztop, &z));
  PetscCall(VecGetArrayRead(sigma, &s));
  PetscCheck(PetscIsInfReal(PetscRealPart(z[0])) && PetscRealPart(z[0]) > 0.0, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
             "loadCsemBackground: %s: /background/layer_ztop[0] = %g (expected +inf)", file, (double)PetscRealPart(z[0]));
  PetscCall(PetscMalloc3(n - 1, &interfaces, n, &sigmaH, n, &sigmaV));
  for (PetscInt k = 0; k < n; k++) {
    if (k > 0) interfaces[k - 1] = PetscRealPart(z[k]);
    sigmaH[k] = PetscRealPart(s[2 * k]);
    sigmaV[k] = PetscRealPart(s[2 * k + 1]);
  }
  PetscCall(VecRestoreArrayRead(sigma, &s));
  PetscCall(VecRestoreArrayRead(ztop, &z));

  PetscCall(createLayeredModel(n, interfaces, sigmaH, sigmaV, model));

  PetscCall(PetscFree3(interfaces, sigmaH, sigmaV));
  PetscCall(VecDestroy(&sigma));
  PetscCall(VecDestroy(&ztop));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Conductivity per material from /background/material_sigma (viewer at /background). */
static PetscErrorCode loadBackgroundMaterials(PetscViewer viewer, const char *file, CsemBackground *background) {
  PetscFunctionBeginUser;

  Vec                sigma;
  const PetscScalar *s;
  PetscInt           n;

  PetscCall(loadSerialVec(viewer, "material_sigma", &sigma));
  PetscCall(VecGetSize(sigma, &n));
  PetscCheck(n > 0 && n % 3 == 0, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
             "loadCsemBackground: %s: /background/material_sigma has %" PetscInt_FMT " entries, expected 3 per material", file, n);

  background->numMaterials = n / 3;
  PetscCall(PetscMalloc1(n, &background->materialSigma));
  PetscCall(VecGetArrayRead(sigma, &s));
  for (PetscInt i = 0; i < n; i++) background->materialSigma[i] = PetscRealPart(s[i]);
  PetscCall(VecRestoreArrayRead(sigma, &s));
  PetscCall(VecDestroy(&sigma));

  for (PetscInt m = 0; m < background->numMaterials; m++) {
    const PetscReal *sm = &background->materialSigma[3 * m];
    PetscCheck(sm[0] > 0.0 && sm[1] > 0.0 && sm[2] > 0.0, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
               "loadCsemBackground: %s: material %" PetscInt_FMT " has background sigma = (%g, %g, %g) (expected > 0)", file, m,
               (double)sm[0], (double)sm[1], (double)sm[2]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Reads the formulation options -csem_formulation, -csem_primary and -csem_background_file.
 *
 * -csem_background_file defaults to the input bundle.
 */
PetscErrorCode readCsemParams(const PetgemParams *params, CsemParams *csemParams) {
  PetscFunctionBeginUser;

  const char *formulations[] = {"total", "secondary"};
  const char *primaries[]    = {"layered", "model"};
  PetscInt    formulation    = CSEM_FORMULATION_TOTAL;
  PetscInt    primary        = CSEM_PRIMARY_LAYERED;

  csemParams->backgroundFile[0] = '\0';
  csemParams->primaryTypeSet    = PETSC_FALSE;

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "CSEM: formulation options (optional)", "PETGEM");
  PetscCall(PetscOptionsEList("-csem_formulation", "Field formulation: 'total' (E) or 'secondary' (E - E_p of a background model)", "csem",
                              formulations, 2, formulations[formulation], &formulation, NULL));
  PetscCall(PetscOptionsEList("-csem_primary",
                              "Background model of the secondary formulation: 'layered' (/background/layer_*) or 'model' "
                              "(/background/material_sigma); default: layered if present",
                              "csem", primaries, 2, primaries[primary], &primary, &csemParams->primaryTypeSet));
  PetscCall(PetscOptionsString("-csem_background_file", "HDF5 file with the /background group (default: the input bundle)", "csem",
                               csemParams->backgroundFile, csemParams->backgroundFile, sizeof(csemParams->backgroundFile), NULL));
  PetscOptionsEnd();

  csemParams->formulation = (CsemFormulation)formulation;
  csemParams->primaryType = (CsemPrimaryType)primary;
  if (csemParams->backgroundFile[0] == '\0')
    PetscCall(PetscStrncpy(csemParams->backgroundFile, params->inputFile, sizeof(csemParams->backgroundFile)));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Loads the background model from the /background group.
 *
 * Without -csem_primary the type is layered if /background/layer_sigma
 * exists and model otherwise.
 */
PetscErrorCode loadCsemBackground(const CsemParams *csemParams, CsemBackground *background) {
  PetscFunctionBeginUser;

  const char *file         = csemParams->backgroundFile;
  PetscBool   hasLayers    = PETSC_FALSE;
  PetscBool   hasMaterials = PETSC_FALSE;
  PetscViewer viewer;

  PetscCall(PetscMemzero(background, sizeof(*background)));
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, file, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerHDF5HasDataset(viewer, "/background/layer_sigma", &hasLayers));
  PetscCall(PetscViewerHDF5HasDataset(viewer, "/background/material_sigma", &hasMaterials));
  PetscCheck(hasLayers || hasMaterials, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
             "loadCsemBackground: %s has no /background/layer_sigma or /background/material_sigma (preprocess with "
             "-background_layers_file or -background_sigma_file)",
             file);

  background->type = csemParams->primaryTypeSet ? csemParams->primaryType : (hasLayers ? CSEM_PRIMARY_LAYERED : CSEM_PRIMARY_MODEL);
  PetscCheck(background->type != CSEM_PRIMARY_LAYERED || hasLayers, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
             "loadCsemBackground: -csem_primary layered needs /background/layer_sigma, absent from %s", file);
  PetscCheck(background->type != CSEM_PRIMARY_MODEL || hasMaterials, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
             "loadCsemBackground: -csem_primary model needs /background/material_sigma, absent from %s", file);

  PetscCall(PetscViewerHDF5PushGroup(viewer, "/background"));
  if (background->type == CSEM_PRIMARY_LAYERED)
    PetscCall(loadBackgroundLayers(viewer, file, &background->layered));
  else
    PetscCall(loadBackgroundMaterials(viewer, file, background));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Frees a background model loaded by loadCsemBackground().
 */
PetscErrorCode destroyCsemBackground(CsemBackground *background) {
  PetscFunctionBeginUser;
  if (background->type == CSEM_PRIMARY_LAYERED) PetscCall(destroyLayeredModel(&background->layered));
  PetscCall(PetscFree(background->materialSigma));
  background->numMaterials = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Dipole moment I L d of a transmitter, d from its azimuth and dip. */
static PetscErrorCode sourceMoment(const CsemSource *src, PetscReal moment[NUM_DIMENSIONS]) {
  PetscFunctionBeginUser;
  PetscReal direction[NUM_DIMENSIONS];
  PetscCall(computeVectorRotation(src->azimuthAngle, src->dipAngle, direction));
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) moment[d] = direction[d] * (src->current * src->length);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Layered dipoles of the sources of one set. */
static PetscErrorCode createSetDipoles(const LayeredModel *model, const CsemSourceSet *set, LayeredDipole **dipoles) {
  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(set->numSources, dipoles));
  for (PetscInt k = 0; k < set->numSources; k++) {
    PetscReal moment[NUM_DIMENSIONS];
    PetscCall(sourceMoment(&set->sourceArray[k], moment));
    PetscCall(createLayeredDipole(model, set->freq, set->sourceArray[k].position, moment, &(*dipoles)[k]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Frees the dipoles of createSetDipoles. */
static PetscErrorCode destroySetDipoles(PetscInt numSources, LayeredDipole **dipoles) {
  PetscFunctionBeginUser;
  for (PetscInt k = 0; k < numSources; k++) PetscCall(destroyLayeredDipole(&(*dipoles)[k]));
  PetscCall(PetscFree(*dipoles));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* E of the dipoles in ctx at the points: values[(f * numPoints + q) * 3 + d]. */
static PetscErrorCode layeredField(PetscInt numPoints, const PetscReal points[], PetscInt numFields, PetscScalar values[], void *ctx) {
  PetscFunctionBeginUser;
  const LayeredDipole *dipoles = (const LayeredDipole *)ctx;
  for (PetscInt f = 0; f < numFields; f++)
    PetscCall(evaluateLayeredDipole(&dipoles[f], numPoints, points, NULL, &values[f * numPoints * NUM_DIMENSIONS], NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PETSC_TRUE if any component differs from the background by more than CSEM_SIGMA_RTOL. */
static PetscBool differsFromBackground(const PetscReal sigma[NUM_CONDUCTIVITY_COMPONENTS],
                                       const PetscReal sigmaB[NUM_CONDUCTIVITY_COMPONENTS]) {
  for (PetscInt d = 0; d < NUM_CONDUCTIVITY_COMPONENTS; d++)
    if (PetscAbsReal(sigma[d] - sigmaB[d]) > CSEM_SIGMA_RTOL * PetscMax(PetscAbsReal(sigma[d]), PetscAbsReal(sigmaB[d]))) return PETSC_TRUE;
  return PETSC_FALSE;
}

/**
 * @brief Sets the background conductivity of every cell and the cells with sigma != sigma_b.
 *
 * The background Vec is created on the DM of the conductivity Vec.
 */
PetscErrorCode createCsemPrimary(const CsemBackground *background, const DM dm, const Grid *grid, Vec conductivity, Vec materialsId,
                                 CsemPrimary *primary) {
  PetscFunctionBeginUser;

  MPI_Comm     comm        = PetscObjectComm((PetscObject)dm);
  PetscInt     counts[3]   = {0, 0, 0}; /* anomaly, crossing, material outside the background table */
  PetscInt     badMaterial = PETSC_INT_MIN;
  DM           dmSigma, dmMat = NULL;
  PetscSection section;
  PetscScalar *sb;

  PetscCall(PetscMemzero(primary, sizeof(*primary)));
  primary->type       = background->type;
  primary->background = background;
  PetscCheck(background->type != CSEM_PRIMARY_MODEL || materialsId, comm, PETSC_ERR_ARG_NULL,
             "createCsemPrimary: the model background needs the material id Vec");

  PetscCall(VecGetDM(conductivity, &dmSigma));
  if (materialsId) PetscCall(VecGetDM(materialsId, &dmMat));
  PetscCall(DMCreateLocalVector(dmSigma, &primary->sigmaBackground));
  PetscCall(VecZeroEntries(primary->sigmaBackground));
  PetscCall(DMGetLocalSection(dmSigma, &section));
  PetscCall(PetscCalloc1(grid->cellEnd - grid->cellStart, &primary->cells));

  PetscCall(VecGetArray(primary->sigmaBackground, &sb));
  for (PetscInt c = grid->cellStart; c < grid->cellEnd; c++) {
    PetscReal sigmaB[NUM_CONDUCTIVITY_COMPONENTS];
    PetscInt  off;
    Cell      cell;

    PetscCall(extractCellCoordinates(dm, c, &cell));
    if (background->type == CSEM_PRIMARY_LAYERED) {
      const LayeredModel *model = &background->layered;
      PetscReal           zMin = PETSC_MAX_REAL, zMax = PETSC_MIN_REAL;
      PetscInt            layer;

      PetscCall(computeCellCentroid(&cell));
      layer     = findLayer(model, cell.centroid[2]);
      sigmaB[0] = sigmaB[1] = model->sigmaH[layer];
      sigmaB[2]             = model->sigmaV[layer];
      for (PetscInt v = 0; v < NUM_VERTICES_PER_CELL; v++) {
        zMin = PetscMin(zMin, cell.coordinates[v * NUM_DIMENSIONS + 2]);
        zMax = PetscMax(zMax, cell.coordinates[v * NUM_DIMENSIONS + 2]);
      }
      for (PetscInt n = 0; n < model->numLayers - 1; n++) {
        const PetscReal tol = 1.0e-6 * (zMax - zMin);
        if (zMin < model->interfaces[n] - tol && zMax > model->interfaces[n] + tol) {
          counts[1]++;
          break;
        }
      }
    } else {
      PetscCall(extractCellMaterialId(dmMat, materialsId, c, &cell));
      if (cell.materialId < 0 || cell.materialId >= background->numMaterials) {
        counts[2]++;
        badMaterial = PetscMax(badMaterial, cell.materialId);
        continue;
      }
      for (PetscInt d = 0; d < NUM_CONDUCTIVITY_COMPONENTS; d++) sigmaB[d] = background->materialSigma[cell.materialId * 3 + d];
    }

    PetscCall(PetscSectionGetOffset(section, c, &off));
    for (PetscInt d = 0; d < NUM_CONDUCTIVITY_COMPONENTS; d++) sb[off + d] = sigmaB[d];
    PetscCall(extractCellConductivity(dmSigma, conductivity, c, &cell));
    if (differsFromBackground(cell.conductivity, sigmaB)) {
      primary->cells[c - grid->cellStart] = PETSC_TRUE;
      counts[0]++;
    }
  }
  PetscCall(VecRestoreArray(primary->sigmaBackground, &sb));

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, counts, 3, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &badMaterial, 1, MPIU_INT, MPI_MAX, comm));
  PetscCheck(counts[2] == 0, comm, PETSC_ERR_ARG_OUTOFRANGE,
             "createCsemPrimary: %" PetscInt_FMT " cell(s) have a material id outside the background table of %" PetscInt_FMT
             " materials (largest %" PetscInt_FMT ")",
             counts[2], background->numMaterials, badMaterial);
  primary->numAnomalyCells  = counts[0];
  primary->numCrossingCells = counts[1];
  if (counts[1] > 0)
    PetscCall(PetscPrintf(comm, "   WARNING: %s cell(s) cross an interface of the layered background; sigma_b is taken at the centroid.\n",
                          formatGroupedInt(counts[1])));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PETSC_TRUE on every rank if x lies in a local cell flagged in cells on some rank; points outside the mesh give PETSC_FALSE. */
static PetscErrorCode pointInCells(const DM dm, const Grid *grid, const PetscBool cells[], const PetscReal x[NUM_DIMENSIONS],
                                   PetscBool *inCells) {
  PetscFunctionBeginUser;

  Vec                point;
  PetscSF            sf = NULL;
  PetscInt           n;
  const PetscSFNode *found;
  PetscScalar       *a;
  PetscMPIInt        flag = 0;

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, NUM_DIMENSIONS, &point));
  PetscCall(VecSetBlockSize(point, NUM_DIMENSIONS));
  PetscCall(VecGetArrayWrite(point, &a));
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) a[d] = x[d];
  PetscCall(VecRestoreArrayWrite(point, &a));
  PetscCall(DMLocatePoints(dm, point, DM_POINTLOCATION_NONE, &sf));
  PetscCall(PetscSFGetGraph(sf, NULL, &n, NULL, &found));
  for (PetscInt i = 0; i < n; i++) {
    const PetscInt c = found[i].index;
    if (c >= grid->cellStart && c < grid->cellEnd && cells[c - grid->cellStart]) flag = 1;
  }
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
  *inCells = flag ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(VecDestroy(&point));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Sums the local values of each unconstrained DOF over the ranks that share it. */
static PetscErrorCode sumSharedDofs(const DM dm, Vec local) {
  PetscFunctionBeginUser;
  Vec global;
  PetscCall(DMGetGlobalVector(dm, &global));
  PetscCall(VecZeroEntries(global));
  PetscCall(DMLocalToGlobal(dm, local, ADD_VALUES, global));
  PetscCall(DMGlobalToLocal(dm, global, INSERT_VALUES, local));
  PetscCall(DMRestoreGlobalVector(dm, &global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Layered E_p,h: Nédélec interpolation of the analytic field on primary->cells.
 * Each DOF is the mean of the values interpolated by the cells of
 * primary->cells that contain it (all ranks; constrained DOFs: local cells).
 */
static PetscErrorCode computeLayeredPrimary(const DM dm, const Grid *grid, CsemPrimary *primary) {
  PetscFunctionBeginUser;

  MPI_Comm       comm = PetscObjectComm((PetscObject)dm);
  const PetscInt nE   = grid->fem.numDofInCell;
  PetscSection   section;
  PetscScalar   *ones;
  PetscInt       numLocal;
  Vec            numCells;

  for (PetscInt s = 0; s < primary->numSets; s++)
    for (PetscInt k = 0; k < primary->sets[s].numSources; k++) {
      const PetscReal *x       = primary->sets[s].sourceArray[k].position;
      PetscBool        inCells = PETSC_FALSE;

      PetscCall(pointInCells(dm, grid, primary->cells, x, &inCells));
      PetscCheck(!inCells, comm, PETSC_ERR_ARG_WRONG,
                 "computeCsemPrimary: source %" PetscInt_FMT " of set %" PetscInt_FMT
                 " at (%g, %g, %g) lies in a cell with sigma != sigma_b or in extraCells (layered background)",
                 k + 1, s + 1, (double)x[0], (double)x[1], (double)x[2]);
    }

  PetscCall(DMGetLocalSection(dm, &section));

  /* Number of cells of primary->cells that contain each DOF */
  PetscCall(DMCreateLocalVector(dm, &numCells));
  PetscCall(VecZeroEntries(numCells));
  PetscCall(PetscMalloc1(nE, &ones));
  for (PetscInt i = 0; i < nE; i++) ones[i] = 1.0;
  for (PetscInt c = grid->cellStart; c < grid->cellEnd; c++)
    if (primary->cells[c - grid->cellStart]) PetscCall(DMPlexVecSetClosure(dm, section, numCells, c, ones, ADD_ALL_VALUES));
  PetscCall(PetscFree(ones));
  PetscCall(sumSharedDofs(dm, numCells));
  PetscCall(VecGetLocalSize(numCells, &numLocal));

  for (PetscInt s = 0; s < primary->numSets; s++) {
    const CsemSourceSet *set = &primary->sets[s];
    LayeredDipole       *dipoles;
    PetscScalar         *coeff;

    PetscCall(createSetDipoles(&primary->background->layered, set, &dipoles));
    PetscCall(PetscMalloc1(nE * set->numSources, &coeff));
    for (PetscInt c = grid->cellStart; c < grid->cellEnd; c++) {
      Cell cell;
      if (!primary->cells[c - grid->cellStart]) continue;
      PetscCall(extractCellCoordinates(dm, c, &cell));
      PetscCall(interpolateNedelecCell(&grid->fem, &cell, layeredField, dipoles, set->numSources, coeff));
      for (PetscInt k = 0; k < set->numSources; k++)
        PetscCall(DMPlexVecSetClosure(dm, section, primary->Ep[s][k], c, &coeff[k * nE], ADD_ALL_VALUES));
    }
    PetscCall(PetscFree(coeff));
    PetscCall(destroySetDipoles(set->numSources, &dipoles));

    for (PetscInt k = 0; k < set->numSources; k++) {
      const PetscScalar *n;
      PetscScalar       *e;
      PetscCall(sumSharedDofs(dm, primary->Ep[s][k]));
      PetscCall(VecGetArrayRead(numCells, &n));
      PetscCall(VecGetArray(primary->Ep[s][k], &e));
      for (PetscInt i = 0; i < numLocal; i++)
        if (PetscRealPart(n[i]) > 0.0) e[i] /= n[i];
      PetscCall(VecRestoreArray(primary->Ep[s][k], &e));
      PetscCall(VecRestoreArrayRead(numCells, &n));
    }
  }
  PetscCall(VecDestroy(&numCells));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Model E_p,h: finite-element solve with the background conductivity, one per set. */
static PetscErrorCode computeModelPrimary(const DM dm, const Grid *grid, CsemPrimary *primary) {
  PetscFunctionBeginUser;

  for (PetscInt s = 0; s < primary->numSets; s++) {
    const CsemSourceSet *set         = &primary->sets[s];
    const PetscScalar    constFactor = iOmegaMu(set->freq);
    Mat                  A, B, G, X;
    IS                   primalVertices = NULL;

    PetscCall(assembleCsemRhs(*set, dm, grid, constFactor, &B));
    PetscCall(assembleMaxwellOperator(dm, grid, primary->sigmaBackground, constFactor, &A, NULL, &G));
    PetscCall(setupPrimalVerticesFromOptions(dm, A, primary->sigmaBackground, &primalVertices));
    PetscCall(solveMaxwellSystem(dm, A, B, G, grid->fem.order, primalVertices, &X, &primary->solveInfo));
    for (PetscInt k = 0; k < set->numSources; k++) {
      Vec x;
      PetscCall(MatDenseGetColumnVecRead(X, k, &x));
      PetscCall(DMGlobalToLocal(dm, x, INSERT_VALUES, primary->Ep[s][k]));
      PetscCall(MatDenseRestoreColumnVecRead(X, k, &x));
    }
    PetscCall(ISDestroy(&primalVertices));
    PetscCall(MatDestroy(&X));
    PetscCall(MatDestroy(&G));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&B));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the primary field E_p,h of every source of the source sets.
 *
 * E_p,h is zero outside the closure of primary->cells (layered) and on the
 * constrained DOFs (model).
 */
PetscErrorCode computeCsemPrimary(PetscInt numSets, const CsemSourceSet sets[], const DM dm, const Grid *grid, const PetscBool extraCells[],
                                  CsemPrimary *primary) {
  PetscFunctionBeginUser;

  MPI_Comm       comm = PetscObjectComm((PetscObject)dm);
  PetscLogDouble t0, t1;

  PetscCall(PetscTime(&t0));
  if (extraCells)
    for (PetscInt c = 0; c < grid->cellEnd - grid->cellStart; c++) primary->cells[c] = (PetscBool)(primary->cells[c] || extraCells[c]);

  primary->numSets = numSets;
  PetscCall(PetscMalloc2(numSets, &primary->sets, numSets, &primary->Ep));
  for (PetscInt s = 0; s < numSets; s++) {
    primary->sets[s] = sets[s];
    PetscCall(PetscMalloc1(sets[s].numSources, &primary->sets[s].sourceArray));
    PetscCall(PetscArraycpy(primary->sets[s].sourceArray, sets[s].sourceArray, sets[s].numSources));
    PetscCall(PetscMalloc1(sets[s].numSources, &primary->Ep[s]));
    for (PetscInt k = 0; k < sets[s].numSources; k++) {
      PetscCall(DMCreateLocalVector(dm, &primary->Ep[s][k]));
      PetscCall(VecZeroEntries(primary->Ep[s][k]));
    }
  }

  if (primary->type == CSEM_PRIMARY_LAYERED)
    PetscCall(computeLayeredPrimary(dm, grid, primary));
  else
    PetscCall(computeModelPrimary(dm, grid, primary));

  PetscCall(PetscTime(&t1));
  primary->time = t1 - t0;
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &primary->time, 1, MPI_DOUBLE, MPI_MAX, comm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Assembles the secondary right-hand sides of every source set.
 *
 * M_e(σ_e - σ_b,e) is computed once per cell and applied to every source of every set.
 */
PetscErrorCode assembleCsemSecondaryRhs(const CsemPrimary *primary, const DM dm, const Grid *grid, Vec conductivity, Mat B[]) {
  PetscFunctionBeginUser;

  const PetscInt nE = grid->fem.numDofInCell;
  Quadrature3D   quadrature3D;
  PetscSection   section;
  DM             dmSigma;
  PetscReal    **Me, **Ke;
  PetscScalar   *be, *constFactor;
  PetscInt      *first, numRhs = 0;
  Vec           *bLocal;

  /* bLocal[first[s] + k]: source k of set s */
  PetscCall(PetscMalloc2(primary->numSets + 1, &first, primary->numSets, &constFactor));
  for (PetscInt s = 0; s < primary->numSets; s++) {
    first[s]       = numRhs;
    constFactor[s] = iOmegaMu(primary->sets[s].freq);
    numRhs += primary->sets[s].numSources;
  }
  first[primary->numSets] = numRhs;

  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(VecGetDM(conductivity, &dmSigma));
  PetscCall(createQuadrature3D(grid->fem.order, &quadrature3D));
  PetscCall(createDenseRows(nE, nE, &Me));
  PetscCall(createDenseRows(nE, nE, &Ke));
  PetscCall(PetscMalloc2(nE, &be, numRhs, &bLocal));
  for (PetscInt r = 0; r < numRhs; r++) {
    PetscCall(DMGetLocalVector(dm, &bLocal[r]));
    PetscCall(VecZeroEntries(bLocal[r]));
  }

  /* b_k = iωμ_k Σ_e M_e(σ_e - σ_b,e) E_p,h|_e */
  for (PetscInt c = grid->cellStart; c < grid->cellEnd; c++) {
    Cell cell, cellB;

    if (!primary->cells[c - grid->cellStart]) continue;
    PetscCall(extractCellCoordinates(dm, c, &cell));
    PetscCall(extractCellConductivity(dmSigma, conductivity, c, &cell));
    PetscCall(extractCellConductivity(dmSigma, primary->sigmaBackground, c, &cellB));
    if (!differsFromBackground(cell.conductivity, cellB.conductivity)) continue;
    for (PetscInt d = 0; d < NUM_CONDUCTIVITY_COMPONENTS; d++) cell.conductivity[d] -= cellB.conductivity[d];
    PetscCall(computeElementalMatrices(&grid->fem, &cell, &quadrature3D, Me, Ke));

    for (PetscInt s = 0; s < primary->numSets; s++)
      for (PetscInt k = 0; k < primary->sets[s].numSources; k++) {
        PetscScalar *ep = NULL;
        PetscInt     n  = nE;
        PetscCall(DMPlexVecGetClosure(dm, section, primary->Ep[s][k], c, &n, &ep));
        for (PetscInt j = 0; j < nE; j++) {
          be[j] = 0.0;
          for (PetscInt l = 0; l < nE; l++) be[j] += Me[j][l] * ep[l];
          be[j] *= constFactor[s];
        }
        PetscCall(DMPlexVecRestoreClosure(dm, section, primary->Ep[s][k], c, &n, &ep));
        PetscCall(DMPlexVecSetClosure(dm, section, bLocal[first[s] + k], c, be, ADD_VALUES));
      }
  }

  for (PetscInt s = 0; s < primary->numSets; s++) {
    PetscCall(MatZeroEntries(B[s]));
    for (PetscInt k = 0; k < primary->sets[s].numSources; k++) {
      Vec bcol;
      PetscCall(MatDenseGetColumnVecWrite(B[s], k, &bcol));
      PetscCall(DMLocalToGlobal(dm, bLocal[first[s] + k], ADD_VALUES, bcol));
      PetscCall(MatDenseRestoreColumnVecWrite(B[s], k, &bcol));
    }
  }
  for (PetscInt r = 0; r < numRhs; r++) PetscCall(DMRestoreLocalVector(dm, &bLocal[r]));

  PetscCall(PetscFree2(be, bLocal));
  PetscCall(PetscFree2(first, constFactor));
  PetscCall(destroyDenseRows(&Ke));
  PetscCall(destroyDenseRows(&Me));
  PetscCall(destroyQuadrature3D(&quadrature3D));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Adds the primary field of one source to the fields at the receivers.
 *
 * Layered: each rank evaluates the receivers of its rows of fields.
 */
PetscErrorCode addCsemPrimaryAtReceivers(const CsemPrimary *primary, PetscInt set, PetscInt source, const DM dm,
                                         const ReceiverInterpolationMatrices *Q, Vec receivers, Vec fields[NUM_RECEIVER_FIELDS]) {
  PetscFunctionBeginUser;

  const CsemSourceSet *sources = &primary->sets[set];

  if (primary->type == CSEM_PRIMARY_LAYERED) {
    const PetscScalar *coords;
    PetscScalar       *EH, *a;
    PetscReal         *points, moment[NUM_DIMENSIONS];
    PetscInt           lo, hi, n, *layers;
    LayeredDipole      dipole;

    PetscCall(VecGetOwnershipRange(fields[RECV_EX], &lo, &hi));
    n = hi - lo;
    PetscCall(PetscMalloc3(NUM_DIMENSIONS * n, &points, 2 * NUM_DIMENSIONS * n, &EH, n, &layers));
    PetscCall(VecGetArrayRead(receivers, &coords));
    for (PetscInt i = 0; i < NUM_DIMENSIONS * n; i++) points[i] = PetscRealPart(coords[NUM_DIMENSIONS * lo + i]);
    PetscCall(VecRestoreArrayRead(receivers, &coords));
    /* On an interface, the layer of the cell that interpolates the receiver */
    for (PetscInt r = 0; r < n; r++)
      layers[r] =
          findLayerOnSide(&primary->background->layered, points[NUM_DIMENSIONS * r + 2], Q->cellCentroids[NUM_DIMENSIONS * (lo + r) + 2]);

    PetscCall(sourceMoment(&sources->sourceArray[source], moment));
    PetscCall(createLayeredDipole(&primary->background->layered, sources->freq, sources->sourceArray[source].position, moment, &dipole));
    PetscCall(evaluateLayeredDipole(&dipole, n, points, layers, EH, &EH[NUM_DIMENSIONS * n]));
    PetscCall(destroyLayeredDipole(&dipole));

    for (PetscInt f = 0; f < NUM_RECEIVER_FIELDS; f++) {
      const PetscScalar *src = (f < RECV_HX) ? EH : &EH[NUM_DIMENSIONS * n];
      const PetscInt     d   = (f < RECV_HX) ? f : f - RECV_HX;
      PetscCall(VecGetArray(fields[f], &a));
      for (PetscInt r = 0; r < n; r++) a[r] += src[NUM_DIMENSIONS * r + d];
      PetscCall(VecRestoreArray(fields[f], &a));
    }
    PetscCall(PetscFree3(points, EH, layers));
  } else {
    Vec x, primaryFields[NUM_RECEIVER_FIELDS];

    PetscCall(DMGetGlobalVector(dm, &x));
    PetscCall(DMLocalToGlobal(dm, primary->Ep[set][source], INSERT_VALUES, x));
    PetscCall(createReceiverFields(Q, primaryFields));
    PetscCall(interpolateReceiverFields(Q, x, iOmegaMu(sources->freq), primaryFields));
    for (PetscInt f = 0; f < NUM_RECEIVER_FIELDS; f++) PetscCall(VecAXPY(fields[f], 1.0, primaryFields[f]));
    PetscCall(destroyReceiverFields(primaryFields));
    PetscCall(DMRestoreGlobalVector(dm, &x));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints the "CSEM formulation" section of the run report.
 */
PetscErrorCode logCsemFormulation(MPI_Comm comm, const CsemPrimary *primary) {
  PetscFunctionBeginUser;

  PetscCall(logSection(comm, "CSEM formulation"));
  PetscCall(logKvStr(comm, "Formulation", "secondary"));
  if (primary->type == CSEM_PRIMARY_LAYERED) {
    PetscCall(logKvf(comm, "Background", "layered, %s layer(s)", formatGroupedInt(primary->background->layered.numLayers)));
    if (primary->numCrossingCells > 0) PetscCall(logKvStr(comm, "Cells across interfaces", formatGroupedInt(primary->numCrossingCells)));
  } else {
    PetscCall(logKvf(comm, "Background", "model, %s material(s)", formatGroupedInt(primary->background->numMaterials)));
    PetscCall(logKvStr(comm, "Background solver", primary->solveInfo.solver));
    PetscCall(
        logKvStr(comm, "Background status", primary->solveInfo.reason > 0 ? "converged" : KSPConvergedReasons[primary->solveInfo.reason]));
  }
  PetscCall(logKvStr(comm, "Anomalous cells", formatGroupedInt(primary->numAnomalyCells)));
  PetscCall(logKvf(comm, "Primary field", "%.3f s", primary->time));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Frees a primary field created by createCsemPrimary().
 */
PetscErrorCode destroyCsemPrimary(CsemPrimary *primary) {
  PetscFunctionBeginUser;
  for (PetscInt s = 0; s < primary->numSets; s++) {
    for (PetscInt k = 0; k < primary->sets[s].numSources; k++) PetscCall(VecDestroy(&primary->Ep[s][k]));
    PetscCall(PetscFree(primary->Ep[s]));
    PetscCall(PetscFree(primary->sets[s].sourceArray));
  }
  PetscCall(PetscFree2(primary->sets, primary->Ep));
  PetscCall(PetscFree(primary->cells));
  PetscCall(VecDestroy(&primary->sigmaBackground));
  primary->numSets = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}
