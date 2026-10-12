/*
 * Filename: csem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * CSEM physics: transmitters, the PEC grid, the dipole right-hand side and the
 * fields at the receivers.
 */

/* PETSc libraries */
#include <petscviewerhdf5.h>

/* PETGEM functions */
#include "assembly.h"
#include "common.h"
#include "constants.h"
#include "csem.h"
#include "fem.h"
#include "grid.h"
#include "io.h"
#include "receiver_interp.h"

/**
 * @brief Reads the /sources group of a bundle as a row table.
 *
 * Row i holds entry i in CSEM_SRC_* column order: frequency, position,
 * current, length, dip and azimuth. Fails when the group is empty or a
 * frequency is not positive.
 */
PetscErrorCode loadSourceRows(const char *bundleFile, PetscInt *numRows, PetscReal **rows) {
  PetscFunctionBeginUser;

  const char        *names[] = {"freq", "position", "current", "length", "dipAngle", "azimuthAngle"};
  const PetscInt     first[] = {CSEM_SRC_FREQ, CSEM_SRC_X, CSEM_SRC_CURRENT, CSEM_SRC_LENGTH, CSEM_SRC_DIP, CSEM_SRC_AZIMUTH};
  const PetscInt     width[] = {1, 3, 1, 1, 1, 1};
  const PetscInt     numVecs = 6;
  PetscViewer        viewer;
  Vec                v;
  const PetscScalar *a;
  PetscInt           n;

  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, bundleFile, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/sources"));
  for (PetscInt i = 0; i < numVecs; i++) {
    PetscCall(loadSerialVec(viewer, names[i], &v));
    PetscCall(VecGetSize(v, &n));
    n /= width[i];
    if (i == 0) {
      *numRows = n;
      PetscCall(PetscMalloc1(n * CSEM_SRC_NUM_COLUMNS, rows));
    }
    PetscCheck(n == *numRows, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
               "%s: /sources/%s has %" PetscInt_FMT " entries, expected %" PetscInt_FMT, bundleFile, names[i], n, *numRows);
    PetscCall(VecGetArrayRead(v, &a));
    for (PetscInt j = 0; j < n; j++) {
      for (PetscInt k = 0; k < width[i]; k++) (*rows)[j * CSEM_SRC_NUM_COLUMNS + first[i] + k] = PetscRealPart(a[j * width[i] + k]);
    }
    PetscCall(VecRestoreArrayRead(v, &a));
    PetscCall(VecDestroy(&v));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCheck(*numRows > 0, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "%s: /sources is empty", bundleFile);
  for (PetscInt i = 0; i < *numRows; i++) {
    const PetscReal f = (*rows)[i * CSEM_SRC_NUM_COLUMNS + CSEM_SRC_FREQ];
    PetscCheck(f > 0.0, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "%s: /sources/freq[%" PetscInt_FMT "] = %g is not positive", bundleFile,
               i, (double)f);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Loads the CSEM transmitter set from the /sources group of the bundle.
 *
 * Forward modeling is monochromatic: the first entry's frequency applies to
 * the whole set.
 */
PetscErrorCode loadCsemSources(const PetgemParams *params, CsemSourceSet *sources) {
  PetscFunctionBeginUser;

  PetscInt   n;
  PetscReal *rows;

  PetscCall(loadSourceRows(params->inputFile, &n, &rows));
  sources->freq       = rows[CSEM_SRC_FREQ];
  sources->numSources = n;
  PetscCall(PetscMalloc1(n, &sources->sourceArray));
  for (PetscInt i = 0; i < n; i++) {
    const PetscReal *r = &rows[i * CSEM_SRC_NUM_COLUMNS];
    CsemSource      *s = &sources->sourceArray[i];
    s->position[0]     = r[CSEM_SRC_X];
    s->position[1]     = r[CSEM_SRC_Y];
    s->position[2]     = r[CSEM_SRC_Z];
    s->current         = r[CSEM_SRC_CURRENT];
    s->length          = r[CSEM_SRC_LENGTH];
    s->dipAngle        = r[CSEM_SRC_DIP];
    s->azimuthAngle    = r[CSEM_SRC_AZIMUTH];
  }
  PetscCall(PetscFree(rows));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints the "CSEM source(s)" section of the forward run report.
 *
 * One source: frequency, position, current, length, dip and azimuth. Several
 * sources: the frequency followed by one sub-block per transmitter.
 */
PetscErrorCode logForwardSources(const CsemSourceSet *sources) {
  PetscFunctionBeginUser;

  MPI_Comm comm = PETSC_COMM_WORLD;

  if (sources->numSources == 1) {
    const CsemSource *s = &sources->sourceArray[0];
    PetscCall(logSection(comm, "CSEM source"));
    PetscCall(logKvReal(comm, "Frequency (Hz)", sources->freq));
    PetscCall(
        logKvf(comm, "Position (m)", "(%s, %s, %s)", formatReal(s->position[0]), formatReal(s->position[1]), formatReal(s->position[2])));
    PetscCall(logKvReal(comm, "Current (A)", s->current));
    PetscCall(logKvReal(comm, "Length (m)", s->length));
    PetscCall(logKvf(comm, "Dip / azimuth (deg)", "%s / %s", formatReal(s->dipAngle), formatReal(s->azimuthAngle)));
  } else {
    PetscCall(logSection(comm, "CSEM sources"));
    PetscCall(logKvReal(comm, "Frequency (Hz)", sources->freq));
    PetscCall(logKvInt(comm, "Number of sources", sources->numSources));
    for (PetscInt i = 0; i < sources->numSources; i++) {
      const CsemSource *s = &sources->sourceArray[i];
      PetscCall(PetscPrintf(comm, "   Source %" PetscInt_FMT ":\n", i + 1));
      PetscCall(PetscPrintf(comm, "     %-22s = (%s, %s, %s)\n", "Position (m)", formatReal(s->position[0]), formatReal(s->position[1]),
                            formatReal(s->position[2])));
      PetscCall(PetscPrintf(comm, "     %-22s = %s\n", "Current (A)", formatReal(s->current)));
      PetscCall(PetscPrintf(comm, "     %-22s = %s\n", "Length (m)", formatReal(s->length)));
      PetscCall(PetscPrintf(comm, "     %-22s = %s / %s\n", "Dip / azimuth (deg)", formatReal(s->dipAngle), formatReal(s->azimuthAngle)));
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Configures the CSEM grid: H(curl) and H1 sections with PETGEM_BC_PEC.
 */
PetscErrorCode setupCsemGrid(PetscInt order, DM *dm, Grid *grid) {
  PetscFunctionBeginUser;
  PetscCall(setupNedelecGrid(order, PETGEM_BC_PEC, dm, grid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Assembles the right-hand side matrix for the CSEM system.
 *
 * Constructs the RHS vectors for every transmitter in the simulation; each
 * column of `B` corresponds to one source. Performs finite-element assembly
 * using H(curl) Nédélec elements, taking the source dipole orientation,
 * rotation, and position into account.
 *
 * For each source the function (a) locates it in the mesh via locatePoint,
 * (b) extracts cell coordinates and computes the Jacobian, (c) maps the
 * source position to reference coordinates (ξ, η, ζ), (d) evaluates the
 * Nédélec basis and its curls there, (e) writes the local contributions
 * into a per-source RHS Vec, and (f) copies the assembled Vec into the
 * corresponding column of `B`. A final complex scaling by iωμ is applied.
 *
 * @note The caller is responsible for destroying the returned matrix `B`.
 */
PetscErrorCode assembleCsemRhs(const CsemSourceSet sources, const DM dm, const Grid *grid, const PetscScalar constFactor, Mat *B) {
  PetscFunctionBeginUser;

  const PetscInt nE = grid->fem.numDofInCell;
  Cell           cell;
  PetscInt       numDofIndices, *dofIndices;
  PetscReal    **Ni, XiEtaZeta[NUM_DIMENSIONS];
  PetscScalar   *closureRhs;
  PetscSection   section;
  Vec            b, bcol;

  /* One right-hand side vector, and a matrix with one column per source */
  PetscCall(createRhsVector(dm, &b));
  PetscCall(createRhsMatrix(b, sources.numSources, B));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscMalloc1(nE, &closureRhs));
  PetscCall(createBasisValues(nE, &Ni));

  for (PetscInt i = 0; i < sources.numSources; i++) {
    const CsemSource *src = &sources.sourceArray[i];
    PetscReal         direction[NUM_DIMENSIONS], moment[NUM_DIMENSIONS];
    PetscInt          cellId;

    PetscCall(VecZeroEntries(b));

    /* Dipole moment I*L along the rotated source axis */
    PetscCall(computeVectorRotation(src->azimuthAngle, src->dipAngle, direction));
    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) moment[d] = direction[d] * (src->current * src->length);

    /* Point source: <N_j, moment> in the owning cell */
    PetscCall(locatePoint(dm, src->position, &cellId));
    if (cellId >= 0) {
      PetscCall(extractCellCoordinates(dm, cellId, &cell));
      PetscCall(tetrahedronXyzToReference(cell.coordinates, src->position, XiEtaZeta));
      PetscCall(evaluateNedelecBasis(&grid->fem, &cell, XiEtaZeta, Ni, NULL));
      for (PetscInt j = 0; j < nE; j++) {
        closureRhs[j] = 0.0;
        for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) closureRhs[j] += Ni[d][j] * moment[d];
      }
      PetscCall(DMPlexGetClosureIndices(dm, section, section, cellId, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
      PetscCall(VecSetValuesLocal(b, numDofIndices, dofIndices, closureRhs, INSERT_VALUES));
      PetscCall(DMPlexRestoreClosureIndices(dm, section, section, cellId, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
    }
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    PetscCall(MatDenseGetColumnVecWrite(*B, i, &bcol));
    PetscCall(VecCopy(b, bcol));
    PetscCall(MatDenseRestoreColumnVecWrite(*B, i, &bcol));
  }

  PetscCall(MatScale(*B, constFactor));

  PetscCall(VecDestroy(&b));
  PetscCall(destroyBasisValues(&Ni));
  PetscCall(PetscFree(closureRhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes electric and magnetic fields at receivers (forward kernel).
 *
 * Locates the receivers in the mesh, interpolates the H(curl) solution X to
 * obtain E at each receiver, derives H via H = curl(E)/(iωμ), and writes a
 * SINGLE HDF5 response file containing every source. The file groups the
 * responses by source (one subgroup per transmitter) and carries run-wide
 * provenance attributes at the root:
 *
 *   /                              root attrs: petgem_version, input_filename,
 *                                              date, order, mpi_tasks,
 *                                              num_sources, frequency
 *   /sources/src{k}/               attrs: frequency, x_pos, y_pos, z_pos,
 *                                         current, length, dip_angle,
 *                                         azimuth_angle
 *   /sources/src{k}/fields/        Ex, Ey, Ez, Hx, Hy, Hz (PETSc Vec)
 *
 * The output filename is `{output_directory}/{output_filename}.h5`. All Vec
 * writes are collective on the kernel's MPI communicator and go through
 * PETSc's HDF5 viewer; no rank-0 gather happens.
 *
 * `receivers` is the serial Vec (PETSC_COMM_SELF, length 3·N_recv) returned
 * by loadModelInputs - passed through so postprocessing does not re-open the
 * input HDF5.
 *
 * @note Assumes 3D H(curl) Nédélec elements. Receivers outside the
 *       computational domain trigger a warning and are skipped.
 */
PetscErrorCode computeCsemResponses(const PetgemParams *params, const CsemSourceSet sources, const DM dm, const Grid *grid, Vec receivers,
                                    const Mat X) {
  PetscFunctionBeginUser;

  /* Variable declarations */
  const PetscScalar             constFactor = iOmegaMu(sources.freq);
  Vec                           fields[NUM_RECEIVER_FIELDS];
  PetscViewer                   viewer;
  char                          outFileName[PETSC_MAX_PATH_LEN];
  char                          groupPath[64];
  ReceiverInterpolationMatrices Q;
  MPI_Comm                      comm = PetscObjectComm((PetscObject)dm);

  /* Receiver interpolation operators, shared by all sources */
  PetscCall(buildReceiverInterpolationMatrices(receivers, dm, grid, &Q));
  PetscCall(createReceiverFields(&Q, fields));

  /* Single output file with the run provenance at the root */
  PetscCall(buildOutputPath(params, ".h5", outFileName, sizeof(outFileName)));
  PetscCall(PetscViewerHDF5Open(comm, outFileName, FILE_MODE_WRITE, &viewer));
  PetscCall(writeRunProvenance(viewer, params, PETGEM_SIM_FM));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "num_sources", PETSC_INT, &sources.numSources));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "frequency", PETSC_REAL, &sources.freq));

  for (PetscInt i = 0; i < sources.numSources; i++) {
    const CsemSource *s = &sources.sourceArray[i];
    Vec               x;

    /* E and H at the receivers for this source */
    PetscCall(MatDenseGetColumnVecRead(X, i, &x));
    PetscCall(interpolateReceiverFields(&Q, x, constFactor, fields));
    PetscCall(MatDenseRestoreColumnVecRead(X, i, &x));

    /* /sources/src{k}/fields/{Ex..Hz} */
    PetscCall(PetscSNPrintf(groupPath, sizeof(groupPath), "/sources/src%" PetscInt_FMT "/fields", i + 1));
    PetscCall(PetscViewerHDF5PushGroup(viewer, groupPath));
    for (PetscInt comp = 0; comp < NUM_RECEIVER_FIELDS; comp++) PetscCall(VecView(fields[comp], viewer));
    PetscCall(PetscViewerHDF5PopGroup(viewer));

    /* /sources/src{k} attributes */
    PetscCall(PetscSNPrintf(groupPath, sizeof(groupPath), "/sources/src%" PetscInt_FMT, i + 1));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "frequency", PETSC_REAL, &sources.freq));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "x_pos", PETSC_REAL, &s->position[0]));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "y_pos", PETSC_REAL, &s->position[1]));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "z_pos", PETSC_REAL, &s->position[2]));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "current", PETSC_REAL, &s->current));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "length", PETSC_REAL, &s->length));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "dip_angle", PETSC_REAL, &s->dipAngle));
    PetscCall(PetscViewerHDF5WriteAttribute(viewer, groupPath, "azimuth_angle", PETSC_REAL, &s->azimuthAngle));
  }

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(destroyReceiverFields(fields));
  PetscCall(destroyReceiverInterpolationMatrices(&Q));

  PetscFunctionReturn(PETSC_SUCCESS);
}
