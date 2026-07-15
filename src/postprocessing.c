/*
 * Filename: postprocessing.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Data postprocessing functions for the PETGEM forward kernel.
 */

/* C libraries */
#include <time.h>

/* PETSc libraries */
#include <petscsys.h>
#include <petscviewerhdf5.h>

/* PETGEM functions */
#include "common.h"
#include "constants.h"
#include "grid.h"
#include "fem.h"
#include "io.h"
#include "postprocessing.h"
#include "receiver_interp.h"
#include "version.h"

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
 * The output filename is `{output_directory}/{output_filename}.h5`; the
 * per-source `_src{k}` suffix used by the legacy one-file-per-source layout
 * is gone. All Vec writes are collective on the kernel's MPI communicator
 * and go through PETSc's native HDF5 viewer (parallel HDF5 when PETSc is
 * built against a parallel HDF5 library); no rank-0 gather happens.
 *
 * `receivers` is the serial Vec (PETSC_COMM_SELF, length 3·N_recv) returned
 * by loadCsemInputs - passed through so postprocessing does not re-open the
 * input HDF5.
 *
 * @param[in] params     Forward-modeling parameters (order, output paths, MPI tasks).
 * @param[in] sources    Transmitter set (one solution column per source).
 * @param[in] dm         DMPlex mesh and H(curl) discretization.
 * @param[in] grid       Finite-element grid descriptor.
 * @param[in] receivers  Serial Vec of 3·N_recv receiver coordinates.
 * @param[in] X          Solution matrix (one column per source).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 *
 * @note Assumes 3D H(curl) Nédélec elements. Receivers outside the
 *       computational domain trigger a warning and are skipped.
 */
PetscErrorCode computeFields(const petgemParams params, 
                             const CsemSourceSet sources,
                             const DM dm, const Grid grid,
                             Vec receivers, const Mat X) {
  PetscFunctionBeginUser;

  /* Variable declarations */
  PetscReal omega;
  PetscScalar constFactor;
  Vec Ex, Ey, Ez, Hx, Hy, Hz;
  PetscViewer viewerOutput;
  char outFileName[PETSC_MAX_PATH_LEN];
  char groupPath[64];
  ReceiverInterpolationMatrices Q;

  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* Compute constant */
  omega = sources.freq * 2.0 * PETSC_PI;
  constFactor = (0.0 + 1.0 * PETSC_i) * (omega * MU);

  /* Build the receiver interpolation operator (shared with im.csem).
   * One global Q assembly handles all receivers and all sources at this
   * frequency.  Per-receiver Ex = QEx * x, etc. - MPI-invariant by
   * construction because each Q row is decided once with global column
   * indexing, irrespective of partition. */
  PetscCall(buildReceiverInterpolationMatrices(params.order, receivers, dm, &grid, &Q));

  /* Allocate output Vecs sized to match Q's row layout (left vector).
   * These are parallel Vecs on the kernel communicator, so VecView through
   * the HDF5 viewer below performs collective MPI-IO writes. */
  PetscCall(MatCreateVecs(Q.QEx, NULL, &Ex));
  PetscCall(MatCreateVecs(Q.QEy, NULL, &Ey));
  PetscCall(MatCreateVecs(Q.QEz, NULL, &Ez));
  PetscCall(MatCreateVecs(Q.QHx, NULL, &Hx));
  PetscCall(MatCreateVecs(Q.QHy, NULL, &Hy));
  PetscCall(MatCreateVecs(Q.QHz, NULL, &Hz));

  /* Build the single output file name: {output_dir}/{output_filename}.h5
   * (shared with im.csem, so both kernels name their products alike). */
  PetscCall(buildOutputPath(&params, ".h5", outFileName, sizeof(outFileName)));

  /* Print message */
  PetscCall(PetscPrintf(comm, "\n Field interpolation:\n"));
  PetscCall(PetscPrintf(comm, "   %-24s = %s\n",                "Input file",          params.inputFile));
  PetscCall(PetscPrintf(comm, "   %-24s = %s\n", "Number of receivers", formatGroupedInt(Q.numReceivers)));
  PetscCall(PetscPrintf(comm, "   %-24s = %s\n",                "Output file",         outFileName));
  PetscCall(PetscPrintf(comm, "   %-24s = %s\n",                "Status",              "Started"));

  /* Open the single HDF5 output file on the kernel communicator. PETSc's
   * HDF5 viewer routes the collective VecView calls below through MPI-IO
   * when PETSc is linked against a parallel HDF5 build. */
  PetscCall(PetscViewerHDF5Open(comm, outFileName, FILE_MODE_WRITE, &viewerOutput));

  /* Root provenance attributes - written ONCE for the whole file. The common
   * block (version, simulation type, input bundle, order, solver, tasks,
   * date) is shared with im.csem; only the forward-specific attributes below
   * are added here. */
  PetscCall(writeRunProvenance(viewerOutput, &params, PETGEM_SIM_FM));
  PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "num_sources", PETSC_INT,  &sources.numSources));
  PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "frequency",   PETSC_REAL, &sources.freq));

  /* Postprocessing fields for each source */
  for (PetscInt i = 0; i < sources.numSources; i++) {

    /* Variable declarations */
    Vec x;

    /* Get the solution column for this source */
    PetscCall(MatDenseGetColumnVecRead(X, i, &x));

    PetscCall(PetscPrintf(comm, "   %-24s = %s of %s\n", "Processing source", formatGroupedInt(i + 1), formatGroupedInt(sources.numSources)));

    /* Apply Q to the H(curl) solution: Ex = QEx*x, Ey = QEy*x, ... */
    PetscCall(MatMult(Q.QEx, x, Ex));
    PetscCall(MatMult(Q.QEy, x, Ey));
    PetscCall(MatMult(Q.QEz, x, Ez));
    PetscCall(MatMult(Q.QHx, x, Hx));
    PetscCall(MatMult(Q.QHy, x, Hy));
    PetscCall(MatMult(Q.QHz, x, Hz));

    PetscCall(MatDenseRestoreColumnVecRead(X, i, &x));

    /* H from curl(E): divide by constFactor = i*omega*mu (Maxwell). */
    PetscCall(VecScale(Hx, 1.0 / constFactor));
    PetscCall(VecScale(Hy, 1.0 / constFactor));
    PetscCall(VecScale(Hz, 1.0 / constFactor));

    /* Per-source group path. Field components land under
     * /sources/src{k}/fields/, and the per-source metadata attributes
     * attach to the /sources/src{k} group itself. */
    snprintf(groupPath, sizeof(groupPath), "/sources/src%" PetscInt_FMT "/fields", i + 1);
    PetscCall(PetscObjectSetName((PetscObject)Ex, "Ex"));
    PetscCall(PetscObjectSetName((PetscObject)Ey, "Ey"));
    PetscCall(PetscObjectSetName((PetscObject)Ez, "Ez"));
    PetscCall(PetscObjectSetName((PetscObject)Hx, "Hx"));
    PetscCall(PetscObjectSetName((PetscObject)Hy, "Hy"));
    PetscCall(PetscObjectSetName((PetscObject)Hz, "Hz"));
    PetscCall(PetscViewerHDF5PushGroup(viewerOutput, groupPath));
    PetscCall(VecView(Ex, viewerOutput));
    PetscCall(VecView(Ey, viewerOutput));
    PetscCall(VecView(Ez, viewerOutput));
    PetscCall(VecView(Hx, viewerOutput));
    PetscCall(VecView(Hy, viewerOutput));
    PetscCall(VecView(Hz, viewerOutput));
    PetscCall(PetscViewerHDF5PopGroup(viewerOutput));

    /* Per-source metadata at /sources/src{k}. */
    {
      const CsemSource *s = &sources.sourceArray[i];
      snprintf(groupPath, sizeof(groupPath), "/sources/src%" PetscInt_FMT, i + 1);
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, groupPath, "frequency",     PETSC_REAL, &sources.freq));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, groupPath, "x_pos",         PETSC_REAL, &s->position[0]));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, groupPath, "y_pos",         PETSC_REAL, &s->position[1]));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, groupPath, "z_pos",         PETSC_REAL, &s->position[2]));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, groupPath, "current",       PETSC_REAL, &s->current));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, groupPath, "length",        PETSC_REAL, &s->length));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, groupPath, "dip_angle",     PETSC_REAL, &s->dipAngle));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, groupPath, "azimuth_angle", PETSC_REAL, &s->azimuthAngle));
    }
  }

  /* Close the single output file. */
  PetscCall(PetscViewerDestroy(&viewerOutput));

  PetscCall(PetscPrintf(comm, "   %-24s = %s\n", "Status", "Finished"));

  /* Free memory */
  PetscCall(VecDestroy(&Ex));
  PetscCall(VecDestroy(&Ey));
  PetscCall(VecDestroy(&Ez));
  PetscCall(VecDestroy(&Hx));
  PetscCall(VecDestroy(&Hy));
  PetscCall(VecDestroy(&Hz));
  PetscCall(destroyReceiverInterpolationMatrices(&Q));

  PetscFunctionReturn(PETSC_SUCCESS);
}
