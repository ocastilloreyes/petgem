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
#include "constants.h"
#include "grid.h"
#include "hvfem.h"
#include "inputs.h"
#include "postprocessing.h"
#include "receiver_interp.h"
#include "version.h"

/**
 * @brief Computes electric (E) and magnetic (H) fields at specified receiver locations.
 *
 * @param[in] params A `csemParams` struct containing simulation parameters such as
 *                   finite element order, output filenames, and MPI task information.
 * @param[in] sources A `CsemSourceSet` struct containing information about sources,
 *                    including number of sources, frequency, positions, and currents.
 * @param[in] dm The PETSc DMPlex object representing the mesh and H(curl) discretization.
 * @param[in] grid A `Grid` struct containing mesh statistics, number of DOFs per cell,
 *                 and other relevant discretization information.
 * @param[in] X The solution matrix (Mat), where each column corresponds to the solution
 *              vector for a specific source.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or an appropriate PETSc error code.
 *
 * @details
 * This function performs the following steps:
 * 1. Loads the receiver coordinates from an HDF5 file into a PETSc Vec.
 * 2. Locates receivers in the computational mesh using `DMLocatePoints`.
 * 3. Allocates PETSc vectors to store the electric (Ex, Ey, Ez) and magnetic (Hx, Hy, Hz) fields.
 * 4. Loops over each source:
 *    - Extracts the solution vector for the source.
 *    - Converts the global solution vector to a local representation.
 *    - Loops over receivers:
 *        * Determines the cell containing the receiver.
 *        * Computes the reference coordinates (Xi, Eta, Zeta) for the receiver.
 *        * Computes Nédélec basis functions and their curls at the receiver location.
 *        * Interpolates the E and H fields using the DOFs in the cell.
 *        * Applies Maxwell's equations to compute H from E (scaling by frequency and permeability).
 *    - Performs parallel assembly of field vectors.
 *    - Writes the computed fields to an HDF5 file, including metadata attributes such as:
 *        + PETGEM version
 *        + Mesh and receivers filenames
 *        + Simulation date
 *        + Source frequency and position
 *        + FEM order and number of MPI tasks
 * 5. Frees all allocated memory and PETSc objects.
 *
 * @note
 * - This function assumes 3D simulations (NUM_DIMENSIONS = 3) and H(curl) elements.
 * - Only receivers located inside the computational domain are considered; others
 *   generate a warning and are ignored.
 * - The magnetic field is computed via H = (1 / (i * omega * mu)) curl(E), following
 *   standard Maxwell equations.
 * - Output files are written in HDF5 format with one file per source, and the filename
 *   is constructed using the output directory, base filename, and source index.
 *
 */
PetscErrorCode computeFields(const csemParams params, const CsemSourceSet sources,
                             const DM dm, const Grid grid,
                             Vec receivers, const Mat X) {
  PetscFunctionBeginUser;

  /* Variable declarations */
  PetscReal omega;
  PetscScalar constFactor;
  PetscInt day, year;
  PetscInt month = 0;
  PetscBool flag;
  Vec Ex, Ey, Ez, Hx, Hy, Hz;
  PetscViewer viewerOutput;
  char date[30], monthStr[4], version[50], idSource[20];
  char formattedDate[11]; // YYYY-MM-DD format (10 chars + null terminator)
  char outFileName[PETSC_MAX_PATH_LEN];
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
  PetscCall(buildReceiverInterpolationMatrices(params.nord,
                                                receivers,
                                                dm, &grid, &Q));

  /* Allocate output Vecs sized to match Q's row layout (left vector). */
  PetscCall(MatCreateVecs(Q.QEx, NULL, &Ex));
  PetscCall(MatCreateVecs(Q.QEy, NULL, &Ey));
  PetscCall(MatCreateVecs(Q.QEz, NULL, &Ez));
  PetscCall(MatCreateVecs(Q.QHx, NULL, &Hx));
  PetscCall(MatCreateVecs(Q.QHy, NULL, &Hy));
  PetscCall(MatCreateVecs(Q.QHz, NULL, &Hz));

  /* Print message */
  PetscCall(PetscPrintf(comm, "\n Compute electric and magnetic fields:\n"));
  PetscCall(PetscPrintf(comm, "   Input file                  = %s\n", params.inputFile));
  PetscCall(PetscPrintf(comm, "   Number of receivers         = %" PetscInt_FMT "\n", Q.numReceivers));
  PetscCall(PetscPrintf(comm, "   Postprocessing status       = Initiated\n"));

  /* Postprocessing fields for each source */
  for (PetscInt i = 0; i < sources.numSources; i++) {

    /* Variable declarations */
    Vec x;

    /* Get the solution column for this source */
    PetscCall(MatDenseGetColumnVecRead(X, i, &x));

    PetscCall(PetscPrintf(comm, "   Computing fields for source = %" PetscInt_FMT "\n", i + 1));

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

    /* Get current date and time */
    PetscCall(PetscGetDate(date, sizeof(date)));

    /* Parse the date to extract month, day, and year */
    sscanf(date, "%*s %3s %" PetscInt_FMT "%*s %" PetscInt_FMT, monthStr, &day, &year);

    /* Compare monthStr with each month abbreviation */
    PetscCall(PetscStrcmp(monthStr, "Jan", &flag));
    if (flag)
      month = 1;
    PetscCall(PetscStrcmp(monthStr, "Feb", &flag));
    if (flag)
      month = 2;
    PetscCall(PetscStrcmp(monthStr, "Mar", &flag));
    if (flag)
      month = 3;
    PetscCall(PetscStrcmp(monthStr, "Apr", &flag));
    if (flag)
      month = 4;
    PetscCall(PetscStrcmp(monthStr, "May", &flag));
    if (flag)
      month = 5;
    PetscCall(PetscStrcmp(monthStr, "Jun", &flag));
    if (flag)
      month = 6;
    PetscCall(PetscStrcmp(monthStr, "Jul", &flag));
    if (flag)
      month = 7;
    PetscCall(PetscStrcmp(monthStr, "Aug", &flag));
    if (flag)
      month = 8;
    PetscCall(PetscStrcmp(monthStr, "Sep", &flag));
    if (flag)
      month = 9;
    PetscCall(PetscStrcmp(monthStr, "Oct", &flag));
    if (flag)
      month = 10;
    PetscCall(PetscStrcmp(monthStr, "Nov", &flag));
    if (flag)
      month = 11;
    PetscCall(PetscStrcmp(monthStr, "Dec", &flag));
    if (flag)
      month = 12;

    /* Format the date as YYYY-MM-DD */
    snprintf(formattedDate, sizeof(formattedDate), "%04" PetscInt_FMT "-%02" PetscInt_FMT "-%02" PetscInt_FMT, year, month, day);

    /* Build output file name robustly */
    PetscCall(PetscStrncpy(outFileName, params.outputDirectory, sizeof(outFileName)));

    /* Add "/" if missing */
    size_t len = strlen(outFileName);
    if (len > 0 && outFileName[len - 1] != '/') {
      PetscCall(PetscStrlcat(outFileName, "/", sizeof(outFileName)));
    }

    PetscCall(PetscStrlcat(outFileName, params.outputFilename, sizeof(outFileName)));
    PetscCall(PetscStrlcat(outFileName, "_src", sizeof(outFileName)));
    snprintf(idSource, sizeof(idSource), "%" PetscInt_FMT, i + 1);
    PetscCall(PetscStrlcat(outFileName, idSource, sizeof(outFileName)));
    PetscCall(PetscStrlcat(outFileName, ".h5", sizeof(outFileName)));

    /* Create hdf5 file */
    PetscCall(PetscPrintf(comm, "   Output filename             = %s\n", outFileName));
    PetscCall(PetscViewerHDF5Open(comm, outFileName, FILE_MODE_WRITE, &viewerOutput));

    /* Field components under /fields/, mirroring the bundle's
     * /fields/model_data layout. */
    PetscCall(PetscObjectSetName((PetscObject)Ex, "Ex"));
    PetscCall(PetscObjectSetName((PetscObject)Ey, "Ey"));
    PetscCall(PetscObjectSetName((PetscObject)Ez, "Ez"));
    PetscCall(PetscObjectSetName((PetscObject)Hx, "Hx"));
    PetscCall(PetscObjectSetName((PetscObject)Hy, "Hy"));
    PetscCall(PetscObjectSetName((PetscObject)Hz, "Hz"));
    PetscCall(PetscViewerHDF5PushGroup(viewerOutput, "/fields"));
    PetscCall(VecView(Ex, viewerOutput));
    PetscCall(VecView(Ey, viewerOutput));
    PetscCall(VecView(Ez, viewerOutput));
    PetscCall(VecView(Hx, viewerOutput));
    PetscCall(VecView(Hy, viewerOutput));
    PetscCall(VecView(Hz, viewerOutput));
    PetscCall(PetscViewerHDF5PopGroup(viewerOutput));

    /* Per-source metadata under /source/, parallel to the bundle's
     * /sources/ group (singular here since each output file is one source). */
    {
      const CsemSource *s = &sources.sourceArray[i];
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, "/source", "frequency",    PETSC_REAL, &sources.freq));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, "/source", "x_pos",        PETSC_REAL, &s->position[0]));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, "/source", "y_pos",        PETSC_REAL, &s->position[1]));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, "/source", "z_pos",        PETSC_REAL, &s->position[2]));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, "/source", "current",      PETSC_REAL, &s->current));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, "/source", "length",       PETSC_REAL, &s->length));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, "/source", "dip_angle",    PETSC_REAL, &s->dipAngle));
      PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, "/source", "azimuth_angle",PETSC_REAL, &s->azimuthAngle));
    }

    /* Top-level provenance attributes - lowercase + underscore, matching
     * the bundle's naming idiom. */
    sprintf(version, "%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "petgem_version", PETSC_STRING, version));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "input_filename", PETSC_STRING, params.inputFile));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "date",           PETSC_STRING, date));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "nord",           PETSC_INT,    &params.nord));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "mpi_tasks",      PETSC_INT,    &params.numMPITasks));

    /* Free memory */
    PetscCall(PetscViewerDestroy(&viewerOutput));
  }

  PetscCall(PetscPrintf(comm, "   Postprocessing status       = Finished\n"));

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
