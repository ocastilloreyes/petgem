/*
 * Filename: postprocessing.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2025-08-05
 *
 * Description:
 * This file contains functions data postprocessing.
 *
 * Usage:
 * Include this file in your source code to utilize the
 * postprocessing functions. For example: #include
 * "postprocessing.h"
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
PetscErrorCode computeFields(const csemParams params, const CsemSourceSet sources, const DM dm, const Grid grid, const Mat X) {
  PetscFunctionBeginUser;

  /* Variable declarations */
  Cell cell;
  PetscReal *XiEtaZeta, **Ni, **NiCurl, **coeffs, **Dx_Ni, **Dy_Ni, **Dz_Ni;
  PetscReal realReceiverCoords[NUM_DIMENSIONS];
  PetscReal omega;
  PetscScalar* closureReceiver;
  PetscScalar tmpFields[6];
  PetscScalar constFactor;
  PetscInt cellID, globalSizeReceivers, numGlobalReceivers;
  PetscInt numReceiversFoundGlobal, numReceiversFoundLocal;
  PetscInt closureSizeReceiver = grid.numDofInCell;
  PetscInt day, year;
  PetscInt month = 0;
  PetscBool flag;
  PetscSF receiverGlobalSF = NULL;
  Vec xLocal, receivers, Ex, Ey, Ez, Hx, Hy, Hz;
  PetscViewer viewerInput, viewerOutput;
  char date[30], monthStr[4], version[50], idSource[20];
  char formattedDate[11]; // YYYY-MM-DD format (10 chars +
                          // null terminator)
  char outFileName[PETSC_MAX_PATH_LEN];
  const PetscSFNode* receiverInCell;
  const PetscInt* receiverFound;
  const PetscScalar* coords;
  PetscSection section;

  PetscMPIInt rank;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* Compute constant */
  omega = sources.freq * 2.0 * PETSC_PI;
  constFactor = (0.0 + 1.0 * PETSC_i) * (omega * MU);

  /* Load receiver data (sequential) */
  PetscCall(VecCreate(PETSC_COMM_SELF, &receivers));
  PetscCall(PetscObjectSetName((PetscObject)receivers, "receivers"));
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, params.receiversFile, FILE_MODE_READ, &viewerInput));
  PetscCall(VecLoad(receivers, viewerInput));
  PetscCall(VecSetBlockSize(receivers, NUM_DIMENSIONS));
  PetscCall(VecGetSize(receivers, &globalSizeReceivers));

  /* Verify receivers vector consistency */
  PetscCheck(globalSizeReceivers % 3 == 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ,
             "   Error: Global size of the receivers "
             "vector (%" PetscInt_FMT ") is not divisible by 3, which is required "
             "for 3D points.\n",
             globalSizeReceivers);

  numGlobalReceivers = globalSizeReceivers / 3;

  /* Check if all the receivers are within the computational
   * domain */
  PetscCall(DMLocatePoints(dm, receivers, DM_POINTLOCATION_REMOVE, &receiverGlobalSF));
  PetscCall(PetscSFGetGraph(receiverGlobalSF, NULL, &numReceiversFoundLocal, &receiverFound, &receiverInCell));
  PetscCallMPI(MPI_Allreduce(&numReceiversFoundLocal, &numReceiversFoundGlobal, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCheck(numReceiversFoundGlobal > 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG,
             "   Error: Receiver coordinates are either "
             "not found or located outside the "
             "computational domain.\n");
  if (numReceiversFoundGlobal != numGlobalReceivers) {
    PetscCall(PetscPrintf(comm, "   Warning: Some receiver coordinates "
                                "are either not found or located "
                                "outside the computational domain. "
                                "Fields will not be computed for these "
                                "receivers.\n"));
  }

  /* Create vectors for electric and magnetic fields */
  PetscCall(VecCreateMPI(comm, numReceiversFoundLocal, PETSC_DECIDE, &Ex));
  PetscCall(VecCreateMPI(comm, numReceiversFoundLocal, PETSC_DECIDE, &Ey));
  PetscCall(VecCreateMPI(comm, numReceiversFoundLocal, PETSC_DECIDE, &Ez));
  PetscCall(VecCreateMPI(comm, numReceiversFoundLocal, PETSC_DECIDE, &Hx));
  PetscCall(VecCreateMPI(comm, numReceiversFoundLocal, PETSC_DECIDE, &Hy));
  PetscCall(VecCreateMPI(comm, numReceiversFoundLocal, PETSC_DECIDE, &Hz));

  /* Allocate memory */
  PetscCall(PetscMalloc1(grid.numDofInCell, &closureReceiver));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &XiEtaZeta));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &NiCurl));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dx_Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dy_Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dz_Ni));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(grid.numDofInCell, &Ni[i]));
    PetscCall(PetscCalloc1(grid.numDofInCell, &NiCurl[i]));
    PetscCall(PetscCalloc1(grid.numDofInCell, &Dx_Ni[i]));
    PetscCall(PetscCalloc1(grid.numDofInCell, &Dy_Ni[i]));
    PetscCall(PetscCalloc1(grid.numDofInCell, &Dz_Ni[i]));
  }

  PetscCall(PetscCalloc1(grid.numDofInCell, &coeffs));
  for (PetscInt i = 0; i < grid.numDofInCell; i++) {
    PetscCall(PetscCalloc1(grid.numDofInCell, &coeffs[i]));
  }

  /* Create the const views for arrays */
  const PetscReal** coeffs_const = (const PetscReal**)coeffs;
  const PetscReal** Dx_Ni_const = (const PetscReal**)Dx_Ni;
  const PetscReal** Dy_Ni_const = (const PetscReal**)Dy_Ni;
  const PetscReal** Dz_Ni_const = (const PetscReal**)Dz_Ni;

  /* Get local vector */
  PetscCall(DMGetLocalVector(dm, &xLocal));

  /* Get section */
  PetscCall(DMGetLocalSection(dm, &section));

  /* Get receiver coordinates*/
  PetscCall(VecGetArrayRead(receivers, &coords));

  /* Print message */
  PetscCall(PetscPrintf(comm, "\n Compute electric and magnetic fields:\n"));
  PetscCall(PetscPrintf(comm, "   Receivers filename          = %s\n", params.receiversFile));
  PetscCall(PetscPrintf(comm, "   Number of receivers         = %" PetscInt_FMT "\n", numGlobalReceivers));
  PetscCall(PetscPrintf(comm, "   Postprocessing status       = Initiated\n"));

  /* Postprocessing fields for each source */
  for (PetscInt i = 0; i < sources.numSources; i++) {

    /* Variable declarations */
    Vec x;

    /* Get data vector and reset output vectors */
    PetscCall(MatDenseGetColumnVecRead(X, i, &x));
    PetscCall(DMGlobalToLocal(dm, x, INSERT_VALUES, xLocal));
    PetscCall(MatDenseRestoreColumnVecRead(X, i, &x));
    PetscCall(VecSet(Ex, 0.0));
    PetscCall(VecSet(Ey, 0.0));
    PetscCall(VecSet(Ez, 0.0));
    PetscCall(VecSet(Hx, 0.0));
    PetscCall(VecSet(Hy, 0.0));
    PetscCall(VecSet(Hz, 0.0));

    PetscCall(PetscPrintf(comm, "   Computing fields for source = %" PetscInt_FMT "\n", i + 1));

    /* Compute fields for receivers */
    for (PetscInt j = 0; j < numReceiversFoundLocal; j++) {
      /* Get cell index in which receiver belongs */
      PetscInt ridx = receiverFound ? receiverFound[j] : j;

      cellID = receiverInCell[j].index;
      realReceiverCoords[0] = PetscRealPart(coords[3 * ridx]);
      realReceiverCoords[1] = PetscRealPart(coords[3 * ridx + 1]);
      realReceiverCoords[2] = PetscRealPart(coords[3 * ridx + 2]);

      /* Get vertices coordinates for cellID */
      PetscCall(extractCellCoordinates(dm, cellID, &cell));

      /* Compute jacobian, inverse jacobian and jacobian
       * determinand for cellID */
      PetscCall(computeCellJacobian(&cell));

      /* Get transitive clousure for cell i */
      PetscCall(extractCellClousure(dm, cellID, &cell));

      /* Compute cell orientation */
      PetscCall(computeCellOrientation(&cell));

      /* Transform xyz source position to XiEtaZeta
       * coordinates (reference tetrahedral element) */
      PetscCall(tetrahedronXYZToReference(cell.coordinates, realReceiverCoords, XiEtaZeta));

      /* Compute basis functions */
      switch (params.nord) {
      case 1:
        /* Compute nedelec coefficients and its derivatives
         */
        PetscCall(computeNedelecOrder1Coefficients(params.nord, coeffs, Dx_Ni, Dy_Ni, Dz_Ni));

        /* Compute basis functions */
        PetscCall(computeNedelecOrder1BasisFunctions(params.nord, XiEtaZeta, (const PetscReal(*)[NUM_DIMENSIONS])cell.jacobian,
                                                     coeffs_const, Ni));

        /* Compute curl basis functions */
        PetscCall(computeNedelecOrder1BasisFunctionCurls(params.nord, Dx_Ni_const, Dy_Ni_const, Dz_Ni_const,
                                                         (const PetscReal(*)[NUM_DIMENSIONS])cell.jacobian, cell.detJacobian, NiCurl));

        break;
      case 2:
        break;
      default:
        break;
      }

      /* Get clousure for receiverInCell */
      PetscCall(DMPlexVecGetClosure(dm, section, xLocal, cellID, &closureSizeReceiver, &closureReceiver));

      /* Reset variables to zero */
      for (PetscInt k = 0; k < NUM_EM_FIELD_COMPONENTS; k++) {
        tmpFields[k] = 0.0 + PETSC_i * 0.0;
      }

      /* Interpolate fields at receiver i */
      for (PetscInt k = 0; k < grid.numDofInCell; k++) {
        tmpFields[0] += (Ni[0][k] * closureReceiver[k] * cell.orientation[4 + k]);     /* Ex */
        tmpFields[1] += (Ni[1][k] * closureReceiver[k] * cell.orientation[4 + k]);     /* Ey */
        tmpFields[2] += (Ni[2][k] * closureReceiver[k] * cell.orientation[4 + k]);     /* Ez */
        tmpFields[3] += (NiCurl[0][k] * closureReceiver[k] * cell.orientation[4 + k]); /* Hx */
        tmpFields[4] += (NiCurl[1][k] * closureReceiver[k] * cell.orientation[4 + k]); /* Hy */
        tmpFields[5] += (NiCurl[2][k] * closureReceiver[k] * cell.orientation[4 + k]); /* Hz */
      }

      /* Following Maxwell equations, compute H fields */
      tmpFields[3] /= constFactor;
      tmpFields[4] /= constFactor;
      tmpFields[5] /= constFactor;

      /* Set values to output vectors */
      PetscCall(VecSetValue(Ex, ridx, tmpFields[0], INSERT_VALUES));
      PetscCall(VecSetValue(Ey, ridx, tmpFields[1], INSERT_VALUES));
      PetscCall(VecSetValue(Ez, ridx, tmpFields[2], INSERT_VALUES));
      PetscCall(VecSetValue(Hx, ridx, tmpFields[3], INSERT_VALUES));
      PetscCall(VecSetValue(Hy, ridx, tmpFields[4], INSERT_VALUES));
      PetscCall(VecSetValue(Hz, ridx, tmpFields[5], INSERT_VALUES));
    }

    /* Perform global assembly */
    PetscCall(VecAssemblyBegin(Ex));
    PetscCall(VecAssemblyBegin(Ey));
    PetscCall(VecAssemblyBegin(Ez));
    PetscCall(VecAssemblyBegin(Hx));
    PetscCall(VecAssemblyBegin(Hy));
    PetscCall(VecAssemblyBegin(Hz));

    PetscCall(VecAssemblyEnd(Ex));
    PetscCall(VecAssemblyEnd(Ey));
    PetscCall(VecAssemblyEnd(Ez));
    PetscCall(VecAssemblyEnd(Hx));
    PetscCall(VecAssemblyEnd(Hy));
    PetscCall(VecAssemblyEnd(Hz));

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

    /* Write output vectors */
    PetscCall(PetscObjectSetName((PetscObject)Ex, "Ex"));
    PetscCall(PetscObjectSetName((PetscObject)Ey, "Ey"));
    PetscCall(PetscObjectSetName((PetscObject)Ez, "Ez"));
    PetscCall(PetscObjectSetName((PetscObject)Hx, "Hx"));
    PetscCall(PetscObjectSetName((PetscObject)Hy, "Hy"));
    PetscCall(PetscObjectSetName((PetscObject)Hz, "Hz"));
    PetscCall(VecView(Ex, viewerOutput));
    PetscCall(VecView(Ey, viewerOutput));
    PetscCall(VecView(Ez, viewerOutput));
    PetscCall(VecView(Hx, viewerOutput));
    PetscCall(VecView(Hy, viewerOutput));
    PetscCall(VecView(Hz, viewerOutput));

    /* Write attributes for data provedance */
    sprintf(version, "%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "Petgem_version", PETSC_STRING, version));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "Mesh_filename", PETSC_STRING, params.meshFile));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "Receivers_filename", PETSC_STRING, params.receiversFile));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "Date", PETSC_STRING, date));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "Nord", PETSC_INT, &params.nord));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "MPI_tasks", PETSC_INT, &params.numMPITasks));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "Source_frequency", PETSC_REAL, &sources.freq));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "Source_x_pos", PETSC_REAL, &sources.sourceArray[i].position[0]));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "Source_y_pos", PETSC_REAL, &sources.sourceArray[i].position[1]));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "Source_z_pos", PETSC_REAL, &sources.sourceArray[i].position[2]));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "Source_current", PETSC_REAL, &sources.sourceArray[i].current));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "Source_length", PETSC_REAL, &sources.sourceArray[i].length));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "Source_dip", PETSC_REAL, &sources.sourceArray[i].dipAngle));
    PetscCall(PetscViewerHDF5WriteAttribute(viewerOutput, NULL, "Source_azimuth", PETSC_REAL, &sources.sourceArray[i].azimuthAngle));

    /* Free memory */
    PetscCall(PetscViewerDestroy(&viewerOutput));
  }

  /* Restore local and global vector */
  PetscCall(DMRestoreLocalVector(dm, &xLocal));
  PetscCall(VecRestoreArrayRead(receivers, &coords));

  PetscCall(PetscPrintf(comm, "   Postprocessing status       = Finished\n"));

  /* Free memory */
  PetscCall(PetscViewerDestroy(&viewerInput));

  PetscCall(PetscSFDestroy(&receiverGlobalSF));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscFree(NiCurl[i]));
    PetscCall(PetscFree(Ni[i]));
    PetscCall(PetscFree(Dx_Ni[i]));
    PetscCall(PetscFree(Dy_Ni[i]));
    PetscCall(PetscFree(Dz_Ni[i]));
  }
  PetscCall(PetscFree(Ni));
  PetscCall(PetscFree(NiCurl));
  PetscCall(PetscFree(Dx_Ni));
  PetscCall(PetscFree(Dy_Ni));
  PetscCall(PetscFree(Dz_Ni));
  PetscCall(PetscFree(closureReceiver));
  PetscCall(PetscFree(XiEtaZeta));
  PetscCall(VecDestroy(&receivers));
  PetscCall(VecDestroy(&Ex));
  PetscCall(VecDestroy(&Ey));
  PetscCall(VecDestroy(&Ez));
  PetscCall(VecDestroy(&Hx));
  PetscCall(VecDestroy(&Hy));
  PetscCall(VecDestroy(&Hz));

  for (PetscInt i = 0; i < grid.numDofInCell; i++) {
    PetscCall(PetscFree(coeffs[i]));
  }
  PetscCall(PetscFree(coeffs));

  PetscFunctionReturn(PETSC_SUCCESS);
}
