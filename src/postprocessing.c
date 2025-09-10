/*
 * Filename: postprocessing.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2025-08-05
 *
 * Description:
 * This file contains functions data postprocessing. 
 *
 * Usage:
 * Include this file in your source code to utilize the postprocessing functions. 
 * For example:
 * #include "postprocessing.h"
*/

/* C libraries */ 
#include <time.h>

/* PETSc libraries */
#include <petscsys.h>
#include <petscviewerhdf5.h>

/* PETGEM functions */ 
#include "constants.h"
#include "inputs.h"
#include "grid.h"
#include "hvfem.h"
#include "postprocessing.h"
#include "version.h"


/**
 * @brief Computes electric (E) and magnetic (H) fields at specified receiver locations.
 *
 * The computed electric (E) and magnetic (H) field components for each source are saved
 * to separate HDF5 files. Metadata about the simulation is also written as attributes
 * to the output files.
 *
 * @param[in] dm The DMPlex object representing the mesh topology and H(curl) discretization.
 * @param[in] X The solution matrix (Mat), where each column corresponds to the solution vector for a specific source.
 * @param[in] grid A Grid struct containing mesh statistics and DOF information.
 * @param[in] sources A setSource struct containing source parameters (frequency, positions, etc.).
 * @param[in] params A Params struct containing simulation parameters (basis order, mode, output settings, etc.).
 * @return PetscErrorCode PETSC_SUCCESS on success, or an error code otherwise.
 */
PetscErrorCode computeFields(DM dm, Mat X, Grid grid, CsemSourceSet sources, Params params){
    PetscFunctionBeginUser;
    MPI_Comm comm = PetscObjectComm((PetscObject)dm);
    
    PetscCall(PetscPrintf(comm, "\n Compute electric and magnetic fields:\n"));
    PetscCall(PetscPrintf(comm, "   Receivers filename      = %s\n", params.receiversFile));
        
    /* Variable declarations */
    PetscReal   jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS];
    PetscReal   **basisFunctions, **curlBasisFunctions, *XiEtaZeta;
    PetscReal   realReceiverCoords[NUM_DIMENSIONS];
    PetscReal   omega;

    PetscScalar *cellCoords = NULL, *closureReceiver;
    PetscScalar tmpFields[6];
    PetscScalar constFactor;

    PetscInt    numCoords, cell, globalSizeReceivers, numGlobalReceivers;
    PetscInt    numReceiversFoundGlobal, numReceiversFoundLocal;
    PetscInt    closureSizeReceiver = grid.numDofInCell;
    PetscInt    cellOrientation[10];
    PetscInt    day, year;
    PetscInt    month = 0;
    PetscMPIInt rank;

    PetscBool   isDG, flag;
    PetscSF     receiverGlobalSF = NULL;
    
    Vec         xLocal, receivers, Ex, Ey, Ez, Hx, Hy, Hz;

    PetscViewer viewerInput, viewerOutput; 

    /* Compute constant */
    omega = sources.freq * 2.0 * PETSC_PI;
    constFactor = (0.0 + 1.0*PETSC_i) * (omega * MU);      
    
    char  date[30];
    char  formattedDate[11];  // YYYY-MM-DD format (10 chars + null terminator)
    char  monthStr[4];
    char  version[50];
    char  outFileName[PETSC_MAX_PATH_LEN];
    char  idSource[20];
    const PetscSFNode   *receiverInCell;
    const PetscInt      *receiverFound;   
    const PetscScalar   *arrayCoords, *coords;
    PetscSection section;

    /* Load receiver data (sequential) */     
    PetscCall(VecCreate(PETSC_COMM_SELF, &receivers));
    PetscCall(PetscObjectSetName((PetscObject)receivers,"receivers"));
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, params.receiversFile, FILE_MODE_READ, &viewerInput));
    PetscCall(VecLoad(receivers, viewerInput));
    PetscCall(VecSetBlockSize(receivers, NUM_DIMENSIONS));
    PetscCall(VecGetSize(receivers, &globalSizeReceivers));
    
    /* Verify receivers vector consistency */
    PetscCheck(globalSizeReceivers % 3 == 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_SIZ, "   Error: Global size of the receivers vector (%" PetscInt_FMT ") is not divisible by 3, which is required for 3D points.\n", globalSizeReceivers);
    
    numGlobalReceivers = globalSizeReceivers/3;
    PetscCall(PetscPrintf(comm, "   Number of receivers     = %" PetscInt_FMT "\n", numGlobalReceivers));
    
    /* Check if all the receivers are within the computational domain */
    PetscCall(DMLocatePoints(dm, receivers, DM_POINTLOCATION_REMOVE, &receiverGlobalSF));
    PetscCall(PetscSFGetGraph(receiverGlobalSF, NULL, &numReceiversFoundLocal, &receiverFound, &receiverInCell));
    PetscCallMPI(MPI_Allreduce(&numReceiversFoundLocal, &numReceiversFoundGlobal, 1, MPIU_INT, MPI_SUM, comm));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));

    PetscCheck(numReceiversFoundGlobal > 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "   Error: Receiver coordinates are either not found or located outside the computational domain.\n");
    if (numReceiversFoundGlobal != numGlobalReceivers){
        PetscCall(PetscPrintf(comm, "   Warning: Some receiver coordinates are either not found or located outside the computational domain. Fields will not be computed for these receivers.\n"));
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
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &basisFunctions));
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &curlBasisFunctions));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscCalloc1(grid.numDofInCell, &basisFunctions[i]));
        PetscCall(PetscCalloc1(grid.numDofInCell, &curlBasisFunctions[i]));
    }

    /* Get local vector */
    PetscCall(DMGetLocalVector(dm, &xLocal));
    
    /* Get section */
    PetscCall(DMGetLocalSection(dm, &section));

    /* Get receiver coordinates*/
    PetscCall(VecGetArrayRead(receivers, &coords));

    PetscCall(PetscPrintf(comm, "   Postprocessing status   = Initiated\n"));

    /* Postprocessing fields for each source */
    for (PetscInt i = 0; i < sources.numSources; i++) {
        PetscCall(PetscPrintf(comm, "       Computing fields for source %" PetscInt_FMT "\n", i+1));

        Vec x;
        PetscCall(MatDenseGetColumnVecRead(X, i, &x)); 
        PetscCall(DMGlobalToLocal(dm, x, INSERT_VALUES, xLocal));
        PetscCall(MatDenseRestoreColumnVecRead(X, i, &x)); 

        /* Reset vectors */
        PetscCall(VecSet(Ex, 0.0));
        PetscCall(VecSet(Ey, 0.0));
        PetscCall(VecSet(Ez, 0.0));
        PetscCall(VecSet(Hx, 0.0));
        PetscCall(VecSet(Hy, 0.0));
        PetscCall(VecSet(Hz, 0.0));

        /* Compute fields for receivers */
        for (PetscInt j = 0; j < numReceiversFoundLocal; j++){
            /* Get cell index in which receiver belongs */
            PetscInt ridx = receiverFound ? receiverFound[j] : j;

            cell = receiverInCell[j].index;
            realReceiverCoords[0] = PetscRealPart(coords[3*ridx]);
            realReceiverCoords[1] = PetscRealPart(coords[3*ridx + 1]);
            realReceiverCoords[2] = PetscRealPart(coords[3*ridx + 2]);
    
            /* Get cell coordinates */ 
            PetscCall(DMPlexGetCellCoordinates(dm, cell, &isDG, &numCoords, &arrayCoords, &cellCoords));

            /* Compute jacobian and its inverse for receiverInCell */
            PetscCall(computeJacobian(cellCoords, jacobian, invJacobian));
        
            /* Transform xyz receiver position to XiEtaZeta coordinates (reference tetrahedral element) */
            PetscCall(tetrahedronXYZToXiEtaZeta(cellCoords, realReceiverCoords, XiEtaZeta));
        
            /* Restore cell coordinates */ 
            PetscCall(DMPlexRestoreCellCoordinates(dm, cell, &isDG, &numCoords, &arrayCoords, &cellCoords));

            /* Compute cell orientation */
            PetscCall(computeCellOrientation(dm, cell, cellOrientation));

            /* Compute basis functions for receiverInCell */
            PetscCall(computeBasisFunctions(params.nord, cellOrientation, jacobian, invJacobian, XiEtaZeta, basisFunctions, curlBasisFunctions));

            /* Get clousure for receiverInCell */
            PetscCall(DMPlexVecGetClosure(dm, section, xLocal, cell, &closureSizeReceiver, &closureReceiver));

            /* Reset variables to zero */
            for (PetscInt k = 0; k < 6; k++){
                tmpFields[k] = 0.0 + PETSC_i*0.0;
            }

            /* Interpolate fields at receiver i */
            for (PetscInt k=0; k<grid.numDofInCell; k++){
                tmpFields[0] += (basisFunctions[0][k] * closureReceiver[k]);        /* Ex */ 
                tmpFields[1] += (basisFunctions[1][k] * closureReceiver[k]);        /* Ey */ 
                tmpFields[2] += (basisFunctions[2][k] * closureReceiver[k]);        /* Ez */ 
                tmpFields[3] += (curlBasisFunctions[0][k] * closureReceiver[k]);    /* Hx */ 
                tmpFields[4] += (curlBasisFunctions[1][k] * closureReceiver[k]);    /* Hy */ 
                tmpFields[5] += (curlBasisFunctions[2][k] * closureReceiver[k]);    /* Hz */ 
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
        PetscCall(PetscStrcmp(monthStr, "Jan", &flag)); if (flag) month = 1;
        PetscCall(PetscStrcmp(monthStr, "Feb", &flag)); if (flag) month = 2;
        PetscCall(PetscStrcmp(monthStr, "Mar", &flag)); if (flag) month = 3;
        PetscCall(PetscStrcmp(monthStr, "Apr", &flag)); if (flag) month = 4;
        PetscCall(PetscStrcmp(monthStr, "May", &flag)); if (flag) month = 5;
        PetscCall(PetscStrcmp(monthStr, "Jun", &flag)); if (flag) month = 6;
        PetscCall(PetscStrcmp(monthStr, "Jul", &flag)); if (flag) month = 7;
        PetscCall(PetscStrcmp(monthStr, "Aug", &flag)); if (flag) month = 8;
        PetscCall(PetscStrcmp(monthStr, "Sep", &flag)); if (flag) month = 9;
        PetscCall(PetscStrcmp(monthStr, "Oct", &flag)); if (flag) month = 10;
        PetscCall(PetscStrcmp(monthStr, "Nov", &flag)); if (flag) month = 11;
        PetscCall(PetscStrcmp(monthStr, "Dec", &flag)); if (flag) month = 12;

        /* Format the date as YYYY-MM-DD */
        snprintf(formattedDate, sizeof(formattedDate), "%04" PetscInt_FMT "-%02" PetscInt_FMT "-%02" PetscInt_FMT, year, month, day);

        /* Build output file name robustly */
        PetscCall(PetscStrncpy(outFileName, params.outputDirectory, sizeof(outFileName)));

        /* Add "/" if missing */
        size_t len = strlen(outFileName);
        if (len > 0 && outFileName[len-1] != '/') {
            PetscCall(PetscStrlcat(outFileName, "/", sizeof(outFileName)));
        }

        PetscCall(PetscStrlcat(outFileName, params.outputFilename, sizeof(outFileName)));
        PetscCall(PetscStrlcat(outFileName, "_src", sizeof(outFileName)));
        snprintf(idSource, sizeof(idSource), "%" PetscInt_FMT, i+1);    
        PetscCall(PetscStrlcat(outFileName, idSource, sizeof(outFileName)));
        PetscCall(PetscStrlcat(outFileName, ".h5", sizeof(outFileName)));

        /* Create hdf5 file */
        PetscCall(PetscPrintf(comm, "       Output filename   = %s\n", outFileName));
        PetscCall(PetscViewerHDF5Open(comm, outFileName, FILE_MODE_WRITE, &viewerOutput));    
    
        /* Write output vectors */
        PetscCall(PetscObjectSetName((PetscObject)Ex,"Ex"));
        PetscCall(PetscObjectSetName((PetscObject)Ey,"Ey"));
        PetscCall(PetscObjectSetName((PetscObject)Ez,"Ez"));
        PetscCall(PetscObjectSetName((PetscObject)Hx,"Hx"));
        PetscCall(PetscObjectSetName((PetscObject)Hy,"Hy"));
        PetscCall(PetscObjectSetName((PetscObject)Hz,"Hz"));
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
    
    PetscCall(PetscPrintf(comm, "   Postprocessing status   = Finished\n"));

    /* Free memory */
    PetscCall(PetscViewerDestroy(&viewerInput));
    
    PetscCall(PetscSFDestroy(&receiverGlobalSF));
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++){
        PetscCall(PetscFree(curlBasisFunctions[i]));
        PetscCall(PetscFree(basisFunctions[i]));        
    }
    PetscCall(PetscFree(curlBasisFunctions));
    PetscCall(PetscFree(basisFunctions));    
    PetscCall(PetscFree(closureReceiver));    
    PetscCall(PetscFree(XiEtaZeta));
    PetscCall(VecDestroy(&receivers));    
    PetscCall(VecDestroy(&Ex));
    PetscCall(VecDestroy(&Ey));
    PetscCall(VecDestroy(&Ez));
    PetscCall(VecDestroy(&Hx));
    PetscCall(VecDestroy(&Hy));
    PetscCall(VecDestroy(&Hz));
    
    PetscFunctionReturn(PETSC_SUCCESS);
}
