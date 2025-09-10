/*
 * Filename: transmitter.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2025-08-05
 *
 * Description:
 * This file contains functions for transmitter (CSEM or MT) in a PETGEM simulation. 
 * It includes functions for parsing source data based on user-provided parameters. 
 * The functions in this file facilitate the setup and configuration of the PETGEM code.
 *
 * Usage:
 * Include this file in your source code to utilize the transmitter functions. 
 * For example:
 * #include "transmitter.h" 
*/

/* C libraries */ 

/* PETSc libraries */
#include <petscsys.h>

/* PETGEM functions */ 
#include "transmitter.h"


/**
 * @brief Reads CSEM source parameters from a file.
 *
 * Opens the file specified in @p params.sourceFilename. Reads the number of sources
 * and the source frequency. Allocates memory for @p sources->sourceArray.
 *
 * @param[out] sources Pointer to the setCsemSource struct to be populated.
 * @param[in] params A CsemParams struct containing simulation parameters, including source filename.
 * @return PetscErrorCode PETSC_SUCCESS on successful reading and parsing. Returns error codes on file open/read errors or format inconsistencies.
 */
PetscErrorCode setupCsemSource(CsemSourceSet* sources, Params params) {
    PetscFunctionBeginUser;

    /* Variables declaration */
    PetscInt    ret; 
    FILE        *inputFile;
    
    /* Open source file */
    inputFile = fopen(params.sourceFilename, "r");
    PetscCheck(inputFile, PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, "Exiting: Error opening source file.\n");

    /* Read the number of sources */
    ret = fscanf(inputFile, "%" PetscInt_FMT "", &sources->numSources);
    PetscCheck(ret == 1, PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "Exiting: Error reading number of sources.\n");

    /* Allocate memory for the sources */
    PetscCall(PetscMalloc1(sources->numSources, &sources->sourceArray));

    /* Read source frequency */
    ret = fscanf(inputFile, "%lf", &sources->freq);
    PetscCheck(ret == 1, PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "Exiting: Error reading source frequency.\n");

    /* Read the data from the file into the CSEM sources array */
    for (PetscInt i = 0; i < sources->numSources; i++) {
        ret = fscanf(inputFile, "%lf %lf %lf %lf %lf %lf %lf", 
            &sources->sourceArray[i].position[0], &sources->sourceArray[i].position[1],
            &sources->sourceArray[i].position[2], &sources->sourceArray[i].current, 
            &sources->sourceArray[i].length, &sources->sourceArray[i].dipAngle, 
            &sources->sourceArray[i].azimuthAngle);
        PetscCheck(ret == 7, PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "Exiting: Error reading CSEM source data. Verify source file format.\n");
    }
        
    /* Close file */
    fclose(inputFile);

    /* Print source data */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nCSEM source data:\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Freq (Hz)         = %g\n", sources->freq));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Number of sources = %" PetscInt_FMT "\n", sources->numSources));
    for (PetscInt i = 0; i < sources->numSources; i++) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Data for source %" PetscInt_FMT ":\n", i+1));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Current         = %g\n", sources->sourceArray[i].current));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Length          = %g\n", sources->sourceArray[i].length));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Dip             = %g\n", sources->sourceArray[i].dipAngle));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Azimuth         = %g\n", sources->sourceArray[i].azimuthAngle));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "     Position (xyz)  = [%g, %g, %g]\n", sources->sourceArray[i].position[0], sources->sourceArray[i].position[1], sources->sourceArray[i].position[2]));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}
