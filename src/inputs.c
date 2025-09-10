/*
 * Filename: solver.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2025-09-04
 *
 * Description:
 * This file contains functions for handling input data and user parameters in a PETGEM simulation. 
 * It includes functions for parsing input data, and processing user-provided parameters. 
 * The functions in this file facilitate the setup and configuration of the PETGEM code.
 *
 * Usage:
 * Include this file in your source code to utilize the input functions. 
 * For example:
 * #include "inputs.h"
*/

/* C libraries */ 
#include <time.h>
#include <stdio.h>
#include <sys/stat.h>

/* PETSc libraries */
#include <petscsys.h>

/* PETGEM functions */ 
#include "common.h"
#include "inputs.h"  


/**
 * @brief Reads and validates user-provided parameters from PETSc options.
 * @param[out] params Pointer to the CsemParams struct to be populated.
 * @param[in] size The total number of MPI tasks (MPI_Comm_size).
 * @return PetscErrorCode PETSC_SUCCESS on successful parsing and validation.
 *         Returns error codes if mandatory parameters are missing or invalid.
 * @details Parses command-line options or options file entries for:
 *          - `-mesh_filename`: Path to the mesh file (mandatory).
 *          - `-receivers_filename`: Path to the receivers file (mandatory).
 *          - `-output_dir`: Output directory path (mandatory).
 *          - `-output_filename`: Base name for output files (mandatory).
 *          - `-nord`: Finite element basis order (1-6, mandatory).
  *          - `-source_filename`: Path to the source definition file (mandatory).
 *          Stores the parsed values in the `params` struct. Validates mandatory parameters
 *          and the range/type of `nord` and `mode`. Stores the MPI size. Creates the output directory.
 */
PetscErrorCode readCsemParams(CsemParams *params, PetscMPIInt size) {

    PetscFunctionBeginUser;
    
    /* Variables declaration */
    char  meshFilename[PETSC_MAX_PATH_LEN];
    char  receiversFilename[PETSC_MAX_PATH_LEN];
    char  outputDir[PETSC_MAX_PATH_LEN];
    char  outputFilename[PETSC_MAX_PATH_LEN];
    char  sourceFilename[PETSC_MAX_PATH_LEN];
    PetscBool   meshFilenameIsPresent, receiversFilenameIsPresent, nordIsPresent, sourceFilenameIsPresent;
    PetscBool   outputDirIsPresent, outputFilenameIsPresent;
    PetscInt    nord; /* Basis order = 1, 2, 3, 4, 5, 6 */    
    
    /* Read mesh filename (hdf5 format) */
    PetscCall(PetscOptionsGetString(NULL, NULL, "-mesh_filename", meshFilename, sizeof(meshFilename), &meshFilenameIsPresent));  
    PetscCheck(meshFilenameIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Exiting: Mesh file missing. Mandatory parameter required for simulation.\n");
    PetscCall(PetscStrncpy(params->meshFile, meshFilename, sizeof(params->meshFile)));

    /* Read receivers filename (hdf5 format) */
    PetscCall(PetscOptionsGetString(NULL, NULL, "-receivers_filename", receiversFilename, sizeof(receiversFilename), &receiversFilenameIsPresent));  
    PetscCheck(receiversFilenameIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Exiting: Receivers file missing. Mandatory parameter required for simulation.\n");
    PetscCall(PetscStrncpy(params->receiversFile, receiversFilename, sizeof(params->receiversFile)));

    /* Read output directory */
    PetscCall(PetscOptionsGetString(NULL, NULL, "-output_dir", outputDir, sizeof(outputDir), &outputDirIsPresent));  
    PetscCheck(outputDirIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Exiting: Output directory missing. Mandatory parameter required for simulation.\n");
    PetscCall(PetscStrncpy(params->outputDirectory, outputDir, sizeof(params->outputDirectory)));

    /* Read output filename */
    PetscCall(PetscOptionsGetString(NULL, NULL, "-output_filename", outputFilename, sizeof(outputFilename), &outputFilenameIsPresent));  
    PetscCheck(outputFilenameIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Exiting: Output filename missing. Mandatory parameter required for simulation.\n");
    PetscCall(PetscStrncpy(params->outputFilename, outputFilename, sizeof(params->outputFilename)));
    
    /* Read basis order (nord = 1, 2, 3, 4, 5, 6) */
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-nord", &nord, &nordIsPresent));
    PetscCheck(nordIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Exiting: Nord parameter missing. Mandatory parameter required for simulation.\n");
    /* Check is a valid basis order */
    nordIsPresent = (nord >= 1 && nord <= 6) ? PETSC_TRUE : PETSC_FALSE;
    PetscCheck(nordIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Exiting: Nord parameter out of valid range (nord = 1, 2, 3, 4, 5, 6).\n");
    params->nord = nord;
    
    /* Read source filename */
    PetscCall(PetscOptionsGetString(NULL, NULL, "-source_filename", sourceFilename, sizeof(sourceFilename), &sourceFilenameIsPresent));  
    PetscCheck(sourceFilenameIsPresent, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Exiting: Source filename missing. Mandatory parameter required for simulation.\n");
    PetscCall(PetscStrncpy(params->sourceFilename, sourceFilename, sizeof(params->sourceFilename)));
    
    /* Number of MPI tasks */ 
    params->numMPITasks = size;
   
    /* Create output directory */
    createDirectory(outputDir);
    
    PetscFunctionReturn(PETSC_SUCCESS);
}
