/*
 * Filename: common.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2024-05-28
 *
 * Description:
 * This file contains a collection of functions for common utility functions that are used
 * throughout the PETGEM toolkit. These functions include operations for printing and timers.
 *
 * Usage:
 * Include this file in your source code to utilize the common functions. 
 * For example:
 * #include "common.h"
 * 
*/

/* C libraries */ 
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <string.h>

/* PETSc libraries */
#include <petscsys.h>

/* PETGEM funcions*/
#include "common.h"
#include "version.h"

// =============================================================================
// Function: printHeader
// =============================================================================

/**
 * @brief Prints a header with PETGEM project information and the current year.
 *
 * This function prints a formatted header to the console, containing
 * information about the PETGEM project, including its name, purpose,
 * GitHub repository, website, and the names and affiliations of the
 * developers. It also includes the current year.
 *
 * The header is printed using PETSc's parallel printing functions,
 * ensuring that the output is consistent across all processes in the
 * PETSc communicator.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful completion, or an error code otherwise.
 */
PetscErrorCode printHeader(){
    
    PetscFunctionBeginUser;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "----------------------------------------------------------------------------\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-                                                                          -\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-                                   PETGEM                                 -\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-          Parallel Edge-based Tool for Electromagnetic Modelling          -\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-                                                                          -\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-          GitHub Repository: github.com/ocastilloreyes/petgem             -\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-                      Website: https://petgem.bsc.es/                     -\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-                                                                          -\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "----------------------------------------------------------------------------\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-                                                                          -\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-                         Octavio Castillo-Reyes                           -\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-            Universitat Politècnica de Catalunya (UPC) - 2024             -\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-               Barcelona Supercomputing Center (BSC) - 2024               -\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-                                                                          -\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "----------------------------------------------------------------------------\n"));

    PetscFunctionReturn(PETSC_SUCCESS);

}

// =============================================================================
// Function: printFooter
// =============================================================================

/**
 * @brief Prints the finalization time of the PETGEM simulation.
 *
 * This function retrieves the current local time and prints it to the
 * console in the format: "Finished: YYYY-MM-DD HH:MM:SS".
 * The printing is done using PETSc's parallel printing functions to ensure
 * consistency across all processes in the PETSc communicator.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful completion, or an error code otherwise.
 */
PetscErrorCode printFooter(){

    PetscFunctionBeginUser;

    /* Variables declaration */
    char date[30];

    /* Get date*/
    PetscCall(PetscGetDate(date, 30));
    
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n Finished: %s", date)); 
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n PETGEM version: %d.%d.%d\n", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)); 
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "----------------------------------------------------------------------------\n"));

    PetscFunctionReturn(PETSC_SUCCESS);

}

// =============================================================================
// Function: createDirectory
// =============================================================================

/**
 * @brief Creates a directory if it does not already exist.
 *
 * This function checks if the specified directory exists. If the directory
 * does not exist, it attempts to create it with the specified permissions.
 * The function uses PETSc error handling to report any issues encountered
 * during the creation of the directory.
 *
 * @param[in] path  A constant character pointer to the path of the directory to be created.
 * @return PetscErrorCode PETSC_SUCCESS on successful completion. If the
 *         directory creation fails, it returns an appropriate PETSc error code.
 */
PetscErrorCode createDirectory(const char *path) {
     
    PetscFunctionBeginUser;
    
    /* Verify if the directory exists */
    struct stat st;
    if (stat(path, &st) == 0) {
        /* Directory exists */
        if (S_ISDIR(st.st_mode)) {
            PetscFunctionReturn(PETSC_SUCCESS);
        } else {
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, " Path exists but is not a directory: %s", path);
        }
    } else {
        /* Directory doesn't exist, create it */
        if (mkdir(path, 0755) && errno != EEXIST) {
            SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, " Error when creating output directory: %s", path);
        }
    }
    
    PetscFunctionReturn(PETSC_SUCCESS);
}