/*
  Filename: inputs.h
  Author: Octavio Castillo Reyes (UPC/BSC)
  Date: 2024-10-02
 
  Description:
  This file contains a collection of definitions for user input functions that are used
  throughout the PETGEM project. These functions include operations for printing and timers.

  Usage:
  Include this file in your source code to utilize the input functions. 
  For example:
  #include "inputs.h" 
*/

#ifndef INPUTS_H
#define INPUTS_H

/* =============================================================================
   Declaration of structures
   =============================================================================
*/
typedef struct {
    char meshFile[PETSC_MAX_PATH_LEN];
    char receiversFile[PETSC_MAX_PATH_LEN];
    char outputDirectory[PETSC_MAX_PATH_LEN];
    char outputFilename[PETSC_MAX_PATH_LEN];
    char sourceFilename[PETSC_MAX_PATH_LEN];
    char mode[PETSC_MAX_PATH_LEN];
    PetscInt nord;
    PetscMPIInt numMPITasks;
} Params;


/* =============================================================================
   Declaration of functions
   =============================================================================
*/
PetscErrorCode readUserParams(Params *params, PetscMPIInt size);

#endif

