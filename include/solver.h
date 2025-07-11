/*
  Filename: solver.h
  Author: Octavio Castillo Reyes (UPC/BSC)
  Date: 2024-10-02
 
  Description:
  This file contains a collection of definitions for solver functions that are used
  throughout the PETGEM project.

  Usage:
  Include this file in your source code to utilize the solver functions. 
  For example:
  #include "solver.h" 
*/

#ifndef SOLVER_H
#define SOLVER_H

#include <petsc.h>
#include <petscdmplex.h>
#include "inputs.h"

/* =============================================================================
   Declaration of functions
   =============================================================================
*/   
PetscErrorCode solveSystem(DM dm, Mat A, Mat B, Mat G, Params params, Mat *X);

#endif
