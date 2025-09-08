/*
  Filename: postprocessing.h
  Author: Octavio Castillo Reyes (UPC/BSC)
  Date: 2025-09-05
 
  Description:
  This file contains a collection of definitions for postprocessing functions that are used
  throughout the PETGEM project.

  Usage:
  Include this file in your source code to utilize the postprocessing functions. 
  For example:
  #include "postprocessing.h" 
*/

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <petsc.h>
#include <petscdmplex.h>
#include "inputs.h"
#include "grid.h"
#include "transmitter.h"

PetscErrorCode computeFields(DM dm, Mat X, Grid grid, setSource source, Params params);

#endif
