/*
  Filename: assembly.h
  Author: Octavio Castillo Reyes (UPC/BSC)
  Date: 2024-10-02
 
  Description:
  This file contains a collection of definitions for assembly functions that are used
  throughout the PETGEM project.
 
  Usage:
  Include this file in your source code to utilize the assembly functions. 
  For example:
  #include "assembly.h" 
*/

#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#include <petsc.h>
#include "inputs.h"
#include "grid.h"
#include "transmitter.h"

PetscErrorCode assembleCsemSystem(DM dm, Vec resistivity, Grid grid, setCsemSource sources, CsemParams params, Mat *A, Mat *B, Mat *G);

#endif

