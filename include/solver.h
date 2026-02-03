/*
  Filename: solver.h
  Author: Octavio Castillo Reyes (UPC/BSC)
  Date: 2025-09-05

  Description:
  This file contains a collection of definitions for solver
  functions that are used throughout the PETGEM project.

  Usage:
  Include this file in your source code to utilize the
  solver functions. For example: #include "solver.h"
*/

#ifndef SOLVER_H
#define SOLVER_H

#include "inputs.h"
#include <petsc.h>
#include <petscdmplex.h>

PetscErrorCode solveCsemSystem(const csemParams params, const DM dm, const Mat A, const Mat B, const Mat G, Mat* X);

#endif
