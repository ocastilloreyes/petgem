/*
  Filename: hvfem.h
  Author: Octavio Castillo Reyes (UPC/BSC)
  Date: 2024-10-02
 
  Description:
  This file contains a collection of definitions for high-order vector finite element functions that 
  are used throughout the PETGEM project.
 
  Usage:
  Include this file in your source code to utilize the hvfem functions. 
  For example:
  #include "hvfem.h" 
*/

#ifndef HVFEM_H
#define HVFEM_H

#include <petsc.h>
#include <petscdmplex.h>
#include "constants.h"

/* =============================================================================
   Declaration of functions
   =============================================================================
*/
PetscErrorCode computeJacobian(PetscScalar *cellCoords, PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS]);

PetscErrorCode computeNumGaussPoints3D(PetscInt nord, PetscInt *numGaussPoints);

PetscErrorCode computeGaussPoints3D(PetscInt numPoints, PetscReal **points, PetscReal *weights);

PetscErrorCode vectorRotation(PetscReal azimuth, PetscReal dip, PetscReal rotationVector[NUM_DIMENSIONS]);

PetscErrorCode tetrahedronXYZToXiEtaZeta(PetscScalar *cellCoords, PetscReal point[NUM_DIMENSIONS], PetscReal XiEtaZeta[NUM_DIMENSIONS]);

PetscErrorCode computeElementalMatrix(PetscInt nord, PetscInt cellOrientation[10], PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscInt numGaussPoints, PetscReal **gaussPoints, PetscReal *weigths, PetscReal *cellResistivity, PetscReal **Me, PetscReal **Ke);

PetscErrorCode computeBasisFunctions(PetscInt nord, PetscInt cellOrientation[10], PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], PetscReal *point, PetscReal **basisFunctions, PetscReal **curlBasisFunctions);

PetscErrorCode computeCellOrientation(DM dm, PetscInt cell, PetscInt cellOrientation[10]);

PetscErrorCode computeCellOrientation(DM dm, PetscInt cell, PetscInt cellOrientation[10]);

PetscErrorCode computeElementalGradientMatrix(PetscInt cellOrientation[10], PetscReal **gradientMatrix);

PetscErrorCode computeElementalGradientMatrix2(PetscInt nord, PetscInt cellOrientation[10], PetscInt numGaussPoints, PetscReal **gaussPoints, PetscReal *weigths);

#endif
