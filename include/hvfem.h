/*
  Filename: hvfem.h
  Author: Octavio Castillo Reyes (UPC/BSC)
  Date: 2025-09-05

  Description:
  This file contains a collection of definitions for
  high-order vector finite element functions that are used
  throughout the PETGEM project.

  Usage:
  Include this file in your source code to utilize the hvfem
  functions. For example: #include "hvfem.h"
*/

#ifndef HVFEM_H
#define HVFEM_H

#include "constants.h"
#include "grid.h"
#include <petsc.h>
#include <petscdmplex.h>

typedef struct {
  PetscInt numPoints;
  PetscReal* points;
  PetscReal* weights;
} Quadrature1D;

typedef struct {
  PetscInt numPoints;
  PetscReal** points;
  PetscReal* weights;
} Quadrature2D;

typedef struct {
  PetscInt numPoints;
  PetscReal** points;
  PetscReal* weights;
} Quadrature3D;

PetscErrorCode printCellEntities(DM dm, PetscInt cell);

PetscErrorCode computeCellJacobian(Cell* cell);

PetscErrorCode computeCellOrientation(Cell* cell);

PetscErrorCode computeNum1DQuadraturePoints(const PetscInt nord, Quadrature1D* quadrature);

PetscErrorCode computeNum2DQuadraturePoints(const PetscInt nord, Quadrature2D* quadrature);

PetscErrorCode computeNum3DQuadraturePoints(const PetscInt nord, Quadrature3D* quadrature);

PetscErrorCode compute1DQuadraturePoints(Quadrature1D* quadrature);

PetscErrorCode compute3DQuadraturePoints(Quadrature3D* quadrature);

PetscErrorCode tetrahedronXYZToReference(const PetscReal coordinates[NUM_VERTICES_PER_CELL * NUM_DIMENSIONS],
                                         const PetscReal point[NUM_DIMENSIONS], PetscReal XiEtaZeta[NUM_DIMENSIONS]);

PetscErrorCode computeVectorRotation(const PetscReal azimuth, const PetscReal dip, PetscReal rotationVector[NUM_DIMENSIONS]);

PetscErrorCode computeElementalMatrices(const PetscInt nord, const PetscInt numDofInCell, const Cell* cell, const Quadrature3D* quadrature,
                                        PetscReal** Me, PetscReal** Ke);

PetscErrorCode computeNedelecOrder1BasisFunctions(const PetscInt nord, const PetscReal point[NUM_DIMENSIONS],
                                                  const PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], const PetscReal* const* coeffs,
                                                  PetscReal** Ni);

PetscErrorCode computeNedelecOrder1BasisFunctionCurls(const PetscInt nord, const PetscReal* const* Dx_Ni, const PetscReal* const* Dy_Ni,
                                                      const PetscReal* const* Dz_Ni,
                                                      const PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS], const PetscReal detJacobian,
                                                      PetscReal** NiCurl);

PetscErrorCode computeNedelecOrder1Coefficients(const PetscInt nord, PetscReal** coeffs, PetscReal** Dx_Ni, PetscReal** Dy_Ni,
                                                PetscReal** Dz_Ni);

PetscErrorCode computeElementalGradientMatrix(const PetscInt nord, const PetscInt numDofInCell, const PetscInt numH1DofInCell,
                                              const Cell* cell, const Quadrature1D* quadrature, PetscReal** gradientMatrix);

PetscErrorCode printCellEntities(const DM dm, const PetscInt cell);

PetscErrorCode checkDiscreteGradientKernel(const PetscReal* M, const PetscReal* G, const PetscInt m, const PetscInt n, const PetscInt cell);

#endif
