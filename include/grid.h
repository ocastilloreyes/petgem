/*
  Filename: grid.h
  Author: Octavio Castillo Reyes (UPC/BSC)
  Date: 2025-09-05

  Description:
  This file contains a collection of definitions for grid
  functions that are used throughout the PETGEM project.
  These functions are based on DMPLex provided by PETSc.

  Usage:
  Include this file in your source code to utilize the grid
  functions. For example: #include "grid.h"
*/

#ifndef GRID_H
#define GRID_H

#include "constants.h"
#include "inputs.h"
#include <petsc.h>

typedef struct {
  PetscInt numCellsLocal;     /* Number of local cells        */
  PetscInt numCellsGlobal;    /* Number of global cells */
  PetscInt numFacesLocal;     /* Number of local faces        */
  PetscInt numFacesGlobal;    /* Number of global cells */
  PetscInt numEdgesLocal;     /* Number of local edges        */
  PetscInt numEdgesGlobal;    /* Number of global edges   */
  PetscInt numVerticesLocal;  /* Number of local vertices */
  PetscInt numVerticesGlobal; /* Number of global vertices    */
  PetscInt numDofInVertex;    /* Number of dofs per vertex */
  PetscInt numDofInEdge;      /* Number of dofs per edge      */
  PetscInt numDofInFace;      /* Number of dofs per vertex    */
  PetscInt numDofInVolume;    /* Number of dofs per volume */
  PetscInt numDofInCell;      /* Number of dofs per cell      */
  PetscInt cellStart;         /* Index of global cell start   */
  PetscInt cellEnd;           /* Index of global cell end     */
  PetscInt faceStart;         /* Index of global face start   */
  PetscInt faceEnd;           /* Index of global face end     */
  PetscInt edgeStart;         /* Index of global edge start   */
  PetscInt edgeEnd;           /* Index of global edge end     */
  PetscInt vertexStart;       /* Index of global vertex start */
  PetscInt vertexEnd;         /* Index of global vertex end   */
  PetscInt dim;               /* Number of dimensions         */

  PetscInt numH1DofInCell; /* Number of H1 dofs per cell */
  DM H1dm;
} Grid;

typedef struct {
  PetscReal coordinates[NUM_VERTICES_PER_CELL * NUM_DIMENSIONS]; /* 12 */
  PetscReal resistivity[NUM_RESISTIVITY_COMPONENTS];
  PetscInt closure[MAX_TRANSITIVE_CLOSURE_SIZE];
  PetscInt closureSize;
  PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS];
  PetscReal invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS];
  PetscReal detJacobian;
  PetscInt orientation[10];
} Cell;

PetscErrorCode importGrid(const Params params, DM* odm, Vec* resistivity_output);

PetscErrorCode setupCsemGrid(const Params params, DM* dm, Grid* grid);

PetscErrorCode locatePoint(const DM dm, const PetscReal* position, PetscInt* pointInCell);

PetscErrorCode extractCellCoordinates(DM dm, PetscInt cellID, Cell* cell);

PetscErrorCode extractCellResistivity(DM dmResistivity, Vec resistivity, PetscInt cellID, Cell* cell);

PetscErrorCode extractCellClousure(DM dm, PetscInt cellID, Cell* cell);

#endif
