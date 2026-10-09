/*
 * Filename: box_mesh.h
 * Author: PETGEM test suite
 * Date: 2026-10-09
 *
 * Description:
 * Distributed tetrahedral box mesh for the C harnesses: faces[0] x faces[1] x
 * faces[2] hexahedra, each split into 6 positively oriented Kuhn tetrahedra, built on rank 0 with
 * DMPlexCreateFromCellListPetsc and distributed, and a per-cell
 * conductivity Vec on it.
 */
#ifndef BOX_MESH_H
#define BOX_MESH_H

#include <petsc.h>

/* topShift moves vertex (1, 1, faces[2]) of the top face by topShift in z. */
static void create_box_mesh(const PetscInt faces[3], const PetscReal lower[3], const PetscReal upper[3],
                            PetscReal topShift, DM *dm)
{
  DM          dmDist = NULL;
  PetscMPIInt rank;
  PetscInt    numCells = 0, numVertices = 0, *cells = NULL;
  PetscReal  *coords = NULL;

  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0) {
    const PetscInt nx = faces[0], ny = faces[1], nz = faces[2];
    const PetscInt perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    const PetscBool odd[6]    = {PETSC_FALSE, PETSC_TRUE, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, PETSC_TRUE};
    numVertices = (nx + 1) * (ny + 1) * (nz + 1);
    numCells    = 6 * nx * ny * nz;
    PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(3 * numVertices, &coords));
    PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(4 * numCells, &cells));
    for (PetscInt k = 0; k <= nz; k++)
      for (PetscInt j = 0; j <= ny; j++)
        for (PetscInt i = 0; i <= nx; i++) {
          const PetscInt v = (k * (ny + 1) + j) * (nx + 1) + i;
          coords[3 * v + 0] = lower[0] + (upper[0] - lower[0]) * i / nx;
          coords[3 * v + 1] = lower[1] + (upper[1] - lower[1]) * j / ny;
          coords[3 * v + 2] = lower[2] + (upper[2] - lower[2]) * k / nz;
          if (i == 1 && j == 1 && k == nz) coords[3 * v + 2] += topShift;
        }
    PetscInt c = 0;
    for (PetscInt k = 0; k < nz; k++)
      for (PetscInt j = 0; j < ny; j++)
        for (PetscInt i = 0; i < nx; i++)
          for (PetscInt t = 0; t < 6; t++) {
            PetscInt ijk[3] = {i, j, k};
            for (PetscInt s = 0; s < 4; s++) {
              if (s > 0) ijk[perm[t][s - 1]]++;
              cells[4 * c + s] = (ijk[2] * (ny + 1) + ijk[1]) * (nx + 1) + ijk[0];
            }
            if (odd[t]) {
              const PetscInt tmp = cells[4 * c + 2];
              cells[4 * c + 2]   = cells[4 * c + 3];
              cells[4 * c + 3]   = tmp;
            }
            c++;
          }
  }
  PetscCallAbort(PETSC_COMM_WORLD, DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, 3, numCells, numVertices, 4, PETSC_TRUE, cells, 3, coords, dm));
  PetscCallAbort(PETSC_COMM_WORLD, PetscFree(cells));
  PetscCallAbort(PETSC_COMM_WORLD, PetscFree(coords));
  PetscCallAbort(PETSC_COMM_WORLD, DMPlexDistribute(*dm, 0, NULL, &dmDist));
  if (dmDist) {
    PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(dm));
    *dm = dmDist;
  }
}

/* Per-cell conductivity Vec laid out as loadModelInputs builds it; sigmaAt(centroid) gives the three components. */
static void create_cell_conductivity(DM dm, PetscReal (*sigmaAt)(const PetscReal centroid[3]), Vec *sigma)
{
  DM           dmS;
  PetscSection sec;
  PetscInt     pStart, pEnd, cStart, cEnd;
  PetscScalar *arr;

  PetscCallAbort(PETSC_COMM_WORLD, DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCallAbort(PETSC_COMM_WORLD, DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCallAbort(PETSC_COMM_WORLD, DMClone(dm, &dmS));
  PetscCallAbort(PETSC_COMM_WORLD, PetscSectionCreate(PETSC_COMM_WORLD, &sec));
  PetscCallAbort(PETSC_COMM_WORLD, PetscSectionSetChart(sec, pStart, pEnd));
  for (PetscInt c = cStart; c < cEnd; c++) PetscCallAbort(PETSC_COMM_WORLD, PetscSectionSetDof(sec, c, 3));
  PetscCallAbort(PETSC_COMM_WORLD, PetscSectionSetUp(sec));
  PetscCallAbort(PETSC_COMM_WORLD, DMSetLocalSection(dmS, sec));
  PetscCallAbort(PETSC_COMM_WORLD, PetscSectionDestroy(&sec));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(dmS, sigma));
  PetscCallAbort(PETSC_COMM_WORLD, DMGetLocalSection(dmS, &sec));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetArray(*sigma, &arr));
  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscBool          isDG;
    PetscInt           n, off;
    const PetscScalar *array;
    PetscScalar       *coords = NULL;
    PetscReal          centroid[3] = {0.0, 0.0, 0.0};
    PetscCallAbort(PETSC_COMM_WORLD, DMPlexGetCellCoordinates(dm, c, &isDG, &n, &array, &coords));
    for (PetscInt v = 0; v < n / 3; v++)
      for (PetscInt d = 0; d < 3; d++) centroid[d] += PetscRealPart(coords[3 * v + d]) / (n / 3);
    PetscCallAbort(PETSC_COMM_WORLD, DMPlexRestoreCellCoordinates(dm, c, &isDG, &n, &array, &coords));
    PetscCallAbort(PETSC_COMM_WORLD, PetscSectionGetOffset(sec, c, &off));
    for (PetscInt d = 0; d < 3; d++) arr[off + d] = sigmaAt(centroid);
  }
  PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArray(*sigma, &arr));
  PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(&dmS));
}

#endif /* BOX_MESH_H */
