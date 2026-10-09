/*
 * Filename: box_mesh.h
 * Author: PETGEM test suite
 * Date: 2026-10-09
 *
 * Description:
 * Distributed tetrahedral box mesh for the C harnesses: faces[0] x faces[1] x
 * faces[2] hexahedra, each split into 6 Kuhn tetrahedra, built on rank 0 with
 * DMPlexCreateFromCellListPetsc and distributed.
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

#endif /* BOX_MESH_H */
