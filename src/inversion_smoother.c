/*
 * Filename: inversion_smoother.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Cell-graph smoother, fixed-cell mask and model transform for the CSEM
 * inverse kernel.
 */

#include <petsc.h>
#include <petscdmplex.h>

#include "constants.h"
#include "grid.h"
#include "fem.h"
#include "inversion.h"
#include "inversion_internal.h"

PetscErrorCode createCellScalarDm(DM dm, DM *cellDm) {
  PetscFunctionBeginUser;
  MPI_Comm     comm = PetscObjectComm((PetscObject)dm);
  PetscFV      fvm;
  PetscSection sec;
  PetscInt     pStart, pEnd, cStart, cEnd;

  PetscCall(DMClone(dm, cellDm));
  PetscCall(PetscFVCreate(comm, &fvm));
  PetscCall(PetscFVSetNumComponents(fvm, 1));
  PetscCall(PetscObjectSetName((PetscObject)fvm, "scalar"));
  PetscCall(DMAddField(*cellDm, NULL, (PetscObject)fvm));
  PetscCall(PetscFVDestroy(&fvm));

  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(PetscSectionCreate(comm, &sec));
  PetscCall(PetscSectionSetNumFields(sec, 1));
  PetscCall(PetscSectionSetFieldComponents(sec, 0, 1));
  PetscCall(PetscSectionSetChart(sec, pStart, pEnd));
  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscCall(PetscSectionSetDof(sec, c, 1));
    PetscCall(PetscSectionSetFieldDof(sec, c, 0, 1));
  }
  PetscCall(PetscSectionSetUp(sec));
  PetscCall(DMSetLocalSection(*cellDm, sec));
  PetscCall(PetscSectionDestroy(&sec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Appends v to the growable array (*a, *n, *cap). */
static PetscErrorCode appendInt(PetscInt **a, PetscInt *n, PetscInt *cap, PetscInt v) {
  PetscFunctionBeginUser;
  if (*n == *cap) {
    *cap = PetscMax(2 * *cap, 64);
    PetscCall(PetscRealloc((size_t)*cap * sizeof(PetscInt), a));
  }
  (*a)[(*n)++] = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Cells in [cStart, cEnd), other than cell, sharing a vertex with it; first-seen order, no duplicates. */
static PetscErrorCode vertexNeighbors(DM dm, PetscInt cell, PetscInt cStart, PetscInt cEnd, PetscInt **nb, PetscInt *n, PetscInt *cap) {
  PetscFunctionBeginUser;
  PetscInt closureSize = 0, *closure = NULL;

  *n = 0;
  PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
  for (PetscInt ci = 0; ci < closureSize * 2; ci += 2) {
    PetscInt depth, starSize = 0, *star = NULL;
    PetscCall(DMPlexGetPointDepth(dm, closure[ci], &depth));
    if (depth != 0) continue;
    PetscCall(DMPlexGetTransitiveClosure(dm, closure[ci], PETSC_FALSE, &starSize, &star));
    for (PetscInt s = 0; s < starSize * 2; s += 2) {
      const PetscInt c    = star[s];
      PetscBool      seen = PETSC_FALSE;
      if (c == cell || c < cStart || c >= cEnd) continue;
      for (PetscInt k = 0; k < *n && !seen; k++) seen = (PetscBool)((*nb)[k] == c - cStart);
      if (!seen) PetscCall(appendInt(nb, n, cap, c - cStart));
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, closure[ci], PETSC_FALSE, &starSize, &star));
  }
  PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* CSR graph over the first numLocalCells cells of [cStart, cEnd) in dm: vertex neighbors with normalized 1/dist weights. */
static PetscErrorCode buildCsr(DM dm, PetscInt cStart, PetscInt cEnd, NeighborGraph *graph) {
  PetscFunctionBeginUser;
  const PetscInt numCells = cEnd - cStart;
  PetscReal     *centroid;
  PetscInt      *nb = NULL, n, nbCap = 0, *list = NULL, listLen = 0, listCap = 0;

  PetscCall(PetscMalloc1(numCells * NUM_DIMENSIONS, &centroid));
  for (PetscInt c = cStart; c < cEnd; c++) {
    Cell cell;
    PetscCall(extractCellCoordinates(dm, c, &cell));
    PetscCall(computeCellCentroid(&cell));
    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) centroid[(c - cStart) * NUM_DIMENSIONS + d] = cell.centroid[d];
  }

  PetscCall(PetscMalloc1(graph->numLocalCells + 1, &graph->neighborStart));
  graph->neighborStart[0] = 0;
  for (PetscInt i = 0; i < graph->numLocalCells; i++) {
    n = 0;
    if (!graph->isFixed[i]) PetscCall(vertexNeighbors(dm, cStart + i, cStart, cEnd, &nb, &n, &nbCap));
    if (n == 0) PetscCall(appendInt(&list, &listLen, &listCap, i));
    for (PetscInt j = 0; j < n; j++) PetscCall(appendInt(&list, &listLen, &listCap, nb[j]));
    graph->neighborStart[i + 1] = listLen;
  }
  graph->neighborList = list;

  PetscCall(PetscMalloc1(listLen, &graph->neighborWeights));
  for (PetscInt i = 0; i < graph->numLocalCells; i++) {
    const PetscInt   start = graph->neighborStart[i], end = graph->neighborStart[i + 1];
    const PetscReal *ci   = &centroid[i * NUM_DIMENSIONS];
    PetscReal        wsum = 0.0;

    if (end - start == 1 && list[start] == i) {
      graph->neighborWeights[start] = 1.0;
      continue;
    }
    for (PetscInt j = start; j < end; j++) {
      const PetscReal *cj = &centroid[list[j] * NUM_DIMENSIONS];
      const PetscReal  d =
          PetscSqrtReal((cj[0] - ci[0]) * (cj[0] - ci[0]) + (cj[1] - ci[1]) * (cj[1] - ci[1]) + (cj[2] - ci[2]) * (cj[2] - ci[2]));
      graph->neighborWeights[j] = (d > 0.0) ? 1.0 / d : 0.0;
      wsum += graph->neighborWeights[j];
    }
    if (wsum > 0.0) {
      for (PetscInt j = start; j < end; j++) graph->neighborWeights[j] /= wsum;
    }
  }

  PetscCall(PetscFree(nb));
  PetscCall(PetscFree(centroid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* overlap=1 copy of dm under vertex adjacency, or NULL when dm is not distributed. */
static PetscErrorCode createVertexOverlapDm(DM dm, DM *dmOver) {
  PetscFunctionBeginUser;
  PetscBool useCone, useClosure;
  PetscSF   sf;

  /* Smoothing neighbors share a vertex: distribute the overlap with closure adjacency, then restore dm */
  PetscCall(DMGetBasicAdjacency(dm, &useCone, &useClosure));
  PetscCall(DMSetBasicAdjacency(dm, PETSC_FALSE, PETSC_TRUE));
  PetscCall(DMPlexDistributeOverlap(dm, 1, &sf, dmOver));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(DMSetBasicAdjacency(dm, useCone, useClosure));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Builds the CSR neighbor smoothing graph from DMPlex topology.
 *
 * Fixed cells (materialId in imParams->fixedMaterials) keep only a
 * self-reference. Other cells link to every cell sharing a vertex with
 * weights 1/dist(centroids), normalized. On more than one rank the graph is
 * built on an overlap=1 DM, so cross-rank neighbors are included and the
 * smoothed result does not depend on the partition.
 */
PetscErrorCode buildNeighborSmoothingGraph(const DM dm, const Grid *grid, const ImParams *imParams, Vec materialsId, NeighborGraph *graph) {
  PetscFunctionBeginUser;
  PetscMPIInt size;
  DM          dmMaterialsId, dmOver = NULL;

  PetscCall(PetscMemzero(graph, sizeof(*graph)));
  graph->numLocalCells = grid->cellEnd - grid->cellStart;
  PetscCall(PetscCalloc1(graph->numLocalCells, &graph->isFixed));
  PetscCall(VecGetDM(materialsId, &dmMaterialsId));
  for (PetscInt i = 0; i < graph->numLocalCells; i++) {
    Cell cell;
    PetscCall(extractCellMaterialId(dmMaterialsId, materialsId, grid->cellStart + i, &cell));
    for (PetscInt j = 0; j < imParams->numFixedMaterials; j++) {
      if (cell.materialId == imParams->fixedMaterials[j]) graph->isFixed[i] = PETSC_TRUE;
    }
  }

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  if (size > 1) PetscCall(createVertexOverlapDm(dm, &dmOver));
  if (dmOver) {
    PetscInt cStart, cEnd;
    PetscCall(createCellScalarDm(dmOver, &graph->dmOverlap));
    PetscCall(DMDestroy(&dmOver));
    PetscCall(DMCreateLocalVector(graph->dmOverlap, &graph->overlapLocal));
    PetscCall(DMCreateGlobalVector(graph->dmOverlap, &graph->overlapGlobal));
    /* Owned cells come first in the overlap DM */
    PetscCall(DMPlexGetHeightStratum(graph->dmOverlap, 0, &cStart, &cEnd));
    PetscCall(buildCsr(graph->dmOverlap, cStart, cEnd, graph));
  } else {
    PetscCall(buildCsr(dm, grid->cellStart, grid->cellEnd, graph));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* out[i] = diagWeight in[i] + sum_j w_ij in[j] for the non-fixed owned cells. */
static void jacobiSweep(const NeighborGraph *graph, PetscReal diagWeight, const PetscScalar *in, PetscScalar *out) {
  for (PetscInt i = 0; i < graph->numLocalCells; i++) {
    PetscScalar v;
    if (graph->isFixed[i]) continue;
    v = diagWeight * in[i];
    for (PetscInt j = graph->neighborStart[i]; j < graph->neighborStart[i + 1]; j++) {
      v += graph->neighborWeights[j] * in[graph->neighborList[j]];
    }
    out[i] = v;
  }
}

/* Copies the owned cells between the 1-DOF/cell vector v and the overlap work vector. */
static PetscErrorCode copyOwned(const NeighborGraph *graph, Vec from, Vec to) {
  PetscFunctionBeginUser;
  const PetscScalar *src;
  PetscScalar       *dst;
  PetscCall(VecGetArrayRead(from, &src));
  PetscCall(VecGetArray(to, &dst));
  PetscCall(PetscArraycpy(dst, src, graph->numLocalCells));
  PetscCall(VecRestoreArray(to, &dst));
  PetscCall(VecRestoreArrayRead(from, &src));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode applyGraphSmoothing(const NeighborGraph *graph, PetscReal diagWeight, Vec v) {
  PetscFunctionBeginUser;
  Vec          work = graph->dmOverlap ? graph->overlapLocal : v;
  PetscScalar *arr, *snap;
  PetscInt     n;

  PetscCall(VecGetLocalSize(work, &n));
  PetscCall(PetscMalloc1(n, &snap));
  if (graph->dmOverlap) PetscCall(copyOwned(graph, v, work));
  for (PetscInt sweep = 0; sweep < 2; sweep++) {
    if (graph->dmOverlap) {
      PetscCall(DMLocalToGlobal(graph->dmOverlap, work, INSERT_VALUES, graph->overlapGlobal));
      PetscCall(DMGlobalToLocal(graph->dmOverlap, graph->overlapGlobal, INSERT_VALUES, work));
    }
    PetscCall(VecGetArray(work, &arr));
    PetscCall(PetscArraycpy(snap, arr, n));
    jacobiSweep(graph, diagWeight, snap, arr);
    PetscCall(VecRestoreArray(work, &arr));
  }
  if (graph->dmOverlap) PetscCall(copyOwned(graph, work, v));
  PetscCall(PetscFree(snap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Builds a 0/1 mask vector that freezes fixed cells.
 *
 * mask = 0 on fixed cells and 1 elsewhere, on dmInversion (1 DOF/cell).
 */
PetscErrorCode buildNotFixedMask(const NeighborGraph *graph, DM dmInversion, const Grid *grid, Vec *maskGlobal, Vec *maskLocal) {
  PetscFunctionBeginUser;
  PetscSection sec;
  PetscScalar *arr;

  PetscCall(DMCreateLocalVector(dmInversion, maskLocal));
  PetscCall(VecSet(*maskLocal, 1.0));
  PetscCall(DMGetLocalSection(dmInversion, &sec));
  PetscCall(VecGetArray(*maskLocal, &arr));
  for (PetscInt i = 0; i < graph->numLocalCells; i++) {
    PetscInt off;
    if (!graph->isFixed[i]) continue;
    PetscCall(PetscSectionGetOffset(sec, grid->cellStart + i, &off));
    arr[off] = 0.0;
  }
  PetscCall(VecRestoreArray(*maskLocal, &arr));

  PetscCall(DMCreateGlobalVector(dmInversion, maskGlobal));
  PetscCall(VecSet(*maskGlobal, 0.0));
  PetscCall(DMLocalToGlobal(dmInversion, *maskLocal, INSERT_VALUES, *maskGlobal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Recovers conductivity sigma = 1 / exp(smooth(X) + X0) (isotropic).
 *
 * X and X0 are global Vecs on dmInversion; sigmaModel is the local
 * 3-DOF/cell conductivity Vec. X itself is not modified.
 */
PetscErrorCode applyLogToSigma(DM dmInversion, DM dmConductivity, const Vec X, const Vec X0, Vec sigmaModel, const Grid *grid,
                               const NeighborGraph *graph) {
  PetscFunctionBeginUser;
  PetscSection       sec;
  Vec                xLocal, x0Local;
  const PetscScalar *x, *x0;
  PetscScalar       *s;

  PetscCall(DMGetLocalVector(dmInversion, &xLocal));
  PetscCall(DMGetLocalVector(dmInversion, &x0Local));
  PetscCall(DMGlobalToLocal(dmInversion, X, INSERT_VALUES, xLocal));
  PetscCall(DMGlobalToLocal(dmInversion, X0, INSERT_VALUES, x0Local));
  PetscCall(applyGraphSmoothing(graph, 0.0, xLocal));

  PetscCall(DMGetLocalSection(dmConductivity, &sec));
  PetscCall(VecGetArrayRead(xLocal, &x));
  PetscCall(VecGetArrayRead(x0Local, &x0));
  PetscCall(VecGetArray(sigmaModel, &s));
  for (PetscInt i = grid->cellStart; i < grid->cellEnd; i++) {
    const PetscInt    li    = i - grid->cellStart;
    const PetscScalar sigma = 1.0 / PetscExpScalar(PetscRealPart(x[li] + x0[li]));
    PetscInt          off;
    PetscCall(PetscSectionGetOffset(sec, i, &off));
    for (PetscInt comp = 0; comp < NUM_CONDUCTIVITY_COMPONENTS; comp++) s[off + comp] = sigma;
  }
  PetscCall(VecRestoreArray(sigmaModel, &s));
  PetscCall(VecRestoreArrayRead(x0Local, &x0));
  PetscCall(VecRestoreArrayRead(xLocal, &x));
  PetscCall(DMRestoreLocalVector(dmInversion, &xLocal));
  PetscCall(DMRestoreLocalVector(dmInversion, &x0Local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroyNeighborGraph(NeighborGraph *graph) {
  PetscFunctionBeginUser;
  PetscCall(PetscFree(graph->isFixed));
  PetscCall(PetscFree(graph->neighborStart));
  PetscCall(PetscFree(graph->neighborList));
  PetscCall(PetscFree(graph->neighborWeights));
  PetscCall(VecDestroy(&graph->overlapLocal));
  PetscCall(VecDestroy(&graph->overlapGlobal));
  PetscCall(DMDestroy(&graph->dmOverlap));
  PetscFunctionReturn(PETSC_SUCCESS);
}
