/*
 * Filename: inversion_smoother.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Cell-graph smoother for the CSEM inverse kernel.
 */

/*
 * Public surface:
 *   buildNeighborSmoothingGraph — DMPlex vertex-star adjacency on the
 *                                 owned-cell DM, 1/dist weights
 *   setupParallelSmoothingGraph — overlap=1 graph + ghost-aware Vec
 *                                 workspace; production path for MPI > 1
 *   applyGaussSeidelSmoothing   — forward+reverse Gauss-Seidel.
 *                                 At MPI > 1 uses the parallel block-Jacobi
 *                                 path with one ghost exchange between
 *                                 sweeps; at MPI == 1 uses a direct
 *                                 sequential sweep on the owned cells.
 *   destroyNeighborGraph        — frees both graph variants
 *   buildNotFixedMask           — 0/1 Vec used to zero gradient at fixed cells
 *   applyLogToSigma             — sigma = 1/exp(smoothed_X + X0), MATLAB tempX path
 */

#include <stdio.h>
#include <stdlib.h>

#include <petsc.h>
#include <petscdmplex.h>

#include "constants.h"
#include "grid.h"
#include "hvfem.h"
#include "inversion.h"
#include "inversion_internal.h"

/* ================================================================== */
/* buildNeighborSmoothingGraph                                         */
/*                                                                     */
/* For each local cell i:                                              */
/*   - find all cells sharing at least one vertex via DMPlex star      */
/*   - exclude fixed cells (material_id in iparams->fixedMaterials)   */
/*   - weight w_ij = 1/dist(centroid_i, centroid_j), normalized        */
/* Fixed cells: those with no valid neighbors keep self-reference.    */
/* ================================================================== */
PetscErrorCode buildNeighborSmoothingGraph(const DM          dm,
                                           const Grid       *grid,
                                           const invParams  *iparams,
                                           Vec               materialsID,
                                           NeighborGraph    *graph)
{
  PetscFunctionBeginUser;

  /* Variables declaration */
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  PetscInt numCells = grid->numCellsLocal;

  graph->numLocalCells = numCells;
  PetscCall(PetscCalloc1(numCells + 1, &graph->neighborStart));
  PetscCall(PetscCalloc1(numCells,     &graph->isFixed));

  /* Initialize parallel-smoother fields to NULL/false so destroyNeighborGraph
   * stays safe even if setupParallelSmoothingGraph is never called (the
   * single-rank fallback leaves them this way). */
  graph->hasParallelGraph  = PETSC_FALSE;
  graph->dmInversionOver   = NULL;
  graph->oNeighborStart    = NULL;
  graph->oNeighborList     = NULL;
  graph->oNeighborWeights  = NULL;
  graph->oLocalScratch     = NULL;
  graph->oGlobalScratch    = NULL;

  /* ---- Pass 1: count neighbors per cell to size CSR arrays ---- */
  /* We use a temporary dynamic approach: collect neighbors per cell */

  /* Allocate temporary per-cell neighbor lists (worst-case 200 neighbors) */
  PetscInt maxNeighbors = 200;
  PetscInt *tmpNeighbors;
  PetscCall(PetscMalloc1((PetscInt)(numCells * maxNeighbors), &tmpNeighbors));
  PetscInt *tmpCount;
  PetscCall(PetscCalloc1(numCells, &tmpCount));

  /* Get the DM associated with the materialsID Vec for extractCellMaterialID */
  DM dmMaterialsID;
  PetscCall(VecGetDM(materialsID, &dmMaterialsID));

  /* Precompute centroids for all local cells */
  PetscReal *centroids;
  PetscCall(PetscMalloc1(numCells * NUM_DIMENSIONS, &centroids));

  for (PetscInt i = grid->cellStart; i < grid->cellEnd; i++) {
    PetscInt li = i - grid->cellStart;
    Cell     cell;
    PetscCall(extractCellCoordinates(dm, i, &cell));
    PetscCall(computeCellCentroid(&cell));
    centroids[li * NUM_DIMENSIONS + 0] = cell.centroid[0];
    centroids[li * NUM_DIMENSIONS + 1] = cell.centroid[1];
    centroids[li * NUM_DIMENSIONS + 2] = cell.centroid[2];

    /* Mark fixed cells: material_id matches any entry in fixedMaterials */
    Cell matCell;
    PetscCall(extractCellMaterialID(dmMaterialsID, materialsID, i, &matCell));
    graph->isFixed[li] = PETSC_FALSE;
    for (PetscInt k = 0; k < iparams->numFixedMaterials; k++) {
      if (matCell.material_id == iparams->fixedMaterials[k]) {
        graph->isFixed[li] = PETSC_TRUE;
        break;
      }
    }
  }

  /* ---- Diagnostic: per-material cell-count histogram (global) ----
   * Prints the distribution of material_id values across all ranks.
   * A well-behaved load produces an MPI-count-invariant histogram; any
   * variation pinpoints a DMPlex HDF5 load / SF-composition bug. */
  {
    const PetscInt HIST_CAP = 16;  /* max material ids tracked */
    PetscInt localHist[16];
    PetscInt globalHist[16];
    for (PetscInt k = 0; k < HIST_CAP; k++) localHist[k] = 0;

    for (PetscInt i = grid->cellStart; i < grid->cellEnd; i++) {
      Cell matCell;
      PetscCall(extractCellMaterialID(dmMaterialsID, materialsID, i, &matCell));
      if (matCell.material_id >= 0 && matCell.material_id < HIST_CAP)
        localHist[matCell.material_id]++;
    }
    PetscCallMPI(MPI_Allreduce(localHist, globalHist, HIST_CAP,
                                MPIU_INT, MPI_SUM, comm));

    PetscInt nFixedLocal = 0, nFreeLocal = 0;
    for (PetscInt li = 0; li < numCells; li++) {
      if (graph->isFixed[li]) nFixedLocal++;
      else                    nFreeLocal++;
    }
    PetscInt nFixedGlobal, nFreeGlobal;
    PetscCallMPI(MPI_Allreduce(&nFixedLocal, &nFixedGlobal, 1, MPIU_INT,
                                MPI_SUM, comm));
    PetscCallMPI(MPI_Allreduce(&nFreeLocal,  &nFreeGlobal,  1, MPIU_INT,
                                MPI_SUM, comm));

    if (iparams->verbose) {
      PetscCall(PetscPrintf(comm, "\n Material-ID histogram (global):\n"));
      for (PetscInt k = 0; k < HIST_CAP; k++) {
        if (globalHist[k] > 0)
          PetscCall(PetscPrintf(comm,
            "   id %2" PetscInt_FMT " : %" PetscInt_FMT " cells\n",
            k, globalHist[k]));
      }
      PetscCall(PetscPrintf(comm,
        "   Fixed: %" PetscInt_FMT "  Non-fixed: %" PetscInt_FMT
        "  Total: %" PetscInt_FMT "\n",
        nFixedGlobal, nFreeGlobal, nFixedGlobal + nFreeGlobal));
    } else {
      PetscCall(PetscPrintf(comm,
        "\n Cells: total=%" PetscInt_FMT "  fixed=%" PetscInt_FMT
        "  non-fixed=%" PetscInt_FMT "  (-inv_verbose for per-id histogram)\n",
        nFixedGlobal + nFreeGlobal, nFixedGlobal, nFreeGlobal));
    }

    /* Centroid-of-mass per material_id.  Cell numbering changes with MPI
     * rank count, but the physical centroid of each cell does not.  So if
     * the load correctly maps material_id to the right PHYSICAL cells,
     * these means are MPI-invariant.  A shift across rank counts is proof
     * of a data-to-cell scramble. */
    PetscReal localSum[16 * 3];
    PetscReal globalSum[16 * 3];
    for (PetscInt k = 0; k < HIST_CAP * 3; k++) localSum[k] = 0.0;

    for (PetscInt i = grid->cellStart; i < grid->cellEnd; i++) {
      PetscInt li = i - grid->cellStart;
      Cell     matCell;
      PetscCall(extractCellMaterialID(dmMaterialsID, materialsID, i, &matCell));
      if (matCell.material_id < 0 || matCell.material_id >= HIST_CAP) continue;
      PetscInt base = matCell.material_id * 3;
      localSum[base + 0] += centroids[li * NUM_DIMENSIONS + 0];
      localSum[base + 1] += centroids[li * NUM_DIMENSIONS + 1];
      localSum[base + 2] += centroids[li * NUM_DIMENSIONS + 2];
    }
    PetscCallMPI(MPI_Allreduce(localSum, globalSum, HIST_CAP * 3,
                                MPIU_REAL, MPI_SUM, comm));

    if (iparams->verbose) {
      PetscCall(PetscPrintf(comm, "\n Mean centroid per material (global):\n"));
      PetscCall(PetscPrintf(comm,
        "   id    N      mean_x          mean_y          mean_z\n"));
      for (PetscInt k = 0; k < HIST_CAP; k++) {
        if (globalHist[k] == 0) continue;
        PetscReal nInv = 1.0 / (PetscReal)globalHist[k];
        PetscCall(PetscPrintf(comm,
          "   %2" PetscInt_FMT "  %6" PetscInt_FMT
          "  %+.6e  %+.6e  %+.6e\n",
          k, globalHist[k],
          (double)(globalSum[k * 3 + 0] * nInv),
          (double)(globalSum[k * 3 + 1] * nInv),
          (double)(globalSum[k * 3 + 2] * nInv)));
      }
    }
  }

  /* Build neighbor lists using transitive closure */
  for (PetscInt i = grid->cellStart; i < grid->cellEnd; i++) {
    PetscInt  li = i - grid->cellStart;

    if (graph->isFixed[li]) {
      /* Fixed: self-reference only */
      tmpNeighbors[li * maxNeighbors + 0] = li;
      tmpCount[li]                         = 1;
      continue;
    }

    /* Get the 4 vertices of this cell */
    PetscInt        closureSize = 0;
    PetscInt       *closure     = NULL;
    PetscCall(DMPlexGetTransitiveClosure(dm, i, PETSC_TRUE,
                                         &closureSize, &closure));

    /* Collect all cells sharing any vertex.
       In DMPlex the direct support of a vertex is edges (depth 1),
       not cells.  We use the upward transitive closure (star) of
       each vertex to find all cells that contain it.              */
    PetscInt cnt = 0;
    for (PetscInt ci = 0; ci < closureSize * 2; ci += 2) {
      PetscInt point = closure[ci];
      /* Only vertices (depth 0) */
      PetscInt pdepth = 0;
      PetscCall(DMPlexGetPointDepth(dm, point, &pdepth));
      if (pdepth != 0) continue;

      /* Get upward star of this vertex to find all cells */
      PetscInt        starSize = 0;
      PetscInt       *star     = NULL;
      PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE,
                                           &starSize, &star));
      for (PetscInt s = 0; s < starSize * 2; s += 2) {
        PetscInt neighbor = star[s];
        if (neighbor == i) continue;
        if (neighbor < grid->cellStart || neighbor >= grid->cellEnd) continue;
        PetscInt ln = neighbor - grid->cellStart;
        /* Keep fixed (air/padding) neighbors in the list — they act as a
         * fixed-value bath the smoother averages non-fixed boundary cells
         * toward.  applyGaussSeidelSmoothing skips updating fixed cells
         * themselves (their stored value is preserved), so dropping them
         * here produced sharp anomaly contours in MATLAB-comparable runs:
         * boundary non-fixed cells lost most of their averaging support.
         * Including them restores MATLAB-style smooth transitions. */

        /* Check for duplicate */
        PetscBool found = PETSC_FALSE;
        for (PetscInt k = 0; k < cnt; k++) {
          if (tmpNeighbors[li * maxNeighbors + k] == ln) {
            found = PETSC_TRUE;
            break;
          }
        }
        if (!found && cnt < maxNeighbors) {
          tmpNeighbors[li * maxNeighbors + cnt] = ln;
          cnt++;
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE,
                                               &starSize, &star));
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, i, PETSC_TRUE,
                                             &closureSize, &closure));

    if (cnt == 0) {
      /* No valid neighbors: self-reference */
      tmpNeighbors[li * maxNeighbors + 0] = li;
      tmpCount[li]                         = 1;
    } else {
      tmpCount[li] = cnt;
    }
  }

  /* ---- Build CSR structure ---- */
  graph->neighborStart[0] = 0;
  for (PetscInt i = 0; i < numCells; i++)
    graph->neighborStart[i + 1] = graph->neighborStart[i] + tmpCount[i];

  PetscInt totalNeighbors = graph->neighborStart[numCells];
  PetscCall(PetscMalloc1(totalNeighbors, &graph->neighborList));
  PetscCall(PetscMalloc1(totalNeighbors, &graph->neighborWeights));

  for (PetscInt i = 0; i < numCells; i++) {
    PetscInt start = graph->neighborStart[i];
    PetscInt cnt   = tmpCount[i];

    if (graph->isFixed[i] || cnt == 1) {
      /* self-reference or no neighbors */
      graph->neighborList[start]    = i;
      graph->neighborWeights[start] = 1.0;
      continue;
    }

    PetscReal cx = centroids[i * NUM_DIMENSIONS + 0];
    PetscReal cy = centroids[i * NUM_DIMENSIONS + 1];
    PetscReal cz = centroids[i * NUM_DIMENSIONS + 2];
    PetscReal wsum = 0.0;

    for (PetscInt k = 0; k < cnt; k++) {
      PetscInt  ln = tmpNeighbors[i * maxNeighbors + k];
      PetscReal dx = centroids[ln * NUM_DIMENSIONS + 0] - cx;
      PetscReal dy = centroids[ln * NUM_DIMENSIONS + 1] - cy;
      PetscReal dz = centroids[ln * NUM_DIMENSIONS + 2] - cz;
      PetscReal w  = 1.0 / PetscSqrtReal(dx*dx + dy*dy + dz*dz);
      graph->neighborList[start + k]    = ln;
      graph->neighborWeights[start + k] = w;
      wsum += w;
    }
    /* Normalize */
    for (PetscInt k = 0; k < cnt; k++)
      graph->neighborWeights[start + k] /= wsum;
  }

  /* Diagnostic: per-cell neighbor-count distribution (non-fixed cells only).
   * Boundary cells with very few neighbors are the ones that produce sharp
   * anomaly contours — they have weak averaging support.  Healthy mesh
   * regions show most non-fixed cells in the 10-19 buckets. */
  {
    const PetscInt NB_BINS = 8;
    PetscInt nbHist[8] = {0,0,0,0,0,0,0,0};
    /* Bin edges: [0], [1-2], [3-5], [6-9], [10-14], [15-19], [20-29], [30+] */
    for (PetscInt i = 0; i < numCells; i++) {
      if (graph->isFixed[i]) continue;
      PetscInt c = graph->neighborStart[i + 1] - graph->neighborStart[i];
      PetscInt b = (c == 0) ? 0
                : (c <  3) ? 1
                : (c <  6) ? 2
                : (c < 10) ? 3
                : (c < 15) ? 4
                : (c < 20) ? 5
                : (c < 30) ? 6 : 7;
      nbHist[b]++;
    }
    PetscInt nbHistG[8];
    PetscCallMPI(MPI_Allreduce(nbHist, nbHistG, NB_BINS, MPIU_INT,
                                MPI_SUM, comm));
    if (iparams->verbose) {
      PetscCall(PetscPrintf(comm,
        "\n Smoother neighbor-count histogram (non-fixed cells, global):\n"
        "       0      1-2     3-5     6-9   10-14   15-19   20-29   30+\n"
        "   %6" PetscInt_FMT " %6" PetscInt_FMT " %6" PetscInt_FMT
        " %6" PetscInt_FMT " %6" PetscInt_FMT " %6" PetscInt_FMT
        " %6" PetscInt_FMT " %6" PetscInt_FMT "\n",
        nbHistG[0], nbHistG[1], nbHistG[2], nbHistG[3],
        nbHistG[4], nbHistG[5], nbHistG[6], nbHistG[7]));
    }
  }

  PetscCall(PetscFree(tmpNeighbors));
  PetscCall(PetscFree(tmpCount));
  PetscCall(PetscFree(centroids));

  PetscCall(PetscPrintf(comm,
    "\n Neighbor smoothing graph:\n"
    "   Local cells         = %" PetscInt_FMT "\n"
    "   Total neighbor arcs = %" PetscInt_FMT "\n",
    numCells, totalNeighbors));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* applyGaussSeidelSmoothing                                           */
/*                                                                     */
/* Forward + reverse Gauss-Seidel sweep using the precomputed CSR      */
/* neighbor graph.  Operates on a partition-local Vec — non-owned     */
/* DOFs not visited.                                                   */
/* diagWeight < 0 is a sentinel that bypasses the smoother (used by   */
/* the dev FD-gradient check).                                        */
/* ================================================================== */
PetscErrorCode applyGaussSeidelSmoothing(const NeighborGraph *graph,
                                         PetscReal            diagWeight,
                                         Vec                  v)
{
  PetscFunctionBeginUser;

  /* Sentinel: a negative diagWeight bypasses the smoother entirely.
   * Used by runFdGradientCheck so the FD test sees a self-consistent
   * objective/gradient pair (no non-symmetric S confusing the chain rule). */
  if (diagWeight < 0.0) PetscFunctionReturn(PETSC_SUCCESS);

  /* ============================================================== */
  /* Fully-parallel block-Jacobi Gauss-Seidel path. Each rank does  */
  /* a forward sweep on its OWNED cells using local + ghost values, */
  /* exchanges ghosts, does a reverse sweep, exchanges ghosts. No   */
  /* rank-0 bottleneck. Result is a valid GS smoother but not bit-  */
  /* identical to the sequential rank-0 path (sequential ordering   */
  /* across all cells is impossible to replicate in parallel; see   */
  /* setupParallelSmoothingGraph docstring).                        */
  /* ============================================================== */
  if (graph->hasParallelGraph) {
    PetscInt nOwned = graph->numLocalCells;

    /* 1. Copy owned values from input local Vec into the overlap-1
     *    local scratch (owned slot only; ghost slot still stale). */
    {
      const PetscScalar *arrIn;
      PetscScalar       *arrOver;
      PetscCall(VecGetArrayRead(v, &arrIn));
      PetscCall(VecGetArray(graph->oLocalScratch, &arrOver));
      for (PetscInt i = 0; i < nOwned; i++) arrOver[i] = arrIn[i];
      PetscCall(VecRestoreArray(graph->oLocalScratch, &arrOver));
      PetscCall(VecRestoreArrayRead(v, &arrIn));
    }

    /* 2. Refresh ghosts: L→G (owners contribute) then G→L (pull ghosts). */
    PetscCall(DMLocalToGlobal(graph->dmInversionOver, graph->oLocalScratch,
                               INSERT_VALUES, graph->oGlobalScratch));
    PetscCall(DMGlobalToLocal(graph->dmInversionOver, graph->oGlobalScratch,
                               INSERT_VALUES, graph->oLocalScratch));

    /* 3. Forward sweep on owned cells. */
    {
      PetscScalar *arrOver;
      PetscCall(VecGetArray(graph->oLocalScratch, &arrOver));
      for (PetscInt i = 0; i < nOwned; i++) {
        if (graph->isFixed[i]) continue;
        PetscScalar newVal = diagWeight * arrOver[i];
        for (PetscInt k = graph->oNeighborStart[i];
             k < graph->oNeighborStart[i + 1]; k++)
          newVal += graph->oNeighborWeights[k] * arrOver[graph->oNeighborList[k]];
        arrOver[i] = newVal;
      }
      PetscCall(VecRestoreArray(graph->oLocalScratch, &arrOver));
    }

    /* 4. Refresh ghosts so other ranks see updated owned values. */
    PetscCall(DMLocalToGlobal(graph->dmInversionOver, graph->oLocalScratch,
                               INSERT_VALUES, graph->oGlobalScratch));
    PetscCall(DMGlobalToLocal(graph->dmInversionOver, graph->oGlobalScratch,
                               INSERT_VALUES, graph->oLocalScratch));

    /* 5. Reverse sweep. */
    {
      PetscScalar *arrOver;
      PetscCall(VecGetArray(graph->oLocalScratch, &arrOver));
      for (PetscInt i = nOwned - 1; i >= 0; i--) {
        if (graph->isFixed[i]) continue;
        PetscScalar newVal = diagWeight * arrOver[i];
        for (PetscInt k = graph->oNeighborStart[i];
             k < graph->oNeighborStart[i + 1]; k++)
          newVal += graph->oNeighborWeights[k] * arrOver[graph->oNeighborList[k]];
        arrOver[i] = newVal;
      }

      /* 6. Copy owned slot back into input local Vec. */
      PetscScalar *arrOut;
      PetscCall(VecGetArray(v, &arrOut));
      for (PetscInt i = 0; i < nOwned; i++) arrOut[i] = arrOver[i];
      PetscCall(VecRestoreArray(v, &arrOut));
      PetscCall(VecRestoreArray(graph->oLocalScratch, &arrOver));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* ============================================================== */
  /* Single-rank path: forward+reverse sweep on the input local Vec. */
  /* The setup phase leaves hasParallelGraph=false at MPI=1 because  */
  /* the local mesh IS the global mesh; this loop produces the       */
  /* correct sequential GS result with no overhead.                  */
  /* ============================================================== */
  PetscInt     N;
  PetscScalar *arr;
  PetscCall(VecGetLocalSize(v, &N));
  PetscCall(VecGetArray(v, &arr));

  /* Forward sweep */
  for (PetscInt i = 0; i < graph->numLocalCells && i < N; i++) {
    if (graph->isFixed[i]) continue;
    PetscScalar newVal = diagWeight * arr[i];
    for (PetscInt k = graph->neighborStart[i];
         k < graph->neighborStart[i + 1]; k++) {
      PetscInt nb = graph->neighborList[k];
      if (nb < N) newVal += graph->neighborWeights[k] * arr[nb];
    }
    arr[i] = newVal;
  }

  /* Reverse sweep */
  for (PetscInt i = graph->numLocalCells - 1; i >= 0; i--) {
    if (i >= N) continue;
    if (graph->isFixed[i]) continue;
    PetscScalar newVal = diagWeight * arr[i];
    for (PetscInt k = graph->neighborStart[i];
         k < graph->neighborStart[i + 1]; k++) {
      PetscInt nb = graph->neighborList[k];
      if (nb < N) newVal += graph->neighborWeights[k] * arr[nb];
    }
    arr[i] = newVal;
  }

  PetscCall(VecRestoreArray(v, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* ================================================================== */
/* setupParallelSmoothingGraph                                        */
/*                                                                     */
/* Builds the per-rank overlap=1 neighbor graph used by the fully-    */
/* parallel block-Jacobi GS path of applyGaussSeidelSmoothing. Steps: */
/*   1. DMPlexDistributeOverlap(dm, 1, …) → permanent overlap=1 EM-DM */
/*      kept alive for the lifetime of the inversion run.             */
/*   2. Clone the EM-DM-over and install a 1-DOF/cell PetscSection on */
/*      it, identical in shape to the original dmInversion section.   */
/*      The owned-cell DOFs are at the SAME local positions as in     */
/*      dmInversion (0..nOwned-1), so direct memcpy-style transfer    */
/*      works between them.                                           */
/*   3. Walk vertex stars on the overlap-1 DM for each owned cell;    */
/*      neighbor list now includes ghost cells (local-overlap-1       */
/*      indices); compute 1/dist weights and normalize.               */
/*   4. Allocate persistent local + global Vec scratch on the new DM. */
/*                                                                     */
/* Result is NOT bit-identical to the sequential rank-0 sweep —       */
/* parallel GS uses one-step-stale ghost values during each sweep,    */
/* whereas sequential GS sees the most recently updated value of      */
/* every neighbor. Recovered model still matches; same convergence.   */
/* ================================================================== */
PetscErrorCode setupParallelSmoothingGraph(NeighborGraph *graph,
                                            const DM       dm,
                                            const Grid    *grid)
{
  PetscFunctionBeginUser;

  MPI_Comm    comm = PetscObjectComm((PetscObject)dm);
  PetscMPIInt size;
  PetscCallMPI(MPI_Comm_size(comm, &size));

  if (size == 1) PetscFunctionReturn(PETSC_SUCCESS);

  /* ---- 1. Permanent overlap=1 EM-DM ---- */
  DM      dmEMOver = NULL;
  PetscSF sfDist;
  PetscCall(DMPlexDistributeOverlap(dm, 1, &sfDist, &dmEMOver));
  PetscCall(PetscSFDestroy(&sfDist));
  if (!dmEMOver) PetscFunctionReturn(PETSC_SUCCESS);

  /* ---- 2. Build dmInversionOver: clone + install 1-DOF/cell section ---- */
  DM dmInvOver;
  PetscCall(DMClone(dmEMOver, &dmInvOver));

  /* Register a 1-component cell-centered field (matches createInversionDM). */
  PetscFV fvm;
  PetscCall(PetscFVCreate(comm, &fvm));
  PetscCall(PetscFVSetNumComponents(fvm, 1));
  PetscCall(PetscObjectSetName((PetscObject)fvm, "scalar"));
  PetscCall(DMAddField(dmInvOver, NULL, (PetscObject)fvm));
  PetscCall(PetscFVDestroy(&fvm));

  PetscInt cStartO, cEndO;
  PetscCall(DMPlexGetHeightStratum(dmInvOver, 0, &cStartO, &cEndO));

  PetscInt pStart, pEnd;
  PetscCall(DMPlexGetChart(dmInvOver, &pStart, &pEnd));

  PetscSection sec;
  PetscCall(PetscSectionCreate(comm, &sec));
  PetscCall(PetscSectionSetNumFields(sec, 1));
  PetscCall(PetscSectionSetFieldComponents(sec, 0, 1));
  PetscCall(PetscSectionSetChart(sec, pStart, pEnd));
  for (PetscInt i = cStartO; i < cEndO; i++) {
    PetscCall(PetscSectionSetDof(sec, i, 1));
    PetscCall(PetscSectionSetFieldDof(sec, i, 0, 1));
  }
  PetscCall(PetscSectionSetUp(sec));
  PetscCall(DMSetLocalSection(dmInvOver, sec));
  PetscCall(PetscSectionDestroy(&sec));

  /* dmEMOver is no longer needed; dmInvOver clone holds its own ref. */
  PetscCall(DMDestroy(&dmEMOver));

  graph->dmInversionOver = dmInvOver;

  /* ---- 3. Walk vertex stars; build local-overlap-1 neighbor graph ---- */
  PetscInt nCellsOver = cEndO - cStartO;
  PetscInt nOwned     = grid->cellEnd - grid->cellStart;

  PetscReal *centroids;
  PetscCall(PetscMalloc1(nCellsOver * NUM_DIMENSIONS, &centroids));
  for (PetscInt i = cStartO; i < cEndO; i++) {
    Cell cell;
    PetscCall(extractCellCoordinates(dmInvOver, i, &cell));
    PetscCall(computeCellCentroid(&cell));
    centroids[(i - cStartO) * NUM_DIMENSIONS + 0] = cell.centroid[0];
    centroids[(i - cStartO) * NUM_DIMENSIONS + 1] = cell.centroid[1];
    centroids[(i - cStartO) * NUM_DIMENSIONS + 2] = cell.centroid[2];
  }

  PetscInt  maxNeighbors = 200;
  PetscInt *tmpNeighbors;
  PetscInt *tmpCount;
  PetscCall(PetscMalloc1(nOwned * maxNeighbors, &tmpNeighbors));
  PetscCall(PetscCalloc1(nOwned, &tmpCount));

  for (PetscInt i = cStartO; i < cStartO + nOwned; i++) {
    PetscInt liOver  = i - cStartO;
    PetscInt liOwned = i - cStartO;  /* same in this DM (owned first) */

    if (graph->isFixed[liOwned]) {
      tmpNeighbors[liOver * maxNeighbors + 0] = liOwned;  /* self */
      tmpCount[liOver] = 1;
      continue;
    }

    PetscInt  closureSize = 0;
    PetscInt *closure     = NULL;
    PetscCall(DMPlexGetTransitiveClosure(dmInvOver, i, PETSC_TRUE,
                                          &closureSize, &closure));
    PetscInt cnt = 0;
    for (PetscInt ci = 0; ci < closureSize * 2; ci += 2) {
      PetscInt point = closure[ci];
      PetscInt pdepth = 0;
      PetscCall(DMPlexGetPointDepth(dmInvOver, point, &pdepth));
      if (pdepth != 0) continue;

      PetscInt  starSize = 0;
      PetscInt *star     = NULL;
      PetscCall(DMPlexGetTransitiveClosure(dmInvOver, point, PETSC_FALSE,
                                            &starSize, &star));
      for (PetscInt s = 0; s < starSize * 2; s += 2) {
        PetscInt nb = star[s];
        if (nb == i) continue;
        if (nb < cStartO || nb >= cEndO) continue;
        PetscInt nbLi = nb - cStartO;

        PetscBool found = PETSC_FALSE;
        for (PetscInt k = 0; k < cnt; k++)
          if (tmpNeighbors[liOver * maxNeighbors + k] == nbLi) {
            found = PETSC_TRUE; break;
          }
        if (!found && cnt < maxNeighbors) {
          tmpNeighbors[liOver * maxNeighbors + cnt] = nbLi;
          cnt++;
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dmInvOver, point, PETSC_FALSE,
                                                &starSize, &star));
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dmInvOver, i, PETSC_TRUE,
                                              &closureSize, &closure));

    if (cnt == 0) {
      tmpNeighbors[liOver * maxNeighbors + 0] = liOwned;
      tmpCount[liOver] = 1;
    } else {
      tmpCount[liOver] = cnt;
    }
  }

  /* CSR */
  PetscCall(PetscMalloc1(nOwned + 1, &graph->oNeighborStart));
  graph->oNeighborStart[0] = 0;
  for (PetscInt i = 0; i < nOwned; i++)
    graph->oNeighborStart[i + 1] = graph->oNeighborStart[i] + tmpCount[i];

  PetscInt totalArcs = graph->oNeighborStart[nOwned];
  PetscCall(PetscMalloc1(totalArcs, &graph->oNeighborList));
  PetscCall(PetscMalloc1(totalArcs, &graph->oNeighborWeights));

  for (PetscInt i = 0; i < nOwned; i++) {
    PetscInt start = graph->oNeighborStart[i];
    PetscInt cnt   = tmpCount[i];

    if (graph->isFixed[i] || cnt == 1) {
      graph->oNeighborList[start]    = i;
      graph->oNeighborWeights[start] = 1.0;
      continue;
    }

    PetscReal cx = centroids[i * NUM_DIMENSIONS + 0];
    PetscReal cy = centroids[i * NUM_DIMENSIONS + 1];
    PetscReal cz = centroids[i * NUM_DIMENSIONS + 2];
    PetscReal wsum = 0.0;
    for (PetscInt k = 0; k < cnt; k++) {
      PetscInt  ln = tmpNeighbors[i * maxNeighbors + k];
      PetscReal nx = centroids[ln * NUM_DIMENSIONS + 0];
      PetscReal ny = centroids[ln * NUM_DIMENSIONS + 1];
      PetscReal nz = centroids[ln * NUM_DIMENSIONS + 2];
      PetscReal d  = PetscSqrtReal((nx-cx)*(nx-cx) +
                                   (ny-cy)*(ny-cy) +
                                   (nz-cz)*(nz-cz));
      graph->oNeighborList[start + k]    = ln;
      graph->oNeighborWeights[start + k] = (d > 0.0) ? 1.0 / d : 0.0;
      wsum += graph->oNeighborWeights[start + k];
    }
    if (wsum > 0.0)
      for (PetscInt k = 0; k < cnt; k++)
        graph->oNeighborWeights[start + k] /= wsum;
  }

  PetscCall(PetscFree(tmpNeighbors));
  PetscCall(PetscFree(tmpCount));
  PetscCall(PetscFree(centroids));

  /* ---- 4. Persistent scratch Vecs on dmInvOver ---- */
  PetscCall(DMCreateLocalVector(dmInvOver, &graph->oLocalScratch));
  PetscCall(DMCreateGlobalVector(dmInvOver, &graph->oGlobalScratch));

  graph->hasParallelGraph = PETSC_TRUE;

  /* Per-rank report (collective via PetscPrintf with PETSC_COMM_WORLD).
   * Each rank's owned-cell + arc count differs by partition; emit on
   * rank 0 only as a summary. */
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscInt totalArcsGlobal = 0;
  PetscCallMPI(MPI_Reduce(&totalArcs, &totalArcsGlobal, 1, MPIU_INT,
                           MPI_SUM, 0, comm));
  if (rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
      "\n Parallel smoother graph (block-Jacobi, overlap=1):\n"
      "   Total arcs (global) = %" PetscInt_FMT "\n"
      "   Per-iter cost       = 2 ghost exchanges + local sweep\n",
      totalArcsGlobal));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* buildNotFixedMask                                                   */
/*                                                                     */
/* Builds a 0/1 mask vector on dmInversion (1 DOF/cell):              */
/*   mask[cell] = 0  if cell->isFixed                                 */
/*   mask[cell] = 1  otherwise                                        */
/*                                                                     */
/* The mask replaces the fragile DMPlexVecSetClosure-on-global-Vec    */
/* pattern for zeroing gradient entries at fixed elements.            */
/* ================================================================== */
PetscErrorCode buildNotFixedMask(const NeighborGraph *graph,
                                  DM                  dmInversion,
                                  const Grid         *grid,
                                  Vec                *maskGlobal,
                                  Vec                *maskLocal)
{
  PetscFunctionBeginUser;

  PetscCall(DMCreateLocalVector(dmInversion, maskLocal));
  PetscCall(VecSet(*maskLocal, 1.0));

  /* Zero the local mask entry for each fixed cell using the
   * section offset — local section has exactly 1 DOF/cell. */
  PetscSection localSec;
  PetscCall(DMGetLocalSection(dmInversion, &localSec));

  PetscScalar *arr;
  PetscCall(VecGetArray(*maskLocal, &arr));
  for (PetscInt li = 0; li < graph->numLocalCells; li++) {
    if (!graph->isFixed[li]) continue;
    PetscInt cellID = grid->cellStart + li;
    PetscInt offset;
    PetscCall(PetscSectionGetOffset(localSec, cellID, &offset));
    arr[offset] = 0.0;
  }
  PetscCall(VecRestoreArray(*maskLocal, &arr));

  /* Global mirror — INSERT_VALUES is unambiguous because each cell
   * is owned by exactly one rank for a cell-based 1-DOF section. */
  PetscCall(DMCreateGlobalVector(dmInversion, maskGlobal));
  PetscCall(VecSet(*maskGlobal, 0.0));
  PetscCall(DMLocalToGlobal(dmInversion, *maskLocal, INSERT_VALUES,
                             *maskGlobal));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* applyLogToSigma                                                     */
/*                                                                     */
/* sigma = 1 / exp(X_smooth + X0)   (isotropic)                      */
/*                                                                     */
/* X and X0 are global Vecs on dmInversion (1 DOF/cell).              */
/* sigmaModel is a local Vec on dmConductivity (3 DOF/cell).           */
/* Components 0-2 (res_x, res_y, res_z) are set to sigma.            */
/*                                                                     */
/* graph/diagWeight: apply the same Gauss-Seidel smoothing to a local */
/* copy of X before the exp conversion, matching MATLAB's tempX       */
/* smoothing after each L-BFGS step. X itself is NOT modified.        */
/* ================================================================== */
PetscErrorCode applyLogToSigma(DM dmInversion, DM dmConductivity,
                                const Vec X, const Vec X0,
                                Vec sigmaModel, const Grid *grid,
                                const NeighborGraph *graph,
                                PetscReal diagWeight,
                                Vec xPostSmoothOut)
{
  PetscFunctionBeginUser;

  /* Scatter global X and X0 to local Vecs on the 1-DOF inversion DM */
  Vec xLocal, x0Local;
  PetscCall(DMCreateLocalVector(dmInversion, &xLocal));
  PetscCall(DMCreateLocalVector(dmInversion, &x0Local));
  PetscCall(DMGlobalToLocal(dmInversion, X,  INSERT_VALUES, xLocal));
  PetscCall(DMGlobalToLocal(dmInversion, X0, INSERT_VALUES, x0Local));

  /* MATLAB applies forward + reverse Gauss-Seidel smoothing to the
   * model perturbation (tempX = X_lbfgs) before converting to sigma.
   * Apply the same sweep here on a local copy so that the assembled
   * conductivity is always spatially smooth, matching the reference.
   * The optimizer's X is not modified — only the physical sigma changes. */
  PetscCall(applyGaussSeidelSmoothing(graph, diagWeight, xLocal));

  /* Expose the smoothed tempX to the caller for VTU diagnostics (MATLAB
   * X_guanghuahou). Copy happens before exp() so the user sees the same
   * log-space quantity the reference writes. */
  if (xPostSmoothOut) PetscCall(VecCopy(xLocal, xPostSmoothOut));

  /* Get section offsets for the conductivity Vec (3 DOF/cell) */
  PetscSection resSec;
  PetscCall(DMGetLocalSection(dmConductivity, &resSec));

  const PetscScalar *xArr, *x0Arr;
  PetscScalar       *sArr;
  PetscCall(VecGetArrayRead(xLocal,   &xArr));
  PetscCall(VecGetArrayRead(x0Local,  &x0Arr));
  PetscCall(VecGetArray(sigmaModel,   &sArr));

  for (PetscInt i = grid->cellStart; i < grid->cellEnd; i++) {
    PetscInt li = i - grid->cellStart;
    PetscInt resOff;
    PetscCall(PetscSectionGetOffset(resSec, i, &resOff));

    PetscScalar sigma = 1.0 / PetscExpScalar(
        PetscRealPart(xArr[li] + x0Arr[li]));

    sArr[resOff + 0] = sigma;  /* res_x */
    sArr[resOff + 1] = sigma;  /* res_y */
    sArr[resOff + 2] = sigma;  /* res_z */
  }

  PetscCall(VecRestoreArray(sigmaModel,   &sArr));
  PetscCall(VecRestoreArrayRead(x0Local,  &x0Arr));
  PetscCall(VecRestoreArrayRead(xLocal,   &xArr));
  PetscCall(VecDestroy(&xLocal));
  PetscCall(VecDestroy(&x0Local));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* destroyNeighborGraph                                                */
/* ================================================================== */
PetscErrorCode destroyNeighborGraph(NeighborGraph *graph)
{
  PetscFunctionBeginUser;
  /* Owned-cell graph (always allocated by buildNeighborSmoothingGraph). */
  PetscCall(PetscFree(graph->neighborStart));
  PetscCall(PetscFree(graph->neighborList));
  PetscCall(PetscFree(graph->neighborWeights));
  PetscCall(PetscFree(graph->isFixed));

  /* Parallel block-Jacobi resources (set by setupParallelSmoothingGraph;
   * remain NULL on a single MPI rank). */
  if (graph->oNeighborStart)   PetscCall(PetscFree(graph->oNeighborStart));
  if (graph->oNeighborList)    PetscCall(PetscFree(graph->oNeighborList));
  if (graph->oNeighborWeights) PetscCall(PetscFree(graph->oNeighborWeights));
  if (graph->oLocalScratch)    PetscCall(VecDestroy(&graph->oLocalScratch));
  if (graph->oGlobalScratch)   PetscCall(VecDestroy(&graph->oGlobalScratch));
  if (graph->dmInversionOver)  PetscCall(DMDestroy(&graph->dmInversionOver));
  PetscFunctionReturn(PETSC_SUCCESS);
}
