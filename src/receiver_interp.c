/*
 * Filename: receiver_interp.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Shared receiver-interpolation operator for both CSEM kernels.
 */

/*
 * Notes:
 * Conventions (sign, dofSigns, basis evaluation order) match the forward
 * kernel; the column indexing uses the global section so cross-rank ghost
 * DOFs are routed correctly.
 *
 * MPI-invariance of Q requires TWO things, both handled below:
 *   (1) COLUMNS: closure DOFs addressed through the GLOBAL section, so a
 *       receiver's row carries the same global column ids on every rank.
 *   (2) ROWS: each receiver assembled EXACTLY ONCE. Because `receivers` is a
 *       replicated (COMM_SELF) Vec, every rank locates every receiver against
 *       its local cells, and a receiver on a partition boundary is found by
 *       several ranks; with ADD_VALUES those duplicates would accumulate and
 *       make Q (and Q*x, the misfit, the inversion) rank-count-dependent. Each
 *       receiver is therefore assigned to a single owner (the lowest rank that
 *       located it) and skipped on every other rank. */

#include "common.h"
#include "receiver_interp.h"

#include "constants.h"
#include "fem.h"
#include "grid.h"

#include <petsc.h>
#include <petscdmplex.h>
#include <petscviewerhdf5.h>

/**
 * @brief Decodes a global DOF index returned by DMPlexGetClosureIndices
 *        against a global PetscSection.
 *
 * PETSc encodes ghost DOFs (owned by another rank but reachable through
 * the closure) as `-(global_offset + 1)`, and owned DOFs as the offset
 * itself.  This helper restores the positive global offset uniformly,
 * documenting the convention so the call site doesn't carry the magic
 * `-(g + 1)` formula inline.
 *
 * @param[in] rawIdx Value pulled from the globalIdx[] array.
 *
 * @return Non-negative global DOF offset.
 */
static inline PetscInt decodeGlobalDOF(PetscInt rawIdx)
{
  return (rawIdx >= 0) ? rawIdx : -(rawIdx + 1);
}

/**
 * @brief Builds the six receiver-interpolation matrices QEx..QHz.
 *
 * Q = {QEx, QEy, QEz, QHx, QHy, QHz} maps an H(curl) DOF vector to the
 * electric/magnetic field components evaluated at the receiver positions.
 * Used by both kernels (fm.csem in postprocessing.c, im.csem in inversion.c).
 * The full contract - receiver-location semantics, sign convention,
 * MPI-invariance argument, and the divide-by-(iωμ) factor for H - is
 * documented in include/receiver_interp.h.
 *
 * @param[in]  order       Nédélec basis order (dispatched via fem->ops, 1..6).
 * @param[in]  receivers  Serial Vec of 3·N_recv reals (caller-owned).
 * @param[in]  dm         H(curl) DM the solution lives on.
 * @param[in]  grid       Grid struct produced by setupCsemGrid.
 * @param[out] Q          Output struct holding QEx..QHz.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode buildReceiverInterpolationMatrices(PetscInt    order,
                                                  Vec         receivers,
                                                  const DM    dm,
                                                  const Grid *grid,
                                                  ReceiverInterpolationMatrices *Q)
{
  PetscFunctionBeginUser;

  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* Basis evaluation goes through evaluateNedelecBasis (see include/fem.h), which builds on the PETGEM-style reference Nedelec element covering
   * order = 1..6. */
  PetscCheck(order >= 1 && order <= 6, comm, PETSC_ERR_SUP, "buildReceiverInterpolationMatrices: order must be in 1..6  (got %" PetscInt_FMT ")", order);

  /* `receivers` is owned by the caller (produced by loadCsemInputs from the /receivers Vec inside the unified input HDF5). This routine only
   * reads it: locates the points in the mesh and assembles the Q matrices. */
  PetscInt globalSizeReceivers, numGlobalReceivers;
  PetscCall(VecGetSize(receivers, &globalSizeReceivers));

  PetscCheck(globalSizeReceivers % NUM_DIMENSIONS == 0, comm, PETSC_ERR_ARG_SIZ,
             "Receiver vector size %" PetscInt_FMT " not divisible by %d", globalSizeReceivers, NUM_DIMENSIONS);

  numGlobalReceivers = globalSizeReceivers / NUM_DIMENSIONS;
  Q->numReceivers    = numGlobalReceivers;

  /* Locate receivers in mesh */
  PetscSF            receiverSF = NULL;
  PetscInt           numFound;
  const PetscSFNode *recvInCell;
  const PetscInt    *recvFound;

  PetscCall(DMLocatePoints(dm, receivers, DM_POINTLOCATION_REMOVE, &receiverSF));
  PetscCall(PetscSFGetGraph(receiverSF, NULL, &numFound, &recvFound, &recvInCell));

 /* Receivers on partition boundaries may be located by multiple ranks. Assign ownership to the lowest-ranked locator so each receiver row 
  * is assembled exactly once, avoiding MPI-decomposition-dependent double counting. Receivers exactly on an interface can still show 
  * small rank-dependent differences because different adjacent cells may be selected for interpolation. */
  PetscMPIInt rank, nprocs;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &nprocs));
  PetscInt  *foundByRank, *ownerRank;
  PetscBool *rowAssembled;
  PetscCall(PetscMalloc3(numGlobalReceivers, &foundByRank, numGlobalReceivers, &ownerRank, numGlobalReceivers, &rowAssembled));
  for (PetscInt r = 0; r < numGlobalReceivers; r++) { 
    foundByRank[r] = nprocs; rowAssembled[r] = PETSC_FALSE; 
  }
  
  for (PetscInt i = 0; i < numFound; i++) {
    PetscInt ridx = recvFound ? recvFound[i] : i;
    if (recvInCell[i].index >= 0 && ridx >= 0 && ridx < numGlobalReceivers && rank < foundByRank[ridx]) {
      foundByRank[ridx] = rank;
    }
  }
  PetscCallMPI(MPI_Allreduce(foundByRank, ownerRank, numGlobalReceivers, MPIU_INT, MPI_MIN, comm));

  /* Global DOF count for matrix column size */
  Vec  tmpVec;
  PetscInt M, m;
  PetscCall(DMCreateGlobalVector(dm, &tmpVec));
  PetscCall(VecGetSize(tmpVec, &M));
  PetscCall(VecGetLocalSize(tmpVec, &m));
  PetscCall(VecDestroy(&tmpVec));
  Q->numDof = M;

  /* Create 6 sparse matrices (AIJ), Nrec rows x Ndof cols */
  PetscCall(MatCreate(comm, &Q->QEx));
  PetscCall(MatSetSizes(Q->QEx, PETSC_DECIDE, m, numGlobalReceivers, M));
  PetscCall(MatSetType(Q->QEx, MATAIJ));
  PetscCall(MatSetUp(Q->QEx));

  PetscCall(MatDuplicate(Q->QEx, MAT_DO_NOT_COPY_VALUES, &Q->QEy));
  PetscCall(MatDuplicate(Q->QEx, MAT_DO_NOT_COPY_VALUES, &Q->QEz));
  PetscCall(MatDuplicate(Q->QEx, MAT_DO_NOT_COPY_VALUES, &Q->QHx));
  PetscCall(MatDuplicate(Q->QEx, MAT_DO_NOT_COPY_VALUES, &Q->QHy));
  PetscCall(MatDuplicate(Q->QEx, MAT_DO_NOT_COPY_VALUES, &Q->QHz));

  /* Allocate FEM basis arrays */
  PetscReal **Ni, **NiCurl, *XiEtaZeta;

  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &NiCurl));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &XiEtaZeta));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(grid->numDofInCell, &Ni[i]));
    PetscCall(PetscCalloc1(grid->numDofInCell, &NiCurl[i]));
  }

  PetscSection    section;
  const PetscScalar *coords;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(VecGetArrayRead(receivers, &coords));

  /* Fill Q matrices row by row (one row per receiver) */
  PetscInt numSkipped = 0;
  for (PetscInt i = 0; i < numFound; i++) {
    PetscInt ridx  = recvFound ? recvFound[i] : i;
    PetscInt cellID = recvInCell[i].index;
    if (cellID < 0) {
      continue;
    }

    /* Only the canonical owner assembles this receiver's row, and only once (a rank may locate the same on-interface 
    * receiver in two local cells). Every other locating rank skips it, so the row is written exactly once. */
    if (rank != ownerRank[ridx] || rowAssembled[ridx]) {
      continue;
    }
    rowAssembled[ridx] = PETSC_TRUE;

    PetscReal recvCoords[NUM_DIMENSIONS];
    recvCoords[0] = PetscRealPart(coords[3 * ridx]);
    recvCoords[1] = PetscRealPart(coords[3 * ridx + 1]);
    recvCoords[2] = PetscRealPart(coords[3 * ridx + 2]);

    Cell cell;
    PetscCall(extractCellCoordinates(dm, cellID, &cell));
    PetscCall(computeCellJacobian(&cell));

    /* If the located cell is degenerate (near-zero volume), search neighboring cells sharing a vertex 
       for a valid alternative. DMLocatePoints may place receivers on partition boundaries or in thin 
       surface cells where |detJ| <= PETSC_SMALL.       */
    if (PetscAbsReal(cell.detJacobian) <= PETSC_SMALL) {
      PetscInt        altCell = -1;
      PetscInt        closureSize = 0;
      PetscInt       *closure     = NULL;
      PetscCall(DMPlexGetTransitiveClosure(dm, cellID, PETSC_TRUE, &closureSize, &closure));
      for (PetscInt j = 0; j < closureSize * 2; j += 2) {
        PetscInt point = closure[j];
        PetscInt pdepth;
        PetscCall(DMPlexGetPointDepth(dm, point, &pdepth));
        
        if (pdepth != 0) {
          continue;  /* only vertices */
        }
        
        PetscInt        supportSize;
        const PetscInt *support;
        PetscCall(DMPlexGetSupportSize(dm, point, &supportSize));
        PetscCall(DMPlexGetSupport(dm, point, &support));
        
        for (PetscInt k = 0; k < supportSize; k++) {
          PetscInt neighbor = support[k];
          if (neighbor == cellID) {
            continue;
          }
          if (neighbor < grid->cellStart || neighbor >= grid->cellEnd) {
            continue;
          }
          
          Cell ncell;
          PetscCall(extractCellCoordinates(dm, neighbor, &ncell));
          PetscCall(computeCellJacobian(&ncell));
          
          if (PetscAbsReal(ncell.detJacobian) > PETSC_SMALL) {
            altCell = neighbor;
            break;
          }
        }
        if (altCell >= 0) {
          break;
        }
      }
      
      PetscCall(DMPlexRestoreTransitiveClosure(dm, cellID, PETSC_TRUE, &closureSize, &closure));
      
      if (altCell >= 0) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,
          "   WARNING: receiver %" PetscInt_FMT " in degenerate cell %" PetscInt_FMT
          " (|detJ| = %.2e); relocated to cell %" PetscInt_FMT ".\n",
          ridx, cellID, (double)PetscAbsReal(cell.detJacobian), altCell));
        
        cellID = altCell;
        PetscCall(extractCellCoordinates(dm, cellID, &cell));
        PetscCall(computeCellJacobian(&cell));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,
          "   WARNING: receiver %" PetscInt_FMT " in degenerate cell %" PetscInt_FMT
          " (|detJ| = %.2e, coords = [%.4e, %.4e, %.4e]); no valid neighbor"
          " found, skipping.\n",
          ridx, cellID, (double)PetscAbsReal(cell.detJacobian),
          (double)recvCoords[0], (double)recvCoords[1], (double)recvCoords[2]));
        
        numSkipped++;
        continue;
      }
    }

    PetscCall(tetrahedronXYZToReference(cell.coordinates, recvCoords, XiEtaZeta));

    /* order-agnostic basis + curl evaluation (1..6). The returned values and curls are already physical and geometrically oriented, 
     * matching the forward assembly, so no per-DOF sign correction is applied. */
    PetscCall(evaluateNedelecBasis(&grid->fem, &cell, XiEtaZeta, Ni, NiCurl));

    /* Get BOTH local and global DOF indices for this cell's closure.
     *
     * - Local section idxSection => local indices.  Negative entries mark
     *   BC-constrained DOFs (skip: they aren't free variables).
     * - Global section idxSection => global indices, with PETSc's ghost  encoding: a ghost DOF (owned by another rank) is returned as
     *   -(gOff + 1).  Owned DOFs are returned as a non-negative gOff.
     *
     * The two calls traverse the closure in the SAME order (driven by dmplex + useClosurePermutation=PETSC_TRUE), so slot k lines up */
    PetscSection globalSection;
    PetscCall(DMGetGlobalSection(dm, &globalSection));

    PetscInt  numLocal, *localIdx;
    PetscCall(DMPlexGetClosureIndices(dm, section, section, cellID, PETSC_TRUE, &numLocal, &localIdx, NULL, NULL));
    PetscInt  numGlobal, *globalIdxRaw;
    PetscCall(DMPlexGetClosureIndices(dm, section, globalSection, cellID, PETSC_TRUE, &numGlobal, &globalIdxRaw, NULL, NULL));

    for (PetscInt j = 0; j < grid->numDofInCell; j++) {
      if (localIdx[j] < 0) {
        continue;                       /* BC-constrained: skip */
      }
      PetscInt gidx = decodeGlobalDOF(globalIdxRaw[j]);
      if (gidx >= Q->numDof) {
        continue;                     /* safety */
      }

      PetscCall(MatSetValue(Q->QEx, ridx, gidx, Ni[0][j], ADD_VALUES));
      PetscCall(MatSetValue(Q->QEy, ridx, gidx, Ni[1][j], ADD_VALUES));
      PetscCall(MatSetValue(Q->QEz, ridx, gidx, Ni[2][j], ADD_VALUES));
      PetscCall(MatSetValue(Q->QHx, ridx, gidx, NiCurl[0][j], ADD_VALUES));
      PetscCall(MatSetValue(Q->QHy, ridx, gidx, NiCurl[1][j], ADD_VALUES));
      PetscCall(MatSetValue(Q->QHz, ridx, gidx, NiCurl[2][j], ADD_VALUES));
    }

    PetscCall(DMPlexRestoreClosureIndices(dm, section, globalSection, cellID, PETSC_TRUE, &numGlobal, &globalIdxRaw, NULL, NULL));
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, cellID, PETSC_TRUE, &numLocal, &localIdx, NULL, NULL));
  }

  PetscCall(VecRestoreArrayRead(receivers, &coords));

  if (numSkipped > 0) {
    PetscCall(PetscPrintf(comm, "\n   WARNING: %s receiver(s) skipped due to degenerate cells.\n", formatGroupedInt(numSkipped)));
  }

  /* Assemble all Q matrices */
  PetscCall(MatAssemblyBegin(Q->QEx, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Q->QEy, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Q->QEz, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Q->QHx, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Q->QHy, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(Q->QHz, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Q->QEx, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Q->QEy, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Q->QEz, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Q->QHx, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Q->QHy, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Q->QHz, MAT_FINAL_ASSEMBLY));

  /* Free FEM basis memory */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscFree(Ni[i]));
    PetscCall(PetscFree(NiCurl[i]));
  }
  PetscCall(PetscFree(Ni));
  PetscCall(PetscFree(NiCurl));
  PetscCall(PetscFree(XiEtaZeta));
  PetscCall(PetscSFDestroy(&receiverSF));
  PetscCall(PetscFree3(foundByRank, ownerRank, rowAssembled));
  /* Note: `receivers` Vec is owned by the caller and not destroyed here. */

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Destroys the six Mats owned by a ReceiverInterpolationMatrices struct.
 *
 * Idempotent on individual fields (MatDestroy handles NULL). The Q struct
 * itself is stack-owned by the caller and is not freed.
 *
 * @param[in,out] Q  Struct whose QEx..QHz matrices are destroyed.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode destroyReceiverInterpolationMatrices(ReceiverInterpolationMatrices *Q)
{
  PetscFunctionBeginUser;
  PetscCall(MatDestroy(&Q->QEx));
  PetscCall(MatDestroy(&Q->QEy));
  PetscCall(MatDestroy(&Q->QEz));
  PetscCall(MatDestroy(&Q->QHx));
  PetscCall(MatDestroy(&Q->QHy));
  PetscCall(MatDestroy(&Q->QHz));
  PetscFunctionReturn(PETSC_SUCCESS);
}
