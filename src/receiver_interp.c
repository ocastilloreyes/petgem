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
 * See receiver_interp.h for the interface and rationale.  The body of
 * buildReceiverInterpolationMatrices was moved out of inversion.c so
 * that postprocessing.c (fm.csem) and inversion.c (im.csem) can both
 * obtain Ex/Ey/Ez/Hx/Hy/Hz at receivers via a common Q*x MatMult,
 * guaranteeing MPI-invariant outputs at any rank count.
 *
 * Conventions (sign, dofSigns, basis evaluation order) match the forward
 * kernel; the column indexing uses the global section so cross-rank ghost
 * DOFs are routed correctly.
 */

#include "receiver_interp.h"

#include "constants.h"
#include "grid.h"
#include "hvfem.h"

#include <petsc.h>
#include <petscdmplex.h>
#include <petscviewerhdf5.h>

PetscErrorCode buildReceiverInterpolationMatrices(PetscInt    nord,
                                                  Vec         receivers,
                                                  const DM    dm,
                                                  const Grid *grid,
                                                  PetscBool   verbose,
                                                  ReceiverInterpolationMatrices *Q)
{
  PetscFunctionBeginUser;

  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* Basis evaluation is dispatched through evaluateNedelecBasis (see
   * include/hvfem.h) which routes to the per-order ops table — works for
   * any nord that registers a NedelecOps entry. The unified hierarchical
   * basis covers nord = 1..6. */
  PetscCheck(nord >= 1 && nord <= 6, comm, PETSC_ERR_SUP,
             "buildReceiverInterpolationMatrices: nord must be in 1..6 "
             "(got %" PetscInt_FMT ")", nord);

  /* `receivers` is owned by the caller (produced by loadCsemInputs from
   * the /receivers Vec inside the unified input HDF5). This routine only
   * reads it: locates the points in the mesh and assembles the Q matrices. */
  PetscInt globalSizeReceivers, numGlobalReceivers;
  PetscCall(VecGetSize(receivers, &globalSizeReceivers));

  PetscCheck(globalSizeReceivers % NUM_DIMENSIONS == 0,
             comm, PETSC_ERR_ARG_SIZ,
             "Receiver vector size %" PetscInt_FMT " not divisible by %d",
             globalSizeReceivers, NUM_DIMENSIONS);

  numGlobalReceivers = globalSizeReceivers / NUM_DIMENSIONS;
  Q->numReceivers    = numGlobalReceivers;

  /* Locate receivers in mesh */
  PetscSF            receiverSF = NULL;
  PetscInt           numFound;
  const PetscSFNode *recvInCell;
  const PetscInt    *recvFound;

  PetscCall(DMLocatePoints(dm, receivers, DM_POINTLOCATION_REMOVE,
                           &receiverSF));
  PetscCall(PetscSFGetGraph(receiverSF, NULL, &numFound,
                            &recvFound, &recvInCell));

  /* Diagnostic: total leaves reported by DMLocatePoints across all ranks.
   * Expected = numGlobalReceivers (one leaf per receiver).  Any excess is
   * a receiver found in >1 cell (shared face / edge / vertex) and will
   * double-accumulate Q rows via ADD_VALUES below. */
  {
    PetscInt localLeaves = numFound, globalLeaves = 0;
    PetscInt dupOnThisRank = 0;
    if (numFound > 0 && recvFound) {
      PetscInt *seen;
      PetscCall(PetscCalloc1(numGlobalReceivers, &seen));
      for (PetscInt j = 0; j < numFound; j++) {
        PetscInt ridx = recvFound[j];
        if (ridx >= 0 && ridx < numGlobalReceivers) {
          seen[ridx]++;
          if (seen[ridx] > 1) dupOnThisRank++;
        }
      }
      PetscCall(PetscFree(seen));
    }
    PetscInt globalDup = 0;
    PetscCallMPI(MPI_Allreduce(&localLeaves, &globalLeaves, 1, MPIU_INT,
                                MPI_SUM, comm));
    PetscCallMPI(MPI_Allreduce(&dupOnThisRank, &globalDup, 1, MPIU_INT,
                                MPI_SUM, comm));
    if (verbose) {
      PetscCall(PetscPrintf(comm,
        "\n DMLocatePoints diagnostic:\n"
        "   Global receivers     = %" PetscInt_FMT "\n"
        "   Total leaves found   = %" PetscInt_FMT
        " (excess = %" PetscInt_FMT ")\n"
        "   Intra-rank duplicates= %" PetscInt_FMT "\n",
        numGlobalReceivers, globalLeaves,
        globalLeaves - numGlobalReceivers, globalDup));
    }
  }

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
  PetscReal **Ni, **NiCurl, **coeffs, **Dx_Ni, **Dy_Ni, **Dz_Ni, *XiEtaZeta;

  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &NiCurl));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dx_Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dy_Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dz_Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &XiEtaZeta));
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    PetscCall(PetscCalloc1(grid->numDofInCell, &Ni[d]));
    PetscCall(PetscCalloc1(grid->numDofInCell, &NiCurl[d]));
    PetscCall(PetscCalloc1(grid->numDofInCell, &Dx_Ni[d]));
    PetscCall(PetscCalloc1(grid->numDofInCell, &Dy_Ni[d]));
    PetscCall(PetscCalloc1(grid->numDofInCell, &Dz_Ni[d]));
  }
  PetscCall(PetscCalloc1(grid->numDofInCell, &coeffs));
  for (PetscInt d = 0; d < grid->numDofInCell; d++)
    PetscCall(PetscCalloc1(grid->numDofInCell, &coeffs[d]));

  PetscInt *dofSigns;
  PetscCall(PetscCalloc1(grid->numDofInCell, &dofSigns));

  PetscSection    section;
  const PetscScalar *coords;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(VecGetArrayRead(receivers, &coords));

  /* Fill Q matrices row by row (one row per receiver) */
  PetscInt numSkipped = 0;
  for (PetscInt j = 0; j < numFound; j++) {
    PetscInt ridx  = recvFound ? recvFound[j] : j;
    PetscInt cellID = recvInCell[j].index;
    if (cellID < 0) continue;

    PetscReal recvCoords[NUM_DIMENSIONS];
    recvCoords[0] = PetscRealPart(coords[3 * ridx]);
    recvCoords[1] = PetscRealPart(coords[3 * ridx + 1]);
    recvCoords[2] = PetscRealPart(coords[3 * ridx + 2]);

    Cell cell;
    PetscCall(extractCellCoordinates(dm, cellID, &cell));
    PetscCall(computeCellJacobian(&cell));

    /* If the located cell is degenerate (near-zero volume), search
       neighboring cells sharing a vertex for a valid alternative.
       DMLocatePoints may place receivers on partition boundaries
       or in thin surface cells where |detJ| <= PETSC_SMALL.       */
    if (PetscAbsReal(cell.detJacobian) <= PETSC_SMALL) {
      PetscInt        altCell = -1;
      PetscInt        closureSize = 0;
      PetscInt       *closure     = NULL;
      PetscCall(DMPlexGetTransitiveClosure(dm, cellID, PETSC_TRUE,
                                           &closureSize, &closure));
      for (PetscInt ci = 0; ci < closureSize * 2; ci += 2) {
        PetscInt point = closure[ci];
        PetscInt pdepth;
        PetscCall(DMPlexGetPointDepth(dm, point, &pdepth));
        if (pdepth != 0) continue;  /* only vertices */
        PetscInt        supportSize;
        const PetscInt *support;
        PetscCall(DMPlexGetSupportSize(dm, point, &supportSize));
        PetscCall(DMPlexGetSupport(dm, point, &support));
        for (PetscInt s = 0; s < supportSize; s++) {
          PetscInt neighbor = support[s];
          if (neighbor == cellID) continue;
          if (neighbor < grid->cellStart || neighbor >= grid->cellEnd) continue;
          Cell ncell;
          PetscCall(extractCellCoordinates(dm, neighbor, &ncell));
          PetscCall(computeCellJacobian(&ncell));
          if (PetscAbsReal(ncell.detJacobian) > PETSC_SMALL) {
            altCell = neighbor;
            break;
          }
        }
        if (altCell >= 0) break;
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, cellID, PETSC_TRUE,
                                               &closureSize, &closure));
      if (altCell >= 0) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,
          "   Warning: receiver %" PetscInt_FMT " in degenerate cell %" PetscInt_FMT
          " (|detJ|=%.2e), relocated to cell %" PetscInt_FMT "\n",
          ridx, cellID, (double)PetscAbsReal(cell.detJacobian), altCell));
        cellID = altCell;
        PetscCall(extractCellCoordinates(dm, cellID, &cell));
        PetscCall(computeCellJacobian(&cell));
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,
          "   Warning: receiver %" PetscInt_FMT " in degenerate cell %" PetscInt_FMT
          " (|detJ|=%.2e, coords=[%.4e,%.4e,%.4e]), no valid neighbor found — skipping\n",
          ridx, cellID, (double)PetscAbsReal(cell.detJacobian),
          (double)recvCoords[0], (double)recvCoords[1], (double)recvCoords[2]));
        numSkipped++;
        continue;
      }
    }

    PetscCall(extractCellClousure(dm, cellID, &cell));
    PetscCall(computeCellOrientation(&cell));
    PetscCall(tetrahedronXYZToReference(cell.coordinates, recvCoords,
                                        XiEtaZeta));

    /* Nord-agnostic basis + curl evaluation through the per-order ops
     * table (cf. include/hvfem.h). Same routine assembly.c uses, so any
     * basis order with a registered NedelecOps works here uniformly. */
    PetscCall(evaluateNedelecBasis(&grid->fem, &cell, XiEtaZeta,
                                   coeffs, Dx_Ni, Dy_Ni, Dz_Ni,
                                   Ni, NiCurl));

    /* Per-DOF sign convention — same routine the forward assembly uses,
     * so Q aligns with the physical (signed) field evaluation. */
    PetscCall(buildDofSigns(&cell, &grid->fem, dofSigns));

    /* Get BOTH local and global DOF indices for this cell's closure.
     *
     * - Local section idxSection => local indices.  Negative entries mark
     *   BC-constrained DOFs (skip: they aren't free variables).
     * - Global section idxSection => global indices, with PETSc's ghost
     *   encoding: a ghost DOF (owned by another rank) is returned as
     *   -(gOff + 1).  Owned DOFs are returned as a non-negative gOff.
     *
     * The two calls traverse the closure in the SAME order (driven by
     * dmplex + useClosurePermutation=PETSC_TRUE), so slot k lines up.
     *
     * The prior code used the LOCAL index as a GLOBAL column to MatSetValue.
     * That works on 1 rank (local==global) but aliases across ranks on N>1,
     * producing a mis-addressed Q.  ‖Q‖_F is invariant (same set of values
     * written) but Q*x differs. */
    PetscSection globalSection;
    PetscCall(DMGetGlobalSection(dm, &globalSection));

    PetscInt  numLocal, *localIdx;
    PetscCall(DMPlexGetClosureIndices(dm, section, section, cellID,
                                      PETSC_TRUE, &numLocal, &localIdx,
                                      NULL, NULL));
    PetscInt  numGlobal, *globalIdxRaw;
    PetscCall(DMPlexGetClosureIndices(dm, section, globalSection, cellID,
                                      PETSC_TRUE, &numGlobal, &globalIdxRaw,
                                      NULL, NULL));

    for (PetscInt k = 0; k < grid->numDofInCell; k++) {
      if (localIdx[k] < 0) continue;         /* BC-constrained: skip */
      PetscInt gidx = globalIdxRaw[k];
      if (gidx < 0) gidx = -(gidx + 1);      /* decode ghost DOF */
      if (gidx < 0 || gidx >= Q->numDof) continue;  /* safety */

      PetscReal ori = (PetscReal)dofSigns[k];
      PetscCall(MatSetValue(Q->QEx, ridx, gidx,
                            Ni[0][k] * ori, ADD_VALUES));
      PetscCall(MatSetValue(Q->QEy, ridx, gidx,
                            Ni[1][k] * ori, ADD_VALUES));
      PetscCall(MatSetValue(Q->QEz, ridx, gidx,
                            Ni[2][k] * ori, ADD_VALUES));
      PetscCall(MatSetValue(Q->QHx, ridx, gidx,
                            NiCurl[0][k] * ori, ADD_VALUES));
      PetscCall(MatSetValue(Q->QHy, ridx, gidx,
                            NiCurl[1][k] * ori, ADD_VALUES));
      PetscCall(MatSetValue(Q->QHz, ridx, gidx,
                            NiCurl[2][k] * ori, ADD_VALUES));
    }

    PetscCall(DMPlexRestoreClosureIndices(dm, section, globalSection, cellID,
                                          PETSC_TRUE, &numGlobal,
                                          &globalIdxRaw, NULL, NULL));
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, cellID,
                                          PETSC_TRUE, &numLocal, &localIdx,
                                          NULL, NULL));
  }

  PetscCall(VecRestoreArrayRead(receivers, &coords));

  if (numSkipped > 0) {
    PetscCall(PetscPrintf(comm,
      "\n   Warning: %" PetscInt_FMT " receiver(s) skipped due to degenerate cells\n",
      numSkipped));
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
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    PetscCall(PetscFree(Ni[d]));
    PetscCall(PetscFree(NiCurl[d]));
    PetscCall(PetscFree(Dx_Ni[d]));
    PetscCall(PetscFree(Dy_Ni[d]));
    PetscCall(PetscFree(Dz_Ni[d]));
  }
  for (PetscInt d = 0; d < grid->numDofInCell; d++)
    PetscCall(PetscFree(coeffs[d]));
  PetscCall(PetscFree(Ni));
  PetscCall(PetscFree(NiCurl));
  PetscCall(PetscFree(Dx_Ni));
  PetscCall(PetscFree(Dy_Ni));
  PetscCall(PetscFree(Dz_Ni));
  PetscCall(PetscFree(coeffs));
  PetscCall(PetscFree(XiEtaZeta));
  PetscCall(PetscFree(dofSigns));
  PetscCall(PetscSFDestroy(&receiverSF));
  /* Note: `receivers` Vec is owned by the caller and not destroyed here. */

  PetscCall(PetscPrintf(comm,
    "\n Receiver interpolation matrices:\n"
    "   Receivers           = %" PetscInt_FMT "\n"
    "   DOFs                = %" PetscInt_FMT "\n",
    numGlobalReceivers, M));

  /* Diagnostic: Q-matrix norms + nnz.  These are global reductions — if
   * the Q entries and sparsity are correct, they must be MPI-invariant.
   * Any drift across rank counts pinpoints the Q assembly bug (cell
   * vertex ordering, dofSigns, Jacobian, basis evaluation). */
  if (verbose) {
    PetscReal qExNorm, qEyNorm, qEzNorm, qHxNorm;
    MatInfo   info;
    PetscCall(MatNorm(Q->QEx, NORM_FROBENIUS, &qExNorm));
    PetscCall(MatNorm(Q->QEy, NORM_FROBENIUS, &qEyNorm));
    PetscCall(MatNorm(Q->QEz, NORM_FROBENIUS, &qEzNorm));
    PetscCall(MatNorm(Q->QHx, NORM_FROBENIUS, &qHxNorm));
    PetscCall(MatGetInfo(Q->QEx, MAT_GLOBAL_SUM, &info));
    PetscCall(PetscPrintf(comm,
      "   ||QEx||_F = %.8e  nnz(QEx) = %.0f\n"
      "   ||QEy||_F = %.8e  ||QEz||_F = %.8e  ||QHx||_F = %.8e\n",
      (double)qExNorm, (double)info.nz_used,
      (double)qEyNorm, (double)qEzNorm, (double)qHxNorm));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

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
