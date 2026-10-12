/*
 * Filename: receiver_interp.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Shared receiver-interpolation operator (fm.csem, im.csem, fm.mt).
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
static inline PetscInt decodeGlobalDof(PetscInt rawIdx) {
  return (rawIdx >= 0) ? rawIdx : -(rawIdx + 1);
}

/* Lowest rank that located each receiver (nprocs when none did). */
static PetscErrorCode electReceiverOwners(MPI_Comm comm, PetscInt numReceivers, PetscInt numFound, const PetscInt *found,
                                          const PetscSFNode *inCell, PetscInt *owner) {
  PetscFunctionBeginUser;
  PetscMPIInt rank, nprocs;
  PetscInt   *mine;

  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &nprocs));
  PetscCall(PetscMalloc1(numReceivers, &mine));
  for (PetscInt r = 0; r < numReceivers; r++) mine[r] = nprocs;
  for (PetscInt i = 0; i < numFound; i++) {
    const PetscInt r = found ? found[i] : i;
    if (inCell[i].index >= 0 && r >= 0 && r < numReceivers) mine[r] = rank;
  }
  PetscCallMPI(MPI_Allreduce(mine, owner, numReceivers, MPIU_INT, MPI_MIN, comm));
  PetscCall(PetscFree(mine));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* First local cell sharing a vertex with cellId whose Jacobian is not degenerate, or -1. */
static PetscErrorCode findNonDegenerateNeighbor(DM dm, const Grid *grid, PetscInt cellId, PetscInt *neighborCell) {
  PetscFunctionBeginUser;
  PetscInt closureSize = 0, *closure = NULL;

  *neighborCell = -1;
  PetscCall(DMPlexGetTransitiveClosure(dm, cellId, PETSC_TRUE, &closureSize, &closure));
  for (PetscInt i = 0; i < closureSize * 2 && *neighborCell < 0; i += 2) {
    const PetscInt *support;
    PetscInt        depth, supportSize;

    PetscCall(DMPlexGetPointDepth(dm, closure[i], &depth));
    if (depth != 0) continue;
    PetscCall(DMPlexGetSupportSize(dm, closure[i], &supportSize));
    PetscCall(DMPlexGetSupport(dm, closure[i], &support));
    for (PetscInt j = 0; j < supportSize; j++) {
      const PetscInt c = support[j];
      Cell           cell;
      if (c == cellId || c < grid->cellStart || c >= grid->cellEnd) continue;
      PetscCall(extractCellCoordinates(dm, c, &cell));
      PetscCall(computeCellJacobian(&cell));
      if (PetscAbsReal(cell.detJacobian) > PETSC_SMALL) {
        *neighborCell = c;
        break;
      }
    }
  }
  PetscCall(DMPlexRestoreTransitiveClosure(dm, cellId, PETSC_TRUE, &closureSize, &closure));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Adds receiver row ridx (basis and curl at point in cellId) to every matrix of Q. */
static PetscErrorCode addReceiverRow(DM dm, const Grid *grid, PetscInt ridx, const PetscReal point[NUM_DIMENSIONS], const Cell *cell,
                                     PetscInt cellId, PetscReal **Ni, PetscReal **NiCurl, ReceiverInterpolationMatrices *Q) {
  PetscFunctionBeginUser;
  PetscSection section, globalSection;
  PetscReal    XiEtaZeta[NUM_DIMENSIONS];
  PetscInt     numLocal, numGlobal, *localIdx, *globalIdx;

  PetscCall(tetrahedronXyzToReference(cell->coordinates, point, XiEtaZeta));
  PetscCall(evaluateNedelecBasis(&grid->fem, cell, XiEtaZeta, Ni, NiCurl));

  /* Local indices flag constrained DOFs (< 0); global indices give the columns, in the same closure order */
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &globalSection));
  PetscCall(DMPlexGetClosureIndices(dm, section, section, cellId, PETSC_TRUE, &numLocal, &localIdx, NULL, NULL));
  PetscCall(DMPlexGetClosureIndices(dm, section, globalSection, cellId, PETSC_TRUE, &numGlobal, &globalIdx, NULL, NULL));
  for (PetscInt j = 0; j < grid->fem.numDofInCell; j++) {
    const PetscInt gidx = decodeGlobalDof(globalIdx[j]);
    if (localIdx[j] < 0 || gidx >= Q->numDof) continue;
    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
      PetscCall(MatSetValue(Q->mat[RECV_EX + d], ridx, gidx, Ni[d][j], ADD_VALUES));
      PetscCall(MatSetValue(Q->mat[RECV_HX + d], ridx, gidx, NiCurl[d][j], ADD_VALUES));
    }
  }
  PetscCall(DMPlexRestoreClosureIndices(dm, section, globalSection, cellId, PETSC_TRUE, &numGlobal, &globalIdx, NULL, NULL));
  PetscCall(DMPlexRestoreClosureIndices(dm, section, section, cellId, PETSC_TRUE, &numLocal, &localIdx, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Builds the receiver-interpolation matrices Q->mat[RECV_EX..RECV_HZ].
 *
 * Each receiver row is assembled once, by the lowest rank that locates it, with
 * global column indices. A receiver in a degenerate cell (|detJ| <= PETSC_SMALL)
 * moves to a non-degenerate vertex neighbor, or is skipped with a warning.
 * The contract is documented in include/receiver_interp.h.
 */
PetscErrorCode buildReceiverInterpolationMatrices(Vec receivers, const DM dm, const Grid *grid, ReceiverInterpolationMatrices *Q) {
  PetscFunctionBeginUser;

  MPI_Comm           comm = PetscObjectComm((PetscObject)dm);
  PetscMPIInt        rank;
  PetscSF            receiverSf = NULL;
  PetscInt           size, m, numFound, numSkipped = 0, *ownerRank;
  PetscBool         *rowAssembled;
  const PetscSFNode *recvInCell;
  const PetscInt    *recvFound;
  const PetscScalar *coords;
  PetscReal        **Ni, **NiCurl;
  Vec                v;

  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(VecGetSize(receivers, &size));
  PetscCheck(size % NUM_DIMENSIONS == 0, comm, PETSC_ERR_ARG_SIZ, "Receiver vector size %" PetscInt_FMT " not divisible by %d", size,
             NUM_DIMENSIONS);
  Q->numReceivers = size / NUM_DIMENSIONS;

  PetscCall(DMLocatePoints(dm, receivers, DM_POINTLOCATION_REMOVE, &receiverSf));
  PetscCall(PetscSFGetGraph(receiverSf, NULL, &numFound, &recvFound, &recvInCell));
  PetscCall(PetscMalloc2(Q->numReceivers, &ownerRank, Q->numReceivers, &rowAssembled));
  PetscCall(PetscArrayzero(rowAssembled, Q->numReceivers));
  PetscCall(electReceiverOwners(comm, Q->numReceivers, numFound, recvFound, recvInCell, ownerRank));

  /* Q matrices: numReceivers x numDof, columns laid out like the solution */
  PetscCall(DMGetGlobalVector(dm, &v));
  PetscCall(VecGetSize(v, &Q->numDof));
  PetscCall(VecGetLocalSize(v, &m));
  PetscCall(DMRestoreGlobalVector(dm, &v));
  PetscCall(MatCreate(comm, &Q->mat[0]));
  PetscCall(MatSetSizes(Q->mat[0], PETSC_DECIDE, m, Q->numReceivers, Q->numDof));
  PetscCall(MatSetType(Q->mat[0], MATAIJ));
  PetscCall(MatSetUp(Q->mat[0]));
  for (PetscInt comp = 1; comp < NUM_RECEIVER_FIELDS; comp++) PetscCall(MatDuplicate(Q->mat[0], MAT_DO_NOT_COPY_VALUES, &Q->mat[comp]));

  PetscCall(createBasisValues(grid->fem.numDofInCell, &Ni));
  PetscCall(createBasisValues(grid->fem.numDofInCell, &NiCurl));
  PetscCall(VecGetArrayRead(receivers, &coords));

  for (PetscInt i = 0; i < numFound; i++) {
    const PetscInt ridx   = recvFound ? recvFound[i] : i;
    PetscInt       cellId = recvInCell[i].index;
    PetscReal      point[NUM_DIMENSIONS];
    Cell           cell;

    /* A rank may locate the same on-interface receiver in two local cells */
    if (cellId < 0 || rank != ownerRank[ridx] || rowAssembled[ridx]) continue;
    rowAssembled[ridx] = PETSC_TRUE;
    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) point[d] = PetscRealPart(coords[NUM_DIMENSIONS * ridx + d]);

    PetscCall(extractCellCoordinates(dm, cellId, &cell));
    PetscCall(computeCellJacobian(&cell));
    if (PetscAbsReal(cell.detJacobian) <= PETSC_SMALL) {
      PetscInt neighbor;
      PetscCall(findNonDegenerateNeighbor(dm, grid, cellId, &neighbor));
      if (neighbor < 0) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,
                              "   WARNING: receiver %" PetscInt_FMT " in degenerate cell %" PetscInt_FMT
                              " (|detJ| = %.2e, coords = [%.4e, %.4e, %.4e]); no valid neighbor found, skipping.\n",
                              ridx, cellId, (double)PetscAbsReal(cell.detJacobian), (double)point[0], (double)point[1], (double)point[2]));
        numSkipped++;
        continue;
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF,
                            "   WARNING: receiver %" PetscInt_FMT " in degenerate cell %" PetscInt_FMT
                            " (|detJ| = %.2e); relocated to cell %" PetscInt_FMT ".\n",
                            ridx, cellId, (double)PetscAbsReal(cell.detJacobian), neighbor));
      cellId = neighbor;
      PetscCall(extractCellCoordinates(dm, cellId, &cell));
      PetscCall(computeCellJacobian(&cell));
    }
    PetscCall(addReceiverRow(dm, grid, ridx, point, &cell, cellId, Ni, NiCurl, Q));
  }
  PetscCall(VecRestoreArrayRead(receivers, &coords));

  if (numSkipped > 0) {
    PetscCall(PetscPrintf(comm, "\n   WARNING: %s receiver(s) skipped due to degenerate cells.\n", formatGroupedInt(numSkipped)));
  }

  for (PetscInt comp = 0; comp < NUM_RECEIVER_FIELDS; comp++) PetscCall(MatAssemblyBegin(Q->mat[comp], MAT_FINAL_ASSEMBLY));
  for (PetscInt comp = 0; comp < NUM_RECEIVER_FIELDS; comp++) PetscCall(MatAssemblyEnd(Q->mat[comp], MAT_FINAL_ASSEMBLY));

  PetscCall(destroyBasisValues(&Ni));
  PetscCall(destroyBasisValues(&NiCurl));
  PetscCall(PetscSFDestroy(&receiverSf));
  PetscCall(PetscFree2(ownerRank, rowAssembled));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Destroys the matrices owned by a ReceiverInterpolationMatrices struct.
 */
PetscErrorCode destroyReceiverInterpolationMatrices(ReceiverInterpolationMatrices *Q) {
  PetscFunctionBeginUser;
  for (PetscInt comp = 0; comp < NUM_RECEIVER_FIELDS; comp++) PetscCall(MatDestroy(&Q->mat[comp]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Creates the receiver field Vecs, named Ex, Ey, Ez, Hx, Hy, Hz.
 */
PetscErrorCode createReceiverFields(const ReceiverInterpolationMatrices *Q, Vec fields[NUM_RECEIVER_FIELDS]) {
  PetscFunctionBeginUser;
  const char *names[NUM_RECEIVER_FIELDS] = {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"};
  for (PetscInt comp = 0; comp < NUM_RECEIVER_FIELDS; comp++) {
    PetscCall(MatCreateVecs(Q->mat[comp], NULL, &fields[comp]));
    PetscCall(PetscObjectSetName((PetscObject)fields[comp], names[comp]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Interpolates E and H = curl(E)/(iωμ) at the receivers for one solution.
 */
PetscErrorCode interpolateReceiverFields(const ReceiverInterpolationMatrices *Q, Vec x, PetscScalar constFactor,
                                         Vec fields[NUM_RECEIVER_FIELDS]) {
  PetscFunctionBeginUser;
  for (PetscInt comp = 0; comp < NUM_RECEIVER_FIELDS; comp++) PetscCall(MatMult(Q->mat[comp], x, fields[comp]));
  for (PetscInt comp = RECV_HX; comp <= RECV_HZ; comp++) PetscCall(VecScale(fields[comp], 1.0 / constFactor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Destroys the Vecs created by createReceiverFields().
 *
 * @param[in,out] fields  Field Vecs.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode destroyReceiverFields(Vec fields[NUM_RECEIVER_FIELDS]) {
  PetscFunctionBeginUser;
  for (PetscInt comp = 0; comp < NUM_RECEIVER_FIELDS; comp++) PetscCall(VecDestroy(&fields[comp]));
  PetscFunctionReturn(PETSC_SUCCESS);
}
