/*
 * Filename: inversion.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * CSEM inverse kernel based on L-BFGS optimization (Nocedal 1980)
 * with an adjoint-state gradient and graph (Jacobi) gradient smoothing.
 */

/*
 * Notes:
 * Inversion parameterization: log-resistivity perturbation X = log(rho) - X0
 * Objective: ||W(dObs - Ex)||^2 / Ndata + lambda*||X||^2/N
 * Gradient:  adjoint method (forward + adjoint solve per frequency)
 */

/* PETSc libraries */
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>

/* PETGEM headers */
#include "common.h"
#include "assembly.h"
#include "constants.h"
#include "grid.h"
#include "fem.h"
#include "io.h"
#include "inversion.h"
#include "inversion_internal.h"
#include "lbfgs.h"
#include "solver.h"
#include "csem.h"

/* Adds the wall time since t0 to *acc. */
static PetscErrorCode addElapsed(PetscLogDouble t0, PetscLogDouble *acc) {
  PetscFunctionBeginUser;
  PetscLogDouble now;
  PetscCall(PetscTime(&now));
  *acc += now - t0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Creates the KSP that serves both the forward and the adjoint solve.
 *
 * Defaults to FGMRES + PCBDDC with the discrete gradient G; the options
 * database can select a direct factorization when A is AIJ.
 *
 * @param[in]  order  Basis order.
 * @param[in]  A      Operator (MATIS for PCBDDC, AIJ for a direct solve).
 * @param[in]  G      Discrete gradient for PCBDDC.
 * @param[out] ksp    KSP bound to A.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode createFrequencyKsp(PetscInt order, Mat A, Mat G, KSP *ksp) {
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)A), ksp));
  PetscCall(KSPSetOperators(*ksp, A, A));
  PetscCall(KSPSetType(*ksp, KSPFGMRES));
  PetscCall(setupBddcFromPetgemGradient(*ksp, A, G, order, NULL));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Source RHS, observed row k of dObs (row-major numFreqs x numReceivers) and data weights of one frequency. */
static PetscErrorCode setupFrequencyData(InversionContext *ctx, const PetscScalar *dObs, PetscInt k, InversionFrequency *f) {
  PetscFunctionBeginUser;
  const ImCsemSource *isrc         = &ctx->imParams->imSources[k];
  const PetscInt      numReceivers = ctx->Q->numReceivers;
  const PetscReal     errorLevel   = ctx->imParams->errorLevel;
  const PetscScalar  *obs          = &dObs[k * numReceivers];
  CsemSource          src          = isrc->dipole;
  CsemSourceSet       one          = {isrc->freq, 1, &src};
  PetscScalar        *d, *w;
  Mat                 B;
  Vec                 col;

  f->iwmu = iOmegaMu(isrc->freq);

  PetscCall(assembleCsemRhs(one, ctx->dm, &ctx->grid, f->iwmu, &B));
  PetscCall(DMCreateGlobalVector(ctx->dm, &f->B));
  PetscCall(MatDenseGetColumnVecRead(B, 0, &col));
  PetscCall(VecCopy(col, f->B));
  PetscCall(MatDenseRestoreColumnVecRead(B, 0, &col));
  PetscCall(MatDestroy(&B));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, numReceivers, &f->dObs));
  PetscCall(VecDuplicate(f->dObs, &f->W));
  PetscCall(VecGetArray(f->dObs, &d));
  PetscCall(VecGetArray(f->W, &w));
  for (PetscInt r = 0; r < numReceivers; r++) {
    const PetscReal absVal = PetscAbsScalar(obs[r]);
    d[r]                   = obs[r];
    w[r]                   = (absVal > 0.0) ? 1.0 / (absVal * errorLevel) : 0.0;
  }
  PetscCall(VecRestoreArray(f->W, &w));
  PetscCall(VecRestoreArray(f->dObs, &d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Allocates the iterate-independent workspace of the inversion.
 *
 * Quadrature and elemental buffers, solution Vecs, K and G (assembled once;
 * K is the fused operator with constFactor = 0), the Ms pattern, and for each
 * frequency its RHS, observed data, weights, operator and KSP.
 *
 * @param[in,out] ctx   Inversion context.
 * @param[in]     dObs  Observed Ex, row-major numFreqs x numReceivers.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode setupInversionWorkspace(InversionContext *ctx, const PetscScalar *dObs) {
  PetscFunctionBeginUser;
  const PetscInt numDof = ctx->grid.fem.numDofInCell;
  char           matType[PETSC_MAX_PATH_LEN];
  PetscBool      matTypeSet;

  PetscCall(createQuadrature3D(ctx->grid.fem.order, &ctx->quad3d));
  PetscCall(createDenseRows(numDof, numDof, &ctx->Me));
  PetscCall(createDenseRows(numDof, numDof, &ctx->Ke));

  PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->x));
  PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->nB));
  PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->nx));
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)ctx->dm), PETSC_DECIDE, ctx->Q->numReceivers, &ctx->ExRecv));
  PetscCall(VecDuplicate(ctx->ExRecv, &ctx->adjSrc));

  /* MATIS (PCBDDC) unless the parameter file chose another operator type */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_mat_type", matType, sizeof(matType), &matTypeSet));
  if (!matTypeSet) PetscCall(DMSetMatType(ctx->dm, MATIS));
  PetscCall(assembleMaxwellOperator(ctx->dm, &ctx->grid, ctx->conductivity, 0.0, &ctx->K, NULL, &ctx->G));
  PetscCall(MatDuplicate(ctx->K, MAT_DO_NOT_COPY_VALUES, &ctx->Ms));

  ctx->numFreqs = ctx->imParams->numFreqs;
  PetscCall(PetscCalloc1(ctx->numFreqs, &ctx->freq));
  for (PetscInt k = 0; k < ctx->numFreqs; k++) {
    InversionFrequency *f = &ctx->freq[k];
    PetscCall(setupFrequencyData(ctx, dObs, k, f));
    PetscCall(MatDuplicate(ctx->K, MAT_DO_NOT_COPY_VALUES, &f->A));
    PetscCall(createFrequencyKsp(ctx->grid.fem.order, f->A, ctx->G, &f->ksp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroyInversionWorkspace(InversionContext *ctx) {
  PetscFunctionBeginUser;
  PetscCall(destroyQuadrature3D(&ctx->quad3d));
  PetscCall(destroyDenseRows(&ctx->Me));
  PetscCall(destroyDenseRows(&ctx->Ke));
  PetscCall(VecDestroy(&ctx->x));
  PetscCall(VecDestroy(&ctx->nB));
  PetscCall(VecDestroy(&ctx->nx));
  PetscCall(VecDestroy(&ctx->ExRecv));
  PetscCall(VecDestroy(&ctx->adjSrc));
  for (PetscInt k = 0; k < ctx->numFreqs; k++) {
    PetscCall(VecDestroy(&ctx->freq[k].B));
    PetscCall(VecDestroy(&ctx->freq[k].dObs));
    PetscCall(VecDestroy(&ctx->freq[k].W));
    PetscCall(MatDestroy(&ctx->freq[k].A));
    PetscCall(KSPDestroy(&ctx->freq[k].ksp));
  }
  PetscCall(PetscFree(ctx->freq));
  ctx->numFreqs = 0;
  PetscCall(MatDestroy(&ctx->K));
  PetscCall(MatDestroy(&ctx->Ms));
  PetscCall(MatDestroy(&ctx->G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves with ksp and records its time, iterations and failure reason. */
static PetscErrorCode solve(InversionContext *c, KSP ksp, Vec rhs, Vec sol) {
  PetscFunctionBeginUser;
  PetscLogDouble     t0;
  PetscInt           its;
  KSPConvergedReason reason;

  PetscCall(PetscTime(&t0));
  PetscCall(KSPSolve(ksp, rhs, sol));
  PetscCall(addElapsed(t0, &c->tSolver));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  c->numSolves++;
  if (reason < 0 && c->failReason == 0) c->failReason = reason;
  c->kspItsMinRow = PetscMin(c->kspItsMinRow, its);
  c->kspItsMaxRow = PetscMax(c->kspItsMaxRow, its);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Adds the adjoint gradient of one frequency to c->DfDm.
 *
 * Per cell: DfDm[e] += Re( sum_j (sum_k -2 iwmu Me[j][k] x_e[k]) nx_e[j] ),
 * with Me the unit-conductivity mass matrix.
 *
 * @param[in,out] c        Inversion context.
 * @param[in]     xLocal   Forward solution (local).
 * @param[in]     nxLocal  Adjoint solution (local).
 * @param[in]     iwmu     i omega mu.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode accumulateGradient(InversionContext *c, Vec xLocal, Vec nxLocal, PetscScalar iwmu) {
  PetscFunctionBeginUser;
  const PetscInt n = c->grid.fem.numDofInCell;
  PetscSection   section;

  PetscCall(DMGetLocalSection(c->dm, &section));
  for (PetscInt i = c->grid.cellStart; i < c->grid.cellEnd; i++) {
    Cell         cell;
    PetscInt     closureSize = n;
    PetscScalar *x_e = NULL, *nx_e = NULL, contrib = 0.0, grad;

    PetscCall(extractCellCoordinates(c->dm, i, &cell));
    cell.conductivity[0] = cell.conductivity[1] = cell.conductivity[2] = 1.0;
    PetscCall(computeElementalMatrices(&c->grid.fem, &cell, &c->quad3d, c->Me, c->Ke));

    PetscCall(DMPlexVecGetClosure(c->dm, section, xLocal, i, &closureSize, &x_e));
    PetscCall(DMPlexVecGetClosure(c->dm, section, nxLocal, i, &closureSize, &nx_e));
    for (PetscInt j = 0; j < n; j++) {
      PetscScalar iGj = 0.0;
      for (PetscInt k = 0; k < n; k++) iGj += (-2.0 * iwmu * c->Me[j][k]) * x_e[k];
      contrib += iGj * nx_e[j];
    }
    PetscCall(DMPlexVecRestoreClosure(c->dm, section, xLocal, i, &closureSize, &x_e));
    PetscCall(DMPlexVecRestoreClosure(c->dm, section, nxLocal, i, &closureSize, &nx_e));

    grad = PetscRealPart(contrib);
    PetscCall(DMPlexVecSetClosure(c->dmInversion, NULL, c->DfDm, i, &grad, ADD_VALUES));
  }
  PetscCall(VecAssemblyBegin(c->DfDm));
  PetscCall(VecAssemblyEnd(c->DfDm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Sum over receivers of |W (dObs - Ex)|^2; c->adjSrc = W^2 conj(dObs - Ex) on the owned receivers. */
static PetscErrorCode weightedMisfit(InversionContext *c, const InversionFrequency *f, PetscReal *misfit) {
  PetscFunctionBeginUser;
  const PetscScalar *ex, *d, *w;
  PetscScalar       *a;
  PetscInt           start, end;
  PetscReal          local = 0.0;

  PetscCall(VecGetOwnershipRange(c->ExRecv, &start, &end));
  PetscCall(VecGetArrayRead(c->ExRecv, &ex));
  PetscCall(VecGetArrayRead(f->dObs, &d));
  PetscCall(VecGetArrayRead(f->W, &w));
  PetscCall(VecGetArray(c->adjSrc, &a));
  for (PetscInt r = 0; r < end - start; r++) {
    const PetscInt    rGlobal = start + r; /* global receiver index into the sequential dObs and W */
    const PetscScalar resid   = d[rGlobal] - ex[r];
    const PetscReal   wr      = PetscRealPart(w[rGlobal]);
    const PetscScalar dtD     = wr * resid;
    local += PetscRealPart(dtD * PetscConj(dtD));
    a[r] = wr * wr * PetscConj(resid);
  }
  PetscCall(VecRestoreArray(c->adjSrc, &a));
  PetscCall(VecRestoreArrayRead(f->W, &w));
  PetscCall(VecRestoreArrayRead(f->dObs, &d));
  PetscCall(VecRestoreArrayRead(c->ExRecv, &ex));
  PetscCallMPI(MPI_Allreduce(&local, misfit, 1, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)c->dm)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* One frequency: A_f = K - iwmu Ms, forward solve, misfit, adjoint solve, gradient contribution. */
static PetscErrorCode evaluateFrequency(InversionContext *c, InversionFrequency *f, PetscReal *misfit) {
  PetscFunctionBeginUser;
  const Mat      QEx = c->Q->mat[RECV_EX];
  PetscLogDouble t0;
  Vec            xLocal, nxLocal;

  PetscCall(PetscTime(&t0));
  PetscCall(MatCopy(c->K, f->A, SAME_NONZERO_PATTERN));
  PetscCall(MatAXPY(f->A, -f->iwmu, c->Ms, SAME_NONZERO_PATTERN));
  PetscCall(addElapsed(t0, &c->tAssembly));

  PetscCall(solve(c, f->ksp, f->B, c->x));
  PetscCall(MatMult(QEx, c->x, c->ExRecv));
  PetscCall(weightedMisfit(c, f, misfit));
  PetscCall(MatMultTranspose(QEx, c->adjSrc, c->nB));
  PetscCall(solve(c, f->ksp, c->nB, c->nx));

  PetscCall(DMGetLocalVector(c->dm, &xLocal));
  PetscCall(DMGetLocalVector(c->dm, &nxLocal));
  PetscCall(DMGlobalToLocal(c->dm, c->x, INSERT_VALUES, xLocal));
  PetscCall(DMGlobalToLocal(c->dm, c->nx, INSERT_VALUES, nxLocal));
  PetscCall(accumulateGradient(c, xLocal, nxLocal, f->iwmu));
  PetscCall(DMRestoreLocalVector(c->dm, &xLocal));
  PetscCall(DMRestoreLocalVector(c->dm, &nxLocal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* G = mask( scatter( smooth( mask( -sigma DfDm ) ) / numData ) + 2 lambda/N X ). */
static PetscErrorCode assembleGradient(InversionContext *c, Vec X, PetscInt numData, PetscInt numCells, Vec G) {
  PetscFunctionBeginUser;
  PetscSection       sec;
  PetscScalar       *df;
  const PetscScalar *sigma;

  /* Chain rule d/dlog(rho) = -sigma d/dsigma (component 0 = sigma_x) */
  PetscCall(DMGetLocalSection(c->dmConductivity, &sec));
  PetscCall(VecGetArray(c->DfDm, &df));
  PetscCall(VecGetArrayRead(c->conductivity, &sigma));
  for (PetscInt i = c->grid.cellStart; i < c->grid.cellEnd; i++) {
    const PetscInt li = i - c->grid.cellStart;
    PetscInt       off;
    PetscCall(PetscSectionGetOffset(sec, i, &off));
    df[li] = -df[li] * sigma[off];
  }
  PetscCall(VecRestoreArrayRead(c->conductivity, &sigma));
  PetscCall(VecRestoreArray(c->DfDm, &df));

  PetscCall(VecPointwiseMult(c->DfDm, c->DfDm, c->notFixedMaskLocal));
  PetscCall(applyGraphSmoothing(c->graph, c->imParams->diagGradientWeight, c->DfDm));
  PetscCall(VecScale(c->DfDm, 1.0 / (PetscReal)numData));
  PetscCall(VecZeroEntries(G));
  PetscCall(DMLocalToGlobal(c->dmInversion, c->DfDm, ADD_VALUES, G));
  PetscCall(VecAXPY(G, 2.0 * c->imParams->lambda / (PetscReal)numCells, X));
  PetscCall(VecPointwiseMult(G, G, c->notFixedMaskGlobal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Objective, gradient and data RMS at X (LbfgsProblem.objgrad).
 *
 * F = sum_f |W (dObs - Ex)|^2 / Ndata + lambda ||X||^2 / N, with
 * Ndata = 2 numReceivers numFreqs (real and imaginary parts).
 *
 * @param[in]  ctx   InversionContext.
 * @param[in]  X     Log-perturbation iterate.
 * @param[out] F     Objective.
 * @param[out] G     Gradient.
 * @param[out] rms   Data RMS.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode inversionObjGrad(void *ctx, Vec X, PetscReal *F, Vec G, PetscReal *rms) {
  PetscFunctionBeginUser;
  InversionContext *c       = (InversionContext *)ctx;
  const PetscInt    numData = c->Q->numReceivers * c->numFreqs * 2;
  PetscReal         misfit  = 0.0, regTerm;
  PetscScalar       xDot;
  PetscInt          numCells;
  PetscLogDouble    t0;

  c->numEvals++;

  PetscCall(applyLogToSigma(c->dmInversion, c->dmConductivity, X, c->X0, c->conductivity, &c->grid, c->graph));
  PetscCall(PetscTime(&t0));
  PetscCall(refillMassMatrix(c->dm, &c->grid, c->conductivity, &c->quad3d, c->Me, c->Ke, c->Ms));
  PetscCall(addElapsed(t0, &c->tAssembly));

  PetscCall(VecZeroEntries(c->DfDm));
  for (PetscInt k = 0; k < c->numFreqs; k++) {
    PetscReal m;
    PetscCall(evaluateFrequency(c, &c->freq[k], &m));
    misfit += m;
  }

  PetscCall(VecDot(X, X, &xDot));
  PetscCall(VecGetSize(X, &numCells));
  regTerm = c->imParams->lambda * PetscRealPart(xDot) / (PetscReal)numCells;
  *F      = misfit / (PetscReal)numData + regTerm;
  *rms    = PetscSqrtReal(misfit / (PetscReal)numData);
  PetscCall(assembleGradient(c, X, numData, numCells, G));

  c->allRms[c->numEvals - 1] = *rms;
  c->lastRms                 = *rms;
  c->lastRegTerm             = regTerm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief L-BFGS monitor: VTU snapshot on accepted steps and one table row.
 *
 * Columns: iteration, evaluations, RMS, the Tikhonov term, ||g||/max(1,||X||),
 * the accepted step, the min-max KSP iterations and the wall time since the
 * previous row.
 *
 * @param[in] ctx        InversionContext.
 * @param[in] iter       Iteration (0 = initial model).
 * @param[in] gradRatio  ||g|| / max(1, ||X||).
 * @param[in] step       Accepted step length (unused on row 0).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode inversionMonitor(void *ctx, PetscInt iter, PetscReal gradRatio, PetscReal step) {
  PetscFunctionBeginUser;
  InversionContext *c        = (InversionContext *)ctx;
  MPI_Comm          comm     = PetscObjectComm((PetscObject)c->dm);
  const PetscInt    interval = c->imParams->snapshotInterval;
  PetscLogDouble    now;
  char              its[32], stp[16];

  if (iter == 0) {
    PetscCall(
        PetscPrintf(comm, "   %4s %5s %9s %12s %10s %8s %9s %8s\n", "Iter", "Evals", "RMS", "λΦm", "|g|/|x|", "Step", "KSP its", "t (s)"));
  } else {
    c->acceptedIter++;
    if (interval > 0 && c->acceptedIter % interval == 0) {
      PetscLogDouble t0;
      PetscCall(PetscTime(&t0));
      PetscCall(writeInversionSnapshotVtu(c, c->acceptedIter));
      PetscCall(addElapsed(t0, &c->tOutput));
      c->numSnapshots++;
    }
  }

  if (c->kspItsMaxRow == 0 && c->kspItsMinRow == PETSC_INT_MAX) {
    PetscCall(PetscStrncpy(its, "-", sizeof(its)));
  } else if (c->kspItsMinRow == c->kspItsMaxRow) {
    PetscCall(PetscSNPrintf(its, sizeof(its), "%" PetscInt_FMT, c->kspItsMaxRow));
  } else {
    PetscCall(PetscSNPrintf(its, sizeof(its), "%" PetscInt_FMT "-%" PetscInt_FMT, c->kspItsMinRow, c->kspItsMaxRow));
  }
  if (iter == 0)
    PetscCall(PetscStrncpy(stp, "-", sizeof(stp)));
  else
    snprintf(stp, sizeof(stp), "%.3g", (double)step);

  PetscCall(PetscTime(&now));
  PetscCall(PetscPrintf(comm, "   %4" PetscInt_FMT " %5" PetscInt_FMT " %9.4f %10.3e %10.3e %8s %9s %8.1f\n", iter, c->numEvals,
                        (double)c->lastRms, (double)c->lastRegTerm, (double)gradRatio, stp, its, (double)(now - c->tRow)));
  c->kspItsMinRow = PETSC_INT_MAX;
  c->kspItsMaxRow = 0;
  c->tRow         = now;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* X0 = log(1 / sigma_x) per cell. */
static PetscErrorCode initialLogResistivity(DM dmInversion, DM dmConductivity, Vec conductivity, const Grid *grid, Vec X0) {
  PetscFunctionBeginUser;
  PetscSection       sec;
  const PetscScalar *sigma;
  PetscScalar       *x0;
  Vec                local;

  PetscCall(DMCreateLocalVector(dmInversion, &local));
  PetscCall(DMGetLocalSection(dmConductivity, &sec));
  PetscCall(VecGetArrayRead(conductivity, &sigma));
  PetscCall(VecGetArray(local, &x0));
  for (PetscInt i = grid->cellStart; i < grid->cellEnd; i++) {
    PetscInt off;
    PetscCall(PetscSectionGetOffset(sec, i, &off));
    x0[i - grid->cellStart] = PetscLogReal(1.0 / PetscRealPart(sigma[off]));
  }
  PetscCall(VecRestoreArray(local, &x0));
  PetscCall(VecRestoreArrayRead(conductivity, &sigma));
  PetscCall(DMLocalToGlobal(dmInversion, local, INSERT_VALUES, X0));
  PetscCall(VecDestroy(&local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode logInversionAssembly(const InversionContext *ctx, InversionStats *stats) {
  PetscFunctionBeginUser;
  MPI_Comm comm = PetscObjectComm((PetscObject)ctx->dm);
  MatType  mtype;
  PetscCall(MatGetType(ctx->freq[0].A, &mtype));
  PetscCall(getSolverName(ctx->freq[0].ksp, stats->solver, sizeof(stats->solver)));
  PetscCall(logSection(comm, "Assembly"));
  PetscCall(logKvStr(comm, "Matrix type", mtype));
  PetscCall(logKvInt(comm, "Right-hand sides", ctx->numFreqs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Top-level inversion driver.
 *
 * Builds the smoothing graph, receiver operators, inversion DM and workspace,
 * runs L-BFGS from the starting model and writes the HDF5 results.
 */
PetscErrorCode runCsemInversion(const ImParams *imParams, const DM dm, const Grid *grid, Vec conductivity, Vec materialsId, Vec receivers,
                                InversionStats *stats) {
  PetscFunctionBeginUser;
  PetscLogDouble                t0, t1, t2, t3;
  NeighborGraph                 graph;
  ReceiverInterpolationMatrices Q;
  PetscScalar                  *dObs;
  DM                            dmConductivity, dmInversion;
  Vec                           X0, X, DfDm, maskGlobal, maskLocal;
  PetscReal                    *allRms;
  LbfgsResult                   result;

  PetscCall(PetscTime(&t0));
  PetscCall(buildNeighborSmoothingGraph(dm, grid, imParams, materialsId, &graph));
  PetscCall(buildReceiverInterpolationMatrices(receivers, dm, grid, &Q));
  PetscCall(loadObservedDataset(imParams, Q.numReceivers, &dObs));
  PetscCall(VecGetDM(conductivity, &dmConductivity));
  PetscCall(createCellScalarDm(dmConductivity, &dmInversion));

  PetscCall(DMCreateGlobalVector(dmInversion, &X0));
  PetscCall(DMCreateGlobalVector(dmInversion, &X));
  PetscCall(DMCreateLocalVector(dmInversion, &DfDm));
  PetscCall(initialLogResistivity(dmInversion, dmConductivity, conductivity, grid, X0));
  PetscCall(VecZeroEntries(X));
  PetscCall(buildNotFixedMask(&graph, dmInversion, grid, &maskGlobal, &maskLocal));
  /* One RMS per evaluation: the initial one plus at most LBFGS_MAX_LINE_SEARCH per iteration */
  PetscCall(PetscCalloc1(imParams->maxIter * LBFGS_MAX_LINE_SEARCH + 1, &allRms));

  InversionContext ctx = {
      .imParams           = imParams,
      .dm                 = dm,
      .dmConductivity     = dmConductivity,
      .dmInversion        = dmInversion,
      .grid               = *grid,
      .conductivity       = conductivity,
      .X0                 = X0,
      .DfDm               = DfDm,
      .Q                  = &Q,
      .graph              = &graph,
      .notFixedMaskGlobal = maskGlobal,
      .notFixedMaskLocal  = maskLocal,
      .allRms             = allRms,
      .lastRms            = PETSC_INFINITY,
      .kspItsMinRow       = PETSC_INT_MAX,
  };
  PetscCall(setupInversionWorkspace(&ctx, dObs));
  PetscCall(PetscFree(dObs));
  PetscCall(logInversionAssembly(&ctx, stats));

  {
    const LbfgsProblem problem = {inversionObjGrad, inversionMonitor, &ctx};
    const LbfgsOptions opts    = {imParams->lbfgsMemory, imParams->maxIter,   imParams->gtol,
                                  imParams->rmsTol,      imParams->rmsRelTol, imParams->rmsStallWindow};
    PetscCall(logSection(PetscObjectComm((PetscObject)dm), "L-BFGS optimization"));
    PetscCall(PetscTime(&t1));
    ctx.tRow = t1;
    PetscCall(lbfgsOptimize(&problem, &opts, X, &result));
    PetscCall(PetscTime(&t2));
  }
  const PetscLogDouble tOutputLoop = ctx.tOutput;

  /* Model of the last accepted X (a failed line search leaves the rejected trial in conductivity) */
  PetscCall(applyLogToSigma(dmInversion, dmConductivity, X, X0, conductivity, grid, &graph));
  if (imParams->snapshotInterval > 0 && (ctx.acceptedIter == 0 || ctx.acceptedIter % imParams->snapshotInterval != 0)) {
    PetscCall(writeInversionSnapshotVtu(&ctx, ctx.acceptedIter));
    ctx.numSnapshots++;
  }
  PetscCall(writeInversionResults(imParams, dmConductivity, conductivity, X, allRms, result.numIters, result.numEvals,
                                  LbfgsReasons[result.reason]));
  PetscCall(PetscTime(&t3));

  stats->tSetup       = t1 - t0;
  stats->tAssembly    = ctx.tAssembly;
  stats->tSolver      = ctx.tSolver;
  stats->tGradient    = PetscMax(0.0, (t2 - t1) - ctx.tAssembly - ctx.tSolver - tOutputLoop);
  stats->tOutput      = tOutputLoop + (t3 - t2);
  stats->numSolves    = ctx.numSolves;
  stats->failReason   = ctx.failReason;
  stats->numSnapshots = ctx.numSnapshots;

  PetscCall(destroyInversionWorkspace(&ctx));
  PetscCall(VecDestroy(&X0));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&DfDm));
  PetscCall(VecDestroy(&maskGlobal));
  PetscCall(VecDestroy(&maskLocal));
  PetscCall(DMDestroy(&dmInversion));
  PetscCall(PetscFree(allRms));
  PetscCall(destroyNeighborGraph(&graph));
  PetscCall(destroyReceiverInterpolationMatrices(&Q));
  PetscFunctionReturn(PETSC_SUCCESS);
}
