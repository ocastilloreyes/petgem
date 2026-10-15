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
 * Objective: ||W(dObs - d)||^2 / Ndata + lambda*||X||^2/N, d the fitted components (Ex..Hz, Zij = Ei/Hj)
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

/* Source RHS, observed row k of every component of dObs and data weights of one frequency; NaN data get weight 0. */
static PetscErrorCode setupFrequencyData(InversionContext *ctx, const PetscScalar *dObs, PetscInt k, InversionFrequency *f) {
  PetscFunctionBeginUser;
  const ImCsemSource *isrc         = &ctx->imParams->imSources[k];
  const PetscInt      numReceivers = ctx->Q->numReceivers;
  const PetscReal     errorLevel   = ctx->imParams->errorLevel;
  CsemSource          src          = isrc->dipole;
  CsemSourceSet       one          = {isrc->freq, 1, &src};
  Mat                 B;
  Vec                 col;

  f->iwmu = iOmegaMu(isrc->freq);

  PetscCall(assembleCsemRhs(one, ctx->dm, &ctx->grid, f->iwmu, &B));
  PetscCall(DMCreateGlobalVector(ctx->dm, &f->B));
  PetscCall(MatDenseGetColumnVecRead(B, 0, &col));
  PetscCall(VecCopy(col, f->B));
  PetscCall(MatDenseRestoreColumnVecRead(B, 0, &col));
  PetscCall(MatDestroy(&B));

  for (PetscInt i = 0; i < ctx->imParams->numComponents; i++) {
    const PetscScalar *obs = &dObs[(i * ctx->numFreqs + k) * numReceivers];
    PetscScalar       *d, *w;

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, numReceivers, &f->dObs[i]));
    PetscCall(VecDuplicate(f->dObs[i], &f->W[i]));
    PetscCall(VecGetArray(f->dObs[i], &d));
    PetscCall(VecGetArray(f->W[i], &w));
    for (PetscInt r = 0; r < numReceivers; r++) {
      const PetscBool missing = (PetscBool)(PetscIsNanReal(PetscRealPart(obs[r])) || PetscIsNanReal(PetscImaginaryPart(obs[r])));
      const PetscReal absVal  = missing ? 0.0 : PetscAbsScalar(obs[r]);
      d[r]                    = missing ? 0.0 : obs[r];
      w[r]                    = (absVal > 0.0) ? 1.0 / (absVal * errorLevel) : 0.0;
      if (!missing) ctx->numData += 2;
    }
    PetscCall(VecRestoreArray(f->W[i], &w));
    PetscCall(VecRestoreArray(f->dObs[i], &d));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Allocates the iterate-independent workspace of the inversion.
 *
 * Quadrature and elemental buffers, solution Vecs, K and G (assembled once;
 * K is the fused operator with constFactor = 0), the Ms pattern, the operator
 * and KSP shared by the frequencies, and for each frequency its RHS, observed
 * data and weights.
 *
 * @param[in,out] ctx   Inversion context.
 * @param[in]     dObs  Observed data, component-major numComponents x numFreqs x numReceivers.
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
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)ctx->dm), PETSC_DECIDE, ctx->Q->numReceivers, &ctx->dPre));
  for (PetscInt i = 0; i < ctx->imParams->numComponents; i++) {
    ReceiverField fields[2];
    imDataComponentFields(ctx->imParams->components[i], &fields[0], &fields[1]);
    for (PetscInt j = 0; j < 2; j++) {
      if (fields[j] == NUM_RECEIVER_FIELDS || ctx->recv[fields[j]]) continue;
      PetscCall(VecDuplicate(ctx->dPre, &ctx->recv[fields[j]]));
      PetscCall(VecDuplicate(ctx->dPre, &ctx->adjSrc[fields[j]]));
    }
  }

  /* MATIS (PCBDDC) unless the parameter file chose another operator type */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_mat_type", matType, sizeof(matType), &matTypeSet));
  if (!matTypeSet) PetscCall(DMSetMatType(ctx->dm, MATIS));
  PetscCall(assembleMaxwellOperator(ctx->dm, &ctx->grid, ctx->conductivity, 0.0, &ctx->K, NULL, &ctx->G));
  PetscCall(MatDuplicate(ctx->K, MAT_DO_NOT_COPY_VALUES, &ctx->Ms));

  ctx->numFreqs = ctx->imParams->numFreqs;
  ctx->numData  = 0;
  PetscCall(PetscCalloc1(ctx->numFreqs, &ctx->freq));
  for (PetscInt k = 0; k < ctx->numFreqs; k++) {
    InversionFrequency *f = &ctx->freq[k];
    PetscCall(setupFrequencyData(ctx, dObs, k, f));
  }
  PetscCall(MatDuplicate(ctx->K, MAT_DO_NOT_COPY_VALUES, &ctx->A));
  PetscCall(createFrequencyKsp(ctx->grid.fem.order, ctx->A, ctx->G, &ctx->ksp));
  PetscCheck(ctx->numData > 0, PetscObjectComm((PetscObject)ctx->dm), PETSC_ERR_ARG_WRONG,
             "No observed data: every value is missing (NaN)");
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
  PetscCall(VecDestroy(&ctx->dPre));
  for (PetscInt f = 0; f < NUM_RECEIVER_FIELDS; f++) {
    PetscCall(VecDestroy(&ctx->recv[f]));
    PetscCall(VecDestroy(&ctx->adjSrc[f]));
  }
  for (PetscInt k = 0; k < ctx->numFreqs; k++) {
    PetscCall(VecDestroy(&ctx->freq[k].B));
    for (PetscInt i = 0; i < ctx->imParams->numComponents; i++) {
      PetscCall(VecDestroy(&ctx->freq[k].dObs[i]));
      PetscCall(VecDestroy(&ctx->freq[k].W[i]));
    }
  }
  PetscCall(MatDestroy(&ctx->A));
  PetscCall(KSPDestroy(&ctx->ksp));
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

/* Fields at the receivers for the solution c->x; H is divided by i omega mu. */
static PetscErrorCode interpolateFields(InversionContext *c, PetscScalar iwmu) {
  PetscFunctionBeginUser;
  for (PetscInt f = 0; f < NUM_RECEIVER_FIELDS; f++) {
    if (!c->recv[f]) continue;
    PetscCall(MatMult(c->Q->mat[f], c->x, c->recv[f]));
    if (f >= RECV_HX) PetscCall(VecScale(c->recv[f], 1.0 / iwmu));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* c->dPre = component i of the fitted data, from c->recv. */
static PetscErrorCode componentData(InversionContext *c, PetscInt i) {
  PetscFunctionBeginUser;
  ReceiverField      num, den;
  const PetscScalar *e, *h = NULL;
  PetscScalar       *p;
  PetscInt           n;

  imDataComponentFields(c->imParams->components[i], &num, &den);
  if (den == NUM_RECEIVER_FIELDS) {
    PetscCall(VecCopy(c->recv[num], c->dPre));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecGetLocalSize(c->dPre, &n));
  PetscCall(VecGetArrayRead(c->recv[num], &e));
  PetscCall(VecGetArrayRead(c->recv[den], &h));
  PetscCall(VecGetArray(c->dPre, &p));
  for (PetscInt r = 0; r < n; r++) p[r] = e[r] / h[r];
  PetscCall(VecRestoreArray(c->dPre, &p));
  PetscCall(VecRestoreArrayRead(c->recv[den], &h));
  PetscCall(VecRestoreArrayRead(c->recv[num], &e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Sum over receivers of |W (dObs - d)|^2 for component i, d = c->dPre. With adjoint, adds J^T a to c->adjSrc,
 * a = W^2 conj(dObs - d) and J the derivative of d with respect to the interpolated fields.
 */
static PetscErrorCode weightedMisfit(InversionContext *c, const InversionFrequency *f, PetscInt i, PetscBool adjoint, PetscReal *misfit) {
  PetscFunctionBeginUser;
  ReceiverField      num, den;
  const PetscScalar *pre, *d, *w, *e = NULL, *h = NULL;
  PetscScalar       *aNum = NULL, *aDen = NULL;
  PetscInt           start, end;
  PetscReal          local = 0.0;

  imDataComponentFields(c->imParams->components[i], &num, &den);
  PetscCall(VecGetOwnershipRange(c->dPre, &start, &end));
  PetscCall(VecGetArrayRead(c->dPre, &pre));
  PetscCall(VecGetArrayRead(f->dObs[i], &d));
  PetscCall(VecGetArrayRead(f->W[i], &w));
  if (adjoint) {
    PetscCall(VecGetArray(c->adjSrc[num], &aNum));
    if (den != NUM_RECEIVER_FIELDS) {
      PetscCall(VecGetArray(c->adjSrc[den], &aDen));
      PetscCall(VecGetArrayRead(c->recv[num], &e));
      PetscCall(VecGetArrayRead(c->recv[den], &h));
    }
  }
  for (PetscInt r = 0; r < end - start; r++) {
    const PetscInt    rGlobal = start + r; /* global receiver index into the sequential dObs and W */
    const PetscScalar resid   = d[rGlobal] - pre[r];
    const PetscReal   wr      = PetscRealPart(w[rGlobal]);
    const PetscScalar dtD     = wr * resid;
    local += PetscRealPart(dtD * PetscConj(dtD));
    if (!adjoint) continue;
    {
      const PetscScalar a = wr * wr * PetscConj(resid);
      if (den == NUM_RECEIVER_FIELDS) {
        aNum[r] += (num >= RECV_HX) ? a / f->iwmu : a;
      } else {
        aNum[r] += a / h[r];
        aDen[r] -= a * e[r] / (h[r] * h[r] * f->iwmu);
      }
    }
  }
  if (adjoint) {
    PetscCall(VecRestoreArray(c->adjSrc[num], &aNum));
    if (den != NUM_RECEIVER_FIELDS) {
      PetscCall(VecRestoreArrayRead(c->recv[den], &h));
      PetscCall(VecRestoreArrayRead(c->recv[num], &e));
      PetscCall(VecRestoreArray(c->adjSrc[den], &aDen));
    }
  }
  PetscCall(VecRestoreArrayRead(f->W[i], &w));
  PetscCall(VecRestoreArrayRead(f->dObs[i], &d));
  PetscCall(VecRestoreArrayRead(c->dPre, &pre));
  PetscCallMPI(MPI_Allreduce(&local, misfit, 1, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)c->dm)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* One frequency: A_f = K - iwmu Ms, forward solve, misfit, adjoint solve, gradient contribution. */
static PetscErrorCode evaluateFrequency(InversionContext *c, InversionFrequency *f, PetscReal *misfit) {
  PetscFunctionBeginUser;
  PetscLogDouble t0;
  PetscBool      first = PETSC_TRUE;
  Vec            xLocal, nxLocal;

  PetscCall(PetscTime(&t0));
  PetscCall(MatCopy(c->K, c->A, SAME_NONZERO_PATTERN));
  PetscCall(MatAXPY(c->A, -f->iwmu, c->Ms, SAME_NONZERO_PATTERN));
  PetscCall(addElapsed(t0, &c->tAssembly));

  PetscCall(solve(c, c->ksp, f->B, c->x));
  PetscCall(interpolateFields(c, f->iwmu));
  for (PetscInt fld = 0; fld < NUM_RECEIVER_FIELDS; fld++)
    if (c->adjSrc[fld]) PetscCall(VecZeroEntries(c->adjSrc[fld]));
  *misfit = 0.0;
  for (PetscInt i = 0; i < c->imParams->numComponents; i++) {
    PetscReal m;
    PetscCall(componentData(c, i));
    PetscCall(weightedMisfit(c, f, i, PETSC_TRUE, &m));
    *misfit += m;
  }
  for (PetscInt fld = 0; fld < NUM_RECEIVER_FIELDS; fld++) {
    if (!c->adjSrc[fld]) continue;
    if (first)
      PetscCall(MatMultTranspose(c->Q->mat[fld], c->adjSrc[fld], c->nB));
    else
      PetscCall(MatMultTransposeAdd(c->Q->mat[fld], c->adjSrc[fld], c->nB, c->nB));
    first = PETSC_FALSE;
  }
  PetscCall(solve(c, c->ksp, c->nB, c->nx));

  PetscCall(DMGetLocalVector(c->dm, &xLocal));
  PetscCall(DMGetLocalVector(c->dm, &nxLocal));
  PetscCall(DMGlobalToLocal(c->dm, c->x, INSERT_VALUES, xLocal));
  PetscCall(DMGlobalToLocal(c->dm, c->nx, INSERT_VALUES, nxLocal));
  PetscCall(accumulateGradient(c, xLocal, nxLocal, f->iwmu));
  PetscCall(DMRestoreLocalVector(c->dm, &xLocal));
  PetscCall(DMRestoreLocalVector(c->dm, &nxLocal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Data misfit of the current conductivity and its derivative in c->DfDm, summed over frequencies. */
static PetscErrorCode evaluateMisfit(InversionContext *c, PetscReal *misfit) {
  PetscFunctionBeginUser;
  PetscLogDouble t0;

  PetscCall(PetscTime(&t0));
  PetscCall(refillMassMatrix(c->dm, &c->grid, c->conductivity, &c->quad3d, c->Me, c->Ke, c->Ms));
  PetscCall(addElapsed(t0, &c->tAssembly));

  *misfit = 0.0;
  PetscCall(VecZeroEntries(c->DfDm));
  for (PetscInt k = 0; k < c->numFreqs; k++) {
    PetscReal m = 0.0;
    PetscCall(evaluateFrequency(c, &c->freq[k], &m));
    *misfit += m;
  }
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
 * F = sum_f |W (dObs - d)|^2 / Ndata + lambda ||X||^2 / N, with
 * Ndata = 2 x the observed values that are not missing (real and imaginary parts).
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
  const PetscInt    numData = c->numData;
  PetscReal         misfit, regTerm;
  PetscScalar       xDot;
  PetscInt          numCells;

  c->numEvals++;

  PetscCall(applyLogToSigma(c->dmInversion, c->dmConductivity, X, c->X0, c->conductivity, &c->grid, c->graph));
  PetscCall(evaluateMisfit(c, &misfit));

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
 * Columns: iteration, evaluations, RMS, the Tikhonov term, ||g||, ||g||/max(1,||X||),
 * the accepted step, the min-max KSP iterations and the wall time since the
 * previous row.
 *
 * @param[in] ctx        InversionContext.
 * @param[in] iter       Iteration (0 = initial model).
 * @param[in] gnorm      ||g||.
 * @param[in] gradRatio  ||g|| / max(1, ||X||).
 * @param[in] step       Accepted step length (unused on row 0).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode inversionMonitor(void *ctx, PetscInt iter, PetscReal gnorm, PetscReal gradRatio, PetscReal step) {
  PetscFunctionBeginUser;
  InversionContext *c        = (InversionContext *)ctx;
  MPI_Comm          comm     = PetscObjectComm((PetscObject)c->dm);
  const PetscInt    interval = c->imParams->snapshotInterval;
  PetscLogDouble    now;
  char              its[32], stp[16];

  if (iter == 0) {
    PetscCall(PetscPrintf(comm, "   %4s %5s %9s %12s %10s %10s %8s %9s %8s\n", "Iter", "Evals", "RMS", "λΦm", "|g|", "|g|/|x|", "Step",
                          "KSP its", "t (s)"));
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

  c->gradNorms[iter] = gnorm;
  PetscCall(PetscTime(&now));
  PetscCall(PetscPrintf(comm, "   %4" PetscInt_FMT " %5" PetscInt_FMT " %9.4f %10.3e %10.3e %10.3e %8s %9s %8.1f\n", iter, c->numEvals,
                        (double)c->lastRms, (double)c->lastRegTerm, (double)gnorm, (double)gradRatio, stp, its, (double)(now - c->tRow)));
  c->kspItsMinRow = PETSC_INT_MAX;
  c->kspItsMaxRow = 0;
  c->tRow         = now;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Fitted components at the receivers for the current conductivity, one forward solve per frequency.
 *
 * @param[in,out] c     Inversion context (conductivity holds the model).
 * @param[out]    dPre  Predicted data, component-major numComponents x numFreqs x numReceivers, on rank 0 (NULL elsewhere; caller frees).
 * @param[out]    rms   Data RMS of dPre against the observed data.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode predictData(InversionContext *c, PetscScalar **dPre, PetscReal *rms) {
  PetscFunctionBeginUser;
  const PetscInt     numReceivers  = c->Q->numReceivers;
  const PetscInt     numComponents = c->imParams->numComponents;
  PetscReal          misfit        = 0.0;
  PetscMPIInt        rank;
  VecScatter         toZero;
  Vec                preAll;
  KSPConvergedReason reason;

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)c->dm), &rank));
  *dPre = NULL;
  if (rank == 0) PetscCall(PetscCalloc1(numComponents * c->numFreqs * numReceivers, dPre));
  PetscCall(refillMassMatrix(c->dm, &c->grid, c->conductivity, &c->quad3d, c->Me, c->Ke, c->Ms));
  PetscCall(VecScatterCreateToZero(c->dPre, &toZero, &preAll));
  for (PetscInt k = 0; k < c->numFreqs; k++) {
    InversionFrequency *f = &c->freq[k];

    PetscCall(MatCopy(c->K, c->A, SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(c->A, -f->iwmu, c->Ms, SAME_NONZERO_PATTERN));
    PetscCall(KSPSolve(c->ksp, f->B, c->x));
    PetscCall(KSPGetConvergedReason(c->ksp, &reason));
    if (reason < 0 && c->failReason == 0) c->failReason = reason;
    PetscCall(interpolateFields(c, f->iwmu));
    for (PetscInt i = 0; i < numComponents; i++) {
      PetscReal m;
      PetscCall(componentData(c, i));
      PetscCall(weightedMisfit(c, f, i, PETSC_FALSE, &m));
      misfit += m;

      PetscCall(VecScatterBegin(toZero, c->dPre, preAll, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(toZero, c->dPre, preAll, INSERT_VALUES, SCATTER_FORWARD));
      if (rank == 0) {
        const PetscScalar *p;
        PetscCall(VecGetArrayRead(preAll, &p));
        PetscCall(PetscArraycpy(&(*dPre)[(i * c->numFreqs + k) * numReceivers], p, numReceivers));
        PetscCall(VecRestoreArrayRead(preAll, &p));
      }
    }
  }
  PetscCall(VecScatterDestroy(&toZero));
  PetscCall(VecDestroy(&preAll));
  *rms = PetscSqrtReal(misfit / (PetscReal)c->numData);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Multiplies sigma (all three components) of every local cell by exp(-eps dir), i.e. log(rho) += eps dir, from sigma0. */
static PetscErrorCode perturbConductivity(InversionContext *c, Vec sigma0, Vec dirLocal, PetscReal eps) {
  PetscFunctionBeginUser;
  PetscSection       sec;
  const PetscScalar *s0, *dir;
  PetscScalar       *s;

  PetscCall(DMGetLocalSection(c->dmConductivity, &sec));
  PetscCall(VecGetArrayRead(sigma0, &s0));
  PetscCall(VecGetArrayRead(dirLocal, &dir));
  PetscCall(VecGetArray(c->conductivity, &s));
  for (PetscInt i = c->grid.cellStart; i < c->grid.cellEnd; i++) {
    const PetscReal factor = PetscExpReal(-eps * PetscRealPart(dir[i - c->grid.cellStart]));
    PetscInt        off;
    PetscCall(PetscSectionGetOffset(sec, i, &off));
    for (PetscInt j = 0; j < 3; j++) s[off + j] = s0[off + j] * factor;
  }
  PetscCall(VecRestoreArray(c->conductivity, &s));
  PetscCall(VecRestoreArrayRead(dirLocal, &dir));
  PetscCall(VecRestoreArrayRead(sigma0, &s0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Compares the adjoint derivative of the data misfit with central differences.
 *
 * The direction perturbs log(rho) of the cells that are not fixed by a smooth
 * function of the centroid. The model smoothing is not applied.
 *
 * @param[in,out] c  Inversion context (conductivity holds the starting model, restored on exit).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode runGradientCheck(InversionContext *c) {
  PetscFunctionBeginUser;
  const PetscReal   *eps  = c->imParams->checkSteps;
  const PetscInt     n    = c->imParams->numCheckSteps;
  MPI_Comm           comm = PetscObjectComm((PetscObject)c->dm);
  PetscReal          fd[IM_MAX_GRADIENT_CHECK_STEPS], adj[IM_MAX_GRADIENT_CHECK_STEPS], m0, mPlus, mMinus;
  PetscScalar        dot, *dir, *df;
  const PetscScalar *mask, *sigma;
  PetscSection       sec;
  Vec                sigma0, dirLocal, dirGlobal, gLocal, gGlobal;

  PetscCall(VecDuplicate(c->conductivity, &sigma0));
  PetscCall(VecCopy(c->conductivity, sigma0));
  PetscCall(DMCreateLocalVector(c->dmInversion, &dirLocal));
  PetscCall(DMCreateGlobalVector(c->dmInversion, &dirGlobal));
  PetscCall(VecDuplicate(dirLocal, &gLocal));
  PetscCall(VecDuplicate(dirGlobal, &gGlobal));

  /* dir = mask (1 + sin(x/L + 2y/L + 3z/L)) / 2, L the 1 km scale */
  PetscCall(VecGetArray(dirLocal, &dir));
  PetscCall(VecGetArrayRead(c->notFixedMaskLocal, &mask));
  for (PetscInt i = c->grid.cellStart; i < c->grid.cellEnd; i++) {
    const PetscInt li   = i - c->grid.cellStart;
    PetscReal      x[3] = {0.0, 0.0, 0.0};
    Cell           cell;
    PetscCall(extractCellCoordinates(c->dm, i, &cell));
    for (PetscInt v = 0; v < 4; v++)
      for (PetscInt d = 0; d < 3; d++) x[d] += 0.25 * cell.coordinates[3 * v + d];
    dir[li] = PetscRealPart(mask[li]) * 0.5 * (1.0 + PetscSinReal((x[0] + 2.0 * x[1] + 3.0 * x[2]) / 1000.0));
  }
  PetscCall(VecRestoreArrayRead(c->notFixedMaskLocal, &mask));
  PetscCall(VecRestoreArray(dirLocal, &dir));
  PetscCall(VecZeroEntries(dirGlobal));
  PetscCall(DMLocalToGlobal(c->dmInversion, dirLocal, INSERT_VALUES, dirGlobal));

  /* Adjoint: d misfit / d log(rho) = -sigma d misfit / d sigma */
  PetscCall(evaluateMisfit(c, &m0));
  PetscCall(DMGetLocalSection(c->dmConductivity, &sec));
  PetscCall(VecCopy(c->DfDm, gLocal));
  PetscCall(VecGetArray(gLocal, &df));
  PetscCall(VecGetArrayRead(c->conductivity, &sigma));
  for (PetscInt i = c->grid.cellStart; i < c->grid.cellEnd; i++) {
    PetscInt off;
    PetscCall(PetscSectionGetOffset(sec, i, &off));
    df[i - c->grid.cellStart] *= -sigma[off];
  }
  PetscCall(VecRestoreArrayRead(c->conductivity, &sigma));
  PetscCall(VecRestoreArray(gLocal, &df));
  PetscCall(VecZeroEntries(gGlobal));
  PetscCall(DMLocalToGlobal(c->dmInversion, gLocal, ADD_VALUES, gGlobal));
  PetscCall(VecDot(gGlobal, dirGlobal, &dot));

  PetscCall(logSection(comm, "Gradient check"));
  PetscCall(logKvStr(comm, "Data misfit", formatReal(m0)));
  PetscCall(PetscPrintf(comm, "   %10s %16s %16s %12s\n", "eps", "central diff.", "adjoint", "rel. error"));
  for (PetscInt j = 0; j < n; j++) {
    PetscCall(perturbConductivity(c, sigma0, dirLocal, eps[j]));
    PetscCall(evaluateMisfit(c, &mPlus));
    PetscCall(perturbConductivity(c, sigma0, dirLocal, -eps[j]));
    PetscCall(evaluateMisfit(c, &mMinus));
    fd[j]  = (mPlus - mMinus) / (2.0 * eps[j]);
    adj[j] = PetscRealPart(dot);
    PetscCall(PetscPrintf(comm, "   %10.1e %16.8e %16.8e %12.3e\n", (double)eps[j], (double)fd[j], (double)adj[j],
                          (double)(PetscAbsReal(fd[j] - adj[j]) / PetscMax(PetscAbsReal(adj[j]), PETSC_SMALL))));
  }
  PetscCall(VecCopy(sigma0, c->conductivity));
  PetscCall(writeGradientCheck(c->imParams, comm, eps, fd, adj, n));

  PetscCall(VecDestroy(&sigma0));
  PetscCall(VecDestroy(&dirLocal));
  PetscCall(VecDestroy(&dirGlobal));
  PetscCall(VecDestroy(&gLocal));
  PetscCall(VecDestroy(&gGlobal));
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
  PetscCall(MatGetType(ctx->A, &mtype));
  PetscCall(getSolverName(ctx->ksp, stats->solver, sizeof(stats->solver)));
  PetscCall(logSection(comm, "Assembly"));
  PetscCall(logKvStr(comm, "Matrix type", mtype));
  PetscCall(logKvInt(comm, "Right-hand sides", ctx->numFreqs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* L-BFGS from X, then the model of the last accepted X, its predicted data and the HDF5 results. */
static PetscErrorCode runLbfgs(InversionContext *c, Vec X, PetscLogDouble t0, InversionStats *stats) {
  PetscFunctionBeginUser;
  const ImParams *imParams = c->imParams;
  MPI_Comm        comm     = PetscObjectComm((PetscObject)c->dm);
  PetscLogDouble  t1, t2, t3;
  PetscReal       rmsPre = 0.0;
  PetscScalar    *dPre;
  LbfgsResult     result;

  {
    const LbfgsProblem problem = {inversionObjGrad, inversionMonitor, c};
    const LbfgsOptions opts    = {imParams->lbfgsMemory, imParams->maxIter,   imParams->gtol,
                                  imParams->rmsTol,      imParams->rmsRelTol, imParams->rmsStallWindow};
    PetscCall(logSection(comm, "L-BFGS optimization"));
    PetscCall(PetscTime(&t1));
    c->tRow = t1;
    PetscCall(lbfgsOptimize(&problem, &opts, X, &result));
    PetscCall(PetscTime(&t2));
  }
  const PetscLogDouble tOutputLoop = c->tOutput;

  /* Model of the last accepted X (a failed line search leaves the rejected trial in conductivity) */
  PetscCall(applyLogToSigma(c->dmInversion, c->dmConductivity, X, c->X0, c->conductivity, &c->grid, c->graph));
  if (imParams->snapshotInterval > 0 && (c->acceptedIter == 0 || c->acceptedIter % imParams->snapshotInterval != 0)) {
    PetscCall(writeInversionSnapshotVtu(c, c->acceptedIter));
    c->numSnapshots++;
  }
  PetscCall(predictData(c, &dPre, &rmsPre));
  PetscCall(logKvf(comm, "RMS of predicted data", "%.4f", (double)rmsPre));
  PetscCall(writeInversionResults(imParams, c->dmConductivity, c->conductivity, X, c->allRms, result.numIters, result.numEvals,
                                  LbfgsReasons[result.reason], c->gradNorms, c->acceptedIter + 1, dPre, c->Q->numReceivers, rmsPre));
  PetscCall(PetscFree(dPre));
  PetscCall(PetscTime(&t3));

  stats->tSetup    = t1 - t0;
  stats->tAssembly = c->tAssembly;
  stats->tSolver   = c->tSolver;
  stats->tGradient = PetscMax(0.0, (t2 - t1) - c->tAssembly - c->tSolver - tOutputLoop);
  stats->tOutput   = tOutputLoop + (t3 - t2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Top-level inversion driver.
 *
 * Builds the smoothing graph, receiver operators, inversion DM and workspace,
 * runs L-BFGS from the starting model and writes the HDF5 results, or with
 * -im_gradient_check runs the gradient check instead.
 */
PetscErrorCode runCsemInversion(const ImParams *imParams, const DM dm, const Grid *grid, Vec conductivity, Vec materialsId, Vec receivers,
                                InversionStats *stats) {
  PetscFunctionBeginUser;
  PetscLogDouble                t0, t1, t2;
  NeighborGraph                 graph;
  ReceiverInterpolationMatrices Q;
  PetscScalar                  *dObs;
  DM                            dmConductivity, dmInversion;
  Vec                           X0, X, DfDm, maskGlobal, maskLocal;
  PetscReal                    *allRms, *gradNorms;

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
  PetscCall(PetscCalloc1(imParams->maxIter + 1, &gradNorms));

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
      .gradNorms          = gradNorms,
      .lastRms            = PETSC_INFINITY,
      .kspItsMinRow       = PETSC_INT_MAX,
  };
  PetscCall(setupInversionWorkspace(&ctx, dObs));
  PetscCall(PetscFree(dObs));
  PetscCall(logInversionAssembly(&ctx, stats));

  if (imParams->gradientCheck) {
    PetscCall(PetscTime(&t1));
    PetscCall(runGradientCheck(&ctx));
    PetscCall(PetscTime(&t2));
    stats->tSetup    = t1 - t0;
    stats->tAssembly = ctx.tAssembly;
    stats->tSolver   = ctx.tSolver;
    stats->tGradient = PetscMax(0.0, (t2 - t1) - ctx.tAssembly - ctx.tSolver);
    stats->tOutput   = 0.0;
  } else {
    PetscCall(runLbfgs(&ctx, X, t0, stats));
  }
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
  PetscCall(PetscFree(gradNorms));
  PetscCall(destroyNeighborGraph(&graph));
  PetscCall(destroyReceiverInterpolationMatrices(&Q));
  PetscFunctionReturn(PETSC_SUCCESS);
}
