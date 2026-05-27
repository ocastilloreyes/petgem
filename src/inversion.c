/*
 * Filename: inversion.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * CSEM inverse kernel based on L-BFGS optimization (Nocedal 1980)
 * with an adjoint-state gradient and Gauss-Seidel gradient smoothing.
 */

/*
 * Notes:
 * Algorithmic reference: /petgem_inv_new/Ex_inv.m (MATLAB prototype).
 *
 * Inversion parameterization: log-conductivity perturbation X = log(rho) - X0
 * Objective: ||W(dObs - Ex)||^2 + lambda*||X||^2/N
 * Gradient:  adjoint method (forward + adjoint solve per frequency)
 */

/* C libraries */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* PETSc libraries */
#include <petsc.h>
#include <petscdmplex.h>
#include <petscksp.h>

/* PETGEM headers */
#include "assembly.h"
#include "constants.h"
#include "grid.h"
#include "hvfem.h"
#include "inputs.h"
#include "inversion.h"
#include "inversion_internal.h"
#include "transmitter.h"

/* ================================================================== */
/* computeGradientContribution                                         */
/*                                                                     */
/* For each local cell ie:                                             */
/*   idKdm = -2 * constFactor * Me_e                                  */
/*   iG    = idKdm * x_e                                              */
/*   DfDm[ie] += real( iG . nx_e )   (plain transpose, MATLAB iG.'*inx) */
/*                                                                     */
/* The 3D quadrature and Me/Ke buffers are provided by the caller     */
/* (allocated once on the InversionContext, see setupInversionWorkspace).*/
/* That hoist is purely a lifetime change - the values produced by    */
/* each cell evaluation are byte-identical to the previous version.   */
/* ================================================================== */
PetscErrorCode computeGradientContribution(const DM          dm,
                                           const Grid       *grid,
                                           const Vec         conductivity,
                                           const Vec         xLocal,
                                           const Vec         nxLocal,
                                           PetscScalar       constFactor,
                                           DM                dmInversion,
                                           Vec               DfDm,
                                           const Quadrature3D *quadrature_3d,
                                           PetscReal       **Me,
                                           PetscReal       **Ke)
{
  PetscFunctionBeginUser;
  /* nord lives in grid->fem.ops via the supplied quadrature */

  /* Get local section and conductivity DM */
  PetscSection section;
  DM           dmConductivity;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(VecGetDM(conductivity, &dmConductivity));

  /* Loop over local cells */
  for (PetscInt i = grid->cellStart; i < grid->cellEnd; i++) {
    Cell cell;

    PetscCall(extractCellCoordinates(dm, i, &cell));
    PetscCall(computeCellJacobian(&cell));
    PetscCall(extractCellConductivity(dmConductivity, conductivity, i, &cell));
    PetscCall(extractCellClousure(dm, i, &cell));
    PetscCall(computeCellOrientation(&cell));

    /* Zero elemental matrices */
    PetscCall(PetscArrayzero(Me[0], grid->numDofInCell * grid->numDofInCell));
    PetscCall(PetscArrayzero(Ke[0], grid->numDofInCell * grid->numDofInCell));

    /* The gradient formula requires the UNSCALED mass matrix:
     *   dA/d(sigma_e) = -iωμ · Me_unscaled   where   Me_unscaled = ∫ N·N dV
     * computeElementalMatrices fills Me = ∫ N·(sigma·I)·N dV (= MATLAB Mes).
     * Setting conductivity to 1 makes it compute Me_unscaled directly.
     * sigma is still available from c->conductivity for the chain rule later. */
    cell.conductivity[0] = 1.0;
    cell.conductivity[1] = 1.0;
    cell.conductivity[2] = 1.0;
    PetscCall(computeElementalMatrices(&grid->fem, &cell, quadrature_3d, Me, Ke));

    /* Get DOF closure indices */
    PetscInt  numDofIdx, *dofIdx;
    PetscCall(DMPlexGetClosureIndices(dm, section, section, i,
                                      PETSC_TRUE, &numDofIdx, &dofIdx,
                                      NULL, NULL));

    /* Extract local solution values for forward (x_e) and adjoint (nx_e) */
    PetscInt      closureSize = grid->numDofInCell;
    PetscScalar  *x_e  = NULL;
    PetscScalar  *nx_e = NULL;
    PetscCall(DMPlexVecGetClosure(dm, section, xLocal,  i,
                                  &closureSize, &x_e));
    PetscCall(DMPlexVecGetClosure(dm, section, nxLocal, i,
                                  &closureSize, &nx_e));

    /* Compute gradient contribution (plain transpose, matches MATLAB iG.'*inx):
     *   iG[j] = sum_k (-2*constFactor * Me[j][k]) * x_e[k]
     *   contrib = real( sum_j iG[j] * nx_e[j] )               */
    PetscScalar contrib = 0.0 + PETSC_i * 0.0;
    for (PetscInt j = 0; j < grid->numDofInCell; j++) {
      PetscScalar iGj = 0.0;
      for (PetscInt k = 0; k < grid->numDofInCell; k++)
        iGj += (-2.0 * constFactor * Me[j][k]) * x_e[k];
      contrib += iGj * nx_e[j];
    }

    /* Accumulate into DfDm (1 DOF/cell on dmInversion) */
    PetscReal   gradReal = PetscRealPart(contrib);
    PetscScalar gradScalar = gradReal;
    PetscCall(DMPlexVecSetClosure(dmInversion, NULL, DfDm, i,
                                  &gradScalar, ADD_VALUES));

    PetscCall(DMPlexVecRestoreClosure(dm, section, xLocal,  i,
                                      &closureSize, &x_e));
    PetscCall(DMPlexVecRestoreClosure(dm, section, nxLocal, i,
                                      &closureSize, &nx_e));
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, i,
                                          PETSC_TRUE, &numDofIdx, &dofIdx,
                                          NULL, NULL));
  }

  PetscCall(VecAssemblyBegin(DfDm));
  PetscCall(VecAssemblyEnd(DfDm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* setupInversionWorkspace / destroyInversionWorkspace                 */
/*                                                                     */
/* Lifetime hoist of every workspace that does not depend on the       */
/* L-BFGS iterate X.  Called once before the optimization loop and     */
/* freed once after, replacing the per-callback allocate/free dance.   */
/*                                                                     */
/* What gets precomputed:                                              */
/*   • 3D quadrature points + weights (depend only on iparams->nord)   */
/*   • Per-cell Me / Ke elemental-matrix buffers (numDof² each)        */
/*   • Reusable global Vecs b, x, nB, nx and parallel Ex_recv          */
/*   • Per-frequency RHS Vec  Bvec_per_freq[ifre]                      */
/*     (assembleCsemRHS depends only on source + freq, NOT σ)          */
/*   • Per-frequency observed-Ex row Vec  dObsRow_per_freq[ifre]       */
/*   • Per-frequency data weights Vec  Wf_per_freq[ifre]               */
/*                                                                     */
/* All produced values are byte-identical to what the previous per-    */
/* iteration code computed; only the allocation lifetime changes.      */
/* ================================================================== */
static PetscErrorCode setupInversionWorkspace(InversionContext *ctx)
{
  PetscFunctionBeginUser;

  MPI_Comm  comm = PetscObjectComm((PetscObject)ctx->dm);
  PetscInt  numFreqs    = ctx->iparams->numFreqs;
  PetscInt  numDof      = ctx->grid.numDofInCell;
  PetscInt  numReceivers = ctx->Q->numReceivers;
  PetscReal errorLevel  = ctx->iparams->errorLevel;

  /* ---- 3D quadrature (depends only on nord) ---- */
  PetscCall(computeNum3DQuadraturePoints(ctx->iparams->nord, &ctx->quad3d));
  PetscCall(PetscCalloc1(ctx->quad3d.numPoints, &ctx->quad3d.points));
  for (PetscInt i = 0; i < ctx->quad3d.numPoints; i++)
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &ctx->quad3d.points[i]));
  PetscCall(PetscCalloc1(ctx->quad3d.numPoints, &ctx->quad3d.weights));
  PetscCall(compute3DQuadraturePoints(&ctx->quad3d));
  ctx->quad3dInited = PETSC_TRUE;

  /* ---- Me / Ke row-of-pointers buffers (zeroed per cell in the loop) ---- */
  PetscCall(PetscCalloc1(numDof * numDof, &ctx->MeBuf));
  PetscCall(PetscCalloc1(numDof * numDof, &ctx->KeBuf));
  PetscCall(PetscCalloc1(numDof,          &ctx->MeRows));
  PetscCall(PetscCalloc1(numDof,          &ctx->KeRows));
  for (PetscInt i = 0; i < numDof; i++) {
    ctx->MeRows[i] = ctx->MeBuf + i * numDof;
    ctx->KeRows[i] = ctx->KeBuf + i * numDof;
  }

  /* ---- Per-iter reusable Vecs ---- */
  PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->bVec));
  PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->xVec));
  PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->nBvec));
  PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->nxVec));
  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, numReceivers, &ctx->ExRecvVec));
  /* wcdtDvec mirrors the parallel layout of ExRecvVec so the local-receiver
   * loop and MatMultTranspose use consistent row ownership. */
  PetscCall(VecDuplicate(ctx->ExRecvVec, &ctx->wcdtDvec));

  /* ---- Per-frequency RHS / observed-row / weights ---- */
  ctx->numFreqsAlloc = numFreqs;
  PetscCall(PetscCalloc1(numFreqs, &ctx->Bvec_per_freq));
  PetscCall(PetscCalloc1(numFreqs, &ctx->Wf_per_freq));
  PetscCall(PetscCalloc1(numFreqs, &ctx->dObsRow_per_freq));

  /* Build a quiet csemParams stub for assembleCsemRHS (it only reads
   * nord, numMPITasks and quiet - same fields the iter loop used to fill). */
  csemParams stub;
  PetscCall(PetscMemzero(&stub, sizeof(stub)));
  stub.nord = ctx->iparams->nord;
  PetscCallMPI(MPI_Comm_size(comm, &stub.numMPITasks));
  stub.quiet = PETSC_TRUE;

  for (PetscInt ifre = 0; ifre < numFreqs; ifre++) {
    const InvCsemSource *isrc = &ctx->iparams->invSources[ifre];

    /* Build a single-source CsemSourceSet so assembleCsemRHS can do its
     * usual work.  Lives on the stack - assembleCsemRHS copies what it
     * needs into the returned Mat. */
    CsemSource one;
    one.position[0]  = isrc->position[0];
    one.position[1]  = isrc->position[1];
    one.position[2]  = isrc->position[2];
    one.current      = isrc->current;
    one.length       = isrc->length;
    one.dipAngle     = isrc->dipAngle;
    one.azimuthAngle = isrc->azimuthAngle;

    CsemSourceSet setOne;
    setOne.freq        = isrc->freq;
    setOne.numSources  = 1;
    setOne.sourceArray = &one;

    Mat Bmat;
    PetscCall(assembleCsemRHS(stub, setOne, ctx->dm, ctx->grid, &Bmat));

    PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->Bvec_per_freq[ifre]));
    {
      Vec bcol;
      PetscCall(MatDenseGetColumnVecRead(Bmat, 0, &bcol));
      PetscCall(VecCopy(bcol, ctx->Bvec_per_freq[ifre]));
      PetscCall(MatDenseRestoreColumnVecRead(Bmat, 0, &bcol));
    }
    PetscCall(MatDestroy(&Bmat));

    /* Observed-Ex row: extract column ifre from the [numFreqs × numRec]
     * dense Mat dObs (row-major in the underlying storage).  This was
     * being repeated every callback for no reason. */
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, numReceivers, &ctx->dObsRow_per_freq[ifre]));
    {
      const PetscScalar *arr;
      PetscCall(MatDenseGetArrayRead(ctx->dObs, &arr));
      PetscScalar *rArr;
      PetscCall(VecGetArray(ctx->dObsRow_per_freq[ifre], &rArr));
      for (PetscInt r = 0; r < numReceivers; r++)
        rArr[r] = arr[ifre + numFreqs * r];
      PetscCall(VecRestoreArray(ctx->dObsRow_per_freq[ifre], &rArr));
      PetscCall(MatDenseRestoreArrayRead(ctx->dObs, &arr));
    }

    /* Per-frequency weights Wf[r] = 1 / (|dObs[r]| * errorLevel). */
    PetscCall(VecDuplicate(ctx->dObsRow_per_freq[ifre], &ctx->Wf_per_freq[ifre]));
    {
      const PetscScalar *dArr;
      PetscScalar       *wArr;
      PetscCall(VecGetArrayRead(ctx->dObsRow_per_freq[ifre], &dArr));
      PetscCall(VecGetArray(ctx->Wf_per_freq[ifre], &wArr));
      for (PetscInt r = 0; r < numReceivers; r++) {
        PetscReal absVal = PetscAbsScalar(dArr[r]);
        wArr[r] = (absVal > 0.0) ? 1.0 / (absVal * errorLevel) : 0.0;
      }
      PetscCall(VecRestoreArray(ctx->Wf_per_freq[ifre], &wArr));
      PetscCall(VecRestoreArrayRead(ctx->dObsRow_per_freq[ifre], &dArr));
    }
  }

  /* ---- K, Ms (template), G_BDDC built once ----
   * K is σ-independent (μ_r = I, no σ enters the curl-curl integrand).
   * G_BDDC is integer ±1 vertex incidence - purely topological.
   * Both can be reused for every L-BFGS iteration.
   *
   * Ms's values produced here come from whatever σ is in `conductivity`
   * at setup time (= the initial σ derived from X0).  Those values get
   * overwritten by the per-iter assembleCsemMsRefill call, so we only
   * actually keep Ms for its sparsity pattern (= K's pattern, attached
   * to the right local-to-global mapping). */
  csemParams kandmStub;
  PetscCall(PetscMemzero(&kandmStub, sizeof(kandmStub)));
  kandmStub.nord        = ctx->iparams->nord;
  kandmStub.numMPITasks = stub.numMPITasks;
  kandmStub.quiet       = PETSC_TRUE;
  PetscCall(assembleCsemKandM(kandmStub, ctx->dm, ctx->grid,
                               ctx->conductivity,
                               0.0,             /* constFactor (unused: K/Ms mode) */
                               &ctx->Kmat, &ctx->Msmat,
                               NULL /* canonical G - skip */,
                               &ctx->Gmat_BDDC));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tear-down counterpart to setupInversionWorkspace: free everything that
 * helper allocated on the context (quadrature, Me/Ke buffers, per-iter
 * reusable Vecs, per-frequency precomputes, and the cached K / Ms / G_BDDC
 * matrices).  Safe to call on a context that was zero-initialised but
 * never set up (every PetscFree/VecDestroy/MatDestroy handles NULL). */
static PetscErrorCode destroyInversionWorkspace(InversionContext *ctx)
{
  PetscFunctionBeginUser;

  if (ctx->quad3dInited) {
    PetscCall(PetscFree(ctx->quad3d.weights));
    for (PetscInt i = 0; i < ctx->quad3d.numPoints; i++)
      PetscCall(PetscFree(ctx->quad3d.points[i]));
    PetscCall(PetscFree(ctx->quad3d.points));
    ctx->quad3dInited = PETSC_FALSE;
  }
  PetscCall(PetscFree(ctx->MeBuf));
  PetscCall(PetscFree(ctx->KeBuf));
  PetscCall(PetscFree(ctx->MeRows));
  PetscCall(PetscFree(ctx->KeRows));

  PetscCall(VecDestroy(&ctx->bVec));
  PetscCall(VecDestroy(&ctx->xVec));
  PetscCall(VecDestroy(&ctx->nBvec));
  PetscCall(VecDestroy(&ctx->nxVec));
  PetscCall(VecDestroy(&ctx->ExRecvVec));
  PetscCall(VecDestroy(&ctx->wcdtDvec));

  for (PetscInt ifre = 0; ifre < ctx->numFreqsAlloc; ifre++) {
    PetscCall(VecDestroy(&ctx->Bvec_per_freq[ifre]));
    PetscCall(VecDestroy(&ctx->Wf_per_freq[ifre]));
    PetscCall(VecDestroy(&ctx->dObsRow_per_freq[ifre]));
  }
  PetscCall(PetscFree(ctx->Bvec_per_freq));
  PetscCall(PetscFree(ctx->Wf_per_freq));
  PetscCall(PetscFree(ctx->dObsRow_per_freq));
  ctx->numFreqsAlloc = 0;

  PetscCall(MatDestroy(&ctx->Kmat));
  PetscCall(MatDestroy(&ctx->Msmat));
  PetscCall(MatDestroy(&ctx->Gmat_BDDC));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* createInversionDM                                                   */
/*                                                                     */
/* Creates a DM with 1 DOF per cell for the inversion variables.       */
/* Clones the DMPlex topology from dmConductivity and installs a new    */
/* PetscSection with a single scalar DOF on each cell.                 */
/*                                                                     */
/* A 1-component PetscFV is registered as field 0 so that cell-centered */
/* data can be written to VTU (DMPlexVTKWriteAll_VTU traverses          */
/* registered fields; without one, DMGetField fails).                   */
/* ================================================================== */
PetscErrorCode createInversionDM(DM dmConductivity, const Grid *grid, DM *dmInv)
{
  PetscFunctionBeginUser;

  MPI_Comm comm = PetscObjectComm((PetscObject)dmConductivity);

  PetscCall(DMClone(dmConductivity, dmInv));

  /* Register a 1-component cell-centered field so VTK/VTU writers can
   * resolve DMGetField. The section below still defines the DOF layout;
   * we only need a valid field *object* to be present. */
  PetscFV fvm;
  PetscCall(PetscFVCreate(comm, &fvm));
  PetscCall(PetscFVSetNumComponents(fvm, 1));
  PetscCall(PetscObjectSetName((PetscObject)fvm, "scalar"));
  PetscCall(DMAddField(*dmInv, NULL, (PetscObject)fvm));
  PetscCall(PetscFVDestroy(&fvm));

  PetscInt pStart, pEnd;
  PetscCall(DMPlexGetChart(dmConductivity, &pStart, &pEnd));

  PetscSection sec;
  PetscCall(PetscSectionCreate(comm, &sec));
  PetscCall(PetscSectionSetNumFields(sec, 1));
  PetscCall(PetscSectionSetFieldComponents(sec, 0, 1));
  PetscCall(PetscSectionSetChart(sec, pStart, pEnd));
  for (PetscInt i = grid->cellStart; i < grid->cellEnd; i++) {
    PetscCall(PetscSectionSetDof(sec, i, 1));
    PetscCall(PetscSectionSetFieldDof(sec, i, 0, 1));
  }
  PetscCall(PetscSectionSetUp(sec));
  PetscCall(DMSetLocalSection(*dmInv, sec));
  PetscCall(PetscSectionDestroy(&sec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* createInvKSP                                                        */
/*                                                                     */
/* Same setup logic as solveCsemSystem but returns without solving.   */
/* Caller calls solveInvSystem twice (forward + adjoint) then         */
/* KSPDestroy. MUMPS factorization is triggered on first KSPSolve.   */
/* ================================================================== */
PetscErrorCode createInvKSP(const invParams *iparams,
                             const DM          dm,
                             const Mat         A,
                             const Mat         G,
                             KSP              *ksp)
{
  PetscFunctionBeginUser;

  MPI_Comm  comm    = PetscObjectComm((PetscObject)dm);
  PetscBool ismatis = PETSC_FALSE;

  PetscCall(KSPCreate(comm, ksp));
  PetscCall(KSPSetOperators(*ksp, A, A));

  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (ismatis && G) {
    PC pc;
    PetscCall(KSPGetPC(*ksp, &pc));
    PetscCall(PCSetType(pc, PCBDDC));
    PetscCall(PCBDDCSetDiscreteGradient(pc, G, iparams->nord, 0,
                                        PETSC_TRUE, PETSC_TRUE));
  }
  PetscCall(KSPSetFromOptions(*ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* solveInvSystem                                                      */
/* ================================================================== */
PetscErrorCode solveInvSystem(const KSP ksp, const Vec rhs, Vec sol)
{
  PetscFunctionBeginUser;
  PetscCall(KSPSolve(ksp, rhs, sol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* inversionObjGrad                                                    */
/*                                                                     */
/* Objective + gradient callback for the L-BFGS optimizer:            */
/*  1. Recover sigma from X                                           */
/*  2. Reassemble LHS (uses updated sigma)                            */
/*  3. Frequency loop: forward solve → field interpolation →          */
/*     residual → adjoint RHS → adjoint solve → gradient             */
/*  4. Chain rule + smoothing + Tikhonov                              */
/* ================================================================== */
PetscErrorCode inversionObjGrad(Vec X, PetscReal *F, Vec Gvec,
                                void *ctx)
{
  InversionContext *c = (InversionContext *)ctx;
  MPI_Comm          comm = PetscObjectComm((PetscObject)c->dm);
  PetscInt          numFreqs    = c->iparams->numFreqs;
  PetscInt          numReceivers = c->Q->numReceivers;
  PetscReal         lambda       = c->iparams->lambda;

  PetscFunctionBeginUser;

  c->iterCount++;

  /* ---- 1. Recover conductivity from smoothed X (1-DOF → 4-DOF) ----
   * applyLogToSigma applies the same GS smoothing MATLAB uses on tempX
   * before computing sigma.  diagGradientWeight=0 matches diag_gwight=0.
   * Capture pre/post-smoothing X for VTU diagnostics when enabled. */
  if (c->XPreSmooth) PetscCall(VecCopy(X, c->XPreSmooth));
  {
    PetscReal fwdDiagWeight = c->bypassSmoother ? -1.0 : 0.0;
    PetscCall(applyLogToSigma(c->dmInversion, c->dmConductivity,
                              X, c->X0, c->conductivity, &c->grid,
                              c->graph, fwdDiagWeight, c->XPostSmooth));
  }

  /* ---- 2. Refill Ms(σ) with the current iterate's σ ----
   * K (curl-curl stiffness, σ-independent) and G_BDDC (topological,
   * σ-independent) were built once in setupInversionWorkspace and live
   * on the context - they are reused for every L-BFGS iteration.
   * Only Ms needs to be recomputed when σ changes.
   *
   * assembleCsemMsRefill walks the local cells, shares the per-cell
   * setup helper (prepareCellForAssembly) with assembleCsemKandM, and
   * overwrites Ms in-place using the cached sparsity pattern. */
  csemParams fwdParams;
  PetscCall(PetscMemzero(&fwdParams, sizeof(fwdParams)));
  fwdParams.nord = c->iparams->nord;
  PetscCallMPI(MPI_Comm_size(comm, &fwdParams.numMPITasks));
  fwdParams.quiet = PETSC_TRUE;

  PetscLogDouble tA0, tA1;   /* phase-timer scratch (assembly / solver) */
  PetscCall(PetscTime(&tA0));
  PetscCall(assembleCsemMsRefill(fwdParams, c->dm, c->grid,
                                  c->conductivity,
                                  &c->quad3d, c->MeRows, c->KeRows,
                                  c->Msmat));
  PetscCall(PetscTime(&tA1));
  c->tAssembly += tA1 - tA0;

  Mat Kmat = c->Kmat;
  Mat Msmat = c->Msmat;
  Mat Gmat  = c->Gmat_BDDC;

  /* ---- 3. Zero gradient accumulator ---- */
  PetscCall(VecZeroEntries(c->DfDm));

  /* ---- 4. Misfit accumulators ---- */
  PetscReal reduil_fi  = 0.0;
  PetscInt  numData    = numReceivers * numFreqs * 2; /* real+imag */

  /* Pre-allocated workspace (lives on InversionContext, set up once
   * before the L-BFGS loop). The five Vecs and the per-freq RHS / Wf /
   * dObsRow arrays are all reused across iterations - identical numerical
   * values to recomputing them each call, just without the allocations. */
  Vec b       = c->bVec;
  Vec x       = c->xVec;
  Vec nB      = c->nBvec;
  Vec nx      = c->nxVec;
  Vec Ex_recv = c->ExRecvVec;
  Vec wcdtD_mpi = c->wcdtDvec;

  /* One A matrix for the whole iteration (numFreqs systems share its
   * sparsity pattern). Allocate with DO_NOT_COPY_VALUES; each freq does
   * MatCopy(K → A) + MatAXPY(-Const · Ms). Replaces numFreqs× MatDuplicate
   * (COPY_VALUES) with one allocation + numFreqs× value copy. */
  Mat A;
  PetscCall(PetscTime(&tA0));
  PetscCall(MatDuplicate(Kmat, MAT_DO_NOT_COPY_VALUES, &A));
  PetscCall(PetscTime(&tA1));
  c->tAssembly += tA1 - tA0;

  /* ---- 5. Frequency loop ---- */
  for (PetscInt ifre = 0; ifre < numFreqs; ifre++) {
    const InvCsemSource *isrc = &c->iparams->invSources[ifre];

    PetscReal   omega = isrc->freq * 2.0 * PETSC_PI;
    PetscScalar Const = PETSC_i * omega * MU;

    /* A_f = K - iωμ·Ms.  SAME_NONZERO_PATTERN lets PETSc skip symbolic. */
    PetscCall(PetscTime(&tA0));
    PetscCall(MatCopy(Kmat, A, SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(A, -Const, Msmat, SAME_NONZERO_PATTERN));
    PetscCall(PetscTime(&tA1));
    c->tAssembly += tA1 - tA0;

    /* RHS, observed-Ex row, and per-freq weights are all precomputed
     * in setupInversionWorkspace - they only depend on source + dObs +
     * errorLevel, none of which change during L-BFGS. */
    PetscCall(VecCopy(c->Bvec_per_freq[ifre], b));
    Vec dObs_row = c->dObsRow_per_freq[ifre];
    Vec Wf       = c->Wf_per_freq[ifre];

    /* Factorize and solve forward system: A_f * x = b */
    KSP ksp;
    PetscCall(PetscTime(&tA0));
    PetscCall(createInvKSP(c->iparams, c->dm, A, Gmat, &ksp));
    PetscCall(solveInvSystem(ksp, b, x));
    PetscCall(PetscTime(&tA1));
    c->tSolver += tA1 - tA0;

    /* Convert global x to local for field interpolation & gradient */
    Vec xLocal;
    PetscCall(DMGetLocalVector(c->dm, &xLocal));
    PetscCall(DMGlobalToLocal(c->dm, x, INSERT_VALUES, xLocal));

    /* Compute Ex at receivers: Ex_recv = QEx * x */
    PetscCall(MatMult(c->Q->QEx, x, Ex_recv));

    /* Compute misfit and adjoint RHS */
    PetscInt exStart, exEnd;
    PetscCall(VecGetOwnershipRange(Ex_recv, &exStart, &exEnd));
    PetscInt locNRec = exEnd - exStart;

    const PetscScalar *exArr, *dArr, *wArr;
    PetscCall(VecGetArrayRead(Ex_recv, &exArr));
    PetscCall(VecGetArrayRead(dObs_row, &dArr));
    PetscCall(VecGetArrayRead(Wf, &wArr));

    PetscScalar *wcArr;
    PetscCall(VecGetArray(wcdtD_mpi, &wcArr));

    PetscReal local_fi = 0.0;
    for (PetscInt lr = 0; lr < locNRec; lr++) {
      PetscInt    r     = exStart + lr;
      PetscScalar resid = dArr[r] - exArr[lr];
      PetscReal   w     = PetscRealPart(wArr[r]);
      PetscScalar dtD   = w * resid;
      local_fi += PetscRealPart(dtD * PetscConj(dtD));
      wcArr[lr] = w * w * PetscConj(resid);
    }
    PetscCall(VecRestoreArray(wcdtD_mpi, &wcArr));
    PetscCall(VecRestoreArrayRead(Wf, &wArr));
    PetscCall(VecRestoreArrayRead(dObs_row, &dArr));
    PetscCall(VecRestoreArrayRead(Ex_recv, &exArr));

    /* Sum local misfit contributions across all ranks */
    PetscReal global_fi;
    PetscCallMPI(MPI_Allreduce(&local_fi, &global_fi, 1, MPIU_REAL, MPI_SUM, comm));
    reduil_fi += global_fi;

    /* Adjoint RHS: nB = QEx^T * wcdtD_mpi */
    PetscCall(MatMultTranspose(c->Q->QEx, wcdtD_mpi, nB));

    /* Adjoint solve: A_f * nx = nB  (reuse factorization from forward) */
    PetscCall(PetscTime(&tA0));
    PetscCall(solveInvSystem(ksp, nB, nx));
    PetscCall(PetscTime(&tA1));
    c->tSolver += tA1 - tA0;

    /* Local adjoint solution for gradient accumulation */
    Vec nxLocal;
    PetscCall(DMGetLocalVector(c->dm, &nxLocal));
    PetscCall(DMGlobalToLocal(c->dm, nx, INSERT_VALUES, nxLocal));

    /* Accumulate per-element gradient (1 DOF/cell) */
    PetscCall(computeGradientContribution(c->dm, &c->grid,
                                          c->conductivity,
                                          xLocal, nxLocal, Const,
                                          c->dmInversion, c->DfDm,
                                          &c->quad3d, c->MeRows, c->KeRows));

    /* Cleanup frequency-level objects (workspace stays alive on ctx). */
    PetscCall(DMRestoreLocalVector(c->dm, &xLocal));
    PetscCall(DMRestoreLocalVector(c->dm, &nxLocal));
    PetscCall(KSPDestroy(&ksp));
  } /* end frequency loop */

  /* Destroy A (per-iter scratch). K, Ms and G_BDDC live on the context
   * and are released once at runCsemInversion teardown. */
  PetscCall(MatDestroy(&A));

  /* ---- 5. RMS ---- */
  /* RMS = sqrt( sum_freq sum_rec |W*(d_obs - E_x)|^2 / Ndata )
   * W_r   = 1 / (|d_obs_r| * errorLevel)   (amplitude-relative weight)
   * Ndata = numReceivers * numFreqs * 2     (factor 2: real + imag parts) */
  PetscReal dataMisfit = reduil_fi / (PetscReal)numData;
  PetscReal rms        = PetscSqrtReal(dataMisfit);
  if (c->allRMS) c->allRMS[c->iterCount - 1] = rms;

  /* ---- 6. Objective function (Tikhonov regularization) ---- */
  /* F = dataMisfit + lambda * ||X||^2 / N_cells
   * Matches MATLAB: reduil_fi/(N*M*2) + lamda*dot(X,X)/nElems */
  PetscScalar xDot;
  PetscCall(VecDot(X, X, &xDot));
  PetscReal xNorm2 = PetscRealPart(xDot);
  PetscInt  nGlobal;
  PetscCall(VecGetSize(X, &nGlobal));
  PetscReal regTerm = lambda * xNorm2 / (PetscReal)nGlobal;
  *F = dataMisfit + regTerm;
  /* Stash data/reg split so lbfgsOptimize can render the one-line summary. */
  c->lastDataMisfit = dataMisfit;
  c->lastRegTerm    = regTerm;

  /* ---- 7. Chain rule: DfDm *= -sigma  (matching MATLAB DfDm.*(-sigma)) ----
   * DfDm is 1 DOF/cell on dmInversion; sigma comes from component 0
   * of the 4-DOF/cell conductivity Vec. */
  {
    PetscSection resSec;
    PetscCall(DMGetLocalSection(c->dmConductivity, &resSec));
    PetscScalar       *dfArr;
    const PetscScalar *sArr;
    PetscCall(VecGetArray(c->DfDm, &dfArr));
    PetscCall(VecGetArrayRead(c->conductivity, &sArr));
    for (PetscInt i = c->grid.cellStart; i < c->grid.cellEnd; i++) {
      PetscInt li = i - c->grid.cellStart;
      PetscInt resOff;
      PetscCall(PetscSectionGetOffset(resSec, i, &resOff));
      dfArr[li] = -dfArr[li] * sArr[resOff]; /* component 0 = sigma_x */
    }
    PetscCall(VecRestoreArrayRead(c->conductivity, &sArr));
    PetscCall(VecRestoreArray(c->DfDm, &dfArr));
  }

  /* Capture raw gradient (MATLAB dfdm0) before fixed-zero and smoothing. */
  if (c->DfDmRaw) PetscCall(VecCopy(c->DfDm, c->DfDmRaw));

  /* Zero local gradient at fixed elements via the persistent mask. */
  PetscCall(VecPointwiseMult(c->DfDm, c->DfDm, c->notFixedMaskLocal));

  /* ---- 8. Gradient smoothing (forward + reverse Gauss-Seidel) ----
   * DfDm is already 1 value per cell - apply smoothing directly.
   * bypassSmoother (FD check mode) passes the sentinel -1.0 to skip it. */
  {
    PetscReal gradDiagWeight = c->bypassSmoother
                                 ? -1.0
                                 : c->iparams->diagGradientWeight;
    PetscCall(applyGaussSeidelSmoothing(c->graph, gradDiagWeight, c->DfDm));
  }

  /* ---- 9. Scale local gradient by 1/numData ---- */
  PetscCall(VecScale(c->DfDm, 1.0 / (PetscReal)numData));

  /* ---- 10. Scatter local DfDm to global gradient Gvec ----
   * DMLocalToGlobal on dmInversion (1 DOF/cell). With overlap=0
   * and cell-based DOFs, each cell belongs to exactly one rank. */
  PetscCall(VecZeroEntries(Gvec));
  PetscCall(DMLocalToGlobal(c->dmInversion, c->DfDm, ADD_VALUES, Gvec));

  /* ---- 11. Tikhonov regularization gradient: Gvec += 2*lambda/N * X ---- */
  PetscCall(VecAXPY(Gvec, 2.0 * lambda / (PetscReal)nGlobal, X));

  /* ---- 12. Zero gradient at fixed elements in global Gvec via mask ---- */
  PetscCall(VecPointwiseMult(Gvec, Gvec, c->notFixedMaskGlobal));

  /* Capture final gradient (MATLAB DfDM) for VTU diagnostics. */
  if (c->DfDmFinal) PetscCall(VecCopy(Gvec, c->DfDmFinal));

  /* Publish the most recent RMS so the optimizer loop can use it
   * for early-stopping, matching MATLAB's `rms <= 1.05` exit. */
  c->lastRMS = rms;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* runFdGradientCheck                                                   */
/*                                                                     */
/* Diagnostic: compare the adjoint-based gradient against a centered    */
/* finite-difference approximation on `ncells` non-fixed cells.        */
/*                                                                     */
/* For each selected global cell index i:                              */
/*   G_adj = G[i] returned by inversionObjGrad at the baseline X       */
/*   FD    = (F(X + eps*e_i) - F(X - eps*e_i)) / (2*eps)               */
/* Prints ratio = FD / G_adj per cell. Centered FD has O(eps^2)        */
/* truncation error; forward FD was O(eps), which dominates the signal */
/* for cells with small |G| and produced spurious mixed ratios.        */
/*                                                                     */
/* Interpretation:                                                     */
/*   ratio ~ +1 : gradient correct, bug lies in smoother/mask/mapping. */
/*   ratio ~ -1 : sign flip (commonly in dsigma/dX = -sigma).          */
/*   ratio O(1) but != 1 : missing scalar factor (mu0, i*omega, W...). */
/*   ratio random/huge  : indexing bug between X and G.                */
/*                                                                     */
/* Side effects: leaves ctx->conductivity in the last perturbed state. */
/* Callers that continue past this function (currently none) must      */
/* trigger a fresh objgrad before trusting ctx->conductivity.          */
/* ================================================================== */
PetscErrorCode runFdGradientCheck(InversionContext *ctx, Vec X,
                                          PetscInt ncells, PetscReal eps)
{
  PetscFunctionBeginUser;

  MPI_Comm    comm = PetscObjectComm((PetscObject)X);
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCheck(ncells > 0 && ncells <= INV_MAX_FD_CHECK_CELLS, comm,
             PETSC_ERR_ARG_OUTOFRANGE,
             "-inv_dev_fd_check must be in [1,%d] "
             "(bump INV_MAX_FD_CHECK_CELLS in include/constants.h to raise)",
             INV_MAX_FD_CHECK_CELLS);

  /* ---- Gather the not-fixed mask to rank 0 and pick ncells well-spaced
   *      non-fixed global cell indices.                               */
  Vec        maskAll;
  VecScatter sMask;
  PetscCall(VecScatterCreateToZero(ctx->notFixedMaskGlobal, &sMask, &maskAll));
  PetscCall(VecScatterBegin(sMask, ctx->notFixedMaskGlobal, maskAll,
                            INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd  (sMask, ctx->notFixedMaskGlobal, maskAll,
                            INSERT_VALUES, SCATTER_FORWARD));

  PetscInt chosen[INV_MAX_FD_CHECK_CELLS] = {0};
  PetscInt numChosen = 0;
  if (rank == 0) {
    PetscInt           Ntotal;
    const PetscScalar *mArr;
    PetscCall(VecGetSize(maskAll, &Ntotal));
    PetscCall(VecGetArrayRead(maskAll, &mArr));
    PetscInt stride = PetscMax((PetscInt)1, Ntotal / (ncells + 1));
    PetscInt start  = stride / 2;
    for (PetscInt k = 0; k < ncells && numChosen < ncells; k++) {
      PetscInt idx   = start + k * stride;
      PetscInt probe = 0;
      while (probe < Ntotal &&
             PetscRealPart(mArr[(idx + probe) % Ntotal]) <= 0.5) probe++;
      if (probe < Ntotal) {
        PetscInt pick = (idx + probe) % Ntotal;
        /* avoid duplicates if stride collapsed */
        PetscBool dup = PETSC_FALSE;
        for (PetscInt j = 0; j < numChosen; j++)
          if (chosen[j] == pick) { dup = PETSC_TRUE; break; }
        if (!dup) chosen[numChosen++] = pick;
      }
    }
    PetscCall(VecRestoreArrayRead(maskAll, &mArr));
  }
  PetscCallMPI(MPI_Bcast(&numChosen, 1, MPIU_INT, 0, comm));
  if (numChosen > 0)
    PetscCallMPI(MPI_Bcast(chosen, numChosen, MPIU_INT, 0, comm));
  PetscCall(VecScatterDestroy(&sMask));
  PetscCall(VecDestroy(&maskAll));

  PetscCheck(numChosen > 0, comm, PETSC_ERR_PLIB,
             "FD check: could not select any non-fixed cells");

  PetscCall(PetscPrintf(comm,
    "\n === Finite-difference gradient check ===\n"
    "   Non-fixed cells   : %" PetscInt_FMT "\n"
    "   Epsilon           : %g\n"
    "   Scheme            : centered difference (O(eps^2))\n"
    "   Smoother          : BYPASSED (self-consistent F vs G)\n\n",
    numChosen, (double)eps));

  /* Bypass the forward + gradient smoothers for the duration of the
   * check. The smoother is non-self-adjoint (forward+reverse GS with
   * neighbour averaging), so with it active FD(F(X)) != G even when
   * the adjoint machinery is correct. Bypassing makes F(X) = F(sigma(X,X0))
   * and G(X) = dF/dX exactly, so ratios ~ 1 iff the gradient is right. */
  PetscBool savedBypass = ctx->bypassSmoother;
  ctx->bypassSmoother   = PETSC_TRUE;

  /* ---- Baseline: F0 and G from one adjoint solve ---- */
  Vec Gbase;
  PetscCall(VecDuplicate(X, &Gbase));
  PetscReal F0;
  PetscCall(inversionObjGrad(X, &F0, Gbase, ctx));

  /* ---- Replicate X and Gbase to all ranks for cell-local lookups ---- */
  Vec        Xall, Gall;
  VecScatter sX, sG;
  PetscCall(VecScatterCreateToAll(X,     &sX, &Xall));
  PetscCall(VecScatterCreateToAll(Gbase, &sG, &Gall));
  PetscCall(VecScatterBegin(sX, X,     Xall, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd  (sX, X,     Xall, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(sG, Gbase, Gall, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd  (sG, Gbase, Gall, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(PetscPrintf(comm,
    "   Baseline F0 = %14.6e\n\n", (double)F0));
  PetscCall(PetscPrintf(comm,
    "   %8s   %14s   %14s   %14s   %10s   %12s\n",
    "cell_idx", "G_adjoint", "FD_approx", "|FD-G|",
    "ratio", "|FD-G|/|G|"));
  PetscCall(PetscPrintf(comm,
    "   --------   --------------   --------------   --------------"
    "   ----------   ------------\n"));

  /* Ownership range for targeted VecSetValue - only the owner sets the
   * value so INSERT_VALUES stays unambiguous under MPI. */
  PetscInt lo, hi;
  PetscCall(VecGetOwnershipRange(X, &lo, &hi));

  PetscReal sumRatio = 0.0, maxAbsErr = 0.0;
  PetscInt  nValid = 0;

  for (PetscInt k = 0; k < numChosen; k++) {
    PetscInt idx = chosen[k];

    /* Read xi and Gi from the all-ranks replicas */
    const PetscScalar *xArr, *gArr;
    PetscCall(VecGetArrayRead(Xall, &xArr));
    PetscCall(VecGetArrayRead(Gall, &gArr));
    PetscScalar xi = xArr[idx];
    PetscReal   Gi = PetscRealPart(gArr[idx]);
    PetscCall(VecRestoreArrayRead(Xall, &xArr));
    PetscCall(VecRestoreArrayRead(Gall, &gArr));

    /* Centered difference: evaluate at X + eps*e_i and X - eps*e_i */
    if (idx >= lo && idx < hi)
      PetscCall(VecSetValue(X, idx, xi + eps, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(X));
    PetscCall(VecAssemblyEnd(X));

    PetscReal Fplus;
    PetscCall(inversionObjGrad(X, &Fplus, Gbase, ctx));

    if (idx >= lo && idx < hi)
      PetscCall(VecSetValue(X, idx, xi - eps, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(X));
    PetscCall(VecAssemblyEnd(X));

    PetscReal Fminus;
    PetscCall(inversionObjGrad(X, &Fminus, Gbase, ctx));

    /* Restore X[idx] */
    if (idx >= lo && idx < hi)
      PetscCall(VecSetValue(X, idx, xi, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(X));
    PetscCall(VecAssemblyEnd(X));

    PetscReal fd     = (Fplus - Fminus) / (2.0 * eps);
    PetscReal absErr = PetscAbsReal(fd - Gi);
    PetscReal ratio  = (PetscAbsReal(Gi) > 1e-30) ? (fd / Gi) : 0.0;
    PetscReal relErr = (PetscAbsReal(Gi) > 1e-30)
                         ? absErr / PetscAbsReal(Gi) : absErr;

    PetscCall(PetscPrintf(comm,
      "   %8" PetscInt_FMT "   %+14.6e   %+14.6e   %14.6e   %+10.4f   %12.4e\n",
      idx, (double)Gi, (double)fd, (double)absErr,
      (double)ratio, (double)relErr));

    if (PetscAbsReal(Gi) > 1e-30) {
      sumRatio += ratio;
      nValid++;
    }
    if (absErr > maxAbsErr) maxAbsErr = absErr;
  }

  PetscCall(PetscPrintf(comm,
    "\n   mean ratio = %+.4f   (over %" PetscInt_FMT " cells with |G|>1e-30)\n",
    (nValid > 0) ? (double)(sumRatio / nValid) : 0.0, nValid));
  PetscCall(PetscPrintf(comm,
    "   max |FD-G| = %.4e\n", (double)maxAbsErr));

  PetscCall(PetscPrintf(comm,
    "\n Interpretation:\n"
    "   ratio ~ +1     gradient correct; bug is in smoother/mask/mapping\n"
    "   ratio ~ -1     sign flip (check log-Jacobian dsigma/dX = -sigma)\n"
    "   ratio O(1)!=1  missing scalar factor (mu0, i*omega, W vs W^2)\n"
    "   ratio huge/nan indexing bug between X[i] and G[i]\n\n"));

  PetscCall(VecDestroy(&Xall));
  PetscCall(VecDestroy(&Gall));
  PetscCall(VecScatterDestroy(&sX));
  PetscCall(VecScatterDestroy(&sG));
  PetscCall(VecDestroy(&Gbase));

  ctx->bypassSmoother = savedBypass;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ================================================================== */
/* runCsemInversion                                                    */
/* ================================================================== */
PetscErrorCode runCsemInversion(const invParams  *iparams,
                                const DM          dm,
                                const Grid       *grid,
                                Vec               conductivity,
                                Vec               materialsID,
                                Vec               receivers,
                                PetscLogDouble   *tAssemblyOut,
                                PetscLogDouble   *tSolverOut)
{
  PetscFunctionBeginUser;

  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* Validate preconditions: loadCsemInputs returns NULL for conductivity
   * and materialsID only if it was called with an empty inputFile, which
   * readCsemParams already errors out on. Re-check here for safety. */
  PetscCheck(conductivity, comm, PETSC_ERR_ARG_NULL,
             "conductivity Vec is NULL - loadCsemInputs did not populate "
             "the model. Check that -input_filename is valid.");
  PetscCheck(materialsID, comm, PETSC_ERR_ARG_NULL,
             "materialsID Vec is NULL - loadCsemInputs did not populate "
             "the model. Check that -input_filename is valid.");
  PetscCheck(receivers, comm, PETSC_ERR_ARG_NULL,
             "receivers Vec is NULL - loadCsemInputs did not return /receivers.");

  /* ---- Build neighbor smoothing graph ---- */
  NeighborGraph graph;
  PetscCall(buildNeighborSmoothingGraph(dm, grid, iparams, materialsID, &graph));

  /* ---- Build receiver Q matrices ---- */
  ReceiverInterpolationMatrices Q;
  PetscCall(buildReceiverInterpolationMatrices(iparams->nord,
                                                receivers,
                                                dm, grid, &Q));

  /* ---- Load observed data from the unified bundle (/observed/Ex) ---- */
  Mat dObs;
  PetscCall(loadObservedData(iparams->bundleFile, iparams->numFreqs,
                              Q.numReceivers, &dObs));

  /* ---- Get conductivity DM (3 DOF/cell) for scatter operations ---- */
  DM dmConductivity;
  PetscCall(VecGetDM(conductivity, &dmConductivity));

  /* ---- Create inversion DM (1 DOF/cell) for X, X0, gradient ---- */
  DM dmInversion;
  PetscCall(createInversionDM(dmConductivity, grid, &dmInversion));

  /* ---- Build the parallel block-Jacobi smoother graph (multi-rank only).
   * No-op on a single rank; on >1 ranks, switches applyGaussSeidelSmoothing
   * to a fully-parallel forward+reverse Gauss-Seidel path that uses one
   * layer of ghost cells (overlap=1) and exchanges them between sweeps -
   * O(local cells) per call, no rank-0 bottleneck. Eliminates the partition-
   * boundary seams that the original partition-local sweep produced.
   *
   * Result is mathematically NOT bit-identical to the sequential rank-0
   * path (any fully-parallel GS variant differs in summation order / use
   * of one-step-stale ghost values), but the recovered model matches -
   * same constraint already accepted for MUMPS-induced trajectory drift. */
  PetscCall(setupParallelSmoothingGraph(&graph, dm, grid));

  /* ---- Initial model X0 = log(1/sigma_x) ----
   * X and X0 are global distributed Vecs on dmInversion (1 DOF/cell)
   * so that L-BFGS operates on a properly distributed scalar variable.
   * DfDm is a local Vec on dmInversion used as a scratch accumulator
   * inside the callback, then scattered to the global gradient via
   * DMLocalToGlobal.                                                  */
  Vec X0, X, DfDm;
  PetscCall(DMCreateGlobalVector(dmInversion, &X0));
  PetscCall(DMCreateGlobalVector(dmInversion, &X));
  PetscCall(DMCreateLocalVector(dmInversion, &DfDm));

  {
    /* Compute X0 = log(1/sigma_x) from component 0 of conductivity */
    Vec X0_local;
    PetscCall(DMCreateLocalVector(dmInversion, &X0_local));

    PetscSection resSec;
    PetscCall(DMGetLocalSection(dmConductivity, &resSec));
    const PetscScalar *sArr;
    PetscScalar       *x0Arr;
    PetscCall(VecGetArrayRead(conductivity, &sArr));
    PetscCall(VecGetArray(X0_local, &x0Arr));
    for (PetscInt i = grid->cellStart; i < grid->cellEnd; i++) {
      PetscInt li = i - grid->cellStart;
      PetscInt resOff;
      PetscCall(PetscSectionGetOffset(resSec, i, &resOff));
      x0Arr[li] = PetscLogReal(1.0 / PetscRealPart(sArr[resOff]));
    }
    PetscCall(VecRestoreArray(X0_local, &x0Arr));
    PetscCall(VecRestoreArrayRead(conductivity, &sArr));
    PetscCall(DMLocalToGlobal(dmInversion, X0_local, INSERT_VALUES, X0));
    PetscCall(VecDestroy(&X0_local));
  }
  PetscCall(VecZeroEntries(X));   /* perturbation starts at 0 */

  /* ---- Build inversion context ---- */
  /* allRMS must hold one entry per objgrad call, not per L-BFGS iteration.
   * The line search calls objgrad up to maxLs=20 times per iteration,
   * plus one initial evaluation, so worst case is maxIter*(20+1)+1. */
  PetscInt  allRMSSize = iparams->maxIter * 21 + 1;
  PetscReal *allRMS;
  PetscCall(PetscCalloc1(allRMSSize, &allRMS));

  /* Persistent 0/1 mask for fixed cells (built once, used every
   * objgrad call to replace fragile DMPlexVecSetClosure patterns). */
  Vec notFixedMaskGlobal, notFixedMaskLocal;
  PetscCall(buildNotFixedMask(&graph, dmInversion, grid,
                               &notFixedMaskGlobal, &notFixedMaskLocal));

  /* Diagnostic snapshot buffers - allocated only when VTU snapshots
   * are enabled. NULL entries in the context disable the corresponding
   * capture inside inversionObjGrad / applyLogToSigma. */
  Vec DfDmRaw = NULL, DfDmFinal = NULL, XPreSmooth = NULL, XPostSmooth = NULL;
  if (iparams->snapshotInterval > 0) {
    PetscCall(DMCreateLocalVector (dmInversion, &DfDmRaw));
    PetscCall(DMCreateGlobalVector(dmInversion, &DfDmFinal));
    PetscCall(DMCreateGlobalVector(dmInversion, &XPreSmooth));
    PetscCall(DMCreateLocalVector (dmInversion, &XPostSmooth));
  }

  InversionContext ctx = {
    .iparams            = iparams,
    .dm                 = dm,
    .dmConductivity     = dmConductivity,
    .dmInversion        = dmInversion,
    .grid               = *grid,
    .conductivity       = conductivity,
    .X0                 = X0,
    .DfDm               = DfDm,
    .dObs               = dObs,
    .Wweights           = NULL,   /* superseded by per-freq Wf_per_freq[] */
    .Q                  = &Q,
    .graph              = &graph,
    .notFixedMaskGlobal = notFixedMaskGlobal,
    .notFixedMaskLocal  = notFixedMaskLocal,
    .DfDmRaw            = DfDmRaw,
    .DfDmFinal          = DfDmFinal,
    .XPreSmooth         = XPreSmooth,
    .XPostSmooth        = XPostSmooth,
    .allRMS             = allRMS,
    .lastRMS            = PETSC_INFINITY,
    .lastDataMisfit     = 0.0,
    .lastRegTerm        = 0.0,
    .iterCount          = 0,
    .acceptedIter       = 0,
    /* -inv_no_smoother disables both smoothers for the whole run (diagnostic
     * for the nord>=2 / multi-rank boundary-artifact investigation). The FD
     * check still toggles bypassSmoother around its own evaluations. */
    .bypassSmoother     = iparams->smootherOff,
    /* Workspace fields (quad3d, MeRows, KeRows, b/x/nB/nx/Ex_recv,
     * Bvec_per_freq, Wf_per_freq, dObsRow_per_freq) are zero-initialized
     * by C designated-init and populated by setupInversionWorkspace next. */
  };

  /* Precompute everything that doesn't depend on the L-BFGS iterate X. */
  PetscCall(setupInversionWorkspace(&ctx));

  /* ---- Optional: FD gradient check at iter 0, then exit ---- */
  if (iparams->fdCheckCells > 0) {
    PetscReal fdEps = 1.0e-3;
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-inv_dev_fd_check_eps",
                                  &fdEps, NULL));
    PetscCall(runFdGradientCheck(&ctx, X, iparams->fdCheckCells, fdEps));
    PetscCall(PetscPrintf(comm,
      " FD check complete; skipping L-BFGS "
      "(remove -inv_dev_fd_check to run the inversion).\n"));
    goto fd_cleanup;
  }

  /* ---- Run L-BFGS optimization ----
   * PETSc TAO is unavailable with --with-scalar-type=complex (all TAO
   * solver registrations are guarded by #if !defined(PETSC_USE_COMPLEX)).
   * We use a custom L-BFGS implementation matching the MATLAB Fortran
   * reference (Nocedal 1980 two-loop recursion).                     */
  PetscCall(PetscPrintf(comm, "\n L-BFGS inversion started"
    " (M=%" PetscInt_FMT ", maxIter=%" PetscInt_FMT ")\n",
    iparams->lbfgsMemory, iparams->maxIter));
  if (iparams->smootherOff)
    PetscCall(PetscPrintf(comm,
      "   [diagnostic] -inv_no_smoother: BOTH smoothers disabled this run\n"));

  PetscInt    numIters;
  const char *reasonStr;
  PetscCall(lbfgsOptimize(inversionObjGrad, &ctx,
                          X, iparams->lbfgsMemory, iparams->maxIter,
                          iparams->gtol,
                          &ctx.lastRMS, iparams->rmsTol,
                          &numIters, &reasonStr));

  /* Print RMS history (one entry per objgrad call, including line searches) */
  PetscCall(PetscPrintf(comm, "\n RMS history (all objgrad evaluations):\n"));
  for (PetscInt it = 0; it < ctx.iterCount; it++)
    PetscCall(PetscPrintf(comm,
      "   eval %" PetscInt_FMT " : RMS = %g\n",
      it + 1, (double)allRMS[it]));

  /* ---- Write results to HDF5 ---- */
  PetscCall(writeInversionResults(iparams, dmConductivity,
                                  conductivity, X,
                                  allRMS, ctx.iterCount, reasonStr));

fd_cleanup:
  /* ---- Cleanup ---- */
  PetscCall(destroyInversionWorkspace(&ctx));
  PetscCall(VecDestroy(&X0));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&DfDm));
  PetscCall(VecDestroy(&notFixedMaskGlobal));
  PetscCall(VecDestroy(&notFixedMaskLocal));
  if (DfDmRaw)     PetscCall(VecDestroy(&DfDmRaw));
  if (DfDmFinal)   PetscCall(VecDestroy(&DfDmFinal));
  if (XPreSmooth)  PetscCall(VecDestroy(&XPreSmooth));
  if (XPostSmooth) PetscCall(VecDestroy(&XPostSmooth));
  PetscCall(DMDestroy(&dmInversion));
  PetscCall(MatDestroy(&dObs));
  PetscCall(PetscFree(allRMS));
  PetscCall(destroyNeighborGraph(&graph));
  PetscCall(destroyReceiverInterpolationMatrices(&Q));

  /* Hand the accumulated phase timers back to the caller (im.csem) so it
   * can report an Assembly/Solver breakdown consistent with fm.csem. */
  if (tAssemblyOut) *tAssemblyOut = ctx.tAssembly;
  if (tSolverOut)   *tSolverOut   = ctx.tSolver;

  PetscFunctionReturn(PETSC_SUCCESS);
}

