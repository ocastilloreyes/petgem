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
 *
 * MPI reproducibility (important):
 *   The recovered model is reproducible across MPI task counts and mesh
 *   partitionings to CONVERGENCE TOLERANCE, not bitwise. Every cross-rank
 *   reduction in the pipeline - the misfit MPI_Allreduce here, and the
 *   VecDot/VecNorm reductions inside the L-BFGS driver - sums partial
 *   results in a rank-count-dependent order, and IEEE addition is non-
 *   associative, so the last bits of the objective and gradient differ
 *   between, say, 4 and 336 ranks. This perturbs the optimizer trajectory
 *   exactly like the already-accepted MUMPS factorization drift. The
 *   per-cell adjoint gradient and the Nedelec sign/ordering convention are
 *   partition-INDEPENDENT (topological); only the global scalar reductions
 *   and the Jacobi smoother carry the FP-order sensitivity. Do NOT read
 *   "partition-independent results" as "bit-identical across rank counts".
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
#include "solver.h"
#include "transmitter.h"

/**
 * @brief Allocates all iterate-independent inversion workspace.
 *
 * Lifetime hoist of every workspace that does not depend on the L-BFGS
 * iterate X. Called once before the optimization loop; freed by
 * destroyInversionWorkspace, replacing the per-callback allocate/free dance.
 *
 * What gets precomputed:
 *   - 3D quadrature points + weights (depend only on iparams->fm.nord);
 *   - Per-cell Me / Ke elemental-matrix buffers (numDof² each);
 *   - Reusable global Vecs b, x, nB, nx and parallel Ex_recv;
 *   - Per-frequency RHS Vec Bvec_per_freq[ifre]
 *     (assembleCsemRHS depends only on source + freq, NOT σ);
 *   - Per-frequency observed-Ex row Vec dObsRow_per_freq[ifre];
 *   - Per-frequency data weights Vec Wf_per_freq[ifre].
 *
 * All produced values are byte-identical to what the previous per-iteration
 * code computed; only the allocation lifetime changes.
 *
 * @param[in,out] ctx  Inversion context whose workspace is populated.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
static PetscErrorCode setupInversionWorkspace(InversionContext *ctx)
{
  PetscFunctionBeginUser;

  MPI_Comm  comm = PetscObjectComm((PetscObject)ctx->dm);
  PetscInt  numFreqs    = ctx->iparams->numFreqs;
  PetscInt  numDof      = ctx->grid.numDofInCell;
  PetscInt  numReceivers = ctx->Q->numReceivers;
  PetscReal errorLevel  = ctx->iparams->errorLevel;

  /* ---- 3D quadrature (depends only on nord) ---- */
  PetscCall(computeNum3DQuadraturePoints(ctx->iparams->fm.nord, &ctx->quad3d));
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

  /* Build a quiet fmParams stub for assembleCsemRHS (it only reads
   * nord, numMPITasks and quiet - same fields the iter loop used to fill). */
  fmParams stub;
  PetscCall(PetscMemzero(&stub, sizeof(stub)));
  stub.nord = ctx->iparams->fm.nord;
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
  fmParams kandmStub;
  PetscCall(PetscMemzero(&kandmStub, sizeof(kandmStub)));
  kandmStub.nord        = ctx->iparams->fm.nord;
  kandmStub.numMPITasks = stub.numMPITasks;
  kandmStub.quiet       = PETSC_TRUE;
  PetscCall(assembleCsemKandM(kandmStub, ctx->dm, ctx->grid,
                               ctx->conductivity,
                               0.0,             /* constFactor (unused: K/Ms mode) */
                               &ctx->Kmat, &ctx->Msmat,
                               &ctx->Gmat_BDDC));

  /* ---- Persistent per-frequency A_f and KSP (built once, reused) ----
   * A_f shares K's nonzero pattern; we allocate it without copying values
   * (the inversionObjGrad loop refills it via MatCopy(K)+MatAXPY(-Const·Ms)
   * every evaluation).  The KSP is bound to A_f here so the symbolic
   * factorization (MUMPS analysis) / PCBDDC topological setup is computed
   * once on the first solve and reused for every subsequent iteration -
   * only numeric refactorization is repeated when A_f's values change. */
  PetscCall(PetscCalloc1(numFreqs, &ctx->Avec_per_freq));
  PetscCall(PetscCalloc1(numFreqs, &ctx->ksp_per_freq));
  for (PetscInt ifre = 0; ifre < numFreqs; ifre++) {
    PetscCall(MatDuplicate(ctx->Kmat, MAT_DO_NOT_COPY_VALUES,
                           &ctx->Avec_per_freq[ifre]));
    PetscCall(createInvKSP(ctx->iparams, ctx->dm,
                           ctx->Avec_per_freq[ifre], ctx->Gmat_BDDC,
                           &ctx->ksp_per_freq[ifre]));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Tear-down counterpart to setupInversionWorkspace: free everything that
 * helper allocated on the context (quadrature, Me/Ke buffers, per-iter
 * reusable Vecs, per-frequency precomputes, and the cached K / Ms / G_BDDC
 * matrices).  Safe to call on a context that was zero-initialised but
 * never set up (every PetscFree/VecDestroy/MatDestroy handles NULL). */
/**
 * @brief Frees the workspace allocated by setupInversionWorkspace.
 *
 * @param[in,out] ctx  Inversion context whose workspace buffers are freed.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
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

  /* Per-frequency persistent solvers + system matrices (NULL-safe: the
   * arrays may be NULL if setup bailed before allocating them). Freed
   * before numFreqsAlloc is reset since the loop bounds depend on it. */
  if (ctx->ksp_per_freq) {
    for (PetscInt ifre = 0; ifre < ctx->numFreqsAlloc; ifre++)
      PetscCall(KSPDestroy(&ctx->ksp_per_freq[ifre]));
    PetscCall(PetscFree(ctx->ksp_per_freq));
  }
  if (ctx->Avec_per_freq) {
    for (PetscInt ifre = 0; ifre < ctx->numFreqsAlloc; ifre++)
      PetscCall(MatDestroy(&ctx->Avec_per_freq[ifre]));
    PetscCall(PetscFree(ctx->Avec_per_freq));
  }

  ctx->numFreqsAlloc = 0;

  PetscCall(MatDestroy(&ctx->Kmat));
  PetscCall(MatDestroy(&ctx->Msmat));
  PetscCall(MatDestroy(&ctx->Gmat_BDDC));

  PetscFunctionReturn(PETSC_SUCCESS);
}
/**
 * @brief Accumulates the per-element adjoint gradient into DfDm.
 *
 * For each local cell ie:
 *   idKdm = -2 · constFactor · Me_e,
 *   iG    = idKdm · x_e,
 *   DfDm[ie] += real( iG · nx_e )   (plain transpose, MATLAB iG.'*inx).
 * The 3D quadrature and Me/Ke buffers are provided by the caller (allocated
 * once on the InversionContext via setupInversionWorkspace); the hoist is a
 * pure lifetime change - the values produced by each cell evaluation are
 * byte-identical to the per-callback version.
 *
 * @param[in]     dm             H(curl) DM.
 * @param[in]     grid           Finite-element grid descriptor.
 * @param[in]     conductivity   Current conductivity Vec.
 * @param[in]     xLocal         Forward solution (local).
 * @param[in]     nxLocal        Adjoint solution (local).
 * @param[in]     constFactor    Frequency factor iωμ.
 * @param[in]     dmInversion    DM for the inversion field (1 DOF/cell).
 * @param[in,out] DfDm           Gradient accumulator (local, 1 DOF/cell).
 * @param[in]     quadrature_3d  3D quadrature workspace.
 * @param[in,out] Me             Scratch elemental mass buffer.
 * @param[in,out] Ke             Scratch elemental stiffness buffer.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
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

    /* Extract local solution values for forward (x_e) and adjoint (nx_e).
     * DMPlexVecGetClosure traverses the closure in the SAME order the
     * forward assembly used to fill Me (and Me already carries the per-DOF
     * orientation signs from computeElementalMatrices), so the quadratic
     * form below is sign- and ordering-consistent with fm.csem without any
     * separate closure-index lookup. */
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
  }

  PetscCall(VecAssemblyBegin(DfDm));
  PetscCall(VecAssemblyEnd(DfDm));

  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Creates a DM with 1 DOF per cell for the inversion variables.
 *
 * Clones the DMPlex topology from dmConductivity and installs a new
 * PetscSection with a single scalar DOF on each cell. A 1-component PetscFV
 * is registered as field 0 so cell-centered data can be written to VTU
 * (DMPlexVTKWriteAll_VTU traverses registered fields; without one, DMGetField
 * fails).
 *
 * @param[in]  dmConductivity  DM whose topology is cloned.
 * @param[in]  grid            Finite-element grid descriptor.
 * @param[out] dmInv           New 1-DOF/cell inversion DM.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
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

/**
 * @brief Creates a KSP for the inversion (same setup as solveCsemSystem, no solve).
 *
 * The caller invokes solveInvSystem twice (forward + adjoint) then KSPDestroy.
 * The MUMPS factorization is triggered on the first KSPSolve.
 *
 * @param[in]  iparams  Inversion parameters.
 * @param[in]  dm       H(curl) DM.
 * @param[in]  A        System matrix.
 * @param[in]  Gbddc    Topological discrete-gradient operator for PCBDDC.
 * @param[out] ksp      Created KSP bound to A.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode createInvKSP(const imParams *iparams,
                             const DM          dm,
                             const Mat         A,
                             const Mat         Gbddc,
                             KSP              *ksp)
{
  PetscFunctionBeginUser;

  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  PetscCall(KSPCreate(comm, ksp));
  PetscCall(KSPSetOperators(*ksp, A, A));

  /* `Gbddc` is the forward-formulation TOPOLOGICAL discrete-gradient operator
   * (Nédélec→H1 vertex incidence) consumed by PCBDDC. It has nothing to do
   * with the inversion gradient ∂F/∂X built by the L-BFGS layer — distinct
   * names so the two never get conflated. The inverse path passes
   * iparams->fm.nord as the order argument (preserves prior behaviour). */
  PetscCall(petgemConfigureBDDCFromGradient(*ksp, A, Gbddc, iparams->fm.nord));
  PetscCall(KSPSetFromOptions(*ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Solves A·sol = rhs using an already-created (factored) KSP.
 *
 * @param[in]  ksp  KSP previously created by createInvKSP.
 * @param[in]  rhs  Right-hand side vector.
 * @param[out] sol  Solution vector.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode solveInvSystem(const KSP ksp, const Vec rhs, Vec sol)
{
  PetscFunctionBeginUser;
  PetscCall(KSPSolve(ksp, rhs, sol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Objective + gradient callback for the L-BFGS optimizer.
 *
 * Per call:
 *   1. Recover σ from X (applyLogToSigma).
 *   2. Reassemble the LHS using the updated σ.
 *   3. Frequency loop: forward solve → field interpolation → residual →
 *      adjoint RHS → adjoint solve → gradient accumulation.
 *   4. Chain rule + smoothing + Tikhonov regularization.
 *
 * @param[in]  X     Current log-perturbation iterate.
 * @param[out] F     Objective value at X.
 * @param[out] Gvec  Gradient at X.
 * @param[in]  ctx   InversionContext pointer (cast from void*).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
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
   * before computing sigma.  diagGradientWeight=0 matches diag_gwight=0. */
  PetscCall(applyLogToSigma(c->dmInversion, c->dmConductivity,
                            X, c->X0, c->conductivity, &c->grid,
                            c->graph, 0.0, NULL));

  /* ---- 2. Refill Ms(σ) with the current iterate's σ ----
   * K (curl-curl stiffness, σ-independent) and G_BDDC (topological,
   * σ-independent) were built once in setupInversionWorkspace and live
   * on the context - they are reused for every L-BFGS iteration.
   * Only Ms needs to be recomputed when σ changes.
   *
   * assembleCsemMsRefill walks the local cells, shares the per-cell
   * setup helper (prepareCellForAssembly) with assembleCsemKandM, and
   * overwrites Ms in-place using the cached sparsity pattern. */
  fmParams fwdParams;
  PetscCall(PetscMemzero(&fwdParams, sizeof(fwdParams)));
  fwdParams.nord = c->iparams->fm.nord;
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

  /* ---- 5. Frequency loop ---- */
  for (PetscInt ifre = 0; ifre < numFreqs; ifre++) {
    const InvCsemSource *isrc = &c->iparams->invSources[ifre];

    PetscReal   omega = isrc->freq * 2.0 * PETSC_PI;
    PetscScalar Const = PETSC_i * omega * MU;

    /* Persistent per-frequency system matrix and solver (allocated once in
     * setupInversionWorkspace). A_f is refilled in place each evaluation:
     * A_f = K - iωμ·Ms.  SAME_NONZERO_PATTERN guarantees the symbolic
     * factorization / PCBDDC topological setup attached to ksp is preserved -
     * KSPSolve detects the matrix-value change and redoes ONLY the numeric
     * factorization. */
    Mat A   = c->Avec_per_freq[ifre];
    KSP ksp = c->ksp_per_freq[ifre];

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

    /* Solve forward system: A_f * x = b (numeric refactor on the cached
     * symbolic factorization, triggered by the in-place A_f value update). */
    PetscCall(PetscTime(&tA0));
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

    /* Cleanup frequency-level objects. The per-frequency A_f and ksp live
     * on the context (reused next evaluation); only the borrowed local
     * vectors are returned here. */
    PetscCall(DMRestoreLocalVector(c->dm, &xLocal));
    PetscCall(DMRestoreLocalVector(c->dm, &nxLocal));
  } /* end frequency loop */

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

  /* Zero local gradient at fixed elements via the persistent mask. */
  PetscCall(VecPointwiseMult(c->DfDm, c->DfDm, c->notFixedMaskLocal));

  /* ---- 8. Gradient smoothing (forward + reverse Gauss-Seidel) ----
   * DfDm is already 1 value per cell - apply smoothing directly. */
  PetscCall(applyGaussSeidelSmoothing(c->graph, c->iparams->diagGradientWeight,
                                      c->DfDm));

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

  /* Publish the most recent RMS so the optimizer loop can use it
   * for early-stopping, matching MATLAB's `rms <= 1.05` exit. */
  c->lastRMS = rms;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Top-level inversion driver.
 *
 * Builds the inversion DM, neighbor smoothing graph, workspace, and KSPs;
 * runs the custom L-BFGS optimizer over the per-frequency objective/gradient
 * callback; writes the final HDF5 results; and returns accumulated assembly
 * and solver wall-clock times for the kernel timer report. `receivers` is
 * the serial Vec produced by loadCsemInputs; it is consumed by
 * buildReceiverInterpolationMatrices and the caller retains ownership.
 *
 * @param[in]  iparams       Inversion parameters.
 * @param[in]  dm            H(curl) DM.
 * @param[in]  grid          Finite-element grid descriptor.
 * @param[in]  conductivity  Initial per-cell conductivity Vec.
 * @param[in]  materialsID   Per-cell material-id Vec.
 * @param[in]  receivers     Serial Vec of 3·N_recv receiver coordinates.
 * @param[out] tAssemblyOut  Accumulated assembly time (seconds).
 * @param[out] tSolverOut    Accumulated solver time (seconds).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode runCsemInversion(const imParams  *iparams,
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
  PetscCall(buildReceiverInterpolationMatrices(iparams->fm.nord,
                                                receivers,
                                                dm, grid, &Q));

  /* ---- Load observed data via the source-agnostic abstraction ----
   * Backend (external /observed/Ex vs fm-native /sources/src{k}/fields/Ex)
   * and file are selected from iparams (-inv_observed_mode / -inv_observed_file). */
  Mat dObs;
  PetscCall(loadObservedDataset(iparams, Q.numReceivers, &dObs));

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
    .Q                  = &Q,
    .graph              = &graph,
    .notFixedMaskGlobal = notFixedMaskGlobal,
    .notFixedMaskLocal  = notFixedMaskLocal,
    .allRMS             = allRMS,
    .lastRMS            = PETSC_INFINITY,
    .lastDataMisfit     = 0.0,
    .lastRegTerm        = 0.0,
    .iterCount          = 0,
    .acceptedIter       = 0,
    /* Workspace fields (quad3d, MeRows, KeRows, b/x/nB/nx/Ex_recv,
     * Bvec_per_freq, Wf_per_freq, dObsRow_per_freq) are zero-initialized
     * by C designated-init and populated by setupInversionWorkspace next. */
  };

  /* Precompute everything that doesn't depend on the L-BFGS iterate X. */
  PetscCall(setupInversionWorkspace(&ctx));

  /* ---- Run L-BFGS optimization ----
   * PETSc TAO is unavailable with --with-scalar-type=complex (all TAO
   * solver registrations are guarded by #if !defined(PETSC_USE_COMPLEX)).
   * We use a custom L-BFGS implementation matching the MATLAB Fortran
   * reference (Nocedal 1980 two-loop recursion).                     */
  PetscCall(PetscPrintf(comm, "\n L-BFGS optimization:\n"));
  PetscCall(PetscPrintf(comm, "   %-24s = %" PetscInt_FMT "\n", "L-BFGS memory (M)", iparams->lbfgsMemory));
  PetscCall(PetscPrintf(comm, "   %-24s = %" PetscInt_FMT "\n", "Max iterations",    iparams->maxIter));
  PetscCall(PetscPrintf(comm, "   %-24s = %s\n",                "Status",            "Started"));

  PetscInt    numIters;
  const char *reasonStr;
  PetscCall(lbfgsOptimize(inversionObjGrad, &ctx,
                          X, iparams->lbfgsMemory, iparams->maxIter,
                          iparams->gtol,
                          &ctx.lastRMS, iparams->rmsTol,
                          &numIters, &reasonStr));

  /* Print RMS history (one entry per objgrad call, including line searches) */
  PetscCall(PetscPrintf(comm, "\n RMS history (per objgrad evaluation):\n"));
  PetscCall(PetscPrintf(comm, "        eval         RMS\n"));
  for (PetscInt it = 0; it < ctx.iterCount; it++)
    PetscCall(PetscPrintf(comm,
      "   %9" PetscInt_FMT "   %.6g\n",
      it + 1, (double)allRMS[it]));

  /* ---- Write results to HDF5 ---- */
  PetscCall(writeInversionResults(iparams, dmConductivity,
                                  conductivity, X,
                                  allRMS, ctx.iterCount, reasonStr));

  /* ---- Cleanup ---- */
  PetscCall(destroyInversionWorkspace(&ctx));
  PetscCall(VecDestroy(&X0));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&DfDm));
  PetscCall(VecDestroy(&notFixedMaskGlobal));
  PetscCall(VecDestroy(&notFixedMaskLocal));
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

