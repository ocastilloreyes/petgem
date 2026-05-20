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
/* ================================================================== */
PetscErrorCode computeGradientContribution(const invParams *iparams,
                                           const DM          dm,
                                           const Grid       *grid,
                                           const Vec         conductivity,
                                           const Vec         xLocal,
                                           const Vec         nxLocal,
                                           PetscScalar       constFactor,
                                           DM                dmInversion,
                                           Vec               DfDm)
{
  PetscFunctionBeginUser;

  /* Quadrature for elemental mass matrix */
  Quadrature3D quadrature_3d;
  Quadrature1D quadrature_1d;

  PetscCall(computeNum3DQuadraturePoints(iparams->nord, &quadrature_3d));
  PetscCall(computeNum1DQuadraturePoints(iparams->nord, &quadrature_1d));

  PetscCall(PetscCalloc1(quadrature_3d.numPoints, &quadrature_3d.points));
  PetscCall(PetscCalloc1(quadrature_1d.numPoints, &quadrature_1d.points));
  for (PetscInt i = 0; i < quadrature_3d.numPoints; i++)
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &quadrature_3d.points[i]));
  PetscCall(PetscCalloc1(quadrature_3d.numPoints, &quadrature_3d.weights));
  PetscCall(PetscCalloc1(quadrature_1d.numPoints, &quadrature_1d.weights));
  PetscCall(compute3DQuadraturePoints(&quadrature_3d));
  PetscCall(compute1DQuadraturePoints(&quadrature_1d));

  /* Allocate elemental matrices */
  PetscReal **Me, **Ke, **gradientMatrix;
  PetscCall(PetscCalloc1(grid->numDofInCell, &Me));
  PetscCall(PetscCalloc1(grid->numDofInCell, &Ke));
  PetscCall(PetscCalloc1(grid->numDofInCell, &gradientMatrix));
  PetscCall(PetscCalloc1(grid->numDofInCell * grid->numDofInCell, &Me[0]));
  PetscCall(PetscCalloc1(grid->numDofInCell * grid->numDofInCell, &Ke[0]));
  PetscCall(PetscCalloc1(grid->numDofInCell * grid->numH1DofInCell,
                         &gradientMatrix[0]));
  for (PetscInt i = 1; i < grid->numDofInCell; i++) {
    Me[i]            = Me[i - 1]            + grid->numDofInCell;
    Ke[i]            = Ke[i - 1]            + grid->numDofInCell;
    gradientMatrix[i] = gradientMatrix[i - 1] + grid->numH1DofInCell;
  }

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
    PetscCall(computeElementalMatrices(&grid->fem, &cell, &quadrature_3d, Me, Ke));

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

  /* Free quadrature memory */
  PetscCall(PetscFree(quadrature_3d.weights));
  PetscCall(PetscFree(quadrature_1d.weights));
  for (PetscInt i = 0; i < quadrature_3d.numPoints; i++)
    PetscCall(PetscFree(quadrature_3d.points[i]));
  PetscCall(PetscFree(quadrature_3d.points));
  PetscCall(PetscFree(quadrature_1d.points));

  PetscCall(PetscFree(Me[0]));
  PetscCall(PetscFree(Ke[0]));
  PetscCall(PetscFree(gradientMatrix[0]));
  PetscCall(PetscFree(Me));
  PetscCall(PetscFree(Ke));
  PetscCall(PetscFree(gradientMatrix));

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
  PetscReal         errorLevel   = c->iparams->errorLevel;

  PetscFunctionBeginUser;

  c->iterCount++;
  if (c->iparams->verbose)
    PetscCall(PetscPrintf(comm,
      "\n Inversion iteration %" PetscInt_FMT ":\n", c->iterCount));

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

  /* ---- 2. Assemble K (stiffness) and M(σ) (mass) once per iteration.
   *         These are frequency-independent; only the scalar iωμ changes
   *         per frequency. assembleCsemKandM is the unified forward/
   *         inverse assembly function (forward kernel uses the same
   *         call site in fm_csem.c). */
  csemParams fwdParams;
  PetscCall(PetscMemzero(&fwdParams, sizeof(fwdParams)));
  fwdParams.nord = c->iparams->nord;
  PetscCallMPI(MPI_Comm_size(comm, &fwdParams.numMPITasks));
  /* Suppress per-call assembly headers in the L-BFGS loop unless the user
   * asked for verbose output; otherwise every iteration spams the log
   * with "Assembly RHS / Vector size / Initiated / Finished" × Nfreq. */
  fwdParams.quiet = c->iparams->verbose ? PETSC_FALSE : PETSC_TRUE;

  /* Inverse kernel needs K and Ms separately (forms A_f per frequency
   * via MatDuplicate + MatAXPY inside the frequency loop below). Pass
   * a non-NULL Ms pointer to select K/Ms mode; constFactor is unused
   * in that mode. The canonical Π^Ned G is skipped (NULL); only the
   * BDDC structural-hint gradient (G_BDDC) is built. */
  Mat Kmat = NULL, Msmat = NULL, Gmat = NULL;
  PetscCall(assembleCsemKandM(fwdParams, c->dm, c->grid,
                              c->conductivity,
                              0.0,            /* constFactor (unused in K/Ms mode) */
                              &Kmat, &Msmat,
                              NULL /* canonical G — skip */,
                              &Gmat /* G_BDDC for PCBDDC */));

  /* ---- 3. Zero gradient accumulator ---- */
  PetscCall(VecZeroEntries(c->DfDm));

  /* ---- 4. Pre-allocate reusable Vecs (avoid create/destroy per freq) ---- */
  PetscReal reduil_fi  = 0.0;
  PetscInt  numData    = numReceivers * numFreqs * 2; /* real+imag */

  Vec b, x, nB, nx;
  PetscCall(DMCreateGlobalVector(c->dm, &b));
  PetscCall(DMCreateGlobalVector(c->dm, &x));
  PetscCall(DMCreateGlobalVector(c->dm, &nB));
  PetscCall(DMCreateGlobalVector(c->dm, &nx));

  Vec Ex_recv;
  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, numReceivers, &Ex_recv));
  Vec dObs_row;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, numReceivers, &dObs_row));
  Vec Wf;
  PetscCall(VecDuplicate(dObs_row, &Wf));

  /* ---- 5. Frequency loop ---- */
  for (PetscInt ifre = 0; ifre < numFreqs; ifre++) {
    const InvCsemSource *isrc = &c->iparams->invSources[ifre];

    PetscReal   omega = isrc->freq * 2.0 * PETSC_PI;
    PetscScalar Const = PETSC_i * omega * MU;

    if (c->iparams->verbose)
      PetscCall(PetscPrintf(comm,
        "   Frequency %" PetscInt_FMT " = %g Hz\n", ifre + 1,
        (double)isrc->freq));

    /* Form A_f = K - iωμ·Ms via MatDuplicate + MatAXPY.
     * SAME_NONZERO_PATTERN lets PETSc skip symbolic analysis. */
    Mat A;
    PetscCall(MatDuplicate(Kmat, MAT_COPY_VALUES, &A));
    PetscCall(MatAXPY(A, -Const, Msmat, SAME_NONZERO_PATTERN));

    /* Assemble RHS (source vector b) — still needs per-freq source info */
    CsemSource tmpSource;
    tmpSource.position[0]  = isrc->position[0];
    tmpSource.position[1]  = isrc->position[1];
    tmpSource.position[2]  = isrc->position[2];
    tmpSource.current      = isrc->current;
    tmpSource.length       = isrc->length;
    tmpSource.dipAngle     = isrc->dipAngle;
    tmpSource.azimuthAngle = isrc->azimuthAngle;

    CsemSourceSet tmpSrc;
    tmpSrc.freq        = isrc->freq;
    tmpSrc.numSources  = 1;
    tmpSrc.sourceArray = &tmpSource;

    Mat Bmat = NULL;
    PetscCall(assembleCsemRHS(fwdParams, tmpSrc,
                               c->dm, c->grid, &Bmat));

    /* Extract single source column from Bmat into Vec b */
    {
      Vec bcol;
      PetscCall(MatDenseGetColumnVecRead(Bmat, 0, &bcol));
      PetscCall(VecCopy(bcol, b));
      PetscCall(MatDenseRestoreColumnVecRead(Bmat, 0, &bcol));
    }

    /* Factorize and solve forward system: A_f * x = b */
    KSP ksp;
    PetscCall(createInvKSP(c->iparams, c->dm, A, Gmat, &ksp));
    PetscCall(solveInvSystem(ksp, b, x));

    /* Convert global x to local for field interpolation & gradient */
    Vec xLocal;
    PetscCall(DMGetLocalVector(c->dm, &xLocal));
    PetscCall(DMGlobalToLocal(c->dm, x, INSERT_VALUES, xLocal));

    /* Compute Ex at receivers: Ex_recv = QEx * x */
    PetscCall(MatMult(c->Q->QEx, x, Ex_recv));

    /* ---- Iter-0 per-frequency norms diagnostic (verbose only).
     * Useful for MPI-invariance debugging; norms are global reductions
     * so any rank-count drift localizes the bug.  See -inv_verbose. */
    PetscReal iter0_aFnorm = 0.0, iter0_bNorm = 0.0, iter0_xNorm = 0.0,
              iter0_exNorm = 0.0;
    if (c->iterCount == 1 && c->iparams->verbose) {
      PetscCall(MatNorm(A,       NORM_FROBENIUS, &iter0_aFnorm));
      PetscCall(VecNorm(b,       NORM_2,         &iter0_bNorm));
      PetscCall(VecNorm(x,       NORM_2,         &iter0_xNorm));
      PetscCall(VecNorm(Ex_recv, NORM_2,         &iter0_exNorm));
    }

    /* Get observed Ex for this frequency */
    {
      const PetscScalar *arr;
      PetscCall(MatDenseGetArrayRead(c->dObs, &arr));
      PetscScalar *rArr;
      PetscCall(VecGetArray(dObs_row, &rArr));
      for (PetscInt r = 0; r < numReceivers; r++)
        rArr[r] = arr[ifre + numFreqs * r];
      PetscCall(VecRestoreArray(dObs_row, &rArr));
      PetscCall(MatDenseRestoreArrayRead(c->dObs, &arr));
    }

    /* ---- Iter-0 per-frequency norms diagnostic (verbose only).
     * Each row that differs across MPI rank counts localizes the bug:
     *   ||A||_F     diff -> K or M(sigma) assembly (shared forward code)
     *   ||b||_2     diff -> source RHS assembly
     *   ||x||_2     diff -> solver (unexpected with MUMPS + seq. analysis)
     *   ||Ex||_2    diff -> receiver interpolation Q_Ex
     *   ||dObs||_2  diff -> observed-data load                       */
    if (c->iterCount == 1 && c->iparams->verbose) {
      PetscReal dobsNorm;
      PetscCall(VecNorm(dObs_row, NORM_2, &dobsNorm));
      PetscCall(PetscPrintf(comm,
        "     [iter-0] ||A||_F = %.8e  ||b|| = %.8e  ||x|| = %.8e\n"
        "              ||Ex||  = %.8e  ||dObs|| = %.8e\n",
        (double)iter0_aFnorm, (double)iter0_bNorm, (double)iter0_xNorm,
        (double)iter0_exNorm, (double)dobsNorm));
    }

    /* Data weights: W_f = 1 / (|dObs_f| * errorLevel) */
    {
      const PetscScalar *dArr;
      PetscScalar       *wArr;
      PetscCall(VecGetArrayRead(dObs_row, &dArr));
      PetscCall(VecGetArray(Wf, &wArr));
      for (PetscInt r = 0; r < numReceivers; r++) {
        PetscReal absVal = PetscAbsScalar(dArr[r]);
        wArr[r] = (absVal > 0.0) ? 1.0 / (absVal * errorLevel) : 0.0;
      }
      PetscCall(VecRestoreArray(Wf, &wArr));
      PetscCall(VecRestoreArrayRead(dObs_row, &dArr));
    }

    /* Compute misfit and adjoint RHS */
    PetscInt exStart, exEnd;
    PetscCall(VecGetOwnershipRange(Ex_recv, &exStart, &exEnd));
    PetscInt locNRec = exEnd - exStart;

    const PetscScalar *exArr, *dArr, *wArr;
    PetscCall(VecGetArrayRead(Ex_recv, &exArr));
    PetscCall(VecGetArrayRead(dObs_row, &dArr));
    PetscCall(VecGetArrayRead(Wf, &wArr));

    Vec wcdtD_mpi;
    PetscCall(VecCreateMPI(comm, locNRec, numReceivers, &wcdtD_mpi));
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
    PetscCall(VecDestroy(&wcdtD_mpi));

    /* Adjoint solve: A_f * nx = nB  (reuse factorization from forward) */
    PetscCall(solveInvSystem(ksp, nB, nx));

    /* Local adjoint solution for gradient accumulation */
    Vec nxLocal;
    PetscCall(DMGetLocalVector(c->dm, &nxLocal));
    PetscCall(DMGlobalToLocal(c->dm, nx, INSERT_VALUES, nxLocal));

    /* Accumulate per-element gradient (1 DOF/cell) */
    PetscCall(computeGradientContribution(c->iparams, c->dm, &c->grid,
                                          c->conductivity,
                                          xLocal, nxLocal, Const,
                                          c->dmInversion, c->DfDm));

    /* Cleanup frequency-level objects (keep reusable Vecs alive) */
    PetscCall(DMRestoreLocalVector(c->dm, &xLocal));
    PetscCall(DMRestoreLocalVector(c->dm, &nxLocal));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&Bmat));
  } /* end frequency loop */

  /* Destroy iteration-level matrices and reusable Vecs */
  PetscCall(MatDestroy(&Kmat));
  PetscCall(MatDestroy(&Msmat));
  PetscCall(MatDestroy(&Gmat));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&nB));
  PetscCall(VecDestroy(&nx));
  PetscCall(VecDestroy(&Ex_recv));
  PetscCall(VecDestroy(&dObs_row));
  PetscCall(VecDestroy(&Wf));

  /* ---- 5. RMS ---- */
  /* RMS = sqrt( sum_freq sum_rec |W*(d_obs - E_x)|^2 / Ndata )
   * W_r   = 1 / (|d_obs_r| * errorLevel)   (amplitude-relative weight)
   * Ndata = numReceivers * numFreqs * 2     (factor 2: real + imag parts) */
  PetscReal dataMisfit = reduil_fi / (PetscReal)numData;
  PetscReal rms        = PetscSqrtReal(dataMisfit);
  if (c->allRMS) c->allRMS[c->iterCount - 1] = rms;
  if (c->iparams->verbose)
    PetscCall(PetscPrintf(comm,
      "   RMS  = %g"
      "  [Ndata = %" PetscInt_FMT " = %" PetscInt_FMT " rec * %" PetscInt_FMT
      " freq * 2]\n",
      (double)rms, numData, numReceivers, numFreqs));

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
  if (c->iparams->verbose)
    PetscCall(PetscPrintf(comm,
      "   F    = %g  (data = %g  reg = %g)\n",
      (double)*F, (double)dataMisfit, (double)regTerm));
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
   * DfDm is already 1 value per cell — apply smoothing directly.
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

  /* Ownership range for targeted VecSetValue — only the owner sets the
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
                                Vec               receivers)
{
  PetscFunctionBeginUser;

  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* Validate preconditions: loadCsemInputs returns NULL for conductivity
   * and materialsID only if it was called with an empty inputFile, which
   * readCsemParams already errors out on. Re-check here for safety. */
  PetscCheck(conductivity, comm, PETSC_ERR_ARG_NULL,
             "conductivity Vec is NULL — loadCsemInputs did not populate "
             "the model. Check that -input_filename is valid.");
  PetscCheck(materialsID, comm, PETSC_ERR_ARG_NULL,
             "materialsID Vec is NULL — loadCsemInputs did not populate "
             "the model. Check that -input_filename is valid.");
  PetscCheck(receivers, comm, PETSC_ERR_ARG_NULL,
             "receivers Vec is NULL — loadCsemInputs did not return /receivers.");

  /* ---- Build neighbor smoothing graph ---- */
  NeighborGraph graph;
  PetscCall(buildNeighborSmoothingGraph(dm, grid, iparams, materialsID, &graph));

  /* ---- Build receiver Q matrices ---- */
  ReceiverInterpolationMatrices Q;
  PetscCall(buildReceiverInterpolationMatrices(iparams->nord,
                                                receivers,
                                                dm, grid,
                                                iparams->verbose, &Q));

  /* ---- Load observed data ---- */
  Mat dObs;
  PetscCall(loadObservedData(iparams, Q.numReceivers, &dObs));

  /* ---- Get conductivity DM (3 DOF/cell) for scatter operations ---- */
  DM dmConductivity;
  PetscCall(VecGetDM(conductivity, &dmConductivity));

  /* ---- Create inversion DM (1 DOF/cell) for X, X0, gradient ---- */
  DM dmInversion;
  PetscCall(createInversionDM(dmConductivity, grid, &dmInversion));

  /* ---- Build the parallel block-Jacobi smoother graph (multi-rank only).
   * No-op on a single rank; on >1 ranks, switches applyGaussSeidelSmoothing
   * to a fully-parallel forward+reverse Gauss-Seidel path that uses one
   * layer of ghost cells (overlap=1) and exchanges them between sweeps —
   * O(local cells) per call, no rank-0 bottleneck. Eliminates the partition-
   * boundary seams that the original partition-local sweep produced.
   *
   * Result is mathematically NOT bit-identical to the sequential rank-0
   * path (any fully-parallel GS variant differs in summation order / use
   * of one-step-stale ghost values), but the recovered model matches —
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

  /* Diagnostic snapshot buffers — allocated only when VTU snapshots
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
    .Wweights           = NULL,   /* weights computed per-frequency in callback */
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
    .bypassSmoother     = PETSC_FALSE,
  };

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

  PetscFunctionReturn(PETSC_SUCCESS);
}

