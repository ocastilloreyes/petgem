/*
 * Filename: lbfgs.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Limited-Memory BFGS optimizer (Nocedal 1980) used by the CSEM
 * inverse kernel.
 */

/*
 * Notes:
 * PETSc TAO is unavailable when the library is built with
 * --with-scalar-type=complex (every TAO solver is #ifdef'd out under
 * PETSC_USE_COMPLEX), so we ship a custom implementation matching the
 * MATLAB Fortran reference.
 *
 * The optimizer is intentionally generic in its callback signature
 * (InversionObjGradFn) but currently couples to InversionContext for
 * two things: VTU snapshot dispatch on accepted steps and the one-line
 * iter summary that pulls RMS / regularization terms from the most
 * recent objective evaluation.
 *
 * Extracted from src/inversion.c during the refactor that split the
 * 2400-line monolith into focused translation units. Behavior preserved
 * byte-for-byte - only the file boundary moved.
 */

#include <math.h>

#include <petsc.h>

#include "inversion.h"

/* ================================================================== */
/* lbfgsOptimize                                                       */
/*                                                                     */
/* Limited-Memory BFGS optimizer (Nocedal 1980).                      */
/*                                                                     */
/* Implements the two-loop recursion for H*g approximation and a       */
/* backtracking Armijo line search.  Designed to work with complex     */
/* PetscScalar (TAO is unavailable when PETSC_USE_COMPLEX is set).    */
/*                                                                     */
/* All optimization variables are stored with zero imaginary part;     */
/* VecDot/VecAXPY on such data reduce to standard real operations.    */
/*                                                                     */
/* Reference: Nocedal & Wright, "Numerical Optimization", Ch. 7       */
/*            MATLAB Fortran reference: petgem_inv_new/lbfgs_matlab/   */
/* ================================================================== */
PetscErrorCode lbfgsOptimize(InversionObjGradFn objgrad, void *ctx,
                             Vec X, PetscInt M, PetscInt maxIter,
                             PetscReal gtol,
                             const PetscReal *rmsPtr, PetscReal rmsTol,
                             PetscInt *numIters, const char **reasonStr)
{
  PetscFunctionBeginUser;

  MPI_Comm comm = PetscObjectComm((PetscObject)X);

  /* Allocate work vectors */
  Vec G;          /* current gradient */
  Vec d;          /* search direction */
  Vec Xnew, Gnew; /* trial point and gradient */
  PetscCall(VecDuplicate(X, &G));
  PetscCall(VecDuplicate(X, &d));
  PetscCall(VecDuplicate(X, &Xnew));
  PetscCall(VecDuplicate(X, &Gnew));

  /* L-BFGS memory: s[i] = X_{k+1} - X_k, y[i] = G_{k+1} - G_k */
  Vec *S, *Y;
  PetscReal *rho;       /* rho[i] = 1 / (y_i^T s_i) */
  PetscReal *alpha;     /* workspace for two-loop recursion */
  PetscCall(PetscMalloc1(M, &S));
  PetscCall(PetscMalloc1(M, &Y));
  PetscCall(PetscMalloc1(M, &rho));
  PetscCall(PetscMalloc1(M, &alpha));
  for (PetscInt i = 0; i < M; i++) {
    PetscCall(VecDuplicate(X, &S[i]));
    PetscCall(VecDuplicate(X, &Y[i]));
  }

  /* Evaluate initial objective and gradient */
  PetscReal f;
  PetscCall(objgrad(X, &f, G, ctx));

  PetscReal gnorm, xnorm;
  PetscCall(VecNorm(G, NORM_2, &gnorm));
  PetscCall(VecNorm(X, NORM_2, &xnorm));
  xnorm = PetscMax(1.0, xnorm);

  {
    InversionContext *ictx0 = (InversionContext *)ctx;
    PetscCall(PetscPrintf(comm,
      "   Iter %3d : RMS = %8.4f  F = %10.4e  reg = %9.3e  ||g|| = %9.3e\n",
      0, (double)ictx0->lastRMS, (double)f,
      (double)ictx0->lastRegTerm, (double)gnorm));
  }

  /* Convergence check before iteration */
  if (gnorm / xnorm <= gtol) {
    *numIters  = 0;
    *reasonStr = "CONVERGED_GRTOL";
    goto cleanup;
  }

  /* L-BFGS main loop */
  PetscInt bound = 0;   /* number of stored pairs (up to M) */
  PetscInt ptr   = 0;   /* circular buffer pointer */
  *reasonStr = "DIVERGED_MAXITS";

  for (PetscInt iter = 0; iter < maxIter; iter++) {

    /* ---- Two-loop recursion: compute d = -H_k * G ---- */
    PetscCall(VecCopy(G, d));     /* q = G */

    /* First loop (backward over stored pairs) */
    for (PetscInt ii = 0; ii < bound; ii++) {
      PetscInt idx = (ptr - 1 - ii + M) % M;
      PetscScalar dotval;
      PetscCall(VecDot(S[idx], d, &dotval));
      alpha[idx] = rho[idx] * PetscRealPart(dotval);
      PetscCall(VecAXPY(d, -alpha[idx], Y[idx]));
    }

    /* Scale by gamma = s^T y / (y^T y) for the most recent pair */
    if (bound > 0) {
      PetscInt last = (ptr - 1 + M) % M;
      PetscScalar sys, yty;
      PetscCall(VecDot(S[last], Y[last], &sys));
      PetscCall(VecDot(Y[last], Y[last], &yty));
      PetscReal gamma = PetscRealPart(sys) / PetscRealPart(yty);
      PetscCall(VecScale(d, gamma));
    }

    /* Second loop (forward over stored pairs) */
    for (PetscInt ii = bound - 1; ii >= 0; ii--) {
      PetscInt idx = (ptr - 1 - ii + M) % M;
      PetscScalar dotval;
      PetscCall(VecDot(Y[idx], d, &dotval));
      PetscReal beta = rho[idx] * PetscRealPart(dotval);
      PetscCall(VecAXPY(d, alpha[idx] - beta, S[idx]));
    }

    /* d = -H*G (negate) */
    PetscCall(VecScale(d, -1.0));

    /* ---- Backtracking Armijo line search ---- */
    PetscScalar gTd_scalar;
    PetscCall(VecDot(G, d, &gTd_scalar));
    PetscReal gTd = PetscRealPart(gTd_scalar);

    if (gTd >= 0.0) {
      /* Not a descent direction - fall back to steepest descent */
      PetscCall(PetscPrintf(comm,
        "   L-BFGS: positive curvature detected, resetting to steepest descent\n"));
      PetscCall(VecCopy(G, d));
      PetscCall(VecScale(d, -1.0));
      PetscCall(VecDot(G, d, &gTd_scalar));
      gTd = PetscRealPart(gTd_scalar);
      bound = 0;
      ptr   = 0;
    }

    /* Initial step size: 1.0 for L-BFGS, 1/||G|| for first (steepest descent) */
    PetscReal stp = 1.0;
    if (iter == 0 && bound == 0) {
      stp = 1.0 / gnorm;
    }

    PetscReal fnew;
    PetscReal c1 = 1e-4;       /* Armijo sufficient decrease parameter */
    PetscInt  maxLs = 20;      /* max line search steps */
    PetscBool lsOk = PETSC_FALSE;

    for (PetscInt ls = 0; ls < maxLs; ls++) {
      /* Xnew = X + stp * d */
      PetscCall(VecCopy(X, Xnew));
      PetscCall(VecAXPY(Xnew, stp, d));

      PetscCall(objgrad(Xnew, &fnew, Gnew, ctx));

      if (!PetscIsNanReal(fnew) && fnew <= f + c1 * stp * gTd) {
        lsOk = PETSC_TRUE;
        break;
      }
      stp *= 0.5;
    }

    if (!lsOk) {
      PetscCall(PetscPrintf(comm,
        "   L-BFGS: line search failed at iter %" PetscInt_FMT "\n", iter + 1));
      *reasonStr = "DIVERGED_LS_FAILURE";
      *numIters  = iter + 1;
      goto cleanup;
    }

    /* ---- Store s_k = Xnew - X, y_k = Gnew - G ---- */
    /* Check curvature BEFORE writing to S[ptr]/Y[ptr], so that a
       rejected pair does not corrupt the circular buffer.
       ys = (Gnew - G)^T (Xnew - X) = stp * (Gnew^T d - G^T d) */
    PetscScalar gnewTd_scalar;
    PetscCall(VecDot(Gnew, d, &gnewTd_scalar));
    PetscReal ys = stp * (PetscRealPart(gnewTd_scalar) - gTd);

    if (ys > 1e-30) {
      PetscCall(VecCopy(Xnew, S[ptr]));
      PetscCall(VecAXPY(S[ptr], -1.0, X));    /* S[ptr] = Xnew - X */
      PetscCall(VecCopy(Gnew, Y[ptr]));
      PetscCall(VecAXPY(Y[ptr], -1.0, G));    /* Y[ptr] = Gnew - G */
      rho[ptr] = 1.0 / ys;
      ptr   = (ptr + 1) % M;
      bound = PetscMin(bound + 1, M);
    } else {
      /* Curvature condition not met - skip storing this pair but keep
       * existing history.  Resetting to steepest descent was too
       * aggressive: stale history is still better than no history. */
      PetscCall(PetscPrintf(comm,
        "   L-BFGS: skipping update (y^T s = %g), keeping history\n",
        (double)ys));
    }

    /* ---- Accept step ---- */
    PetscCall(VecCopy(Xnew, X));
    PetscCall(VecCopy(Gnew, G));
    f = fnew;

    /* ---- VTU snapshot (accepted steps only) ---- */
    {
      InversionContext *ictx = (InversionContext *)ctx;
      ictx->acceptedIter++;
      if (ictx->iparams->snapshotInterval > 0 &&
          ictx->acceptedIter % ictx->iparams->snapshotInterval == 0) {
        PetscCall(writeInversionSnapshotVTU(ictx, ictx->acceptedIter));
      }
    }

    /* ---- Convergence check ---- */
    PetscCall(VecNorm(G, NORM_2, &gnorm));
    PetscCall(VecNorm(X, NORM_2, &xnorm));
    xnorm = PetscMax(1.0, xnorm);

    {
      InversionContext *ictxIter = (InversionContext *)ctx;
      PetscCall(PetscPrintf(comm,
        "   Iter %3" PetscInt_FMT " : RMS = %8.4f  F = %10.4e  reg = %9.3e"
        "  ||g|| = %9.3e  step = %6.3g\n",
        iter + 1, (double)ictxIter->lastRMS, (double)f,
        (double)ictxIter->lastRegTerm, (double)gnorm, (double)stp));
    }

    if (PetscIsNanReal(gnorm) || PetscIsNanReal(f)) {
      PetscCall(PetscPrintf(comm,
        "   L-BFGS: NaN detected at iter %" PetscInt_FMT "\n", iter + 1));
      *reasonStr = "DIVERGED_NAN";
      *numIters  = iter + 1;
      goto cleanup;
    }

    if (gnorm / xnorm <= gtol) {
      *numIters  = iter + 1;
      *reasonStr = "CONVERGED_GRTOL";
      goto cleanup;
    }

    /* MATLAB-style RMS early exit: stop when the data misfit drops
     * to the noise-level threshold (e.g. 1.05 in Ex_inv.m). */
    if (rmsPtr && rmsTol > 0.0 && *rmsPtr <= rmsTol) {
      *numIters  = iter + 1;
      *reasonStr = "CONVERGED_RMSTOL";
      goto cleanup;
    }
  }

  *numIters = maxIter;

cleanup:
  PetscCall(PetscPrintf(comm, "\n   L-BFGS finished: %s  (%" PetscInt_FMT " iterations)\n",
                        *reasonStr, *numIters));
  for (PetscInt i = 0; i < M; i++) {
    PetscCall(VecDestroy(&S[i]));
    PetscCall(VecDestroy(&Y[i]));
  }
  PetscCall(PetscFree(S));
  PetscCall(PetscFree(Y));
  PetscCall(PetscFree(rho));
  PetscCall(PetscFree(alpha));
  PetscCall(VecDestroy(&G));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&Xnew));
  PetscCall(VecDestroy(&Gnew));

  PetscFunctionReturn(PETSC_SUCCESS);
}
