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

#include "common.h"
#include "inversion.h"

/**
 * @brief Limited-Memory BFGS optimizer (Nocedal 1980).
 *
 * Implements the two-loop recursion for H·g approximation with a
 * backtracking Armijo line search. Designed to work with complex
 * PetscScalar (TAO is unavailable when PETSC_USE_COMPLEX is set). All
 * optimization variables are stored with zero imaginary part, so
 * VecDot/VecAXPY on them reduce to standard real operations.
 *
 * Two early-stop modes are supported when the caller maintains an RMS
 * value via `rmsPtr`:
 *   - Absolute: stop when *rmsPtr ≤ rmsTol (matches MATLAB's
 *     `rms <= 1.05` exit in Ex_inv.m).
 *   - Adaptive plateau: stop when the relative drop is below
 *     `-inv_rms_rtol` for `-inv_rms_stall_window` consecutive
 *     iterations (default 1e-3 / 3 iters).
 *
 * References: Nocedal & Wright, "Numerical Optimization", Ch. 7;
 *             MATLAB Fortran reference at petgem_inv_new/lbfgs_matlab/.
 *
 * @param[in]     objgrad    Objective/gradient callback.
 * @param[in]     ctx        Opaque context passed to objgrad (an
 *                           InversionContext * in current usage).
 * @param[in,out] X          Initial iterate; final iterate on return.
 * @param[in]     M          L-BFGS memory size.
 * @param[in]     maxIter    Maximum number of iterations.
 * @param[in]     gtol       Gradient-norm convergence tolerance.
 * @param[in]     rmsPtr     Optional pointer to a caller-updated RMS value.
 * @param[in]     rmsTol     RMS early-stop threshold (≤0 disables).
 * @param[out]    numIters   Number of iterations performed.
 * @param[out]    reasonStr  Human-readable convergence/stop reason.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
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

  /* L-BFGS memory: s[i] = X_{k+1} - X_k, y[i] = G_{k+1} - G_k.
   * VecDuplicateVecs creates the array of Vec handles AND the Vecs in a
   * single call - one round-trip instead of an explicit per-slot loop. */
  Vec *S, *Y;
  PetscReal *rho;       /* rho[i] = 1 / (y_i^T s_i) */
  PetscReal *alpha;     /* workspace for two-loop recursion */
  PetscCall(VecDuplicateVecs(X, M, &S));
  PetscCall(VecDuplicateVecs(X, M, &Y));
  PetscCall(PetscMalloc1(M, &rho));
  PetscCall(PetscMalloc1(M, &alpha));

  /* Evaluate initial objective and gradient */
  PetscReal f;
  PetscCall(objgrad(X, &f, G, ctx));

  PetscReal gnorm, xnorm;
  PetscCall(VecNorm(G, NORM_2, &gnorm));
  PetscCall(VecNorm(X, NORM_2, &xnorm));
  xnorm = PetscMax(1.0, xnorm);

  {
    InversionContext *ictx0 = (InversionContext *)ctx;
    /* Tabular trace: header row printed once, then one row per iteration
     * (including the initial iterate at iter 0 below).  Columns are
     * iter, RMS, F (objective), reg (Tikhonov term), ||g||, step length. */
    PetscCall(PetscPrintf(comm,
      "   %4s   %10s   %10s   %10s   %10s   %10s\n",
      "iter", "RMS", "F", "reg", "||g||", "step"));
    PetscCall(PetscPrintf(comm,
      "   %4d   %10.4f   %10.4e   %10.4e   %10.4e   %10s\n",
      0, (double)ictx0->lastRMS, (double)f,
      (double)ictx0->lastRegTerm, (double)gnorm, "-"));
  }

  /* Convergence check before iteration */
  if (gnorm / xnorm <= gtol) {
    *numIters  = 0;
    *reasonStr = "CONVERGED_GRTOL";
    goto cleanup;
  }

  /* Adaptive RMS-plateau stop: in addition to the absolute rmsTol, stop
   * when the data misfit stops improving (relative drop below rmsRelTol for
   * `rmsStallWindow` consecutive iterations). This is model-adaptive - it
   * derives the stopping point from the RMS history itself rather than a
   * fixed threshold - so a case that plateaus above rmsTol (e.g. order>=2
   * settling near RMS~1.15) stops once it has converged instead of grinding
   * to maxIter and overfitting. Disable with -inv_rms_rtol 0. */
  PetscReal rmsRelTol     = 1.0e-3;   /* <0.1% improvement/iter => plateau */
  PetscInt  rmsStallWindow = 3;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-inv_rms_rtol", &rmsRelTol, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-inv_rms_stall_window",
                               &rmsStallWindow, NULL));
  PetscReal prevRms     = (rmsPtr) ? *rmsPtr : PETSC_INFINITY;
  PetscInt  rmsStallCnt = 0;

  /* L-BFGS main loop */
  PetscInt bound = 0;   /* number of stored pairs (up to M) */
  PetscInt ptr   = 0;   /* circular buffer pointer */
  *reasonStr = "DIVERGED_MAXITS";

  for (PetscInt iter = 0; iter < maxIter; iter++) {

    /* ---- Two-loop recursion: compute d = -H_k * G ---- */
    PetscCall(VecCopy(G, d));     /* q = G */

    /* First loop (backward over stored pairs) */
    for (PetscInt j = 0; j < bound; j++) {
      PetscInt idx = (ptr - 1 - j + M) % M;
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
    for (PetscInt j = bound - 1; j >= 0; j--) {
      PetscInt idx = (ptr - 1 - j + M) % M;
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
        "   L-BFGS: positive curvature detected; resetting to steepest descent.\n"));
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
        "   L-BFGS: line search failed at iter %" PetscInt_FMT ".\n", iter + 1));
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
        "   L-BFGS: skipping update (y^T s = %g); keeping history.\n",
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
        "   %4" PetscInt_FMT "   %10.4f   %10.4e   %10.4e   %10.4e   %10.4e\n",
        iter + 1, (double)ictxIter->lastRMS, (double)f,
        (double)ictxIter->lastRegTerm, (double)gnorm, (double)stp));
    }

    if (PetscIsNanReal(gnorm) || PetscIsNanReal(f)) {
      PetscCall(PetscPrintf(comm,
        "   L-BFGS: NaN detected at iter %" PetscInt_FMT ".\n", iter + 1));
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

    /* Adaptive RMS-plateau exit: stop once the misfit stops improving for
     * `rmsStallWindow` consecutive iterations (relative drop < rmsRelTol).
     * Prevents over-iterating / overfitting when the absolute rmsTol is
     * unreachable for this model. */
    if (rmsPtr && rmsRelTol > 0.0) {
      PetscReal denom = PetscMax(prevRms, PETSC_SMALL);
      PetscReal rel   = (prevRms - *rmsPtr) / denom;   /* fractional drop */
      if (rel < rmsRelTol) rmsStallCnt++;
      else                 rmsStallCnt = 0;
      prevRms = *rmsPtr;
      if (rmsStallCnt >= rmsStallWindow) {
        *numIters  = iter + 1;
        *reasonStr = "CONVERGED_RMS_STALL";
        goto cleanup;
      }
    }
  }

  *numIters = maxIter;

cleanup:
  PetscCall(PetscPrintf(comm, "\n   %-24s = %s (%s iterations)\n",
                        "L-BFGS exit reason", *reasonStr, formatGroupedInt(*numIters)));
  PetscCall(VecDestroyVecs(M, &S));
  PetscCall(VecDestroyVecs(M, &Y));
  PetscCall(PetscFree(rho));
  PetscCall(PetscFree(alpha));
  PetscCall(VecDestroy(&G));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&Xnew));
  PetscCall(VecDestroy(&Gnew));

  PetscFunctionReturn(PETSC_SUCCESS);
}
