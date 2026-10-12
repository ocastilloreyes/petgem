/*
 * Filename: lbfgs.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Limited-Memory BFGS optimizer (Nocedal 1980) used by the CSEM
 * inverse kernel. References: Nocedal & Wright, "Numerical Optimization", Ch. 7.
 */

#include <petsc.h>

#include "common.h"
#include "lbfgs.h"

const char *const LbfgsReasons[] = {"CONVERGED_GRTOL", "CONVERGED_RMSTOL",    "CONVERGED_RMS_STALL",
                                    "DIVERGED_MAXITS", "DIVERGED_LS_FAILURE", "DIVERGED_NAN"};

/* Armijo sufficient-decrease parameter */
#define LBFGS_ARMIJO_C1 1e-4

/* Circular buffer of (s, y) pairs: s = X_{k+1} - X_k, y = G_{k+1} - G_k, rho = 1 / (y^T s). */
typedef struct {
  PetscInt   memory;
  PetscInt   bound; /* stored pairs (<= memory) */
  PetscInt   ptr;   /* next slot */
  Vec       *S, *Y;
  PetscReal *rho, *alpha;
} LbfgsHistory;

static PetscErrorCode createHistory(Vec X, PetscInt memory, LbfgsHistory *h) {
  PetscFunctionBeginUser;
  h->memory = memory;
  h->bound  = 0;
  h->ptr    = 0;
  PetscCall(VecDuplicateVecs(X, memory, &h->S));
  PetscCall(VecDuplicateVecs(X, memory, &h->Y));
  PetscCall(PetscMalloc2(memory, &h->rho, memory, &h->alpha));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroyHistory(LbfgsHistory *h) {
  PetscFunctionBeginUser;
  PetscCall(VecDestroyVecs(h->memory, &h->S));
  PetscCall(VecDestroyVecs(h->memory, &h->Y));
  PetscCall(PetscFree2(h->rho, h->alpha));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode realDot(Vec a, Vec b, PetscReal *v) {
  PetscFunctionBeginUser;
  PetscScalar dot;
  PetscCall(VecDot(a, b, &dot));
  *v = PetscRealPart(dot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* d = -H G by the two-loop recursion. */
static PetscErrorCode searchDirection(LbfgsHistory *h, Vec G, Vec d) {
  PetscFunctionBeginUser;
  const PetscInt M = h->memory;

  PetscCall(VecCopy(G, d));
  for (PetscInt i = 0; i < h->bound; i++) {
    const PetscInt idx = (h->ptr - 1 - i + M) % M;
    PetscReal      sd  = 0.0;
    PetscCall(realDot(h->S[idx], d, &sd));
    h->alpha[idx] = h->rho[idx] * sd;
    PetscCall(VecAXPY(d, -h->alpha[idx], h->Y[idx]));
  }
  if (h->bound > 0) {
    const PetscInt last = (h->ptr - 1 + M) % M;
    PetscReal      sy = 0.0, yy = 0.0;
    PetscCall(realDot(h->S[last], h->Y[last], &sy));
    PetscCall(realDot(h->Y[last], h->Y[last], &yy));
    PetscCall(VecScale(d, sy / yy));
  }
  for (PetscInt i = h->bound - 1; i >= 0; i--) {
    const PetscInt idx = (h->ptr - 1 - i + M) % M;
    PetscReal      yd  = 0.0;
    PetscCall(realDot(h->Y[idx], d, &yd));
    PetscCall(VecAXPY(d, h->alpha[idx] - h->rho[idx] * yd, h->S[idx]));
  }
  PetscCall(VecScale(d, -1.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Stores the accepted pair when it satisfies the curvature condition y^T s > 0. */
static PetscErrorCode updateHistory(LbfgsHistory *h, Vec X, Vec Xnew, Vec G, Vec Gnew, PetscReal ys, MPI_Comm comm) {
  PetscFunctionBeginUser;
  if (ys <= 1e-30) {
    PetscCall(PetscPrintf(comm, "   L-BFGS: skipping update (y^T s = %g); keeping history.\n", (double)ys));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecCopy(Xnew, h->S[h->ptr]));
  PetscCall(VecAXPY(h->S[h->ptr], -1.0, X));
  PetscCall(VecCopy(Gnew, h->Y[h->ptr]));
  PetscCall(VecAXPY(h->Y[h->ptr], -1.0, G));
  h->rho[h->ptr] = 1.0 / ys;
  h->ptr         = (h->ptr + 1) % h->memory;
  h->bound       = PetscMin(h->bound + 1, h->memory);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Backtracking Armijo search along d from (X, f) with slope gTd; *ok is false when it fails. */
static PetscErrorCode lineSearch(const LbfgsProblem *problem, Vec X, PetscReal f, Vec d, PetscReal gTd, PetscReal *stp, Vec Xnew,
                                 PetscReal *fnew, Vec Gnew, PetscReal *rms, PetscInt *numEvals, PetscBool *ok) {
  PetscFunctionBeginUser;
  *ok = PETSC_FALSE;
  for (PetscInt ls = 0; ls < LBFGS_MAX_LINE_SEARCH; ls++) {
    PetscCall(VecCopy(X, Xnew));
    PetscCall(VecAXPY(Xnew, *stp, d));
    PetscCall(problem->objgrad(problem->ctx, Xnew, fnew, Gnew, rms));
    (*numEvals)++;
    if (!PetscIsNanReal(*fnew) && *fnew <= f + LBFGS_ARMIJO_C1 * *stp * gTd) {
      *ok = PETSC_TRUE;
      break;
    }
    *stp *= 0.5;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gradientRatio(Vec G, Vec X, PetscReal *ratio) {
  PetscFunctionBeginUser;
  PetscReal gnorm, xnorm;
  PetscCall(VecNorm(G, NORM_2, &gnorm));
  PetscCall(VecNorm(X, NORM_2, &xnorm));
  *ratio = gnorm / PetscMax(1.0, xnorm);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode logExit(MPI_Comm comm, const LbfgsOptions *opts, const LbfgsResult *result, PetscReal rms, PetscReal gradRatio) {
  PetscFunctionBeginUser;
  char detail[128];

  switch (result->reason) {
  case LBFGS_CONVERGED_RMSTOL:
    PetscCall(PetscSNPrintf(detail, sizeof(detail), "RMS %.4f <= %s", (double)rms, formatCompactReal(opts->rmsTol)));
    break;
  case LBFGS_CONVERGED_GRTOL:
    PetscCall(PetscSNPrintf(detail, sizeof(detail), "|g|/|x| %.3e <= %s", (double)gradRatio, formatCompactReal(opts->gtol)));
    break;
  case LBFGS_CONVERGED_RMS_STALL:
    PetscCall(PetscSNPrintf(detail, sizeof(detail), "RMS drop < %s for %" PetscInt_FMT " iterations, RMS %.4f",
                            formatCompactReal(opts->rmsRelTol), opts->rmsStallWindow, (double)rms));
    break;
  case LBFGS_DIVERGED_LS_FAILURE:
    PetscCall(PetscSNPrintf(detail, sizeof(detail), "no sufficient decrease after %d trial steps", LBFGS_MAX_LINE_SEARCH));
    break;
  case LBFGS_DIVERGED_NAN:
    PetscCall(PetscStrncpy(detail, "NaN in objective or gradient", sizeof(detail)));
    break;
  case LBFGS_DIVERGED_MAXITS:
    PetscCall(PetscStrncpy(detail, "max iterations reached", sizeof(detail)));
    break;
  }
  PetscCall(PetscPrintf(comm, "\n"));
  PetscCall(logKvStr(comm, "Exit reason", LbfgsReasons[result->reason]));
  PetscCall(logKvStr(comm, "Stopping test", detail));
  PetscCall(logKvInt(comm, "Iterations", result->numIters));
  PetscCall(logKvInt(comm, "Evaluations", result->numEvals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode lbfgsOptimize(const LbfgsProblem *problem, const LbfgsOptions *opts, Vec X, LbfgsResult *result) {
  PetscFunctionBeginUser;

  MPI_Comm     comm = PetscObjectComm((PetscObject)X);
  LbfgsHistory hist;
  Vec          G, d, Xnew, Gnew;
  PetscReal    f, fnew, rms, prevRms, gnorm, gradRatio;
  PetscInt     stallCount = 0;

  PetscCall(VecDuplicate(X, &G));
  PetscCall(VecDuplicate(X, &d));
  PetscCall(VecDuplicate(X, &Xnew));
  PetscCall(VecDuplicate(X, &Gnew));
  PetscCall(createHistory(X, opts->memory, &hist));

  result->numIters = 0;
  result->numEvals = 1;
  result->reason   = LBFGS_DIVERGED_MAXITS;

  PetscCall(problem->objgrad(problem->ctx, X, &f, G, &rms));
  PetscCall(VecNorm(G, NORM_2, &gnorm));
  PetscCall(gradientRatio(G, X, &gradRatio));
  PetscCall(problem->monitor(problem->ctx, 0, gradRatio, 0.0));
  prevRms = rms;
  if (gradRatio <= opts->gtol) result->reason = LBFGS_CONVERGED_GRTOL;

  for (PetscInt iter = 0; iter < opts->maxIter && result->reason == LBFGS_DIVERGED_MAXITS; iter++) {
    PetscReal gTd = 0.0, gnewTd = 0.0, stp = 1.0;
    PetscBool ok;

    PetscCall(searchDirection(&hist, G, d));
    PetscCall(realDot(G, d, &gTd));
    if (gTd >= 0.0) {
      PetscCall(PetscPrintf(comm, "   L-BFGS: positive curvature detected; resetting to steepest descent.\n"));
      PetscCall(VecCopy(G, d));
      PetscCall(VecScale(d, -1.0));
      PetscCall(realDot(G, d, &gTd));
      hist.bound = 0;
      hist.ptr   = 0;
    }
    if (iter == 0 && hist.bound == 0) stp = 1.0 / gnorm;

    PetscCall(lineSearch(problem, X, f, d, gTd, &stp, Xnew, &fnew, Gnew, &rms, &result->numEvals, &ok));
    if (!ok) {
      result->reason   = LBFGS_DIVERGED_LS_FAILURE;
      result->numIters = iter + 1;
      break;
    }

    /* y^T s = stp (Gnew^T d - G^T d) */
    PetscCall(realDot(Gnew, d, &gnewTd));
    PetscCall(updateHistory(&hist, X, Xnew, G, Gnew, stp * (gnewTd - gTd), comm));
    PetscCall(VecCopy(Xnew, X));
    PetscCall(VecCopy(Gnew, G));
    f                = fnew;
    result->numIters = iter + 1;

    PetscCall(VecNorm(G, NORM_2, &gnorm));
    PetscCall(gradientRatio(G, X, &gradRatio));
    PetscCall(problem->monitor(problem->ctx, iter + 1, gradRatio, stp));

    if (PetscIsNanReal(gnorm) || PetscIsNanReal(f)) {
      result->reason = LBFGS_DIVERGED_NAN;
    } else if (gradRatio <= opts->gtol) {
      result->reason = LBFGS_CONVERGED_GRTOL;
    } else if (opts->rmsTol > 0.0 && rms <= opts->rmsTol) {
      result->reason = LBFGS_CONVERGED_RMSTOL;
    } else if (opts->rmsRelTol > 0.0) {
      const PetscReal drop = (prevRms - rms) / PetscMax(prevRms, PETSC_SMALL);
      stallCount           = (drop < opts->rmsRelTol) ? stallCount + 1 : 0;
      prevRms              = rms;
      if (stallCount >= opts->rmsStallWindow) result->reason = LBFGS_CONVERGED_RMS_STALL;
    }
  }

  PetscCall(logExit(comm, opts, result, rms, gradRatio));

  PetscCall(destroyHistory(&hist));
  PetscCall(VecDestroy(&G));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&Xnew));
  PetscCall(VecDestroy(&Gnew));
  PetscFunctionReturn(PETSC_SUCCESS);
}
