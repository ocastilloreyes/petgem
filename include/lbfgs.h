/*
 * Filename: lbfgs.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Limited-Memory BFGS optimizer (Nocedal 1980) with a backtracking Armijo
 * line search and RMS-based stopping tests.
 */

#ifndef LBFGS_H
#define LBFGS_H

#include <petsc.h>

/** @brief Maximum backtracking steps per line search. */
#define LBFGS_MAX_LINE_SEARCH 20

/** @brief Why lbfgsOptimize stopped; LbfgsReasons[] holds the names. */
typedef enum {
  LBFGS_CONVERGED_GRTOL,     /**< ||g|| / max(1, ||x||) <= gtol. */
  LBFGS_CONVERGED_RMSTOL,    /**< RMS <= rmsTol. */
  LBFGS_CONVERGED_RMS_STALL, /**< Relative RMS drop < rmsRelTol for rmsStallWindow iterations. */
  LBFGS_DIVERGED_MAXITS,     /**< maxIter iterations performed. */
  LBFGS_DIVERGED_LS_FAILURE, /**< No sufficient decrease within LBFGS_MAX_LINE_SEARCH steps. */
  LBFGS_DIVERGED_NAN         /**< NaN in the objective or the gradient. */
} LbfgsReason;

/** @brief Names of the LbfgsReason values, e.g. "CONVERGED_GRTOL". */
extern const char *const LbfgsReasons[];

/** @brief Optimizer settings. */
typedef struct {
  PetscInt  memory;         /**< Number of stored (s, y) pairs. */
  PetscInt  maxIter;        /**< Maximum number of iterations. */
  PetscReal gtol;           /**< Relative gradient-norm tolerance. */
  PetscReal rmsTol;         /**< RMS stopping threshold (<= 0 disables). */
  PetscReal rmsRelTol;      /**< Relative RMS-drop threshold (<= 0 disables). */
  PetscInt  rmsStallWindow; /**< Stalled iterations before stopping. */
} LbfgsOptions;

/** @brief Problem callbacks; ctx is passed through unchanged. */
typedef struct {
  /** Objective f, gradient G and data RMS at X. */
  PetscErrorCode (*objgrad)(void *ctx, Vec X, PetscReal *f, Vec G, PetscReal *rms);
  /** Called after the initial evaluation (iter 0) and after every accepted step. */
  PetscErrorCode (*monitor)(void *ctx, PetscInt iter, PetscReal gnorm, PetscReal gradRatio, PetscReal step);
  void *ctx;
} LbfgsProblem;

/** @brief Outcome of lbfgsOptimize. */
typedef struct {
  LbfgsReason reason;   /**< Stop reason. */
  PetscInt    numIters; /**< Accepted iterations. */
  PetscInt    numEvals; /**< Objective-gradient evaluations. */
} LbfgsResult;

/**
 * @brief Minimizes problem->objgrad from X with L-BFGS.
 *
 * Two-loop recursion for the search direction, backtracking Armijo line
 * search, and stops on the gradient, RMS and RMS-plateau tests. Variables are
 * real-valued (zero imaginary part). Prints the exit summary.
 *
 * @param[in]     problem  Objective/gradient and monitor callbacks.
 * @param[in]     opts     Optimizer settings.
 * @param[in,out] X        Initial iterate; last accepted iterate on return.
 * @param[out]    result   Stop reason and counters.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode lbfgsOptimize(const LbfgsProblem *problem, const LbfgsOptions *opts, Vec X, LbfgsResult *result);

#endif /* LBFGS_H */
