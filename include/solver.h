/*
 * Filename: solver.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Prototypes for the linear-system solver routines used by the
 * PETGEM kernels.
 */

#ifndef SOLVER_H
#define SOLVER_H

#include "io.h"
#include <petsc.h>
#include <petscdmplex.h>

/**
 * @brief Convergence summary of a linear solve, for the run report.
 */
typedef struct {
  PetscInt           its;     /**< KSP iterations (last right-hand side). */
  KSPConvergedReason reason;  /**< KSP converged reason (last right-hand side). */
  char               solver[32];  /**< PC type, or factorization package for LU/Cholesky. */
} SolveInfo;

/**
 * @brief Solves the linear system A·X = B using PETSc KSP.
 *
 * When A is a MATIS matrix and G is provided, the preconditioner is set to
 * PCBDDC and G is registered via PCBDDCSetDiscreteGradient at order = order to
 * capture the curl kernel for H(curl) problems. G is the high-order discrete
 * gradient G: Nédélec_order -> P_order H1 from assembleCsemKandM.
 *
 * @param[in]  dm    DMPlex mesh; its communicator drives the parallel solve.
 * @param[in]  A     System matrix (H(curl) FEM operator).
 * @param[in]  B     Right-hand side matrix, one column per source.
 * @param[in]  G     Discrete-gradient hint for PCBDDC; may be NULL when A is
 *                   not of type MATIS.
 * @param[in]  order  Nédélec basis order (registered with the gradient in PCBDDC).
 * @param[in]  primalVertices  Optional PCBDDC primal vertices (may be NULL).
 * @param[out] X     Solution matrix, created internally (caller destroys).
 * @param[out] info  Iterations, converged reason and solver name.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode solveCsemSystem(const DM dm, const Mat A, const Mat B, const Mat G,
                               const PetscInt order, IS primalVertices, Mat* X, SolveInfo *info);

/**
 * @brief Returns the solver name of a configured KSP: the PC type, or the
 *        factorization package for LU/Cholesky.
 *
 * @param[in]  ksp   KSP after KSPSetFromOptions().
 * @param[out] name  Solver name.
 * @param[in]  len   Size of @p name.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode getSolverName(KSP ksp, char name[], size_t len);

/**
 * @brief Configures `ksp`'s PC as PCBDDC with the discrete-gradient hint, if
 *        the operator is MATIS and a gradient was supplied.
 *
 * Shared helper that captures the single BDDC policy used by both the
 * forward solver (solveCsemSystem) and the inverse solver (setupForwardKSP):
 * when A is of type MATIS and Gbddc is non-NULL, set PC to PCBDDC and
 * register Gbddc (the high-order discrete gradient) via
 * PCBDDCSetDiscreteGradient(..., order, 0, PETSC_TRUE, PETSC_TRUE). When the
 * conditions are not met the function is a no-op so the caller's existing
 * default PC stays in place.
 *
 * @param[in,out] ksp   KSP whose preconditioner is configured.
 * @param[in]     A     System matrix (probed for MATIS).
 * @param[in]     G     Discrete-gradient hint (may be NULL).
 * @param[in]     order  Nédélec basis order registered with the gradient.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode setupBDDCFromPetgemGradient(KSP ksp, Mat A, Mat Gbddc, PetscInt order,
                                           IS primalVertices);

#endif
