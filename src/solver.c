/*
 * Filename: solver.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Linear-system solver routines for the PETGEM kernels.
 */

/* C libraries */

/* PETSc libraries */
#include "common.h"
#include "io.h"
#include <petscdmplex.h>
#include <petscksp.h>
#include <petscsys.h>

/* PETGEM functions */

/**
 * @brief Configures a KSP's preconditioner as PCBDDC with the discrete-gradient
 *        hint, when the operator is distributed (MATIS) and a gradient is given.
 *
 * Shared helper that captures the single PCBDDC + Nédélec discrete-gradient
 * policy used by BOTH kernels: the forward solver (solveCsemSystem) and the
 * inverse solver (createInvKSP). When @p A is of type MATIS and @p G is
 * non-NULL, the preconditioner is set to PCBDDC and @p G is registered via
 * PCBDDCSetDiscreteGradient(pc, G, 1, 0, PETSC_TRUE, PETSC_TRUE). When those
 * conditions are not met the call is a no-op, so the caller's default PC stays.
 *
 * @p G is the high-order discrete gradient G : Nédélec_order -->
 * P_order H1 produced by assembleCsemKandM (buildDiscreteGradientMatrix: each
 * H(curl) DOF's gradient is resolved against the full P_order H1 closure, so
 * grad(phi_k) = sum_i G_ik N_i exactly and K·G = 0). PCBDDC reads its
 * sparsity (GᵀG) to build the curl-kernel coarse space.
 *
 * The discrete gradient is the high-order operator G : Nedelec_order ->
 * P_order H1, so it is registered at order = order (matching the column space of
 * G); the coarse-space collapse is driven by G's sparsity/values together with
 * that order argument.
 *
 * @param[in,out] ksp    KSP whose preconditioner is configured.
 * @param[in]     A      System matrix (probed for the MATIS type).
 * @param[in]     G      Discrete-gradient hint; may be NULL.
 * @param[in]     order   Nedelec basis order registered with the gradient.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode setupBDDCFromPetgemGradient(KSP ksp, Mat A, Mat G, PetscInt order)
{
  PetscFunctionBeginUser;
  PetscBool ismatis = PETSC_FALSE;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (ismatis && G) {
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCBDDC));
    PetscCall(PCBDDCSetDiscreteGradient(pc, G, order, 0, PETSC_TRUE, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Solves the linear system A·X = B with PETSc KSP, all right-hand sides
 *        at once.
 *
 * Drives a single KSP over every column of @p B via KSPMatSolve, which is how
 * the forward kernel solves for all CSEM sources simultaneously. The
 * preconditioner is set up through setupBDDCFromPetgemGradient: when @p A is
 * of type MATIS and @p G is supplied, PCBDDC is configured with the exact
 * order-p discrete gradient to capture the curl kernel of the H(curl)
 * operator; otherwise PETSc's default PC is used. The KSP type, tolerances and
 * PC settings stay overridable through the PETSc options database
 * (KSPSetFromOptions).
 *
 * Steps:
 *   1. Create the KSP on @p dm's communicator and set @p A as both the operator
 *      and the preconditioner matrix.
 *   2. Configure PCBDDC from @p G (a no-op unless @p A is MATIS and @p G given).
 *   3. Read solver options with KSPSetFromOptions.
 *   4. Allocate the dense solution matrix @p X, matched to @p B's layout and
 *      @p A's vector type.
 *   5. Solve all right-hand sides with KSPMatSolve, reporting progress.
 *   6. Destroy the KSP.
 *
 * @param[in]  dm    DMPlex mesh; its communicator drives the parallel solve.
 * @param[in]  A     System matrix (H(curl) FEM operator).
 * @param[in]  B     Right-hand side matrix, one column per source.
 * @param[in]  G     High-order discrete-gradient hint for PCBDDC; may be NULL
 *                   when @p A is not of type MATIS.
 * @param[in]  order  Nedelec basis order (registered with the gradient in PCBDDC).
 * @param[out] X     Solution matrix, created internally (the caller destroys it).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 *
 * @note @p B must have a size and ordering consistent with @p A, and @p G (when
 *       used) must match the DOF ordering of @p A.
 */
PetscErrorCode solveCsemSystem(const DM dm, const Mat A, const Mat B, const Mat G, const PetscInt order, Mat* X) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  KSP ksp;
  PetscInt M, N, m, n;
  VecType vtype;

  /* Create KSP object */
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* Setup solver and run it */
  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));

  /* G is the high-order discrete gradient (assembleCsemKandM); BDDC reads its sparsity to build the curl-kernel coarse space. */
  PetscCall(setupBDDCFromPetgemGradient(ksp, A, G, order));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(MatGetSize(B, &M, &N));
  PetscCall(MatGetLocalSize(B, &m, &n));
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatCreateDenseFromVecType(comm, vtype, m, n, M, N, m, NULL, X));
  PetscCall(PetscPrintf(comm, "\n Linear solve:\n"));
  PetscCall(PetscPrintf(comm, "   %-24s = %s\n", "Number of systems", formatGroupedInt(N)));
  PetscCall(PetscPrintf(comm, "   %-24s = %s\n",                "Status",            "Started"));

  PetscCall(KSPMatSolve(ksp, B, *X));

  PetscCall(PetscPrintf(comm, "   %-24s = %s\n", "Status", "Finished"));
  PetscCall(KSPDestroy(&ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}
