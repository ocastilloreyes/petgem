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
#include "solver.h"
#include <petscdmplex.h>
#include <petscksp.h>
#include <petscsys.h>

/* PETGEM functions */

/**
 * @brief Configures PCBDDC for H(curl) systems using PETGEM's discrete gradient.
 *
 * Shared helper used by both the forward and inverse solvers. When the operator
 * matrix @p A is of type MATIS and a discrete-gradient matrix @p G is provided,
 * the routine:
 *
 *   1. Forces the preconditioner type to PCBDDC.
 *   2. Explicitly registers a single local field with
 *      PCBDDCSetDofsSplittingLocal().
 *   3. Registers the high-order discrete gradient through
 *      PCBDDCSetDiscreteGradient().
 *
 * The explicit field split is required to avoid PCBDDC falling back to
 * DMCreateFieldDecomposition() on the DM attached to the MATIS operator.
 * For high-order Nédélec spaces, DMPlex may generate field index sets whose
 * block size equals the number of DOFs associated with a mesh edge. PCBDDC
 * can interpret that block size as multiple fields, which leads to an
 * incorrect coarse-edge reconstruction in PCBDDCNedelecSupport() for
 * order >= 2. By prescribing a single field containing all local DOFs,
 * the decomposition becomes unambiguous and the discrete-gradient coarse
 * space is built correctly.
 *
 * The matrix @p G is PETGEM's exact high-order discrete gradient
 *
 *    G : H1(P_order) -> H(curl)(Nedelec_order)
 *
 * assembled in assembleCsemKandM(). Its rows correspond to H(curl) DOFs and
 * its columns to nodal H1 DOFs. PCBDDC uses the topology encoded in @p G,
 * together with the supplied Nédélec order, to identify the gradient kernel
 * and construct the associated coarse-space components.
 *
 * The routine is a no-op when either:
 *
 *   - @p A is not of type MATIS, or
 *   - @p G is NULL.
 *
 * In those cases the caller's preconditioner configuration is left unchanged.
 *
 * @param[in,out] ksp
 *     KSP whose preconditioner is configured.
 *
 * @param[in] A
 *     System matrix. The PCBDDC setup is applied only when @p A is a MATIS
 *     matrix.
 *
 * @param[in] G
 *     High-order discrete-gradient operator mapping nodal H1 DOFs to
 *     H(curl) DOFs. May be NULL.
 *
 * @param[in] order
 *     Polynomial order of the Nédélec space associated with @p G.
 *     Passed directly to PCBDDCSetDiscreteGradient().
 *
 * @return
 *     PETSC_SUCCESS on success, or a PETSc error code otherwise.
 *
 * @note
 *     The local field supplied to PCBDDCSetDofsSplittingLocal() spans all
 *     local DOFs of the MATIS local matrix. This intentionally disables the
 *     automatic field decomposition obtained from DMPlex and avoids the
 *     high-order field-identification issue described above.
 */
PetscErrorCode setupBDDCFromPetgemGradient(KSP ksp, Mat A, Mat G, PetscInt order, IS primalVertices)
{
  PetscFunctionBeginUser;
  PetscBool ismatis = PETSC_FALSE;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (ismatis && G) {
    PC       pc;
    Mat      lA;
    IS       allDofs;
    PetscInt nloc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCBDDC));
    PetscCall(MatISGetLocalMat(A, &lA));
    PetscCall(MatGetSize(lA, &nloc, NULL));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)A), nloc, 0, 1, &allDofs));
    PetscCall(PCBDDCSetDofsSplittingLocal(pc, 1, &allDofs));
    PetscCall(ISDestroy(&allDofs));
    PetscCall(PCBDDCSetDiscreteGradient(pc, G, order, 0, PETSC_TRUE, PETSC_TRUE));
    /* Pin a high-conductivity channel to the coarse space.
     *
     * DO NOT pass every DOF of the channel. An earlier version of this comment
     * claimed PCBDDC discards whatever is not on the subdomain interface, so
     * that passing the whole channel was free. It is not: handing it all 252947
     * steel dofs of the test3 model, 39 % of the mesh, left PCSetUp still
     * running after 52 minutes on a problem that solves in 118 s. The caller
     * filters to the interface itself, in buildHighSigmaPrimalIS. */
    if (primalVertices) {
      /* The list is in GLOBAL numbering, so PCBDDC maps it to its own local
       * space itself (PCBDDCGlobalToLocal). Passing local indices taken from
       * the DMPlex section does not work: that section is larger than the MATIS
       * local matrix, so the indices overflow it or name unrelated dofs. */
      /* ISGetSize sums the local sizes, and a shared dof is listed by every
       * rank that sees it, so np OVER-COUNTS the set by a factor near two.
       * PCBDDC deduplicates; the honest figure is printed by
       * buildHighSigmaPrimalIS as "distinct dofs". Labelled here as listed so
       * the two numbers cannot be mistaken for each other again. */
      PetscInt np, N;
      PetscCall(ISGetSize(primalVertices, &np));
      PetscCall(MatGetSize(A, &N, NULL));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A),
                            "  BDDC primal vertices   = %" PetscInt_FMT " listed (with cross-rank"
                            " duplicates) of %" PetscInt_FMT " global dofs\n",
                            np, N));
      PetscCall(PCBDDCSetPrimalVerticesIS(pc, primalVertices));
    }
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
 *   5. Solve all right-hand sides with KSPMatSolve.
 *   6. Record iterations, converged reason and solver name.
 *   7. Destroy the KSP.
 *
 * @param[in]  dm    DMPlex mesh; its communicator drives the parallel solve.
 * @param[in]  A     System matrix (H(curl) FEM operator).
 * @param[in]  B     Right-hand side matrix, one column per source.
 * @param[in]  G     High-order discrete-gradient hint for PCBDDC; may be NULL
 *                   when @p A is not of type MATIS.
 * @param[in]  order  Nedelec basis order (registered with the gradient in PCBDDC).
 * @param[in]  primalVertices  Optional PCBDDC primal vertices (may be NULL).
 * @param[out] X     Solution matrix, created internally (the caller destroys it).
 * @param[out] info  Iterations, converged reason and solver name.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 *
 * @note @p B must have a size and ordering consistent with @p A, and @p G (when
 *       used) must match the DOF ordering of @p A.
 */
PetscErrorCode solveCsemSystem(const DM dm, const Mat A, const Mat B, const Mat G, const PetscInt order, IS primalVertices, Mat* X, SolveInfo *info) {

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
  PetscCall(setupBDDCFromPetgemGradient(ksp, A, G, order, primalVertices));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(MatGetSize(B, &M, &N));
  PetscCall(MatGetLocalSize(B, &m, &n));
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatCreateDenseFromVecType(comm, vtype, m, n, M, N, m, NULL, X));
  PetscCall(KSPMatSolve(ksp, B, *X));

  PetscCall(KSPGetIterationNumber(ksp, &info->its));
  PetscCall(KSPGetConvergedReason(ksp, &info->reason));
  PetscCall(getSolverName(ksp, info->solver, sizeof(info->solver)));
  PetscCall(KSPDestroy(&ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Returns the solver name of a configured KSP: the PC type, or the
 *        factorization package for LU/Cholesky.
 *
 * @param[in]  ksp   KSP after KSPSetFromOptions().
 * @param[out] name  Solver name.
 * @param[in]  len   Size of @p name.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode getSolverName(KSP ksp, char name[], size_t len)
{
  PetscFunctionBeginUser;

  PC            pc;
  PCType        ptype;
  MatSolverType pkg = NULL;
  PetscBool     isfactor;

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCGetType(pc, &ptype));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)pc, &isfactor, PCLU, PCCHOLESKY, ""));
  if (isfactor) PetscCall(PCFactorGetMatSolverType(pc, &pkg));
  PetscCall(PetscStrncpy(name, pkg ? pkg : (ptype ? ptype : "none"), len));

  PetscFunctionReturn(PETSC_SUCCESS);
}
