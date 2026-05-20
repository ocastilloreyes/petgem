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
#include "inputs.h"
#include <petscdmplex.h>
#include <petscksp.h>
#include <petscsys.h>

/* PETGEM functions */

/**
 * @brief Solves the linear system AX = B using PETSc KSP.
 *
 * This function solves multiple linear systems (one per column of B) using
 * PETSc's KSP solver. If the system matrix A is of type MATIS and a discrete
 * gradient matrix G is provided, the solver configures a PCBDDC preconditioner
 * and sets the discrete gradient to improve convergence for H(curl) problems.
 *
 * @param[in] dm The PETSc DMPlex object representing the mesh. Its communicator
 *               is used for parallel solver setup.
 * @param[in] A The system matrix (Mat) assembled for the simulation. Should be
 *              compatible with the discretization (H(curl) FEM).
 * @param[in] B The right-hand side matrix (Mat), with one column per source.
 * @param[in] G Optional discrete gradient matrix (Mat). Required for MATIS matrices
 *              to set up the PCBDDC preconditioner correctly.
 * @param[out] X Pointer to the solution matrix (Mat) that will be created and
 *               populated with the solution vectors corresponding to each column
 *               of B.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful solve, or an appropriate
 *         PETSc error code otherwise.
 *
 * @details
 * The function performs the following steps:
 * 1. Creates a KSP solver object and sets A as both the operator and preconditioner matrix.
 * 2. Checks if A is of type MATIS:
 *    - If so and G is provided, configures the KSP preconditioner as PCBDDC.
 *    - Calls PCBDDCSetDiscreteGradient with G, FEM order, and default orientation settings.
 * 3. Reads solver options from the command line via KSPSetFromOptions.
 * 4. Creates a dense solution matrix X compatible with the vector type of A.
 * 5. Solves the system(s) using KSPMatSolve for all columns of B.
 * 6. Prints progress messages before and after solving.
 * 7. Destroys the KSP object and returns success.
 *
 * @note
 * - The function supports multiple right-hand sides (columns in B) efficiently.
 * - For MATIS matrices, the discrete gradient G is essential to enforce the
 *   kernel of the curl operator in H(curl) FEM.
 * - The solution matrix X is created internally; the caller is responsible for
 *   destroying it after use.
 * - Solver options (KSP type, tolerances, preconditioner settings, etc.) can
 *   be controlled via PETSc options database.
 *
 * @warning
 * - Ensure B has the correct size and ordering consistent with A.
 * - G must be compatible with the ordering of DOFs in A if MATIS/PCBDDC is used.
 */
PetscErrorCode solveCsemSystem(const DM dm, const Mat A, const Mat B, const Mat G, Mat* X) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  KSP ksp;
  PetscBool ismatis = PETSC_FALSE;
  PetscInt M, N, m, n;
  VecType vtype;

  /* Create KSP object */
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* Setup solver and run it */
  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));

  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));

  /* PCBDDC + Nédélec discrete-gradient operator at every supported order.
   *
   * G here is the TOPOLOGICAL gradient G_BDDC : Nédélec_k → P_nord H1
   * from assembleCsemKandM (lowest-Whitney vertex incidence per mesh
   * edge, higher-order edge / face / volume rows zero, K·G_BDDC ≠ 0
   * by design). PCBDDCSetDiscreteGradient is called with order = 1:
   * BDDC uses G as a structural hint to identify the curl-kernel
   * coarse space (∇P_1 ⊂ Nédélec_1 ⊂ Nédélec_k), and higher-order
   * H(curl) DOFs are static-condensed internally.
   *
   * The mathematically exact, cross-cell-consistent canonical Π^Ned
   * gradient against P_nord H1 is also produced by assembleCsemKandM
   * (the separate `G` matrix output) but is NOT passed to PCBDDC
   * because its denser face/volume couplings violate the edge-cluster
   * nnz budget BDDC checks during PCBDDCNedelecSupport ("SIZE OF EDGE
   * > EXTCOL SECOND PASS" at nord ≥ 3). The canonical G is used for
   * K·G analysis and any consumer that needs the actual algebraic
   * gradient. */
  if (ismatis && G) {
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCBDDC));
    PetscCall(PCBDDCSetDiscreteGradient(pc, G, 1, 0,
                                        PETSC_TRUE, PETSC_TRUE));
  }
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(MatGetSize(B, &M, &N));
  PetscCall(MatGetLocalSize(B, &m, &n));
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatCreateDenseFromVecType(comm, vtype, m, n, M, N, m, NULL, X));
  PetscCall(PetscPrintf(comm, "\n Solution of %" PetscInt_FMT " linear systems:\n", N));
  PetscCall(PetscPrintf(comm, "   Solver process    = Initiated\n"));

  PetscCall(KSPMatSolve(ksp, B, *X));

  PetscCall(PetscPrintf(comm, "   Solver process    = Finished\n"));
  PetscCall(KSPDestroy(&ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}
