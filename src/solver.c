/*
 * Filename: solver.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Linear solves of the Maxwell system: KSP setup, PCBDDC with the discrete
 * gradient, and the optional BDDC primal vertices.
 */

/* PETSc libraries */
#include <petscdmplex.h>
#include <petscksp.h>
#include <petscsys.h>

/* PETGEM functions */
#include "assembly.h"
#include "common.h"
#include "io.h"
#include "solver.h"

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
 * assembled in assembleMaxwellOperator(). Its rows correspond to H(curl) DOFs and
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
 * @note
 *     The local field supplied to PCBDDCSetDofsSplittingLocal() spans all
 *     local DOFs of the MATIS local matrix, which disables the automatic
 *     field decomposition obtained from DMPlex.
 */
PetscErrorCode setupBddcFromPetgemGradient(KSP ksp, Mat A, Mat G, PetscInt order, IS primalVertices) {
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
    /* Pin a high-conductivity channel to the coarse space. Only its interface
     * DOFs are passed (filtered in buildHighSigmaPrimalIs); the whole channel
     * makes PCSetUp impractically slow. */
    if (primalVertices) {
      /* The list is in GLOBAL numbering, so PCBDDC maps it to its own local
       * space itself (PCBDDCGlobalToLocal). Passing local indices taken from
       * the DMPlex section does not work: that section is larger than the MATIS
       * local matrix, so the indices overflow it or name unrelated dofs. */
      /* ISGetSize sums the local sizes, and a shared dof is listed by every
       * rank that sees it, so np OVER-COUNTS the set by a factor near two.
       * PCBDDC deduplicates; the honest figure is printed by
       * buildHighSigmaPrimalIs as "distinct dofs". Labeled here as listed so
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
 * the forward kernels solve for all right-hand sides simultaneously. The
 * preconditioner is set up through setupBddcFromPetgemGradient: when @p A is
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
 * @note @p B must have a size and ordering consistent with @p A, and @p G (when
 *       used) must match the DOF ordering of @p A.
 */
PetscErrorCode solveMaxwellSystem(const DM dm, const Mat A, const Mat B, const Mat G, const PetscInt order, IS primalVertices, Mat *X,
                                  SolveInfo *info) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  KSP      ksp;
  PetscInt M, N, m, n;
  VecType  vtype;

  /* Create KSP object */
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* Setup solver and run it */
  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));

  /* G is the high-order discrete gradient (assembleMaxwellOperator); BDDC reads its sparsity to build the curl-kernel coarse space. */
  PetscCall(setupBddcFromPetgemGradient(ksp, A, G, order, primalVertices));
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
PetscErrorCode getSolverName(KSP ksp, char name[], size_t len) {
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

/* Cross-check a primal index set against the operator itself.
 *
 * buildHighSigmaPrimalIs emits indices taken from the DM's GLOBAL section, on
 * the assumption that they coincide with the global row numbering of the MATIS
 * operator handed to PCBDDC. If that assumption is wrong the set is not merely
 * incomplete, it is a list of the WRONG dofs, and PCBDDC would dutifully make
 * those primal - which looks like the fix under-performing rather than like a
 * numbering bug.
 *
 * The conductivity enters the operator through the mass term, so a dof inside a
 * conductor carries a diagonal larger than an ordinary one by roughly the
 * conductivity contrast. Comparing |diag| on the selected rows against |diag|
 * over all rows therefore tests the numbering with no reference to how the
 * indices were built. If the selected rows are not dramatically larger, the
 * indices are wrong.
 */
static PetscErrorCode verifyPrimalIsAgainstOperator(Mat A, IS primal) {
  PetscFunctionBeginUser;
  if (!primal) PetscFunctionReturn(PETSC_SUCCESS);

  Vec                d;
  const PetscInt    *ix;
  const PetscScalar *da;
  PetscInt           n, lo, hi, nloc, nin = 0;
  PetscReal          selMin = PETSC_MAX_REAL, selMax = 0.0, allMin = PETSC_MAX_REAL, allMax = 0.0;

  PetscCall(MatCreateVecs(A, NULL, &d));
  PetscCall(MatGetDiagonal(A, d));
  PetscCall(VecGetOwnershipRange(d, &lo, &hi));
  PetscCall(VecGetLocalSize(d, &nloc));
  PetscCall(VecGetArrayRead(d, &da));

  for (PetscInt i = 0; i < nloc; ++i) {
    const PetscReal a = PetscAbsScalar(da[i]);
    allMin            = PetscMin(allMin, a);
    allMax            = PetscMax(allMax, a);
  }
  PetscCall(ISGetLocalSize(primal, &n));
  PetscCall(ISGetIndices(primal, &ix));
  for (PetscInt i = 0; i < n; ++i) {
    if (ix[i] < lo || ix[i] >= hi) continue; /* listed by its owner elsewhere */
    const PetscReal a = PetscAbsScalar(da[ix[i] - lo]);
    selMin            = PetscMin(selMin, a);
    selMax            = PetscMax(selMax, a);
    ++nin;
  }
  PetscCall(ISRestoreIndices(primal, &ix));
  PetscCall(VecRestoreArrayRead(d, &da));

  PetscReal rmin[2] = {selMin, allMin}, rmax[2] = {selMax, allMax};
  PetscInt  cnt[2] = {nin, n};
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, rmin, 2, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)A)));
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, rmax, 2, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)A)));
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, cnt, 2, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)A)));

  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A),
                        "   BDDC primal check      = %" PetscInt_FMT " of %" PetscInt_FMT " indices fell in a local row range\n"
                        "                            |diag| selected  [%.3e, %.3e]\n"
                        "                            |diag| all rows  [%.3e, %.3e]\n",
                        cnt[0], cnt[1], (double)rmin[0], (double)rmax[0], (double)rmin[1], (double)rmax[1]));
  if (cnt[0] != cnt[1]) {
    /* Not an error: dofs owned by another rank fall outside this row range and
     * are checked by their owner. */
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A),
                          "                            %" PetscInt_FMT " listed by a rank that does not own"
                          " the row (checked by its owner)\n",
                          cnt[1] - cnt[0]));
  }
  if (cnt[0] > 0 && rmin[0] <= rmax[1] / 1.0e3) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "   WARNING: the smallest selected |diag| is not far above the general\n"
                                                           "            range, so the selection is probably not the conductor.\n"));
  }
  PetscCall(VecDestroy(&d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Collects the local DOFs supported on cells more conductive than a
 *        threshold, for use as PCBDDC primal vertices.
 *
 * BDDC is robust to coefficient jumps as long as those jumps do not cut the
 * subdomain interface. A high-conductivity filament threading the domain, such
 * as the steel casing of the cased-well benchmark, is cut by every partition,
 * and the iteration count then grows with the number of subdomains instead of
 * staying flat.
 *
 * Making the DOFs of the channel primal keeps them continuous across the
 * interface. PCBDDC discards the ones that are not on the interface, so every
 * DOF of every conductive cell may be passed; the number that survives is
 * reported by -ksp_view.
 *
 * Selection is by conductivity rather than by material id, so no knowledge of
 * the tag numbering is required: the steel sits at 2.4e6 S/m and the next most
 * conductive material at 1 S/m, so any threshold between the two picks out the
 * channel exactly.
 *
 * The returned indices are global, as required by PCBDDCSetPrimalVerticesIS,
 * which maps them to its own local space (PCBDDCGlobalToLocal). Local closure
 * indices taken from the DMPlex section must not be used: that section is
 * larger than the MATIS local matrix, so the indices either overflow it or
 * name unrelated DOFs.
 *
 * @param[in]  dm            DMPlex mesh carrying the H(curl) section.
 * @param[in]  conductivity  Per-cell conductivity Vec (its DM comes from VecGetDM).
 * @param[in]  threshold     Cells whose largest sigma exceeds this are selected.
 * @param[out] primal        IS of global DOF indices on the DM's communicator,
 *                           empty on ranks that hold no qualifying cells. Always
 *                           created when the threshold is positive, because the
 *                           creation is collective. The caller destroys it.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
static PetscErrorCode buildHighSigmaPrimalIs(const DM dm, const Vec conductivity, const PetscReal threshold, IS *primal) {
  PetscFunctionBeginUser;
  PetscAssertPointer(primal, 4);
  *primal = NULL;
  if (threshold <= 0.0) PetscFunctionReturn(PETSC_SUCCESS);

  DM           dmConductivity;
  PetscSection section, gsection;
  Cell         cell;
  PetscInt     cStart, cEnd, nsel = 0, cap = 0;
  PetscInt    *sel = NULL;
  /* Interface dofs of a high-sigma cell that this rank does not own. They are
   * skipped here on the assumption that the owning rank lists them - which
   * holds only if that rank also has a high-sigma cell on this dof. Where the
   * conductor BOUNDARY falls on the subdomain interface it does not, and the
   * dof is then listed by nobody. Counted so the size of that hole is visible
   * instead of assumed. */
  PetscInt nforeign = 0;

  PetscCall(VecGetDM(conductivity, &dmConductivity));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &gsection));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  /* share[i] = how many ranks hold local dof i. Ones in the local vector,
   * summed into the global one, scattered back: anything above 1 is shared,
   * i.e. sits on a subdomain interface. */
  Vec                gshare, lshare;
  const PetscScalar *share;
  PetscCall(DMCreateGlobalVector(dm, &gshare));
  PetscCall(DMCreateLocalVector(dm, &lshare));
  PetscCall(VecSet(lshare, 1.0));
  PetscCall(VecZeroEntries(gshare));
  PetscCall(DMLocalToGlobal(dm, lshare, ADD_VALUES, gshare));
  PetscCall(DMGlobalToLocal(dm, gshare, INSERT_VALUES, lshare));
  PetscCall(VecGetArrayRead(lshare, &share));

  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscCall(extractCellConductivity(dmConductivity, conductivity, c, &cell));
    PetscReal smax = cell.conductivity[0];
    for (PetscInt d = 1; d < NUM_DIMENSIONS; ++d) {
      if (cell.conductivity[d] > smax) smax = cell.conductivity[d];
    }
    if (smax <= threshold) continue;

    /* Both closures of the same cell, in the same order: the local one says
     * whether a dof is shared, the global one is what PCBDDC is given. */
    PetscInt  ndof, ndofl;
    PetscInt *idx, *lidx;
    PetscCall(DMPlexGetClosureIndices(dm, section, gsection, c, PETSC_TRUE, &ndof, &idx, NULL, NULL));
    PetscCall(DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE, &ndofl, &lidx, NULL, NULL));
    if (nsel + ndof > cap) {
      cap = PetscMax(2 * cap, nsel + ndof);
      PetscCall(PetscRealloc(cap * sizeof(PetscInt), &sel));
    }
    for (PetscInt j = 0; j < ndof && j < ndofl; ++j) {
      /* A global section gives negative indices for dofs this rank does not
       * own. Skip them: the owning rank lists them, and PCBDDC accepts any
       * process listing any global node. */
      const PetscInt  l      = lidx[j] < 0 ? -(lidx[j] + 1) : lidx[j];
      const PetscBool shared = (PetscBool)(PetscRealPart(share[l]) > 1.5);
      if (idx[j] < 0) {
        if (shared) ++nforeign;
        continue;
      }
      if (shared) sel[nsel++] = idx[j];
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE, &ndofl, &lidx, NULL, NULL));
    PetscCall(DMPlexRestoreClosureIndices(dm, section, gsection, c, PETSC_TRUE, &ndof, &idx, NULL, NULL));
  }

  PetscCall(VecRestoreArrayRead(lshare, &share));
  PetscCall(VecDestroy(&gshare));
  PetscCall(VecDestroy(&lshare));

  /* ISCreateGeneral on the DM's communicator is COLLECTIVE, so every rank must
   * call it - including the ones that hold no steel at all, which is most of
   * them once the selection is restricted to the interface. Guarding it with
   * `if (nsel)` deadlocked 896 ranks for nine hours on 23 September 2026. An
   * empty IS is legal and is what those ranks contribute. */
  PetscCall(PetscSortRemoveDupsInt(&nsel, sel));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), nsel, sel, PETSC_COPY_VALUES, primal));

  {
    /* COUNT DISTINCT DOFS, not listed ones. DMPlexGetClosureIndices with a
     * global section returns a non-negative index for a shared dof on EVERY
     * rank that sees it, not only on its owner, so the same global dof is
     * listed several times and summing local sizes over-counts it - by a factor
     * near two in practice. PCBDDC deduplicates internally, so the index set is
     * correct either way; it is the reported number that was wrong, and it was
     * wrong in a way that made the coarse space look as though it had absorbed
     * only half of what it was given.
     *
     * Marking a global vector with INSERT_VALUES and summing it counts each dof
     * once, whoever listed it, at the cost of one assembly. */
    Vec         mark;
    PetscInt    tot[2] = {nsel, nforeign};
    PetscScalar ndistinct;

    PetscCall(DMCreateGlobalVector(dm, &mark));
    PetscCall(VecZeroEntries(mark));
    for (PetscInt i = 0; i < nsel; ++i) {
      const PetscScalar one = 1.0;
      PetscCall(VecSetValues(mark, 1, &sel[i], &one, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(mark));
    PetscCall(VecAssemblyEnd(mark));
    PetscCall(VecSum(mark, &ndistinct));
    PetscCall(VecDestroy(&mark));

    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, tot, 2, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)dm)));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                          "   BDDC primal selection  = %" PetscInt_FMT " distinct dofs"
                          "  (%" PetscInt_FMT " listed, so %.2fx duplicated across ranks;"
                          " %" PetscInt_FMT " skipped as not owned)\n",
                          (PetscInt)PetscRealPart(ndistinct), tot[0],
                          tot[0] > 0 ? (double)tot[0] / PetscMax(PetscRealPart(ndistinct), 1.0) : 1.0, tot[1]));
  }
  PetscCall(PetscFree(sel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Builds the optional PCBDDC primal vertices from the options database.
 *
 * Off unless -petgem_bddc_primal_sigma <S/m> is given, in which case the DOFs
 * of the cells above that conductivity are selected with
 * buildHighSigmaPrimalIs. -petgem_bddc_primal_check verifies the selection
 * against the diagonal of @p A.
 */
PetscErrorCode setupPrimalVerticesFromOptions(const DM dm, const Mat A, const Vec conductivity, IS *primal) {
  PetscFunctionBeginUser;

  PetscReal primalSigma = 0.0;
  PetscBool checkPrimal = PETSC_FALSE;
  PetscInt  nlocal = 0, ntotal = 0;

  *primal = NULL;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-petgem_bddc_primal_sigma", &primalSigma, NULL));
  if (primalSigma <= 0.0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(buildHighSigmaPrimalIs(dm, conductivity, primalSigma, primal));
  if (*primal) PetscCall(ISGetLocalSize(*primal, &nlocal));
  PetscCallMPI(MPI_Reduce(&nlocal, &ntotal, 1, MPIU_INT, MPI_SUM, 0, PETSC_COMM_WORLD));
  PetscCall(logKvf(PETSC_COMM_WORLD, "BDDC primal candidates", "%s listed dofs on cells above %g S/m (see distinct count above)",
                   formatGroupedInt(ntotal), (double)primalSigma));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-petgem_bddc_primal_check", &checkPrimal, NULL));
  if (checkPrimal) PetscCall(verifyPrimalIsAgainstOperator(A, *primal));

  PetscFunctionReturn(PETSC_SUCCESS);
}
