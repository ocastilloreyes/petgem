/*
 * Filename: solver.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Linear solves of the Maxwell system: KSP setup, PCBDDC with the discrete
 * gradient, BDDC on the SPD auxiliary matrix with a gradient-space correction,
 * and the optional BDDC primal vertices.
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
    if (primalVertices) {
      PetscInt np, N;
      PetscCall(ISGetSize(primalVertices, &np));
      PetscCall(MatGetSize(A, &N, NULL));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A),
                            "  BDDC primal vertices   = %" PetscInt_FMT " listed of %" PetscInt_FMT " global dofs\n", np, N));
      PetscCall(PCBDDCSetPrimalVerticesIS(pc, primalVertices));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Re(A) + sgn Im(A), i.e. K + omega mu sigma M, as a MATIS copy of A. */
static PetscErrorCode buildSpdAuxiliary(Mat A, Mat *P) {
  PetscFunctionBeginUser;
  Mat                L;
  Vec                d;
  const PetscScalar *da;
  PetscScalar       *a;
  PetscInt           n;
  PetscReal          sgn = 0.0;
  MatInfo            info;

  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, P));
  PetscCall(MatISGetLocalMat(*P, &L));
  PetscCall(MatGetSize(L, &n, NULL));
  PetscCall(MatCreateVecs(L, &d, NULL));
  PetscCall(MatGetDiagonal(L, d));
  PetscCall(VecGetArrayRead(d, &da));
  for (PetscInt i = 0; i < n; i++) sgn += PetscImaginaryPart(da[i]);
  PetscCall(VecRestoreArrayRead(d, &da));
  PetscCall(VecDestroy(&d));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &sgn, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)A)));
  sgn = sgn >= 0.0 ? 1.0 : -1.0;
  PetscCall(MatGetInfo(L, MAT_LOCAL, &info));
  PetscCall(MatSeqAIJGetArray(L, &a));
  for (PetscInt k = 0; k < (PetscInt)info.nz_used; k++) a[k] = PetscRealPart(a[k]) + sgn * PetscImaginaryPart(a[k]);
  PetscCall(MatSeqAIJRestoreArray(L, &a));
  PetscCall(MatISRestoreLocalMat(*P, &L));
  PetscCall(MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* BDDC on the SPD auxiliary matrix with the gradient-space correction C = G S^-1 G^T, S = G^T P G. */
typedef struct {
  KSP bddc; /* preonly + PCBDDC on the BDDC matrix */
  KSP ks;   /* solver for S, prefix petgem_grad_ */
  Mat A, G, S;
  Vec r, w, t, z;
} GradCorrection;

/* y += C r */
static PetscErrorCode applyGradientSpace(GradCorrection *c, Vec r, Vec y) {
  PetscFunctionBeginUser;
  PetscCall(MatMultHermitianTranspose(c->G, r, c->t));
  PetscCall(KSPSolve(c->ks, c->t, c->z));
  PetscCall(MatMultAdd(c->G, c->z, y, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = C x; y += BDDC (x - A y); y += C (x - A y) */
static PetscErrorCode applyGradCorrection(PC pc, Vec x, Vec y) {
  PetscFunctionBeginUser;
  GradCorrection *c;
  PetscCall(PCShellGetContext(pc, &c));
  PetscCall(VecZeroEntries(y));
  PetscCall(applyGradientSpace(c, x, y));
  PetscCall(MatMult(c->A, y, c->r));
  PetscCall(VecAYPX(c->r, -1.0, x));
  PetscCall(KSPSolve(c->bddc, c->r, c->w));
  PetscCall(VecAXPY(y, 1.0, c->w));
  PetscCall(MatMult(c->A, y, c->r));
  PetscCall(VecAYPX(c->r, -1.0, x));
  PetscCall(applyGradientSpace(c, c->r, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode viewGradCorrection(PC pc, PetscViewer viewer) {
  PetscFunctionBeginUser;
  GradCorrection *c;
  PetscCall(PCShellGetContext(pc, &c));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  BDDC on the SPD auxiliary matrix:\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(KSPView(c->bddc, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Gradient-space solver:\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(KSPView(c->ks, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode destroyGradCorrection(PC pc) {
  PetscFunctionBeginUser;
  GradCorrection *c;
  PetscCall(PCShellGetContext(pc, &c));
  PetscCall(KSPDestroy(&c->bddc));
  PetscCall(KSPDestroy(&c->ks));
  PetscCall(MatDestroy(&c->S));
  PetscCall(VecDestroy(&c->r));
  PetscCall(VecDestroy(&c->w));
  PetscCall(VecDestroy(&c->t));
  PetscCall(VecDestroy(&c->z));
  PetscCall(PetscFree(c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* S = G^T P G; nodes that only touch constrained edges get a unit diagonal. */
static PetscErrorCode buildGradientSpaceMatrix(Mat P, Mat G, Mat *S) {
  PetscFunctionBeginUser;
  Mat          Paij;
  Vec          d;
  PetscScalar *dd;
  PetscInt     nl;

  PetscCall(MatConvert(P, MATAIJ, MAT_INITIAL_MATRIX, &Paij));
  PetscCall(MatPtAP(Paij, G, MAT_INITIAL_MATRIX, PETSC_DETERMINE, S));
  PetscCall(MatDestroy(&Paij));
  PetscCall(MatCreateVecs(*S, &d, NULL));
  PetscCall(MatGetDiagonal(*S, d));
  PetscCall(VecGetLocalSize(d, &nl));
  PetscCall(VecGetArray(d, &dd));
  for (PetscInt i = 0; i < nl; i++) dd[i] = PetscAbsScalar(dd[i]) > 0.0 ? 0.0 : 1.0;
  PetscCall(VecRestoreArray(d, &dd));
  PetscCall(MatSetOption(*S, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatDiagonalSet(*S, d, ADD_VALUES));
  PetscCall(VecDestroy(&d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PC of ksp: BDDC on the SPD part of Abddc, corrected in the gradient space of the SPD part of A. */
static PetscErrorCode setupGradCorrection(KSP ksp, Mat A, Mat Abddc, Mat G, PetscInt order, IS primalVertices) {
  PetscFunctionBeginUser;
  GradCorrection *c;
  PC              pc, ipc, gpc;
  Mat             P, Pb;
  PetscInt        ns;
  PetscBool       ismatis = PETSC_FALSE, flg;
  MPI_Comm        comm    = PetscObjectComm((PetscObject)A);

  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  PetscCheck(ismatis && G, comm, PETSC_ERR_ARG_WRONG, "-petgem_bddc_spd needs a MATIS operator and the discrete gradient");
  PetscCall(buildSpdAuxiliary(A, &P));
  if (Abddc == A) {
    PetscCall(PetscObjectReference((PetscObject)P));
    Pb = P;
  } else
    PetscCall(buildSpdAuxiliary(Abddc, &Pb));

  PetscCall(PetscNew(&c));
  c->A = A;
  c->G = G;

  /* BDDC with the adaptive coarse space (default threshold 1.2) */
  PetscCall(PetscOptionsHasName(NULL, NULL, "-pc_bddc_adaptive_threshold", &flg));
  if (!flg) PetscCall(PetscOptionsSetValue(NULL, "-pc_bddc_adaptive_threshold", "1.2"));
  PetscCall(KSPCreate(comm, &c->bddc));
  PetscCall(KSPSetOperators(c->bddc, Pb, Pb));
  PetscCall(setupBddcFromPetgemGradient(c->bddc, Pb, G, order, primalVertices));
  PetscCall(KSPSetType(c->bddc, KSPPREONLY));
  PetscCall(KSPGetPC(c->bddc, &ipc));
  PetscCall(PCSetFromOptions(ipc));

  /* Gradient-space solver, MUMPS LU by default */
  PetscCall(buildGradientSpaceMatrix(P, G, &c->S));
  PetscCall(MatGetSize(c->S, &ns, NULL));
  PetscCall(KSPCreate(comm, &c->ks));
  PetscCall(KSPSetOperators(c->ks, c->S, c->S));
  PetscCall(KSPSetOptionsPrefix(c->ks, "petgem_grad_"));
  PetscCall(KSPSetType(c->ks, KSPPREONLY));
  PetscCall(KSPGetPC(c->ks, &gpc));
  PetscCall(PCSetType(gpc, PCLU));
  PetscCall(PCFactorSetMatSolverType(gpc, MATSOLVERMUMPS));
  PetscCall(KSPSetFromOptions(c->ks));
  PetscCall(MatCreateVecs(c->S, &c->z, &c->t));
  PetscCall(MatCreateVecs(A, &c->r, &c->w));
  PetscCall(MatDestroy(&Pb));
  PetscCall(MatDestroy(&P));
  PetscCall(logKvf(comm, "Gradient correction", "%s nodal dofs", formatGroupedInt(ns)));

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, c));
  PetscCall(PCShellSetApply(pc, applyGradCorrection));
  PetscCall(PCShellSetDestroy(pc, destroyGradCorrection));
  PetscCall(PCShellSetView(pc, viewGradCorrection));
  PetscCall(PCShellSetName(pc, "bddc+gradient"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Solves the linear system A·X = B with PETSc KSP, all right-hand sides
 *        at once.
 *
 * Drives a single KSP over every column of @p B via KSPMatSolve. Without
 * @p Abddc the preconditioner is set up through setupBddcFromPetgemGradient:
 * when @p A is of type MATIS and @p G is supplied, PCBDDC is configured with the
 * exact order-p discrete gradient; otherwise PETSc's default PC is used. With
 * @p Abddc the preconditioner is PCBDDC on the SPD part of @p Abddc
 * (K + omega mu sigma M), with an adaptive coarse space, preceded and followed
 * by the gradient-space correction G (G^T P G)^-1 G^T, P the SPD part of @p A.
 * The solver of G^T P G takes the prefix petgem_grad_. The KSP type, tolerances
 * and PC settings stay overridable through the PETSc options database.
 *
 * @note @p B must have a size and ordering consistent with @p A, and @p G (when
 *       used) must match the DOF ordering of @p A.
 */
PetscErrorCode solveMaxwellSystem(const DM dm, const Mat A, const Mat Abddc, const Mat B, const Mat G, const PetscInt order,
                                  IS primalVertices, Mat *X, SolveInfo *info) {

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
  if (Abddc) {
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(setupGradCorrection(ksp, A, Abddc, G, order, primalVertices));
  } else {
    /* G is the high-order discrete gradient (assembleMaxwellOperator); BDDC reads its sparsity to build the curl-kernel coarse space. */
    PetscCall(setupBddcFromPetgemGradient(ksp, A, G, order, primalVertices));
    PetscCall(KSPSetFromOptions(ksp));
  }

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

/* Prints |diag(A)| on the primal rows against all rows; conductor rows carry a much larger diagonal. */
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
 * The DOFs on the subdomain interface of every cell whose largest
 * conductivity exceeds @p threshold. The indices are global, as
 * PCBDDCSetPrimalVerticesIS expects.
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
  /* Interface dofs of selected cells owned by another rank */
  PetscInt nforeign = 0;

  PetscCall(VecGetDM(conductivity, &dmConductivity));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &gsection));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  /* Number of ranks holding each local dof */
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

    /* Local closure for the sharing count, global closure for the indices */
    PetscInt  ndof, ndofl;
    PetscInt *idx, *lidx;
    PetscCall(DMPlexGetClosureIndices(dm, section, gsection, c, PETSC_TRUE, &ndof, &idx, NULL, NULL));
    PetscCall(DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE, &ndofl, &lidx, NULL, NULL));
    if (nsel + ndof > cap) {
      cap = PetscMax(2 * cap, nsel + ndof);
      PetscCall(PetscRealloc(cap * sizeof(PetscInt), &sel));
    }
    for (PetscInt j = 0; j < ndof && j < ndofl; ++j) {
      /* Negative global index: owned by another rank, which lists it */
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

  /* Collective: ranks without selected cells contribute an empty set */
  PetscCall(PetscSortRemoveDupsInt(&nsel, sel));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), nsel, sel, PETSC_COPY_VALUES, primal));

  {
    /* Distinct dofs: a shared dof is listed by every rank that holds it */
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
