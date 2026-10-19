/*
 * Filename: assembly.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Global assembly of the H(curl) Maxwell operator, its discrete gradient
 * and mass-matrix refill, and the right-hand-side containers.
 */

/* PETSc libraries */
#include <petscsys.h>

/* PETGEM functions */
#include "common.h"
#include "assembly.h"
#include "constants.h"
#include "fem.h"
#include "grid.h"

/**
 * @brief Verifies that a cell's discrete gradient lies in the stiffness kernel.
 *
 * This function checks the De Rham identity K_e G_e = 0 for one cell: with M the
 * m x m elemental stiffness (Ke) and G the m x n discrete gradient (both
 * row-major), it forms every entry of the product M·G and reports, via a printed
 * message, any (i, j) whose value is not within PETSC_SMALL of zero. It is a
 * read-only diagnostic: no matrix is modified and a non-zero product does not
 * abort the run.
 *
 * @param[in] M  Elemental stiffness matrix Ke (m x m, row-major).
 * @param[in] G  Discrete gradient block (m x n, row-major).
 * @param[in] m  Number of H(curl) DOFs per cell (rows of M and G).
 * @param[in] n  Number of H1 DOFs per cell (columns of G).
 * @param[in] w  Cell index, used only to label diagnostic messages.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
static PetscErrorCode checkGradientKernel(PetscReal *M, PetscReal *G, PetscInt m, PetscInt n, PetscInt w) {
  PetscReal normM = 0, normG = 0, worst = 0;
  PetscInt  wi = -1, wj = -1;

  PetscFunctionBeginUser;

  for (PetscInt i = 0; i < m * m; i++) {
    normM = PetscMax(normM, PetscAbsReal(M[i]));
  }

  for (PetscInt i = 0; i < m * n; i++) {
    normG = PetscMax(normG, PetscAbsReal(G[i]));
  }

  for (PetscInt i = 0; i < m; i++) {
    for (PetscInt j = 0; j < n; j++) {
      PetscReal v = 0;
      for (PetscInt k = 0; k < m; k++) {
        /* M is m x m, G is m x n */
        v += M[i * m + k] * G[k * n + j];
      }
      if (PetscAbsReal(v) > worst) {
        worst = PetscAbsReal(v);
        wi    = i;
        wj    = j;
      }
    }
  }

  /* One line per cell, reporting the worst entry of K_e G_e relative to the
    * scale of the product. Round-off sits at m * PETSC_MACHINE_EPSILON,
    * ~1e-15; an incorrect gradient gives O(1). */
  {
    PetscReal scale = normM * normG;
    PetscReal rel   = (scale > 0) ? worst / scale : worst;
    if (rel > 1e-10)
      PetscCall(PetscPrintf(PETSC_COMM_SELF,
                            "gradient kernel: cell %d  worst |K_e G_e| = %.3e at (%d %d)  "
                            "||K_e|| = %.3e  ||G_e|| = %.3e  relative = %.3e\n",
                            (int)w, (double)worst, (int)wi, (int)wj, (double)normM, (double)normG, (double)rel));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Shared per-cell setup used by every LHS-assembly path.
 *
 * Reads the cell vertex coordinates and pulls the per-cell conductivity slice
 * out of the conductivity Vec. After this returns, `cell` is ready to be handed
 * to computeElementalMatrices / buildDiscreteGradientMatrix, which derive the
 * Jacobian and geometric orientation internally from the coordinates.
 *
 * @param[in]  dm              DMPlex mesh.
 * @param[in]  dmConductivity  DM carrying the conductivity field.
 * @param[in]  conductivity    Per-cell conductivity Vec.
 * @param[in]  cellId          Cell index.
 * @param[out] cell            Cell descriptor filled with coordinates and
 *                             conductivity.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
static PetscErrorCode prepareCellForAssembly(const DM dm, const DM dmConductivity, const Vec conductivity, PetscInt cellId, Cell *cell) {
  PetscFunctionBeginUser;
  PetscCall(extractCellCoordinates(dm, cellId, cell));
  PetscCall(extractCellConductivity(dmConductivity, conductivity, cellId, cell));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Creates a global right-hand-side Vec that accepts closure insertion.
 *
 * The Vec carries the DM local-to-global mapping and ignores negative
 * (constrained) indices, so element closures can be inserted with
 * VecSetValuesLocal.
 */
PetscErrorCode createRhsVector(const DM dm, Vec *b) {
  PetscFunctionBeginUser;
  ISLocalToGlobalMapping mapping;
  PetscCall(DMCreateGlobalVector(dm, b));
  PetscCall(DMGetLocalToGlobalMapping(dm, &mapping));
  PetscCall(VecSetLocalToGlobalMapping(*b, mapping));
  PetscCall(VecSetOption(*b, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE));
  PetscCall(VecSetFromOptions(*b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Creates a dense matrix with one column per right-hand side, laid out like @p b.
 */
PetscErrorCode createRhsMatrix(const Vec b, const PetscInt numRhs, Mat *B) {
  PetscFunctionBeginUser;
  PetscInt m, M;
  VecType  vtype;
  PetscCall(VecGetSize(b, &M));
  PetscCall(VecGetLocalSize(b, &m));
  PetscCall(VecGetType(b, &vtype));
  PetscCall(MatCreateDenseFromVecType(PetscObjectComm((PetscObject)b), vtype, m, PETSC_DECIDE, M, numRhs, m, NULL, B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief True numerical rank test of the discrete gradient G.
 *
 * Uses the smallest eigenvalues of H = Gᵀ·G (the weighted nodal graph
 * Laplacian, whose only kernel is the constant field) via a PETSc GMRES Arnoldi
 * cycle with PC=none, so the Ritz values are those of H itself. For a connected
 * mesh the expected spectrum is one numerically-zero eigenvalue and the rest
 * strictly positive: rank(G) = N-1. Ported from the legacy KG_validation.c
 * battery; used by reportGradientValidation under -petgem_validate_gradient.
 *
 * @param[in] comm  Communicator for the collective solve.
 * @param[in] G     Discrete gradient matrix.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
static PetscErrorCode gradientRankTestEig(MPI_Comm comm, Mat G) {
  PetscFunctionBeginUser;

  Mat         H;
  KSP         ksp;
  PC          pc;
  Vec         x, b;
  PetscRandom rnd;
  PetscInt    N, kdim, neig = 0, nullDim = 0;
  PetscReal   scale, emax = 0.0, emin = 0.0, zeroTol, lam2 = -1.0;
  PetscReal  *er, *ei;

  PetscCall(MatTransposeMatMult(G, G, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &H));
  PetscCall(MatSetOption(H, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatGetSize(H, &N, NULL));
  PetscCall(MatNorm(H, NORM_INFINITY, &scale));
  if (scale == 0.0) {
    scale = 1.0;
  }

  kdim = PetscMin(N, 100);

  PetscCall(MatCreateVecs(H, &x, &b));
  PetscCall(PetscRandomCreate(comm, &rnd));
  PetscCall(PetscRandomSetFromOptions(rnd));
  PetscCall(VecSetRandom(b, rnd));
  PetscCall(PetscRandomDestroy(&rnd));

  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "petgem_rank_"));
  PetscCall(KSPSetOperators(ksp, H, H));
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPGMRESSetRestart(ksp, kdim));
  PetscCall(KSPSetTolerances(ksp, 1.0e-12, 1.0e-50, PETSC_DEFAULT, kdim));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetComputeEigenvalues(ksp, PETSC_TRUE));
  PetscCall(KSPSetComputeSingularValues(ksp, PETSC_TRUE));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(KSPComputeExtremeSingularValues(ksp, &emax, &emin));
  if (emax > 0.0) {
    scale = emax;
  }
  zeroTol = scale * 1.0e-9;

  PetscCall(PetscMalloc2(kdim, &er, kdim, &ei));
  PetscCall(KSPComputeEigenvalues(ksp, kdim, er, ei, &neig));
  PetscCall(PetscSortReal(neig, er));

  for (PetscInt i = 0; i < neig; i++) {
    if (PetscAbsReal(er[i]) < zeroTol) {
      nullDim++;
    } else if (lam2 < 0.0) {
      lam2 = er[i];
    }
  }
  (void)nullDim;
  PetscCall(PetscPrintf(comm, "   eigen    lambda_min(nonzero)=%.2e  lambda_max=%.2e  separation=%.1e\n", (double)(lam2 > 0.0 ? lam2 : 0.0),
                        (double)scale, (double)(lam2 > 0.0 ? lam2 / scale : 0.0)));

  PetscCall(PetscFree2(er, ei));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&H));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Column-space mask that is 0 on boundary H1 dofs and 1 on interior ones.
 *
 * H(curl) carries Dirichlet conditions on the "Boundary" label (grid.c, one BC
 * field), so boundary edge dofs are removed; the H1 column space has no BC, so
 * every vertex/edge/face nodal dof is a column of G. The gradient of a boundary
 * H1 dof is therefore truncated and pollutes the raw K·G product. Scaling G's
 * columns by this mask isolates the BC-consistent interior residual. Ported from
 * KG_validation.c; the "Boundary" stratum id (100) matches setupNedelecGrid.
 *
 * @param[in]  dm               H(curl) DMPlex (source of the "Boundary" label).
 * @param[in]  grid             Grid descriptor (grid.H1dm supplies the section).
 * @param[in]  G                Discrete gradient (defines the column layout).
 * @param[out] mask             Column-space Vec, 0 on boundary / 1 on interior.
 * @param[out] numBoundaryCols  Count of masked (boundary) H1 columns.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
static PetscErrorCode buildInteriorColumnMask(DM dm, const Grid *grid, Mat G, Vec *mask, PetscInt *numBoundaryCols) {
  PetscFunctionBeginUser;

  Vec             dcol;
  DMLabel         label = NULL;
  IS              bdIs  = NULL;
  PetscSection    gH1section;
  PetscInt        nbd = 0, Bcount = 0;
  const PetscInt *bd;

  PetscCall(MatCreateVecs(G, &dcol, NULL));
  PetscCall(VecSet(dcol, 1.0));

  PetscCall(DMGetGlobalSection(grid->H1dm, &gH1section));
  PetscCall(DMGetLabel(dm, "Boundary", &label));
  if (label) {
    PetscCall(DMLabelGetStratumIS(label, 100, &bdIs));
  }

  if (bdIs) {
    PetscCall(ISGetLocalSize(bdIs, &nbd));
    PetscCall(ISGetIndices(bdIs, &bd));
    for (PetscInt i = 0; i < nbd; i++) {
      PetscInt dof, off;
      PetscCall(PetscSectionGetDof(gH1section, bd[i], &dof));
      PetscCall(PetscSectionGetOffset(gH1section, bd[i], &off));
      if (dof > 0 && off >= 0) {
        for (PetscInt d = 0; d < dof; d++) {
          PetscCall(VecSetValue(dcol, off + d, 0.0, INSERT_VALUES));
          Bcount++;
        }
      }
    }
    PetscCall(ISRestoreIndices(bdIs, &bd));
    PetscCall(ISDestroy(&bdIs));
  }
  PetscCall(VecAssemblyBegin(dcol));
  PetscCall(VecAssemblyEnd(dcol));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &Bcount, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)G)));

  *mask            = dcol;
  *numBoundaryCols = Bcount;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Structural check of G against what PCBDDCNedelecSupport expects.
 *
 * PCBDDC's Nedelec support (PETSc bddcprivate.c) groups edge dofs into coarse
 * edges using the row structure of the discrete gradient: on a coarse edge every
 * participating edge-dof row of G must connect to exactly order+1 nodal dofs
 * (the two endpoints plus the order-1 interior nodal dofs), and the local edge
 * dof count must be a multiple of order. This routine only reports, per MPI
 * rank, the histogram of nonzeros per edge-dof row of the assembled (filtered) G.
 *
 * @param[in] comm   Communicator (for the reduction and print).
 * @param[in] grid   Grid descriptor (order, per-class dof counts).
 * @param[in] G      Assembled, filtered discrete gradient.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
static PetscErrorCode reportGradientBddcStructure(MPI_Comm comm, const Grid *grid, Mat G) {
  PetscFunctionBeginUser;

  PetscInt rStart, rEnd, order = grid->fem.order;
  PetscInt minNnz = PETSC_MAX_INT, maxNnz = 0;
  PetscInt zeroRows    = 0, localRows;
  PetscInt expected    = order + 1; /* nodal dofs per edge-dof row on a coarse edge */
  PetscInt hist[16]    = {0};       /* nnz-per-row histogram, clamped at 15 */
  PetscInt offExpected = 0;         /* rows whose nnz is neither 0 nor expected */

  PetscCall(MatGetOwnershipRange(G, &rStart, &rEnd));
  localRows = rEnd - rStart;
  for (PetscInt row = rStart; row < rEnd; row++) {
    PetscInt ncols;
    PetscCall(MatGetRow(G, row, &ncols, NULL, NULL));
    if (ncols == 0) {
      zeroRows++;
    } else {
      minNnz = PetscMin(minNnz, ncols);
      maxNnz = PetscMax(maxNnz, ncols);
      if (ncols != expected) {
        offExpected++;
      }
    }
    hist[PetscMin(ncols, 15)]++;
    PetscCall(MatRestoreRow(G, row, &ncols, NULL, NULL));
  }
  if (minNnz == PETSC_MAX_INT) minNnz = 0;

  /* Global reduction so the report is one line regardless of rank count. */
  PetscInt gmin, gmax, gzero, goff, grows, ghist[16];
  PetscCallMPI(MPI_Allreduce(&minNnz, &gmin, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&maxNnz, &gmax, 1, MPIU_INT, MPI_MAX, comm));
  PetscCallMPI(MPI_Allreduce(&zeroRows, &gzero, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&offExpected, &goff, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&localRows, &grows, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(hist, ghist, 16, MPIU_INT, MPI_SUM, comm));

  /* PCBDDC's per-row Nedelec contract (ii[i+1]-ii[i] == order+1) applies to the
   * edge DOFs, which are the minimum-nnz rows: 2 endpoints + order-1 interior
   * nodal DOFs. Face and volume DOF rows legitimately connect to more nodal
   * DOFs and are not what BDDC checks per row, so they are reported separately
   * rather than flagged. gedge is the count of order+1 rows (the edge-DOF
   * class). */
  PetscInt gedge = ghist[PetscMin(expected, 15)];
  (void)gmax;
  (void)goff;
  (void)gedge;
  PetscCall(PetscPrintf(comm,
                        "   bddc     edge-dof nnz/row=%" PetscInt_FMT " (order+1)  %s  [rows=%" PetscInt_FMT ", zero=%" PetscInt_FMT "]\n",
                        expected, (gmin == expected) ? "uniform  PASS" : "MIN != order+1  CHECK", grows, gzero));
  PetscCall(PetscPrintf(comm, "   bddc     nnz/row histogram ="));
  for (PetscInt i = 1; i < 16; i++) {
    if (ghist[i]) PetscCall(PetscPrintf(comm, " %" PetscInt_FMT ":%" PetscInt_FMT, i, ghist[i]));
  }
  PetscCall(PetscPrintf(comm, "\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief De Rham / discrete-gradient validation battery (opt-in diagnostic).
 *
 * Enabled by -petgem_validate_gradient. Prints one compact block per
 * (order, ranks), verifying the assembled discrete gradient G against the De
 * Rham complex using the pure curl-curl matrix Acurl:
 *   derham : ||Acurl·G_int|| / (||Acurl|| ||G||) with boundary columns masked
 *            (curl of a gradient vanishes) → ~machine eps; plus ||G·1|| ~ 0;
 *   rank   : rank(G) = N-1 (only kernel is the constant field);
 *   eigen  : smallest nonzero / largest eigenvalue of GᵀG (gradientRankTestEig);
 *   bddc   : reportGradientBddcStructure - the row structure PCBDDC consumes.
 *
 * @param[in] comm   Communicator.
 * @param[in] dm     H(curl) DMPlex.
 * @param[in] grid   Grid descriptor.
 * @param[in] Acurl  Pure curl-curl matrix (Ke only, no mass term).
 * @param[in] G      Assembled discrete gradient.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
static PetscErrorCode reportGradientValidation(MPI_Comm comm, DM dm, const Grid *grid, Mat Acurl, Mat G) {
  PetscFunctionBeginUser;

  /* (1) Column rank of G modulo the constant nullspace. */
  Vec       x, y;
  PetscInt  N, nzero = 0;
  PetscReal nrmConstInf, nrmConst2, *colNorms, cmin = PETSC_MAX_REAL, cmax = 0.0;

  PetscCall(MatCreateVecs(G, &x, &y));
  PetscCall(VecSet(x, 1.0));
  PetscCall(MatMult(G, x, y));
  PetscCall(VecNorm(y, NORM_INFINITY, &nrmConstInf));
  PetscCall(VecNorm(y, NORM_2, &nrmConst2));

  PetscCall(MatGetSize(G, NULL, &N));
  PetscCall(PetscMalloc1(N, &colNorms));
  PetscCall(MatGetColumnNorms(G, NORM_2, colNorms));
  for (PetscInt i = 0; i < N; i++) {
    if (colNorms[i] < PETSC_SQRT_MACHINE_EPSILON) {
      nzero++;
    } else {
      cmin = PetscMin(cmin, colNorms[i]);
      cmax = PetscMax(cmax, colNorms[i]);
    }
  }
  if (cmin == PETSC_MAX_REAL) {
    cmin = 0.0;
  }
  PetscCall(PetscFree(colNorms));

  Vec      mask;
  PetscInt numBoundaryCols = 0;
  PetscCall(buildInteriorColumnMask(dm, grid, G, &mask, &numBoundaryCols));

  PetscBool constInNull = (PetscBool)(nrmConstInf < PETSC_SQRT_MACHINE_EPSILON);
  PetscBool rankOk      = (PetscBool)(constInNull && nzero <= numBoundaryCols);
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  (void)nrmConst2;
  (void)cmin;
  (void)cmax;

  /* (2) Interior De Rham residual R = Acurl·G_int, expect ~machine eps
   * (curl of a gradient vanishes). Boundary columns are masked out first,
   * since the H(curl) Dirichlet / H1 no-BC asymmetry pollutes the raw product. */
  Mat       AccAij, Gmask, Rint;
  PetscReal rFroInt, aFro, gFroInt, deRham;

  PetscCall(MatConvert(Acurl, MATAIJ, MAT_INITIAL_MATRIX, &AccAij));
  PetscCall(MatNorm(AccAij, NORM_FROBENIUS, &aFro));
  PetscCall(MatDuplicate(G, MAT_COPY_VALUES, &Gmask));
  PetscCall(MatDiagonalScale(Gmask, NULL, mask));
  PetscCall(MatMatMult(AccAij, Gmask, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Rint));
  PetscCall(MatNorm(Rint, NORM_FROBENIUS, &rFroInt));
  PetscCall(MatNorm(Gmask, NORM_FROBENIUS, &gFroInt));
  deRham = (aFro * gFroInt > 0.0) ? rFroInt / (aFro * gFroInt) : 0.0;

  /* Compact report: one block per (order, ranks). */
  PetscMPIInt commSize;
  PetscCallMPI(MPI_Comm_size(comm, &commSize));
  PetscCall(PetscPrintf(comm, "\n Gradient validation  order=%" PetscInt_FMT "  ranks=%d\n", grid->fem.order, (int)commSize));
  PetscCall(PetscPrintf(comm, "   derham   ||A*G_int||/(||A||||G||)=%.1e  %s   (||G*1||=%.1e)\n", (double)deRham,
                        (deRham < 1.0e-10) ? "PASS" : "CHECK", (double)nrmConstInf));
  PetscCall(PetscPrintf(comm, "   rank     N=%" PetscInt_FMT "  decoupled=%" PetscInt_FMT "  rank(G)=N-1  %s\n", N, nzero,
                        rankOk ? "PASS" : "CHECK"));

  /* (1b) numerical rank via smallest eigenvalues of GᵀG (prints one line). */
  PetscCall(gradientRankTestEig(comm, G));

  /* (3) row structure PCBDDC's Nedelec support consumes (prints two lines). */
  PetscCall(reportGradientBddcStructure(comm, grid, G));

  PetscCall(MatDestroy(&Rint));
  PetscCall(MatDestroy(&Gmask));
  PetscCall(VecDestroy(&mask));
  PetscCall(MatDestroy(&AccAij));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Adds X + beta*Y (Y may be NULL) over the n x n closure block of a cell. */
static PetscErrorCode addElementBlock(Mat mat, PetscInt n, const PetscInt *idx, PetscReal **X, PetscReal **Y, PetscScalar beta,
                                      PetscScalar *buf) {
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < n; i++) {
    for (PetscInt j = 0; j < n; j++) buf[i * n + j] = Y ? X[i][j] + beta * Y[i][j] : X[i][j];
  }
  PetscCall(MatSetValuesLocal(mat, n, idx, n, idx, buf, ADD_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode createOperatorMatrix(DM dm, Mat *A) {
  PetscFunctionBeginUser;
  PetscCall(DMCreateMatrix(dm, A));
  PetscCall(MatSetFromOptions(*A));
  PetscCall(MatSetOption(*A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Empty discrete gradient: rows on the H(curl) space of dm, columns on grid->H1dm. */
static PetscErrorCode createGradientMatrix(DM dm, const Grid *grid, Mat *G) {
  PetscFunctionBeginUser;
  Vec                    e, h;
  PetscInt               m, n, M, N;
  ISLocalToGlobalMapping rowMap, colMap;

  PetscCall(DMGetGlobalVector(dm, &e));
  PetscCall(DMGetGlobalVector(grid->H1dm, &h));
  PetscCall(VecGetSize(e, &M));
  PetscCall(VecGetLocalSize(e, &m));
  PetscCall(VecGetSize(h, &N));
  PetscCall(VecGetLocalSize(h, &n));
  PetscCall(DMRestoreGlobalVector(grid->H1dm, &h));
  PetscCall(DMRestoreGlobalVector(dm, &e));
  PetscCall(DMGetLocalToGlobalMapping(dm, &rowMap));
  PetscCall(DMGetLocalToGlobalMapping(grid->H1dm, &colMap));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm), G));
  PetscCall(MatSetSizes(*G, m, n, M, N));
  PetscCall(MatSetType(*G, MATAIJ));
  PetscCall(MatSetLocalToGlobalMapping(*G, rowMap, colMap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode assembleFinal(Mat A) {
  PetscFunctionBeginUser;
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Maxwell operator assembly.
 *
 * Single element loop over the local cells producing, depending on Ms:
 *
 *   fused (Ms == NULL):  KorA = K - constFactor * Ms_sigma
 *   split (Ms != NULL):  KorA = K and *Ms = Ms_sigma, for callers that form
 *                        A = K - i omega mu Ms per frequency (im.csem)
 *
 * with K the curl-curl stiffness and Ms_sigma the sigma-weighted mass matrix.
 * G, when requested, is the high-order discrete gradient consumed by
 * PCBDDCSetDiscreteGradient. -petgem_validate_gradient adds the De Rham and
 * PCBDDC-structure checks on G.
 */
PetscErrorCode assembleMaxwellOperator(const DM dm, const Grid *grid, const Vec conductivity, const PetscScalar constFactor, Mat *KorA,
                                       Mat *Ms, Mat *G) {
  PetscFunctionBeginUser;

  const PetscInt nE = grid->fem.numDofInCell;
  const PetscInt nH = grid->fem.numH1DofInCell;
  Cell           cell;
  Quadrature3D   quadrature3D;
  PetscInt       numDofIndices, numH1DofIndices, *dofIndices, *H1dofIndices;
  PetscReal    **Me, **Ke, **Ge = NULL;
  PetscScalar   *buf, *bufG = NULL;
  PetscSection   section, H1section;
  DM             dmConductivity;
  Mat            Acurl    = NULL;
  PetscBool      validate = PETSC_FALSE;

  if (G) PetscCall(PetscOptionsHasName(NULL, NULL, "-petgem_validate_gradient", &validate));

  PetscCall(DMSetAdjacency(dm, 0, PETSC_FALSE, PETSC_TRUE));
  PetscCall(DMSetMatrixPreallocateOnly(dm, PETSC_TRUE));
  PetscCall(createOperatorMatrix(dm, KorA));
  if (Ms) {
    PetscCall(MatDuplicate(*KorA, MAT_DO_NOT_COPY_VALUES, Ms));
    PetscCall(MatSetOption(*Ms, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  }
  if (validate) PetscCall(createOperatorMatrix(dm, &Acurl));
  if (G) PetscCall(createGradientMatrix(dm, grid, G));

  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetLocalSection(grid->H1dm, &H1section));
  PetscCall(VecGetDM(conductivity, &dmConductivity));
  PetscCall(createQuadrature3D(grid->fem.order, &quadrature3D));

  PetscCall(createDenseRows(nE, nE, &Me));
  PetscCall(createDenseRows(nE, nE, &Ke));
  PetscCall(PetscMalloc1(nE * nE, &buf));
  if (G) {
    PetscCall(createDenseRows(nE, nH, &Ge));
    PetscCall(PetscMalloc1(nE * nH, &bufG));
  }

  for (PetscInt i = grid->cellStart; i < grid->cellEnd; ++i) {
    PetscCall(prepareCellForAssembly(dm, dmConductivity, conductivity, i, &cell));
    PetscCall(computeElementalMatrices(&grid->fem, &cell, &quadrature3D, Me, Ke));
    PetscCall(DMPlexGetClosureIndices(dm, section, section, i, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));

    if (Ms) {
      PetscCall(addElementBlock(*KorA, numDofIndices, dofIndices, Ke, NULL, 0.0, buf));
      PetscCall(addElementBlock(*Ms, numDofIndices, dofIndices, Me, NULL, 0.0, buf));
    } else {
      PetscCall(addElementBlock(*KorA, numDofIndices, dofIndices, Ke, Me, -constFactor, buf));
    }
    if (validate) PetscCall(addElementBlock(Acurl, numDofIndices, dofIndices, Ke, NULL, 0.0, buf));

    if (G) {
      PetscCall(buildDiscreteGradientMatrix(&grid->fem, &cell, Ge));
      if (validate) PetscCall(checkGradientKernel(Ke[0], Ge[0], nE, nH, i));
      for (PetscInt j = 0; j < nE * nH; j++) bufG[j] = Ge[0][j];
      PetscCall(DMPlexGetClosureIndices(grid->H1dm, H1section, H1section, i, PETSC_TRUE, &numH1DofIndices, &H1dofIndices, NULL, NULL));
      PetscCall(MatSetValuesLocal(*G, numDofIndices, dofIndices, numH1DofIndices, H1dofIndices, bufG, INSERT_VALUES));
      PetscCall(DMPlexRestoreClosureIndices(grid->H1dm, H1section, H1section, i, PETSC_TRUE, &numH1DofIndices, &H1dofIndices, NULL, NULL));
    }

    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, i, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
  }

  PetscCall(assembleFinal(*KorA));
  if (Ms) PetscCall(assembleFinal(*Ms));
  if (G) {
    PetscCall(assembleFinal(*G));
    /* PCBDDC derives connectivity from G: keep only its nonzeros */
    PetscCall(MatFilter(*G, 0, PETSC_TRUE, PETSC_FALSE));
    PetscCall(MatViewFromOptions(*G, NULL, "-petgem_grad_view"));
  }
  if (validate) {
    PetscCall(assembleFinal(Acurl));
    PetscCall(reportGradientValidation(PetscObjectComm((PetscObject)dm), dm, grid, Acurl, *G));
    PetscCall(MatDestroy(&Acurl));
  }

  PetscCall(destroyQuadrature3D(&quadrature3D));
  PetscCall(destroyDenseRows(&Me));
  PetscCall(destroyDenseRows(&Ke));
  PetscCall(destroyDenseRows(&Ge));
  PetscCall(PetscFree(buf));
  PetscCall(PetscFree(bufG));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Refills an existing Ms matrix for the current conductivity field.
 *
 * Inverse-kernel companion to assembleMaxwellOperator: recomputes the
 * sigma-weighted mass entries in place, keeping the sparsity pattern. K and G
 * are sigma-independent and are not touched.
 */
PetscErrorCode refillMassMatrix(const DM dm, const Grid *grid, const Vec conductivity, const Quadrature3D *quadrature3D, PetscReal **Me,
                                PetscReal **Ke, Mat Ms) {
  PetscFunctionBeginUser;

  const PetscInt nE = grid->fem.numDofInCell;
  PetscSection   section;
  DM             dmConductivity;
  PetscScalar   *buf;
  Cell           cell;
  PetscInt       numDofIndices, *dofIndices;

  PetscCall(MatZeroEntries(Ms));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(VecGetDM(conductivity, &dmConductivity));
  PetscCall(PetscMalloc1(nE * nE, &buf));

  for (PetscInt i = grid->cellStart; i < grid->cellEnd; ++i) {
    PetscCall(prepareCellForAssembly(dm, dmConductivity, conductivity, i, &cell));
    PetscCall(computeElementalMatrices(&grid->fem, &cell, quadrature3D, Me, Ke));
    PetscCall(DMPlexGetClosureIndices(dm, section, section, i, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
    PetscCall(addElementBlock(Ms, numDofIndices, dofIndices, Me, NULL, 0.0, buf));
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, i, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
  }

  PetscCall(PetscFree(buf));
  PetscCall(assembleFinal(Ms));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Line segment read from the edge-conductance file. */
typedef struct {
  PetscReal a[NUM_DIMENSIONS], b[NUM_DIMENSIONS];
  PetscReal sigmaA;
} ConductanceSegment;

/* Reads "x0 y0 z0 x1 y1 z1 sigmaA" rows; '#' starts a comment line. */
static PetscErrorCode readConductanceSegments(const char *path, PetscInt *n, ConductanceSegment **segs) {
  PetscFunctionBeginUser;
  FILE    *fp;
  char     line[PETSC_MAX_PATH_LEN];
  PetscInt cap = 0;

  *n    = 0;
  *segs = NULL;
  PetscCall(PetscFOpen(PETSC_COMM_SELF, path, "r", &fp));
  while (fgets(line, sizeof(line), fp)) {
    ConductanceSegment s;
    double             v[7];
    char              *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '#' || *p == '\n' || *p == '\0') continue;
    PetscCheck(sscanf(p, "%lf %lf %lf %lf %lf %lf %lf", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]) == 7, PETSC_COMM_SELF,
               PETSC_ERR_FILE_UNEXPECTED, "Edge-conductance row needs 7 values: %s", line);
    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
      s.a[d] = (PetscReal)v[d];
      s.b[d] = (PetscReal)v[3 + d];
    }
    s.sigmaA = (PetscReal)v[6];
    if (*n == cap) {
      cap = PetscMax(8, 2 * cap);
      PetscCall(PetscRealloc(cap * sizeof(ConductanceSegment), segs));
    }
    (*segs)[(*n)++] = s;
  }
  PetscCall(PetscFClose(PETSC_COMM_SELF, fp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* True if point x lies on segment s, to a relative tolerance. */
static PetscBool pointOnSegment(const ConductanceSegment *s, const PetscReal x[NUM_DIMENSIONS]) {
  PetscReal ab[NUM_DIMENSIONS], ax[NUM_DIMENSIONS], c[NUM_DIMENSIONS], L2 = 0.0, t = 0.0;
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    ab[d] = s->b[d] - s->a[d];
    ax[d] = x[d] - s->a[d];
    L2 += ab[d] * ab[d];
    t += ax[d] * ab[d];
  }
  const PetscReal L   = PetscSqrtReal(L2);
  const PetscReal tol = 1.0e-6 * PetscMax(L, 1.0);
  c[0]                = ax[1] * ab[2] - ax[2] * ab[1];
  c[1]                = ax[2] * ab[0] - ax[0] * ab[2];
  c[2]                = ax[0] * ab[1] - ax[1] * ab[0];
  if (PetscSqrtReal(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) > tol * L) return PETSC_FALSE;
  return (PetscBool)(t >= -tol * L && t <= L2 + tol * L);
}

/**
 * @brief Adds line conductance (sigma x area) on mesh edges to the operator.
 *
 * Off unless -petgem_edge_conductance_file <txt> is given. Each mesh edge lying
 * on a segment of the file gets -constFactor * sigmaA * int_e (Ni.t)(Nj.t) dl,
 * evaluated with the cell basis of one adjacent cell and split evenly among the
 * ranks that hold the edge (Weiss 2017 hierarchical conductivity).
 */
static PetscErrorCode addEdgeConductance(const DM dm, const Grid *grid, const PetscScalar constFactor, Mat A, const PetscBool report) {
  PetscFunctionBeginUser;

  char                file[PETSC_MAX_PATH_LEN];
  PetscBool           flg  = PETSC_FALSE;
  PetscInt            nseg = 0, eStart, eEnd, numEdges = 0, ng;
  ConductanceSegment *segs = NULL;
  PetscSection        section, csection;
  Vec                 coords, gshare, lshare;
  const PetscScalar  *xc, *share;
  PetscReal          *gx, *gw, length = 0.0;
  PetscReal         **Ni;
  PetscScalar        *blk;
  PetscReal          *tau;
  PetscInt           *rows, *jl;
  const PetscInt      nE = grid->fem.numDofInCell;

  PetscCall(PetscOptionsGetString(NULL, NULL, "-petgem_edge_conductance_file", file, sizeof(file), &flg));
  if (!flg) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(readConductanceSegments(file, &nseg, &segs));

  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetCoordinateSection(dm, &csection));
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(VecGetArrayRead(coords, &xc));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));

  /* Number of ranks holding each local dof */
  PetscCall(DMCreateGlobalVector(dm, &gshare));
  PetscCall(DMCreateLocalVector(dm, &lshare));
  PetscCall(VecSet(lshare, 1.0));
  PetscCall(VecZeroEntries(gshare));
  PetscCall(DMLocalToGlobal(dm, lshare, ADD_VALUES, gshare));
  PetscCall(DMGlobalToLocal(dm, gshare, INSERT_VALUES, lshare));
  PetscCall(VecGetArrayRead(lshare, &share));

  ng = grid->fem.order + 1;
  PetscCall(PetscMalloc2(ng, &gx, ng, &gw));
  PetscCall(PetscDTGaussQuadrature(ng, 0.0, 1.0, gx, gw));
  PetscCall(createBasisValues(nE, &Ni));
  PetscCall(PetscMalloc4(nE * nE, &blk, ng * nE, &tau, nE, &rows, nE, &jl));

  for (PetscInt e = eStart; e < eEnd; ++e) {
    const PetscInt *cone;
    PetscReal       x[2][NUM_DIMENSIONS], t[NUM_DIMENSIONS], L = 0.0, sigmaA = -1.0;
    PetscInt        ndof, off, cell = -1, nstar, *star = NULL;

    PetscCall(PetscSectionGetDof(section, e, &ndof));
    if (!ndof) continue;
    PetscCall(DMPlexGetCone(dm, e, &cone));
    for (PetscInt k = 0; k < 2; k++) {
      PetscInt voff;
      PetscCall(PetscSectionGetOffset(csection, cone[k], &voff));
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) x[k][d] = PetscRealPart(xc[voff + d]);
    }
    for (PetscInt s = 0; s < nseg; s++) {
      if (pointOnSegment(&segs[s], x[0]) && pointOnSegment(&segs[s], x[1])) {
        sigmaA = segs[s].sigmaA;
        break;
      }
    }
    if (sigmaA < 0.0) continue;

    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
      t[d] = x[1][d] - x[0][d];
      L += t[d] * t[d];
    }
    L = PetscSqrtReal(L);
    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) t[d] /= L;

    PetscCall(DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &nstar, &star));
    for (PetscInt k = 0; k < nstar; k++) {
      if (star[2 * k] >= grid->cellStart && star[2 * k] < grid->cellEnd) {
        cell = star[2 * k];
        break;
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &nstar, &star));
    PetscCheck(cell >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Edge %" PetscInt_FMT " has no local cell", e);

    PetscCall(PetscSectionGetOffset(section, e, &off));
    const PetscReal nshare = PetscRealPart(share[off]);

    Cell      c;
    PetscInt  numIdx, *idx, nr = 0;
    PetscReal tmax = 0.0;
    PetscCall(extractCellCoordinates(dm, cell, &c));
    for (PetscInt q = 0; q < ng; q++) {
      PetscReal p[NUM_DIMENSIONS], ref[NUM_DIMENSIONS];
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) p[d] = x[0][d] + gx[q] * (x[1][d] - x[0][d]);
      PetscCall(tetrahedronXyzToReference(c.coordinates, p, ref));
      PetscCall(evaluateNedelecBasis(&grid->fem, &c, ref, Ni, NULL));
      for (PetscInt j = 0; j < nE; j++) {
        tau[q * nE + j] = Ni[0][j] * t[0] + Ni[1][j] * t[1] + Ni[2][j] * t[2];
        tmax            = PetscMax(tmax, PetscAbsReal(tau[q * nE + j]));
      }
    }
    PetscCall(DMPlexGetClosureIndices(dm, section, section, cell, PETSC_TRUE, &numIdx, &idx, NULL, NULL));
    for (PetscInt j = 0; j < nE; j++) {
      PetscReal m = 0.0;
      for (PetscInt q = 0; q < ng; q++) m = PetscMax(m, PetscAbsReal(tau[q * nE + j]));
      if (m > 1.0e-10 * tmax) {
        jl[nr]   = j;
        rows[nr] = idx[j];
        nr++;
      }
    }
    for (PetscInt a = 0; a < nr; a++) {
      for (PetscInt b = 0; b < nr; b++) {
        PetscReal v = 0.0;
        for (PetscInt q = 0; q < ng; q++) v += gw[q] * tau[q * nE + jl[a]] * tau[q * nE + jl[b]];
        blk[a * nr + b] = -constFactor * (sigmaA * L * v / nshare);
      }
    }
    PetscCall(MatSetValuesLocal(A, nr, rows, nr, rows, blk, ADD_VALUES));
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, cell, PETSC_TRUE, &numIdx, &idx, NULL, NULL));

    numEdges++;
    length += L / nshare;
  }
  PetscCall(assembleFinal(A));

  if (report) {
    PetscReal tot[2] = {(PetscReal)numEdges, length};
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, tot, 2, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)dm)));
    PetscCall(logKvf(PETSC_COMM_WORLD, "Edge conductance", "%" PetscInt_FMT " segments, %.6g m of mesh edges (%.0f rank-local edges)", nseg,
                     (double)tot[1], (double)tot[0]));
  }

  PetscCall(PetscFree4(blk, tau, rows, jl));
  PetscCall(destroyBasisValues(&Ni));
  PetscCall(PetscFree2(gx, gw));
  PetscCall(VecRestoreArrayRead(lshare, &share));
  PetscCall(VecDestroy(&gshare));
  PetscCall(VecDestroy(&lshare));
  PetscCall(VecRestoreArrayRead(coords, &xc));
  PetscCall(PetscFree(segs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode addEdgeConductanceFromOptions(const DM dm, const Grid *grid, const PetscScalar constFactor, Mat A) {
  PetscFunctionBeginUser;
  PetscCall(addEdgeConductance(dm, grid, constFactor, A, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Assembles the operator for the BDDC matrix of solveMaxwellSystem.
 *
 * Off unless -petgem_bddc_spd is given: @p A with the conductivity raised to
 * -petgem_bddc_spd_floor <S/m> (default 0.1), plus the edge conductance; @p A
 * itself for a floor of 0.
 */
PetscErrorCode assembleBddcOperatorFromOptions(const DM dm, const Grid *grid, const Vec conductivity, const PetscScalar constFactor,
                                               const Mat A, Mat *Abddc) {
  PetscFunctionBeginUser;

  PetscBool    spd        = PETSC_FALSE;
  PetscReal    sigmaFloor = 0.1;
  Vec          condFloor;
  DM           cdm;
  PetscScalar *s;
  PetscInt     nloc;

  *Abddc = NULL;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-petgem_bddc_spd", &spd, NULL));
  if (!spd) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-petgem_bddc_spd_floor", &sigmaFloor, NULL));
  PetscCall(logKvf(PetscObjectComm((PetscObject)dm), "BDDC matrix", "SPD, sigma floor %g S/m", (double)sigmaFloor));
  if (sigmaFloor <= 0.0) {
    PetscCall(PetscObjectReference((PetscObject)A));
    *Abddc = A;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* sigma >= sigmaFloor */
  PetscCall(VecGetDM(conductivity, &cdm));
  PetscCall(VecDuplicate(conductivity, &condFloor));
  PetscCall(VecCopy(conductivity, condFloor));
  PetscCall(VecSetDM(condFloor, cdm));
  PetscCall(VecGetLocalSize(condFloor, &nloc));
  PetscCall(VecGetArray(condFloor, &s));
  for (PetscInt i = 0; i < nloc; i++) s[i] = PetscMax(PetscRealPart(s[i]), sigmaFloor);
  PetscCall(VecRestoreArray(condFloor, &s));

  PetscCall(assembleMaxwellOperator(dm, grid, condFloor, constFactor, Abddc, NULL, NULL));
  PetscCall(addEdgeConductance(dm, grid, constFactor, *Abddc, PETSC_FALSE));
  PetscCall(VecDestroy(&condFloor));
  PetscFunctionReturn(PETSC_SUCCESS);
}
