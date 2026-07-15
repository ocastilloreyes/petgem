/*
 * Filename: mms.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-07-08
 *
 * Description:
 * Single entry point for the Method-of-Manufactured-Solutions (MMS)
 * verification of the fm.csem high-order Nedelec discretization. runForward
 * dispatches here on -mms; everything MMS lives in this file.
 *
 * Three levels, selected by runtime flags:
 *   -mms                 Level 1 (standard): one run performs the Galerkin
 *                        solve and the L2-projection best-approximation, and
 *                        records both error norms plus the solve residual.
 *   -mms_diagnostics     Level 2: adds the over-integrated (quadrature-check)
 *                        error norms and the conditioning probe (kappa, floor).
 *   -mms_conditioning    Level 3: re-solves at a set of mass-term (frequency)
 *                        scales to vary the operator conditioning kappa(A).
 *
 * Outputs (in params.outputDirectory, HDF5 as in the rest of PETGEM):
 *   {output_filename}.h5        one file per run     (Level 1/2)
 *   {output_filename}_s<i>.h5   one file per scale   (Level 3)
 * Each file stores the metrics as root-group attributes.
 */

#include <stdio.h>
#include <petsc.h>
#include <petscviewerhdf5.h>

#include "assembly.h"
#include "common.h"
#include "constants.h"
#include "fem.h"
#include "grid.h"
#include "io.h"
#include "mms.h"
#include "solver.h"
#include "transmitter.h"
#include "version.h"

/* Over-integration added to the basis order for the quadrature check. */
#define MMS_DIAG_QUAD_EXTRA 3

/* Mass-term (frequency) scales swept by the Level-3 conditioning sweep. */
static const PetscReal MMS_COND_SCALES[] = {1.0, 100.0, 10000.0};
#define MMS_COND_NUM_SCALES 3

/**
 * @brief Global relative L2 / H(curl) error norms at one quadrature rule.
 *
 * Reconstructs E_h and curl E_h at the cell quadrature points from the ghosted
 * DOF array, integrates the squared differences against E* / curl E* with the
 * |detJ| measure, MPI-reduces, and returns the relative norms.
 */
static PetscErrorCode mmsErrorNormsAtQuad(const DM dm, const Grid grid, PetscSection section,
                                          const PetscScalar *xarr, PetscInt quadOrder,
                                          PetscReal *relL2, PetscReal *relHcurl) {
  PetscFunctionBeginUser;

  Cell cell;
  Quadrature3D q;
  PetscReal **Ni, **NiCurl;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  PetscCall(computeNum3DQuadraturePoints(quadOrder, &q));
  PetscCall(PetscCalloc1(q.numPoints, &q.points));
  for (PetscInt i = 0; i < q.numPoints; i++) {
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &q.points[i]));
  }
  PetscCall(PetscCalloc1(q.numPoints, &q.weights));
  PetscCall(compute3DQuadraturePoints(&q));

  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &NiCurl));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(grid.numDofInCell, &Ni[i]));
    PetscCall(PetscCalloc1(grid.numDofInCell, &NiCurl[i]));
  }

  PetscReal l2sq = 0.0, curlsq = 0.0;

  for (PetscInt c = grid.cellStart; c < grid.cellEnd; ++c) {

    PetscCall(extractCellCoordinates(dm, c, &cell));
    PetscCall(computeCellJacobian(&cell));
    const PetscReal absdet = PetscAbsReal(cell.detJacobian);

    PetscInt numLocal, *localIdx;
    PetscCall(DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE, &numLocal, &localIdx, NULL, NULL));

    for (PetscInt qp = 0; qp < q.numPoints; qp++) {
      const PetscReal *ref = q.points[qp];

      PetscReal xphys[NUM_DIMENSIONS];
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
        xphys[d] = cell.coordinates[d]
                 + cell.jacobian[0][d] * ref[0]
                 + cell.jacobian[1][d] * ref[1]
                 + cell.jacobian[2][d] * ref[2];
      }

      PetscCall(evaluateNedelecBasis(&grid.fem, &cell, ref, Ni, NiCurl));

      PetscScalar Eh[NUM_DIMENSIONS] = {0.0, 0.0, 0.0};
      PetscScalar Ch[NUM_DIMENSIONS] = {0.0, 0.0, 0.0};
      for (PetscInt j = 0; j < grid.numDofInCell; j++) {
        if (localIdx[j] < 0) {
          continue;
        }
        const PetscScalar coeff = xarr[localIdx[j]];
        for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
          Eh[d] += coeff * (PetscScalar)Ni[d][j];
          Ch[d] += coeff * (PetscScalar)NiCurl[d][j];
        }
      }

      PetscScalar Eex[NUM_DIMENSIONS], Cex[NUM_DIMENSIONS];
      mmsExactE(xphys, Eex);
      mmsExactCurlE(xphys, Cex);

      const PetscReal wdet = q.weights[qp] * absdet;
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
        const PetscScalar de = Eh[d] - Eex[d];
        const PetscScalar dc = Ch[d] - Cex[d];
        l2sq   += wdet * PetscRealPart(de * PetscConj(de));
        curlsq += wdet * PetscRealPart(dc * PetscConj(dc));
      }
    }

    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE, &numLocal, &localIdx, NULL, NULL));
  }

  PetscReal partial[2] = {l2sq, curlsq};
  PetscReal global[2]  = {0.0, 0.0};
  PetscCallMPI(MPI_Allreduce(partial, global, 2, MPIU_REAL, MPI_SUM, comm));

  *relL2    = PetscSqrtReal(global[0])             / MMS_E_L2_NORM;
  *relHcurl = PetscSqrtReal(global[0] + global[1]) / MMS_E_HCURL_NORM;

  PetscCall(PetscFree(q.weights));
  for (PetscInt i = 0; i < q.numPoints; i++) {
    PetscCall(PetscFree(q.points[i]));
  }
  PetscCall(PetscFree(q.points));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscFree(Ni[i]));
    PetscCall(PetscFree(NiCurl[i]));
  }
  PetscCall(PetscFree(Ni));
  PetscCall(PetscFree(NiCurl));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Relative error norms for column 0 of a solution matrix. */
static PetscErrorCode mmsColumnErrors(const DM dm, const Grid grid, const Mat X,
                                      PetscInt quadOrder, PetscReal *relL2, PetscReal *relHcurl) {
  PetscFunctionBeginUser;
  PetscSection section;
  Vec x, xloc;
  const PetscScalar *xarr;
  PetscCall(MatDenseGetColumnVecRead(X, 0, &x));
  PetscCall(DMGetLocalVector(dm, &xloc));
  PetscCall(DMGlobalToLocalBegin(dm, x, INSERT_VALUES, xloc));
  PetscCall(DMGlobalToLocalEnd(dm, x, INSERT_VALUES, xloc));
  PetscCall(MatDenseRestoreColumnVecRead(X, 0, &x));
  PetscCall(VecGetArrayRead(xloc, &xarr));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(mmsErrorNormsAtQuad(dm, grid, section, xarr, quadOrder, relL2, relHcurl));
  PetscCall(VecRestoreArrayRead(xloc, &xarr));
  PetscCall(DMRestoreLocalVector(dm, &xloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/** @brief Backward error ||A x - b|| / ||b|| for column 0. */
static PetscErrorCode mmsResidual(const Mat A, const Mat B, const Mat X, PetscReal *relRes) {
  PetscFunctionBeginUser;
  Vec xc, bc, r;
  PetscReal rn, bn;
  PetscCall(MatDenseGetColumnVecRead(X, 0, &xc));
  PetscCall(MatDenseGetColumnVecRead(B, 0, &bc));
  PetscCall(MatCreateVecs(A, NULL, &r));
  PetscCall(MatMult(A, xc, r));
  PetscCall(VecAXPY(r, -1.0, bc));
  PetscCall(VecNorm(r, NORM_2, &rn));
  PetscCall(VecNorm(bc, NORM_2, &bn));
  *relRes = (bn > 0.0) ? rn / bn : rn;
  PetscCall(VecDestroy(&r));
  PetscCall(MatDenseRestoreColumnVecRead(X, 0, &xc));
  PetscCall(MatDenseRestoreColumnVecRead(B, 0, &bc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Conditioning probe: effective condition number and forward-error floor
 * of the operator, estimated from a relatively perturbed RHS re-solve. The
 * amplification of the perturbation gives kappa(A), and kappa(A)*eps is the
 * attainable forward-error floor.
 */
static PetscErrorCode mmsConditionProbe(const DM dm, const Grid grid, const Mat A,
                                        const Mat B, const Mat X,
                                        PetscReal *condEst, PetscReal *floorEst) {
  PetscFunctionBeginUser;
  Mat Bpert, X2;
  Vec bp, dpert, xc, x2c, dx;
  PetscReal bn, dn, dxn, xn;
  const PetscReal delta = 1.0e-8;

  PetscCall(MatDuplicate(B, MAT_COPY_VALUES, &Bpert));
  PetscCall(MatDenseGetColumnVecWrite(Bpert, 0, &bp));
  PetscCall(VecDuplicate(bp, &dpert));
  PetscCall(VecSetRandom(dpert, NULL));
  PetscCall(VecNorm(bp, NORM_2, &bn));
  PetscCall(VecNorm(dpert, NORM_2, &dn));
  if (bn > 0.0 && dn > 0.0) {
    PetscCall(VecAXPY(bp, delta * bn / dn, dpert));
  }
  PetscCall(VecDestroy(&dpert));
  PetscCall(MatDenseRestoreColumnVecWrite(Bpert, 0, &bp));

  PetscCall(solveCsemSystem(dm, A, Bpert, NULL, grid.fem.order, &X2));

  PetscCall(MatDenseGetColumnVecRead(X,  0, &xc));
  PetscCall(MatDenseGetColumnVecRead(X2, 0, &x2c));
  PetscCall(VecDuplicate(xc, &dx));
  PetscCall(VecCopy(x2c, dx));
  PetscCall(VecAXPY(dx, -1.0, xc));
  PetscCall(VecNorm(dx, NORM_2, &dxn));
  PetscCall(VecNorm(xc, NORM_2, &xn));
  *condEst  = (xn > 0.0) ? (dxn / xn) / delta : -1.0;
  *floorEst = (*condEst >= 0.0) ? *condEst * PETSC_MACHINE_EPSILON : -1.0;
  PetscCall(VecDestroy(&dx));
  PetscCall(MatDenseRestoreColumnVecRead(X2, 0, &x2c));
  PetscCall(MatDenseRestoreColumnVecRead(X,  0, &xc));
  PetscCall(MatDestroy(&X2));
  PetscCall(MatDestroy(&Bpert));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Assemble and solve one MMS system.
 *
 * useForcing = PETSC_TRUE : A = K - iωμ·Ms, b = ∫ f*·N   (the Galerkin solve).
 * useForcing = PETSC_FALSE: A = mass (Ms, σ=1),  b = ∫ E*·N (the L2 projection).
 * G is never built (MMS uses a MUMPS direct solve, not PCBDDC). Caller destroys
 * A, B, X.
 */
static PetscErrorCode mmsAssembleSolve(const petgemParams params, const CsemSourceSet sources,
                                       const DM dm, const Grid grid, const Vec conductivity,
                                       const PetscScalar constFactor, const PetscBool useForcing,
                                       Mat *A, Mat *B, Mat *X) {
  PetscFunctionBeginUser;
  PetscCall(assembleCsemMMSRHS(params, sources, dm, grid, conductivity, useForcing, B));
  if (useForcing) {
    PetscCall(assembleCsemKandM(params, dm, grid, conductivity, constFactor, A, NULL, NULL));
  } else {
    Mat Kthrow;
    PetscCall(assembleCsemKandM(params, dm, grid, conductivity, constFactor, &Kthrow, A, NULL));
    PetscCall(MatDestroy(&Kthrow));
  }
  PetscCall(solveCsemSystem(dm, *A, *B, NULL, grid.fem.order, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Write one MMS measurement to {dir}/{fname}.h5.
 *
 * PETGEM writes its responses in HDF5 (computeFields); the MMS metrics follow
 * the same convention. One file is written per run, with all scalars stored as
 * root-group attributes (plus the petgem_version / date provenance attributes
 * used elsewhere), so a directory of these files reconstructs the full table.
 * Collective on `comm` (all ranks carry the same, already-reduced values).
 */
static PetscErrorCode mmsWriteH5(MPI_Comm comm, const char *dir, const char *fname,
                                 PetscInt order, PetscInt dofs, PetscReal omegaScale,
                                 PetscReal solveL2, PetscReal solveHcurl,
                                 PetscReal projL2, PetscReal projHcurl, PetscReal residual,
                                 PetscReal solveL2hi, PetscReal solveHcurlhi,
                                 PetscReal condEst, PetscReal floorEst) {
  PetscFunctionBeginUser;
  char path[PETSC_MAX_PATH_LEN], version[64], date[30];
  PetscViewer v;

  PetscCall(PetscStrncpy(path, dir, sizeof(path)));
  size_t len = strlen(path);
  if (len > 0 && path[len - 1] != '/') {
    PetscCall(PetscStrlcat(path, "/", sizeof(path)));
  }
  PetscCall(PetscStrlcat(path, fname, sizeof(path)));
  PetscCall(PetscStrlcat(path, ".h5", sizeof(path)));

  PetscCall(PetscViewerHDF5Open(comm, path, FILE_MODE_WRITE, &v));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "order",          PETSC_INT,  &order));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "dofs",           PETSC_INT,  &dofs));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "omega_scale",    PETSC_REAL, &omegaScale));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "solve_L2",       PETSC_REAL, &solveL2));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "solve_Hcurl",    PETSC_REAL, &solveHcurl));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "proj_L2",        PETSC_REAL, &projL2));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "proj_Hcurl",     PETSC_REAL, &projHcurl));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "residual",       PETSC_REAL, &residual));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "solve_L2_hi",    PETSC_REAL, &solveL2hi));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "solve_Hcurl_hi", PETSC_REAL, &solveHcurlhi));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "cond_est",       PETSC_REAL, &condEst));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "floor_est",      PETSC_REAL, &floorEst));

  snprintf(version, sizeof(version), "%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "petgem_version", PETSC_STRING, version));
  PetscCall(PetscGetDate(date, sizeof(date)));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "date", PETSC_STRING, date));

  PetscCall(PetscViewerDestroy(&v));
  PetscCall(PetscPrintf(comm, "   %-14s %s\n", "wrote", path));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Level 3: re-solve at a set of mass-term scales to vary kappa(A).
 *
 * Self-contained (its scale=1 point is the physical solve), so it is run instead
 * of the standard Level-1 pass. Writes one HDF5 file per scale,
 * {output_filename}_s<i>.h5, carrying omega_scale, the solve norms, and the
 * conditioning estimate (proj / residual / hi columns are set to -1).
 */
static PetscErrorCode mmsConditioningSweep(const petgemParams params, CsemSourceSet sources,
                                           const DM dm, const Grid grid, const Vec conductivity) {
  PetscFunctionBeginUser;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  const PetscReal baseFreq = sources.freq;

  PetscInt Ndof;
  {
    Vec t;
    PetscCall(DMCreateGlobalVector(dm, &t));
    PetscCall(VecGetSize(t, &Ndof));
    PetscCall(VecDestroy(&t));
  }

  PetscCall(PetscPrintf(comm, "\n MMS conditioning sweep (order %" PetscInt_FMT "):\n", grid.fem.order));
  for (PetscInt i = 0; i < MMS_COND_NUM_SCALES; i++) {
    /* Scale the mass term (frequency); E* stays exact because the operator and
     * f* are both rebuilt from the same scaled ω. */
    sources.freq = baseFreq * MMS_COND_SCALES[i];
    const PetscReal    omega = sources.freq * 2.0 * PETSC_PI;
    const PetscScalar  cf    = (0.0 + 1.0 * PETSC_i) * (omega * MU);

    Mat A, B, X;
    PetscReal L2, Hc, cond, floor;
    PetscCall(mmsAssembleSolve(params, sources, dm, grid, conductivity, cf, PETSC_TRUE, &A, &B, &X));
    PetscCall(mmsColumnErrors(dm, grid, X, grid.fem.order, &L2, &Hc));
    PetscCall(mmsConditionProbe(dm, grid, A, B, X, &cond, &floor));
    PetscCall(PetscPrintf(comm, "   scale %9.0f : kappa %.2e   L2 %.3e   H(curl) %.3e\n",
                          (double)MMS_COND_SCALES[i], (double)cond, (double)L2, (double)Hc));

    char cfname[PETSC_MAX_PATH_LEN];
    PetscCall(PetscSNPrintf(cfname, sizeof(cfname), "%s_s%" PetscInt_FMT, params.outputFilename, i));
    PetscCall(mmsWriteH5(comm, params.outputDirectory, cfname, grid.fem.order, Ndof,
                         MMS_COND_SCALES[i], L2, Hc, -1.0, -1.0, -1.0, -1.0, -1.0, cond, floor));

    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&B));
    PetscCall(MatDestroy(&X));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Complete MMS verification for one (order, mesh). Single entry point.
 *
 * Level 1 (-mms): Galerkin solve + L2 projection, both error norms and the
 * solve residual -> one HDF5 file {output_filename}.h5.
 * Level 2 (-mms_diagnostics): also the over-integrated solve norms and the
 * kappa/floor conditioning probe (extra attributes in the same file).
 * Level 3 (-mms_conditioning): instead runs the mass-term scale sweep -> one
 * HDF5 file per scale, {output_filename}_s<i>.h5.
 */
PetscErrorCode runMMSVerification(const petgemParams params, const DM dm, const Grid grid,
                                  const Vec conductivity, const CsemSourceSet sources) {
  PetscFunctionBeginUser;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  PetscBool diagnostics = PETSC_FALSE, conditioning = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mms_diagnostics",  &diagnostics,  NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mms_conditioning", &conditioning, NULL));

  /* Level 3: the conditioning sweep is self-contained (its scale=1 point is the
   * physical solve), so it runs instead of the standard Level-1 pass. */
  if (conditioning) {
    PetscCall(mmsConditioningSweep(params, sources, dm, grid, conductivity));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  const PetscReal    omega       = sources.freq * 2.0 * PETSC_PI;
  const PetscScalar  constFactor = (0.0 + 1.0 * PETSC_i) * (omega * MU);
  const PetscInt     p           = grid.fem.order;

  /* Level 1: Galerkin solve and L2-projection best-approximation. */
  Mat A, Bs, Xs, M, Bp, Xp;
  PetscCall(mmsAssembleSolve(params, sources, dm, grid, conductivity, constFactor, PETSC_TRUE,  &A, &Bs, &Xs));
  PetscCall(mmsAssembleSolve(params, sources, dm, grid, conductivity, constFactor, PETSC_FALSE, &M, &Bp, &Xp));

  PetscReal sL2 = 0.0, sHc = 0.0, pL2 = 0.0, pHc = 0.0, residual = -1.0;
  PetscCall(mmsColumnErrors(dm, grid, Xs, p, &sL2, &sHc));
  PetscCall(mmsColumnErrors(dm, grid, Xp, p, &pL2, &pHc));
  PetscCall(mmsResidual(A, Bs, Xs, &residual));

  /* Level 2: over-integrated solve norms + conditioning probe. */
  PetscReal sL2hi = sL2, sHchi = sHc, cond = -1.0, floor = -1.0;
  if (diagnostics) {
    PetscCall(mmsColumnErrors(dm, grid, Xs, p + MMS_DIAG_QUAD_EXTRA, &sL2hi, &sHchi));
    PetscCall(mmsConditionProbe(dm, grid, A, Bs, Xs, &cond, &floor));
  }

  PetscInt Ndof;
  PetscCall(MatGetSize(A, &Ndof, NULL));

  /* Report. */
  PetscCall(PetscPrintf(comm, "\n MMS verification (order %" PetscInt_FMT ", %s DOFs):\n", p, formatGroupedInt(Ndof)));
  PetscCall(PetscPrintf(comm, "   %-14s L2 %.6e   H(curl) %.6e\n", "solve",      (double)sL2, (double)sHc));
  PetscCall(PetscPrintf(comm, "   %-14s L2 %.6e   H(curl) %.6e\n", "projection", (double)pL2, (double)pHc));
  PetscCall(PetscPrintf(comm, "   %-14s %.3e\n", "residual", (double)residual));
  if (diagnostics) {
    const PetscReal qs = (sL2 > 0.0) ? PetscAbsReal(sL2hi - sL2) / sL2 : 0.0;
    PetscCall(PetscPrintf(comm, "   %-14s quad-sens %.2e   kappa %.2e   floor %.2e\n",
                          "diagnostics", (double)qs, (double)cond, (double)floor));
  }

  /* One HDF5 file per run (omega_scale = 1; the Level-2 attributes are -1 /
   * duplicate when diagnostics are off). */
  PetscCall(mmsWriteH5(comm, params.outputDirectory, params.outputFilename, p, Ndof,
                       1.0, sL2, sHc, pL2, pHc, residual, sL2hi, sHchi, cond, floor));

  PetscCall(MatDestroy(&A));  PetscCall(MatDestroy(&Bs)); PetscCall(MatDestroy(&Xs));
  PetscCall(MatDestroy(&M));  PetscCall(MatDestroy(&Bp)); PetscCall(MatDestroy(&Xp));

  PetscFunctionReturn(PETSC_SUCCESS);
}
