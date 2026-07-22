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

/**
 * @brief Additional quadrature order used by MMS diagnostic checks.
 *
 * Added to the finite-element basis order when recomputing MMS error
 * norms under over-integration. The resulting higher-order quadrature
 * provides a simple sensitivity check for quadrature-induced error in
 * the reported verification metrics.
 */
#define MMS_DIAG_QUAD_EXTRA 3


/**
 * @brief Mass-term scaling factors used by the Level-3 conditioning sweep.
 *
 * Each value scales the physical frequency (and therefore the mass term
 * iωμM) before reassembling and solving the MMS system. The sweep probes
 * how solution accuracy and estimated conditioning change as the operator
 * becomes increasingly mass dominated.
 */
static const PetscReal MMS_COND_SCALES[] = {1.0, 100.0, 10000.0};


/**
 * @brief Number of mass-term scales evaluated by the conditioning sweep.
 *
 * Defines the number of entries stored in MMS_COND_SCALES and the number
 * of MMS solves performed during a Level-3 conditioning analysis.
 */
#define MMS_COND_NUM_SCALES 3


/**
 * @brief Computes global relative L2 and H(curl) MMS error norms.
 *
 * Reconstructs the discrete electric field E_h and curl(E_h) at the
 * quadrature points of every local cell from the ghosted DOF vector.
 * The squared differences against the manufactured exact solution
 * E* and curl(E*) are integrated using the cell Jacobian determinant
 * and quadrature weights, summed over all MPI ranks, and normalized
 * by the reference MMS norms.
 *
 * The resulting values measure discretization error only; solver
 * residuals are handled separately by mmsResidual().
 *
 * @param[in]  dm         DMPlex mesh.
 * @param[in]  grid       Finite-element discretization information.
 * @param[in]  section    Local DOF layout for dm.
 * @param[in]  xarr       Ghosted solution-vector entries.
 * @param[in]  quadOrder  Quadrature order used for integration.
 * @param[out] relL2      Relative L2 error norm.
 * @param[out] relHcurl   Relative H(curl) error norm.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
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


/**
 * @brief Computes MMS error norms for the first column of a solution matrix.
 *
 * Extracts column 0 from the dense solution matrix, creates the required
 * local ghosted representation through DMGlobalToLocal, and evaluates the
 * relative L2 and H(curl) MMS error norms using
 * mmsErrorNormsAtQuad().
 *
 * PETGEM MMS solves currently produce a single right-hand side, so the
 * first matrix column contains the solution of interest.
 *
 * @param[in]  dm         DMPlex mesh.
 * @param[in]  grid       Finite-element discretization information.
 * @param[in]  X          Dense matrix containing the solution vector.
 * @param[in]  quadOrder  Quadrature order used for error integration.
 * @param[out] relL2      Relative L2 error norm.
 * @param[out] relHcurl   Relative H(curl) error norm.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
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


/**
 * @brief Computes the rela*ive residual norm of an MMS solve.* *
 * Forms the residual vector r * A x - b using column 0 of the sol*tion
 * and right-hand-side matric*s, computes ||r||₂ and ||b||₂, and*returns
 * the backward-error esti*ate ||A x - b||₂ / ||b||₂.
 *
 * I* the right-hand side is identicall* zero, the absolute residual
 * no*m is returned instead.
 *
 * @para*[in]  A       System matrix.
 * @p*ram[in]  B       Right-hand-side m*trix.
 * @param[in]  X       Solut*on matrix.
 * @param[out] relRes  *elative residual norm.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
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
 * @brief Estimates conditioning and attainable forward accuracy.
 *
 * Perturbs the right-hand side by a small relative amount, resolves the
 * linear system, and measures the induced relative solution change. The
 * amplification factor provides an estimate of the operator condition
 * number, while kappa(A) multiplied by machine precision estimates the
 * practical forward-error floor imposed by finite-precision arithmetic.
 *
 * The estimate is intended as a diagnostic indicator rather than a
 * rigorous condition-number computation.
 *
 * @param[in]  dm        DMPlex mesh.
 * @param[in]  grid      Finite-element discretization information.
 * @param[in]  A         System matrix.
 * @param[in]  B         Right-hand-side matrix.
 * @param[in]  X         Reference solution matrix.
 * @param[out] condEst   Estimated condition number.
 * @param[out] floorEst  Estimated forward-error floor.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
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
 * @brief Assembles and solves one MMS verification system.
 *
 * Builds the MMS right-hand side and associated operator, then solves
 * the resulting linear system and returns the assembled matrices and
 * solution.
 *
 * When useForcing is PETSC_TRUE, the routine assembles the physical
 * Galerkin MMS problem:
 *
 *   A = K - iωμM
 *   b = ∫ f* · N
 *
 * When useForcing is PETSC_FALSE, the routine assembles the L2
 * projection problem:
 *
 *   A = M
 *   b = ∫ E* · N
 *
 * The caller assumes ownership of the returned matrices and solution.
 *
 * @param[in]  params        PETGEM runtime parameters.
 * @param[in]  sources       MMS source configuration.
 * @param[in]  dm            DMPlex mesh.
 * @param[in]  grid          Finite-element discretization information.
 * @param[in]  conductivity  Cell conductivity field.
 * @param[in]  constFactor   Frequency-dependent mass coefficient.
 * @param[in]  useForcing    Select Galerkin solve or L2 projection.
 * @param[out] A             Assembled system matrix.
 * @param[out] B             Assembled right-hand-side matrix.
 * @param[out] X             Computed solution matrix.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
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
 * @brief Writes MMS verification metrics to an HDF5 results file.
 *
 * Creates {dir}/{fname}.h5 and stores all MMS measurements as root-group
 * HDF5 attributes, together with PETGEM version and execution-date
 * provenance information. One file corresponds to one MMS run or one
 * conditioning-sweep point.
 *
 * All numerical values are assumed to be globally reduced before the
 * call. The operation is collective on the supplied communicator.
 *
 * @param[in] comm          MPI communicator used for HDF5 I/O.
 * @param[in] dir           Output directory.
 * @param[in] fname         Output filename without extension.
 * @param[in] order         FEM basis order.
 * @param[in] dofs          Global number of degrees of freedom.
 * @param[in] omegaScale    Frequency scaling factor.
 * @param[in] solveL2       Galerkin-solve relative L2 error.
 * @param[in] solveHcurl    Galerkin-solve relative H(curl) error.
 * @param[in] projL2        Projection relative L2 error.
 * @param[in] projHcurl     Projection relative H(curl) error.
 * @param[in] residual      Relative solve residual.
 * @param[in] solveL2hi     Over-integrated L2 error.
 * @param[in] solveHcurlhi  Over-integrated H(curl) error.
 * @param[in] condEst       Estimated condition number.
 * @param[in] floorEst      Estimated forward-error floor.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
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
 * @brief Performs the MMS Level-3 conditioning study.
 *
 * Reassembles and resolves the MMS problem for each mass-term scaling
 * factor defined in MMS_COND_SCALES. For every solve, the routine
 * computes discretization errors, estimates operator conditioning, and
 * writes a dedicated HDF5 results file.
 *
 * The scale=1 case corresponds to the physical MMS problem, making the
 * sweep self-contained and allowing it to replace the standard MMS
 * verification workflow.
 *
 * @param[in] params        PETGEM runtime parameters.
 * @param[in] sources       MMS source configuration.
 * @param[in] dm            DMPlex mesh.
 * @param[in] grid          Finite-element discretization information.
 * @param[in] conductivity  Cell conductivity field.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
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
 * @brief Executes the complete MMS verification workflow.
 *
 * Serves as the single entry point for all MMS validation modes.
 *
 * Standard MMS execution (Level 1) performs a Galerkin solve and an
 * L2-projection solve, computes relative L2 and H(curl) errors, evaluates
 * the solve residual, reports the results, and writes a single HDF5 file.
 *
 * With -mms_diagnostics enabled (Level 2), the routine additionally
 * recomputes the solution norms using an over-integrated quadrature rule
 * and estimates operator conditioning and the corresponding forward-error
 * floor.
 *
 * With -mms_conditioning enabled (Level 3), the standard workflow is
 * skipped and replaced by a mass-term scaling sweep performed by
 * mmsConditioningSweep().
 *
 * @param[in] params        PETGEM runtime parameters.
 * @param[in] dm            DMPlex mesh.
 * @param[in] grid          Finite-element discretization information.
 * @param[in] conductivity  Cell conductivity field.
 * @param[in] sources       MMS source configuration.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
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
