/*
 * Filename: mms.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-08-01
 *
 * Description:
 * Method-of-Manufactured-Solutions (MMS) verification program for the fm.csem
 * high-order Nedelec discretization (build/mms, `make mms`). The manufactured
 * solution lives in tests/mms/mms.h.
 *
 * The verification measures the relative error of the Galerkin solution against
 * the exact field E*, in two norms:
 *
 *   L2      ||E* - E_h||_L2 / ||E*||_L2
 *   energy  |||E* - E_h||| / |||E*|||,
 *           |||v|||^2 = ||curl v||^2 + omega mu0 INT (sigma v) . conj(v)
 *
 * The energy norm is the one the operator induces; Cea quasi-optimality holds
 * in it with the constant sqrt(2), independent of h, p, omega and sigma. Both
 * norms converge as O(h^p) for the first-kind Nedelec family of degree p, and
 * as O(Ndof^(-p/3)) against the unknown count. See data/test1/README.md for the
 * design, the theory and the post-processing protocol.
 *
 * Three levels, selected by runtime flags:
 *   (default)            Level 1 (standard): one run performs the Galerkin
 *                        solve and the L2-projection best-approximation, and
 *                        records both error norms plus the solve residual.
 *   -mms_diagnostics     Level 2: adds the error norms recomputed under an
 *                        over-integrated quadrature rule, which bounds the
 *                        quadrature error committed on the trigonometric f*.
 *   -mms_drop_mass       Level 3 (negative control): assembles the RHS from an
 *                        incomplete forcing (mass term dropped) while still
 *                        measuring against E*. The error then plateaus at O(1)
 *                        instead of converging.
 *
 * Output (in params.outputDirectory, HDF5 as in the rest of PETGEM):
 *   {output_filename}.h5    one file per run, metrics as root-group attributes.
 */

#include <stdio.h>
#include <petsc.h>
#include <petscviewerhdf5.h>

#include "assembly.h"
#include "common.h"
#include "constants.h"
#include "csem.h"
#include "fem.h"
#include "grid.h"
#include "io.h"
#include "mms.h"
#include "runtime.h"
#include "solver.h"
#include "version.h"

/**
 * @brief Additional quadrature order used by the MMS diagnostic check.
 *
 * Added to the finite-element basis order when recomputing the MMS error norms
 * under over-integration. The production rule is exact to degree 2*order+1,
 * which covers the polynomial mass integrand but not the trigonometric
 * manufactured field; the higher-order rule bounds the quadrature error left in
 * the reported metrics.
 */
#define MMS_DIAG_QUAD_EXTRA 3

/**
 * @brief Relative tolerance on the mesh bounding box against MMS_L.
 *
 * The manufactured field is tied to the corner-at-origin cube [0,MMS_L]^3: on
 * any other domain n x E* != 0 on the boundary. A loose tolerance suffices,
 * since this guards against a different mesh, not round-off in the coordinates.
 */
#define MMS_DOMAIN_TOL 1.0e-6

/**
 * @brief Aborts unless the mesh is the cube the manufactured solution assumes.
 *
 * Compares the mesh bounding box against [0,MMS_L]^3. E* has a vanishing
 * tangential trace only on that domain.
 *
 * @param[in] dm  DMPlex mesh.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode mmsCheckDomain(const DM dm) {
  PetscFunctionBeginUser;
  PetscReal bbmin[NUM_DIMENSIONS], bbmax[NUM_DIMENSIONS];
  PetscCall(DMGetBoundingBox(dm, bbmin, bbmax));
  const PetscReal tol = MMS_DOMAIN_TOL * MMS_L;
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    PetscCheck(PetscAbsReal(bbmin[d]) <= tol && PetscAbsReal(bbmax[d] - MMS_L) <= tol, PetscObjectComm((PetscObject)dm),
               PETSC_ERR_ARG_WRONG,
               "MMS: mesh is not the cube [0,%g]^3 the manufactured solution assumes "
               "(axis %" PetscInt_FMT " spans [%g,%g]). n x E* = 0 holds only on that "
               "domain; use data/test1/mesh.geo or change MMS_L in tests/mms/mms.h.",
               (double)MMS_L, d, (double)bbmin[d], (double)bbmax[d]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes global relative L2 and energy MMS error norms.
 *
 * Reconstructs the discrete electric field E_h and curl(E_h) at the
 * quadrature points of every local cell from the ghosted DOF vector.
 * The squared differences against the manufactured exact solution E* and
 * curl(E*) are integrated using the cell Jacobian determinant and quadrature
 * weights, summed over all MPI ranks, and normalized by the exact norms of E*.
 *
 * The energy norm weights the L2 part by omega*mu0*sigma, taking sigma from the
 * cell itself so an anisotropic or heterogeneous conductivity is handled
 * exactly as the operator handles it:
 *
 *   |||v|||^2 = INT |curl v|^2 + omega mu0 INT sum_d sigma_d |v_d|^2
 *
 * The resulting values measure discretization error only; the solver residual
 * is handled separately by mmsResidual().
 *
 * @param[in]  dm         DMPlex mesh.
 * @param[in]  grid       Finite-element discretization information.
 * @param[in]  section    Local DOF layout for dm.
 * @param[in]  xarr       Ghosted solution-vector entries.
 * @param[in]  quadOrder     Quadrature order used for integration.
 * @param[in]  omega         Angular frequency, for the energy-norm weight.
 * @param[in]  conductivity  Per-cell conductivity Vec (the energy weight).
 * @param[out] relL2         Relative L2 error norm.
 * @param[out] relEnergy     Relative energy error norm.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode mmsErrorNormsAtQuad(const DM dm, const Grid *grid, PetscSection section, const PetscScalar *xarr, PetscInt quadOrder,
                                          PetscReal omega, const Vec conductivity, PetscReal *relL2, PetscReal *relEnergy) {
  PetscFunctionBeginUser;

  Cell         cell;
  Quadrature3D quad;
  PetscReal  **Ni, **NiCurl;
  DM           dmConductivity;
  MPI_Comm     comm = PetscObjectComm((PetscObject)dm);

  PetscCall(VecGetDM(conductivity, &dmConductivity));

  /* The closed-form denominators assume a conductivity uniform over the domain.
   * Track the spread so a heterogeneous bundle is rejected. */
  PetscReal sigMin[NUM_DIMENSIONS] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal sigMax[NUM_DIMENSIONS] = {-PETSC_MAX_REAL, -PETSC_MAX_REAL, -PETSC_MAX_REAL};

  PetscCall(createQuadrature3D(quadOrder, &quad));
  PetscCall(createBasisValues(grid->fem.numDofInCell, &Ni));
  PetscCall(createBasisValues(grid->fem.numDofInCell, &NiCurl));

  /* l2sq is the plain L2 error; masssq is its sigma-weighted counterpart, the
   * L2 half of the energy norm. They differ whenever sigma != 1. */
  PetscReal l2sq = 0.0, curlsq = 0.0, masssq = 0.0;

  for (PetscInt c = grid->cellStart; c < grid->cellEnd; ++c) {

    PetscCall(extractCellCoordinates(dm, c, &cell));
    PetscCall(computeCellJacobian(&cell));
    const PetscReal absdet = PetscAbsReal(cell.detJacobian);

    /* The energy weight is the cell's own conductivity, matching Me. */
    PetscCall(extractCellConductivity(dmConductivity, conductivity, c, &cell));
    const PetscReal sigma[NUM_DIMENSIONS] = {cell.conductivity[0], cell.conductivity[1], cell.conductivity[2]};
    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
      sigMin[d] = PetscMin(sigMin[d], sigma[d]);
      sigMax[d] = PetscMax(sigMax[d], sigma[d]);
    }

    PetscInt numLocal, *localIdx;
    PetscCall(DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE, &numLocal, &localIdx, NULL, NULL));

    for (PetscInt q = 0; q < quad.numPoints; q++) {
      const PetscReal *ref = quad.points[q];

      PetscReal xphys[NUM_DIMENSIONS];
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
        xphys[d] = cell.coordinates[d] + cell.jacobian[0][d] * ref[0] + cell.jacobian[1][d] * ref[1] + cell.jacobian[2][d] * ref[2];
      }

      PetscCall(evaluateNedelecBasis(&grid->fem, &cell, ref, Ni, NiCurl));

      PetscScalar Eh[NUM_DIMENSIONS] = {0.0, 0.0, 0.0};
      PetscScalar Ch[NUM_DIMENSIONS] = {0.0, 0.0, 0.0};
      for (PetscInt j = 0; j < grid->fem.numDofInCell; j++) {
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

      const PetscReal wdet = quad.weights[q] * absdet;
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
        const PetscScalar de  = Eh[d] - Eex[d];
        const PetscScalar dc  = Ch[d] - Cex[d];
        const PetscReal   de2 = PetscRealPart(de * PetscConj(de));
        l2sq += wdet * de2;
        masssq += wdet * sigma[d] * de2;
        curlsq += wdet * PetscRealPart(dc * PetscConj(dc));
      }
    }

    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE, &numLocal, &localIdx, NULL, NULL));
  }

  PetscReal partial[3] = {l2sq, curlsq, masssq};
  PetscReal global[3]  = {0.0, 0.0, 0.0};
  PetscCallMPI(MPI_Allreduce(partial, global, 3, MPIU_REAL, MPI_SUM, comm));

  /* Denominators: the closed-form norms of E*, evaluated with the domain's
   * uniform sigma. Reduced across ranks so every rank agrees and a rank that
   * owns no cell contributes nothing. */
  PetscReal sigmaRef[NUM_DIMENSIONS], sigmaLo[NUM_DIMENSIONS];
  PetscCallMPI(MPI_Allreduce(sigMax, sigmaRef, NUM_DIMENSIONS, MPIU_REAL, MPI_MAX, comm));
  PetscCallMPI(MPI_Allreduce(sigMin, sigmaLo, NUM_DIMENSIONS, MPIU_REAL, MPI_MIN, comm));
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    PetscCheck(PetscAbsReal(sigmaRef[d] - sigmaLo[d]) <= PETSC_SMALL * PetscAbsReal(sigmaRef[d]), comm, PETSC_ERR_ARG_WRONG,
               "MMS: conductivity is not uniform over the domain (component %" PetscInt_FMT
               " spans [%g,%g]). The closed-form reference norms assume one material; a "
               "piecewise-constant sigma would also make E* non-smooth and cap the "
               "convergence rate below p.",
               d, (double)sigmaLo[d], (double)sigmaRef[d]);
  }

  *relL2     = PetscSqrtReal(global[0]) / mmsRefNormL2();
  *relEnergy = PetscSqrtReal(global[1] + omega * MU * global[2]) / mmsRefNormEnergy(omega, sigmaRef);

  PetscCall(destroyQuadrature3D(&quad));
  PetscCall(destroyBasisValues(&Ni));
  PetscCall(destroyBasisValues(&NiCurl));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes MMS error norms for the first column of a solution matrix.
 *
 * Extracts column 0 from the dense solution matrix, creates the required
 * local ghosted representation through DMGlobalToLocal, and evaluates the
 * relative L2 and energy MMS error norms using mmsErrorNormsAtQuad().
 *
 * PETGEM MMS solves produce a single right-hand side, so the first matrix
 * column contains the solution of interest.
 *
 * @param[in]  dm         DMPlex mesh.
 * @param[in]  grid       Finite-element discretization information.
 * @param[in]  X          Dense matrix containing the solution vector.
 * @param[in]  quadOrder     Quadrature order used for error integration.
 * @param[in]  omega         Angular frequency, for the energy-norm weight.
 * @param[in]  conductivity  Per-cell conductivity Vec (the energy weight).
 * @param[out] relL2         Relative L2 error norm.
 * @param[out] relEnergy     Relative energy error norm.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode mmsColumnErrors(const DM dm, const Grid *grid, const Mat X, PetscInt quadOrder, PetscReal omega,
                                      const Vec conductivity, PetscReal *relL2, PetscReal *relEnergy) {
  PetscFunctionBeginUser;
  PetscSection       section;
  Vec                x, xloc;
  const PetscScalar *xarr;
  PetscCall(MatDenseGetColumnVecRead(X, 0, &x));
  PetscCall(DMGetLocalVector(dm, &xloc));
  PetscCall(DMGlobalToLocalBegin(dm, x, INSERT_VALUES, xloc));
  PetscCall(DMGlobalToLocalEnd(dm, x, INSERT_VALUES, xloc));
  PetscCall(MatDenseRestoreColumnVecRead(X, 0, &x));
  PetscCall(VecGetArrayRead(xloc, &xarr));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(mmsErrorNormsAtQuad(dm, grid, section, xarr, quadOrder, omega, conductivity, relL2, relEnergy));
  PetscCall(VecRestoreArrayRead(xloc, &xarr));
  PetscCall(DMRestoreLocalVector(dm, &xloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the relative residual norm of an MMS solve.
 *
 * Forms the residual r = A x - b using column 0 of the solution and
 * right-hand-side matrices, computes ||r||_2 and ||b||_2, and returns the
 * backward-error estimate ||A x - b||_2 / ||b||_2.
 *
 * With a direct solver this sits at round-off, showing the algebraic error is
 * negligible next to the discretization error.
 *
 * If the right-hand side is identically zero, the absolute residual norm is
 * returned instead.
 *
 * @param[in]  A       System matrix.
 * @param[in]  B       Right-hand-side matrix.
 * @param[in]  X       Solution matrix.
 * @param[out] relRes  Relative residual norm.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode mmsResidual(const Mat A, const Mat B, const Mat X, PetscReal *relRes) {
  PetscFunctionBeginUser;
  Vec       xc, bc, r;
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
 * @brief Assembles the volumetric manufactured-source RHS for MMS verification.
 *
 * MMS counterpart of assembleCsemRhs: instead of a Dirac dipole located in a
 * single cell, it integrates the manufactured forcing f* (tests/mms/mms.h) over
 * every local cell,
 *
 *     b_j = sum_cells sum_q  w_q * detJ_cell * ( N_j(x_q) . f*(x_q) ),
 *
 * reusing the quadrature rule and (weights, detJ) measure as
 * computeElementalMatrices, so the RHS is consistent with the operator
 * A = K - i omega mu Ms that assembleMaxwellOperator builds. Contributions from cells
 * sharing an edge/face DOF accumulate (ADD_VALUES); boundary DOFs are skipped
 * (negative closure indices + VEC_IGNORE_NEGATIVE_INDICES), which is exact since
 * n x E* = 0. The single manufactured RHS is written into a one-column dense B.
 *
 * @param[in]  omega         Angular frequency.
 * @param[in]  dm            DMPlex mesh and H(curl) discretization.
 * @param[in]  grid          Finite-element grid descriptor.
 * @param[in]  conductivity  Per-cell conductivity Vec (diagonal sigma in f*).
 * @param[in]  kind          Which right-hand side to build (see MmsRhsKind).
 * @param[out] B             One-column dense RHS matrix, created by this call.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 *
 * @note The caller is responsible for destroying the returned matrix `B`.
 */
static PetscErrorCode assembleMmsRhs(const PetscReal omega, const DM dm, const Grid *grid, const Vec conductivity, const MmsRhsKind kind,
                                     Mat *B) {
  PetscFunctionBeginUser;

  /* The negative control uses the same operator and only alters f*. */
  const PetscBool useForcing = (PetscBool)(kind != MMS_RHS_PROJECTION);
  const PetscBool dropMass   = (PetscBool)(kind == MMS_RHS_FORCING_NO_MASS);

  /* Variables declaration */
  Cell         cell;
  Quadrature3D quadrature3D;
  PetscInt     numDofIndices, *dofIndices;
  PetscReal  **Ni;
  PetscScalar *closureRhs;
  PetscSection section;
  DM           dmConductivity;
  Vec          b, bcol;
  MPI_Comm     comm = PetscObjectComm((PetscObject)dm);

  /* One right-hand side (the single manufactured forcing) */
  PetscCall(createRhsVector(dm, &b));
  PetscCall(createRhsMatrix(b, 1, B));

  /* DM section and conductivity DM. */
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(VecGetDM(conductivity, &dmConductivity));

  /* Same quadrature rule as the LHS element integrals. */
  PetscCall(createQuadrature3D(grid->fem.order, &quadrature3D));

  /* Basis-value and closure scratch. */
  PetscCall(PetscCalloc1(grid->fem.numDofInCell, &closureRhs));
  PetscCall(createBasisValues(grid->fem.numDofInCell, &Ni));

  const char *what = (kind == MMS_RHS_PROJECTION)        ? "L2 moments of E*"
                     : (kind == MMS_RHS_FORCING_NO_MASS) ? "INCOMPLETE forcing f* (negative control)"
                                                         : "volumetric forcing f*";
  PetscCall(logKvStr(comm, "MMS RHS", what));

  PetscCall(VecZeroEntries(b));

  /* Volumetric integral of f* against the Nedelec basis, cell by cell. */
  for (PetscInt c = grid->cellStart; c < grid->cellEnd; ++c) {

    /* Geometry + conductivity for cell c; computeCellJacobian fills
     * cell.jacobian (rows v1-v0, v2-v0, v3-v0) and cell.detJacobian, which
     * match femComputeJacobian used by evaluateNedelecBasis / the LHS. */
    PetscCall(extractCellCoordinates(dm, c, &cell));
    PetscCall(extractCellConductivity(dmConductivity, conductivity, c, &cell));
    PetscCall(computeCellJacobian(&cell));
    const PetscReal det                   = cell.detJacobian;
    const PetscReal sigma[NUM_DIMENSIONS] = {cell.conductivity[0], cell.conductivity[1], cell.conductivity[2]};

    PetscCall(DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));

    for (PetscInt j = 0; j < grid->fem.numDofInCell; j++) {
      closureRhs[j] = 0.0;
    }

    for (PetscInt q = 0; q < quadrature3D.numPoints; q++) {
      const PetscReal *ref = quadrature3D.points[q];

      /* Physical image of the reference quadrature point: x = v0 + F*ref,
       * with F = J^T, i.e. x_d = coords_d + sum_k jacobian[k][d]*ref[k]. */
      PetscReal xphys[NUM_DIMENSIONS];
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
        xphys[d] = cell.coordinates[d] + cell.jacobian[0][d] * ref[0] + cell.jacobian[1][d] * ref[1] + cell.jacobian[2][d] * ref[2];
      }

      /* useForcing: integrate f* (the MMS solve RHS). Otherwise integrate the
       * SIGMA-WEIGHTED exact field -> the moments of the projection baseline.
       * The operator paired with this RHS is Me = INT (sigma N_j) . N_k, so the
       * sigma weight makes the solution the sigma-weighted L2 projection of E*.
       * Without it the solve would converge to sigma^-1 E*. */
      PetscScalar F[NUM_DIMENSIONS];
      if (useForcing) {
        mmsForcingF(xphys, omega, sigma, dropMass, F);
      } else {
        mmsExactE(xphys, F);
        for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
          F[d] *= (PetscScalar)sigma[d];
        }
      }

      /* Physical, oriented basis values at this reference point (no curls). */
      PetscCall(evaluateNedelecBasis(&grid->fem, &cell, ref, Ni, NULL));

      const PetscReal wdet = quadrature3D.weights[q] * det;
      for (PetscInt j = 0; j < grid->fem.numDofInCell; j++) {
        PetscScalar dot = 0.0;
        for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
          dot += (PetscScalar)Ni[d][j] * F[d];
        }
        closureRhs[j] += wdet * dot;
      }
    }

    /* Accumulate this cell's contribution (shared DOFs sum across cells). */
    PetscCall(VecSetValuesLocal(b, numDofIndices, dofIndices, closureRhs, ADD_VALUES));
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
  }

  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  /* Copy the assembled RHS into the single column of B (no iωμ scaling: f* already carries the frequency/permeability factor). */
  PetscCall(MatDenseGetColumnVecWrite(*B, 0, &bcol));
  PetscCall(VecCopy(b, bcol));
  PetscCall(MatDenseRestoreColumnVecWrite(*B, 0, &bcol));

  PetscCall(VecDestroy(&b));

  /* Free memory */
  PetscCall(destroyQuadrature3D(&quadrature3D));
  PetscCall(destroyBasisValues(&Ni));
  PetscCall(PetscFree(closureRhs));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Assembles and solves one MMS verification system.
 *
 * Builds the requested MMS right-hand side and the matching operator, solves
 * the resulting linear system, and returns the assembled matrices and solution.
 *
 * For MMS_RHS_FORCING (and the MMS_RHS_FORCING_NO_MASS control) the routine
 * assembles the physical Galerkin problem:
 *
 *   A = K - i omega mu Ms
 *   b = INT f* . N
 *
 * For MMS_RHS_PROJECTION it assembles the L2 projection problem instead:
 *
 *   A = Ms
 *   b = INT E* . N
 *
 * whose solution is the best-approximation baseline the Galerkin error is
 * compared against.
 *
 * The caller assumes ownership of the returned matrices and solution.
 *
 * @param[in]  omega         Angular frequency.
 * @param[in]  dm            DMPlex mesh.
 * @param[in]  grid          Finite-element discretization information.
 * @param[in]  conductivity  Cell conductivity field.
 * @param[in]  constFactor   Frequency-dependent mass coefficient.
 * @param[in]  kind          Which right-hand side to build.
 * @param[out] A             Assembled system matrix.
 * @param[out] B             Assembled right-hand-side matrix.
 * @param[out] X             Computed solution matrix.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode mmsAssembleSolve(const PetscReal omega, const DM dm, const Grid *grid, const Vec conductivity,
                                       const PetscScalar constFactor, const MmsRhsKind kind, Mat *A, Mat *B, Mat *X) {
  PetscFunctionBeginUser;
  PetscCall(assembleMmsRhs(omega, dm, grid, conductivity, kind, B));
  if (kind == MMS_RHS_PROJECTION) {
    Mat Kthrow;
    PetscCall(assembleMaxwellOperator(dm, grid, conductivity, constFactor, &Kthrow, A, NULL));
    PetscCall(MatDestroy(&Kthrow));
  } else {
    PetscCall(assembleMaxwellOperator(dm, grid, conductivity, constFactor, A, NULL, NULL));
  }
  SolveInfo info;
  PetscCall(solveMaxwellSystem(dm, *A, *B, NULL, grid->fem.order, NULL, X, &info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Writes MMS verification metrics to an HDF5 results file.
 *
 * Creates {dir}/{fname}.h5 and stores all MMS measurements as root-group
 * HDF5 attributes, together with PETGEM version and execution-date provenance
 * information. One file corresponds to one MMS run; data/test1/postprocessing_mms.py
 * reads a whole directory of them.
 *
 * All numerical values are assumed to be globally reduced before the call. The
 * operation is collective on the supplied communicator.
 *
 * @param[in] comm             MPI communicator used for HDF5 I/O.
 * @param[in] dir              Output directory.
 * @param[in] fname            Output filename without extension.
 * @param[in] order            FEM basis order.
 * @param[in] dofs             Global number of degrees of freedom.
 * @param[in] cells            Global number of cells (gives the mesh spacing).
 * @param[in] meshH            Lattice spacing h = L (6/cells)^(1/3).
 * @param[in] solveL2          Galerkin-solve relative L2 error.
 * @param[in] solveEnergy      Galerkin-solve relative energy error.
 * @param[in] projL2           Projection relative L2 error.
 * @param[in] projEnergy       Projection relative energy error.
 * @param[in] residual         Relative solve residual.
 * @param[in] solveL2hi        Over-integrated L2 error.
 * @param[in] solveEnergyhi    Over-integrated energy error.
 * @param[in] forcingComplete  0 on a negative-control run, 1 otherwise.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode mmsWriteH5(MPI_Comm comm, const char *dir, const char *fname, PetscInt order, PetscInt dofs, PetscInt cells,
                                 PetscReal meshH, PetscReal solveL2, PetscReal solveEnergy, PetscReal projL2, PetscReal projEnergy,
                                 PetscReal residual, PetscReal solveL2hi, PetscReal solveEnergyhi, PetscInt forcingComplete) {
  PetscFunctionBeginUser;
  char        path[PETSC_MAX_PATH_LEN], version[64], date[30];
  PetscViewer v;

  PetscCall(PetscStrncpy(path, dir, sizeof(path)));
  size_t len = strlen(path);
  if (len > 0 && path[len - 1] != '/') {
    PetscCall(PetscStrlcat(path, "/", sizeof(path)));
  }
  PetscCall(PetscStrlcat(path, fname, sizeof(path)));
  PetscCall(PetscStrlcat(path, ".h5", sizeof(path)));

  PetscCall(PetscViewerHDF5Open(comm, path, FILE_MODE_WRITE, &v));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "order", PETSC_INT, &order));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "dofs", PETSC_INT, &dofs));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "cells", PETSC_INT, &cells));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "mesh_h", PETSC_REAL, &meshH));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "solve_L2", PETSC_REAL, &solveL2));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "solve_energy", PETSC_REAL, &solveEnergy));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "proj_L2", PETSC_REAL, &projL2));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "proj_energy", PETSC_REAL, &projEnergy));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "residual", PETSC_REAL, &residual));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "solve_L2_hi", PETSC_REAL, &solveL2hi));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "solve_energy_hi", PETSC_REAL, &solveEnergyhi));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "forcing_complete", PETSC_INT, &forcingComplete));

  /* Provenance of the manufactured solution, so a result file records the
   * design point it was produced under. */
  {
    PetscReal Lref = MMS_L;
    PetscInt  mode = MMS_MODE;
    PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "mms_L", PETSC_REAL, &Lref));
    PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "mms_mode", PETSC_INT, &mode));
  }

  snprintf(version, sizeof(version), "%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "petgem_version", PETSC_STRING, version));
  PetscCall(PetscGetDate(date, sizeof(date)));
  PetscCall(PetscViewerHDF5WriteAttribute(v, NULL, "date", PETSC_STRING, date));

  PetscCall(PetscViewerDestroy(&v));
  PetscCall(PetscPrintf(comm, "   %-14s %s\n", "wrote", path));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Executes the complete MMS verification workflow.
 *
 * Single entry point for all MMS validation modes.
 *
 * Standard execution (Level 1) performs a Galerkin solve and an L2-projection
 * solve, computes relative L2 and energy errors for both, evaluates the solve
 * residual, reports the results and writes a single HDF5 file.
 *
 * With -mms_diagnostics (Level 2) the routine additionally recomputes the
 * solution norms under an over-integrated quadrature rule, bounding the
 * quadrature error committed on the trigonometric forcing.
 *
 * With -mms_drop_mass (Level 3) the right-hand side is built from an incomplete
 * forcing, so E* is not the solution of the discrete problem and the measured
 * error plateaus instead of converging. The projection pass is skipped there.
 *
 * @param[in] params        Parameters (output paths).
 * @param[in] dm            DMPlex mesh.
 * @param[in] grid          Finite-element discretization information.
 * @param[in] conductivity  Cell conductivity field.
 * @param[in] frequency     Frequency (Hz).
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
static PetscErrorCode runMmsVerification(const PetgemParams *params, const DM dm, const Grid *grid, const Vec conductivity,
                                         const PetscReal frequency) {
  PetscFunctionBeginUser;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  PetscBool diagnostics = PETSC_FALSE, dropMass = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mms_diagnostics", &diagnostics, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mms_drop_mass", &dropMass, NULL));

  /* E* is valid only on [0,MMS_L]^3; refuse to report numbers otherwise. */
  PetscCall(mmsCheckDomain(dm));

  const PetscReal   omega       = angularFrequency(frequency);
  const PetscScalar constFactor = iOmegaMu(frequency);
  const PetscInt    p           = grid->fem.order;
  const MmsRhsKind  solveKind   = dropMass ? MMS_RHS_FORCING_NO_MASS : MMS_RHS_FORCING;

  /* Galerkin solve (the quantity under test). */
  Mat A, Bs, Xs;
  PetscCall(mmsAssembleSolve(omega, dm, grid, conductivity, constFactor, solveKind, &A, &Bs, &Xs));

  PetscReal sL2 = 0.0, sEn = 0.0, pL2 = -1.0, pEn = -1.0, residual = -1.0;
  PetscCall(mmsColumnErrors(dm, grid, Xs, p, omega, conductivity, &sL2, &sEn));
  PetscCall(mmsResidual(A, Bs, Xs, &residual));

  /* L2-projection best-approximation baseline; skipped for the negative
   * control. */
  Mat M = NULL, Bp = NULL, Xp = NULL;
  if (!dropMass) {
    PetscCall(mmsAssembleSolve(omega, dm, grid, conductivity, constFactor, MMS_RHS_PROJECTION, &M, &Bp, &Xp));
    PetscCall(mmsColumnErrors(dm, grid, Xp, p, omega, conductivity, &pL2, &pEn));
  }

  /* Level 2: over-integrated solve norms. */
  PetscReal sL2hi = sL2, sEnhi = sEn;
  if (diagnostics) {
    PetscCall(mmsColumnErrors(dm, grid, Xs, p + MMS_DIAG_QUAD_EXTRA, omega, conductivity, &sL2hi, &sEnhi));
  }

  PetscInt Ndof;
  PetscCall(MatGetSize(A, &Ndof, NULL));

  /* Lattice spacing of the structured cube: 6 tetrahedra per lattice cube, so
   * h = L (6/cells)^(1/3). The abscissa of the convergence study. */
  const PetscReal meshH = MMS_L * PetscCbrtReal(6.0 / (PetscReal)grid->numCellsGlobal);

  /* Report. */
  PetscCall(
      PetscPrintf(comm, "\n MMS verification (order %" PetscInt_FMT ", %s DOFs, h = %.4g m):\n", p, formatGroupedInt(Ndof), (double)meshH));
  if (dropMass) {
    PetscCall(PetscPrintf(comm, "   %-14s mass term dropped from f*; the error does not converge\n", "CONTROL"));
  }
  PetscCall(PetscPrintf(comm, "   %-14s L2 %.6e   energy %.6e\n", "solve", (double)sL2, (double)sEn));
  if (!dropMass) {
    PetscCall(PetscPrintf(comm, "   %-14s L2 %.6e   energy %.6e\n", "projection", (double)pL2, (double)pEn));
  }
  PetscCall(PetscPrintf(comm, "   %-14s %.3e\n", "residual", (double)residual));
  if (diagnostics) {
    const PetscReal qs = (sL2 > 0.0) ? PetscAbsReal(sL2hi - sL2) / sL2 : 0.0;
    PetscCall(PetscPrintf(comm, "   %-14s quad-sens %.2e\n", "diagnostics", (double)qs));
  }

  PetscCall(mmsWriteH5(comm, params->outputDirectory, params->outputFilename, p, Ndof, grid->numCellsGlobal, meshH, sL2, sEn, pL2, pEn,
                       residual, sL2hi, sEnhi, dropMass ? 0 : 1));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Bs));
  PetscCall(MatDestroy(&Xs));
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&Bp));
  PetscCall(MatDestroy(&Xp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static char help[] = "================================================================\n\
 PETGEM  mms  --  MMS order-of-accuracy verification\n\
================================================================\n\
\n\
 Solves the manufactured-solution problem on the [0,1000]^3 cube\n\
 and writes the relative L2 and energy errors of the Galerkin\n\
 solution and of the L2 projection to <output_dir>/<stem>.h5.\n\
\n\
QUICK START\n\
  mpirun -n <np> ./mms -input_filename <bundle> -order <p>\n\
         -output_dir <dir> -output_filename <stem>\n\
\n\
OPTIONS\n\
  -mms_diagnostics  Also report over-integrated error norms.\n\
  -mms_drop_mass    Negative control: mass term dropped from f*.\n\
\n" PETGEM_HELP_FOOTER;

int main(int argc, char **argv) {
  if (isVersionRequest(argc, argv)) return printVersion();

  KernelRun     run;
  PetgemParams  params;
  DM            dm;
  Vec           conductivity, materialsId, receivers;
  CsemSourceSet sources = {0, 0, NULL};
  Grid          grid;

  PetscFunctionBeginUser;
  PetscCall(beginKernelRun(&argc, &argv, help, "mms", "MMS errors", &run));
  PetscCall(readPetgemParams(&params));
  if (run.helpRequested) {
    PetscCall(endHelpRun(&run));
    return 0;
  }

  PetscCall(loadModelInputs(&params, &dm, &conductivity, &materialsId, &receivers));
  PetscCall(loadCsemSources(&params, &sources));
  PetscCall(setupCsemGrid(params.order, &dm, &grid));
  PetscCall(logGridSummary(params.inputFile, dm, &grid));
  PetscCall(runMmsVerification(&params, dm, &grid, conductivity, sources.freq));
  PetscCall(printFooter());

  PetscCall(destroyGrid(&grid));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&conductivity));
  PetscCall(VecDestroy(&materialsId));
  PetscCall(VecDestroy(&receivers));
  PetscCall(PetscFree(sources.sourceArray));
  PetscCall(PetscFinalize());
  return 0;
}
