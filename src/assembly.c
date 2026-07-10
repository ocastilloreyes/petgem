/*
 * Filename: assembly.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * This file contains functions for assembling the linear system
 * (CSEM) in a PETGEM simulation.
 */

/* PETSc libraries */
#include <petscsys.h>

/* PETGEM functions */
#include "common.h"
#include "assembly.h"
#include "constants.h"
#include "fem.h"
#include "grid.h"
#include "io.h"
#include "mms.h"
#include "transmitter.h"


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
 * @param[in]  cellID          Cell index.
 * @param[out] cell            Cell descriptor filled with coordinates and
 *                             conductivity.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
static PetscErrorCode prepareCellForAssembly(const DM dm,
                                             const DM dmConductivity,
                                             const Vec conductivity,
                                             PetscInt cellID,
                                             Cell *cell)
{
  PetscFunctionBeginUser;
  PetscCall(extractCellCoordinates(dm, cellID, cell));
  PetscCall(extractCellConductivity(dmConductivity, conductivity, cellID, cell));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Assembles the right-hand side matrix for the CSEM system.
 *
 * Constructs the RHS vectors for every transmitter in the simulation; each
 * column of `B` corresponds to one source. Performs finite-element assembly
 * using H(curl) Nédélec elements, taking the source dipole orientation,
 * rotation, and position into account.
 *
 * For each source the function (a) locates it in the mesh via locatePoint,
 * (b) extracts cell coordinates and computes the Jacobian, (c) maps the
 * source position to reference coordinates (ξ, η, ζ), (d) evaluates the
 * Nédélec basis and its curls there, (e) writes the local contributions
 * into a per-source RHS Vec, and (f) copies the assembled Vec into the
 * corresponding column of `B`. A final complex scaling by iωμ is applied.
 *
 * @param[in]  params   Forward-modeling parameters (order, MPI tasks).
 * @param[in]  sources  Transmitter set (one column of B per source).
 * @param[in]  dm       DMPlex mesh and H(curl) discretization.
 * @param[in]  grid     Finite-element grid descriptor.
 * @param[in]  constFactor  Fused-mode factor iωμ; ignored when Ms != NULL.
 * @param[out] B        Assembled RHS matrix, created and populated by this call.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful assembly,
 *         or a PETSc error code on failure of vector/matrix creation,
 *         FEM basis evaluation, or point location.
 *
 * @note The caller is responsible for destroying the returned matrix `B`.
 */
PetscErrorCode assembleCsemRHS(const fmParams params, 
                               const CsemSourceSet sources, 
                               const DM dm, 
                               const Grid grid, 
                               const PetscScalar constFactor,
                               Mat* B) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  Cell cell;
  PetscInt m, M, numDofIndices, *dofIndices;
  PetscReal **Ni, *XiEtaZeta;
  PetscScalar *closureRHS;
  PetscSection section;
  Vec b, bcol;
  VecType vtype;
  ISLocalToGlobalMapping mapping;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* Create vector to store one right-hand side */
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMGetLocalToGlobalMapping(dm, &mapping));
  PetscCall(VecSetLocalToGlobalMapping(b, mapping));
  PetscCall(VecSetOption(b, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE));
  PetscCall(VecSetFromOptions(b));

  /* Get vector size */
  PetscCall(VecGetSize(b, &M));
  PetscCall(VecGetLocalSize(b, &m));

  /* Create matrix to hold multiple right-hand sides */
  PetscCall(VecGetType(b, &vtype));
  PetscCall(MatCreateDenseFromVecType(comm, vtype, m, PETSC_DECIDE, M, sources.numSources, m, NULL, B));

  /* Get DM section */
  PetscCall(DMGetLocalSection(dm, &section));

  /* Allocate memory for RHS */
  PetscCall(PetscCalloc1(grid.numDofInCell, &closureRHS));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &XiEtaZeta));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Ni));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(grid.numDofInCell, &Ni[i]));
  }

  /* Print linear system statistics (suppressed when params.quiet) */
  if (!params.quiet) {
    PetscCall(PetscPrintf(comm, "\n RHS assembly:\n"));
    PetscCall(PetscPrintf(comm, "   %-24s = %s\n",                "MPI tasks",   formatGroupedInt(params.numMPITasks)));
    PetscCall(PetscPrintf(comm, "   %-24s = %s\n", "Vector size", formatGroupedInt(M)));
    PetscCall(PetscPrintf(comm, "   %-24s = %s\n",                "Status",      "Started"));
  }

  /* Perform finite element assembly for RHS (one vector per source) */
  for (PetscInt i = 0; i < sources.numSources; i++) {
    /* Local variable declarations */
    PetscReal sourceRotationVector[NUM_DIMENSIONS] = {0.0};
    PetscReal sourceVector[NUM_DIMENSIONS] = {0.0};
    PetscReal Dx[NUM_DIMENSIONS] = {0.0};
    PetscReal Dy[NUM_DIMENSIONS] = {0.0};
    PetscReal Dz[NUM_DIMENSIONS] = {0.0};
    PetscInt cellID;

    /* Reset vector */
    PetscCall(VecZeroEntries(b));

    /* Define dipole for total electric field formulation */
    Dx[0] = sources.sourceArray[i].current * sources.sourceArray[i].length; /* x-directed dipole */
    Dy[1] = sources.sourceArray[i].current * sources.sourceArray[i].length; /* y-directed dipole */
    Dz[2] = sources.sourceArray[i].current * sources.sourceArray[i].length; /* z-directed dipole */

    /* Compute matrices for source rotation */
    PetscCall(computeVectorRotation(sources.sourceArray[i].azimuthAngle, sources.sourceArray[i].dipAngle, sourceRotationVector));

    /* Rotate source and setup electric field */
    for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
      sourceVector[i] = sourceRotationVector[0] * Dx[i] + sourceRotationVector[1] * Dy[i] + sourceRotationVector[2] * Dz[i];
    }

    /* Locate source within computational domain */
    cellID = -1;
    PetscCall(locatePoint(dm, sources.sourceArray[i].position, &cellID));

    /* Insert CSEM source */
    if (cellID >= 0) {

      /* Get vertices coordinates for cellID */
      PetscCall(extractCellCoordinates(dm, cellID, &cell));

      /* Transform xyz source position to XiEtaZeta coordinates (reference tetrahedral element) */
      PetscCall(tetrahedronXYZToReference(cell.coordinates, sources.sourceArray[i].position, XiEtaZeta));

      /* Compute basis functions for cellID (already oriented; no curls needed for the RHS) */
      PetscCall(evaluateNedelecBasis(&grid.fem, &cell, XiEtaZeta, Ni, NULL));

      /* Get closure indices for cellID */
      PetscCall(DMPlexGetClosureIndices(dm, section, section, cellID, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));

      /* Compute contribution for closure */
      for (PetscInt j = 0; j < grid.numDofInCell; j++) {
        closureRHS[j] = 0;
        for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
          closureRHS[j] += (Ni[k][j] * sourceVector[k]);
        }
      }

      /* Add closure to vector */
      PetscCall(VecSetValuesLocal(b, numDofIndices, dofIndices, closureRHS, INSERT_VALUES));

      /* Restore closure indices for cellID */
      PetscCall(DMPlexRestoreClosureIndices(dm, section, section, cellID, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
    }

    /* Perform global assembly for RHS */
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    /* Copy rhs into B matrix */
    PetscCall(MatDenseGetColumnVecWrite(*B, i, &bcol));
    PetscCall(VecCopy(b, bcol));
    PetscCall(MatDenseRestoreColumnVecWrite(*B, i, &bcol));
  }

  PetscCall(VecDestroy(&b));

  /* Apply constant factor */
  PetscCall(MatScale(*B, constFactor));

  /* Print message */
  if (!params.quiet)
    PetscCall(PetscPrintf(comm, "   %-24s = %s\n", "Status", "Finished"));

  /* Free memory */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscFree(Ni[i]));
  }
  PetscCall(PetscFree(Ni));
  PetscCall(PetscFree(XiEtaZeta));
  PetscCall(PetscFree(closureRHS));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Assembles the volumetric manufactured-source RHS for MMS verification.
 *
 * MMS counterpart of assembleCsemRHS: instead of a Dirac dipole located in a
 * single cell, it integrates the manufactured forcing f* (include/mms.h) over
 * every local cell,
 *
 *     b_j = sum_cells sum_q  w_q * detJ_cell * ( N_j(x_q) . f*(x_q) ),
 *
 * reusing the SAME quadrature rule and (weights, detJ) measure as
 * computeElementalMatrices, so the RHS is consistent with the operator
 * A = K - i omega mu Ms that assembleCsemKandM builds. Contributions from cells
 * sharing an edge/face DOF accumulate (ADD_VALUES); boundary DOFs are skipped
 * (negative closure indices + VEC_IGNORE_NEGATIVE_INDICES), which is exact since
 * n x E* = 0. The single manufactured RHS is written into a one-column dense B.
 *
 * @param[in]  params        Forward-modeling parameters (order, MPI tasks).
 * @param[in]  sources       Transmitter set; only sources.freq (-> omega) is used.
 * @param[in]  dm            DMPlex mesh and H(curl) discretization.
 * @param[in]  grid          Finite-element grid descriptor.
 * @param[in]  conductivity  Per-cell conductivity Vec (diagonal sigma in f*).
 * @param[out] B             One-column dense RHS matrix, created by this call.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 *
 * @note The caller is responsible for destroying the returned matrix `B`.
 */
PetscErrorCode assembleCsemMMSRHS(const fmParams params,
                                  const CsemSourceSet sources,
                                  const DM dm,
                                  const Grid grid,
                                  const Vec conductivity,
                                  const PetscBool useForcing,
                                  Mat* B) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  Cell cell;
  Quadrature3D quadrature_3d;
  PetscInt m, M, numDofIndices, *dofIndices;
  PetscReal **Ni;
  PetscScalar *closureRHS;
  PetscSection section;
  DM dmConductivity;
  Vec b, bcol;
  VecType vtype;
  ISLocalToGlobalMapping mapping;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  const PetscReal omega = sources.freq * 2.0 * PETSC_PI;

  /* One right-hand side vector (the single manufactured forcing). */
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMGetLocalToGlobalMapping(dm, &mapping));
  PetscCall(VecSetLocalToGlobalMapping(b, mapping));
  PetscCall(VecSetOption(b, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE));
  PetscCall(VecSetFromOptions(b));
  PetscCall(VecGetSize(b, &M));
  PetscCall(VecGetLocalSize(b, &m));

  /* One-column dense B (MMS has a single RHS). */
  PetscCall(VecGetType(b, &vtype));
  PetscCall(MatCreateDenseFromVecType(comm, vtype, m, PETSC_DECIDE, M, 1, m, NULL, B));

  /* DM section and conductivity DM. */
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(VecGetDM(conductivity, &dmConductivity));

  /* Same quadrature rule as the LHS element integrals. */
  PetscCall(computeNum3DQuadraturePoints(params.order, &quadrature_3d));
  PetscCall(PetscCalloc1(quadrature_3d.numPoints, &quadrature_3d.points));
  for (PetscInt i = 0; i < quadrature_3d.numPoints; i++) {
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &quadrature_3d.points[i]));
  }
  PetscCall(PetscCalloc1(quadrature_3d.numPoints, &quadrature_3d.weights));
  PetscCall(compute3DQuadraturePoints(&quadrature_3d));

  /* Basis-value and closure scratch. */
  PetscCall(PetscCalloc1(grid.numDofInCell, &closureRHS));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Ni));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(grid.numDofInCell, &Ni[i]));
  }

  /* Print statistics (suppressed when params.quiet). */
  if (!params.quiet) {
    PetscCall(PetscPrintf(comm, "\n MMS RHS assembly (%s):\n", useForcing ? "volumetric forcing f*" : "L2 moments of E*"));
    PetscCall(PetscPrintf(comm, "   %-24s = %s\n", "MPI tasks",   formatGroupedInt(params.numMPITasks)));
    PetscCall(PetscPrintf(comm, "   %-24s = %s\n", "Vector size", formatGroupedInt(M)));
    PetscCall(PetscPrintf(comm, "   %-24s = %s\n", "Status",      "Started"));
  }

  PetscCall(VecZeroEntries(b));

  /* Volumetric integral of f* against the Nedelec basis, cell by cell. */
  for (PetscInt c = grid.cellStart; c < grid.cellEnd; ++c) {

    /* Geometry + conductivity for cell c; computeCellJacobian fills
     * cell.jacobian (rows v1-v0, v2-v0, v3-v0) and cell.detJacobian, which
     * match femComputeJacobian used by evaluateNedelecBasis / the LHS. */
    PetscCall(extractCellCoordinates(dm, c, &cell));
    PetscCall(extractCellConductivity(dmConductivity, conductivity, c, &cell));
    PetscCall(computeCellJacobian(&cell));
    const PetscReal det = cell.detJacobian;
    const PetscReal sigma[NUM_DIMENSIONS] = {cell.conductivity[0], cell.conductivity[1], cell.conductivity[2]};

    PetscCall(DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));

    for (PetscInt j = 0; j < grid.numDofInCell; j++) {
      closureRHS[j] = 0.0;
    }

    for (PetscInt q = 0; q < quadrature_3d.numPoints; q++) {
      const PetscReal *ref = quadrature_3d.points[q];

      /* Physical image of the reference quadrature point: x = v0 + F*ref,
       * with F = J^T, i.e. x_d = coords_d + sum_k jacobian[k][d]*ref[k]. */
      PetscReal xphys[NUM_DIMENSIONS];
      for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
        xphys[d] = cell.coordinates[d]
                 + cell.jacobian[0][d] * ref[0]
                 + cell.jacobian[1][d] * ref[1]
                 + cell.jacobian[2][d] * ref[2];
      }

      /* useForcing: integrate f* (the MMS solve RHS). Otherwise integrate the
       * exact field E* itself -> L2 moments for the E3 projection baseline. */
      PetscScalar F[NUM_DIMENSIONS];
      if (useForcing) {
        mmsForcingF(xphys, omega, sigma, F);
      } else {
        mmsExactE(xphys, F);
      }

      /* Physical, oriented basis values at this reference point (no curls). */
      PetscCall(evaluateNedelecBasis(&grid.fem, &cell, ref, Ni, NULL));

      const PetscReal wdet = quadrature_3d.weights[q] * det;
      for (PetscInt j = 0; j < grid.numDofInCell; j++) {
        PetscScalar dot = 0.0;
        for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
          dot += (PetscScalar)Ni[d][j] * F[d];
        }
        closureRHS[j] += wdet * dot;
      }
    }

    /* Accumulate this cell's contribution (shared DOFs sum across cells). */
    PetscCall(VecSetValuesLocal(b, numDofIndices, dofIndices, closureRHS, ADD_VALUES));
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
  }

  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  /* Copy the assembled RHS into the single column of B (no iωμ scaling:
   * f* already carries the frequency/permeability factor). */
  PetscCall(MatDenseGetColumnVecWrite(*B, 0, &bcol));
  PetscCall(VecCopy(b, bcol));
  PetscCall(MatDenseRestoreColumnVecWrite(*B, 0, &bcol));

  PetscCall(VecDestroy(&b));

  if (!params.quiet) {
    PetscCall(PetscPrintf(comm, "   %-24s = %s\n", "Status", "Finished"));
  }

  /* Free memory */
  PetscCall(PetscFree(quadrature_3d.weights));
  for (PetscInt i = 0; i < quadrature_3d.numPoints; i++) {
    PetscCall(PetscFree(quadrature_3d.points[i]));
  }
  PetscCall(PetscFree(quadrature_3d.points));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscFree(Ni[i]));
  }
  PetscCall(PetscFree(Ni));
  PetscCall(PetscFree(closureRHS));

  PetscFunctionReturn(PETSC_SUCCESS);
}

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
static PetscErrorCode checkGradientKernel(PetscReal *M, PetscReal *G, PetscInt m, PetscInt n, PetscInt w)
{
   PetscFunctionBeginUser;
   for (PetscInt i = 0; i < m; i++) {
     for (PetscInt j = 0; j < n; j++) {
       PetscReal v = 0;
       for (PetscInt k = 0; k < m; k++) {
         // M is m x m, G is m x n
         v += M[i*m + k] * G[k * n + j];
       }
       if (!PetscIsCloseAtTol(v, 0, 0, PETSC_SMALL)) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error element %d (%d %d)\n", (int)w, (int)i, (int)j));
     }
   }
   PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief CSEM LHS assembly.
 *
 * Single-pass element loop that assembles the frequency-INDEPENDENT
 * pieces of the CSEM operator and the discrete-gradient hint matrices:
 *
 *   K       - stiffness (curl–curl) matrix, ∫ (μ⁻¹ curl N_i)·curl N_j.
 *   Ms      - mass × σ matrix, ∫ (ε_r ⊙ N_i)·N_j where ε_r encodes σ.
 *   G       - high-order discrete gradient (buildDiscreteGradientMatrix)
 *             Consumed by PCBDDCSetDiscreteGradient.
 *
 * The caller forms A_f = K − iωμ·Ms per frequency via
 *     MatDuplicate(K, MAT_COPY_VALUES, &A);
 *     MatAXPY(A, -iωμ, Ms, SAME_NONZERO_PATTERN);
 * - the forward kernel does this once for the source frequency,
 * the inverse kernel does it inside a frequency loop.
 *
 * G may be passed as NULL to skip its construction (e.g. callers
 * that don't go through PCBDDC).
 *
 * @param[in]  params       Forward-modeling parameters (order, MPI tasks).
 * @param[in]  dm           DMPlex mesh and H(curl) discretization.
 * @param[in]  grid         Finite-element grid descriptor.
 * @param[in]  conductivity Per-cell conductivity Vec.
 * @param[in]  constFactor  Fused-mode factor iωμ; ignored when Ms != NULL.
 * @param[out] KorA         Stiffness K (K/Ms mode) or fused operator A.
 * @param[out] Ms           Mass-σ matrix in K/Ms mode; pass NULL for fused mode.
 * @param[out] G            High-order discrete gradient; pass NULL to skip.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode assembleCsemKandM(const fmParams params,
                                 const DM dm, 
                                 const Grid grid,
                                 const Vec conductivity,
                                 const PetscScalar constFactor,
                                 Mat *KorA, Mat *Ms,
                                 Mat *G) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  Cell cell;
  Quadrature3D quadrature_3d;
  PetscInt m, n, M, N, numDofIndices, numH1DofIndices;
  PetscInt *dofIndices, *H1dofIndices;
  PetscReal **Me, **Ke, **gradientMatrix;
  PetscScalar *closureK, *closureM, *closureGBDDC;
  PetscSection section, H1section;
  DM dmConductivity;
  Vec b, h1v;
  ISLocalToGlobalMapping mapping, H1mapping;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* fused mode: Ms == NULL -> caller wants the frequency-dependent
   *   A = K - constFactor·Ms produced directly (element-level
   *   fusion); no global Ms is built.
   * K/Ms mode: Ms != NULL -> produce K and Ms separately; caller
   *   forms A per frequency via MatDuplicate + MatAXPY. */
  const PetscBool fused = (Ms == NULL) ? PETSC_TRUE : PETSC_FALSE;

  /* Create *KorA via DMCreateMatrix. In K/Ms mode, build Ms by MatDuplicate so it shares K's parallel layout */
  PetscCall(DMSetAdjacency(dm, 0, PETSC_FALSE, PETSC_TRUE));
  PetscCall(DMSetMatrixPreallocateOnly(dm, PETSC_TRUE));
  PetscCall(DMCreateMatrix(dm, KorA));
  PetscCall(MatSetFromOptions(*KorA));
  PetscCall(MatSetOption(*KorA, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  if (!fused) {
    PetscCall(MatDuplicate(*KorA, MAT_DO_NOT_COPY_VALUES, Ms));
    PetscCall(MatSetOption(*Ms, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  }

  /* Create a Vec on the H(curl) DM just to query its sizes (M, m) for the G matrix layout below.  */
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMGetLocalToGlobalMapping(dm, &mapping));

  /* Row/column layout for G: rows index V_h (Nédélec, mapping); columns index the P_order H1 space (nodal+bubble, H1mapping) */
  PetscCall(DMCreateGlobalVector(grid.H1dm, &h1v));
  PetscCall(VecGetSize(b, &M));
  PetscCall(VecGetLocalSize(b, &m));
  PetscCall(VecGetSize(h1v, &N));
  PetscCall(VecGetLocalSize(h1v, &n));
  PetscCall(DMGetLocalToGlobalMapping(grid.H1dm, &H1mapping));
  PetscCall(VecDestroy(&h1v));
  PetscCall(VecDestroy(&b));

  /* Create gradient matrix */
  if (G) {    
    PetscCall(MatCreate(comm, G));
    PetscCall(MatSetSizes(*G, m, n, M, N));
    PetscCall(MatSetType(*G, MATAIJ));
    PetscCall(MatSetLocalToGlobalMapping(*G, mapping, H1mapping));
  }

  /* DM sections: H(curl) on dm and P_order H1 on grid.H1dm. */
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetLocalSection(grid.H1dm, &H1section));

  /* Get the local values of the conductivity components */
  PetscCall(VecGetDM(conductivity, &dmConductivity));

  /* Compute 3D quadrature points (element mass/stiffness; the discrete gradient builds its own reference quadrature internally) */
  PetscCall(computeNum3DQuadraturePoints(params.order, &quadrature_3d));

  PetscCall(PetscCalloc1(quadrature_3d.numPoints, &quadrature_3d.points));
  for (PetscInt i = 0; i < quadrature_3d.numPoints; i++) {
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &quadrature_3d.points[i]));
  }
  PetscCall(PetscCalloc1(quadrature_3d.numPoints, &quadrature_3d.weights));
  PetscCall(compute3DQuadraturePoints(&quadrature_3d));

  /* Allocate memory. closureK is a square scratch buffer sized numDofInCell × numDofInCell; in fused mode it carries the fused
   * A_e = K_e - constFactor·M_e block (still per-cell, no extra memory). closureM is only allocated in K/Ms mode. Canonical-G
   * scratch is only allocated when G is requested. */
  PetscCall(PetscMalloc1(grid.numDofInCell * grid.numDofInCell, &closureK));
  PetscCall(PetscCalloc1(grid.numDofInCell, &Me));
  PetscCall(PetscCalloc1(grid.numDofInCell, &Ke));
  PetscCall(PetscCalloc1(grid.numDofInCell * grid.numDofInCell, &Me[0]));
  PetscCall(PetscCalloc1(grid.numDofInCell * grid.numDofInCell, &Ke[0]));
  for (PetscInt i = 1; i < grid.numDofInCell; i++) {
    Me[i] = Me[i - 1] + grid.numDofInCell;
    Ke[i] = Ke[i - 1] + grid.numDofInCell;
  }

  closureM = NULL;
  if (!fused) {
    PetscCall(PetscMalloc1(grid.numDofInCell * grid.numDofInCell, &closureM));
  }
  /* Discrete-gradient scratch.
   *
   * gradientMatrix is the OUTPUT of buildDiscreteGradientMatrix, sized numDofInCell × numH1DofInCell (all P_order H1 columns, in DMPlex
   * closure order). closureGBDDC is the row-major INSERTION buffer of the same shape passed to MatSetValuesLocal. */
  gradientMatrix = NULL;
  closureGBDDC       = NULL;
  if (G) {
    PetscCall(PetscCalloc1(grid.numDofInCell, &gradientMatrix));
    PetscCall(PetscCalloc1(grid.numDofInCell * grid.numH1DofInCell, &gradientMatrix[0]));
    for (PetscInt i = 1; i < grid.numDofInCell; i++) {
      gradientMatrix[i] = gradientMatrix[i - 1] + grid.numH1DofInCell;
    }
    PetscCall(PetscCalloc1(grid.numDofInCell * grid.numH1DofInCell, &closureGBDDC));
  }

  /* Print linear system statistics (suppressed when params.quiet) */
  if (!params.quiet) {
    PetscCall(PetscPrintf(comm, "\n LHS assembly (K and Ms):\n"));
    PetscCall(PetscPrintf(comm, "   %-24s = %s\n",                                      "MPI tasks",   formatGroupedInt(params.numMPITasks)));
    PetscCall(PetscPrintf(comm, "   %-24s = %s x %s\n",   "Matrix size", formatGroupedInt(M), formatGroupedInt(M)));
    PetscCall(PetscPrintf(comm, "   %-24s = %s\n",                                      "Status",      "Started"));
  }

  /* Perform finite element assembly for LHS */
  for (PetscInt i = grid.cellStart; i < grid.cellEnd; ++i) {

    /* Geometry, conductivity, closure and orientation for cell i */
    PetscCall(prepareCellForAssembly(dm, dmConductivity, conductivity, i, &cell));

    /* Compute mass and stifness matrices for cell i */
    PetscCall(computeElementalMatrices(&grid.fem, &cell, &quadrature_3d, Me, Ke));

    /* Get closure indices for cell i */
    PetscCall(DMPlexGetClosureIndices(dm, section, section, i, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
    PetscCall(DMPlexGetClosureIndices(grid.H1dm, H1section, H1section, i, PETSC_TRUE, &numH1DofIndices, &H1dofIndices, NULL, NULL));

    /* fused: closureK[jk] = K_e[jk] − constFactor·M_e[jk], single MatSetValuesLocal into A.
     * K/Ms : closureK[jk] = K_e[jk]; closureM[jk] = M_e[jk], two MatSetValuesLocal into K and Ms.
     * The (j,k) loop writes every entry of its scratch buffer, so no PetscArrayzero is needed. */
    if (fused) {
      for (PetscInt j = 0; j < grid.numDofInCell; j++) {
        for (PetscInt k = 0; k < grid.numDofInCell; k++) {
          closureK[j * grid.numDofInCell + k] = (PetscScalar)Ke[j][k] - constFactor * (PetscScalar)Me[j][k];
        }
      }
      PetscCall(MatSetValuesLocal(*KorA, numDofIndices, dofIndices, numDofIndices, dofIndices, closureK, ADD_VALUES));
    } else {
      for (PetscInt j = 0; j < grid.numDofInCell; j++) {
        for (PetscInt k = 0; k < grid.numDofInCell; k++) {
          closureK[j * grid.numDofInCell + k] = Ke[j][k];
          closureM[j * grid.numDofInCell + k] = Me[j][k];
        }
      }
      PetscCall(MatSetValuesLocal(*KorA, numDofIndices, dofIndices, numDofIndices, dofIndices, closureK, ADD_VALUES));
      PetscCall(MatSetValuesLocal(*Ms,   numDofIndices, dofIndices, numDofIndices, dofIndices, closureM, ADD_VALUES));
    }

    /* Compute gradient matrix */
    if (G) {
      PetscCall(buildDiscreteGradientMatrix(&grid.fem, &cell, gradientMatrix));
      
      /* Verify the per-cell discrete gradient lies in the kernel of the stiffness (K_e G_e = 0). */
      PetscCall(checkGradientKernel(Ke[0], gradientMatrix[0], grid.numDofInCell, grid.numH1DofInCell, i));
      
      /* Fill closure gradient matrix data */
      for (PetscInt j = 0; j < grid.numDofInCell; j++) {
        for (PetscInt k = 0; k < grid.numH1DofInCell; k++) {
          closureGBDDC[j * grid.numH1DofInCell + k] = gradientMatrix[j][k];
        }
      }
      PetscCall(MatSetValuesLocal(*G, numDofIndices, dofIndices, numH1DofIndices, H1dofIndices, closureGBDDC, INSERT_VALUES));
    }

    /* Restore closure indices for cell i */
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, i, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
    PetscCall(DMPlexRestoreClosureIndices(grid.H1dm, H1section, H1section, i, PETSC_TRUE, &numH1DofIndices, &H1dofIndices, NULL, NULL));
  }

  /* Global assembly for the produced matrices */
  PetscCall(MatAssemblyBegin(*KorA, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*KorA,   MAT_FINAL_ASSEMBLY));
  if (!fused) {
    PetscCall(MatAssemblyBegin(*Ms, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*Ms,   MAT_FINAL_ASSEMBLY));
  }

  if (G) {
    PetscCall(MatAssemblyBegin(*G, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*G,   MAT_FINAL_ASSEMBLY));
    /* The discrete gradient matrix is used to compute mesh connectivity information within the solver. Just use nonzero dofs */
   PetscCall(MatFilter(*G, 0, PETSC_TRUE, PETSC_FALSE));
  }

  /* End of assembly */
  if (!params.quiet)
    PetscCall(PetscPrintf(comm, "   %-24s = %s\n", "Status", "Finished"));

  /* Free memory */
  PetscCall(PetscFree(quadrature_3d.weights));
  for (PetscInt i = 0; i < quadrature_3d.numPoints; i++) {
    PetscCall(PetscFree(quadrature_3d.points[i]));
  }
  PetscCall(PetscFree(quadrature_3d.points));

  PetscCall(PetscFree(Me[0]));
  PetscCall(PetscFree(Ke[0]));
  PetscCall(PetscFree(Me));
  PetscCall(PetscFree(Ke));
  PetscCall(PetscFree(closureK));
  
  if (closureM) { 
    PetscCall(PetscFree(closureM));
  }
  
  if (gradientMatrix) {
    PetscCall(PetscFree(gradientMatrix[0]));
    PetscCall(PetscFree(gradientMatrix));
  }
  if (closureGBDDC) {
    PetscCall(PetscFree(closureGBDDC));
  }
  
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Refills an existing Ms matrix for the current conductivity field.
 *
 * Inverse-kernel companion to assembleCsemKandM: walks the local cells,
 * computes only the mass-matrix entries for the current sigma, and writes them
 * into the supplied Ms matrix. K and G are NOT touched - those are
 * sigma-independent, built once at setup, and reused for every L-BFGS iteration.
 *
 * The per-cell setup is shared with assembleCsemKandM via
 * prepareCellForAssembly (single source of truth for cell geometry,
 * conductivity slice, closure, and orientation). Ke is computed by
 * computeElementalMatrices (it shares basis evaluations with Me) but is
 * discarded - the small extra work is offset by not having to duplicate the
 * basis-evaluation code.
 *
 * @param[in]     params         Forward-modeling parameters (order).
 * @param[in]     dm             DMPlex mesh and H(curl) discretization.
 * @param[in]     grid           Finite-element grid descriptor.
 * @param[in]     conductivity   Current per-cell conductivity Vec.
 * @param[in]     quadrature_3d  Caller-owned 3D quadrature workspace.
 * @param[in,out] Me             Scratch elemental mass buffer (numDofInCell²).
 * @param[in,out] Ke             Scratch elemental stiffness buffer (unused output).
 * @param[in,out] Ms             Pre-allocated matrix refilled with mass-σ entries.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode assembleCsemMsRefill(const fmParams params,
                                    const DM dm, const Grid grid,
                                    const Vec conductivity,
                                    const Quadrature3D *quadrature_3d,
                                    PetscReal **Me, PetscReal **Ke,
                                    Mat Ms)
{
  PetscFunctionBeginUser;
  
  /* order lives in grid->fem.ops via the quadrature */
  (void)params;

  /* Zero stale values from the previous L-BFGS iteration; sparsity is preserved (no allocation churn).  
  * MatSetValuesLocal with ADD_VALUES below would otherwise accumulate on top of the previous iter. */
  PetscCall(MatZeroEntries(Ms));

  PetscSection section;
  DM dmConductivity;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(VecGetDM(conductivity, &dmConductivity));

  PetscScalar *closureM;
  PetscCall(PetscMalloc1(grid.numDofInCell * grid.numDofInCell, &closureM));

  for (PetscInt i = grid.cellStart; i < grid.cellEnd; ++i) {
    Cell cell;

    /* Shared per-cell setup: same call as in assembleCsemKandM. */
    PetscCall(prepareCellForAssembly(dm, dmConductivity, conductivity, i, &cell));

    /* Compute mass and stifness matrices; only Me is consumed here. */
    PetscCall(computeElementalMatrices(&grid.fem, &cell, quadrature_3d, Me, Ke));

    PetscInt numDofIndices, *dofIndices;
    PetscCall(DMPlexGetClosureIndices(dm, section, section, i, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));

    for (PetscInt j = 0; j < grid.numDofInCell; j++) {
      for (PetscInt k = 0; k < grid.numDofInCell; k++) {
        closureM[j * grid.numDofInCell + k] = Me[j][k];
      }
    }

    PetscCall(MatSetValuesLocal(Ms, numDofIndices, dofIndices, numDofIndices, dofIndices, closureM, ADD_VALUES));

    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, i, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
  }

  PetscCall(PetscFree(closureM));

  PetscCall(MatAssemblyBegin(Ms, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Ms,   MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}