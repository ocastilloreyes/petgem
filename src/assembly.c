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
#include "assembly.h"
#include "constants.h"
#include "grid.h"
#include "hvfem.h"
#include "inputs.h"
#include "transmitter.h"


/**
 * @brief Shared per-cell setup used by every LHS-assembly path.
 *
 * Reads geometry, builds the Jacobian, pulls the per-cell conductivity slice
 * out of the conductivity Vec, walks the cell's transitive closure for the
 * vertex ordering, and computes the orientation table. After this returns,
 * `cell` is ready to be handed to computeElementalMatrices / the basis
 * dispatch.
 *
 * @param[in]  dm              DMPlex mesh.
 * @param[in]  dmConductivity  DM carrying the conductivity field.
 * @param[in]  conductivity    Per-cell conductivity Vec.
 * @param[in]  cellID          Cell index.
 * @param[out] cell            Cell descriptor filled with geometry, jacobian,
 *                             conductivity, closure, and orientation.
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
  PetscCall(computeCellJacobian(cell));
  PetscCall(extractCellConductivity(dmConductivity, conductivity, cellID, cell));
  PetscCall(extractCellClousure(dm, cellID, cell));
  PetscCall(computeCellOrientation(cell));
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
 * @param[in]  params   Forward-modeling parameters (nord, MPI tasks).
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
  PetscInt dofSigns[grid.numDofInCell];
  PetscReal **Ni, **NiCurl, **coeffs, **Dx_Ni, **Dy_Ni, **Dz_Ni, *XiEtaZeta;
  PetscScalar *closureRHS;
  PetscSection section;
  Vec b, bcol;
  VecType vtype;
  ISLocalToGlobalMapping mapping;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* Create vector to store one right-hand side */
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMGetLocalToGlobalMapping(dm, &mapping));
  {
    ISLocalToGlobalMapping mappingBS1;
    const PetscInt        *l2gIndices;
    PetscInt               l2gSize;
    PetscCall(ISLocalToGlobalMappingGetSize(mapping, &l2gSize));
    PetscCall(ISLocalToGlobalMappingGetIndices(mapping, &l2gIndices));
    PetscCall(ISLocalToGlobalMappingCreate(comm, 1, l2gSize, l2gIndices,
                                            PETSC_COPY_VALUES, &mappingBS1));
    PetscCall(ISLocalToGlobalMappingRestoreIndices(mapping, &l2gIndices));
    PetscCall(VecSetLocalToGlobalMapping(b, mappingBS1));
    PetscCall(ISLocalToGlobalMappingDestroy(&mappingBS1));
  }
  PetscCall(VecSetOption(b, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE));
  PetscCall(VecSetFromOptions(b));

  /* Create discrete gradient matrix */
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
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &NiCurl));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dx_Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dy_Ni));
  PetscCall(PetscCalloc1(NUM_DIMENSIONS, &Dz_Ni));
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
    PetscCall(PetscCalloc1(grid.numDofInCell, &Ni[i]));
    PetscCall(PetscCalloc1(grid.numDofInCell, &NiCurl[i]));
    PetscCall(PetscCalloc1(grid.numDofInCell, &Dx_Ni[i]));
    PetscCall(PetscCalloc1(grid.numDofInCell, &Dy_Ni[i]));
    PetscCall(PetscCalloc1(grid.numDofInCell, &Dz_Ni[i]));
  }

  PetscCall(PetscCalloc1(grid.numDofInCell, &coeffs));
  for (PetscInt i = 0; i < grid.numDofInCell; i++) {
    PetscCall(PetscCalloc1(grid.numDofInCell, &coeffs[i]));
  }

  /* Print linear system statistics (suppressed when params.quiet) */
  if (!params.quiet) {
    PetscCall(PetscPrintf(comm, "\n RHS assembly:\n"));
    PetscCall(PetscPrintf(comm, "   %-24s = %d\n",                "MPI tasks",   params.numMPITasks));
    PetscCall(PetscPrintf(comm, "   %-24s = %" PetscInt_FMT "\n", "Vector size", M));
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

      /* Compute jacobian, inverse jacobian and jacobian determinand for cellID */
      PetscCall(computeCellJacobian(&cell));

      /* Get transitive clousure for cellID */
      PetscCall(extractCellClousure(dm, cellID, &cell));

      /* Compute orientation for cellID */
      PetscCall(computeCellOrientation(&cell));

      /* Transform xyz source position to XiEtaZeta coordinates (reference tetrahedral element) */
      PetscCall(tetrahedronXYZToReference(cell.coordinates, sources.sourceArray[i].position, XiEtaZeta));

      /* Compute basis functions for cellID (no curls needed for the RHS) */
      PetscCall(evaluateNedelecBasis(&grid.fem, &cell, XiEtaZeta, coeffs, Dx_Ni, Dy_Ni, Dz_Ni, Ni, NULL));

      /* Get closure indices for cellID */
      PetscCall(DMPlexGetClosureIndices(dm, section, section, cellID, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));

      /* Per-DOF sign convention for elemental matrices */
      PetscCall(buildDofSigns(&cell, &grid.fem, dofSigns));

      /* Compute contribution for closure */
      for (PetscInt j = 0; j < grid.numDofInCell; j++) {
        closureRHS[j] = 0;
        for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
          closureRHS[j] += (Ni[k][j] * sourceVector[k] * dofSigns[j]);
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
    PetscCall(PetscFree(NiCurl[i]));
    PetscCall(PetscFree(Dx_Ni[i]));
    PetscCall(PetscFree(Dy_Ni[i]));
    PetscCall(PetscFree(Dz_Ni[i]));
  }
  PetscCall(PetscFree(Ni));
  PetscCall(PetscFree(NiCurl));
  PetscCall(PetscFree(Dx_Ni));
  PetscCall(PetscFree(Dy_Ni));
  PetscCall(PetscFree(Dz_Ni));
  PetscCall(PetscFree(XiEtaZeta));
  PetscCall(PetscFree(closureRHS));

  for (PetscInt i = 0; i < grid.numDofInCell; i++) {
    PetscCall(PetscFree(coeffs[i]));
  }
  PetscCall(PetscFree(coeffs));

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
 *   G_BDDC  - topological lowest-Whitney gradient against the P_nord H¹
 *             DM (grid.H1dm_Pnord) with vertex incidence only. Inter-bubble
 *             columns never enter the sparsity pattern: G_BDDC is built with
 *             MAT_IGNORE_ZERO_ENTRIES and a 4-nnz/row preallocation, so no
 *             MatFilter pass is required. Consumed by
 *             PCBDDCSetDiscreteGradient at order = 1.
 *
 * The caller forms A_f = K − iωμ·Ms per frequency via
 *     MatDuplicate(K, MAT_COPY_VALUES, &A);
 *     MatAXPY(A, -iωμ, Ms, SAME_NONZERO_PATTERN);
 * - the forward kernel does this once for the source frequency,
 * the inverse kernel does it inside a frequency loop.
 *
 * G_BDDC may be passed as NULL to skip its construction (e.g. callers
 * that don't go through PCBDDC).
 */
PetscErrorCode assembleCsemKandM(const fmParams params, 
                                 const DM dm, 
                                 const Grid grid,
                                 const Vec conductivity,
                                 const PetscScalar constFactor,
                                 Mat *KorA, Mat *Ms,
                                 Mat *G_BDDC) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  Cell cell;
  Quadrature3D quadrature_3d;
  Quadrature1D quadrature_1d;
  PetscInt m, n, M, N, numDofIndices, numH1DofIndices;
  PetscInt *dofIndices, *H1dofIndices;
  PetscReal **Me, **Ke, **gradientMatrixBDDC;
  PetscScalar *closureK, *closureM, *closureGBDDC;
  PetscSection section, H1section;
  DM dmConductivity;
  Vec b, h1v;
  ISLocalToGlobalMapping mapping, H1mapping;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* fused mode: Ms == NULL -> caller wants the frequency-dependent
   *   A = K − constFactor·Ms produced directly (element-level
   *   fusion); no global Ms is built.
   * K/Ms mode: Ms != NULL -> produce K and Ms separately; caller
   *   forms A per frequency via MatDuplicate + MatAXPY. */
  const PetscBool fused = (Ms == NULL) ? PETSC_TRUE : PETSC_FALSE;

  /* Create *KorA via DMCreateMatrix. In K/Ms mode, build Ms by
   * MatDuplicate so it shares K's parallel layout AND the L2G mapping
   * attached to the DM.
   *
   * On the MATIS / PCBDDC path MatDuplicate does not always propagate
   * MAT_NEW_NONZERO_ALLOCATION_ERR cleanly, so MatSetValuesLocal on
   * the duplicate aborts with "New nonzero at (0,0) caused a malloc".
   * The option is disabled on both matrices to handle that case. */
  PetscCall(DMSetAdjacency(dm, 0, PETSC_FALSE, PETSC_TRUE));
  PetscCall(DMSetMatrixPreallocateOnly(dm, PETSC_TRUE));
  PetscCall(DMCreateMatrix(dm, KorA));
  PetscCall(MatSetFromOptions(*KorA));
  PetscCall(MatSetOption(*KorA, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  if (!fused) {
    PetscCall(MatDuplicate(*KorA, MAT_DO_NOT_COPY_VALUES, Ms));
    PetscCall(MatSetOption(*Ms, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  }

  /* Create a Vec on the H(curl) DM just to query its sizes (M, m) for
   * the G_BDDC matrix layout below.  The DM's L2G mapping is also
   * fetched here for the Mat-side calls (Mats do not enforce the strict
   * block-size match that trips VecSetLocalToGlobalMapping at nord ≥ 4
   * - see assembleCsemRHS for the equivalent workaround on the Vec
   * side). */
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMGetLocalToGlobalMapping(dm, &mapping));

  /* Row/column layout for G_BDDC: rows index V_h (Nédélec, mapping);
   * columns index the P_nord H1 space (nodal+bubble, H1mapping). */
  PetscCall(DMCreateGlobalVector(grid.H1dm_Pnord, &h1v));
  PetscCall(VecGetSize(b, &M));
  PetscCall(VecGetLocalSize(b, &m));
  PetscCall(VecGetSize(h1v, &N));
  PetscCall(VecGetLocalSize(h1v, &n));
  PetscCall(DMGetLocalToGlobalMapping(grid.H1dm_Pnord, &H1mapping));
  PetscCall(VecDestroy(&h1v));
  PetscCall(VecDestroy(&b));

  /* Topological gradient G_BDDC : Nédélec -> P_nord H1, lowest-Whitney
   * vertex incidence only (face/edge-bubble/volume H1 columns are all
   * zero). PCBDDC's coarse-space algorithm operates on this structural
   * hint at order = 1.
   *
   * Built against grid.H1dm_Pnord with vertex entries placed at the
   * LAST 4 closure positions per row (PETSc closure for P_nord H1 is
   * volume -> face -> edge -> vertex). Each row has at most 4 nonzeros
   * regardless of nord, so we preallocate d_nnz = o_nnz = 4 and turn
   * on MAT_IGNORE_ZERO_ENTRIES - that way the insertion's full
   * P_nord-wide closure buffer doesn't trigger storage for the
   * inter-bubble columns, and no MatFilter pass is needed. */
  if (G_BDDC) {
    PetscCall(MatCreate(comm, G_BDDC));
    PetscCall(MatSetSizes(*G_BDDC, m, n, M, N));
    PetscCall(MatSetType(*G_BDDC, MATAIJ));
    PetscCall(MatSetLocalToGlobalMapping(*G_BDDC, mapping, H1mapping));
    PetscCall(MatSeqAIJSetPreallocation(*G_BDDC, 4, NULL));
    PetscCall(MatMPIAIJSetPreallocation(*G_BDDC, 4, NULL, 4, NULL));
    PetscCall(MatSetOption(*G_BDDC, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
    PetscCall(MatSetOption(*G_BDDC, MAT_NEW_NONZERO_ALLOCATION_ERR,
                           PETSC_TRUE));
  }

  /* DM sections: H(curl) on dm and P_nord H1 on grid.H1dm_Pnord. */
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetLocalSection(grid.H1dm_Pnord, &H1section));

  /* Get the local values of the conductivity components */
  PetscCall(VecGetDM(conductivity, &dmConductivity));

  /* Compute quadrature points (1D and 3D cases) */
  PetscCall(computeNum3DQuadraturePoints(params.nord, &quadrature_3d));
  PetscCall(computeNum1DQuadraturePoints(params.nord, &quadrature_1d));

  PetscCall(PetscCalloc1(quadrature_3d.numPoints, &quadrature_3d.points));
  PetscCall(PetscCalloc1(quadrature_1d.numPoints, &quadrature_1d.points));
  for (PetscInt i = 0; i < quadrature_3d.numPoints; i++) {
    PetscCall(PetscCalloc1(NUM_DIMENSIONS, &quadrature_3d.points[i]));
  }
  PetscCall(PetscCalloc1(quadrature_3d.numPoints, &quadrature_3d.weights));
  PetscCall(PetscCalloc1(quadrature_1d.numPoints, &quadrature_1d.weights));
  PetscCall(compute3DQuadraturePoints(&quadrature_3d));
  PetscCall(compute1DQuadraturePoints(&quadrature_1d));

  /* Allocate memory. closureK is a square scratch buffer sized
   * numDofInCell × numDofInCell; in fused mode it carries the fused
   * A_e = K_e − constFactor·M_e block (still per-cell, no extra
   * memory). closureM is only allocated in K/Ms mode. Canonical-G
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
  /* Topological gradient scratch.
   *
   * gradientMatrixBDDC is the OUTPUT of the topological builder
   * (hierarchicalBuildGradientMatrixTopological), sized
   * numDofInCell × numH1DofInCell (= 4 vertex columns in PETGEM
   * cell-local order).
   *
   * closureGBDDC is the INSERTION buffer for G_BDDC. We pass only the
   * 4 vertex-tail column indices of the P_nord H1 closure to
   * MatSetValuesLocal, so the buffer is sized to match:
   *   numDofInCell × numH1DofInCell  (= 4 cols).
   * The per-cell SetValues walk is 4·numDofInCell instead of
   * numH1DofInCell_Pnord·numDofInCell. */
  gradientMatrixBDDC = NULL;
  closureGBDDC       = NULL;
  if (G_BDDC) {
    PetscCall(PetscCalloc1(grid.numDofInCell, &gradientMatrixBDDC));
    PetscCall(PetscCalloc1(grid.numDofInCell * grid.numH1DofInCell,
                           &gradientMatrixBDDC[0]));
    for (PetscInt i = 1; i < grid.numDofInCell; i++) {
      gradientMatrixBDDC[i] = gradientMatrixBDDC[i - 1] + grid.numH1DofInCell;
    }
    PetscCall(PetscCalloc1(grid.numDofInCell * grid.numH1DofInCell,
                           &closureGBDDC));
  }

  /* Print linear system statistics (suppressed when params.quiet) */
  if (!params.quiet) {
    PetscCall(PetscPrintf(comm, "\n LHS assembly (K and Ms):\n"));
    PetscCall(PetscPrintf(comm, "   %-24s = %d\n",                                      "MPI tasks",   params.numMPITasks));
    PetscCall(PetscPrintf(comm, "   %-24s = %" PetscInt_FMT " x %" PetscInt_FMT "\n",   "Matrix size", M, M));
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
    PetscCall(DMPlexGetClosureIndices(grid.H1dm_Pnord, H1section, H1section, i, PETSC_TRUE, &numH1DofIndices, &H1dofIndices, NULL, NULL));

    /* fused: closureK[jk] = K_e[jk] − constFactor·M_e[jk],
     *        single MatSetValuesLocal into A.
     * K/Ms : closureK[jk] = K_e[jk]; closureM[jk] = M_e[jk],
     *        two MatSetValuesLocal into K and Ms.
     * The (j,k) loop writes every entry of its scratch buffer, so no
     * PetscArrayzero is needed. */
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

    /* Build & insert the topological G_BDDC for PCBDDC.
     *
     * The topological builder fills gradientMatrixBDDC[*][0..3] with
     * the ±1 vertex incidence in PETGEM cell-local vertex order. In
     * the P_nord H1 closure (volume -> face -> edge -> vertex) those 4
     * vertex DOFs sit at the END of H1dofIndices - at positions
     *   [numH1DofIndices − 4 .. numH1DofIndices − 1].
     * We only pass that 4-column slice to MatSetValuesLocal (instead
     * of the full P_nord row width), so PETSc walks 4·numDofInCell
     * entries per cell instead of numH1DofInCell_Pnord·numDofInCell.
     * Combined with MAT_IGNORE_ZERO_ENTRIES, the higher-order H(curl)
     * rows (whose topological gradient is zero) contribute nothing to
     * the sparsity pattern. */
    if (G_BDDC) {
      PetscCall(grid.fem.ops->buildGradientMatrix(&grid.fem, &cell, &quadrature_1d, gradientMatrixBDDC));
      for (PetscInt j = 0; j < grid.numDofInCell; j++) {
        for (PetscInt k = 0; k < grid.numH1DofInCell; k++) {
          closureGBDDC[j * grid.numH1DofInCell + k] = gradientMatrixBDDC[j][k];
        }
      }
      const PetscInt vertex_offset = numH1DofIndices - grid.numH1DofInCell;
      PetscCall(MatSetValuesLocal(*G_BDDC, numDofIndices, dofIndices, grid.numH1DofInCell, H1dofIndices + vertex_offset,
                                  closureGBDDC, INSERT_VALUES));
    }

    /* Restore closure indices for cell i */
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, i, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
    PetscCall(DMPlexRestoreClosureIndices(grid.H1dm_Pnord, H1section, H1section, i, PETSC_TRUE, &numH1DofIndices, &H1dofIndices, NULL, NULL));
  }

  /* Global assembly for the produced matrices. */
  PetscCall(MatAssemblyBegin(*KorA, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*KorA,   MAT_FINAL_ASSEMBLY));
  if (!fused) {
    PetscCall(MatAssemblyBegin(*Ms, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*Ms,   MAT_FINAL_ASSEMBLY));
  }
  if (G_BDDC) {
    PetscCall(MatAssemblyBegin(*G_BDDC, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*G_BDDC,   MAT_FINAL_ASSEMBLY));
  }

  /* End of assembly */
  if (!params.quiet)
    PetscCall(PetscPrintf(comm, "   %-24s = %s\n", "Status", "Finished"));

  /* Setup matrix views for petgem. In fused mode `KorA` carries A
   * directly; the `-petgem_k_view` option views A. */
  PetscCall(MatViewFromOptions(*KorA, NULL,
                               fused ? "-petgem_a_view" : "-petgem_k_view"));
  if (!fused) PetscCall(MatViewFromOptions(*Ms, NULL, "-petgem_ms_view"));
  if (G_BDDC) PetscCall(MatViewFromOptions(*G_BDDC, NULL, "-petgem_grad_bddc_view"));

  /* Free memory */
  PetscCall(PetscFree(quadrature_3d.weights));
  PetscCall(PetscFree(quadrature_1d.weights));
  for (PetscInt i = 0; i < quadrature_3d.numPoints; i++) {
    PetscCall(PetscFree(quadrature_3d.points[i]));
  }
  PetscCall(PetscFree(quadrature_3d.points));
  PetscCall(PetscFree(quadrature_1d.points));

  PetscCall(PetscFree(Me[0]));
  PetscCall(PetscFree(Ke[0]));
  PetscCall(PetscFree(Me));
  PetscCall(PetscFree(Ke));
  PetscCall(PetscFree(closureK));
  if (closureM) PetscCall(PetscFree(closureM));
  if (gradientMatrixBDDC) {
    PetscCall(PetscFree(gradientMatrixBDDC[0]));
    PetscCall(PetscFree(gradientMatrixBDDC));
  }
  if (closureGBDDC) PetscCall(PetscFree(closureGBDDC));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Refills an existing Ms matrix for the current conductivity field.
 *
 * Inverse-kernel companion to assembleCsemKandM: walks the local cells,
 * computes only the mass-matrix entries for the current sigma, and writes them
 * into the supplied Ms matrix. K and G_BDDC are NOT touched - those are
 * sigma-independent, built once at setup, and reused for every L-BFGS iteration.
 *
 * The per-cell setup is shared with assembleCsemKandM via
 * prepareCellForAssembly (single source of truth for cell geometry,
 * conductivity slice, closure, and orientation). Ke is computed by
 * computeElementalMatrices (it shares basis evaluations with Me) but is
 * discarded - the small extra work is offset by not having to duplicate the
 * basis-evaluation code.
 *
 * @param[in]     params         Forward-modeling parameters (nord).
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
  (void)params; /* nord lives in grid->fem.ops via the quadrature */

  /* Zero stale values from the previous L-BFGS iteration; sparsity is
   * preserved (no allocation churn).  MatSetValuesLocal with ADD_VALUES
   * below would otherwise accumulate on top of the previous iter. */
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