/*
 * Filename: assembly.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2024-08-02
 *
 * Description:
 * This file contains functions for assembly linear system
 * (CSEM or MT) in a PETGEM simulation.
 *
 */

/* C libraries */

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
 * @brief Assembles the right-hand side (RHS) matrix for the CSEM system.
 *
 * This function constructs the RHS vectors for all sources in the
 * simulation. Each column of the output matrix `B` corresponds to a
 * different source. The function performs finite element assembly
 * using H(curl) Nédélec elements, taking into account the source
 * dipole orientation, rotation, and position within the tetrahedral mesh.
 *
 * @param[in] params A csemParams struct containing simulation parameters,
 *                   including the basis order (nord) and number of MPI tasks.
 * @param[in] sources A CsemSourceSet struct containing source positions,
 *                    currents, lengths, and dipole orientation angles.
 * @param[in] dm The DMPlex object representing the mesh topology and H(curl)
 *               discretization.
 * @param[in] grid A Grid struct containing mesh statistics and DOF information.
 * @param[out] B Pointer to a Mat object that will be created and populated
 *               with the RHS vectors for all sources.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful assembly, or a PETSc
 *         error code if an error occurs during vector/matrix creation,
 *         FEM basis evaluation, or point location.
 *
 * @details
 * The function performs the following steps:
 * 1. Creates a global vector `b` to store a single source's RHS.
 * 2. Allocates memory for Nédélec basis functions, their curls, and coefficients.
 * 3. Iterates over all sources in `sources->sourceArray`.
 *    - Computes the source vector based on current, length, dip, and azimuth.
 *    - Locates the source in the mesh using `locatePoint`.
 *    - Extracts cell coordinates, computes the Jacobian, and transforms
 *      the source position to reference coordinates (XiEtaZeta).
 *    - Computes Nédélec basis functions and, if applicable, their curls.
 *    - Computes the local contributions of the source to the RHS using
 *      `VecSetValuesLocal`.
 *    - Assembles the local vector into the global RHS vector.
 *    - Copies the assembled vector into the corresponding column of `B`.
 * 4. Applies the complex scaling factor for frequency and magnetic permeability.
 * 5. Frees all allocated memory and prints assembly progress messages.
 *
 * @note
 * - Supports multiple sources, each mapped to a column of the dense matrix `B`.
 *   is partially implemented and can be extended.
 * - The caller is responsible for managing the lifetime of the returned matrix `B`.
 */
PetscErrorCode assembleCsemRHS(const csemParams params, const CsemSourceSet sources, const DM dm, const Grid grid, Mat* B) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  Cell cell;
  PetscInt m, M, numDofIndices, *dofIndices;
  PetscReal **Ni, **NiCurl, **coeffs, **Dx_Ni, **Dy_Ni, **Dz_Ni, *XiEtaZeta;
  PetscReal omega;
  PetscScalar constFactor, *closureRHS;
  PetscSection section;
  Vec b;
  VecType vtype;
  ISLocalToGlobalMapping mapping;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* Compute constant */
  omega = sources.freq * 2.0 * PETSC_PI;
  constFactor = (0.0 + 1.0 * PETSC_i) * (omega * MU);

  /* Create vector to store one right-hand side */
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMGetLocalToGlobalMapping(dm, &mapping));
  PetscCall(VecSetLocalToGlobalMapping(b, mapping));
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

  /* Create the const views for arrays */
  const PetscReal** coeffs_const = (const PetscReal**)coeffs;

  /* Print linear system statistics */
  PetscCall(PetscPrintf(comm, "\n Assembly RHS:\n"));
  PetscCall(PetscPrintf(comm, "   Num of MPI tasks    = %d\n", params.numMPITasks));
  PetscCall(PetscPrintf(comm, "   Vector size         = %" PetscInt_FMT "\n", M));
  PetscCall(PetscPrintf(comm, "   Assembly process    = Initiated\n"));

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

      /* Get vertices coordinates for cell i */
      PetscCall(extractCellCoordinates(dm, cellID, &cell));

      /* Compute jacobian, inverse jacobian and jacobian
       * determinand for cell i */
      PetscCall(computeCellJacobian(&cell));

      /* Get transitive clousure for cell i */
      PetscCall(extractCellClousure(dm, cellID, &cell));

      /* Compute cell orientation */
      PetscCall(computeCellOrientation(&cell));

      /* Transform xyz source position to XiEtaZeta
       * coordinates (reference tetrahedral element) */
      PetscCall(tetrahedronXYZToReference(cell.coordinates, sources.sourceArray[i].position, XiEtaZeta));

      /* Compute basis functions for cellID */
      switch (params.nord) {
      case 1:
        /* Compute nedelec coefficients and its derivatives */
        PetscCall(computeNedelecOrder1Coefficients(params.nord, coeffs, Dx_Ni, Dy_Ni, Dz_Ni));

        /* Compute basis functions */
        PetscCall(computeNedelecOrder1BasisFunctions(params.nord, XiEtaZeta, (const PetscReal(*)[NUM_DIMENSIONS])cell.jacobian,
                                                     coeffs_const, Ni));
        break;
      case 2:
        break;
      default:
        break;
      }

      /* Get closure indices for cellID */
      PetscCall(DMPlexGetClosureIndices(dm, section, section, cellID, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));

      /* Compute contribution for closure */
      for (PetscInt j = 0; j < grid.numDofInCell; j++) {
        closureRHS[j] = 0;
        for (PetscInt k = 0; k < NUM_DIMENSIONS; k++) {
          closureRHS[j] += (Ni[k][j] * sourceVector[k] * cell.orientation[j + 4]);
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
    Vec bcol;
    PetscCall(MatDenseGetColumnVecWrite(*B, i, &bcol));
    PetscCall(VecCopy(b, bcol));
    PetscCall(MatDenseRestoreColumnVecWrite(*B, i, &bcol));
  }

  PetscCall(VecDestroy(&b));

  /* Apply constant factor */
  PetscCall(MatScale(*B, constFactor));

  /* Print message */
  PetscCall(PetscPrintf(comm, "   Assembly process    = Finished\n"));

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
 * @brief Assembles the left-hand side (LHS) matrix and discrete gradient for the CSEM system.
 *
 * Constructs the global system matrix `A` and the discrete gradient matrix `G`
 * for H(curl) finite element discretization of Maxwell's equations. Each cell's
 * elemental mass and stiffness matrices are computed and assembled into the
 * global system. The discrete gradient matrix relates H(curl) edge DOFs to H1 nodal DOFs,
 * required for PCBDDC preconditioning in PETSc.
 *
 * @param[in] params A csemParams struct containing simulation parameters,
 *                   including basis order (nord) and number of MPI tasks.
 * @param[in] sources A CsemSourceSet struct containing source frequency information
 *                    (used to compute the frequency-dependent LHS factor).
 * @param[in] dm The DMPlex object representing the mesh topology and H(curl)
 *               discretization.
 * @param[in] grid A Grid struct containing mesh statistics, number of DOFs per cell,
 *                 and auxiliary H1 DM information for the discrete gradient.
 * @param[in] resistivity A Vec containing the resistivity values at mesh locations.
 * @param[out] A Pointer to a Mat object to be created and populated with the global LHS.
 * @param[out] G Pointer to a Mat object to be created and populated with the discrete gradient.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful assembly, or a PETSc error code
 *         on memory allocation, matrix insertion, or FEM evaluation errors.
 *
 * @details
 * The function performs the following steps:
 * 1. Computes the complex frequency-dependent factor `constFactor = i * omega * mu`.
 * 2. Creates the global system matrix `A` and prepares the local-to-global mappings.
 * 3. Creates the discrete gradient matrix `G` connecting H(curl) DOFs to H1 DOFs.
 * 4. Computes quadrature points and weights for 1D and 3D integration.
 * 5. Allocates memory for elemental mass (Me), stiffness (Ke), and gradient matrices.
 * 6. Iterates over all active cells in the mesh:
 *    - Extracts cell vertex coordinates and computes the Jacobian and orientation.
 *    - Extracts cell resistivity.
 *    - Computes elemental mass and stiffness matrices via numerical quadrature.
 *    - Computes the elemental discrete gradient matrix.
 *    - Assembles the local elemental matrices into the global LHS matrix `A`.
 *    - Inserts the local gradient matrix into the discrete gradient matrix `G` (currently only for first-order elements).
 * 7. Performs global assembly for `A` and `G`.
 * 8. Optionally filters the discrete gradient matrix to include only nonzero DOFs.
 * 9. Prints assembly progress and optionally views matrices via PETSc options.
 * 10. Frees all allocated memory.
 *
 * @note
 * - The discrete gradient matrix `G` is primarily used for PCBDDC preconditioning.
 * - The caller is responsible for destroying matrices `A` and `G` after use.
 */
PetscErrorCode assembleCsemLHS(const csemParams params, const CsemSourceSet sources, const DM dm, const Grid grid, const Vec resistivity,
                               Mat* A, Mat* G) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  Cell cell;
  Quadrature3D quadrature_3d;
  Quadrature1D quadrature_1d;
  PetscInt m, n, M, N, numDofIndices, numH1DofIndices, *dofIndices, *H1dofIndices;
  PetscReal omega, **Me, **Ke, **gradientMatrix;
  PetscScalar constFactor, *closureLHS;
  PetscSection section, H1section;
  DM dmResistivity;
  Vec b, h1v;
  ISLocalToGlobalMapping mapping, H1mapping;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);

  /* Compute constant */
  omega = sources.freq * 2.0 * PETSC_PI;
  constFactor = (0.0 + 1.0 * PETSC_i) * (omega * MU);

  /* Create linear system matrix */
  PetscCall(DMSetAdjacency(dm, 0, PETSC_FALSE, PETSC_TRUE));
  PetscCall(DMSetMatrixPreallocateOnly(dm, PETSC_TRUE));
  PetscCall(DMCreateMatrix(dm, A));
  PetscCall(MatSetFromOptions(*A));

  /* Create vector to store one right-hand side */
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMGetLocalToGlobalMapping(dm, &mapping));
  PetscCall(VecSetLocalToGlobalMapping(b, mapping));
  PetscCall(VecSetOption(b, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE));
  PetscCall(VecSetFromOptions(b));

  /* Create discrete gradient matrix */
  PetscCall(DMCreateGlobalVector(grid.H1dm, &h1v));
  PetscCall(VecGetSize(b, &M));
  PetscCall(VecGetLocalSize(b, &m));
  PetscCall(VecGetSize(h1v, &N));
  PetscCall(VecGetLocalSize(h1v, &n));
  PetscCall(DMGetLocalToGlobalMapping(grid.H1dm, &H1mapping));
  PetscCall(MatCreate(comm, G));
  PetscCall(MatSetSizes(*G, m, n, M, N));
  PetscCall(MatSetType(*G, MATAIJ));
  PetscCall(MatSetLocalToGlobalMapping(*G, mapping, H1mapping));
  PetscCall(VecDestroy(&h1v));
  PetscCall(VecDestroy(&b));

  /* Get DM section */
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetLocalSection(grid.H1dm, &H1section));

  /* Get the local values of the resistivity components */
  PetscCall(VecGetDM(resistivity, &dmResistivity));

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

  /* Allocate memory for LHS */
  PetscCall(PetscCalloc1(grid.numDofInCell * PetscMax(grid.numDofInCell, grid.numH1DofInCell), &closureLHS));
  PetscCall(PetscCalloc1(grid.numDofInCell, &Me));
  PetscCall(PetscCalloc1(grid.numDofInCell, &Ke));
  PetscCall(PetscCalloc1(grid.numDofInCell, &gradientMatrix));
  PetscCall(PetscCalloc1(grid.numDofInCell * grid.numDofInCell, &Me[0]));
  PetscCall(PetscCalloc1(grid.numDofInCell * grid.numDofInCell, &Ke[0]));
  PetscCall(PetscCalloc1(grid.numDofInCell * grid.numH1DofInCell, &gradientMatrix[0]));
  for (PetscInt i = 1; i < grid.numDofInCell; i++) {
    Me[i] = Me[i - 1] + grid.numDofInCell;
    Ke[i] = Ke[i - 1] + grid.numDofInCell;
    gradientMatrix[i] = gradientMatrix[i - 1] + grid.numH1DofInCell;
  }

  /* Print linear system statistics */
  PetscCall(PetscPrintf(comm, "\n Assembly LHS:\n"));
  PetscCall(PetscPrintf(comm, "   Num of MPI tasks    = %d\n", params.numMPITasks));
  PetscCall(PetscPrintf(comm, "   Matrix size         = %" PetscInt_FMT " x %" PetscInt_FMT "\n", M, M));
  PetscCall(PetscPrintf(comm, "   Assembly process    = Initiated\n"));

  /* Perform finite element assembly for LHS */
  for (PetscInt i = grid.cellStart; i < grid.cellEnd; ++i) {

    /* Get vertices coordinates for cell i */
    PetscCall(extractCellCoordinates(dm, i, &cell));

    /* Compute jacobian, inverse jacobian and jacobian
     * determinand for cell i */
    PetscCall(computeCellJacobian(&cell));

    /* Get resistivity for cell i */
    PetscCall(extractCellResistivity(dmResistivity, resistivity, i, &cell));

    /* Get transitive clousure for cell i */
    PetscCall(extractCellClousure(dm, i, &cell));

    /* Compute cell orientation */
    PetscCall(computeCellOrientation(&cell));

    /* Compute mass and stifness matrices for cell i */
    PetscCall(computeElementalMatrices(params.nord, grid.numDofInCell, &cell, &quadrature_3d, Me, Ke));

    /* Compute gradient matrix */
    PetscCall(computeElementalGradientMatrix(params.nord, grid.numDofInCell, grid.numH1DofInCell, &cell, &quadrature_1d, gradientMatrix));

    /* Check that gradientMatrix is the kernel of Ke */
    PetscCall(checkDiscreteGradientKernel(Ke[0], gradientMatrix[0], grid.numDofInCell, grid.numH1DofInCell, i));

    /* Get closure indices for cell i */
    PetscCall(DMPlexGetClosureIndices(dm, section, section, i, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
    PetscCall(DMPlexGetClosureIndices(grid.H1dm, H1section, H1section, i, PETSC_TRUE, &numH1DofIndices, &H1dofIndices, NULL, NULL));

    /* Compute elemental matrix for cell i */
    PetscCall(PetscArrayzero(closureLHS, grid.numDofInCell * grid.numDofInCell));
    for (PetscInt j = 0; j < grid.numDofInCell; j++) {
      for (PetscInt k = 0; k < grid.numDofInCell; k++) {
        closureLHS[j * grid.numDofInCell + k] = Ke[j][k] - (constFactor * Me[j][k]);
      }
    }

    /* Add closure to matrix */
    PetscCall(MatSetValuesLocal(*A, numDofIndices, dofIndices, numDofIndices, dofIndices, closureLHS, ADD_VALUES));

    /* Insert closure to discrete gradient matrix */
    /* XXX TODO higher order*/
    if (params.nord == 1) {
      PetscCall(PetscArrayzero(closureLHS, grid.numDofInCell * grid.numH1DofInCell));
      for (PetscInt j = 0; j < grid.numDofInCell; j++) {
        for (PetscInt k = 0; k < grid.numH1DofInCell; k++) {
          closureLHS[j * grid.numH1DofInCell + k] = gradientMatrix[j][k];
        }
      }

      PetscCall(MatSetValuesLocal(*G, numDofIndices, dofIndices, numH1DofIndices, H1dofIndices, closureLHS, INSERT_VALUES));
    }

    /* Restore closure indices for cell i */
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, i, PETSC_TRUE, &numDofIndices, &dofIndices, NULL, NULL));
    PetscCall(DMPlexRestoreClosureIndices(grid.H1dm, H1section, H1section, i, PETSC_TRUE, &numH1DofIndices, &H1dofIndices, NULL, NULL));
  }

  /* Perform global assembly for LHS */
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(*G, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*G, MAT_FINAL_ASSEMBLY));

  /* The discrete gradient matrix is used to compute mesh
     connectivity information within the solver. Just use
     nonzero dofs */
  /* XXX TODO higher order*/
  if (params.nord == 1) {
    PetscCall(MatFilter(*G, 0, PETSC_TRUE, PETSC_FALSE));
  }

  /* End of assembly */
  PetscCall(PetscPrintf(comm, "   Assembly process    = Finished\n"));

  /* Setup matrix views for petgem */
  PetscCall(MatViewFromOptions(*A, NULL, "-petgem_mat_view"));
  PetscCall(MatViewFromOptions(*G, NULL, "-petgem_grad_view"));

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
  PetscCall(PetscFree(gradientMatrix[0]));
  PetscCall(PetscFree(gradientMatrix));
  PetscCall(PetscFree(Me));
  PetscCall(PetscFree(Ke));
  PetscCall(PetscFree(closureLHS));

  PetscFunctionReturn(PETSC_SUCCESS);
}