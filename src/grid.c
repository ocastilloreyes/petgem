/*
 * Filename: grid.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Grid-handling functions used throughout PETGEM, built on
 * PETSc's DMPlex.
 */

/* C libraries */

/* PETSc libraries */
#include <petscdmplex.h>
#include <petscviewerhdf5.h>

/* PETGEM funcions*/
#include "constants.h"
#include "grid.h"
#include "hvfem.h"
#include "inputs.h"

/**
 * @brief Configures a DMPlex object with H(curl) and H1 sections for CSEM simulations.
 *
 * This function sets up the primary DMPlex object for CSEM modeling
 * using high-order edge (H(curl)) elements of order `params.nord`
 * and a corresponding H1 conforming space. The following steps are performed:
 *
 *   - Sets the number of fields in the DM to 1.
 *   - Creates a "Boundary" label and marks boundary faces with ID 100.
 *   - Computes degrees of freedom (DOFs) per vertex, edge, face, and volume
 *     according to the PETGEM basis order (`params.nord`).
 *   - Creates and attaches a PetscSection for H(curl) elements,
 *     applying boundary conditions on the marked faces.
 *   - Clones the DM to create an H1 conforming DM (stored in `grid->H1dm`).
 *   - Computes local and global counts of vertices, edges, faces, and cells.
 *   - Stores DOF counts, element start/end indices, and dimension in the `grid` struct.
 *   - Prints mesh and HEFEM statistics for verification.
 *
 * @param[in]  params  Struct containing simulation parameters, especially the basis order `params.nord` and mesh filename.
 * @param[inout] dm    Pointer to the DMPlex object to configure with H(curl) and H1 sections.
 * @param[out] grid    Pointer to the Grid struct to be populated with mesh statistics, DOF counts, and the H1 DM.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 *
 * @note The function uses PETSc parallel reductions (MPI_Allreduce) to compute global mesh statistics
 *       and relies on PETSc DMPlex utilities to handle boundary labeling, section creation,
 *       and point numbering. Output is printed collectively using PETSc routines.
 * @note The H1 DM (`grid->H1dm`) is cloned from the H(curl) DM and can be used independently for
 *       additional computations, e.g., scalar potential fields.
 */
PetscErrorCode setupCsemGrid(const csemParams params, DM* dm, Grid* grid) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  DMLabel labelBoundary;
  PetscSection section;
  PetscInt numComp[] = {1};
  PetscInt numBC = 1;
  PetscInt bcField[1] = {0};
  PetscInt numDofInVertex, numDofInEdge, numDofInFace, numDofInVolume, numDofInCell;
  /* Per-depth H1 DOF counts (vertex, edge, face, cell). The H1 space
   * order tracks params.nord so the Nédélec discrete gradient
   *     ∇(P_nord H1)  ⊂  Nédélec_nord
   * is exactly representable, which is the property PCBDDCSetDiscreteGradient
   * relies on at any order. P_nord nodal layout:
   *     vertex:  1
   *     edge  :  nord - 1
   *     face  :  (nord-1)(nord-2)/2
   *     cell  :  (nord-1)(nord-2)(nord-3)/6
   * Total per cell = (nord+1)(nord+2)(nord+3)/6. For nord=1 this collapses
   * to {1,0,0,0} (4 vertex DOFs), reproducing the previous P1 layout. */
  PetscInt numH1Dof[NUM_H1_DOF_PER_CELL];
  PetscInt numCellsLocal = 0, numCellsGlobal = 0, numFacesLocal = 0, numFacesGlobal = 0;
  PetscInt numEdgesLocal = 0, numEdgesGlobal = 0, numVerticesLocal = 0, numVerticesGlobal = 0;
  PetscInt dim, pStart, cellStart, cellEnd, faceStart, faceEnd, edgeStart, edgeEnd, vertexStart, vertexEnd;
  DM H1dm;
  IS boundaryIS, globalPointNumbering;
  const PetscInt* gidxs;

  /* Initial setup for DM object*/
  PetscCall(DMSetNumFields(*dm, 1));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  /*  Create label for dirichlet boundary conditions
      Values for boundaries:
      - boundaries = 100  */
  PetscCall(DMCreateLabel(*dm, "Boundary"));
  PetscCall(DMGetLabel(*dm, "Boundary", &labelBoundary));
  PetscCall(DMPlexMarkBoundaryFaces(*dm, 100, labelBoundary));
  PetscCall(DMPlexLabelComplete(*dm, labelBoundary));

  /*  Create PetscSection. The PETSc convention in 3
     dimensions is to number first cells, then vertices,
     then faces, and then edges. The above statement is not
     always true and we should not rely on that. It may be
     true for meshes read from GMSH files, but not for
     others. We are only guaranteed that points at different
     depths (or different heights, depending from where we
     start looking at the DAG) are numbered contiguously,
     this is why we can get start and end. Also, note that
     this numbering is purely local, because it is used to
     perform local mesh traversals   */
  /* Compute DOFs for PETGEM basis functions at vertex,
   * edges, faces, and volume */
  numDofInVertex = 0;
  numDofInEdge = params.nord;
  numDofInFace = params.nord * (params.nord - 1);
  numDofInVolume = params.nord * (params.nord - 1) * (params.nord - 2) / 2;
  numDofInCell = params.nord * (params.nord + 2) * (params.nord + 3) / 2;
  PetscInt numDof[4] = {numDofInVertex, numDofInEdge, numDofInFace, numDofInVolume};

  /* Get the IS for boundaries */
  PetscCall(DMLabelGetStratumIS(labelBoundary, 100, &boundaryIS));
  PetscCall(DMPlexCreateSection(*dm, NULL, numComp, numDof, numBC, bcField, NULL, &boundaryIS, NULL, &section));
  PetscCall(DMSetLocalSection(*dm, section));
  PetscCall(PetscSectionDestroy(&section));

  /* DM for the P1 H1 space used by the inverse kernel's discrete
   * gradient (assembleCsemKandM). The forward kernel uses H1dm_Pnord
   * (below) so PCBDDC receives the order-k discrete gradient with
   * K·G = 0. */
  numH1Dof[0] = 1;
  numH1Dof[1] = 0;
  numH1Dof[2] = 0;
  numH1Dof[3] = 0;
  const PetscInt numH1DofInCell = NUM_H1_DOF_PER_CELL;

  PetscCall(DMClone(*dm, &H1dm));
  PetscCall(DMSetNumFields(H1dm, 1));
  PetscCall(DMPlexCreateSection(H1dm, NULL, numComp, numH1Dof, 0, NULL,
                                NULL, NULL, NULL, &section));
  PetscCall(DMSetLocalSection(H1dm, section));
  PetscCall(PetscSectionDestroy(&section));

  /* DM for the order-k S_h^k space — P_nord nodal + edge/face/volume
   * bubbles, sized so ∇P_nord = curl-kernel of Nédélec_nord (the De Rham
   * complex). The forward-kernel discrete gradient G : S_h^k → V_h^k
   * built against this basis satisfies K·G = 0 element-wise. For
   * nord = 1 the counts collapse to {1,0,0,0} and this DM is
   * structurally identical to H1dm; for nord >= 2 it adds bubble DOFs. */
  PetscInt numH1Dof_Pnord[NUM_H1_DOF_PER_CELL];
  numH1Dof_Pnord[0] = 1;
  numH1Dof_Pnord[1] = params.nord - 1;
  numH1Dof_Pnord[2] = (params.nord - 1) * (params.nord - 2) / 2;
  numH1Dof_Pnord[3] = (params.nord - 1) * (params.nord - 2) * (params.nord - 3) / 6;
  const PetscInt numH1DofInCell_Pnord =
      (params.nord + 1) * (params.nord + 2) * (params.nord + 3) / 6;

  DM H1dm_Pnord = NULL;
  PetscCall(DMClone(*dm, &H1dm_Pnord));
  PetscCall(DMSetNumFields(H1dm_Pnord, 1));
  PetscCall(DMPlexCreateSection(H1dm_Pnord, NULL, numComp, numH1Dof_Pnord, 0, NULL,
                                NULL, NULL, NULL, &section));
  PetscCall(DMSetLocalSection(H1dm_Pnord, section));
  PetscCall(PetscSectionDestroy(&section));

  /* Create point numbering */
  PetscCall(DMPlexCreatePointNumbering(*dm, &globalPointNumbering));
  PetscCall(ISGetIndices(globalPointNumbering, &gidxs));

  /* pStart is almost always 0, but we support nonzero too
   */
  PetscCall(DMPlexGetChart(*dm, &pStart, NULL));

  /* Get numbering for cells (height 0) */
  PetscCall(DMPlexGetHeightStratum(*dm, 0, &cellStart, &cellEnd));

  /* Get number of local cells */
  numCellsLocal = cellEnd - cellStart;

  /* Get total num of cells by MPI reduction */
  PetscCall(MPI_Allreduce(&numCellsLocal, &numCellsGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

  /* Get numbering for faces (depth 2) */
  PetscCall(DMPlexGetDepthStratum(*dm, 2, &faceStart, &faceEnd));

  /* Compute local number of faces */
  for (PetscInt i = faceStart; i < faceEnd; i++) {
    /* This is the global index of face i, using the
       convention that if it is negative it is not owned in
       parallel */
    if (gidxs[i - pStart] >= 0) {
      numFacesLocal += 1;
    }
  }

  /* Get total num of faces by MPI reduction */
  PetscCall(MPI_Allreduce(&numFacesLocal, &numFacesGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

  /* Get numbering for edges (depth 1) */
  PetscCall(DMPlexGetDepthStratum(*dm, 1, &edgeStart, &edgeEnd));

  /* Compute local number of edges */
  for (PetscInt i = edgeStart; i < edgeEnd; i++) {
    /* This is the global index of edge i, using the
       convention that if it is negative it is not owned in
       parallel */
    if (gidxs[i - pStart] >= 0) {
      numEdgesLocal += 1;
    }
  }

  /* Get total num of edges by MPI reduction */
  PetscCall(MPI_Allreduce(&numEdgesLocal, &numEdgesGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

  /* Get numbering for vertices (depth 0) */
  PetscCall(DMPlexGetDepthStratum(*dm, 0, &vertexStart, &vertexEnd));

  /* Compute local number of vertices */
  for (PetscInt i = vertexStart; i < vertexEnd; i++) {
    /* This is the global index of vertex i, using the
       convention that if it is negative it is not owned in
       parallel */
    if (gidxs[i - pStart] >= 0) {
      numVerticesLocal += 1;
    }
  }

  /* Get total num of vertices by MPI reduction */
  PetscCall(MPI_Allreduce(&numVerticesLocal, &numVerticesGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

  /* Get number of dimensions */
  PetscCall(DMGetDimension(*dm, &dim));

  /* Setup petgemGrid */
  grid->numCellsLocal = numCellsLocal;
  grid->numCellsGlobal = numCellsGlobal;
  grid->numFacesLocal = numFacesLocal;
  grid->numFacesGlobal = numFacesGlobal;
  grid->numEdgesLocal = numEdgesLocal;
  grid->numEdgesGlobal = numEdgesGlobal;
  grid->numVerticesLocal = numVerticesLocal;
  grid->numVerticesGlobal = numVerticesGlobal;
  grid->numDofInVertex = numDofInVertex;
  grid->numDofInEdge = numDofInEdge;
  grid->numDofInFace = numDofInFace;
  grid->numDofInVolume = numDofInVolume;
  grid->numDofInCell = numDofInCell;
  grid->cellStart = cellStart;
  grid->cellEnd = cellEnd;
  grid->faceStart = faceStart;
  grid->faceEnd = faceEnd;
  grid->edgeStart = edgeStart;
  grid->edgeEnd = edgeEnd;
  grid->vertexStart = vertexStart;
  grid->vertexEnd = vertexEnd;
  grid->dim = dim;
  grid->numH1DofInCell       = numH1DofInCell;
  grid->numH1DofInCell_Pnord = numH1DofInCell_Pnord;
  grid->H1dm                 = H1dm;
  grid->H1dm_Pnord           = H1dm_Pnord;

  /* Mirror the FEM space descriptor used by hvfem/assembly/postprocessing.
   *
   * Layout in the per-cell H(curl) vector matches PETSc DMPlex's
   * closure-traversal order (cell -> faces -> edges -> vertices) so that
   * cell-local slot k aligns with dofIndices[k] from DMPlexGetClosureIndices.
   * The per-order shape3DETet permutation table in src/hvfem_hierarchical.c
   * is constructed against this same order.
   *
   *   nord=1: edges only            (edgeDofOffset = 0, rest empty).
   *   nord=2: faces, then edges     (faces 0..7, edges 8..19).
   *   nord>=3: volume, then faces, then edges (volume 0..nVol-1,
   *           faces nVol..nVol+nFace-1, edges nVol+nFace..end).
   *
   * The offset for an empty class is set to numDofInCell so loops can
   * iterate `[offset, offset+count)` and become no-ops when count=0,
   * without needing a per-order switch in the consumer. */
  {
    const PetscInt nEdge = NUM_EDGES_PER_CELL * numDofInEdge;
    const PetscInt nFace = NUM_FACES_PER_CELL * numDofInFace;
    const PetscInt nVol  = numDofInVolume;

    grid->fem.nord            = params.nord;
    grid->fem.numDofInCell    = numDofInCell;
    grid->fem.numH1DofInCell        = numH1DofInCell;
    grid->fem.numH1DofInCell_Pnord  = numH1DofInCell_Pnord;

    grid->fem.numDofPerEdge   = numDofInEdge;
    grid->fem.numDofPerFace   = numDofInFace;
    grid->fem.numDofPerVolume = numDofInVolume;

    grid->fem.numEdgeDof      = nEdge;
    grid->fem.numFaceDof      = nFace;
    grid->fem.numVolumeDof    = nVol;

    if (params.nord == 1) {
      /* edges only */
      grid->fem.edgeDofOffset   = 0;
      grid->fem.faceDofOffset   = numDofInCell;
      grid->fem.volumeDofOffset = numDofInCell;
    } else if (nVol == 0) {
      /* nord=2: faces, then edges (no volume DOFs) */
      grid->fem.faceDofOffset   = 0;
      grid->fem.edgeDofOffset   = nFace;
      grid->fem.volumeDofOffset = numDofInCell;
    } else {
      /* nord >= 3: volume, then faces, then edges. Matches PETSc
       * DMPlexGetClosureIndices order: cell-DOFs first, then face-DOFs,
       * then edge-DOFs. */
      grid->fem.volumeDofOffset = 0;
      grid->fem.faceDofOffset   = nVol;
      grid->fem.edgeDofOffset   = nVol + nFace;
    }

    /* Per-order Nédélec dispatch table. Hot paths (computeElementalMatrices,
     * evaluateNedelecBasis) call through this pointer instead of switching
     * on nord. */
    grid->fem.ops = nedelecOpsForOrder(params.nord);
  }

  /* Print grid data */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n Mesh data:\n"));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Input file       = %s\n", params.inputFile));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Num of vertices  = %" PetscInt_FMT "\n", grid->numVerticesGlobal));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Num of edges     = %" PetscInt_FMT "\n", grid->numEdgesGlobal));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Num of faces     = %" PetscInt_FMT "\n", grid->numFacesGlobal));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Num of cells     = %" PetscInt_FMT "\n", grid->numCellsGlobal));

  /* Print HEFEM statistics */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n HEFEM data:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Basis order             = %" PetscInt_FMT "\n", params.nord));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Num of dofs per vertex  = %" PetscInt_FMT "\n", grid->numDofInVertex));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Num of dofs per edge    = %" PetscInt_FMT "\n", grid->numDofInEdge));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Num of dofs per Face    = %" PetscInt_FMT "\n", grid->numDofInFace));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Num of dofs per volume  = %" PetscInt_FMT "\n", grid->numDofInVolume));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Num of dofs per cell    = %" PetscInt_FMT "\n", grid->numDofInCell));

  /* Restore global numbering and free memory */
  PetscCall(ISRestoreIndices(globalPointNumbering, &gidxs));
  PetscCall(ISDestroy(&globalPointNumbering));
  PetscCall(ISDestroy(&boundaryIS));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Locates the mesh cell containing a given point (e.g., source or receiver position).
 *
 * This function uses `DMLocatePoints` to determine which DMPlex cell owns the specified
 * point coordinates. It performs a parallel check across all MPI ranks to ensure that
 * the point is contained in at least one process. If the point is not found globally,
 * the function raises a PETSc error.
 *
 * @param[in]  dm           DMPlex object representing the computational mesh.
 * @param[in]  position     Array of size 3 containing the [x, y, z] coordinates of the point.
 * @param[out] pointInCell  Pointer to an integer where the index of the containing cell
 *                          will be stored. Set to -1 if the point is not found locally.
 *
 * @return PetscErrorCode   PETSC_SUCCESS on success, or a PETSc error code on failure.
 *
 * @note The search is performed collectively across all MPI processes using
 *       `MPI_Allreduce` with logical OR to ensure the point exists somewhere in the domain.
 *       Each process may return -1 if it does not own the point.
 * @note A PETSc error is raised if the point is not located on any process.
 * @note This function allocates a temporary Vec for point coordinates and a PetscSF
 *       for the search, both of which are destroyed before returning.
 */
PetscErrorCode locatePoint(const DM dm, const PetscReal* position, PetscInt* pointInCell) {
  PetscFunctionBeginUser;

  /* Variable declarations */
  Vec pointCoordinates;
  PetscSF pointSF = NULL;
  PetscInt numPointsFound;
  PetscScalar* inputPointCoordinates;
  const PetscSFNode* pointCell;
  const PetscInt* pointFound;
  PetscBool pointFoundGlobal = PETSC_FALSE;
  PetscBool pointFoundLocal;

  /* Prepare PETSc vector with point coordinates */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, NUM_DIMENSIONS, &pointCoordinates));
  PetscCall(VecSetBlockSize(pointCoordinates, NUM_DIMENSIONS));
  PetscCall(VecGetArrayWrite(pointCoordinates, &inputPointCoordinates));
  inputPointCoordinates[0] = position[0];
  inputPointCoordinates[1] = position[1];
  inputPointCoordinates[2] = position[2];
  PetscCall(VecRestoreArrayWrite(pointCoordinates, &inputPointCoordinates));

  /* Search point within computational domain */
  PetscCall(DMLocatePoints(dm, pointCoordinates, DM_POINTLOCATION_NONE, &pointSF));
  PetscCall(PetscSFGetGraph(pointSF, NULL, &numPointsFound, &pointFound, &pointCell));

  for (PetscInt i = 0; i < numPointsFound; i++) {
    *pointInCell = pointCell[i].index;
  }

  /* Perform validation (at least one MPI task must found
     the point). Each process sets its local flag */
  pointFoundLocal = (*pointInCell < 0) ? PETSC_FALSE : PETSC_TRUE;

  /* Gather all local flags to the master process */
  PetscCallMPI(MPI_Allreduce(&pointFoundLocal, &pointFoundGlobal, 1, MPI_C_BOOL, MPI_LOR, PETSC_COMM_WORLD));

  /* Petsc check */
  PetscCheck(pointFoundGlobal, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG,
             "Exiting: CSEM source position not located. "
             "Verify source parameters or improve mesh "
             "quality.\n");

  /* Free memory */
  PetscCall(VecDestroy(&pointCoordinates));
  PetscCall(PetscSFDestroy(&pointSF));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Extracts the coordinates of the vertices of a given cell.
 *
 * This function retrieves the coordinates of all vertices of the specified
 * cell from a DMPlex object and stores them in a user-provided `Cell` struct.
 * It uses `DMPlexGetCellCoordinates` and `DMPlexRestoreCellCoordinates`
 * to access the cell geometry safely.
 *
 * @param[in]  dm      DMPlex object representing the computational mesh.
 * @param[in]  cellID  Global/local index of the cell to extract coordinates from.
 * @param[out] cell    Pointer to a `Cell` struct where the coordinates of
 *                     the cell's vertices will be stored. The struct is assumed
 *                     to have sufficient space for `NUM_VERTICES_PER_CELL * NUM_DIMENSIONS`.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 *
 * @note The function assumes that each cell has exactly `NUM_VERTICES_PER_CELL` vertices
 *       and each vertex has `NUM_DIMENSIONS` coordinates. If this is not satisfied,
 *       a PETSc error is raised.
 * @note The coordinates are copied as `PetscReal` values into the `cell->coordinates` array.
 * @note Temporary internal arrays returned by `DMPlexGetCellCoordinates` are restored
 *       before the function returns.
 */
PetscErrorCode extractCellCoordinates(const DM dm, const  PetscInt cellID, Cell* cell) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscBool isDG;
  PetscInt numCoords;
  const PetscScalar* arrayCoords;
  PetscScalar* cellCoordsScalar = NULL;

  PetscCall(DMPlexGetCellCoordinates(dm, cellID, &isDG, &numCoords, &arrayCoords, &cellCoordsScalar));

  PetscCheck(numCoords == NUM_VERTICES_PER_CELL * NUM_DIMENSIONS, PETSC_COMM_SELF, PETSC_ERR_SUP,
             "Exiting: unexpected number of cell coordinates.\n");

  for (PetscInt i = 0; i < numCoords; ++i) {
    cell->coordinates[i] = PetscRealPart(cellCoordsScalar[i]);
  }

  PetscCall(DMPlexRestoreCellCoordinates(dm, cellID, &isDG, &numCoords, &arrayCoords, &cellCoordsScalar));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Extracts the conductivity components of a given cell.
 *
 * This function retrieves the conductivity values associated with a specific
 * cell from a PETSc Vec defined on a DMPlex object and stores them in a
 * user-provided `Cell` struct. It uses `DMPlexVecGetClosure` and
 * `DMPlexVecRestoreClosure` to access the local cell data safely.
 *
 * @param[in]  dmConductivity DMPlex object representing the conductivity field layout.
 * @param[in]  conductivity    Vec containing the conductivity values defined on the mesh.
 * @param[in]  cellID         Index of the cell to extract conductivity from.
 * @param[out] cell           Pointer to a `Cell` struct where the conductivity
 *                            components for the cell will be stored. Assumes
 *                            space for `NUM_CONDUCTIVITY_COMPONENTS` entries.
 *
 * @return PetscErrorCode     PETSC_SUCCESS on success, or a PETSc error code otherwise.
 *
 * @note The function assumes each cell has exactly `NUM_CONDUCTIVITY_COMPONENTS` (3) conductivity values.
 *       If this is not satisfied, a PETSc error is raised.
 * @note The conductivity values are copied as `PetscReal` into `cell->conductivity`.
 * @note Temporary internal arrays returned by `DMPlexVecGetClosure` are restored
 *       before the function returns.
 */
PetscErrorCode extractCellConductivity(DM dmConductivity, Vec conductivity, PetscInt cellID, Cell* cell) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt numConductivityComponents;
  PetscScalar* conductivityValues = NULL;

  PetscCall(DMPlexVecGetClosure(dmConductivity, NULL, conductivity, cellID, &numConductivityComponents, &conductivityValues));

  PetscCheck(numConductivityComponents == NUM_CONDUCTIVITY_COMPONENTS, PETSC_COMM_SELF, PETSC_ERR_SUP,
             "Exiting: found conductivity components != 3.\n");

  for (PetscInt i = 0; i < NUM_CONDUCTIVITY_COMPONENTS; i++) {
    cell->conductivity[i] = PetscRealPart(conductivityValues[i]);
  }

  PetscCall(DMPlexVecRestoreClosure(dmConductivity, NULL, conductivity, cellID, &numConductivityComponents, &conductivityValues));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Extracts the material ID of a given cell.
 *
 * This function retrieves the material identifier associated with a specific
 * cell from a PETSc Vec defined on a DMPlex sub-DM (field 1 of the model
 * section, 1 dof per cell) and stores it in a user-provided `Cell` struct.
 * It uses `DMPlexVecGetClosure` and `DMPlexVecRestoreClosure` to access the
 * local cell data safely.
 *
 * The function works in parallel: each MPI rank operates on its local portion
 * of the distributed mesh, and the materials_id Vec is the local vector
 * obtained from the global-to-local scatter performed in loadCsemInputs.
 *
 * @param[in]  dmMaterialsID  DMPlex sub-DM representing the materials_id
 *                            field layout (1 dof per cell).
 * @param[in]  materialsID    Local Vec containing the material IDs defined
 *                            on the local mesh partition.
 * @param[in]  cellID         Index of the cell to extract the material ID from.
 * @param[out] cell           Pointer to a `Cell` struct where the material ID
 *                            will be stored in `cell->material_id`.
 *
 * @return PetscErrorCode     PETSC_SUCCESS on success, or a PETSc error code
 *         otherwise.
 *
 * @note The function assumes each cell has exactly NUM_MATERIALS_ID_COMPONENTS
 *       (1) material ID value. A PETSc error is raised if this is not satisfied.
 * @note The material ID is stored as a floating-point value in the PETSc Vec
 *       (since PETSc scalars are complex); PetscRealPart is used to extract
 *       the real part before casting to PetscInt.
 */
PetscErrorCode extractCellMaterialID(DM dmMaterialsID, Vec materialsID, PetscInt cellID, Cell* cell) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt     numMaterialIDComponents;
  PetscScalar* materialIDValues = NULL;

  PetscCall(DMPlexVecGetClosure(dmMaterialsID, NULL, materialsID, cellID, &numMaterialIDComponents, &materialIDValues));

  PetscCheck(numMaterialIDComponents == NUM_MATERIALS_ID_COMPONENTS, PETSC_COMM_SELF, PETSC_ERR_SUP,
             "Exiting: found materials_id components != 1.\n");

  cell->material_id = (PetscInt)PetscRealPart(materialIDValues[0]);

  PetscCall(DMPlexVecRestoreClosure(dmMaterialsID, NULL, materialsID, cellID, &numMaterialIDComponents, &materialIDValues));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Extracts the transitive closure of a given cell.
 *
 * This function retrieves the transitive closure of a specified cell in a
 * DMPlex mesh and stores it in a user-provided `Cell` struct. The closure
 * contains all points (vertices, edges, faces, and the cell itself) that
 * make up the cell, along with their orientation.
 *
 * @param[in]  dm      DMPlex object representing the computational mesh.
 * @param[in]  cellID  Index of the cell whose transitive closure is to be extracted.
 * @param[out] cell    Pointer to a `Cell` struct where the closure will be stored.
 *                     Assumes space for `MAX_TRANSITIVE_CLOSURE_SIZE * 2` integers.
 *                     The field `cell->closureSize` will store the number of points
 *                     in the closure.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 *
 * @note The function uses `DMPlexGetTransitiveClosure` and `DMPlexRestoreTransitiveClosure`
 *       to access internal DMPlex data.
 * @note Each entry in the closure array is a pair: `(point, orientation)`.
 * @note A PETSc error is raised if the closure size exceeds `MAX_TRANSITIVE_CLOSURE_SIZE`.
 */
PetscErrorCode extractCellClousure(const DM dm, const PetscInt cellID, Cell* cell) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt transitiveClosureSize;
  PetscInt* transitiveClosure = NULL;

  PetscCall(DMPlexGetTransitiveClosure(dm, cellID, PETSC_TRUE, &transitiveClosureSize, &transitiveClosure));

  PetscCheck(transitiveClosureSize < MAX_TRANSITIVE_CLOSURE_SIZE, PETSC_COMM_SELF, PETSC_ERR_SUP,
             "Exiting: clousure size greater than "
             "MAX_TRANSITIVE_CLOSURE_SIZE.\n");

  cell->closureSize = transitiveClosureSize;

  for (PetscInt i = 0; i < transitiveClosureSize * 2; i++) {
    cell->closure[i] = transitiveClosure[i];
  }

  /* Restore transitive closure */
  PetscCall(DMPlexRestoreTransitiveClosure(dm, cellID, PETSC_TRUE, &transitiveClosureSize, &transitiveClosure));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints detailed connectivity and geometric information of a given tetrahedral cell
 *        in a DMPlex mesh.
 *
 * @param[in] dm The DMPlex object representing the unstructured mesh.
 * @param[in] cell The index of the cell whose entities are to be printed.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success.
 *
 * @details
 * This function retrieves and prints the following information for a given cell:
 *
 * 1. **Transitive closure of the cell**:
 *    - Includes all points (vertices, edges, faces) connected to the cell.
 *    - Prints the point index and its orientation.
 *
 * 2. **Face connectivity**:
 *    - Indices of faces associated with the cell.
 *    - For each face:
 *      - Indices of edges forming the face.
 *      - Indices of vertices forming the face.
 *
 * 3. **Edge connectivity**:
 *    - Indices of edges associated with the cell.
 *    - For each edge:
 *      - Indices of the two vertices defining the edge.
 *
 * 4. **Vertex coordinates**:
 *    - Coordinates of each vertex in the cell in 3D space.
 *
 * 5. **Edge midpoints**:
 *    - Computed as the average of the coordinates of the two vertices of the edge.
 *
 * @note
 * - Assumes tetrahedral cells with:
 *     - `NUM_FACES_PER_CELL` = 4
 *     - `NUM_EDGES_PER_CELL` = 6
 *     - `NUM_VERTICES_PER_CELL` = 4
 *     - `NUM_VERTICES_PER_EDGE` = 2
 *     - `NUM_EDGES_PER_FACE` = 3
 *     - `NUM_VERTICES_PER_FACE` = 3
 * - Relies on DMPlex functions:
 *     - `DMPlexGetTransitiveClosure` for retrieving connected points
 *     - `DMPlexGetCone` for face-to-edge and edge-to-vertex connectivity
 *     - `DMPlexGetCellCoordinates` for vertex coordinates
 */
PetscErrorCode printCellEntities(const DM dm, const PetscInt cell) {
  PetscFunctionBeginUser;

  /* Variable declarations */
  PetscInt cellFaces[NUM_FACES_PER_CELL];
  PetscInt cellEdges[NUM_EDGES_PER_CELL];
  PetscInt faceEdges[NUM_FACES_PER_CELL][NUM_EDGES_PER_FACE];
  PetscInt faceVertices[NUM_FACES_PER_CELL][NUM_VERTICES_PER_FACE];
  PetscInt edgeVertices[NUM_EDGES_PER_CELL][NUM_VERTICES_PER_EDGE];

  PetscInt transitiveClosureCellSize;
  PetscInt* transitiveClosureCellPoints = NULL;
  PetscInt transitiveClosureFaceSize;
  PetscInt* transitiveClosureFacePoints = NULL;
  const PetscInt* conePoints;
  PetscInt currentPoint;
  PetscInt currentFace;
  PetscBool isDG;
  PetscInt numCoords;
  const PetscScalar* arrayCoords;
  PetscScalar* cellCoords = NULL;

  PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &transitiveClosureCellSize, &transitiveClosureCellPoints));

  /* Get faces indices for cell, edges for each face, and vertices for each face */
  currentPoint = 2;
  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; i++) {
    /* Face indexes */
    cellFaces[i] = transitiveClosureCellPoints[currentPoint + i * 2];
    PetscCall(DMPlexGetCone(dm, cellFaces[i], &conePoints));

    /* Edges for each face*/
    for (PetscInt j = 0; j < NUM_EDGES_PER_FACE; j++) {
      faceEdges[i][j] = conePoints[j];
    }

    /* Vertices for each face */
    /* Orden convention:
    - Edges indices start on position 2
    - Vertices indices start on position 2 + NUM_EDGES_PER_FACE * 2
    */
    PetscCall(DMPlexGetTransitiveClosure(dm, cellFaces[i], PETSC_TRUE, &transitiveClosureFaceSize, &transitiveClosureFacePoints));
    currentFace = 8;
    for (PetscInt k = 0; k < NUM_VERTICES_PER_FACE; k++) {
      faceVertices[i][k] = transitiveClosureFacePoints[currentFace + k * 2];
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, cellFaces[i], PETSC_TRUE, &transitiveClosureFaceSize, &transitiveClosureFacePoints));
  }

  /* Get edges indices for cell */
  currentPoint = 2 + NUM_FACES_PER_CELL * 2;
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; i++) {
    cellEdges[i] = transitiveClosureCellPoints[currentPoint + i * 2];
    PetscCall(DMPlexGetCone(dm, cellEdges[i], &conePoints));
    for (PetscInt j = 0; j < NUM_VERTICES_PER_EDGE; j++) {
      edgeVertices[i][j] = conePoints[j];
    }
  }

  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\nData for cell %" PetscInt_FMT ":\n", cell));

  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[Cell %" PetscInt_FMT "] transitive closure size = %" PetscInt_FMT "\n ", cell, transitiveClosureCellSize));

  for (PetscInt i = 0; i < transitiveClosureCellSize; i++) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, " [Cell %" PetscInt_FMT "] closure entry %" PetscInt_FMT " = point %" PetscInt_FMT "(orientation %" PetscInt_FMT ")\n ", cell, i,
                                      transitiveClosureCellPoints[2 * i], transitiveClosureCellPoints[2 * i + 1]));
  }
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));

  /* Face --> vertices connectivity */
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Face vertices:\n"));
  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; i++) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "  [Face %" PetscInt_FMT "] vertices = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n", cellFaces[i], faceVertices[i][0],
                                      faceVertices[i][1], faceVertices[i][2]));
  }
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));

  /* Edge --> vertices connectivity */
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Edge vertices:\n"));
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; i++) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[Edge %" PetscInt_FMT "] vertices = (%" PetscInt_FMT ", %" PetscInt_FMT ")\n ", cellEdges[i], edgeVertices[i][0],
                                      edgeVertices[i][1]));
  }
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));

  /* Face --> edges connectivity */
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Face edges:\n"));
  for (PetscInt i = 0; i < NUM_FACES_PER_CELL; i++) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[Face %" PetscInt_FMT "] edges = (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n ", cellFaces[i], faceEdges[i][0],
                                      faceEdges[i][1], faceEdges[i][2]));
  }
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));

  /* Get/print cell coordinates */
  PetscCall(DMPlexGetCellCoordinates(dm, cell, &isDG, &numCoords, &arrayCoords, &cellCoords));

  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Vertex coordinates:\n"));
  currentPoint = 2 + NUM_FACES_PER_CELL * 2 + NUM_EDGES_PER_CELL * 2;
  for (PetscInt i = 0; i < NUM_VERTICES_PER_CELL; i++) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[Vertex %" PetscInt_FMT "] coordinates = (%g, %g, %g)\n",
                                      transitiveClosureCellPoints[currentPoint], PetscRealPart(cellCoords[3 * i + 0]),
                                      PetscRealPart(cellCoords[3 * i + 1]), PetscRealPart(cellCoords[3 * i + 2])));
    currentPoint += 2;
  }
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));

  /* Print edge midpoints */
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Edge midpoints:\n"));
  currentPoint = 2 + NUM_FACES_PER_CELL * 2;
  for (PetscInt i = 0; i < NUM_EDGES_PER_CELL; i++) {
    PetscInt v0 = EDGE_VERTICES[i][0];
    PetscInt v1 = EDGE_VERTICES[i][1];

    PetscReal xm = 0.5 * (PetscRealPart(cellCoords[3 * v0 + 0]) + PetscRealPart(cellCoords[3 * v1 + 0]));
    PetscReal ym = 0.5 * (PetscRealPart(cellCoords[3 * v0 + 1]) + PetscRealPart(cellCoords[3 * v1 + 1]));
    PetscReal zm = 0.5 * (PetscRealPart(cellCoords[3 * v0 + 2]) + PetscRealPart(cellCoords[3 * v1 + 2]));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[Edge %" PetscInt_FMT "] midpoint coordinates = (%g, %g, %g)\n", transitiveClosureCellPoints[currentPoint],
                          xm, ym, zm));
    currentPoint += 2;
  }
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));

  /* Restore transitive clousure and cell coordinates */
  PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &transitiveClosureCellSize, &transitiveClosureCellPoints));
  PetscCall(DMPlexRestoreCellCoordinates(dm, cell, &isDG, &numCoords, &arrayCoords, &cellCoords));

  PetscFunctionReturn(PETSC_SUCCESS);
}



/**
 * @brief Compute the geometric centroid (vertex barycenter) of a tetrahedral cell.
 *
 * Writes the result into `cell->centroid` (NUM_DIMENSIONS reals).  The
 * input `cell->coordinates` must already be populated by
 * extractCellCoordinates.  Used by the inversion smoother to weight
 * neighbour graph edges by inverse cell-to-cell distance.
 *
 * @param[in,out] cell  Cell struct; coordinates read, centroid written.
 * @return PetscErrorCode PETSC_SUCCESS on success.
 */
PetscErrorCode computeCellCentroid(Cell* cell) {
  PetscFunctionBeginUser;

  /* Initialize centroid */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        cell->centroid[i] = 0.0; 
  }

  /* Compute centroid */ 
  for (PetscInt i = 0; i < NUM_VERTICES_PER_CELL; i++) {
    for (PetscInt j = 0; j < NUM_DIMENSIONS; j++) {
            cell->centroid[j] += cell->coordinates[i * NUM_DIMENSIONS + j];
        }
    }

  /* Average */
  for (PetscInt i = 0; i < NUM_DIMENSIONS; i++) {
        cell->centroid[i] /= NUM_VERTICES_PER_CELL; 
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}




