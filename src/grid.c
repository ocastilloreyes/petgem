/*
 * Filename: grid.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2025-06-04
 *
 * Description:
 * This file contains a collection of functions for grid
 * functions that are used throughout the PETGEM. These
 * functions are based on DMPlex provided by PETSc.
 *
 * Usage:
 * Include this file in your source code to utilize the grid
 * functions. For example: #include "grid.h"
 *
 */

/* C libraries */

/* PETSc libraries */
#include <petscdmplex.h>
#include <petscviewerhdf5.h>

/* PETGEM funcions*/
#include "constants.h"
#include "grid.h"
#include "inputs.h"

/**
 * @brief Imports a DMPlex mesh and associated resistivity field from an HDF5 file.
 *
 * This function reads a PETSc-formatted HDF5 file containing a DMPlex mesh
 * (named "petgem_mesh") and a resistivity vector (named "resistivity"). It
 * supports distributed meshes for parallel runs and handles DM cloning
 * so that the output DM can be used independently from the internal
 * load DM.
 *
 * The function performs the following steps:
 *   - Creates and initializes a DMPlex object.
 *   - Loads topology, labels, and coordinates from the HDF5 file.
 *   - Distributes the mesh across MPI ranks if necessary.
 *   - Loads the global resistivity vector and scatters it to a local vector.
 *   - Clones the loaded DM for main computations (`odm`) while maintaining
 *     shared topology and coordinates.
 *   - Processes DM options from the command line (`-dm_vec_type` and `-dm_mat_type`).
 *
 * @param[in]  params                Struct containing simulation parameters,
 *                                   including the HDF5 mesh filename.
 * @param[out] odm                   Pointer to the cloned DMPlex object
 *                                   that will be used in computations.
 * @param[out] resistivity_output    Pointer to a local Vec storing
 *                                   the resistivity values for the local portion
 *                                   of the mesh. The vector holds a unique
 *                                   reference and can be used independently
 *                                   of the DM used for loading.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code
 *         on failure.
 *
 * @note This function uses PETSc viewers, PetscSF objects, and DM distribution
 *       routines to handle parallel mesh loading. All output vectors and DMs
 *       are allocated and ready for use in PETSc parallel computations.
 */
PetscErrorCode importGrid(const csemParams params, DM* odm, Vec* resistivity_output) {
  PetscFunctionBegin;

  /* Variables declaration */
  PetscViewer viewer;
  DM dm, dmDist;
  PetscSF sfLoad, sfDist, sfG;
  PetscSF sfXC = NULL;
  Vec resistivity, globalResistivity;
  char typeName[PETSC_MAX_PATH_LEN];
  PetscBool flg;
  size_t load;

  /* Create and setup DM object */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(PetscStrlen(params.meshFile, &load));
  if (!load) {
    PetscCall(DMSetFromOptions(dm));
    *odm = dm;
    *resistivity_output = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* Must use the same name of mesh used to dump the HDF5
   * file */
  PetscCall(PetscObjectSetName((PetscObject)dm, "petgem_mesh"));
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, params.meshFile, FILE_MODE_READ, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_PETSC));
  PetscCall(DMPlexTopologyLoad(dm, viewer, &sfLoad));
  PetscCall(DMPlexLabelsLoad(dm, viewer, sfLoad));
  PetscCall(DMPlexCoordinatesLoad(dm, viewer, sfLoad));
  PetscCall(DMPlexDistribute(dm, 0, &sfDist, &dmDist));
  if (dmDist) {
    PetscCall(PetscSFCompose(sfLoad, sfDist, &sfXC));
    PetscCall(DMDestroy(&dm));
    dm = dmDist;
    PetscCall(PetscObjectSetName((PetscObject)dm, "petgem_mesh"));
  } else {
    PetscCall(PetscObjectReference((PetscObject)sfLoad));
    sfXC = sfLoad;
  }
  PetscCall(DMViewFromOptions(dm, NULL, "-load_dm_view"));

  PetscCall(DMPlexSectionLoad(dm, viewer, NULL, sfXC, &sfG, NULL));
  PetscCall(DMCreateGlobalVector(dm, &globalResistivity));
  PetscCall(PetscObjectSetName((PetscObject)globalResistivity, "resistivity"));
  PetscCall(DMPlexGlobalVectorLoad(dm, viewer, NULL, sfG, globalResistivity));
  PetscCall(VecViewFromOptions(globalResistivity, NULL, "-load_resistivity_view"));
  PetscCall(DMCreateLocalVector(dm, &resistivity));
  PetscCall(DMGlobalToLocal(dm, globalResistivity, INSERT_VALUES, resistivity));
  PetscCall(VecDestroy(&globalResistivity));

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscSFDestroy(&sfLoad));
  PetscCall(PetscSFDestroy(&sfDist));
  PetscCall(PetscSFDestroy(&sfXC));
  PetscCall(PetscSFDestroy(&sfG));

  /* When cloning, you get a new DM that can have its own
     section and labels The actual mesh (topology,
     coordinates) is shared between the two DMs */
  PetscCall(DMClone(dm, odm));

  /* We can destroy here, because the resistivity vector
   * will hold a unique reference to it */
  PetscCall(DMDestroy(&dm));

  /* Setup output vector */
  *resistivity_output = resistivity;

  /* Process some DM options */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_vec_type", typeName, 256, &flg));
  if (flg)
    PetscCall(DMSetVecType(*odm, typeName));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_mat_type", typeName, 256, &flg));
  if (flg)
    PetscCall(DMSetMatType(*odm, typeName));

  PetscFunctionReturn(PETSC_SUCCESS);
}

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
  PetscInt numH1DofInCell = 4;
  PetscInt numH1Dof[4] = {1, 0, 0, 0};
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

  /* DM for H1 conforming space TODO XXX make it depend on
   * params.nord */
  PetscCall(DMClone(*dm, &H1dm));
  PetscCall(DMSetNumFields(H1dm, 1));
  PetscCall(DMPlexCreateSection(H1dm, NULL, numComp, numH1Dof, 0, NULL, NULL, NULL, NULL, &section));
  PetscCall(DMSetLocalSection(H1dm, section));
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
  for (PetscInt f = faceStart; f < faceEnd; f++) {
    /* This is the global index of face f, using the
       convention that if it is negative it is not owned in
       parallel */
    if (gidxs[f - pStart] >= 0) {
      numFacesLocal += 1;
    }
  }

  /* Get total num of faces by MPI reduction */
  PetscCall(MPI_Allreduce(&numFacesLocal, &numFacesGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

  /* Get numbering for edges (depth 1) */
  PetscCall(DMPlexGetDepthStratum(*dm, 1, &edgeStart, &edgeEnd));

  /* Compute local number of edges */
  for (PetscInt e = edgeStart; e < edgeEnd; e++) {
    /* This is the global index of edge e, using the
       convention that if it is negative it is not owned in
       parallel */
    if (gidxs[e - pStart] >= 0) {
      numEdgesLocal += 1;
    }
  }

  /* Get total num of edges by MPI reduction */
  PetscCall(MPI_Allreduce(&numEdgesLocal, &numEdgesGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

  /* Get numbering for vertices (depth 0) */
  PetscCall(DMPlexGetDepthStratum(*dm, 0, &vertexStart, &vertexEnd));

  /* Compute local number of vertices */
  for (PetscInt v = vertexStart; v < vertexEnd; v++) {
    /* This is the global index of vertex v, using the
       convention that if it is negative it is not owned in
       parallel */
    if (gidxs[v - pStart] >= 0) {
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

  grid->numH1DofInCell = numH1DofInCell;
  grid->H1dm = H1dm;

  /* Print grid data */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n Mesh data:\n"));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Filename         = %s\n", params.meshFile));
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
PetscErrorCode extractCellCoordinates(DM dm, PetscInt cellID, Cell* cell) {
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
 * @brief Extracts the resistivity components of a given cell.
 *
 * This function retrieves the resistivity values associated with a specific
 * cell from a PETSc Vec defined on a DMPlex object and stores them in a
 * user-provided `Cell` struct. It uses `DMPlexVecGetClosure` and
 * `DMPlexVecRestoreClosure` to access the local cell data safely.
 *
 * @param[in]  dmResistivity  DMPlex object representing the resistivity field layout.
 * @param[in]  resistivity    Vec containing the resistivity values defined on the mesh.
 * @param[in]  cellID         Index of the cell to extract resistivity from.
 * @param[out] cell           Pointer to a `Cell` struct where the resistivity
 *                            components for the cell will be stored. Assumes
 *                            space for `NUM_RESISTIVITY_COMPONENTS` entries.
 *
 * @return PetscErrorCode     PETSC_SUCCESS on success, or a PETSc error code otherwise.
 *
 * @note The function assumes each cell has exactly `NUM_RESISTIVITY_COMPONENTS` (3) resistivity values.
 *       If this is not satisfied, a PETSc error is raised.
 * @note The resistivity values are copied as `PetscReal` into `cell->resistivity`.
 * @note Temporary internal arrays returned by `DMPlexVecGetClosure` are restored
 *       before the function returns.
 */
PetscErrorCode extractCellResistivity(DM dmResistivity, Vec resistivity, PetscInt cellID, Cell* cell) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt numResistivityComponents;
  PetscScalar* resistivityValues = NULL;

  PetscCall(DMPlexVecGetClosure(dmResistivity, NULL, resistivity, cellID, &numResistivityComponents, &resistivityValues));

  PetscCheck(numResistivityComponents == NUM_RESISTIVITY_COMPONENTS, PETSC_COMM_SELF, PETSC_ERR_SUP,
             "Exiting: found resistivity components != 3.\n");

  for (PetscInt i = 0; i < NUM_RESISTIVITY_COMPONENTS; i++) {
    cell->resistivity[i] = PetscRealPart(resistivityValues[i]);
  }

  PetscCall(DMPlexVecRestoreClosure(dmResistivity, NULL, resistivity, cellID, &numResistivityComponents, &resistivityValues));

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
PetscErrorCode extractCellClousure(DM dm, PetscInt cellID, Cell* cell) {
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
