/*
 * Filename: grid.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Grid-handling functions used throughout PETGEM, built on PETSc's DMPlex.
 */

/* C libraries */

/* PETSc libraries */
#include <petscdmplex.h>
#include <petscviewerhdf5.h>

/* PETGEM funcions*/
#include "common.h"
#include "constants.h"
#include "grid.h"

/* Number of points in [start, end) owned by this rank. */
static PetscInt countOwnedPoints(const PetscInt *globalNumbering, PetscInt pStart, PetscInt start, PetscInt end) {
  PetscInt n = 0;
  for (PetscInt p = start; p < end; p++) {
    if (globalNumbering[p - pStart] >= 0) n++;
  }
  return n;
}

/* Per-cell H(curl) layout in DMPlex closure order: volume, faces, edges. */
static void setupFemSpace(PetscInt order, FemSpace *fem) {
  fem->order           = order;
  fem->numDofPerEdge   = order;
  fem->numDofPerFace   = order * (order - 1);
  fem->numDofPerVolume = order * (order - 1) * (order - 2) / 2;
  fem->numEdgeDof      = NUM_EDGES_PER_CELL * fem->numDofPerEdge;
  fem->numFaceDof      = NUM_FACES_PER_CELL * fem->numDofPerFace;
  fem->numVolumeDof    = fem->numDofPerVolume;
  fem->numDofInCell    = order * (order + 2) * (order + 3) / 2;
  fem->numH1DofInCell  = (order + 1) * (order + 2) * (order + 3) / 6;

  fem->volumeDofOffset = fem->numVolumeDof ? 0 : fem->numDofInCell;
  fem->faceDofOffset   = fem->numFaceDof ? fem->numVolumeDof : fem->numDofInCell;
  fem->edgeDofOffset   = fem->numVolumeDof + fem->numFaceDof;
}

/* Attaches a one-field local section with numDof[depth] DOFs per point. */
static PetscErrorCode setLocalSection(DM dm, const PetscInt numDof[4], IS *constrained) {
  PetscFunctionBeginUser;
  PetscSection section;
  PetscInt     numComp[] = {1}, bcField[] = {0};
  PetscInt     numBc = constrained ? 1 : 0;

  PetscCall(DMSetNumFields(dm, 1));
  PetscCall(DMPlexCreateSection(dm, NULL, numComp, numDof, numBc, numBc ? bcField : NULL, NULL, constrained, NULL, &section));
  PetscCall(DMSetLocalSection(dm, section));
  PetscCall(PetscSectionDestroy(&section));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Configures a DMPlex object with H(curl) and H1 sections.
 *
 * Marks the outer boundary faces (label "Boundary", value 100), attaches the
 * order-p Nedelec section (boundary DOFs constrained for PETGEM_BC_PEC), clones
 * the DM for the P_order H1 space used by the discrete gradient, and counts the
 * global mesh entities.
 */
PetscErrorCode setupNedelecGrid(PetscInt order, PetgemBoundaryCondition bc, DM *dm, Grid *grid) {
  PetscFunctionBeginUser;

  DMLabel         labelBoundary;
  IS              boundaryIs, globalNumbering;
  const PetscInt *gidxs;
  PetscInt        pStart, start, end;
  PetscInt        localCounts[4], globalCounts[4];

  PetscCall(PetscMemzero(grid, sizeof(*grid)));
  setupFemSpace(order, &grid->fem);
  grid->bc = bc;

  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  /* Outer boundary faces */
  PetscCall(DMCreateLabel(*dm, "Boundary"));
  PetscCall(DMGetLabel(*dm, "Boundary", &labelBoundary));
  PetscCall(DMPlexMarkBoundaryFaces(*dm, 100, labelBoundary));
  PetscCall(DMPlexLabelComplete(*dm, labelBoundary));
  PetscCall(DMLabelGetStratumIS(labelBoundary, 100, &boundaryIs));

  /* H(curl) section: DOFs per vertex, edge, face, volume */
  {
    const PetscInt numDof[4] = {0, grid->fem.numDofPerEdge, grid->fem.numDofPerFace, grid->fem.numDofPerVolume};
    PetscCall(setLocalSection(*dm, numDof, bc == PETGEM_BC_PEC ? &boundaryIs : NULL));
  }
  PetscCall(ISDestroy(&boundaryIs));

  /* P_order H1 section (nodal + bubbles), so that grad(P_order) lies in Nedelec_order */
  {
    const PetscInt numDof[4] = {1, order - 1, (order - 1) * (order - 2) / 2, (order - 1) * (order - 2) * (order - 3) / 6};
    PetscCall(DMClone(*dm, &grid->H1dm));
    PetscCall(setLocalSection(grid->H1dm, numDof, NULL));
  }

  /* Local cell range and global entity counts */
  PetscCall(DMPlexGetHeightStratum(*dm, 0, &grid->cellStart, &grid->cellEnd));
  PetscCall(DMPlexCreatePointNumbering(*dm, &globalNumbering));
  PetscCall(ISGetIndices(globalNumbering, &gidxs));
  PetscCall(DMPlexGetChart(*dm, &pStart, NULL));
  localCounts[0] = grid->cellEnd - grid->cellStart;
  for (PetscInt depth = 2; depth >= 0; depth--) {
    PetscCall(DMPlexGetDepthStratum(*dm, depth, &start, &end));
    localCounts[3 - depth] = countOwnedPoints(gidxs, pStart, start, end);
  }
  PetscCall(ISRestoreIndices(globalNumbering, &gidxs));
  PetscCall(ISDestroy(&globalNumbering));
  PetscCallMPI(MPI_Allreduce(localCounts, globalCounts, 4, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)*dm)));
  grid->numCellsGlobal    = globalCounts[0];
  grid->numFacesGlobal    = globalCounts[1];
  grid->numEdgesGlobal    = globalCounts[2];
  grid->numVerticesGlobal = globalCounts[3];

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroyGrid(Grid *grid) {
  PetscFunctionBeginUser;
  PetscCall(DMDestroy(&grid->H1dm));
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
 * @note The search is performed collectively across all MPI processes using
 *       `MPI_Allreduce` with logical OR to ensure the point exists somewhere in the domain.
 *       Each process may return -1 if it does not own the point.
 * @note A PETSc error is raised if the point is not located on any process.
 * @note This function allocates a temporary Vec for point coordinates and a PetscSF
 *       for the search, both of which are destroyed before returning.
 */
PetscErrorCode locatePoint(const DM dm, const PetscReal *position, PetscInt *pointInCell) {
  PetscFunctionBeginUser;

  /* Variable declarations */
  Vec                pointCoordinates;
  PetscSF            pointSf = NULL;
  PetscInt           numPointsFound;
  PetscScalar       *inputPointCoordinates;
  const PetscSFNode *pointCell;
  const PetscInt    *pointFound;
  PetscBool          pointFoundGlobal = PETSC_FALSE;
  PetscBool          pointFoundLocal;

  *pointInCell = -1;

  /* Prepare PETSc vector with point coordinates */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, NUM_DIMENSIONS, &pointCoordinates));
  PetscCall(VecSetBlockSize(pointCoordinates, NUM_DIMENSIONS));
  PetscCall(VecGetArrayWrite(pointCoordinates, &inputPointCoordinates));
  inputPointCoordinates[0] = position[0];
  inputPointCoordinates[1] = position[1];
  inputPointCoordinates[2] = position[2];
  PetscCall(VecRestoreArrayWrite(pointCoordinates, &inputPointCoordinates));

  /* Search point within computational domain */
  PetscCall(DMLocatePoints(dm, pointCoordinates, DM_POINTLOCATION_NONE, &pointSf));
  PetscCall(PetscSFGetGraph(pointSf, NULL, &numPointsFound, &pointFound, &pointCell));

  for (PetscInt i = 0; i < numPointsFound; i++) {
    *pointInCell = pointCell[i].index;
  }

  /* Perform validation (at least one MPI task must found the point). Each process sets its local flag */
  pointFoundLocal = (*pointInCell < 0) ? PETSC_FALSE : PETSC_TRUE;

  /* Gather all local flags to the master process */
  PetscCallMPI(MPI_Allreduce(&pointFoundLocal, &pointFoundGlobal, 1, MPI_C_BOOL, MPI_LOR, PetscObjectComm((PetscObject)dm)));
  PetscCheck(pointFoundGlobal, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Point (%g, %g, %g) is outside the mesh.",
             (double)position[0], (double)position[1], (double)position[2]);

  /* Free memory */
  PetscCall(VecDestroy(&pointCoordinates));
  PetscCall(PetscSFDestroy(&pointSf));

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
 * @note The function assumes that each cell has exactly `NUM_VERTICES_PER_CELL` vertices
 *       and each vertex has `NUM_DIMENSIONS` coordinates. If this is not satisfied,
 *       a PETSc error is raised.
 * @note The coordinates are copied as `PetscReal` values into the `cell->coordinates` array.
 * @note Temporary internal arrays returned by `DMPlexGetCellCoordinates` are restored
 *       before the function returns.
 */
PetscErrorCode extractCellCoordinates(const DM dm, const PetscInt cellId, Cell *cell) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscBool          isDg;
  PetscInt           numCoords;
  const PetscScalar *arrayCoords;
  PetscScalar       *cellCoordsScalar = NULL;

  PetscCall(DMPlexGetCellCoordinates(dm, cellId, &isDg, &numCoords, &arrayCoords, &cellCoordsScalar));

  PetscCheck(numCoords == NUM_VERTICES_PER_CELL * NUM_DIMENSIONS, PETSC_COMM_SELF, PETSC_ERR_SUP,
             "Exiting: unexpected number of cell coordinates.\n");

  for (PetscInt i = 0; i < numCoords; ++i) {
    cell->coordinates[i] = PetscRealPart(cellCoordsScalar[i]);
  }

  PetscCall(DMPlexRestoreCellCoordinates(dm, cellId, &isDg, &numCoords, &arrayCoords, &cellCoordsScalar));

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
 * @note The function assumes each cell has exactly `NUM_CONDUCTIVITY_COMPONENTS` (3) conductivity values.
 *       If this is not satisfied, a PETSc error is raised.
 * @note The conductivity values are copied as `PetscReal` into `cell->conductivity`.
 * @note Temporary internal arrays returned by `DMPlexVecGetClosure` are restored
 *       before the function returns.
 */
PetscErrorCode extractCellConductivity(DM dmConductivity, Vec conductivity, PetscInt cellId, Cell *cell) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt     numConductivityComponents;
  PetscScalar *conductivityValues = NULL;

  PetscCall(DMPlexVecGetClosure(dmConductivity, NULL, conductivity, cellId, &numConductivityComponents, &conductivityValues));

  PetscCheck(numConductivityComponents == NUM_CONDUCTIVITY_COMPONENTS, PETSC_COMM_SELF, PETSC_ERR_SUP,
             "Exiting: found conductivity components != 3.\n");

  for (PetscInt i = 0; i < NUM_CONDUCTIVITY_COMPONENTS; i++) {
    cell->conductivity[i] = PetscRealPart(conductivityValues[i]);
  }

  PetscCall(DMPlexVecRestoreClosure(dmConductivity, NULL, conductivity, cellId, &numConductivityComponents, &conductivityValues));

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
 * of the distributed mesh, and the materialsId Vec is the local vector
 * obtained from the global-to-local scatter performed in loadModelInputs.
 *
 * @note The function assumes each cell has exactly NUM_MATERIALS_ID_COMPONENTS
 *       (1) material ID value. A PETSc error is raised if this is not satisfied.
 * @note The material ID is stored as a floating-point value in the PETSc Vec
 *       (since PETSc scalars are complex); PetscRealPart is used to extract
 *       the real part before casting to PetscInt.
 */
PetscErrorCode extractCellMaterialId(DM dmMaterialsId, Vec materialsId, PetscInt cellId, Cell *cell) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt     numMaterialIdComponents;
  PetscScalar *materialIdValues = NULL;

  PetscCall(DMPlexVecGetClosure(dmMaterialsId, NULL, materialsId, cellId, &numMaterialIdComponents, &materialIdValues));

  PetscCheck(numMaterialIdComponents == NUM_MATERIALS_ID_COMPONENTS, PETSC_COMM_SELF, PETSC_ERR_SUP,
             "Exiting: found materials_id components != 1.\n");

  cell->materialId = (PetscInt)PetscRealPart(materialIdValues[0]);

  PetscCall(DMPlexVecRestoreClosure(dmMaterialsId, NULL, materialsId, cellId, &numMaterialIdComponents, &materialIdValues));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Compute the geometric centroid (vertex barycenter) of a tetrahedral cell.
 *
 * Writes the result into `cell->centroid` (NUM_DIMENSIONS reals).  The
 * input `cell->coordinates` must already be populated by
 * extractCellCoordinates.  Used by the inversion smoother to weight
 * neighbor graph edges by inverse cell-to-cell distance.
 */
PetscErrorCode computeCellCentroid(Cell *cell) {
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

/**
 * @brief Prints the "Mesh" and "FEM space" sections of the run report.
 *
 * The global DOF count is the size of a global vector of @p dm.
 */
PetscErrorCode logGridSummary(const char *inputFile, const DM dm, const Grid *grid) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  Vec      v;
  PetscInt numDofs;

  PetscCall(DMGetGlobalVector(dm, &v));
  PetscCall(VecGetSize(v, &numDofs));
  PetscCall(DMRestoreGlobalVector(dm, &v));

  PetscCall(logSection(comm, "Mesh"));
  PetscCall(logKvStr(comm, "Input file", inputFile));
  PetscCall(logKvInt(comm, "Number of vertices", grid->numVerticesGlobal));
  PetscCall(logKvInt(comm, "Number of edges", grid->numEdgesGlobal));
  PetscCall(logKvInt(comm, "Number of faces", grid->numFacesGlobal));
  PetscCall(logKvInt(comm, "Number of cells", grid->numCellsGlobal));

  PetscCall(logSection(comm, "FEM space"));
  PetscCall(logKvInt(comm, "Basis order", grid->fem.order));
  PetscCall(logKvInt(comm, "DOFs per vertex", 0));
  PetscCall(logKvInt(comm, "DOFs per edge", grid->fem.numDofPerEdge));
  PetscCall(logKvInt(comm, "DOFs per face", grid->fem.numDofPerFace));
  PetscCall(logKvInt(comm, "DOFs per volume", grid->fem.numDofPerVolume));
  PetscCall(logKvInt(comm, "DOFs per cell", grid->fem.numDofInCell));
  PetscCall(logKvInt(comm, "Global DOFs", numDofs));
  PetscCall(logKvStr(comm, "Boundary condition", grid->bc == PETGEM_BC_PEC ? "PEC (n x E = 0)" : "natural"));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Returns the local outer-boundary faces of the mesh.
 *
 * This function reads stratum 100 of the "Boundary" label created by
 * setupNedelecGrid and keeps the depth-2 points (faces).
 */
PetscErrorCode getBoundaryFaces(const DM dm, IS *faces) {
  PetscFunctionBeginUser;

  DMLabel         label;
  IS              stratum;
  const PetscInt *points = NULL;
  PetscInt       *faceList;
  PetscInt        numPoints = 0, numFaces = 0, faceStart, faceEnd;

  PetscCall(DMGetLabel(dm, "Boundary", &label));
  PetscCheck(label, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "getBoundaryFaces: no \"Boundary\" label (call setupNedelecGrid first)");

  PetscCall(DMPlexGetDepthStratum(dm, 2, &faceStart, &faceEnd));
  PetscCall(DMLabelGetStratumIS(label, 100, &stratum));
  if (stratum) {
    PetscCall(ISGetLocalSize(stratum, &numPoints));
    PetscCall(ISGetIndices(stratum, &points));
  }

  PetscCall(PetscMalloc1(numPoints, &faceList));
  for (PetscInt i = 0; i < numPoints; i++) {
    if (points[i] >= faceStart && points[i] < faceEnd) faceList[numFaces++] = points[i];
  }

  if (stratum) {
    PetscCall(ISRestoreIndices(stratum, &points));
    PetscCall(ISDestroy(&stratum));
  }

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numFaces, faceList, PETSC_OWN_POINTER, faces));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Computes the geometry of an outer-boundary face.
 *
 * This function returns the face's support cell, its vertex coordinates, its
 * area and its unit normal oriented away from the support cell centroid.
 */
PetscErrorCode computeBoundaryFaceGeometry(const DM dm, const PetscInt face, PetscInt *cell,
                                           PetscReal vertices[NUM_VERTICES_PER_FACE][NUM_DIMENSIONS], PetscReal normal[NUM_DIMENSIONS],
                                           PetscReal *area) {
  PetscFunctionBeginUser;

  const PetscInt    *support;
  PetscInt           supportSize, numCoords;
  PetscBool          isDg;
  const PetscScalar *arrayCoords;
  PetscScalar       *coords = NULL;
  Cell               owner;
  PetscReal          e1[NUM_DIMENSIONS], e2[NUM_DIMENSIONS], outward[NUM_DIMENSIONS];
  PetscReal          length, orientation = 0.0;

  PetscCall(DMPlexGetSupportSize(dm, face, &supportSize));
  PetscCheck(supportSize == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "computeBoundaryFaceGeometry: face %" PetscInt_FMT " has %" PetscInt_FMT " support cells, expected 1", face, supportSize);
  PetscCall(DMPlexGetSupport(dm, face, &support));
  *cell = support[0];

  /* Face vertex coordinates */
  PetscCall(DMPlexGetCellCoordinates(dm, face, &isDg, &numCoords, &arrayCoords, &coords));
  PetscCheck(numCoords == NUM_VERTICES_PER_FACE * NUM_DIMENSIONS, PETSC_COMM_SELF, PETSC_ERR_SUP,
             "computeBoundaryFaceGeometry: unexpected number of face coordinates (%" PetscInt_FMT ")", numCoords);
  for (PetscInt v = 0; v < NUM_VERTICES_PER_FACE; v++) {
    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
      vertices[v][d] = PetscRealPart(coords[v * NUM_DIMENSIONS + d]);
    }
  }
  PetscCall(DMPlexRestoreCellCoordinates(dm, face, &isDg, &numCoords, &arrayCoords, &coords));

  /* Support cell centroid */
  PetscCall(extractCellCoordinates(dm, *cell, &owner));
  PetscCall(computeCellCentroid(&owner));

  /* Normal (v1 - v0) x (v2 - v0), oriented away from the cell centroid */
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    e1[d]      = vertices[1][d] - vertices[0][d];
    e2[d]      = vertices[2][d] - vertices[0][d];
    outward[d] = (vertices[0][d] + vertices[1][d] + vertices[2][d]) / 3.0 - owner.centroid[d];
  }
  normal[0] = e1[1] * e2[2] - e1[2] * e2[1];
  normal[1] = e1[2] * e2[0] - e1[0] * e2[2];
  normal[2] = e1[0] * e2[1] - e1[1] * e2[0];

  length = PetscSqrtReal(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
  PetscCheck(length > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "computeBoundaryFaceGeometry: face %" PetscInt_FMT " is degenerate", face);
  *area = 0.5 * length;

  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) orientation += normal[d] * outward[d];
  if (orientation < 0.0) length = -length;
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) normal[d] /= length;

  PetscFunctionReturn(PETSC_SUCCESS);
}
