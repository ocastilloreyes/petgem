/*
 * Filename: grid.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2024-06-04
 *
 * Description:
 * This file contains a collection of functions for grid functions that are used
 * throughout the PETGEM project. These functions are based on DMPlex provided by PETSc.
 *
 * List of functions:
 * 
 * 
 * Usage:
 * Include this file in your source code to utilize the common functions. 
 * For example:
 * #include "grid.h"
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

// =============================================================================
// Function: importGrid
// =============================================================================

/**
 * @brief Imports the mesh topology, coordinates, labels, and resistivity field from an HDF5 file.
 * @param[out] odm Pointer to the DMPlex object to be created and populated.
 * @param[out] resistivity_output Pointer to the Vec that will store the local resistivity values.
 * @param[in] params A Params struct containing simulation parameters, including the mesh filename.
 * @return PetscErrorCode PETSC_SUCCESS on success.
 * @details Reads a PETSc-formatted HDF5 file containing a DMPlex mesh ("petgem_mesh") and
 *          an associated Vec ("resistivity"). It handles mesh distribution for parallel runs.
 *          The function clones the loaded DM for the main computation (`odm`) and returns
 *          the resistivity field as a local Vec. DM options like VecType and MatType are processed.
 */

PetscErrorCode importGrid(DM *odm, Vec *resistivity_output, Params params)
{
    PetscViewer viewer;
    DM          dm, dmDist;
    PetscSF     sfLoad, sfDist, sfG;
    PetscSF     sfXC      = NULL;
    Vec         resistivity, globalResistivity;
    size_t      load;

    PetscFunctionBegin;
    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(PetscStrlen(params.meshFile, &load));
    if (!load) {
      PetscCall(DMSetFromOptions(dm));
      *odm = dm;
      *resistivity_output = NULL;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    /* Must use the same name of mesh used to dump the HDF5 file */
    PetscCall(PetscObjectSetName((PetscObject)dm,"petgem_mesh"));
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, params.meshFile, FILE_MODE_READ, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_PETSC));
    PetscCall(DMPlexTopologyLoad(dm, viewer, &sfLoad));
    PetscCall(DMPlexLabelsLoad(dm, viewer, sfLoad));
    PetscCall(DMPlexCoordinatesLoad(dm, viewer, sfLoad));
    PetscCall(DMPlexDistribute(dm, 0, &sfDist, &dmDist));
    if (dmDist){
        PetscCall(PetscSFCompose(sfLoad, sfDist, &sfXC));
        PetscCall(DMDestroy(&dm));
        dm = dmDist;
        PetscCall(PetscObjectSetName((PetscObject)dm,"petgem_mesh"));
    } else {
        PetscCall(PetscObjectReference((PetscObject)sfLoad));
        sfXC = sfLoad;
    }
    PetscCall(DMViewFromOptions(dm, NULL, "-load_dm_view"));

    PetscCall(DMPlexSectionLoad(dm, viewer, NULL, sfXC, &sfG, NULL));
    PetscCall(DMCreateGlobalVector(dm, &globalResistivity));
    PetscCall(PetscObjectSetName((PetscObject)globalResistivity,"resistivity"));
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

    /* When cloning, you get a new DM that can have its own section and labels
       The actual mesh (topology, coordinates) is shared between the two DMs */
    PetscCall(DMClone(dm, odm));

    /* We can destroy here, because the resistivity vector will hold a unique reference to it */
    PetscCall(DMDestroy(&dm));

    /* Setup output vector */
    *resistivity_output = resistivity;

    /* Process some DM options */
    char typeName[256];
    PetscBool flg;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_vec_type", typeName, 256, &flg));
    if (flg) PetscCall(DMSetVecType(*odm, typeName));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-dm_mat_type", typeName, 256, &flg));
    if (flg) PetscCall(DMSetMatType(*odm, typeName));
    PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Sets up the DMPlex object with appropriate sections for H(curl) and H1 finite elements.
 *
 * Configures the primary DM for H(curl) elements of order @p params.nord. Concretely:
 * - Sets the number of fields to 1.
 * - Creates a "Boundary" label and marks boundary faces (ID 100).
 * - Computes DOFs per vertex, edge, face, and volume based on @p params.nord.
 * - Creates the PetscSection for H(curl) elements, applying boundary conditions to the marked faces.
 * - Computes and stores local and global counts of vertices, edges, faces, and cells in the @p grid struct.
 * - Stores DOF counts, element start/end indices, and dimension in the @p grid struct.
 * - Prints mesh statistics.
 *
 * @param[inout] dm Pointer to the DMPlex object to be configured.
 * @param[out] grid Pointer to the Grid struct to be populated with mesh statistics and DOF info.
 * @param[in] params A Params struct containing simulation parameters, especially the basis order (@p params.nord).
 * @return PetscErrorCode PETSC_SUCCESS on success, or an error code otherwise.
 */
PetscErrorCode setupGrid(DM *dm, Grid *grid, Params params) {

	PetscFunctionBeginUser;

    /* Initial setup for DM object*/
    PetscCall(DMSetNumFields(*dm, 1));
    PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

    /*  Create label for dirichlet boundary conditions
        Values for boundaries:
        - boundaries = 100  */
    DMLabel     labelBoundary;
    PetscCall(DMCreateLabel(*dm, "Boundary"));
    PetscCall(DMGetLabel(*dm, "Boundary", &labelBoundary));
    PetscCall(DMPlexMarkBoundaryFaces(*dm, 100, labelBoundary));
    PetscCall(DMPlexLabelComplete(*dm, labelBoundary));

    /*  Create PetscSection. The PETSc convention in 3 dimensions is to number
        first cells, then vertices, then faces, and then edges.
        The above statement is not always true and we should not rely on that.
        It may be true for meshes read from GMSH files, but not for others.
        We are only guaranteed that points at different depths (or different heights, depending
        from where we start looking at the DAG) are numbered contiguously, this is why we
        can get start and end. Also, note that this numbering is purely local, because it is
        used to perform local mesh traversals   */ 
    
    PetscSection  section;
    PetscInt      numComp[] = {1};
    IS            boundaryIS;
    PetscInt      numBC = 1;
    PetscInt      bcField[1] = {0};
    PetscInt      numDofInVertex, numDofInEdge, numDofInFace, numDofInVolume, numDofInCell;
    
    /* Compute DOFs for PETGEM basis functions at vertex, edges, faces, and volume */
    numDofInVertex      = 0;
    numDofInEdge        = params.nord;
    numDofInFace        = params.nord*(params.nord-1);
    numDofInVolume      = params.nord*(params.nord-1)*(params.nord-2)/2;
    numDofInCell        = params.nord*(params.nord+2)*(params.nord+3)/2;
    PetscInt numDof[4]  = {numDofInVertex, numDofInEdge, numDofInFace, numDofInVolume};

    /* Get the IS for boundaries */
    PetscCall(DMLabelGetStratumIS(labelBoundary, 100, &boundaryIS));
    PetscCall(DMPlexCreateSection(*dm, NULL, numComp, numDof, numBC, bcField, NULL, &boundaryIS, NULL, &section));
    PetscCall(DMSetLocalSection(*dm, section));
    PetscCall(PetscSectionDestroy(&section));

    /* DM for H1 conforming space TODO XXX make it depend on params.nord */
    PetscInt numH1DofInCell = 4;
    PetscInt numH1Dof[4] = {1, 0, 0, 0};
    DM H1dm;

    PetscCall(DMClone(*dm , &H1dm));
    PetscCall(DMSetNumFields(H1dm, 1));
    PetscCall(DMPlexCreateSection(H1dm, NULL, numComp, numH1Dof, 0, NULL, NULL, NULL, NULL, &section));
    PetscCall(DMSetLocalSection(H1dm, section));
    PetscCall(PetscSectionDestroy(&section));

    /* Compute mesh statistics (number of vertices, edges, faces, elements) */
    PetscInt    numCellsLocal=0, numCellsGlobal=0, numFacesLocal=0, numFacesGlobal=0;
    PetscInt    numEdgesLocal=0, numEdgesGlobal=0, numVerticesLocal=0, numVerticesGlobal=0;
    PetscInt    pStart, cellStart, cellEnd, faceStart, faceEnd, edgeStart, edgeEnd, vertexStart, vertexEnd;
    IS          globalPointNumbering;
    const PetscInt *gidxs; 
  
    /* Create point numbering */
    PetscCall(DMPlexCreatePointNumbering(*dm, &globalPointNumbering));
    PetscCall(ISGetIndices(globalPointNumbering, &gidxs));
    
    /* pStart is almost always 0, but we support nonzero too */
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
    for (PetscInt f=faceStart; f<faceEnd; f++){
        /* This is the global index of face f, using the 
           convention that if it is negative it is not owned in parallel */    
        if  (gidxs[f - pStart] >= 0){
            numFacesLocal += 1; 
        }
    }    

    /* Get total num of faces by MPI reduction */
    PetscCall(MPI_Allreduce(&numFacesLocal, &numFacesGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

    /* Get numbering for edges (depth 1) */
    PetscCall(DMPlexGetDepthStratum(*dm, 1, &edgeStart, &edgeEnd)); 

    /* Compute local number of edges */
    for (PetscInt e=edgeStart; e<edgeEnd; e++){
        /* This is the global index of edge e, using the 
           convention that if it is negative it is not owned in parallel */    
        if  (gidxs[e - pStart] >= 0){
            numEdgesLocal += 1; 
        }
    }    

    /* Get total num of edges by MPI reduction */
    PetscCall(MPI_Allreduce(&numEdgesLocal, &numEdgesGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

    /* Get numbering for vertices (depth 0) */
    PetscCall(DMPlexGetDepthStratum(*dm, 0, &vertexStart, &vertexEnd)); 

    /* Compute local number of vertices */
    for (PetscInt v=vertexStart; v<vertexEnd; v++){
        /* This is the global index of vertex v, using the convention
           that if it is negative it is not owned in parallel */
        if  (gidxs[v - pStart] >= 0){
            numVerticesLocal += 1; 
        }
    }    

    /* Get total num of vertices by MPI reduction */
    PetscCall(MPI_Allreduce(&numVerticesLocal, &numVerticesGlobal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

    /* Get number of dimensions */
    PetscInt    dim; 
    PetscCall(DMGetDimension(*dm, &dim));

    /* Setup petgemGrid */  
    grid->numCellsLocal     = numCellsLocal;     
    grid->numCellsGlobal    = numCellsGlobal;    
    grid->numFacesLocal     = numFacesLocal;     
    grid->numFacesGlobal    = numFacesGlobal;    
    grid->numEdgesLocal     = numEdgesLocal;     
    grid->numEdgesGlobal    = numEdgesGlobal;    
    grid->numVerticesLocal  = numVerticesLocal;  
    grid->numVerticesGlobal = numVerticesGlobal; 
    grid->numDofInVertex    = numDofInVertex;    
    grid->numDofInEdge      = numDofInEdge;      
    grid->numDofInFace      = numDofInFace;      
    grid->numDofInVolume    = numDofInVolume;
    grid->numDofInCell      = numDofInCell;    
    grid->cellStart         = cellStart; 
    grid->cellEnd           = cellEnd;           
    grid->faceStart         = faceStart;         
    grid->faceEnd           = faceEnd;           
    grid->edgeStart         = edgeStart;         
    grid->edgeEnd           = edgeEnd;           
    grid->vertexStart       = vertexStart;       
    grid->vertexEnd         = vertexEnd;         
    grid->dim               = dim;         

    grid->numH1DofInCell = numH1DofInCell;
    grid->H1dm = H1dm;

    /* Print petgemGrid data */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n Mesh data:\n"));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Filename         = %s\n", params.meshFile));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Num of vertices  = %" PetscInt_FMT "\n", grid->numVerticesGlobal));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Num of edges     = %" PetscInt_FMT "\n", grid->numEdgesGlobal));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Num of faces     = %" PetscInt_FMT "\n", grid->numFacesGlobal));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   Num of cells     = %" PetscInt_FMT "\n", grid->numCellsGlobal));
    
    /* Restore global numbering and free memory */
    PetscCall(ISRestoreIndices(globalPointNumbering, &gidxs));
    PetscCall(ISDestroy(&globalPointNumbering));
    PetscCall(ISDestroy(&boundaryIS));
    
    PetscFunctionReturn(PETSC_SUCCESS);
}


// =============================================================================
// Function: locateCSEMSource
// =============================================================================

/**
 * @brief Locates the cell containing a given point (e.g., a CSEM source position).
 * @param[in] dm The DMPlex object representing the mesh.
 * @param[in] position Array containing the [x, y, z] coordinates of the point to locate.
 * @param[out] pointInCell Pointer to an integer where the index of the containing cell will be stored. Set to -1 if not found locally.
 * @return PetscErrorCode PETSC_SUCCESS on success.
 * @details Uses `DMLocatePoints` to find which cell owns the given `position`.
 *          Performs an MPI reduction (`MPI_LOR`) to check if the point was found on *any* process.
 *          If the point is not found globally, it triggers a `PetscCheck` error.
 */

PetscErrorCode locatePoint(DM dm, PetscReal *position, PetscInt *pointInCell) {

    PetscFunctionBeginUser;

    /* Declarations */
    Vec         pointCoordinates;
    PetscSF     pointSF = NULL;
    PetscInt    numPointsFound;
    PetscScalar *inputPointCoordinates;
    const PetscSFNode   *pointCell;
    const PetscInt      *pointFound;

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
    
    /* Perform validation (at least one MPI task must found the point) */         
    PetscBool pointFoundGlobal = PETSC_FALSE;
    PetscBool pointFoundLocal;
    
    /* Each process sets its local flag */
    pointFoundLocal = (*pointInCell < 0) ? PETSC_FALSE : PETSC_TRUE;

    /* Gather all local flags to the master process */
    PetscCallMPI(MPI_Allreduce(&pointFoundLocal, &pointFoundGlobal, 1, MPI_INT, MPI_LOR, PETSC_COMM_WORLD));
    
    /* Petsc check */
    PetscCheck(pointFoundGlobal, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Exiting: CSEM source position not located. Verify source parameters or improve mesh quality.\n");

    /* Free memory */
    PetscCall(VecDestroy(&pointCoordinates));
    PetscCall(PetscSFDestroy(&pointSF));

    PetscFunctionReturn(PETSC_SUCCESS);
}
