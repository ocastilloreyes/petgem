/*
 * Filename: grid.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Prototypes for the grid-handling routines used throughout PETGEM,
 * built on PETSc's DMPlex.
 */

#ifndef GRID_H
#define GRID_H

#include "constants.h"
#include "inputs.h"
#include <petsc.h>

/* Forward declaration of the per-order Nédélec dispatch table; the full
 * definition lives in hvfem.h. Callers that need to invoke ops methods
 * include hvfem.h; callers that only pass FEMSpace around do not. */
struct NedelecOps;

/**
 * @brief Finite-element space descriptor.
 *
 * Bundles the fields needed by the hvfem routines (element matrices,
 * gradient matrix, DOF signs) so those signatures do not need to carry
 * nord + DOF counts separately.
 *
 * DOF-class layout inside the per-cell H(curl) vector (length numDofInCell)
 * matches PETSc DMPlex's closure-traversal order (cell -> faces -> edges
 * -> vertices), which is also what shape3DETet's permutation table is
 * built against:
 *   nord=1: edges only         [edges 0..5]                       (6)
 *   nord=2: faces-then-edges   [faces 0..7, edges 8..19]           (20)
 *   nord=3: volume-faces-edges [volume 0..2, faces 3..26,
 *                               edges 27..44]                     (45)
 *   nord=k (>=3) generally: [volume 0..nVol-1, faces nVol..nVol+nFace-1,
 *                            edges nVol+nFace..numDofInCell-1].
 * The *Offset fields mark the first index of each class (== numDofInCell
 * when the class is empty), so loops can iterate `[offset, offset+count)`
 * without a per-order switch.
 */
typedef struct {
  PetscInt nord;                  /**< Basis order (1, 2, 3, ...). */
  PetscInt numDofInCell;          /**< Total H(curl) DOFs per cell. */
  PetscInt numH1DofInCell_Pnord;  /**< P_nord H1 DOFs per cell; column count of
                                   *   the G_BDDC discrete gradient. */

  /* Per-entity DOF counts */
  PetscInt numDofPerEdge;    /**< H(curl) DOFs per edge. */
  PetscInt numDofPerFace;    /**< H(curl) DOFs per face. */
  PetscInt numDofPerVolume;  /**< H(curl) DOFs in the volume (interior). */

  /* Per-class totals (per cell) */
  PetscInt numEdgeDof;       /**< = NUM_EDGES_PER_CELL * numDofPerEdge. */
  PetscInt numFaceDof;       /**< = NUM_FACES_PER_CELL * numDofPerFace. */
  PetscInt numVolumeDof;     /**< = numDofPerVolume. */

  /* Offsets into the faces-first layout above */
  PetscInt faceDofOffset;    /**< First index of the face-DOF class. */
  PetscInt edgeDofOffset;    /**< First index of the edge-DOF class. */
  PetscInt volumeDofOffset;  /**< First index of the volume-DOF class. */

  /** Per-order Nédélec dispatch (coefficients / basis / curls / gradient).
   *  Populated in setupCsemGrid; hot loops call through this table instead
   *  of a switch (nord). */
  const struct NedelecOps *ops;
} FEMSpace;

/**
 * @brief Per-cell orientation data.
 *
 * Built once in computeCellOrientation and consumed by the shape / basis /
 * sign routines. A named struct replaces the earlier opaque
 * `orientation[10]` layout so higher-order codes can extend it (e.g.
 * per-face permutations for nord=3 face DOFs) without touching every accessor.
 */
typedef struct {
  PetscInt faces[NUM_FACES_PER_CELL];     /**< Face-orientation code in {0..5}. */
  PetscInt edgeSigns[NUM_EDGES_PER_CELL]; /**< ±1 sign for each edge. */
} CellOrientation;

/**
 * @brief Mesh statistics and DOF counts for a CSEM grid.
 */
typedef struct {
  PetscInt numCellsLocal;     /**< Number of local cells. */
  PetscInt numCellsGlobal;    /**< Number of global cells. */
  PetscInt numFacesLocal;     /**< Number of local faces. */
  PetscInt numFacesGlobal;    /**< Number of global faces. */
  PetscInt numEdgesLocal;     /**< Number of local edges. */
  PetscInt numEdgesGlobal;    /**< Number of global edges. */
  PetscInt numVerticesLocal;  /**< Number of local vertices. */
  PetscInt numVerticesGlobal; /**< Number of global vertices. */
  PetscInt numDofInVertex;    /**< Number of DOFs per vertex. */
  PetscInt numDofInEdge;      /**< Number of DOFs per edge. */
  PetscInt numDofInFace;      /**< Number of DOFs per face. */
  PetscInt numDofInVolume;    /**< Number of DOFs per volume. */
  PetscInt numDofInCell;      /**< Number of DOFs per cell. */
  PetscInt cellStart;         /**< Index of global cell start. */
  PetscInt cellEnd;           /**< Index of global cell end. */
  PetscInt faceStart;         /**< Index of global face start. */
  PetscInt faceEnd;           /**< Index of global face end. */
  PetscInt edgeStart;         /**< Index of global edge start. */
  PetscInt edgeEnd;           /**< Index of global edge end. */
  PetscInt vertexStart;       /**< Index of global vertex start. */
  PetscInt vertexEnd;         /**< Index of global vertex end. */
  PetscInt dim;               /**< Number of spatial dimensions. */

  PetscInt numH1DofInCell_Pnord; /**< P_nord H1 DOFs per cell
                                  *   (= (nord+1)(nord+2)(nord+3)/6); the column
                                  *   count of the G_BDDC discrete gradient
                                  *   produced by assembleCsemKandM. */
  DM H1dm_Pnord;                 /**< P_nord H1 DM: column space of the exact
                                  *   G_BDDC discrete gradient. For nord = 1 it
                                  *   is the P1 vertex space; for nord >= 2 it
                                  *   adds edge/face/volume bubble DOFs per the
                                  *   De Rham complex. */

  FEMSpace fem; /**< Finite-element space descriptor (mirrors nord + DOF counts). */
} Grid;

/**
 * @brief Per-cell geometry, material, and topology working data.
 */
typedef struct {
  PetscReal coordinates[NUM_VERTICES_PER_CELL * NUM_DIMENSIONS]; /**< Vertex coords (4×3). */
  PetscReal conductivity[NUM_CONDUCTIVITY_COMPONENTS];           /**< Cell conductivity (σx, σy, σz). */
  PetscInt  material_id;                                         /**< Material id of the cell. */
  PetscInt closure[MAX_TRANSITIVE_CLOSURE_SIZE];                 /**< DMPlex transitive closure points. */
  PetscInt closureSize;                                          /**< Number of valid closure entries. */
  PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS];            /**< Geometric Jacobian. */
  PetscReal invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS];         /**< Inverse Jacobian. */
  PetscReal detJacobian;                                         /**< Jacobian determinant. */
  CellOrientation orientation;                                   /**< Face/edge orientation data. */
  PetscReal centroid[NUM_DIMENSIONS];                            /**< Cell centroid (vertex barycenter). */
} Cell;

/**
 * @brief Configures a DMPlex object with H(curl) and H1 sections for CSEM.
 *
 * @param[in]     params  Forward-modeling parameters (nord).
 * @param[in,out] dm      DMPlex mesh configured with the FE sections.
 * @param[out]    grid    Grid struct filled with mesh statistics and FEMSpace.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode setupCsemGrid(const fmParams params, DM* dm, Grid* grid);

/**
 * @brief Locates the mesh cell containing a given point.
 *
 * @param[in]  dm           DMPlex mesh.
 * @param[in]  position     Query point coordinates (x, y, z).
 * @param[out] pointInCell  Index of the containing cell (or a not-found marker).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode locatePoint(const DM dm, const PetscReal* position, PetscInt* pointInCell);

/**
 * @brief Extracts the vertex coordinates of a given cell.
 *
 * @param[in]     dm      DMPlex mesh.
 * @param[in]     cellID  Cell index.
 * @param[in,out] cell    Cell whose coordinates field is filled.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode extractCellCoordinates(const DM dm, const PetscInt cellID, Cell* cell);

/**
 * @brief Extracts the conductivity components of a given cell.
 *
 * @param[in]     dmConductivity  DM carrying the conductivity field.
 * @param[in]     conductivity    Per-cell conductivity Vec.
 * @param[in]     cellID          Cell index.
 * @param[in,out] cell            Cell whose conductivity field is filled.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode extractCellConductivity(DM dmConductivity, Vec conductivity, PetscInt cellID, Cell* cell);

/**
 * @brief Extracts the material id of a given cell.
 *
 * @param[in]     dmMaterialsID  DM carrying the material-id field.
 * @param[in]     materialsID    Per-cell material-id Vec.
 * @param[in]     cellID         Cell index.
 * @param[in,out] cell           Cell whose material_id field is filled.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode extractCellMaterialID(DM dmMaterialsID, Vec materialsID, PetscInt cellID, Cell* cell);

/**
 * @brief Extracts the transitive closure of a given cell.
 *
 * @param[in]     dm      DMPlex mesh.
 * @param[in]     cellID  Cell index.
 * @param[in,out] cell    Cell whose closure/closureSize fields are filled.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode extractCellClousure(const DM dm, const PetscInt cellID, Cell* cell);

/**
 * @brief Computes the geometric centroid (vertex barycenter) of a cell.
 *
 * @param[in,out] cell  Cell whose centroid field is filled.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode computeCellCentroid(Cell* cell);

/**
 * @brief Prints connectivity and geometric information of a tetrahedral cell.
 *
 * @param[in] dm    DMPlex mesh.
 * @param[in] cell  Cell index.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PetscError code otherwise.
 */
PetscErrorCode printCellEntities(const DM dm, const PetscInt cell);

#endif
