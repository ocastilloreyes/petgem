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

/* Finite-element space descriptor. Bundles the fields needed by the
 * hvfem routines (element matrices, gradient matrix, DOF signs) so
 * those signatures do not need to carry nord + DOF counts separately.
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
 * without a per-order switch. */
typedef struct {
  PetscInt nord;                  /* Basis order (1, 2, 3, ...)            */
  PetscInt numDofInCell;          /* Total H(curl) DOFs per cell           */
  PetscInt numH1DofInCell;        /* P1 H1 DOFs per cell (= 4); used by the
                                   * inverse-kernel discrete gradient.     */
  PetscInt numH1DofInCell_Pnord;  /* P_nord H1 DOFs per cell; used by the
                                   * forward-kernel order-k discrete
                                   * gradient (K·G = 0).                   */

  /* Per-entity DOF counts */
  PetscInt numDofPerEdge;    /* H(curl) DOFs per edge                 */
  PetscInt numDofPerFace;    /* H(curl) DOFs per face                 */
  PetscInt numDofPerVolume;  /* H(curl) DOFs in the volume (interior) */

  /* Per-class totals (per cell) */
  PetscInt numEdgeDof;       /* = NUM_EDGES_PER_CELL * numDofPerEdge  */
  PetscInt numFaceDof;       /* = NUM_FACES_PER_CELL * numDofPerFace  */
  PetscInt numVolumeDof;     /* = numDofPerVolume                     */

  /* Offsets into the faces-first layout above */
  PetscInt faceDofOffset;
  PetscInt edgeDofOffset;
  PetscInt volumeDofOffset;

  /* Per-order Nédélec dispatch (coefficients / basis / curls / gradient).
   * Populated in setupCsemGrid; hot loops call through this table instead
   * of a switch (nord). */
  const struct NedelecOps *ops;
} FEMSpace;

/* Per-cell orientation data, built once in computeCellOrientation and
 * consumed by the shape / basis / sign routines.
 *   faces[f]       : face-orientation code in {0..5} (PETGEM convention)
 *   edgeSigns[e]   : ±1 sign for edge e
 * A named struct replaces the earlier opaque `orientation[10]` layout so
 * the higher-order codes can extend it (e.g. per-face permutations for
 * nord=3 face DOFs) without touching every accessor. */
typedef struct {
  PetscInt faces[NUM_FACES_PER_CELL];
  PetscInt edgeSigns[NUM_EDGES_PER_CELL];
} CellOrientation;

typedef struct {
  PetscInt numCellsLocal;     /* Number of local cells        */
  PetscInt numCellsGlobal;    /* Number of global cells */
  PetscInt numFacesLocal;     /* Number of local faces        */
  PetscInt numFacesGlobal;    /* Number of global cells */
  PetscInt numEdgesLocal;     /* Number of local edges        */
  PetscInt numEdgesGlobal;    /* Number of global edges   */
  PetscInt numVerticesLocal;  /* Number of local vertices */
  PetscInt numVerticesGlobal; /* Number of global vertices    */
  PetscInt numDofInVertex;    /* Number of dofs per vertex */
  PetscInt numDofInEdge;      /* Number of dofs per edge      */
  PetscInt numDofInFace;      /* Number of dofs per vertex    */
  PetscInt numDofInVolume;    /* Number of dofs per volume */
  PetscInt numDofInCell;      /* Number of dofs per cell      */
  PetscInt cellStart;         /* Index of global cell start   */
  PetscInt cellEnd;           /* Index of global cell end     */
  PetscInt faceStart;         /* Index of global face start   */
  PetscInt faceEnd;           /* Index of global face end     */
  PetscInt edgeStart;         /* Index of global edge start   */
  PetscInt edgeEnd;           /* Index of global edge end     */
  PetscInt vertexStart;       /* Index of global vertex start */
  PetscInt vertexEnd;         /* Index of global vertex end   */
  PetscInt dim;               /* Number of dimensions         */

  PetscInt numH1DofInCell;       /* P1 H1 DOFs per cell (= 4 always);
                                  * still held by the topological
                                  * gradient builder which writes ±1
                                  * vertex incidences into a 4-column
                                  * scratch buffer before tail-padding
                                  * into the P_nord H1 closure. */
  PetscInt numH1DofInCell_Pnord; /* P_nord H1 DOFs per cell
                                  * (= (nord+1)(nord+2)(nord+3)/6); used
                                  * by the order-k discrete gradients
                                  * (G and G_BDDC) produced by
                                  * assembleCsemKandM. The canonical G
                                  * satisfies K·G = 0 element-wise. */
  DM H1dm;                       /* P1 H1 DM, paired with the inverse-kernel G. */
  DM H1dm_Pnord;                 /* P_nord H1 DM, paired with the forward-kernel
                                  * order-k discrete gradient G. For nord = 1 this
                                  * is a duplicate of H1dm; for nord >= 2 it adds
                                  * edge/face/volume bubble DOFs per the De Rham
                                  * complex. */

  FEMSpace fem; /* Finite-element space descriptor (mirrors nord + DOF counts) */
} Grid;

typedef struct {
  PetscReal coordinates[NUM_VERTICES_PER_CELL * NUM_DIMENSIONS]; /* 12 */
  PetscReal conductivity[NUM_CONDUCTIVITY_COMPONENTS];
  PetscInt  material_id;
  PetscInt closure[MAX_TRANSITIVE_CLOSURE_SIZE];
  PetscInt closureSize;
  PetscReal jacobian[NUM_DIMENSIONS][NUM_DIMENSIONS];
  PetscReal invJacobian[NUM_DIMENSIONS][NUM_DIMENSIONS];
  PetscReal detJacobian;
  CellOrientation orientation;
  PetscReal centroid[NUM_DIMENSIONS];
} Cell;

PetscErrorCode setupCsemGrid(const csemParams params, DM* dm, Grid* grid);

PetscErrorCode locatePoint(const DM dm, const PetscReal* position, PetscInt* pointInCell);

PetscErrorCode extractCellCoordinates(const DM dm, const PetscInt cellID, Cell* cell);

PetscErrorCode extractCellConductivity(DM dmConductivity, Vec conductivity, PetscInt cellID, Cell* cell);

PetscErrorCode extractCellMaterialID(DM dmMaterialsID, Vec materialsID, PetscInt cellID, Cell* cell);

PetscErrorCode extractCellClousure(const DM dm, const PetscInt cellID, Cell* cell);

PetscErrorCode computeCellCentroid(Cell* cell);

PetscErrorCode printCellEntities(const DM dm, const PetscInt cell);

#endif
