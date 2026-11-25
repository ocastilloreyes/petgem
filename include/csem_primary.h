/*
 * Filename: csem_primary.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-11-20
 *
 * Description:
 * Field formulation of fm.csem and im.csem (total or secondary), the
 * background model of the secondary formulation, the primary field E_p,h on
 * the H(curl) space and the secondary right-hand side.
 */

/*
 * Notes:
 * The background model is read from the /background group of the input bundle
 * or of the file given by -csem_background_file:
 *   /background/layer_ztop       Vec of N reals: top of each layer (m, z up),
 *                                layer_ztop[0] = +inf, then strictly decreasing
 *   /background/layer_sigma      Vec of 2N reals: sigma_h, sigma_v per layer (S/m)
 *   /background/material_sigma   Vec of 3M reals: sigma_x, sigma_y, sigma_z per
 *                                material id (S/m)
 */

#ifndef CSEM_PRIMARY_H
#define CSEM_PRIMARY_H

#include <petsc.h>

#include "csem.h"
#include "csem_layered.h"
#include "grid.h"
#include "io.h"
#include "receiver_interp.h"
#include "solver.h"

/** @brief Field formulation. */
typedef enum {
  CSEM_FORMULATION_TOTAL,    /**< Total field E. */
  CSEM_FORMULATION_SECONDARY /**< Secondary field E - E_p of a background model. */
} CsemFormulation;

/** @brief Background model that gives the primary field E_p. */
typedef enum {
  CSEM_PRIMARY_LAYERED, /**< Layered VTI medium, /background/layer_*. */
  CSEM_PRIMARY_MODEL    /**< Conductivity per material on the same mesh, /background/material_sigma. */
} CsemPrimaryType;

/** @brief Formulation options -csem_*. */
typedef struct {
  CsemFormulation formulation;                        /**< -csem_formulation. */
  CsemPrimaryType primaryType;                        /**< -csem_primary. */
  PetscBool       primaryTypeSet;                     /**< PETSC_TRUE if -csem_primary was given. */
  char            backgroundFile[PETSC_MAX_PATH_LEN]; /**< -csem_background_file (default: the input bundle). */
} CsemParams;

/** @brief Background model of the secondary formulation. */
typedef struct {
  CsemPrimaryType type;          /**< Model held by this struct. */
  LayeredModel    layered;       /**< CSEM_PRIMARY_LAYERED: layers, top to bottom. */
  PetscInt        numMaterials;  /**< CSEM_PRIMARY_MODEL: number of materials. */
  PetscReal      *materialSigma; /**< CSEM_PRIMARY_MODEL: sigma_x, sigma_y, sigma_z per material (3 * numMaterials). */
} CsemBackground;

/**
 * @brief Reads the formulation options -csem_formulation, -csem_primary and -csem_background_file.
 *
 * @param[in]  params      Parameters (input file, default of -csem_background_file).
 * @param[out] csemParams  Formulation options.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode readCsemParams(const PetgemParams *params, CsemParams *csemParams);

/**
 * @brief Loads the background model from the /background group.
 *
 * Without -csem_primary the type is layered if /background/layer_sigma
 * exists and model otherwise. Reads on PETSC_COMM_SELF (each rank reads
 * independently).
 *
 * @param[in]  csemParams  Formulation options.
 * @param[out] background  Background model; release it with destroyCsemBackground().
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode loadCsemBackground(const CsemParams *csemParams, CsemBackground *background);

/**
 * @brief Frees a background model loaded by loadCsemBackground().
 *
 * @param[in,out] background  Background model.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode destroyCsemBackground(CsemBackground *background);

/** @brief Background conductivity per cell and primary field E_p,h of one or more source sets. */
typedef struct {
  CsemPrimaryType       type;             /**< Background model. */
  const CsemBackground *background;       /**< Background model (not owned). */
  Vec                   sigmaBackground;  /**< Background conductivity per cell, laid out like the conductivity Vec. */
  PetscBool            *cells;            /**< Cells where E_p,h is set and the right-hand side is assembled (cell - cellStart). */
  PetscInt              numAnomalyCells;  /**< Global number of cells with sigma != sigma_b. */
  PetscInt              numCrossingCells; /**< Global number of cells across an interface of the layered background. */
  PetscInt              numSets;          /**< Number of source sets. */
  CsemSourceSet        *sets;             /**< Source sets (copies). */
  Vec                 **Ep;               /**< Ep[s][k]: E_p,h of source k of set s, local Vec of the H(curl) section. */
  PetscLogDouble        time;             /**< Wall time of computeCsemPrimary, maximum over ranks (s). */
  SolveInfo             solveInfo;        /**< Background solve of CSEM_PRIMARY_MODEL (last set). */
} CsemPrimary;

/**
 * @brief Sets the background conductivity of every cell and the cells with sigma != sigma_b.
 *
 * Layered: (sigma_h, sigma_h, sigma_v) of the layer that contains the cell
 * centroid; cells that cross an interface are counted and reported with a
 * WARNING. Model: the background conductivity of the cell material. A cell
 * differs from the background when |sigma - sigma_b| > 1e-10 max(sigma,
 * sigma_b) in a component.
 *
 * @param[in]  background    Background model (loadCsemBackground); must outlive primary.
 * @param[in]  dm            DMPlex mesh carrying the H(curl) section.
 * @param[in]  grid          Grid filled by setupNedelecGrid().
 * @param[in]  conductivity  Per-cell conductivity Vec.
 * @param[in]  materialsId   Per-cell material id Vec (CSEM_PRIMARY_MODEL; may be NULL otherwise).
 * @param[out] primary       Primary field; release it with destroyCsemPrimary().
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode createCsemPrimary(const CsemBackground *background, const DM dm, const Grid *grid, Vec conductivity, Vec materialsId,
                                 CsemPrimary *primary);

/**
 * @brief Computes the primary field E_p,h of every source of the source sets.
 *
 * Layered: Nédélec interpolation (interpolateNedelecCell) of the analytic
 * field on the cells with sigma != sigma_b and on extraCells, boundary DOFs
 * included; a DOF shared by several of these cells takes the mean of their
 * values, so E_p,h does not depend on the partition. Model: finite-element solve with the background conductivity
 * (assembleMaxwellOperator, assembleCsemRhs, solveMaxwellSystem), one per set.
 * Layered sources may lie outside the mesh; the computation fails if one lies
 * in a cell of primary->cells.
 *
 * @param[in]     numSets     Number of source sets (one frequency each).
 * @param[in]     sets        Source sets (copied).
 * @param[in]     dm          DMPlex mesh carrying the H(curl) section.
 * @param[in]     grid        Grid filled by setupNedelecGrid().
 * @param[in]     extraCells  Further cells where E_p,h is needed (cell - cellStart), or NULL.
 * @param[in,out] primary     Primary field (createCsemPrimary).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode computeCsemPrimary(PetscInt numSets, const CsemSourceSet sets[], const DM dm, const Grid *grid, const PetscBool extraCells[],
                                  CsemPrimary *primary);

/**
 * @brief Assembles the secondary right-hand sides of every source set.
 *
 * Column k of B[s]: b_k = iωμ_s Σ_e M_e(σ_e - σ_b,e) E_p,h|_e over the cells of
 * primary->cells, with the rows of constrained DOFs removed.
 *
 * @param[in]     primary       Primary field (computeCsemPrimary).
 * @param[in]     dm            DMPlex mesh carrying the H(curl) section.
 * @param[in]     grid          Grid filled by setupNedelecGrid().
 * @param[in]     conductivity  Per-cell conductivity Vec.
 * @param[in,out] B             Right-hand sides, B[s] one column per source of set s (createRhsMatrix); overwritten.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode assembleCsemSecondaryRhs(const CsemPrimary *primary, const DM dm, const Grid *grid, Vec conductivity, Mat B[]);

/**
 * @brief Adds the primary field of one source to the fields at the receivers.
 *
 * Layered: analytic E_p and H_p at the receivers of the local rows; a
 * receiver on an interface takes the layer of the cell that interpolates it
 * (Q->cellCentroids). Model: Q E_p,h, with H = Q_H E_p,h / (iωμ).
 *
 * @param[in]     primary    Primary field (computeCsemPrimary).
 * @param[in]     set        Source set.
 * @param[in]     source     Source of the set.
 * @param[in]     dm         DMPlex mesh carrying the H(curl) section.
 * @param[in]     Q          Receiver interpolation matrices.
 * @param[in]     receivers  Receiver coordinates (sequential Vec of 3 * numReceivers).
 * @param[in,out] fields     Fields at the receivers (createReceiverFields).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode addCsemPrimaryAtReceivers(const CsemPrimary *primary, PetscInt set, PetscInt source, const DM dm,
                                         const ReceiverInterpolationMatrices *Q, Vec receivers, Vec fields[NUM_RECEIVER_FIELDS]);

/** @brief Context of addCsemPrimaryField. */
typedef struct {
  const CsemPrimary *primary; /**< Primary field (computeCsemPrimary) of a single source set. */
  DM                 dm;      /**< DMPlex mesh carrying the H(curl) section. */
} CsemPrimaryFieldCtx;

/**
 * @brief Adds the primary field of source k of the single source set to the
 *        fields at the receivers.
 *
 * CsemReceiverPrimaryFn passed to computeCsemResponses() by fm.csem in the
 * secondary formulation.
 *
 * @param[in]     k          Source index.
 * @param[in]     Q          Receiver interpolation matrices.
 * @param[in]     receivers  Receiver coordinates (sequential Vec of 3 * numReceivers).
 * @param[in,out] fields     Fields at the receivers (createReceiverFields).
 * @param[in]     ctx        CsemPrimaryFieldCtx.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode addCsemPrimaryField(PetscInt k, const ReceiverInterpolationMatrices *Q, Vec receivers, Vec fields[NUM_RECEIVER_FIELDS],
                                   void *ctx);

/**
 * @brief Prints the "CSEM formulation" section of the run report.
 *
 * @param[in] comm     Communicator of the report.
 * @param[in] primary  Primary field (computeCsemPrimary).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode logCsemFormulation(MPI_Comm comm, const CsemPrimary *primary);

/**
 * @brief Frees a primary field created by createCsemPrimary().
 *
 * @param[in,out] primary  Primary field.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode destroyCsemPrimary(CsemPrimary *primary);

#endif /* CSEM_PRIMARY_H */
