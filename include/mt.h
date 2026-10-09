/*
 * Filename: mt.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * Magnetotelluric (MT) layer: run options, classification of the box faces
 * Gamma_1..Gamma_6, the 1D conductivity profile of the lateral faces and the
 * 1D boundary field H(z) of Castillo-Reyes et al. (2022), Eq. (11).
 */

#ifndef MT_H
#define MT_H

#include "constants.h"
#include "grid.h"
#include <petsc.h>

/**
 * @brief 1D equation used for the boundary field H(z).
 */
typedef enum {
  MT_1D_EQUATION_PAPER, /**< H'' + iωμσH = 0 (Eq. 11). */
  MT_1D_EQUATION_H      /**< (ρH')' + iωμH = 0, ρ = 1/σ. */
} Mt1DEquation;

/**
 * @brief MT run options.
 */
typedef struct {
  PetscInt     refine1D;   /**< 1D element size = boundary 3D edge length / refine1D. */
  Mt1DEquation equation1D; /**< 1D equation for H(z). */
} MtParams;

/**
 * @brief Faces of the box domain, numbered as Gamma_1..Gamma_6 (Fig. 1).
 */
typedef enum {
  MT_FACE_TOP    = 1, /**< Gamma_1: z = z_max. */
  MT_FACE_YMIN   = 2, /**< Gamma_2: y = y_min. */
  MT_FACE_XMAX   = 3, /**< Gamma_3: x = x_max. */
  MT_FACE_YMAX   = 4, /**< Gamma_4: y = y_max. */
  MT_FACE_XMIN   = 5, /**< Gamma_5: x = x_min. */
  MT_FACE_BOTTOM = 6  /**< Gamma_6: z = z_min. */
} MtBoxFace;

/**
 * @brief Layered conductivity of the lateral faces Gamma_2..Gamma_5.
 *
 * Layer i spans [z[i], z[i+1]], bottom to top.
 */
typedef struct {
  PetscInt   numLayers; /**< Number of layers. */
  PetscReal *z;         /**< Interfaces, ascending (numLayers + 1). */
  PetscReal *sigma;     /**< Layer conductivity (numLayers). */
  PetscReal *h;         /**< Smallest 3D edge length on the layer's faces (numLayers). */
} Mt1DProfile;

/**
 * @brief Nodal 1D field H(z) on a piecewise-linear 1D mesh.
 */
typedef struct {
  PetscInt     numNodes; /**< Number of nodes. */
  PetscReal   *z;        /**< Node coordinates, ascending. */
  PetscScalar *H;        /**< Nodal values. */
} Mt1DField;

/**
 * @brief Reads the MT options -mt_1d_refine and -mt_1d_equation.
 *
 * @param[out] mt  MT options.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode readMtParams(MtParams *mt);

/**
 * @brief Assigns each boundary face to a box face Gamma_1..Gamma_6.
 *
 * Fails if a face normal is not aligned with a coordinate axis or the face
 * does not lie on the domain bounding box.
 *
 * @param[in]  dm     DMPlex mesh.
 * @param[in]  faces  Local boundary faces (getBoundaryFaces).
 * @param[out] tags   Box face of each entry of faces (caller frees).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode classifyMtBoxFaces(const DM dm, IS faces, MtBoxFace **tags);

/**
 * @brief Builds the layered conductivity of the lateral faces Gamma_2..Gamma_5.
 *
 * Gathers (z range, σ, edge length) of every lateral face from all ranks and
 * merges them into layers. Fails if σx != σy on a lateral cell or if two
 * lateral faces disagree on σ at the same height.
 *
 * @param[in]  dm            DMPlex mesh.
 * @param[in]  conductivity  Per-cell conductivity Vec.
 * @param[in]  faces         Local boundary faces (getBoundaryFaces).
 * @param[in]  tags          Box face of each entry of faces (classifyMtBoxFaces).
 * @param[out] profile       Layered conductivity, identical on every rank.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode buildMt1DProfile(const DM dm, const Vec conductivity, IS faces, const MtBoxFace *tags, Mt1DProfile *profile);

/**
 * @brief Solves the 1D boundary problem for H(z).
 *
 * Linear finite elements with H(z_max) = 1 and H(z_min) = 0; nodes at every
 * interface and element size h/refine1D in each layer. Solved redundantly on
 * every rank.
 *
 * @param[in]  profile  Layered conductivity.
 * @param[in]  mt       MT options (refine1D, equation1D).
 * @param[in]  omega    Angular frequency.
 * @param[out] field    Nodal H(z).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode solveMt1D(const Mt1DProfile *profile, const MtParams *mt, const PetscReal omega, Mt1DField *field);

/**
 * @brief Evaluates H(z) by linear interpolation; z is clamped to [z_min, z_max].
 *
 * @param[in]  field  Nodal H(z).
 * @param[in]  z      Height.
 * @param[out] H      H(z).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode evalMt1DField(const Mt1DField *field, const PetscReal z, PetscScalar *H);

/**
 * @brief Frees an Mt1DProfile.
 *
 * @param[in,out] profile  Profile to free.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode destroyMt1DProfile(Mt1DProfile *profile);

/**
 * @brief Frees an Mt1DField.
 *
 * @param[in,out] field  Field to free.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode destroyMt1DField(Mt1DField *field);

#endif /* MT_H */
