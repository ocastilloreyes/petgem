/*
 * Filename: mt.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * Magnetotelluric (MT) layer: run options, classification of the box faces
 * Gamma_1..Gamma_6, the 1D conductivity profile of the lateral faces, the
 * 1D boundary field H(z) of Castillo-Reyes et al. (2022), Eq. (11), the
 * boundary right-hand side, Eq. (10), and the responses at the receivers
 * (impedance, apparent resistivity, phase and tipper, Appendix A).
 */

#ifndef MT_H
#define MT_H

#include "constants.h"
#include "grid.h"
#include "io.h"
#include <petsc.h>

/**
 * @brief 1D equation used for the boundary field H(z).
 */
typedef enum {
  MT_1D_EQUATION_PAPER, /**< H'' + iωμσH = 0 (Eq. 11). */
  MT_1D_EQUATION_H      /**< (ρH')' + iωμH = 0, ρ = 1/σ. */
} Mt1DEquation;

#define MT_NUM_POLARIZATIONS 2 /**< x- and y-polarization. */

/**
 * @brief MT run options.
 */
typedef struct {
  PetscReal    frequency;  /**< Frequency (Hz), from the bundle's /mt/freq. */
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
 * @brief Reads the MT frequency from the bundle's /mt/freq dataset.
 *
 * @param[in]     params  Parameters (input file).
 * @param[in,out] mt      MT options; frequency is set.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode loadMtSettings(const petgemParams *params, MtParams *mt);

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
 * @brief Assembles the MT right-hand side, Eq. (10), for both polarizations.
 *
 *   b_j = -iωμ ∮_Γ N_j · (n × Ĥ) dΓ
 *
 * with Ĥ = (0, H(z), 0) in column 0 (x-polarization) and Ĥ = (H(z), 0, 0) in
 * column 1 (y-polarization). Gamma_6 is skipped (H(z_min) = 0). Requires a
 * grid set up with PETGEM_BC_NATURAL.
 *
 * @param[in]  params  Parameters (order).
 * @param[in]  omega   Angular frequency.
 * @param[in]  dm      DMPlex mesh and H(curl) discretization.
 * @param[in]  grid    Finite-element grid descriptor.
 * @param[in]  faces   Local boundary faces (getBoundaryFaces).
 * @param[in]  tags    Box face of each entry of faces (classifyMtBoxFaces).
 * @param[in]  field   Nodal H(z) (solveMt1D).
 * @param[out] B       Right-hand side matrix, one column per polarization.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode assembleMtBoundaryRHS(const petgemParams params, const PetscReal omega, const DM dm, const Grid grid,
                                     IS faces, const MtBoxFace *tags, const Mt1DField *field, Mat *B);

/**
 * @brief Prints the "MT" section of the run report.
 *
 * @param[in] comm      Communicator.
 * @param[in] mt        MT options.
 * @param[in] faces     Local boundary faces.
 * @param[in] profile   Layered conductivity of the lateral faces.
 * @param[in] field     Nodal H(z).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode logMtSetup(MPI_Comm comm, const MtParams *mt, IS faces, const Mt1DProfile *profile, const Mt1DField *field);

/**
 * @brief Computes the MT responses at the receivers and writes them to HDF5.
 *
 * E = Q_E x and H = Q_H x / (iωμ) for each polarization (columns of X). Per
 * receiver, Z = [E1 E2][H1 H2]^-1 and T = [Hz1 Hz2][H1 H2]^-1 (Eq. A.3),
 * rho_ij = |Z_ij|^2 / (ωμ) (Eq. A.4) and phi_ij = atan2(Im Z_ij, Re Z_ij) in
 * degrees (Eq. A.5). Output file `{output_dir}/{output_filename}.h5`:
 *
 *   /                          provenance attrs, frequency, num_receivers
 *   /polarizations/{x,y}/fields  Ex, Ey, Ez, Hx, Hy, Hz
 *   /impedance                 xx, xy, yx, yy
 *   /apparent_resistivity      xx, xy, yx, yy
 *   /phase                     xx, xy, yx, yy
 *   /tipper                    x, y
 *
 * @param[in] params     Parameters (order, output paths).
 * @param[in] mt         MT options (frequency).
 * @param[in] dm         DMPlex mesh and H(curl) discretization.
 * @param[in] grid       Finite-element grid descriptor.
 * @param[in] receivers  Serial Vec of 3·N_recv receiver coordinates.
 * @param[in] X          Solution matrix, one column per polarization.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode computeMtResponses(const petgemParams params, const MtParams *mt, const DM dm, const Grid grid,
                                  Vec receivers, const Mat X);

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
