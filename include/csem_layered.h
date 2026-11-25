/*
 * Filename: csem_layered.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-11-20
 *
 * Description:
 * Electric and magnetic fields of an electric point dipole in a horizontally
 * layered VTI medium, in the PETGEM frame (z up) and time convention e^{-iωt}.
 */

/*
 * Notes:
 * The wavenumber-domain kernel follows Hunziker et al. (2015), Geophysics 80(1),
 * F1-F18, as implemented in empymod (Werthmüller, 2017, Geophysics 82(6),
 * WB9-WB19), with the direct field in the wavenumber domain. The Hankel
 * transforms use the 201-point J0/J1 digital filter of Key (2009), Geophysics
 * 74(2), F9-F20, with the coefficients distributed by libdlf. Displacement
 * currents are neglected. Relative error against the closed-form whole space:
 * up to about 1e-5 for |z - z_s| < 1e-2 r, below 1e-6 elsewhere.
 */

#ifndef CSEM_LAYERED_H
#define CSEM_LAYERED_H

#include <petsc.h>

/** @brief Horizontally layered VTI model, layers ordered top to bottom. */
typedef struct {
  PetscInt   numLayers;  /**< Number of layers. */
  PetscReal *interfaces; /**< Height of the interface below layer k (m, z up), numLayers - 1 values, decreasing. */
  PetscReal *sigmaH;     /**< Horizontal conductivity per layer (S/m). */
  PetscReal *sigmaV;     /**< Vertical conductivity per layer (S/m). */
} LayeredModel;

/** @brief Electric point dipole in a layered model at one frequency. */
typedef struct {
  PetscInt     numLayers;   /**< Number of layers. */
  PetscReal   *depth;       /**< Depth of the top of each layer (m, z down); depth[0] = -inf. */
  PetscScalar *etaH;        /**< Horizontal conductivity per layer (S/m). */
  PetscScalar *etaV;        /**< Vertical conductivity per layer (S/m). */
  PetscScalar  zeta;        /**< i omega mu0 of the e^{+iωt} kernel. */
  PetscReal    position[3]; /**< Dipole position (m, z up). */
  PetscReal    moment[3];   /**< Dipole moment I*L*direction (A m, z up). */
} LayeredDipole;

/**
 * @brief Creates a layered model from per-layer arrays (copied).
 *
 * A point on an interface belongs to the layer above it.
 *
 * @param[in]  numLayers   Number of layers (>= 1).
 * @param[in]  interfaces  Height of the interface below layer k (m, z up), numLayers - 1 values, strictly decreasing.
 * @param[in]  sigmaH      Horizontal conductivity per layer (S/m, > 0).
 * @param[in]  sigmaV      Vertical conductivity per layer (S/m, > 0).
 * @param[out] model       Layered model; release it with destroyLayeredModel().
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode createLayeredModel(PetscInt numLayers, const PetscReal interfaces[], const PetscReal sigmaH[], const PetscReal sigmaV[],
                                  LayeredModel *model);

/**
 * @brief Frees a model created by createLayeredModel().
 *
 * @param[in,out] model  Layered model.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode destroyLayeredModel(LayeredModel *model);

/**
 * @brief Returns the layer that contains height z.
 *
 * @param[in] model  Layered model.
 * @param[in] z      Height (m, z up).
 *
 * @return Layer k such that interfaces[k] <= z < interfaces[k - 1]; layer 0 has no upper bound and
 *         layer numLayers - 1 no lower bound.
 */
PetscInt findLayer(const LayeredModel *model, PetscReal z);

/**
 * @brief Returns the layer that contains height z, taking the side of zSide on an interface.
 *
 * z lies on an interface when |z - z_i| <= 1e-8 max(1, |z_i|).
 *
 * @param[in] model  Layered model.
 * @param[in] z      Height (m, z up).
 * @param[in] zSide  Height on the side of the interface (m, z up), used only when z lies on an interface.
 *
 * @return findLayer(model, z), or the layer on the side of zSide if z lies on an interface.
 */
PetscInt findLayerOnSide(const LayeredModel *model, PetscReal z, PetscReal zSide);

/**
 * @brief Sets up an electric point dipole in a layered model.
 *
 * @param[in]  model     Layered model.
 * @param[in]  freq      Frequency (Hz).
 * @param[in]  position  Dipole position (m, z up).
 * @param[in]  moment    Dipole moment I*L*direction (A m, z up).
 * @param[out] dipole    Dipole; release it with destroyLayeredDipole().
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode createLayeredDipole(const LayeredModel *model, PetscReal freq, const PetscReal position[3], const PetscReal moment[3],
                                   LayeredDipole *dipole);

/**
 * @brief Evaluates E (and optionally H) of the dipole at points.
 *
 * A point at horizontal offset r < r_c = max(0.04 |z - z_s|, 1e-3 m) from the
 * dipole is evaluated by Lagrange interpolation from the offsets +-r_c, +-2r_c,
 * +-3r_c and +-4r_c along its azimuth (r = 0: the x axis). Points that coincide
 * with the dipole are an error.
 *
 * @param[in]  dipole     Dipole (createLayeredDipole).
 * @param[in]  numPoints  Number of points.
 * @param[in]  points     Point coordinates, points[q * 3 + d] (m, z up).
 * @param[in]  layers     Layer of each point, in the layer or on its boundary (findLayerOnSide); NULL: findLayer.
 * @param[out] E          Electric field, E[q * 3 + d] (V/m).
 * @param[out] H          Magnetic field, H[q * 3 + d] (A/m); NULL to skip.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode evaluateLayeredDipole(const LayeredDipole *dipole, PetscInt numPoints, const PetscReal points[], const PetscInt layers[],
                                     PetscScalar E[], PetscScalar H[]);

/**
 * @brief Frees a dipole created by createLayeredDipole().
 *
 * @param[in,out] dipole  Dipole.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode destroyLayeredDipole(LayeredDipole *dipole);

#endif /* CSEM_LAYERED_H */
