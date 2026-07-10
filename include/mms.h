/*
 * Filename: mms.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-07-07
 *
 * Description:
 * Manufactured solution for the Method-of-Manufactured-Solutions (MMS)
 * order-of-accuracy verification of the PETGEM high-order Nedelec
 * discretization. This header is the C-side single source of truth for the
 * exact field, its curl, and the manufactured forcing; it is consumed by the
 * MMS driver (runMMSVerification in src/mms.c) and the volumetric-RHS assembly
 * (assembleCsemMMSRHS in src/assembly.c).
 *
 * It mirrors, symbol for symbol, tests/mms/mms_reference.py.
 *
 * Domain: unit cube [0,1]^3, single material, homogeneous Dirichlet n x E* = 0.
 *
 *   E*(x,y,z) = ( sin(pi y) sin(pi z),
 *                 sin(pi z) sin(pi x),
 *                 sin(pi x) sin(pi y) )
 *
 *   div E* = 0,   curlcurl E* = 2 pi^2 E*
 *
 * Substituted into the total-field curl-curl operator with a diagonal
 * conductivity tensor sigma = (sigma_x, sigma_y, sigma_z), the forcing is a
 * per-component multiple of E* (matching A = K - i omega mu Ms):
 *
 *   f*_d(x) = ( 2 pi^2 - i omega mu sigma_d ) E*_d(x),   d = 0,1,2
 *
 * n x E* = 0 on d[0,1]^3, so no nonzero-BC code path is needed: E* is
 * consistent with the homogeneous Dirichlet condition fm.csem already imposes
 * (DMPlexMarkBoundaryFaces).
 */

#ifndef MMS_H
#define MMS_H

#include "constants.h"   /* MU, NUM_DIMENSIONS */
#include "grid.h"        /* Grid, and (via io.h) fmParams, CsemSourceSet */
#include <petsc.h>

/* Exact norms of E* on [0,1]^3, used as relative-error denominators.
 *   ||E*||_L2     = sqrt(3)/2
 *   ||E*||_H(curl)= sqrt(3/4 + 3 pi^2/2)
 * (Closed-form; verified in mms_reference.py::_selftest with sympy.) */
#define MMS_E_L2_NORM     0.86602540378443865   /* sqrt(3)/2          */
#define MMS_E_HCURL_NORM  3.94390752951103071   /* sqrt(3/4 + 3pi^2/2) */

/**
 * @brief Exact manufactured field E*(X) at a physical point.
 *
 * @param[in]  X  Physical coordinates (x, y, z).
 * @param[out] E  E*(X), stored as PetscScalar (E* is real-valued).
 */
static inline void mmsExactE(const PetscReal X[NUM_DIMENSIONS], PetscScalar E[NUM_DIMENSIONS]) {
  const PetscReal pi = PETSC_PI;
  E[0] = (PetscScalar)(PetscSinReal(pi * X[1]) * PetscSinReal(pi * X[2]));
  E[1] = (PetscScalar)(PetscSinReal(pi * X[2]) * PetscSinReal(pi * X[0]));
  E[2] = (PetscScalar)(PetscSinReal(pi * X[0]) * PetscSinReal(pi * X[1]));
}

/**
 * @brief Exact curl of the manufactured field, curl E*(X), at a physical point.
 *
 *   (curl E*)_x = pi sin(pi x) [cos(pi y) - cos(pi z)]   (and cyclic).
 *
 * @param[in]  X  Physical coordinates (x, y, z).
 * @param[out] C  curl E*(X), stored as PetscScalar (real-valued).
 */
static inline void mmsExactCurlE(const PetscReal X[NUM_DIMENSIONS], PetscScalar C[NUM_DIMENSIONS]) {
  const PetscReal pi = PETSC_PI;
  const PetscReal sx = PetscSinReal(pi * X[0]), sy = PetscSinReal(pi * X[1]), sz = PetscSinReal(pi * X[2]);
  const PetscReal cx = PetscCosReal(pi * X[0]), cy = PetscCosReal(pi * X[1]), cz = PetscCosReal(pi * X[2]);
  C[0] = (PetscScalar)(pi * sx * (cy - cz));
  C[1] = (PetscScalar)(pi * sy * (cz - cx));
  C[2] = (PetscScalar)(pi * sz * (cx - cy));
}

/**
 * @brief Manufactured forcing f*(X) = (2 pi^2 - i omega mu sigma_d) E*_d(X).
 *
 * This is the volumetric current density the RHS assembly integrates against
 * the Nedelec basis. The per-component conductivity keeps f* consistent with
 * the diagonal mass matrix Ms even for an anisotropic sigma.
 *
 * @param[in]  X      Physical coordinates (x, y, z).
 * @param[in]  omega  Angular frequency 2 pi f.
 * @param[in]  sigma  Diagonal conductivity (sigma_x, sigma_y, sigma_z).
 * @param[out] F      f*(X), complex-valued.
 */
static inline void mmsForcingF(const PetscReal X[NUM_DIMENSIONS], PetscReal omega,
                               const PetscReal sigma[NUM_DIMENSIONS], PetscScalar F[NUM_DIMENSIONS]) {
  const PetscReal pi = PETSC_PI;
  PetscScalar E[NUM_DIMENSIONS];
  mmsExactE(X, E);
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    const PetscScalar s = (PetscScalar)(2.0 * pi * pi) - PETSC_i * (omega * MU * sigma[d]);
    F[d] = s * E[d];
  }
}

/**
 * @brief Runs the complete MMS verification for one (order, mesh).
 *
 * Single entry point dispatched from runForward on -mms. Performs the Galerkin
 * solve and the L2-projection best-approximation and writes both error norms
 * and the solve residual to an HDF5 file (one per run); -mms_diagnostics and
 * -mms_conditioning add the over-integration and conditioning metrics. See
 * src/mms.c.
 *
 * @param[in] params        Forward-modeling parameters (order, output dir).
 * @param[in] dm            DMPlex mesh and H(curl) discretization.
 * @param[in] grid          Finite-element grid descriptor.
 * @param[in] conductivity  Per-cell conductivity Vec.
 * @param[in] sources       Transmitter set; only sources.freq (-> omega) is used.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode runMMSVerification(const fmParams params, const DM dm, const Grid grid,
                                  const Vec conductivity, const CsemSourceSet sources);

#endif /* MMS_H */
