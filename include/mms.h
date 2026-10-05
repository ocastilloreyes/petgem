/*
 * Filename: mms.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-08-01
 *
 * Description:
 * Manufactured solution for the Method-of-Manufactured-Solutions (MMS)
 * order-of-accuracy verification of the PETGEM high-order Nedelec
 * discretization. This header is the C-side single source of truth for the
 * exact field, its curl and the manufactured forcing; it is consumed by the
 * MMS driver (runMMSVerification in src/mms.c) and the volumetric-RHS assembly
 * (assembleCsemMMSRHS in src/assembly.c).
 *
 * It mirrors, symbol for symbol, data/test1/mms_reference.py.
 * data/test1/verify_c_reference.sh compiles data/test1/mms_c_check.c against
 * THIS header and diffs the result against the Python reference.
 *
 * PROBLEM VERIFIED. The operator fm.csem assembles
 * (src/fem.c::femElementalMatrix, src/assembly.c::assembleCsemKandM) is:
 *
 *     Ke = INT curl N_j . curl N_k          (mu_r = 1)
 *     Me = INT (sigma . N_j) . N_k          (sigma diagonal, per cell)
 *     A  = Ke - i omega mu0 Me
 *
 *   =>  curl curl E - i omega mu0 sigma E = f   in Omega = [0,L]^3
 *                    n x E                = 0   on dOmega
 *
 * src/grid.c constrains every boundary DOF out of the PetscSection, so
 * n x E = 0 is the essential condition imposed, and E* satisfies it exactly.
 *
 * MANUFACTURED FIELD (Hodge split: solenoidal + gradient)
 *
 *     k    = m pi / L                        (integer m -> zero tangential trace)
 *     S    = ( sin(ky) sin(kz),              div S = 0, curlcurl S = 2 k^2 S
 *              sin(kz) sin(kx),
 *              sin(kx) sin(ky) )
 *     G    = grad( sin(kx) sin(ky) sin(kz) ) / k          curl G = 0
 *          = ( cos(kx) sin(ky) sin(kz),
 *              sin(kx) cos(ky) sin(kz),
 *              sin(kx) sin(ky) cos(kz) )
 *     E*   = a S + b G
 *
 * curl curl annihilates G, so the gradient component is controlled by the mass
 * term alone. div E* = -3 b k Phi, so the field is not divergence-free.
 *
 * MANUFACTURED FORCING (exact)
 *
 *     curl curl E* = 2 k^2 a S
 *     f*_d = a (2 k^2 - i omega mu0 sigma_d) S_d
 *            - i omega mu0 sigma_d b G_d,        d = 0, 1, 2
 *
 * DESIGN POINT. L = 1000 m, f = 10 Hz, sigma = (0.5, 1.0, 2.0) S/m, m = 2.
 * With sigma_ref = 1 S/m the skin depth is delta = L/(m pi) = 159.155 m,
 *
 *     rho = 2 k^2 / (omega mu0 sigma_ref) = 1
 *     nu  = omega mu0 sigma_ref L^2       = 8 pi^2 = 79.0
 *
 * MMS_L is duplicated in data/test1/mesh.geo and data/test1/mms_reference.py;
 * runMMSVerification checks it against the mesh bounding box at start-up and
 * aborts on a mismatch.
 *
 * See data/test1/README.md for the design, the convergence rates and the
 * verification protocol.
 */

#ifndef MMS_H
#define MMS_H

#include "constants.h" /* MU, NUM_DIMENSIONS */
#include "grid.h"      /* Grid, and (via io.h) petgemParams, CsemSourceSet */
#include <petsc.h>

/* --- design point (must match data/test1/mesh.geo and mms_reference.py) --- */
#define MMS_L 1000.0 /**< Edge length of the cubic domain [0,L]^3 [m].     */
#define MMS_MODE 2   /**< Mode number m; k = m pi / L.                     */
#define MMS_K (MMS_MODE * PETSC_PI / MMS_L) /**< Manufactured wavenumber.  */

/* Amplitudes of the two parts, as (real, imaginary) pairs.
 *
 * PetscCMPLX is used rather than (re + im*PETSC_i): PETSC_i is a global that
 * PETSc assigns inside PetscInitialize, so an amplitude spelt with it evaluates
 * to zero in any context that runs earlier. The real/imaginary parts are kept
 * as separate macros so |a|^2 and |b|^2 below are exact arithmetic.
 */
#define MMS_A_RE 1.0 /**< Re(a), solenoidal amplitude. */
#define MMS_A_IM 0.0 /**< Im(a), solenoidal amplitude. */
#define MMS_B_RE 0.0 /**< Re(b), gradient amplitude.   */
#define MMS_B_IM 1.0 /**< Im(b), gradient amplitude.   */

/** @brief Amplitude of the solenoidal part S (complex). */
#define MMS_A PetscCMPLX(MMS_A_RE, MMS_A_IM)
/** @brief Amplitude of the gradient part G (complex). */
#define MMS_B PetscCMPLX(MMS_B_RE, MMS_B_IM)

/** @brief |a|^2, exact. */
#define MMS_A_ABS2 (MMS_A_RE * MMS_A_RE + MMS_A_IM * MMS_A_IM)
/** @brief |b|^2, exact. */
#define MMS_B_ABS2 (MMS_B_RE * MMS_B_RE + MMS_B_IM * MMS_B_IM)

/**
 * @brief Right-hand side the MMS assembly builds.
 *
 * MMS_RHS_FORCING          INT f* . N          : the Galerkin problem.
 * MMS_RHS_PROJECTION       INT (sigma E*) . N  : sigma-weighted moments of the
 *                          exact field. Paired with the operator Ms this gives
 *                          the sigma-weighted L2 projection of E*, the
 *                          best-approximation baseline. The sigma weight
 *                          matches Ms.
 * MMS_RHS_FORCING_NO_MASS  : negative control. MMS_RHS_FORCING with the
 *                          -i omega mu0 sigma_d terms dropped, so E* is not the
 *                          solution and the error plateaus at O(1).
 */
typedef enum {
  MMS_RHS_FORCING = 0,
  MMS_RHS_PROJECTION,
  MMS_RHS_FORCING_NO_MASS
} MMSRhsKind;

/**
 * @brief Evaluates the trigonometric factors shared by S, G and curl S.
 *
 * @param[in]  X  Physical coordinates (x, y, z).
 * @param[out] s  sin(k x), sin(k y), sin(k z).
 * @param[out] c  cos(k x), cos(k y), cos(k z).
 */
static inline void mmsTrig(const PetscReal X[NUM_DIMENSIONS], PetscReal s[NUM_DIMENSIONS],
                           PetscReal c[NUM_DIMENSIONS]) {
  const PetscReal k = MMS_K;
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    s[d] = PetscSinReal(k * X[d]);
    c[d] = PetscCosReal(k * X[d]);
  }
}

/**
 * @brief Solenoidal part S(X): div S = 0 and curlcurl S = 2 k^2 S.
 *
 * @param[in]  X  Physical coordinates (x, y, z).
 * @param[out] S  S(X), real-valued.
 */
static inline void mmsPartS(const PetscReal X[NUM_DIMENSIONS], PetscReal S[NUM_DIMENSIONS]) {
  PetscReal s[NUM_DIMENSIONS], c[NUM_DIMENSIONS];
  mmsTrig(X, s, c);
  S[0] = s[1] * s[2];
  S[1] = s[2] * s[0];
  S[2] = s[0] * s[1];
}

/**
 * @brief Gradient part G(X) = grad(sin kx sin ky sin kz)/k: curl G = 0.
 *
 * @param[in]  X  Physical coordinates (x, y, z).
 * @param[out] G  G(X), real-valued.
 */
static inline void mmsPartG(const PetscReal X[NUM_DIMENSIONS], PetscReal G[NUM_DIMENSIONS]) {
  PetscReal s[NUM_DIMENSIONS], c[NUM_DIMENSIONS];
  mmsTrig(X, s, c);
  G[0] = c[0] * s[1] * s[2];
  G[1] = s[0] * c[1] * s[2];
  G[2] = s[0] * s[1] * c[2];
}

/**
 * @brief Exact manufactured field E*(X) = a S(X) + b G(X).
 *
 * @param[in]  X  Physical coordinates (x, y, z).
 * @param[out] E  E*(X), complex-valued.
 */
static inline void mmsExactE(const PetscReal X[NUM_DIMENSIONS], PetscScalar E[NUM_DIMENSIONS]) {
  PetscReal S[NUM_DIMENSIONS], G[NUM_DIMENSIONS];
  mmsPartS(X, S);
  mmsPartG(X, G);
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    E[d] = MMS_A * (PetscScalar)S[d] + MMS_B * (PetscScalar)G[d];
  }
}

/**
 * @brief Exact curl of the manufactured field, curl E* = a curl S (curl G = 0).
 *
 *   (curl S)_x = k sin(k x) [cos(k y) - cos(k z)],  cyclically.
 *
 * @param[in]  X  Physical coordinates (x, y, z).
 * @param[out] C  curl E*(X), complex-valued.
 */
static inline void mmsExactCurlE(const PetscReal X[NUM_DIMENSIONS], PetscScalar C[NUM_DIMENSIONS]) {
  const PetscReal k = MMS_K;
  PetscReal s[NUM_DIMENSIONS], c[NUM_DIMENSIONS];
  mmsTrig(X, s, c);
  C[0] = MMS_A * (PetscScalar)(k * s[0] * (c[1] - c[2]));
  C[1] = MMS_A * (PetscScalar)(k * s[1] * (c[2] - c[0]));
  C[2] = MMS_A * (PetscScalar)(k * s[2] * (c[0] - c[1]));
}

/**
 * @brief Manufactured forcing f*(X) = curlcurl E* - i omega mu0 sigma E*.
 *
 *   f*_d = a (2 k^2 - i omega mu0 sigma_d) S_d - i omega mu0 sigma_d b G_d
 *
 * The per-component conductivity keeps f* consistent with the diagonal mass
 * matrix Me for an anisotropic sigma.
 *
 * With dropMass = PETSC_TRUE the mass terms are omitted, leaving the incomplete
 * forcing 2 k^2 a S of the negative control (MMS_RHS_FORCING_NO_MASS).
 *
 * @param[in]  X         Physical coordinates (x, y, z).
 * @param[in]  omega     Angular frequency 2 pi f.
 * @param[in]  sigma     Cell diagonal conductivity (sigma_x, sigma_y, sigma_z).
 * @param[in]  dropMass  Omit the -i omega mu0 sigma_d terms (negative control).
 * @param[out] F         f*(X), complex-valued.
 */
static inline void mmsForcingF(const PetscReal X[NUM_DIMENSIONS], PetscReal omega,
                               const PetscReal sigma[NUM_DIMENSIONS], PetscBool dropMass,
                               PetscScalar F[NUM_DIMENSIONS]) {
  const PetscReal k = MMS_K;
  const PetscReal twok2 = 2.0 * k * k;
  PetscReal S[NUM_DIMENSIONS], G[NUM_DIMENSIONS];
  mmsPartS(X, S);
  mmsPartG(X, G);
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    const PetscScalar imu = dropMass ? (PetscScalar)0.0 : PetscCMPLX(0.0, omega * MU * sigma[d]);
    F[d] = MMS_A * ((PetscScalar)twok2 - imu) * (PetscScalar)S[d] - imu * MMS_B * (PetscScalar)G[d];
  }
}

/**
 * @brief Exact ||E*||_L2 on [0,L]^3, the L2 relative-error denominator.
 *
 *   ||E*||^2 = |a|^2 (3 L^3/4) + |b|^2 (3 L^3/8)
 *
 * S and G are L2-orthogonal componentwise, so no cross term appears.
 *
 * @return ||E*||_L2.
 */
static inline PetscReal mmsRefNormL2(void) {
  const PetscReal L3 = MMS_L * MMS_L * MMS_L;
  return PetscSqrtReal(MMS_A_ABS2 * 0.75 * L3 + MMS_B_ABS2 * 0.375 * L3);
}

/**
 * @brief Exact energy norm |||E*||| on [0,L]^3, the energy denominator.
 *
 *   |||v|||^2 = ||curl v||^2_L2 + omega mu0 INT (sigma v) . conj(v)
 *   |||E*|||^2 = |a|^2 (3 k^2 L^3/2)
 *                + omega mu0 (sigma_x+sigma_y+sigma_z)
 *                  (|a|^2 L^3/4 + |b|^2 L^3/8)
 *
 * This is the norm the operator induces: |a(v, conj(v))| lies between
 * |||v|||^2 and sqrt(2) |||v|||^2, and Cea quasi-optimality holds in it with
 * the constant sqrt(2), independent of h, p, omega and sigma.
 *
 * @param[in] omega  Angular frequency 2 pi f.
 * @param[in] sigma  Diagonal conductivity (uniform over the domain).
 *
 * @return |||E*|||.
 */
static inline PetscReal mmsRefNormEnergy(PetscReal omega, const PetscReal sigma[NUM_DIMENSIONS]) {
  const PetscReal L3 = MMS_L * MMS_L * MMS_L;
  const PetscReal k = MMS_K;
  const PetscReal sigmaSum = sigma[0] + sigma[1] + sigma[2];
  const PetscReal curlPart = MMS_A_ABS2 * 1.5 * k * k * L3;
  const PetscReal massPart =
      omega * MU * sigmaSum * (MMS_A_ABS2 * 0.25 * L3 + MMS_B_ABS2 * 0.125 * L3);
  return PetscSqrtReal(curlPart + massPart);
}

/**
 * @brief Runs the complete MMS verification for one (order, mesh).
 *
 * Single entry point dispatched from runForward on -mms. Performs the Galerkin
 * solve and the L2-projection best-approximation and writes both error norms
 * plus the solve residual to an HDF5 file (one per run); -mms_diagnostics adds
 * the over-integrated norms and -mms_drop_mass selects the negative control.
 * See src/mms.c and data/test1/README.md.
 *
 * @param[in] params        Forward-modeling parameters (order, output dir).
 * @param[in] dm            DMPlex mesh and H(curl) discretization.
 * @param[in] grid          Finite-element grid descriptor.
 * @param[in] conductivity  Per-cell conductivity Vec.
 * @param[in] sources       Transmitter set; only sources.freq (-> omega) is used.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode runMMSVerification(const petgemParams params, const DM dm, const Grid grid,
                                  const Vec conductivity, const CsemSourceSet sources);

#endif /* MMS_H */
