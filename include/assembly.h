/*
 * Filename: assembly.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Function prototypes for the assembly routines used throughout PETGEM.
 */

#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#include "grid.h"
#include "io.h"
#include "mms.h"   /* MMSRhsKind, consumed by assembleCsemMMSRHS */
#include "transmitter.h"
#include "fem.h"
#include <petsc.h>

/**
 * @brief Assembles the CSEM right-hand side matrix.
 *
 * Produces B with one column per transmitter, with Dirichlet boundary DOFs
 * eliminated, ready for solveCsemSystem.
 *
 * @param[in]  params   Forward-modeling parameters (order, MPI tasks).
 * @param[in]  sources  Transmitter set (one column of B per source).
 * @param[in]  dm       DMPlex mesh and H(curl) discretization.
 * @param[in]  grid     Finite-element grid descriptor.
 * @param[in]  constFactor  Fused-mode factor iωμ; ignored when Ms != NULL.
 * @param[out] B        Assembled right-hand side matrix.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode assembleCsemRHS(const petgemParams params,
                               const CsemSourceSet sources,
                               const DM dm,
                               const Grid grid,
                               const PetscScalar constFactor,
                               Mat* B);

/**
 * @brief Assembles the volumetric manufactured-source RHS for MMS verification.
 *
 * Replaces the point-dipole RHS (assembleCsemRHS) when running in MMS mode
 * (-mms). Instead of a Dirac source in one cell, it integrates the manufactured
 * forcing f* (include/mms.h) against the Nedelec basis over EVERY cell:
 *
 *     b_j = sum_cells sum_q  w_q * detJ_cell * ( N_j(x_q) . f*(x_q) )
 *
 * where x_q is the physical image of reference quadrature point q, f* is
 * complex, and the same (weights, detJ) measure as computeElementalMatrices is
 * used so b is consistent with the operator A = K - i omega mu Ms. The single
 * manufactured RHS is written to a one-column dense matrix B. Boundary DOFs are
 * eliminated via VEC_IGNORE_NEGATIVE_INDICES (E* already satisfies n x E* = 0).
 * No final iωμ scaling is applied - f* already carries it.
 *
 * The integrand is selected by `kind` (see MMSRhsKind in include/mms.h):
 * MMS_RHS_FORCING integrates f*, MMS_RHS_PROJECTION integrates the exact field
 * E* itself (the L2 moments of the best-approximation baseline, no iωμ and no
 * sigma), and MMS_RHS_FORCING_NO_MASS integrates the incomplete forcing of the
 * negative control.
 *
 * @param[in]  params        Forward-modeling parameters (order, MPI tasks).
 * @param[in]  sources       Transmitter set; only sources.freq (-> omega) is used.
 * @param[in]  dm            DMPlex mesh and H(curl) discretization.
 * @param[in]  grid          Finite-element grid descriptor.
 * @param[in]  conductivity  Per-cell conductivity Vec (diagonal sigma in f*).
 * @param[in]  kind          Which right-hand side to build.
 * @param[out] B             One-column dense RHS matrix (created by this call).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode assembleCsemMMSRHS(const petgemParams params,
                                  const CsemSourceSet sources,
                                  const DM dm,
                                  const Grid grid,
                                  const Vec conductivity,
                                  const MMSRhsKind kind,
                                  Mat* B);

/**
 * @brief Assembles the CSEM left-hand side operator (unified K/Ms or fused).
 *
 * Two output modes, selected by the `Ms` pointer:
 *
 *   Ms != NULL (K/Ms mode, used by the inverse kernel):
 *     *KorA receives the curl-curl stiffness K, *Ms receives the mass-σ
 *     matrix Ms. The frequency-dependent operator A_f = K - i*omega*mu*Ms is
 *     formed by the caller per frequency via MatDuplicate + MatAXPY.
 *     `constFactor` is ignored in this mode.
 *
 *   Ms == NULL (fused mode, used by the forward kernel):
 *     *KorA receives the frequency-dependent operator A = K - constFactor*Ms
 *     directly, formed by per-cell element-level fusion
 *     A_e = K_e - constFactor*M_e, so Ms is never built as a global matrix.
 *     Saves one full complex matrix from the assembly-phase peak memory,
 *     one MatDuplicate, and one global MatAXPY. The caller passes
 *     `constFactor = iωμ`.
 *
 * Matrices:
 *   - K      : curl-curl stiffness, ∫_K (μ⁻¹ curl Ni)·curl Nj.
 *   - Ms     : mass × σ,            ∫_K (ε_r ⊙ Ni)·Nj.
 *   - G_BDDC : high-order discrete gradient (buildDiscreteGradientMatrix)
 *              consumed by PCBDDCSetDiscreteGradient. Each H(curl) DOF's
 *              gradient is resolved against the full P_order H1 closure, so
 *              grad(phi_k) = sum_i G_ik N_i exactly (K·G = 0). Built against
 *              grid.H1dm_P_order; one solver code path serves every order.
 *
 * @param[in]  params       Forward-modeling parameters (order, MPI tasks).
 * @param[in]  dm           DMPlex mesh and H(curl) discretization.
 * @param[in]  grid         Finite-element grid descriptor.
 * @param[in]  conductivity Per-cell conductivity Vec.
 * @param[in]  constFactor  Fused-mode factor iωμ; ignored when Ms != NULL.
 * @param[out] KorA         Stiffness K (K/Ms mode) or fused operator A.
 * @param[out] Ms           Mass-σ matrix in K/Ms mode; pass NULL for fused mode.
 * @param[out] G            High-order discrete gradient; pass NULL to skip.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode assembleCsemKandM(const petgemParams params, const DM dm, const Grid grid,
                                 const Vec conductivity,
                                 const PetscScalar constFactor,
                                 Mat *KorA, Mat *Ms,
                                 Mat *G_BDDC);
/**
 * @brief Refills an existing Ms matrix for the current conductivity field.
 *
 * Used by the inverse kernel inside the L-BFGS loop: K and G_BDDC are
 * σ-independent and built once at setup via assembleCsemKandM, while Ms
 * must be re-computed every iteration when σ changes. fm.csem does not use
 * this routine; it assembles through the fused single-pass call to
 * assembleCsemKandM.
 *
 * Preconditions:
 *   - `Ms` is already allocated with the same sparsity pattern as the K
 *     produced by assembleCsemKandM for the same mesh / order (typically
 *     `MatDuplicate(K, MAT_DO_NOT_COPY_VALUES, &Ms)`).
 *   - `quadrature_3d`, `Me`, `Ke` are caller-owned workspace buffers of the
 *     same shape used inside assembleCsemKandM (numDofInCell² for Me / Ke),
 *     passed in so a single allocation can serve all iterations.
 *
 * On return Ms holds the σ-dependent mass-matrix entries with its sparsity
 * preserved. Ke is computed by computeElementalMatrices but is unused here
 * (kept in the signature so the caller can share the gradient-pass scratch).
 *
 * @param[in]     params         Forward-modeling parameters (order).
 * @param[in]     dm             DMPlex mesh and H(curl) discretization.
 * @param[in]     grid           Finite-element grid descriptor.
 * @param[in]     conductivity   Current per-cell conductivity Vec.
 * @param[in]     quadrature_3d  Caller-owned 3D quadrature workspace.
 * @param[in,out] Me             Scratch elemental mass buffer (numDofInCell²).
 * @param[in,out] Ke             Scratch elemental stiffness buffer (unused output).
 * @param[in,out] Ms             Pre-allocated matrix refilled with mass-σ entries.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode assembleCsemMsRefill(const petgemParams params,
                                    const DM dm, const Grid grid,
                                    const Vec conductivity,
                                    const Quadrature3D *quadrature_3d,
                                    PetscReal **Me, PetscReal **Ke,
                                    Mat Ms);

/* DOFs on cells more conductive than threshold, for PCBDDC primal vertices.
 * Documented in src/assembly.c. */
PetscErrorCode buildHighSigmaPrimalIS(const DM dm, const Vec conductivity,
                                      const PetscReal threshold, IS *primal);

/* Cross-check a primal index set against the assembled operator: the selected
 * rows must carry far larger diagonals than the rest, because the conductivity
 * enters through the mass term. Catches a wrong global numbering, which would
 * otherwise look like the preconditioner under-performing. */
PetscErrorCode verifyPrimalISAgainstOperator(Mat A, IS primal);

#endif
