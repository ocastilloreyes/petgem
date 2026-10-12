/*
 * Filename: assembly.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Global assembly of the H(curl) Maxwell operator, its discrete gradient
 * and mass-matrix refill, and the right-hand-side containers.
 */

#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#include "grid.h"
#include "fem.h"
#include <petsc.h>

/**
 * @brief Creates a global right-hand-side Vec that accepts closure insertion
 *        (DM local-to-global mapping, negative indices ignored).
 *
 * @param[in]  dm  DMPlex mesh carrying the H(curl) section.
 * @param[out] b   Right-hand-side Vec.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode createRhsVector(const DM dm, Vec *b);

/**
 * @brief Creates a dense matrix with one column per right-hand side, laid out like @p b.
 *
 * @param[in]  b       Template right-hand-side Vec.
 * @param[in]  numRhs  Number of columns.
 * @param[out] B       Dense right-hand-side matrix.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode createRhsMatrix(const Vec b, const PetscInt numRhs, Mat *B);

/**
 * @brief Assembles the Maxwell operator A = K - iωμ·Ms (unified K/Ms or fused).
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
 *   - G      : high-order discrete gradient (buildDiscreteGradientMatrix)
 *              consumed by PCBDDCSetDiscreteGradient. Each H(curl) DOF's
 *              gradient is resolved against the full P_order H1 closure, so
 *              grad(phi_k) = sum_i G_ik N_i exactly (K·G = 0). Built against
 *              grid.H1dm_P_order; one solver code path serves every order.
 *
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
PetscErrorCode assembleMaxwellOperator(const DM dm, const Grid *grid, const Vec conductivity, const PetscScalar constFactor, Mat *KorA,
                                       Mat *Ms, Mat *G);
/**
 * @brief Refills an existing Ms matrix for the current conductivity field.
 *
 * Used by the inverse kernel inside the L-BFGS loop: K and G are
 * σ-independent and built once at setup via assembleMaxwellOperator, while Ms
 * must be re-computed every iteration when σ changes. fm.csem does not use
 * this routine; it assembles through the fused single-pass call to
 * assembleMaxwellOperator.
 *
 * Preconditions:
 *   - `Ms` is already allocated with the same sparsity pattern as the K
 *     produced by assembleMaxwellOperator for the same mesh / order (typically
 *     `MatDuplicate(K, MAT_DO_NOT_COPY_VALUES, &Ms)`).
 *   - `quadrature3D`, `Me`, `Ke` are caller-owned workspace buffers of the
 *     same shape used inside assembleMaxwellOperator (numDofInCell² for Me / Ke),
 *     passed in so a single allocation can serve all iterations.
 *
 * On return Ms holds the σ-dependent mass-matrix entries with its sparsity
 * preserved. Ke is computed by computeElementalMatrices but is unused here
 * (kept in the signature so the caller can share the gradient-pass scratch).
 *
 * @param[in]     dm             DMPlex mesh and H(curl) discretization.
 * @param[in]     grid           Finite-element grid descriptor.
 * @param[in]     conductivity   Current per-cell conductivity Vec.
 * @param[in]     quadrature3D  Caller-owned 3D quadrature workspace.
 * @param[in,out] Me             Scratch elemental mass buffer (numDofInCell²).
 * @param[in,out] Ke             Scratch elemental stiffness buffer (unused output).
 * @param[in,out] Ms             Pre-allocated matrix refilled with mass-σ entries.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode refillMassMatrix(const DM dm, const Grid *grid, const Vec conductivity, const Quadrature3D *quadrature3D, PetscReal **Me,
                                PetscReal **Ke, Mat Ms);

#endif
