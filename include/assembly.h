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
#include "inputs.h"
#include "transmitter.h"
#include <petsc.h>

/**
 * @brief Assembles the CSEM right-hand side matrix.
 *
 * Produces B with one column per transmitter, with Dirichlet boundary DOFs
 * eliminated, ready for solveCsemSystem.
 *
 * @param[in]  params   Forward-modeling parameters (nord, MPI tasks).
 * @param[in]  sources  Transmitter set (one column of B per source).
 * @param[in]  dm       DMPlex mesh and H(curl) discretization.
 * @param[in]  grid     Finite-element grid descriptor.
 * @param[in]  constFactor  Fused-mode factor iωμ; ignored when Ms != NULL.
 * @param[out] B        Assembled right-hand side matrix.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode assembleCsemRHS(const fmParams params, 
                               const CsemSourceSet sources, 
                               const DM dm, 
                               const Grid grid,
                               const PetscScalar constFactor, 
                               Mat* B);

/**
 * @brief Assembles the CSEM left-hand side operator (unified K/Ms or fused).
 *
 * Two output modes, selected by the `Ms` pointer:
 *
 *   Ms != NULL (K/Ms mode, used by the inverse kernel):
 *     *KorA receives the curl-curl stiffness K, *Ms receives the mass-σ
 *     matrix Ms. The frequency-dependent operator A_f = K − iωμ·Ms is
 *     formed by the caller per frequency via MatDuplicate + MatAXPY.
 *     `constFactor` is ignored in this mode.
 *
 *   Ms == NULL (fused mode, used by the forward kernel):
 *     *KorA receives the frequency-dependent operator A = K − constFactor·Ms
 *     directly, formed by per-cell element-level fusion
 *     A_e = K_e − constFactor·M_e, so Ms is never built as a global matrix.
 *     Saves one full complex matrix from the assembly-phase peak memory,
 *     one MatDuplicate, and one global MatAXPY. The caller passes
 *     `constFactor = iωμ`.
 *
 * Matrices:
 *   - K      : curl-curl stiffness, ∫_K (μ⁻¹ curl Ni)·curl Nj.
 *   - Ms     : mass × σ,            ∫_K (ε_r ⊙ Ni)·Nj.
 *   - G_BDDC : lowest-Whitney topological gradient consumed by
 *              PCBDDCSetDiscreteGradient at order = 1. Each H(curl) edge DOF
 *              couples to its two endpoint vertex H1 DOFs (±1); higher-order
 *              rows and inter-bubble columns are zero. Built against
 *              grid.H1dm_Pnord so a single solver code path serves every order.
 *
 * @param[in]  params       Forward-modeling parameters (nord, MPI tasks).
 * @param[in]  dm           DMPlex mesh and H(curl) discretization.
 * @param[in]  grid         Finite-element grid descriptor.
 * @param[in]  conductivity Per-cell conductivity Vec.
 * @param[in]  constFactor  Fused-mode factor iωμ; ignored when Ms != NULL.
 * @param[out] KorA         Stiffness K (K/Ms mode) or fused operator A.
 * @param[out] Ms           Mass-σ matrix in K/Ms mode; pass NULL for fused mode.
 * @param[out] G_BDDC       Topological discrete gradient; pass NULL to skip.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode assembleCsemKandM(const fmParams params, const DM dm, const Grid grid,
                                 const Vec conductivity,
                                 const PetscScalar constFactor,
                                 Mat *KorA, Mat *Ms,
                                 Mat *G_BDDC);

#include "hvfem.h"  /* Quadrature3D used by assembleCsemMsRefill */

/**
 * @brief Refills an existing Ms matrix for the current conductivity field.
 *
 * Used by the inverse kernel inside the L-BFGS loop: K and G_BDDC are
 * σ-independent and built once at setup via assembleCsemKandM, while Ms
 * must be re-computed every iteration when σ changes. fm.csem does NOT use
 * this - its fused single-pass call to assembleCsemKandM is unchanged.
 *
 * Preconditions:
 *   - `Ms` is already allocated with the same sparsity pattern as the K
 *     produced by assembleCsemKandM for the same mesh / nord (typically
 *     `MatDuplicate(K, MAT_DO_NOT_COPY_VALUES, &Ms)`).
 *   - `quadrature_3d`, `Me`, `Ke` are caller-owned workspace buffers of the
 *     same shape used inside assembleCsemKandM (numDofInCell² for Me / Ke),
 *     passed in so a single allocation can serve all iterations.
 *
 * On return Ms holds the σ-dependent mass-matrix entries with its sparsity
 * preserved. Ke is computed by computeElementalMatrices but is unused here
 * (kept in the signature so the caller can share the gradient-pass scratch).
 *
 * @param[in]     params         Forward-modeling parameters (nord).
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
PetscErrorCode assembleCsemMsRefill(const fmParams params,
                                    const DM dm, const Grid grid,
                                    const Vec conductivity,
                                    const Quadrature3D *quadrature_3d,
                                    PetscReal **Me, PetscReal **Ke,
                                    Mat Ms);

#endif
