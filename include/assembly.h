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

PetscErrorCode assembleCsemRHS(const csemParams params, const CsemSourceSet sources, const DM dm, const Grid grid, Mat* B);

/* Unified CSEM LHS assembly.
 *
 * Two output modes, selected by the `Ms` pointer:
 *
 *   Ms != NULL  (K/Ms mode, used by the inverse kernel):
 *     *KorA receives the curl-curl stiffness K, *Ms receives the
 *     mass-σ matrix Ms. The frequency-dependent operator
 *           A_f = K − iωμ · Ms
 *     is formed by the caller per frequency via MatDuplicate +
 *     MatAXPY. `constFactor` is ignored in this mode.
 *
 *   Ms == NULL  (fused mode, used by the forward kernel):
 *     *KorA receives the frequency-dependent operator
 *           A = K − constFactor · Ms
 *     directly, formed by per-cell element-level fusion
 *           A_e = K_e − constFactor · M_e
 *     so that Ms is never built as a global matrix. Saves one full
 *     complex matrix from the assembly-phase peak memory, one
 *     MatDuplicate, and one global MatAXPY. The caller passes
 *     `constFactor = iωμ`.
 *
 *   K       — curl-curl stiffness, ∫_K (μ⁻¹ curl Ni)·curl Nj.
 *   Ms      — mass × σ,           ∫_K (ε_r ⊙ Ni)·Nj.
 *   G       — order-k canonical Π^Ned gradient G : S_h^k → V_h^k
 *             (Ainsworth–Coyle DOF moments, K·G = 0 by construction).
 *             Used for analysis and the K_e·G_e verification.
 *   G_BDDC  — lowest-Whitney topological gradient consumed by
 *             PCBDDCSetDiscreteGradient at order = 1. Each H(curl)
 *             edge DOF couples to its two endpoint vertex H1 DOFs
 *             (±1); higher-order rows and inter-bubble columns are
 *             zero. Built against grid.H1dm_Pnord so a single solver
 *             code path serves every order.
 *
 * Either or both of G / G_BDDC may be passed as NULL to skip that
 * matrix's construction. */
PetscErrorCode assembleCsemKandM(const csemParams params, const DM dm, const Grid grid,
                                 const Vec conductivity,
                                 const PetscScalar constFactor,
                                 Mat *KorA, Mat *Ms,
                                 Mat *G, Mat *G_BDDC);

#endif
