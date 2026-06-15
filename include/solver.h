/*
 * Filename: solver.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Prototypes for the linear-system solver routines used by the
 * PETGEM kernels.
 */

#ifndef SOLVER_H
#define SOLVER_H

#include "inputs.h"
#include <petsc.h>
#include <petscdmplex.h>

/**
 * @brief Solves the linear system A·X = B using PETSc KSP.
 *
 * When A is a MATIS matrix and G is provided, the preconditioner is set to
 * PCBDDC and G is registered via PCBDDCSetDiscreteGradient at order = 1 to
 * capture the curl kernel for H(curl) problems. G is the high-order discrete
 * gradient G_BDDC : Nédélec_nord → P_nord H1 from assembleCsemKandM.
 *
 * @param[in]  dm  DMPlex mesh; its communicator drives the parallel solve.
 * @param[in]  A   System matrix (H(curl) FEM operator).
 * @param[in]  B   Right-hand side matrix, one column per source.
 * @param[in]  G   Discrete-gradient hint for PCBDDC; may be NULL when A is
 *                 not of type MATIS.
 * @param[out] X   Solution matrix, created internally (caller destroys).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode solveCsemSystem(const DM dm, const Mat A, const Mat B, const Mat G, Mat* X);

/**
 * @brief Configures `ksp`'s PC as PCBDDC with the discrete-gradient hint, if
 *        the operator is MATIS and a gradient was supplied.
 *
 * Shared helper that captures the single BDDC policy used by both the
 * forward solver (solveCsemSystem) and the inverse solver (createInvKSP):
 * when A is of type MATIS and Gbddc is non-NULL, set PC to PCBDDC and
 * register Gbddc (the high-order discrete gradient) via
 * PCBDDCSetDiscreteGradient(..., 1, 0, PETSC_TRUE, PETSC_TRUE). When the
 * conditions are not met the function is a no-op so the caller's existing
 * default PC stays in place.
 *
 * @param[in,out] ksp   KSP whose preconditioner is configured.
 * @param[in]     A     System matrix (probed for MATIS).
 * @param[in]     Gbddc Discrete-gradient hint (may be NULL).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode setupBDDCFromPetgemGradient(KSP ksp, Mat A, Mat Gbddc);

#endif
