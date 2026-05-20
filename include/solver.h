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

/* Solve A·X = B. The G argument is the topological lowest-Whitney
 * gradient G_BDDC : Nédélec_k → P_nord H1 from assembleCsemKandM, used
 * as the structural hint for PCBDDCSetDiscreteGradient at order = 1
 * (the canonical P_nord G is too dense for PCBDDC's nnz budget at
 * nord ≥ 3). May be NULL when ismatis is false. */
PetscErrorCode solveCsemSystem(const DM dm, const Mat A, const Mat B, const Mat G, Mat* X);

#endif
