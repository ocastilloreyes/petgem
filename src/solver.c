/*
 * Filename: solver.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2025-08-05
 *
 * Description:
 * This file contains functions solving phase. 
 *
 * Usage:
 * Include this file in your source code to utilize the solver functions. 
 * For example:
 * #include "solver.h"
*/
 
/* C libraries */ 

/* PETSc libraries */
#include <petscsys.h>
#include <petscksp.h> 
#include <petscdmplex.h>
#include "inputs.h"

/* PETGEM functions */ 


/**
 * @brief Solves the linear system AX=B using KSP.
 *
 * If the matrix @p A is of type MATIS and @p G is provided, it configures
 * the preconditioner to PCBDDC and sets the discrete gradient using PCBDDCSetDiscreteGradient.
 *
 * @param[in] dm The DMPlex object (used for communicator).
 * @param[in] A The system matrix (assembled by assembleSystem).
 * @param[in] B The right-hand side matrix (assembled by assembleSystem, one column per source).
 * @param[in] G The discrete gradient matrix (used for PCBDDC setup if @p A is MATIS).
 * @param[in] params A Params struct containing simulation parameters (used for PCBDDC setup).
 * @param[out] X Pointer to the solution matrix (Mat) to be created and populated.
 * @return PetscErrorCode PETSC_SUCCESS on successful solve, or an error code otherwise.
 */
PetscErrorCode solveCsemSystem(DM dm, Mat A, Mat B, Mat G, Params params, Mat *X){
    
    PetscFunctionBeginUser;
    
    /* Create KSP object */
    MPI_Comm comm = PetscObjectComm((PetscObject)dm);
    KSP ksp;

    /* Setup solver and run it */
    PetscCall(KSPCreate(comm, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));

    PetscBool ismatis = PETSC_FALSE;
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
    if (ismatis && G) {
      PC pc;

      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCSetType(pc, PCBDDC));
      PetscCall(PCBDDCSetDiscreteGradient(pc, G, params.nord, 0, PETSC_TRUE, PETSC_TRUE));
    }
    PetscCall(KSPSetFromOptions(ksp));

    PetscInt M, N, m, n;
    VecType vtype;
    PetscCall(MatGetSize(B, &M, &N));
    PetscCall(MatGetLocalSize(B, &m, &n));
    PetscCall(MatGetVecType(A, &vtype));
    PetscCall(MatCreateDenseFromVecType(comm, vtype, m, n, M, N, m, NULL, X));
    PetscCall(PetscPrintf(comm, "\n Solution of %" PetscInt_FMT " linear systems:\n", N));
    PetscCall(PetscPrintf(comm, "   Solver process    = Initiated\n"));
    PetscCall(KSPMatSolve(ksp, B, *X));    
    PetscCall(PetscPrintf(comm, "   Solver process    = Finished\n"));
    PetscCall(KSPDestroy(&ksp));  
    
    PetscFunctionReturn(PETSC_SUCCESS);
}
