/*
 * Filename: kernels.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Public entry points for the PETGEM kernels (runForward, runInverse, runMtForward,
 * runTdCsemForward).
 */

/*
 * Notes:
 * Each kernel is called from its own binary (fm.csem, im.csem, fm.mt, fm.tdcsem) and
 * from the unified dispatcher (petgem). Each one owns the whole PETSc
 * lifecycle (PetscInitialize ... PetscFinalize) and returns the exit status.
 */

#ifndef KERNELS_H
#define KERNELS_H

#include <petsc.h>

/**
 * @brief Runs the CSEM forward-modeling kernel (fm.csem).
 *
 * Owns the complete PETSc lifecycle: calls PetscInitialize at entry and
 * PetscFinalize at exit.
 *
 * @param[in] argc  Argument count from main().
 * @param[in] argv  Argument vector from main().
 *
 * @return int the PetscErrorCode (cast to int) as the process exit status.
 */
int runForward(int argc, char **argv);

/**
 * @brief Runs the CSEM inversion kernel (im.csem).
 *
 * Owns the complete PETSc lifecycle: calls PetscInitialize at entry and
 * PetscFinalize at exit.
 *
 * @param[in] argc  Argument count from main().
 * @param[in] argv  Argument vector from main().
 *
 * @return int the PetscErrorCode (cast to int) as the process exit status.
 */
int runInverse(int argc, char **argv);

/**
 * @brief Runs the MT forward-modeling kernel (fm.mt).
 *
 * Owns the complete PETSc lifecycle: calls PetscInitialize at entry and
 * PetscFinalize at exit.
 *
 * @param[in] argc  Argument count from main().
 * @param[in] argv  Argument vector from main().
 *
 * @return int the PetscErrorCode (cast to int) as the process exit status.
 */
int runMtForward(int argc, char **argv);

/**
 * @brief Runs the time-domain CSEM forward-modeling kernel (fm.tdcsem).
 *
 * Owns the complete PETSc lifecycle: calls PetscInitialize at entry and
 * PetscFinalize at exit.
 *
 * @param[in] argc  Argument count from main().
 * @param[in] argv  Argument vector from main().
 *
 * @return int the PetscErrorCode (cast to int) as the process exit status.
 */
int runTdCsemForward(int argc, char **argv);

#endif /* KERNELS_H */
