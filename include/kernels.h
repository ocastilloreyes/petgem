/*
 * Filename: kernels.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Public entry points for the CSEM kernels (runForward, runInverse).
 */

/*
 * Notes:
 * The forward kernel (runForward) and inverse kernel (runInverse) are
 * exposed as functions so they can be called either from the legacy
 * single-purpose binaries (fm.csem, im.csem) or from the unified
 * dispatcher (petgem).
 *
 * Each function owns its complete PETSc lifecycle: it calls
 * PetscInitialize at entry and PetscFinalize at exit, and returns the
 * PetscErrorCode (cast to int) as the would-be exit status.
 *
 * Usage:
 *     #include "kernels.h"
 *     int main(int argc, char **argv) { return runForward(argc, argv); }
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

#endif /* KERNELS_H */
