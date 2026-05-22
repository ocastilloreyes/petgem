/*
 * Filename: fm_csem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Forward CSEM kernel (runForward). Linked into both the legacy
 * fm.csem binary and the unified petgem dispatcher.
 */

static char fmHelp[] = "PETGEM forward CSEM kernel (runForward / fm.csem).\n\
  Standalone usage:\n\
    mpirun -n <np> ./fm.csem -options_file <file.txt>\n\
  Unified-binary usage:\n\
    mpirun -n <np> ./petgem modeling -options_file <file.txt>\n";

/* C libraries */
#include <stdio.h>
#include <stdlib.h>

/* PETSc functions */
#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>
#include <petscviewerhdf5.h>


/* PETGEM functions */
#include "assembly.h"
#include "common.h"
#include "constants.h"
#include "grid.h"
#include "inputs.h"
#include "io.h"
#include "kernels.h"
#include "postprocessing.h"
#include "solver.h"
#include "transmitter.h"
#include "version.h"

/* Extrae library for performance analysis */
#ifdef USE_EXTRAE
#include "extrae_user_events.h"
#endif

/**
 * @brief Main execution routine for CSEM forward modeling kernel.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return int 0 on success, non-zero on failure.
 * @details Initializes PETSc, parses command-line arguments
 * (including `--version`). Prints the header. Reads user
 * parameters, sets up sources, imports the grid and
 * conductivity, sets up the DM sections and grid structure,
 * assembles the linear system (A, B, G), solves the system
 * (AX=B) using KSP, performs post-processing (computes
 * fields at receivers), prints the footer, and finalizes
 * PETSc, freeing allocated memory. Includes Extrae
 * instrumentation hooks if compiled with USE_EXTRAE.
 */
int runForward(int argc, char** argv) {


  /* ---------------------------------------------------------------- */
  /* Check if the --version option is provided                        */
  /* ---------------------------------------------------------------- */
  if (argc > 1 && strcmp(argv[1], "--version") == 0) {
    printf("PETGEM version %d.%d.%d\n", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
    return 0;
  }

  /* ---------------------------------------------------------------- */
  /* Variables declaration                                            */
  /* ---------------------------------------------------------------- */
  PetscMPIInt     rank, size;
  DM              dm;
  Vec             conductivity, materials_id, receivers;
  Mat             A = NULL, B, X;
  Mat             G_BDDC = NULL;  /* Topological lowest-Whitney G : Nédélec_k → P_nord H1. */
  csemParams      params;
  Grid            grid;
  CsemSourceSet   sources = {0, 0, NULL};
  PetscLogDouble  timers[7];
  PetscLogDouble  start_timer, end_timer;

  /* ---------------------------------------------------------------- */
  /* PETSC initialization                                             */
  /* ---------------------------------------------------------------- */
  PetscFunctionBeginUser;
#ifdef USE_EXTRAE
  Extrae_event(1000, 1);
#endif

  PetscCall(PetscInitialize(&argc, &argv, (char*)0, fmHelp));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Print PETGEM header                                              */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 2);
#endif

  PetscCall(printHeader());

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Parse user parameters                                            */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 3);
#endif

  PetscCall(PetscTime(&start_timer));
  PetscCall(readCsemParams(size, &params));
  PetscCall(PetscTime(&end_timer));
  timers[0] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Load unified PETGEM input: mesh + sigma + materials_id + sources */
  /* + receivers, all from params.inputFile in a single routine.       */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 4);
#endif

  PetscCall(PetscTime(&start_timer));
  PetscCall(loadCsemInputs(&params, &dm, &conductivity, &materials_id,
                           &sources, &receivers));
  PetscCall(PetscTime(&end_timer));
  timers[1] = end_timer - start_timer;
  timers[2] = 0.0;  /* legacy slot, kept for printTimers compatibility */

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Setup grid for FE computations                                   */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 6);
#endif

  PetscCall(PetscTime(&start_timer));
  PetscCall(setupCsemGrid(params, &dm, &grid));
  PetscCall(PetscTime(&end_timer));
  timers[3] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Setup linear system                                              */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 7);
#endif

  PetscCall(PetscTime(&start_timer));
  PetscCall(assembleCsemRHS(params, sources, dm, grid, &B));

  /* Single unified assembly in fused mode: A = K − iωμ·Ms is built
   * directly via element-level fusion (K_e − iωμ·M_e per cell). No
   * global Ms is allocated, no MatDuplicate, no MatAXPY. The
   * topological G_BDDC for PCBDDC is built in the same element loop;
   * the canonical Π^Ned G is skipped (NULL). */
  {
    const PetscReal   omega       = sources.freq * 2.0 * PETSC_PI;
    const PetscScalar constFactor = (0.0 + 1.0 * PETSC_i) * (omega * MU);
    PetscCall(assembleCsemKandM(params, dm, grid, conductivity,
                                constFactor,
                                &A,    /* fused output: A = K − constFactor·Ms */
                                NULL,  /* Ms == NULL selects fused mode        */
                                NULL,  /* canonical G - skip                    */
                                &G_BDDC));
  }
  PetscCall(PetscTime(&end_timer));
  timers[4] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Solve linear system                                              */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 8);
#endif

  PetscCall(PetscTime(&start_timer));
  /* G_BDDC is the topological lowest-Whitney gradient mapping
   * Nédélec_k → P_nord H1 (vertex incidence only). Built against the
   * same DM as the canonical G (grid.H1dm_Pnord), so the same code
   * path serves every order: PCBDDC sees a ±1 sparsity structure at
   * the vertex-DOF tail of the P_nord closure, independent of nord. */
  PetscCall(solveCsemSystem(dm, A, B, G_BDDC, &X));
  PetscCall(PetscTime(&end_timer));
  timers[5] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Postprocessing solution                                          */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 9);
#endif

  PetscCall(PetscTime(&start_timer));
  PetscCall(computeFields(params, sources, dm, grid, receivers, X));
  PetscCall(PetscTime(&end_timer));
  timers[6] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Print timers and footer                                          */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 10);
#endif

  PetscCall(printTimers(timers));
  PetscCall(printFooter());

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Free memory                                                      */
  /* ---------------------------------------------------------------- */
  PetscCall(DMDestroy(&grid.H1dm));
  PetscCall(DMDestroy(&grid.H1dm_Pnord));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&conductivity));
  PetscCall(VecDestroy(&materials_id));
  PetscCall(MatDestroy(&G_BDDC));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&X));
  PetscCall(VecDestroy(&receivers));
  PetscCall(PetscFree(sources.sourceArray));

  /* ---------------------------------------------------------------- */
  /* PETSc finalize                                                   */
  /* ---------------------------------------------------------------- */
  PetscCall(PetscFinalize());
  return 0;
}

