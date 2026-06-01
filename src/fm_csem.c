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
  Mat             G_BDDC = NULL;  /* Topological lowest-Whitney G : Nédélec_k -> P_nord H1 */
  fmParams        params;
  Grid            grid;
  CsemSourceSet   sources = {0, 0, NULL};
  PetscReal       omega;      
  PetscScalar     constFactor;
  PetscLogDouble  timers[6];
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

  /* Register the six fm.csem phase stages with PETSc's logging system.
   * The existing PetscTime-based timer table is preserved unchanged; these
   * stages make `-log_view` produce a per-rank, per-event profile (including
   * KSP / Mat / Vec sub-events) for free.  Stage names match the printTimers
   * row labels so the two reports cross-reference cleanly. */
  PetscLogStage stage_parse, stage_load, stage_grid;
  PetscLogStage stage_assembly, stage_solve, stage_postproc;
  PetscCall(PetscLogStageRegister("Read parameters",     &stage_parse));
  PetscCall(PetscLogStageRegister("Load input bundle",   &stage_load));
  PetscCall(PetscLogStageRegister("Setup grid",          &stage_grid));
  PetscCall(PetscLogStageRegister("Assembly",            &stage_assembly));
  PetscCall(PetscLogStageRegister("Linear solve",        &stage_solve));
  PetscCall(PetscLogStageRegister("Field interpolation", &stage_postproc));

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

  PetscCall(PetscLogStagePush(stage_parse));
  PetscCall(PetscTime(&start_timer));
  PetscCall(readfmParams(size, &params));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[0] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Load input data: mesh, sigma, materials_id, sources, receivers   */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 4);
#endif

  PetscCall(PetscLogStagePush(stage_load));
  PetscCall(PetscTime(&start_timer));
  PetscCall(loadCsemInputs(&params, &dm, &conductivity, &materials_id, &sources, &receivers));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[1] = end_timer - start_timer;
  
#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Setup grid for finite element computations                       */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 5);
#endif

  PetscCall(PetscLogStagePush(stage_grid));
  PetscCall(PetscTime(&start_timer));
  PetscCall(setupCsemGrid(params, &dm, &grid));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[2] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Assembly linear system                                              */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 6);
#endif

  /* Compute constants for assembly phase */
  omega       = sources.freq * 2.0 * PETSC_PI;
  constFactor = (0.0 + 1.0 * PETSC_i) * (omega * MU);
  
  /* Perform assembly */
  PetscCall(PetscLogStagePush(stage_assembly));
  PetscCall(PetscTime(&start_timer));
  PetscCall(assembleCsemRHS(params, sources, dm, grid, &B));
  PetscCall(assembleCsemKandM(params, dm, grid, conductivity, constFactor, &A, NULL, &G_BDDC));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[3] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Solve linear system                                              */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 7);
#endif

  PetscCall(PetscLogStagePush(stage_solve));
  PetscCall(PetscTime(&start_timer));
  PetscCall(solveCsemSystem(dm, A, B, G_BDDC, &X));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[4] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Postprocessing solution                                          */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 8);
#endif

  PetscCall(PetscLogStagePush(stage_postproc));
  PetscCall(PetscTime(&start_timer));
  PetscCall(computeFields(params, sources, dm, grid, receivers, X));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[5] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Print timers and footer                                          */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 9);
#endif

  PetscCall(printTimers(timers));
  PetscCall(printFooter());

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Free memory                                                      */
  /* ---------------------------------------------------------------- */
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

