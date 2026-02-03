static char help[] = "PETGEM kernel for 3D CSEM imaging using high-order vector finite elements.\n\
  Command line usage:\n\
    mpirun -n <np> ./im.csem -options_file <file.txt>\n";

/* C libraries */
#include <stdio.h>
#include <stdlib.h>

/* PETSc functions */
#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>

/* PETGEM functions */
#include "assembly.h"
#include "common.h"
#include "constants.h"
#include "grid.h"
#include "inputs.h"
#include "postprocessing.h"
#include "solver.h"
#include "transmitter.h"
#include "version.h"

/* Extrae library for performance analysis */
#ifdef USE_EXTRAE
#include "extrae_user_events.h"
#endif

int main(int argc, char** argv) {

  /* Check if the --version option is provided */
  if (argc > 1 && strcmp(argv[1], "--version") == 0) {
    printf("PETGEM version %d.%d.%d\n", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
    return 0;
  }

  //   /* Variables declaration */
  PetscMPIInt rank, size;
  //   DM dm;
  //   Vec resistivity;
  //   Mat A, B, X;
  //   Mat G = NULL;
  //   csemParams params;
  //   Grid grid;
  //   CsemSourceSet sources = {0, 0, NULL};
  //   PetscLogDouble timers[7];
  //   PetscLogDouble start_timer, end_timer;

  /* PETSC initialization */
  PetscFunctionBeginUser;
#ifdef USE_EXTRAE
  Extrae_event(1000, 1);
#endif

  PetscCall(PetscInitialize(&argc, &argv, (char*)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

/* Print PETGEM header */
#ifdef USE_EXTRAE
  Extrae_event(1000, 2);
#endif

  PetscCall(printHeader());

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  // /* Parse user parameters */
  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 3);
  // #endif

  //   /* Start timer for read params */
  //   PetscCall(PetscTime(&start_timer));

  //   PetscCall(readCsemParams(size, &params));

  //   /* End timer for read params */
  //   PetscCall(PetscTime(&end_timer));
  //   timers[0] = end_timer - start_timer;

  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 0);
  // #endif

  // /* Create and setup source */
  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 4);
  // #endif

  //   /* Start timer for setup source */
  //   PetscCall(PetscTime(&start_timer));

  //   PetscCall(setupCsemSource(params, &sources));

  //   /* End timer for setup source */
  //   PetscCall(PetscTime(&end_timer));
  //   timers[1] = end_timer - start_timer;

  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 0);
  // #endif

  // /* Import mesh and resistivity model */
  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 5);
  // #endif

  //   /* Start timer for import grid */
  //   PetscCall(PetscTime(&start_timer));

  //   PetscCall(importGrid(params, &dm, &resistivity));

  //   /* End timer for import grid */
  //   PetscCall(PetscTime(&end_timer));
  //   timers[2] = end_timer - start_timer;

  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 0);
  // #endif

  // /* Setup grid for FE computations */
  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 6);
  // #endif

  //   /* Start timer for setup grid */
  //   PetscCall(PetscTime(&start_timer));

  //   PetscCall(setupCsemGrid(params, &dm, &grid));

  //   /* End timer for setup grid */
  //   PetscCall(PetscTime(&end_timer));
  //   timers[3] = end_timer - start_timer;

  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 0);
  // #endif

  // /* Setup linear system */
  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 7);
  // #endif

  //   /* Start timer for assembly grid */
  //   PetscCall(PetscTime(&start_timer));

  //   /* Assemble linear system */
  //   PetscCall(assembleCsemRHS(params, sources, dm, grid, &B));
  //   PetscCall(assembleCsemLHS(params, sources, dm, grid, resistivity, &A, &G));

  //   /* End timer for assembly */
  //   PetscCall(PetscTime(&end_timer));
  //   timers[4] = end_timer - start_timer;

  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 0);
  // #endif

  // /* Solve linear system */
  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 8);
  // #endif

  //   /* Start timer for solver */
  //   PetscCall(PetscTime(&start_timer));

  //   PetscCall(solveCsemSystem(params, dm, A, B, G, &X));

  //   /* End timer for solver */
  //   PetscCall(PetscTime(&end_timer));
  //   timers[5] = end_timer - start_timer;

  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 0);
  // #endif

  // /* Postprocessing solution */
  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 9);
  // #endif

  //   /* Start timer for postprocessing */
  //   PetscCall(PetscTime(&start_timer));

  //   PetscCall(computeFields(params, sources, dm, grid, X));

  //   /* End timer for postprocessing */
  //   PetscCall(PetscTime(&end_timer));
  //   timers[6] = end_timer - start_timer;

  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 0);
  // #endif

  //   /* Print timers */
  //   PetscCall(printTimers(timers));

  // /* Print PETGEM footer */
  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 10);
  // #endif

  //   PetscCall(printFooter());

  // #ifdef USE_EXTRAE
  //   Extrae_event(1000, 0);
  // #endif

  //   /* Free memory */
  //   PetscCall(DMDestroy(&grid.H1dm));
  //   PetscCall(DMDestroy(&dm));
  //   PetscCall(VecDestroy(&resistivity));
  //   PetscCall(MatDestroy(&G));
  //   PetscCall(MatDestroy(&A));
  //   PetscCall(MatDestroy(&B));
  //   PetscCall(MatDestroy(&X));
  //   PetscCall(PetscFree(sources.sourceArray));

  /* PETSc finalize*/
  PetscCall(PetscFinalize());
  return 0;
}