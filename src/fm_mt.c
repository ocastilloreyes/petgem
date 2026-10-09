/*
 * Filename: fm_mt.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * Forward MT kernel (runMtForward). Linked into the fm.mt binary.
 */

static char mtHelp[] =
"================================================================\n\
 PETGEM  fm.mt  --  MT forward modeling kernel\n\
================================================================\n\
\n\
 Computes the 3-D magnetotelluric response on a tetrahedral\n\
 mesh using high-order (1..6) Nedelec finite elements.\n\
\n\
QUICK START\n\
  mpirun -n <np> ./fm.mt -options_file \n\
\n\
REQUIRED\n\
  -options_file <file>  Read options (and PETSc flags) from file.\n\
\n\
EXAMPLES\n\
  mpirun -n 96 ./fm.mt -options_file params.txt\n\
\n\
MORE\n\
  -help        Full option database, incl. advanced PETSc flags.\n\
  -help intro  This concise usage summary, then exit.\n\
  --version    Print the PETGEM version and exit.\n\
\n\
 Docs: github.com/ocastilloreyes/petgem\n\
================================================================\n";

/* C libraries */
#include <stdio.h>
#include <stdlib.h>

/* PETGEM functions */
#include "assembly.h"
#include "common.h"
#include "constants.h"
#include "grid.h"
#include "io.h"
#include "kernels.h"
#include "mt.h"
#include "solver.h"
#include "version.h"

/* Extrae library for performance analysis */
#ifdef USE_EXTRAE
#include "extrae_user_events.h"
#endif

int runMtForward(int argc, char** argv) {

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
  Mat             A = NULL, B, X; /* Global matrix (A), RHS (B),  unknown vector solution (X)*/
  Mat             G = NULL;       /* Discrete gradient for PCBDDC */
  IS              faces;
  MtBoxFace      *tags;
  Mt1DProfile     profile;
  Mt1DField       field;
  petgemParams    params;
  MtParams        mt;
  Grid            grid;
  PetscReal       omega;
  PetscScalar     constFactor;
  PetscLogDouble  timers[6];
  SolveInfo       solveInfo;
  MPI_Comm        comm;
  PetscLogDouble  start_timer, end_timer;
  PetscLogStage stage_parse, stage_load, stage_grid;
  PetscLogStage stage_assembly, stage_solve, stage_postproc;
  PetscBool     helpRequested = PETSC_FALSE;

  /* ---------------------------------------------------------------- */
  /* PETSC initialization                                             */
  /* ---------------------------------------------------------------- */
  PetscFunctionBeginUser;
#ifdef USE_EXTRAE
  Extrae_event(1000, 1);
#endif

  PetscCall(PetscInitialize(&argc, &argv, (char*)0, mtHelp));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* ---------------------------------------------------------------- */
  /* Resolve here the petsc help printing                             */
  /* ---------------------------------------------------------------- */
  PetscCall(PetscOptionsHasHelp(NULL, &helpRequested));

  /* ---------------------------------------------------------------- */
  /* Register the fm.mt phase stages with PETSc's logging system      */
  /* ---------------------------------------------------------------- */
  PetscCall(PetscLogStageRegister("Read parameters",     &stage_parse));
  PetscCall(PetscLogStageRegister("Load input bundle",   &stage_load));
  PetscCall(PetscLogStageRegister("Setup grid",          &stage_grid));
  PetscCall(PetscLogStageRegister("Assembly",            &stage_assembly));
  PetscCall(PetscLogStageRegister("Linear solve",        &stage_solve));
  PetscCall(PetscLogStageRegister("MT responses",        &stage_postproc));

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Print PETGEM header                                              */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 2);
#endif

  if (!helpRequested) PetscCall(printHeader("fm.mt"));

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
  PetscCall(readPetgemParams(size, &params));
  PetscCall(readMtParams(&mt));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[0] = end_timer - start_timer;

  /* Option groups already printed by -help; skip simulation and exit. */
  if (helpRequested) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nTip: 'fm.mt -help intro' shows the concise usage summary.\n"));
    PetscCall(PetscFinalize());
    return 0;
  }

  PetscCheck(!params.mms, comm, PETSC_ERR_SUP, "fm.mt does not support -mms (use fm.csem).");

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ----------------------------------------------------------------------- */
  /* Load input data: mesh, conductivity, materials_id, receivers, frequency */
  /* ----------------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 4);
#endif

  PetscCall(PetscLogStagePush(stage_load));
  PetscCall(PetscTime(&start_timer));
  PetscCall(loadModelInputs(&params, &dm, &conductivity, &materials_id, &receivers));
  PetscCall(loadMtSettings(&params, &mt));
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
  PetscCall(setupNedelecGrid(params, PETGEM_BC_NATURAL, &dm, &grid));
  PetscCall(getBoundaryFaces(dm, &faces));
  PetscCall(classifyMtBoxFaces(dm, faces, &tags));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[2] = end_timer - start_timer;

  PetscCall(logGridSummary(params, dm, &grid));

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Assembly linear system                                           */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 6);
#endif

  /* Compute constants for assembly phase */
  omega       = mt.frequency * 2.0 * PETSC_PI;
  constFactor = (0.0 + 1.0 * PETSC_i) * (omega * MU);

  /* 1D boundary field, boundary RHS and operator */
  PetscCall(PetscLogStagePush(stage_assembly));
  PetscCall(PetscTime(&start_timer));
  PetscCall(buildMt1DProfile(dm, conductivity, faces, tags, &profile));
  PetscCall(solveMt1D(&profile, &mt, omega, &field));
  PetscCall(assembleMtBoundaryRHS(params, omega, dm, grid, faces, tags, &field, &B));
  PetscCall(assembleMaxwellOperator(params, dm, grid, conductivity, constFactor, &A, NULL, &G));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[3] = end_timer - start_timer;

  PetscCall(logMtSetup(comm, &mt, faces, &profile, &field));
  {
    MatType mtype;
    PetscCall(MatGetType(A, &mtype));
    PetscCall(logSection(comm, "Assembly"));
    PetscCall(logKVStr(comm, "Matrix type", mtype));
    PetscCall(logKVInt(comm, "Right-hand sides", MT_NUM_POLARIZATIONS));
  }

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
  PetscCall(solveMaxwellSystem(dm, A, B, G, grid.fem.order, NULL, &X, &solveInfo));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[4] = end_timer - start_timer;
  PetscCall(logSection(comm, "Solve"));
  PetscCall(logKVStr(comm, "Solver", solveInfo.solver));
  PetscCall(logKVStr(comm, "Status", solveInfo.reason > 0 ? "converged" : KSPConvergedReasons[solveInfo.reason]));

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
  PetscCall(computeMtResponses(params, &mt, dm, grid, receivers, X));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[5] = end_timer - start_timer;
  {
    PetscInt nrecv;
    char     outFile[PETSC_MAX_PATH_LEN];
    PetscCall(VecGetSize(receivers, &nrecv));
    PetscCall(buildOutputPath(&params, ".h5", outFile, sizeof(outFile)));
    PetscCall(logSection(comm, "MT responses"));
    PetscCall(logKVInt(comm, "Number of receivers", nrecv / 3));
    PetscCall(logKVStr(comm, "Output file", outFile));
  }

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Print timers and footer                                          */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 9);
#endif

  {
    const char *const   labels[] = {"Load + grid", "Assembly", "Linear solve", "MT responses"};
    const PetscLogDouble times[]  = {timers[0] + timers[1] + timers[2], timers[3], timers[4], timers[5]};
    PetscCall(printTimers(labels, times, 4));
  }
  PetscCall(printFooter());

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Free memory                                                      */
  /* ---------------------------------------------------------------- */
  PetscCall(destroyMt1DField(&field));
  PetscCall(destroyMt1DProfile(&profile));
  PetscCall(PetscFree(tags));
  PetscCall(ISDestroy(&faces));
  PetscCall(DMDestroy(&grid.H1dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&conductivity));
  PetscCall(VecDestroy(&materials_id));
  PetscCall(MatDestroy(&G));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&X));
  PetscCall(VecDestroy(&receivers));

  /* ---------------------------------------------------------------- */
  /* PETSc finalize                                                   */
  /* ---------------------------------------------------------------- */
  PetscCall(PetscFinalize());
  return 0;
}
