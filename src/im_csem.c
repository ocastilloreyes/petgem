/*
 * Filename: im_csem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Inverse CSEM kernel (runInverse). Linked into both the legacy
 * im.csem binary and the unified petgem dispatcher.
 */

static char imHelp[] =
"================================================================\n\
 PETGEM  im.csem  --  CSEM inversion kernel\n\
================================================================\n\
\n\
 Recovers a 3-D conductivity model from CSEM data by L-BFGS with\n\
 an adjoint-state gradient.\n\
\n\
QUICK START\n\
  mpirun -n <np> ./im.csem -options_file params.txt -output_dir out/\n\
\n\
REQUIRED\n\
  -input_filename <file.h5>  Unified PETGEM bundle (mesh, start\n\
                             model, sources, receivers, /observed).\n\
  -output_dir <dir>          Output directory (created if absent).\n\
  -output_filename <stem>    Output stem; writes <dir>/<stem>.h5.\n\
\n\
OPTIONAL  (inversion control; defaults in brackets)\n\
  -order <1..6>              Override bundle basis order.\n\
  -im_max_iter <n>           Max L-BFGS iterations            [80]\n\
  -im_lbfgs_memory <m>       L-BFGS history depth M            [5]\n\
  -im_lambda <r>             Tikhonov regularisation weight  [0.1]\n\
  -im_error_level <r>        Relative data-error level      [0.01]\n\
  -im_gtol <r>               Gradient-norm tolerance        [1e-5]\n\
  -im_rms_tol <r>            Absolute RMS early stop (0=off)   [0]\n\
  -im_rms_rtol <r>           RMS-plateau rel. threshold     [1e-3]\n\
  -im_rms_stall_window <n>   Stalled iters before stopping     [3]\n\
  -im_diag_weight <r>        Gradient-smoother self-weight     [0]\n\
  -im_fixed_materials <ids>  Material IDs frozen in smoothing\n\
  -im_snapshot_interval <n>  Write VTU every N steps (0=off)   [0]\n\
  -im_observed_mode <m>      external | fm_native     [external]\n\
  -im_observed_file <f>      Observed-data file (req. fm_native)\n\
\n\
 Every inversion option is -im_* (matching this kernel and the\n\
 `petgem im` subcommand). \n\
\n\
EXAMPLES\n\
  mpirun -n 56 ./im.csem -options_file params_p1.txt -output_dir out/\n\
  mpirun -n 56 ./im.csem -options_file params.txt -im_max_iter 120 -im_lambda 0.05\n\
\n\
UNIFIED BINARY\n\
  mpirun -n 56 ./petgem im -options_file params_p1.txt\n\
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
#include "common.h"
#include "constants.h"
#include "grid.h"
#include "inversion.h"
#include "io.h"
#include "kernels.h"
#include "version.h"

/* Extrae library for performance analysis */
#ifdef USE_EXTRAE
#include "extrae_user_events.h"
#endif

/**
 * @brief Main execution routine for the CSEM inverse-modeling kernel.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return int 0 on success, non-zero on failure.
 * @details Initializes PETSc, parses command-line arguments
 * (including `--version`).  Reads the unified bundle (mesh +
 * conductivity + materials_id + receivers + /sources +
 * /observed/Ex + case-property defaults), sets up the H(curl)
 * grid, the inversion DM (1 DOF/cell), the smoother neighbor
 * graph, the receiver-interpolation matrices, and the cached
 * K / Ms-template / G_BDDC matrices.  Runs custom L-BFGS
 * (Nocedal 1980) with an adjoint-state gradient until the
 * configured RMS-tolerance or max-iter is hit.  Writes the
 * recovered conductivity, the log-perturbation X, and the
 * RMS history to HDF5, then finalizes PETSc.  Includes Extrae
 * instrumentation hooks if compiled with USE_EXTRAE.
 */
int runInverse(int argc, char **argv) {

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
  imParams        iparams;   /* embeds the shared petgemParams base as iparams.common */
  Grid            grid;
  PetscLogDouble  tAssembly = 0.0, tSolver = 0.0;
  PetscLogDouble  timers[6];
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

  PetscCall(PetscInitialize(&argc, &argv, (char *)0, imHelp));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* ---------------------------------------------------------------- */
  /* Resolve here the petsc help printing                             */
  /* ---------------------------------------------------------------- */
  PetscCall(PetscOptionsHasHelp(NULL, &helpRequested));

  /* ---------------------------------------------------------------- */
  /* Register the im.csem phase stages with PETSc's logging system    */
  /* ---------------------------------------------------------------- */
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

  if (!helpRequested) {
    PetscCall(printHeader());
  }

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
  PetscCall(readPetgemParams(size, &iparams.common));
  PetscCall(readimParams(&iparams));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[0] = end_timer - start_timer;

  /* Option groups already printed by -help; skip inversion and exit. */
  if (helpRequested) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nTip: 'im.csem -help intro' shows the concise usage summary.\n"));
    PetscCall(PetscFinalize());
    return 0;
  }

  /* Pull case-property defaults out of the bundle (error_level, fixed_materials) - CLI overrides applied by readimParams
   * already take precedence via the *FromCLI provenance flags. The bundle path is the shared base's input file 
   * (iparams.common.inputFile). Runs after the -help exit above (it opens the bundle) and folds into the same
   * "Read parameters" bucket. */
  PetscCall(PetscLogStagePush(stage_parse));
  PetscCall(PetscTime(&start_timer));
  PetscCall(loadInversionMetaFromBundle(iparams.common.inputFile, &iparams));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[0] += end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ----------------------------------------------------------------------- */
  /* Load input data: mesh, conductivity, materials_id, sources, receivers   */
  /* ----------------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 4);
#endif

  PetscCall(PetscLogStagePush(stage_load));
  PetscCall(PetscTime(&start_timer));
  PetscCall(loadCsemInputs(&iparams.common, &dm, &conductivity, &materials_id,
                           NULL,           /* no forward CsemSourceSet; setupInversionSources reads /sources */
                           &receivers));
  PetscCall(setupInversionSources(iparams.common.inputFile, &iparams));
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
  PetscCall(setupCsemGrid(iparams.common, &dm, &grid));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[2] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Run inversion                                                    */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 7);
#endif

  PetscCall(PetscLogStagePush(stage_solve));
  PetscCall(PetscTime(&start_timer));
  PetscCall(runCsemInversion(&iparams, dm, &grid, conductivity, materials_id, receivers, &tAssembly, &tSolver));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* Split the inversion wall time across the same six buckets fm.csem uses, so printTimers labels them consistently 
   * for both kernels. tAssembly and tSolver are accumulated inside the L-BFGS loop (Ms refill + A build, and
   * factorize + forward/adjoint solves); the remainder (gradient, smoothing, line search, results I/O, 
   * L-BFGS overhead) lands in the last bucket. */
  timers[3] = tAssembly;
  timers[4] = tSolver;
  timers[5] = (end_timer - start_timer) - tAssembly - tSolver;
  if (timers[5] < 0.0) timers[5] = 0.0;

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
  PetscCall(DMDestroy(&grid.H1dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&conductivity));
  PetscCall(VecDestroy(&materials_id));
  PetscCall(VecDestroy(&receivers));

  /* ---------------------------------------------------------------- */
  /* PETSc finalize                                                   */
  /* ---------------------------------------------------------------- */
  PetscCall(PetscFinalize());
  return 0;
}
