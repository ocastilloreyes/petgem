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
 an adjoint-state gradient (PETSc, Nedelec FEM).\n\
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
  -inv_max_iter <n>          Max L-BFGS iterations            [80]\n\
  -inv_lbfgs_memory <m>      L-BFGS history depth M            [5]\n\
  -inv_lambda <r>            Tikhonov regularisation weight  [0.1]\n\
  -inv_gtol <r>              Gradient-norm tolerance        [1e-5]\n\
  -inv_rms_tol <r>           Absolute RMS early stop (0=off)   [0]\n\
  -inv_error_level <r>       Relative data-error level      [0.01]\n\
  -inv_diag_weight <r>       Gradient-smoother self-weight     [0]\n\
  -inv_fixed_materials <ids> Material IDs frozen in smoothing\n\
  -inv_snapshot_interval <n> Write VTU every N steps (0=off)   [0]\n\
  -inv_observed_mode <m>     external | fm_native     [external]\n\
  -inv_observed_file <f>     Observed-data file (req. fm_native)\n\
  -order <1..6>               Override bundle basis order.\n\
\n\
EXAMPLES\n\
  mpirun -n 56 ./im.csem -options_file params_p1.txt -output_dir out/\n\
  mpirun -n 56 ./im.csem -options_file params.txt -inv_max_iter 120 -inv_lambda 0.05\n\
\n\
UNIFIED BINARY\n\
  mpirun -n 56 ./petgem inverse -options_file params_p1.txt\n\
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

/* PETSc functions */
#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>

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
int runInverse(int argc, char **argv)
{
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
  Vec             resistivity, materials_id, receivers;
  imParams        iparams;   /* embeds the shared fmParams base as iparams.fm */
  Grid            grid;
  PetscLogDouble  timers[7];
  PetscLogDouble  start_timer, end_timer;
  PetscBool       helpRequested = PETSC_FALSE;

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

  /* Resolved once here so the boxed run header is skipped on a -help run
   * (it would otherwise interrupt the option listing) and the clean exit
   * below can reuse it. */
  PetscCall(PetscOptionsHasHelp(NULL, &helpRequested));

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Print PETGEM header                                              */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 2);
#endif

  if (!helpRequested) PetscCall(printHeader());

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Parse user parameters                                            */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 3);
#endif

  PetscLogStage stage_parse, stage_load, stage_grid;
  PetscLogStage stage_assembly, stage_solve, stage_postproc;
  PetscCall(PetscLogStageRegister("Read parameters",       &stage_parse));
  PetscCall(PetscLogStageRegister("Load input bundle",     &stage_load));
  PetscCall(PetscLogStageRegister("Setup grid",            &stage_grid));
  PetscCall(PetscLogStageRegister("Assembly",              &stage_assembly));
  PetscCall(PetscLogStageRegister("Linear solve",          &stage_solve));
  PetscCall(PetscLogStageRegister("Field interpolation",   &stage_postproc));

  PetscCall(PetscLogStagePush(stage_parse));
  PetscCall(PetscTime(&start_timer));
  PetscCall(readfmParams(size, &iparams.fm));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[0] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Parse inversion parameters                                        */
  /* ---------------------------------------------------------------- */
  PetscCall(readInversionParams(&iparams));

  /* A plain -help run has printed the required/optional/inversion option
   * groups during parsing; exit cleanly here instead of running an
   * inversion. (-help intro already exited inside PetscInitialize.) */
  if (helpRequested) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "\nTip: 'im.csem -help intro' shows the concise usage summary.\n"));
    PetscCall(PetscFinalize());
    return 0;
  }

  /* Pull case-property defaults out of the bundle (error_level,
   * fixed_materials) - CLI overrides applied by readInversionParams
   * already take precedence via the *FromCLI provenance flags. The bundle
   * path is the shared base's input file (iparams.fm.inputFile). */
  PetscCall(loadInversionMetaFromBundle(iparams.fm.inputFile, &iparams));

  /* ---------------------------------------------------------------- */
  /* Load unified PETGEM input: mesh + sigma + materials_id +          */
  /* receivers from the bundle. The bundle's /sources group and        */
  /* observed Ex dataset are read below via setupInversionSources and  */
  /* (later, once numReceivers is known) loadObservedData inside       */
  /* runCsemInversion. loadCsemInputs is passed NULL for its forward   */
  /* CsemSourceSet - the inverse kernel reads the same /sources group  */
  /* into its multi-frequency invSources[] via setupInversionSources.  */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 5);
#endif

  PetscCall(PetscLogStagePush(stage_load));
  PetscCall(PetscTime(&start_timer));
  PetscCall(loadCsemInputs(&iparams.fm, &dm, &resistivity, &materials_id,
                           NULL,           /* no forward CsemSourceSet; setupInversionSources reads /sources */
                           &receivers));
  /* Basis order is now single-source: loadCsemInputs fills iparams.fm.order
   * from the bundle's /order when the user did not pass -order (sentinel 0). */
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[2] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Load multi-frequency inversion sources from bundle /sources       */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 4);
#endif

  PetscCall(PetscLogStagePush(stage_load));
  PetscCall(PetscTime(&start_timer));
  PetscCall(setupInversionSources(iparams.fm.inputFile, &iparams));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[1] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Setup grid for FE computations                                    */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 6);
#endif

  PetscCall(PetscLogStagePush(stage_grid));
  PetscCall(PetscTime(&start_timer));
  PetscCall(setupCsemGrid(iparams.fm, &dm, &grid));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[3] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Run inversion                                                     */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 7);
#endif

  PetscLogDouble tAssembly = 0.0, tSolver = 0.0;
  /* The inversion driver internally pushes its own Assembly + Solve sub-
   * stages via runCsemInversion's PetscLogEventBegin/End on every L-BFGS
   * callback; here we wrap the whole driver as the "Linear solve" outer
   * stage so -log_view's hierarchy mirrors what the user sees in the
   * printTimers table (timers[4]/[5] are populated from tAssembly/tSolver). */
  PetscCall(PetscLogStagePush(stage_solve));
  PetscCall(PetscTime(&start_timer));
  PetscCall(runCsemInversion(&iparams,
                              dm, &grid, resistivity, materials_id,
                              receivers, &tAssembly, &tSolver));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* Split the inversion wall time across the same buckets fm.csem uses, so
   * the two kernels' timer reports are consistent. tAssembly and tSolver are
   * accumulated inside the L-BFGS loop (Ms refill + A build, and factorize +
   * forward/adjoint solves); the remainder (gradient, smoothing, line search,
   * results I/O, L-BFGS overhead) lands in the "Postprocessing" bucket. */
  timers[4] = tAssembly;
  timers[5] = tSolver;
  timers[6] = (end_timer - start_timer) - tAssembly - tSolver;
  if (timers[6] < 0.0) timers[6] = 0.0;

  /* ---------------------------------------------------------------- */
  /* Print timers and footer                                           */
  /* ---------------------------------------------------------------- */
  PetscCall(printTimers(timers));
  PetscCall(printFooter());

  /* ---------------------------------------------------------------- */
  /* Free memory                                                       */
  /* ---------------------------------------------------------------- */
  PetscCall(DMDestroy(&grid.H1dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&resistivity));
  PetscCall(VecDestroy(&materials_id));
  PetscCall(VecDestroy(&receivers));

  /* ---------------------------------------------------------------- */
  /* PETSc finalize                                                   */
  /* ---------------------------------------------------------------- */
  PetscCall(PetscFinalize());
  return 0;
}
