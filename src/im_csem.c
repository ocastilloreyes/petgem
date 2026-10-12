/*
 * Filename: im_csem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Inverse CSEM kernel (runInverse), linked into im.csem and petgem.
 */

/* PETGEM functions */
#include "common.h"
#include "constants.h"
#include "grid.h"
#include "inversion.h"
#include "io.h"
#include "kernels.h"
#include "runtime.h"

static char imHelp[] = "================================================================\n\
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
  -im_lambda <r>             Tikhonov regularization weight  [0.1]\n\
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
 `petgem im.csem` subcommand).\n\
\n\
EXAMPLES\n\
  mpirun -n 56 ./im.csem -options_file params_p1.txt -output_dir out/\n\
  mpirun -n 56 ./im.csem -options_file params.txt -im_max_iter 120 -im_lambda 0.05\n\
\n\
UNIFIED BINARY\n\
  mpirun -n 56 ./petgem im.csem -options_file params_p1.txt\n\
\n" PETGEM_HELP_FOOTER;

int runInverse(int argc, char **argv) {

  if (isVersionRequest(argc, argv)) return printVersion();

  /* Variables declaration */
  KernelRun      run;
  DM             dm;
  Vec            conductivity, materialsId, receivers;
  ImParams       imParams; /* imParams.common holds the parameters shared with fm.csem */
  Grid           grid;
  InversionStats stats;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  PetscCall(beginKernelRun(&argc, &argv, imHelp, "im.csem", "Field interpolation", &run));
  comm = PETSC_COMM_WORLD;

  /* Parse user parameters */
  PetscCall(beginPhase(&run, PHASE_PARSE));
  PetscCall(readPetgemParams(&imParams.common));
  PetscCall(readImParams(&imParams));
  PetscCall(endPhase(&run, PHASE_PARSE));
  if (run.helpRequested) {
    PetscCall(endHelpRun(&run));
    return 0;
  }

  /* Case defaults stored in the bundle (error level, fixed materials); CLI values win */
  PetscCall(beginPhase(&run, PHASE_PARSE));
  PetscCall(loadInversionMetaFromBundle(imParams.common.inputFile, &imParams));
  PetscCall(endPhase(&run, PHASE_PARSE));

  /* Load mesh, conductivity, materials, receivers and sources */
  PetscCall(beginPhase(&run, PHASE_LOAD));
  PetscCall(loadModelInputs(&imParams.common, &dm, &conductivity, &materialsId, &receivers));
  PetscCall(setupInversionSources(imParams.common.inputFile, &imParams));
  PetscCall(endPhase(&run, PHASE_LOAD));

  /* Setup grid */
  PetscCall(beginPhase(&run, PHASE_GRID));
  PetscCall(setupCsemGrid(imParams.common.order, &dm, &grid));
  PetscCall(endPhase(&run, PHASE_GRID));
  PetscCall(logGridSummary(imParams.common.inputFile, dm, &grid));
  PetscCall(logInversionSummary(&imParams));

  /* L-BFGS inversion */
  PetscCall(beginPhase(&run, PHASE_SOLVE));
  PetscCall(runCsemInversion(&imParams, dm, &grid, conductivity, materialsId, receivers, &stats));
  PetscCall(endPhase(&run, PHASE_SOLVE));

  {
    char outFile[PETSC_MAX_PATH_LEN];
    PetscCall(buildOutputPath(&imParams.common, ".h5", outFile, sizeof(outFile)));
    PetscCall(logSection(comm, "Solve"));
    PetscCall(logKvStr(comm, "Solver", stats.solver));
    PetscCall(logKvf(comm, "Linear solves", "%s (forward + adjoint)", formatGroupedInt(stats.numSolves)));
    PetscCall(logKvStr(comm, "Status", stats.failReason == 0 ? "converged" : KSPConvergedReasons[stats.failReason]));
    PetscCall(logSection(comm, "Inversion output"));
    PetscCall(logKvStr(comm, "Output file", outFile));
    PetscCall(logKvStr(comm, "Datasets written", "conductivity, log_perturbation, rms_history"));
    PetscCall(logKvInt(comm, "VTU snapshots", stats.numSnapshots));
  }
  {
    const char *const    labels[] = {"Load + grid", "Setup", "Assembly", "Linear solve", "Gradient + L-BFGS", "Output"};
    const PetscLogDouble times[]  = {setupTime(&run), stats.tSetup, stats.tAssembly, stats.tSolver, stats.tGradient, stats.tOutput};
    PetscCall(printRunSummary(labels, times, 6));
  }

  /* Free memory */
  PetscCall(destroyGrid(&grid));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&conductivity));
  PetscCall(VecDestroy(&materialsId));
  PetscCall(VecDestroy(&receivers));

  PetscCall(PetscFinalize());
  return 0;
}
