/*
 * Filename: fm_tdcsem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-12
 *
 * Description:
 * Time-domain forward CSEM kernel (runTdcsemForward), linked into fm.tdcsem
 * and petgem.
 */

/* PETGEM functions */
#include "common.h"
#include "constants.h"
#include "csem.h"
#include "grid.h"
#include "io.h"
#include "kernels.h"
#include "runtime.h"
#include "solver.h"
#include "tdcsem.h"

static char tdHelp[] = "================================================================\n\
 PETGEM  fm.tdcsem  --  time-domain CSEM forward modeling kernel\n\
================================================================\n\
\n\
 Computes the 3-D transient controlled-source EM response on a\n\
 tetrahedral mesh using high-order (1..6) Nedelec finite elements\n\
 and BDF2 or backward Euler time stepping with adaptive step\n\
 doubling.\n\
\n\
QUICK START\n\
  mpirun -n <np> ./fm.tdcsem -options_file params.txt\n\
\n\
REQUIRED\n\
  -input_filename <file.h5>  Unified PETGEM bundle (mesh, model,\n\
                             sources, waveform, receivers).\n\
  -output_dir <dir>          Output directory (created if absent).\n\
  -output_filename <stem>    Output stem; writes <dir>/<stem>.h5.\n\
  -tdcsem_t_end <s>          Final time.\n\
\n\
OPTIONAL  (defaults in brackets)\n\
  -order <1..6>              Override bundle basis order.\n\
  -tdcsem_dt0 <s>            Initial time step  [smallest waveform\n\
                             sample spacing]\n\
  -tdcsem_atsd_steps <n>     Steps before a doubling trial    [50]\n\
  -tdcsem_atsd_rtol <r>      Doubling tolerance, 0 = fixed  [1e-2]\n\
  -tdcsem_scheme <s>         bdf2 | be                    [bdf2]\n\
  -tdcsem_dc_ksp_*, -tdcsem_dc_pc_*\n\
                             DC field solver (step-off)\n\
                             [preonly, cholesky, mumps]\n\
\n\
EXAMPLES\n\
  mpirun -n 96 ./fm.tdcsem -options_file params.txt\n\
\n\
UNIFIED BINARY\n\
  mpirun -n 96 ./petgem fm.tdcsem -options_file params.txt\n\
\n" PETGEM_HELP_FOOTER;

/**
 * @brief Runs the time-domain CSEM forward-modeling kernel (fm.tdcsem).
 */
int runTdcsemForward(int argc, char **argv) {

  if (isVersionRequest(argc, argv)) return printVersion();

  /* Variables declaration */
  KernelRun     run;
  DM            dm;
  Vec           conductivity, materialsId, receivers;
  PetgemParams  params;
  TdParams      td = {0};
  Grid          grid;
  CsemSourceSet sources = {0, 0, NULL};
  Mat           K, Ms, G, B0, E, E1, E2, Enext, Etrial, R, swap;
  IS            primalVertices = NULL;
  TdStepper     current = {0}, trial = {0};
  TdCharge      charge  = {0};
  TdStats       stats   = {0};
  TdMonitor     monitor = {0};
  TdOutput      out;
  PetscReal     t, tSource;
  MPI_Comm      comm;

  PetscFunctionBeginUser;
  PetscCall(beginKernelRun(&argc, &argv, tdHelp, "fm.tdcsem", "Field interpolation", &run));
  comm = PETSC_COMM_WORLD;

  /* Parse user parameters */
  PetscCall(beginPhase(&run, PHASE_PARSE));
  PetscCall(readPetgemParams(&params));
  PetscCall(readTdParams(&td));
  PetscCall(endPhase(&run, PHASE_PARSE));
  if (run.helpRequested) {
    PetscCall(endHelpRun(&run));
    return 0;
  }

  /* Load mesh, conductivity, materials, receivers, sources and waveform */
  PetscCall(beginPhase(&run, PHASE_LOAD));
  PetscCall(loadModelInputs(&params, &dm, &conductivity, &materialsId, &receivers));
  PetscCall(loadTdSettings(&params, &sources, &td));
  PetscCall(endPhase(&run, PHASE_LOAD));

  /* Setup grid */
  PetscCall(beginPhase(&run, PHASE_GRID));
  PetscCall(setupCsemGrid(params.order, &dm, &grid));
  PetscCall(endPhase(&run, PHASE_GRID));
  PetscCall(logGridSummary(params.inputFile, dm, &grid));
  PetscCall(logTdSetup(&sources, &td));

  /* Assembly: K, Ms, G and the dipole term B0 = -mu <N, p> */
  PetscCall(beginPhase(&run, PHASE_ASSEMBLY));
  PetscCall(assembleCsemRhs(sources, dm, &grid, -MU, &B0));
  PetscCall(assembleTdOperators(dm, &grid, conductivity, &K, &Ms, &G));
  PetscCall(MatDuplicate(B0, MAT_DO_NOT_COPY_VALUES, &E));
  PetscCall(MatDuplicate(B0, MAT_DO_NOT_COPY_VALUES, &Enext));
  PetscCall(MatDuplicate(B0, MAT_DO_NOT_COPY_VALUES, &Etrial));
  PetscCall(MatDuplicate(B0, MAT_DO_NOT_COPY_VALUES, &E1));
  PetscCall(MatDuplicate(B0, MAT_DO_NOT_COPY_VALUES, &E2));
  PetscCall(MatDuplicate(B0, MAT_DO_NOT_COPY_VALUES, &R));
  PetscCall(MatZeroEntries(E));
  PetscCall(MatZeroEntries(E1));
  PetscCall(MatZeroEntries(E2));
  PetscCall(endPhase(&run, PHASE_ASSEMBLY));
  PetscCall(logAssembly(comm, K, sources.numSources));

  /* Receiver interpolation */
  PetscCall(beginPhase(&run, PHASE_OUTPUT));
  PetscCall(setupTdOutput(sources, dm, &grid, receivers, &out));
  PetscCall(endPhase(&run, PHASE_OUTPUT));

  /* Time stepping with adaptive step doubling */
  PetscCall(beginPhase(&run, PHASE_SOLVE));
  PetscCall(setupPrimalVerticesFromOptions(dm, K, conductivity, &primalVertices));
  PetscCall(setupTdCharge(dm, &grid, Ms, G, (PetscBool)(td.waveform[0] != 0.0), &charge));
  PetscCall(createTdStepper(K, Ms, G, grid.fem.order, primalVertices, td.a[0], td.dt0, &current, &stats));
  t       = td.time[0];
  tSource = td.time[td.numSamples - 1];
  /* Step-off: start from the DC field of s(t0), steady for t < t0 */
  if (td.waveform[0] != 0.0) {
    PetscCall(computeTdDcField(&charge, Ms, B0, td.waveform[0], E));
    PetscCall(MatCopy(E, E1, SAME_NONZERO_PATTERN));
    PetscCall(MatCopy(E, E2, SAME_NONZERO_PATTERN));
  }
  PetscCall(recordTdSample(&out, E, t));
  PetscCall(beginTdMonitor(comm, &stats, &monitor));
  while (t < td.tEnd) {
    const PetscReal dt     = current.dt;
    PetscReal       ratio  = -1.0, res;
    const char     *result = "-";

    for (PetscInt i = 0; i < td.atsdSteps && t < td.tEnd; i++) {
      PetscCall(advanceTdStep(&current, Ms, B0, &td, t, E, E1, Enext, R, &stats));
      swap = E2, E2 = E1, E1 = E, E = Enext, Enext = swap;
      t += current.dt;
      stats.numSteps++;
      PetscCall(recordTdSample(&out, E, t));
    }

    /* Doubling trial: one 2dt step against two dt steps */
    if (td.atsdRtol > 0.0 && t >= tSource && t + 2.0 * current.dt <= td.tEnd) {
      PetscReal diff, norm;

      if (!trial.ksp) {
        PetscCall(createTdStepper(K, Ms, G, grid.fem.order, primalVertices, td.a[0], 2.0 * current.dt, &trial, &stats));
      } else if (trial.dt != 2.0 * current.dt) {
        PetscCall(refillTdStepper(K, Ms, td.a[0], 2.0 * current.dt, &trial, &stats));
      }
      PetscCall(advanceTdStep(&trial, Ms, B0, &td, t, E, E2, Etrial, R, &stats));
      for (PetscInt i = 0; i < 2; i++) {
        PetscCall(advanceTdStep(&current, Ms, B0, &td, t, E, E1, Enext, R, &stats));
        swap = E2, E2 = E1, E1 = E, E = Enext, Enext = swap;
        t += current.dt;
        stats.numSteps++;
        PetscCall(recordTdSample(&out, E, t));
      }
      PetscCall(MatAXPY(Etrial, -1.0, E, SAME_NONZERO_PATTERN));
      PetscCall(normTdCurl(K, Etrial, &diff));
      PetscCall(MatCopy(E, Enext, SAME_NONZERO_PATTERN));
      PetscCall(MatAXPY(Enext, -1.0, E2, SAME_NONZERO_PATTERN));
      PetscCall(normTdCurl(K, Enext, &norm));
      ratio = diff / norm;
      if (diff <= td.atsdRtol * norm) {
        TdStepper spare = current;

        /* The replaced operator is refilled as the next trial */
        current = trial;
        trial   = spare;
        swap = E1, E1 = E2, E2 = swap;
        stats.numDoubled++;
        result = "accepted";
      } else {
        stats.numRejected++;
        result = "rejected";
      }
    }
    PetscCall(computeTdChargeResidual(&charge, Ms, B0, evaluateWaveform(&td, t), E, &res));
    PetscCall(logTdStep(comm, &stats, t, dt, ratio, result, res, &monitor));
  }
  PetscCall(getSolverName(current.ksp, stats.solver, sizeof(stats.solver)));
  PetscCall(endPhase(&run, PHASE_SOLVE));
  PetscCall(logTdSolve(comm, &stats, &charge, current.dt, t));

  /* Time series at the receivers */
  PetscCall(beginPhase(&run, PHASE_OUTPUT));
  PetscCall(writeTdOutput(&params, sources, &out));
  PetscCall(endPhase(&run, PHASE_OUTPUT));
  PetscCall(logReceiverOutput(comm, run.outputStage, &params, receivers));
  PetscCall(printForwardSummary(&run));

  /* Free memory */
  PetscCall(destroyTdOutput(&out));
  PetscCall(destroyTdStepper(&current));
  PetscCall(destroyTdStepper(&trial));
  PetscCall(destroyTdCharge(&charge));
  PetscCall(ISDestroy(&primalVertices));
  PetscCall(MatDestroy(&K));
  PetscCall(MatDestroy(&Ms));
  PetscCall(MatDestroy(&G));
  PetscCall(MatDestroy(&B0));
  PetscCall(MatDestroy(&E));
  PetscCall(MatDestroy(&Enext));
  PetscCall(MatDestroy(&Etrial));
  PetscCall(MatDestroy(&E1));
  PetscCall(MatDestroy(&E2));
  PetscCall(MatDestroy(&R));
  PetscCall(destroyGrid(&grid));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&conductivity));
  PetscCall(VecDestroy(&materialsId));
  PetscCall(VecDestroy(&receivers));
  PetscCall(PetscFree(sources.sourceArray));
  PetscCall(destroyTdParams(&td));

  PetscCall(PetscFinalize());
  return 0;
}
