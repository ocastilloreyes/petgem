/*
 * Filename: runtime.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * Run scaffolding shared by the PETGEM kernels: PETSc start-up, timed log
 * stages (with optional Extrae events) and the common run-report sections.
 */

/* C libraries */
#include <stdio.h>
#include <string.h>

/* PETGEM functions */
#include "common.h"
#include "runtime.h"
#include "version.h"

/* Extrae library for performance analysis */
#ifdef USE_EXTRAE
#include "extrae_user_events.h"
#endif

#define EXTRAE_EVENT_TYPE  1000
#define EXTRAE_INIT        1
#define EXTRAE_HEADER      2
#define EXTRAE_FIRST_PHASE 3
#define EXTRAE_SUMMARY     9

static void extraeEvent(int value) {
#ifdef USE_EXTRAE
  Extrae_event(EXTRAE_EVENT_TYPE, value);
#else
  (void)value;
#endif
}

PetscBool isVersionRequest(int argc, char **argv) {
  return (PetscBool)(argc > 1 && strcmp(argv[1], "--version") == 0);
}

int printVersion(void) {
  printf("PETGEM version %d.%d.%d\n", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
  return 0;
}

PetscErrorCode beginKernelRun(int *argc, char ***argv, const char help[], const char *name, const char *outputStage, KernelRun *run) {
  PetscFunctionBeginUser;

  const char *stageNames[NUM_PHASES] = {"Read parameters", "Load input bundle", "Setup grid", "Assembly", "Linear solve", outputStage};

  extraeEvent(EXTRAE_INIT);
  PetscCall(PetscInitialize(argc, argv, NULL, help));
  PetscCall(PetscMemzero(run, sizeof(*run)));
  run->name        = name;
  run->outputStage = outputStage;
  PetscCall(PetscOptionsHasHelp(NULL, &run->helpRequested));
  for (PetscInt p = 0; p < NUM_PHASES; p++) PetscCall(PetscLogStageRegister(stageNames[p], &run->stage[p]));
  extraeEvent(0);

  extraeEvent(EXTRAE_HEADER);
  if (!run->helpRequested) PetscCall(printHeader(name));
  extraeEvent(0);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode beginPhase(KernelRun *run, KernelPhase phase) {
  PetscFunctionBeginUser;
  extraeEvent(EXTRAE_FIRST_PHASE + (int)phase);
  PetscCall(PetscLogStagePush(run->stage[phase]));
  PetscCall(PetscTime(&run->start));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode endPhase(KernelRun *run, KernelPhase phase) {
  PetscFunctionBeginUser;
  PetscLogDouble now;
  PetscCall(PetscTime(&now));
  run->elapsed[phase] += now - run->start;
  PetscCall(PetscLogStagePop());
  extraeEvent(0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode endHelpRun(const KernelRun *run) {
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nTip: '%s -help intro' shows the concise usage summary.\n", run->name));
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printRunSummary(const char *const labels[], const PetscLogDouble times[], PetscInt n) {
  PetscFunctionBeginUser;
  extraeEvent(EXTRAE_SUMMARY);
  PetscCall(printTimers(labels, times, n));
  PetscCall(printFooter());
  extraeEvent(0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscLogDouble setupTime(const KernelRun *run) {
  return run->elapsed[PHASE_PARSE] + run->elapsed[PHASE_LOAD] + run->elapsed[PHASE_GRID];
}

PetscErrorCode printForwardSummary(const KernelRun *run) {
  PetscFunctionBeginUser;
  const char *const    labels[] = {"Load + grid", "Assembly", "Linear solve", run->outputStage};
  const PetscLogDouble times[]  = {setupTime(run), run->elapsed[PHASE_ASSEMBLY], run->elapsed[PHASE_SOLVE], run->elapsed[PHASE_OUTPUT]};
  PetscCall(printRunSummary(labels, times, 4));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode logAssembly(MPI_Comm comm, Mat A, PetscInt numRhs) {
  PetscFunctionBeginUser;
  MatType type;
  PetscCall(MatGetType(A, &type));
  PetscCall(logSection(comm, "Assembly"));
  PetscCall(logKvStr(comm, "Matrix type", type));
  PetscCall(logKvInt(comm, "Right-hand sides", numRhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode logSolve(MPI_Comm comm, const SolveInfo *info) {
  PetscFunctionBeginUser;
  PetscCall(logSection(comm, "Solve"));
  PetscCall(logKvStr(comm, "Solver", info->solver));
  PetscCall(logKvStr(comm, "Status", info->reason > 0 ? "converged" : KSPConvergedReasons[info->reason]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode logReceiverOutput(MPI_Comm comm, const char *title, const PetgemParams *params, Vec receivers) {
  PetscFunctionBeginUser;
  PetscInt n;
  char     outFile[PETSC_MAX_PATH_LEN];
  PetscCall(VecGetSize(receivers, &n));
  PetscCall(buildOutputPath(params, ".h5", outFile, sizeof(outFile)));
  PetscCall(logSection(comm, title));
  PetscCall(logKvInt(comm, "Number of receivers", n / 3));
  PetscCall(logKvStr(comm, "Output file", outFile));
  PetscFunctionReturn(PETSC_SUCCESS);
}
