/*
 * Filename: fm_csem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Forward CSEM kernel (runForward), linked into fm.csem and petgem.
 */

/* PETGEM functions */
#include "assembly.h"
#include "common.h"
#include "constants.h"
#include "csem.h"
#include "csem_primary.h"
#include "grid.h"
#include "io.h"
#include "kernels.h"
#include "runtime.h"
#include "solver.h"

static char fmHelp[] = "================================================================\n\
 PETGEM  fm.csem  --  CSEM forward modeling kernel\n\
================================================================\n\
\n\
 Computes the 3-D controlled-source EM response on a tetrahedral\n\
 mesh using high-order (1..6) Nedelec finite elements.\n\
\n\
QUICK START\n\
  mpirun -n <np> ./fm.csem -options_file params.txt\n\
\n\
REQUIRED\n\
  -input_filename <file.h5>  Unified PETGEM bundle (mesh, model,\n\
                             sources, receivers).\n\
  -output_dir <dir>          Output directory (created if absent).\n\
  -output_filename <stem>    Output stem; writes <dir>/<stem>.h5.\n\
\n\
OPTIONAL  (defaults in brackets)\n\
  -order <1..6>              Override bundle basis order.\n\
\n\
FIELD FORMULATION  (defaults in brackets)\n\
  -csem_formulation <f>      total | secondary            [total]\n\
  -csem_primary <p>          Background of the secondary field:\n\
                             layered | model  [layered if the\n\
                             background has layers, else model]\n\
  -csem_background_file <f>  HDF5 file with the /background group\n\
                             [the input bundle]\n\
\n\
EXAMPLES\n\
  mpirun -n 96 ./fm.csem -options_file params.txt\n\
\n\
UNIFIED BINARY\n\
  mpirun -n 96 ./petgem fm.csem -options_file params.txt\n\
\n" PETGEM_HELP_FOOTER;

/* Context of addPrimaryField. */
typedef struct {
  const CsemPrimary *primary;
  DM                 dm;
} PrimaryFieldCtx;

/* Adds the primary field of source k of the single source set to the receiver fields. */
static PetscErrorCode addPrimaryField(PetscInt k, const ReceiverInterpolationMatrices *Q, Vec receivers, Vec fields[NUM_RECEIVER_FIELDS],
                                      void *ctx) {
  PetscFunctionBeginUser;
  const PrimaryFieldCtx *c = (const PrimaryFieldCtx *)ctx;
  PetscCall(addCsemPrimaryAtReceivers(c->primary, 0, k, c->dm, Q, receivers, fields));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int runForward(int argc, char **argv) {

  if (isVersionRequest(argc, argv)) return printVersion();

  /* Variables declaration */
  KernelRun      run;
  DM             dm;
  Vec            conductivity, materialsId, receivers;
  PetgemParams   params;
  CsemParams     csemParams;
  CsemBackground background;
  CsemPrimary    primary;
  PetscBool      secondary;
  Grid           grid;
  CsemSourceSet  sources = {0, 0, NULL};
  Mat            A, B, X, G;
  IS             primalVertices = NULL;
  PetscScalar    constFactor;
  SolveInfo      solveInfo;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  PetscCall(beginKernelRun(&argc, &argv, fmHelp, "fm.csem", "Field interpolation", &run));
  comm = PETSC_COMM_WORLD;

  /* Parse user parameters */
  PetscCall(beginPhase(&run, PHASE_PARSE));
  PetscCall(readPetgemParams(&params));
  PetscCall(readCsemParams(&params, &csemParams));
  PetscCall(endPhase(&run, PHASE_PARSE));
  if (run.helpRequested) {
    PetscCall(endHelpRun(&run));
    return 0;
  }

  /* Load mesh, conductivity, materials, receivers and sources */
  PetscCall(beginPhase(&run, PHASE_LOAD));
  PetscCall(loadModelInputs(&params, &dm, &conductivity, &materialsId, &receivers));
  PetscCall(loadCsemSources(&params, &sources));
  secondary = (csemParams.formulation == CSEM_FORMULATION_SECONDARY) ? PETSC_TRUE : PETSC_FALSE;
  if (secondary) PetscCall(loadCsemBackground(&csemParams, &background));
  PetscCall(endPhase(&run, PHASE_LOAD));

  /* Setup grid */
  PetscCall(beginPhase(&run, PHASE_GRID));
  PetscCall(setupCsemGrid(params.order, &dm, &grid));
  PetscCall(endPhase(&run, PHASE_GRID));
  PetscCall(logGridSummary(params.inputFile, dm, &grid));
  PetscCall(logForwardSources(&sources));

  /* Assembly */
  constFactor = iOmegaMu(sources.freq);
  PetscCall(beginPhase(&run, PHASE_ASSEMBLY));
  if (secondary) {
    Vec b;
    PetscCall(createCsemPrimary(&background, dm, &grid, conductivity, materialsId, &primary));
    PetscCall(computeCsemPrimary(1, &sources, dm, &grid, NULL, &primary));
    PetscCall(createRhsVector(dm, &b));
    PetscCall(createRhsMatrix(b, sources.numSources, &B));
    PetscCall(VecDestroy(&b));
    PetscCall(assembleCsemSecondaryRhs(&primary, dm, &grid, conductivity, &B));
  } else
    PetscCall(assembleCsemRhs(sources, dm, &grid, constFactor, &B));
  PetscCall(assembleMaxwellOperator(dm, &grid, conductivity, constFactor, &A, NULL, &G));
  PetscCall(endPhase(&run, PHASE_ASSEMBLY));
  PetscCall(logAssembly(comm, A, sources.numSources));
  if (secondary) PetscCall(logCsemFormulation(comm, &primary));

  /* Solve */
  PetscCall(beginPhase(&run, PHASE_SOLVE));
  PetscCall(setupPrimalVerticesFromOptions(dm, A, conductivity, &primalVertices));
  PetscCall(solveMaxwellSystem(dm, A, B, G, grid.fem.order, primalVertices, &X, &solveInfo));
  PetscCall(endPhase(&run, PHASE_SOLVE));
  PetscCall(logSolve(comm, &solveInfo));

  /* Fields at the receivers */
  PetscCall(beginPhase(&run, PHASE_OUTPUT));
  {
    PrimaryFieldCtx ctx = {&primary, dm};
    PetscCall(computeCsemResponses(&params, sources, dm, &grid, receivers, X, secondary ? addPrimaryField : NULL, &ctx));
  }
  PetscCall(endPhase(&run, PHASE_OUTPUT));
  PetscCall(logReceiverOutput(comm, run.outputStage, &params, receivers));
  PetscCall(printForwardSummary(&run));

  /* Free memory */
  if (secondary) {
    PetscCall(destroyCsemPrimary(&primary));
    PetscCall(destroyCsemBackground(&background));
  }
  PetscCall(ISDestroy(&primalVertices));
  PetscCall(MatDestroy(&G));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&X));
  PetscCall(destroyGrid(&grid));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&conductivity));
  PetscCall(VecDestroy(&materialsId));
  PetscCall(VecDestroy(&receivers));
  PetscCall(PetscFree(sources.sourceArray));

  PetscCall(PetscFinalize());
  return 0;
}
