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
OPTIONAL\n\
  -order <1..6>              Override bundle basis order.\n\
\n\
CASED WELLS\n\
  -petgem_edge_conductance_file <txt>\n\
                             Conductance on mesh edges, one segment per\n\
                             row: x0 y0 z0 x1 y1 z1 sigmaA (m, S m).\n\
  -petgem_bddc_spd           BDDC on K + w mu sigma M with an adaptive\n\
                             coarse space and a gradient-space correction.\n\
  -petgem_bddc_spd_floor <S/m>\n\
                             Conductivity floor of the BDDC matrix (0.1).\n\
  -petgem_grad_<opt>         Solver of the gradient space (MUMPS LU).\n\
\n\
EXAMPLES\n\
  mpirun -n 96 ./fm.csem -options_file params.txt\n\
\n\
UNIFIED BINARY\n\
  mpirun -n 96 ./petgem fm.csem -options_file params.txt\n\
\n" PETGEM_HELP_FOOTER;

int runForward(int argc, char **argv) {

  if (isVersionRequest(argc, argv)) return printVersion();

  /* Variables declaration */
  KernelRun     run;
  DM            dm;
  Vec           conductivity, materialsId, receivers;
  PetgemParams  params;
  Grid          grid;
  CsemSourceSet sources = {0, 0, NULL};
  Mat           A, Abddc, B, X, G;
  IS            primalVertices = NULL;
  PetscScalar   constFactor;
  SolveInfo     solveInfo;
  MPI_Comm      comm;

  PetscFunctionBeginUser;
  PetscCall(beginKernelRun(&argc, &argv, fmHelp, "fm.csem", "Field interpolation", &run));
  comm = PETSC_COMM_WORLD;

  /* Parse user parameters */
  PetscCall(beginPhase(&run, PHASE_PARSE));
  PetscCall(readPetgemParams(&params));
  PetscCall(endPhase(&run, PHASE_PARSE));
  if (run.helpRequested) {
    PetscCall(endHelpRun(&run));
    return 0;
  }

  /* Load mesh, conductivity, materials, receivers and sources */
  PetscCall(beginPhase(&run, PHASE_LOAD));
  PetscCall(loadModelInputs(&params, &dm, &conductivity, &materialsId, &receivers));
  PetscCall(loadCsemSources(&params, &sources));
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
  PetscCall(assembleCsemRhs(sources, dm, &grid, constFactor, &B));
  PetscCall(assembleMaxwellOperator(dm, &grid, conductivity, constFactor, &A, NULL, &G));
  PetscCall(addEdgeConductanceFromOptions(dm, &grid, constFactor, A));
  PetscCall(assembleBddcOperatorFromOptions(dm, &grid, conductivity, constFactor, A, &Abddc));
  PetscCall(endPhase(&run, PHASE_ASSEMBLY));
  PetscCall(logAssembly(comm, A, sources.numSources));

  /* Solve */
  PetscCall(beginPhase(&run, PHASE_SOLVE));
  PetscCall(setupPrimalVerticesFromOptions(dm, A, conductivity, &primalVertices));
  PetscCall(solveMaxwellSystem(dm, A, Abddc, B, G, grid.fem.order, primalVertices, &X, &solveInfo));
  PetscCall(endPhase(&run, PHASE_SOLVE));
  PetscCall(logSolve(comm, &solveInfo));

  /* Fields at the receivers */
  PetscCall(beginPhase(&run, PHASE_OUTPUT));
  PetscCall(computeCsemResponses(&params, sources, dm, &grid, receivers, X));
  PetscCall(endPhase(&run, PHASE_OUTPUT));
  PetscCall(logReceiverOutput(comm, run.outputStage, &params, receivers));
  PetscCall(printForwardSummary(&run));

  /* Free memory */
  PetscCall(ISDestroy(&primalVertices));
  PetscCall(MatDestroy(&G));
  PetscCall(MatDestroy(&Abddc));
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
