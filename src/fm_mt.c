/*
 * Filename: fm_mt.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * Forward MT kernel (runMtForward), linked into fm.mt and petgem.
 */

/* PETGEM functions */
#include "assembly.h"
#include "common.h"
#include "constants.h"
#include "grid.h"
#include "io.h"
#include "kernels.h"
#include "mt.h"
#include "runtime.h"
#include "solver.h"

static char mtHelp[] = "================================================================\n\
 PETGEM  fm.mt  --  MT forward modeling kernel\n\
================================================================\n\
\n\
 Computes the 3-D magnetotelluric response on a tetrahedral\n\
 mesh using high-order (1..6) Nedelec finite elements.\n\
\n\
QUICK START\n\
  mpirun -n <np> ./fm.mt -options_file params.txt\n\
\n\
REQUIRED\n\
  -input_filename <file.h5>  Unified PETGEM bundle (mesh, model,\n\
                             frequency, receivers).\n\
  -output_dir <dir>          Output directory (created if absent).\n\
  -output_filename <stem>    Output stem; writes <dir>/<stem>.h5.\n\
\n\
OPTIONAL  (defaults in brackets)\n\
  -order <1..6>              Override bundle basis order.\n\
  -mt_1d_refine <n>          1D element size = 3D edge / n    [10]\n\
  -mt_1d_equation <e>        paper | h                    [paper]\n\
\n\
EXAMPLES\n\
  mpirun -n 96 ./fm.mt -options_file params.txt\n\
\n\
UNIFIED BINARY\n\
  mpirun -n 96 ./petgem fm.mt -options_file params.txt\n\
\n" PETGEM_HELP_FOOTER;

int runMtForward(int argc, char **argv) {

  if (isVersionRequest(argc, argv)) return printVersion();

  /* Variables declaration */
  KernelRun    run;
  DM           dm;
  Vec          conductivity, materialsId, receivers;
  Mat          A, B, X, G;
  IS           faces;
  MtBoxFace   *tags;
  Mt1DProfile  profile;
  Mt1DField    field;
  PetgemParams params;
  MtParams     mtParams;
  Grid         grid;
  PetscScalar  constFactor;
  SolveInfo    solveInfo;
  MPI_Comm     comm;

  PetscFunctionBeginUser;
  PetscCall(beginKernelRun(&argc, &argv, mtHelp, "fm.mt", "MT responses", &run));
  comm = PETSC_COMM_WORLD;

  /* Parse user parameters */
  PetscCall(beginPhase(&run, PHASE_PARSE));
  PetscCall(readPetgemParams(&params));
  PetscCall(readMtParams(&mtParams));
  PetscCall(endPhase(&run, PHASE_PARSE));
  if (run.helpRequested) {
    PetscCall(endHelpRun(&run));
    return 0;
  }

  /* Load mesh, conductivity, materials, receivers and frequency */
  PetscCall(beginPhase(&run, PHASE_LOAD));
  PetscCall(loadModelInputs(&params, &dm, &conductivity, &materialsId, &receivers));
  PetscCall(loadMtSettings(&params, &mtParams));
  PetscCall(endPhase(&run, PHASE_LOAD));

  /* Setup grid and boundary faces */
  PetscCall(beginPhase(&run, PHASE_GRID));
  PetscCall(setupNedelecGrid(params.order, PETGEM_BC_NATURAL, &dm, &grid));
  PetscCall(getBoundaryFaces(dm, &faces));
  PetscCall(classifyMtBoxFaces(dm, faces, &tags));
  PetscCall(endPhase(&run, PHASE_GRID));
  PetscCall(logGridSummary(params.inputFile, dm, &grid));

  /* 1D boundary field, boundary right-hand side and operator */
  constFactor = iOmegaMu(mtParams.frequency);
  PetscCall(beginPhase(&run, PHASE_ASSEMBLY));
  PetscCall(buildMt1DProfile(dm, conductivity, faces, tags, &profile));
  PetscCall(solveMt1D(&profile, &mtParams, &field));
  PetscCall(assembleMtBoundaryRhs(mtParams.frequency, dm, &grid, faces, tags, &field, &B));
  PetscCall(assembleMaxwellOperator(dm, &grid, conductivity, constFactor, &A, NULL, &G));
  PetscCall(endPhase(&run, PHASE_ASSEMBLY));
  PetscCall(logMtSetup(comm, &mtParams, faces, &profile, &field));
  PetscCall(logAssembly(comm, A, MT_NUM_POLARIZATIONS));

  /* Solve */
  PetscCall(beginPhase(&run, PHASE_SOLVE));
  PetscCall(solveMaxwellSystem(dm, A, B, G, grid.fem.order, NULL, &X, &solveInfo));
  PetscCall(endPhase(&run, PHASE_SOLVE));
  PetscCall(logSolve(comm, &solveInfo));

  /* Impedance, apparent resistivity, phase and tipper at the receivers */
  PetscCall(beginPhase(&run, PHASE_OUTPUT));
  PetscCall(computeMtResponses(&params, &mtParams, dm, &grid, receivers, X));
  PetscCall(endPhase(&run, PHASE_OUTPUT));
  PetscCall(logReceiverOutput(comm, run.outputStage, &params, receivers));
  PetscCall(printForwardSummary(&run));

  /* Free memory */
  PetscCall(destroyMt1DField(&field));
  PetscCall(destroyMt1DProfile(&profile));
  PetscCall(PetscFree(tags));
  PetscCall(ISDestroy(&faces));
  PetscCall(destroyGrid(&grid));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&conductivity));
  PetscCall(VecDestroy(&materialsId));
  PetscCall(MatDestroy(&G));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&X));
  PetscCall(VecDestroy(&receivers));

  PetscCall(PetscFinalize());
  return 0;
}
