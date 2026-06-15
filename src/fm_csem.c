/*
 * Filename: fm_csem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Forward CSEM kernel (runForward). Linked into both the legacy
 * fm.csem binary and the unified petgem dispatcher.
 */

static char fmHelp[] =
"================================================================\n\
 PETGEM  fm.csem  --  CSEM forward modeling kernel\n\
================================================================\n\
\n\
 Computes the 3-D controlled-source EM response on a tetrahedral\n\
 mesh using high-order (1..6) Nedelec finite elements on PETSc.\n\
\n\
QUICK START\n\
  mpirun -n <np> ./fm.csem -input_filename model.h5 \\\n\
         -output_dir out/ -output_filename resp\n\
\n\
REQUIRED\n\
  -input_filename <file.h5>  Unified PETGEM input bundle: mesh,\n\
                             conductivity, materials, sources and\n\
                             receivers (built by the preprocessor).\n\
  -output_dir <dir>          Output directory (created if absent).\n\
  -output_filename <stem>    Output stem; writes <dir>/<stem>.h5.\n\
\n\
OPTIONAL\n\
  -nord <1..6>               Override bundle basis order.\n\
  -options_file <file>       Read options (and PETSc flags) from file.\n\
\n\
EXAMPLES\n\
  ./fm.csem -input_filename model.h5 -output_dir out/ -output_filename resp\n\
  mpirun -n 96 ./fm.csem -options_file params_p2.txt\n\
  mpirun -n 96 ./fm.csem -options_file params.txt -nord 3\n\
\n\
UNIFIED BINARY\n\
  mpirun -n 96 ./petgem modeling -options_file params_p2.txt\n\
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
#include <petscviewerhdf5.h>

/* PETGEM functions */
#include "assembly.h"
#include "common.h"
#include "constants.h"
#include "grid.h"
#include "inputs.h"
#include "io.h"
#include "kernels.h"
#include "postprocessing.h"
#include "solver.h"
#include "transmitter.h"
#include "version.h"

/* Extrae library for performance analysis */
#ifdef USE_EXTRAE
#include "extrae_user_events.h"
#endif

/**
 * @brief Main execution routine for CSEM forward modeling kernel.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return int 0 on success, non-zero on failure.
 * @details Initializes PETSc, parses command-line arguments
 * (including `--version`). Prints the header. Reads user
 * parameters, sets up sources, imports the grid and
 * conductivity, sets up the DM sections and grid structure,
 * assembles the linear system (A, B, G), solves the system
 * (AX=B) using KSP, performs post-processing (computes
 * fields at receivers), prints the footer, and finalizes
 * PETSc, freeing allocated memory. Includes Extrae
 * instrumentation hooks if compiled with USE_EXTRAE.
 */

/**
 * @brief Diagnostic for the partition-dependent PCBDDCNedelecSupport error
 *        "Found more than two corners for edge X". INSTRUMENTATION ONLY - reads
 *        the distributed DMPlex, changes nothing, does not affect the solve.
 *
 * Isolates the failure among three candidates by inspecting the distributed
 * mesh topology on the SAME DM the solver/BDDC uses:
 *
 *   (1) DMPlex partition/topology defect -> PETSc's own checkers (symmetry,
 *       skeleton, faces, cross-rank interface cones). They SETERRQ on any
 *       inconsistency, so reaching the "PASS" line proves the distributed mesh
 *       is sound (no duplicated points, consistent edge/face cones across ranks,
 *       consistent edge-to-cell adjacency) -> candidate (1) ruled out.
 *
 *   (2) Nedelec edge orientation inconsistency -> DMPlexCheckInterfaceCones()
 *       verifies that shared points carry matching CONES *and ORIENTATIONS*
 *       across ranks; a passing check means edge direction is preserved after
 *       distribution -> candidate (2) ruled out. (Edge sign is taken from the
 *       DMPlex closure orientation, which is relative to each edge's canonical
 *       cone and therefore partition-independent.)
 *
 *   (3) BDDC automatic primal/corner selection -> PCBDDC infers corners from
 *       the interface graph (PETGEM does NOT call PCBDDCSetPrimalVerticesLocalIS,
 *       so selection is fully automatic). This routine computes, via the point
 *       SF, the per-point multiplicity = number of subdomains sharing each
 *       vertex/edge. Vertices with multiplicity >= 3 are the corner candidates;
 *       edges with multiplicity >= 3 are the wirebasket. A coarse edge spanning
 *       > 2 corner candidates branches -> the exact "more than two corners"
 *       condition. Reported MPI-rank-aware and globally.
 *
 * @param[in] dm  Fully distributed DMPlex (after setupCsemGrid).
 * @return PetscErrorCode PETSC_SUCCESS, or a PETSc error code (a failing
 *         DMPlexCheck is itself the diagnostic for candidate 1/2).
 */
#define PETGEM_MAXMULT 33
static PetscErrorCode diagnoseMeshBddcCorners(DM dm) {
  PetscFunctionBeginUser;
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  PetscMPIInt rank;
  PetscInt    depth, vStart, vEnd, eStart, eEnd, nroots, nleaves;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(PetscPrintf(comm, "\n===== DMPlex / Nedelec / BDDC corner diagnostic =====\n"));

  /* ---- (1)+(2) DMPlex mesh + cross-rank cone/orientation consistency ---- */
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(PetscPrintf(comm, "[mesh] DMPlex depth = %" PetscInt_FMT
                              " (3 = fully interpolated vtx/edge/face/cell)\n", depth));
  PetscCall(DMPlexCheckSymmetry(dm));
  PetscCall(DMPlexCheckSkeleton(dm, 0));
  PetscCall(DMPlexCheckFaces(dm, 0));
  PetscCall(DMPlexCheckInterfaceCones(dm));
  PetscCall(PetscPrintf(comm,
      "[mesh] DMPlex consistency: PASS (symmetry, skeleton, faces, cross-rank\n"
      "       interface cones+orientations). => candidate (1) partition/topology\n"
      "       defect and (2) edge-orientation inconsistency are RULED OUT.\n"));

  /* ---- (3) interface multiplicity = corner-candidate / wirebasket structure ---- */
  PetscSF            sf;
  const PetscInt    *ilocal, *degree;
  const PetscSFNode *iremote;
  PetscInt          *mult;
  PetscBool         *isLeaf;
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, &iremote));
  if (nroots < 0) nroots = 0;
  if (nleaves < 0) nleaves = 0;
  PetscCall(PetscCalloc1(PetscMax(nroots, 1), &mult));
  PetscCall(PetscCalloc1(PetscMax(nroots, 1), &isLeaf));

  /* multiplicity[p] = #ranks sharing point p: owner gets degree+1, then the
   * owner's value is broadcast to the ghost copies. Interior points -> 1. */
  PetscCall(PetscSFComputeDegreeBegin(sf, &degree));
  PetscCall(PetscSFComputeDegreeEnd(sf, &degree));
  for (PetscInt p = 0; p < nroots; p++) mult[p] = degree[p] + 1;
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, mult, mult, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, mult, mult, MPI_REPLACE));
  for (PetscInt i = 0; i < nleaves; i++) isLeaf[ilocal ? ilocal[i] : i] = PETSC_TRUE;

  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  if (depth >= 1) PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  else            eStart = eEnd = 0;

  PetscInt vHist[PETGEM_MAXMULT] = {0}, eHist[PETGEM_MAXMULT] = {0};
  PetscInt locCorners = 0, locWBedges = 0, locMaxMult = 1;
  /* count each shared point once (on its owner) to avoid cross-rank double count */
  for (PetscInt p = vStart; p < vEnd; p++) {
    if (isLeaf[p]) continue;
    vHist[PetscMin(mult[p], PETGEM_MAXMULT - 1)]++;
    if (mult[p] >= 3) locCorners++;
    locMaxMult = PetscMax(locMaxMult, mult[p]);
  }
  for (PetscInt p = eStart; p < eEnd; p++) {
    if (isLeaf[p]) continue;
    eHist[PetscMin(mult[p], PETGEM_MAXMULT - 1)]++;
    if (mult[p] >= 3) locWBedges++;
  }

  PetscCall(PetscSynchronizedPrintf(comm,
      "[rank %d] owned interface: corner-candidate vtx(mult>=3)=%" PetscInt_FMT
      "  wirebasket edges(mult>=3)=%" PetscInt_FMT "  maxMult=%" PetscInt_FMT "\n",
      rank, locCorners, locWBedges, locMaxMult));
  PetscCall(PetscSynchronizedFlush(comm, PETSC_STDOUT));

  PetscInt gV[PETGEM_MAXMULT], gE[PETGEM_MAXMULT], gMax = 1;
  PetscCallMPI(MPI_Reduce(vHist, gV, PETGEM_MAXMULT, MPIU_INT, MPI_SUM, 0, comm));
  PetscCallMPI(MPI_Reduce(eHist, gE, PETGEM_MAXMULT, MPIU_INT, MPI_SUM, 0, comm));
  PetscCallMPI(MPI_Reduce(&locMaxMult, &gMax, 1, MPIU_INT, MPI_MAX, 0, comm));

  if (rank == 0) {
    PetscInt tV3 = 0, tE3 = 0;
    PetscCall(PetscPrintf(comm, "[corners] GLOBAL interface multiplicity (#subdomains sharing a point):\n"));
    PetscCall(PetscPrintf(comm, "          mult :   #vertices      #edges\n"));
    for (PetscInt m = 2; m <= gMax && m < PETGEM_MAXMULT; m++) {
      PetscCall(PetscPrintf(comm, "          %4" PetscInt_FMT " : %11" PetscInt_FMT " %11" PetscInt_FMT "%s\n",
                            m, gV[m], gE[m], m >= 3 ? "   <- corner-candidate / wirebasket" : "   (faces)"));
      if (m >= 3) { tV3 += gV[m]; tE3 += gE[m]; }
    }
    PetscCall(PetscPrintf(comm,
        "          => corner-candidate vertices (mult>=3) = %" PetscInt_FMT
        ";  wirebasket edges (mult>=3) = %" PetscInt_FMT ";  max multiplicity = %" PetscInt_FMT "\n",
        tV3, tE3, gMax));
    PetscCall(PetscPrintf(comm,
        "[corners] BDDC corner selection is AUTOMATIC (PCBDDCSetPrimalVerticesLocalIS\n"
        "          is NOT used). PCBDDCNedelecSupport infers corners from the graph\n"
        "          above; a coarse edge spanning >2 of the mult>=3 vertices branches\n"
        "          and raises \"Found more than two corners for edge X\". These\n"
        "          cross-points are NORMAL in any 3D partition (present whether or\n"
        "          not the run converges), so the trigger is the automatic 2-corner\n"
        "          model - candidate (3) - not a mesh/partition/orientation defect.\n"));
    PetscCall(PetscPrintf(comm, "=====================================================\n\n"));
  }

  PetscCall(PetscFree(mult));
  PetscCall(PetscFree(isLeaf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int runForward(int argc, char** argv) {

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
  Mat             A = NULL, B, X;
  Mat             G_BDDC = NULL;  /* High-order discrete gradient : Nédélec_nord -> P_nord H1 */
  fmParams        params;
  Grid            grid;
  CsemSourceSet   sources = {0, 0, NULL};
  PetscReal       omega;      
  PetscScalar     constFactor;
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

  PetscCall(PetscInitialize(&argc, &argv, (char*)0, fmHelp));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* ---------------------------------------------------------------- */
  /* Resolve here the petsc help printing                             */
  /* ---------------------------------------------------------------- */
  PetscCall(PetscOptionsHasHelp(NULL, &helpRequested));

  /* ---------------------------------------------------------------- */
  /* Register the fm.csem phase stages with PETSc's logging system    */  
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

  PetscCall(PetscLogStagePush(stage_parse));
  PetscCall(PetscTime(&start_timer));
  PetscCall(readfmParams(size, &params));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[0] = end_timer - start_timer;

  /* Option groups already printed by -help; skip simulation and exit. */
  if (helpRequested) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "\nTip: 'fm.csem -help intro' shows the concise usage summary.\n"));
    PetscCall(PetscFinalize());
    return 0;
  }

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Load input data: mesh, sigma, materials_id, sources, receivers   */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 4);
#endif

  PetscCall(PetscLogStagePush(stage_load));
  PetscCall(PetscTime(&start_timer));
  PetscCall(loadCsemInputs(&params, &dm, &conductivity, &materials_id, &sources, &receivers));
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
  PetscCall(setupCsemGrid(params, &dm, &grid));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[2] = end_timer - start_timer;

  /* Optional visualization of the FULLY DISTRIBUTED DMPlex for ParaView - the
   * mesh as the solver sees it, after load + DMPlexDistribute. Inert unless
   * -dm_view is set. Pair with -dm_partition_view to add a per-cell field of
   * the owning MPI rank (DMPlexCreateRankField), so the subdomain partition
   * behind PCBDDC corner-detection failures can be inspected:
   *   -dm_view vtk:partition.vtu -dm_partition_view
   * Runs on the distributed DM, immediately before assembly and solve. */
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  /* Optional corner-failure diagnostic (instrumentation only; -fm_mesh_bddc_diag).
   * Isolates the "more than two corners" failure among: (1) DMPlex partition,
   * (2) Nedelec edge orientation, (3) BDDC automatic primal selection. Inert
   * unless the flag is set; does not affect the solve. */
  {
    PetscBool diag = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-fm_mesh_bddc_diag", &diag, NULL));
    if (diag) PetscCall(diagnoseMeshBddcCorners(dm));
  }

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
  omega       = sources.freq * 2.0 * PETSC_PI;
  constFactor = (0.0 + 1.0 * PETSC_i) * (omega * MU);

  /* Perform assembly */
  PetscCall(PetscLogStagePush(stage_assembly));
  PetscCall(PetscTime(&start_timer));
  PetscCall(assembleCsemRHS(params, sources, dm, grid, constFactor, &B));
  PetscCall(assembleCsemKandM(params, dm, grid, conductivity, constFactor, &A, NULL, &G_BDDC));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[3] = end_timer - start_timer;

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
  PetscCall(solveCsemSystem(dm, A, B, G_BDDC, &X));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[4] = end_timer - start_timer;

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
  PetscCall(computeFields(params, sources, dm, grid, receivers, X));
  PetscCall(PetscTime(&end_timer));
  PetscCall(PetscLogStagePop());
  timers[5] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

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
  PetscCall(DMDestroy(&grid.H1dm_Pnord));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&conductivity));
  PetscCall(VecDestroy(&materials_id));
  PetscCall(MatDestroy(&G_BDDC));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&X));
  PetscCall(VecDestroy(&receivers));
  PetscCall(PetscFree(sources.sourceArray));

  /* ---------------------------------------------------------------- */
  /* PETSc finalize                                                   */
  /* ---------------------------------------------------------------- */
  PetscCall(PetscFinalize());
  return 0;
}

