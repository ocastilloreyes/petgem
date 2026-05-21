/*
 * Filename: im_csem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Inverse CSEM kernel (runInverse). Linked into both the legacy
 * im.csem binary and the unified petgem dispatcher.
 */

static char imHelp[] = "PETGEM inverse CSEM kernel (runInverse / im.csem).\n\
  Standalone usage:\n\
    mpirun -n <np> ./im.csem -options_file <file.txt>\n\
  Unified-binary usage:\n\
    mpirun -n <np> ./petgem inverse -options_file <file.txt>\n";

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
#include "inputs.h"
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
 * conductivity + materials_id + receivers + /inv_sources +
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
  csemParams      params;
  invParams       iparams;
  Grid            grid;
  PetscLogDouble  timers[7];
  PetscLogDouble  start_timer, end_timer;

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

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Print PETGEM header                                              */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 2);
#endif

  PetscCall(printHeader());

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Parse user parameters                                            */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 3);
#endif

  PetscCall(PetscTime(&start_timer));
  PetscCall(readCsemParams(size, &params));
  PetscCall(PetscTime(&end_timer));
  timers[0] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Parse inversion parameters                                        */
  /* ---------------------------------------------------------------- */
  PetscCall(readInversionParams(&iparams));

  /* Stash the bundle path on iparams so runCsemInversion can load
   * /observed/Ex from the same file (avoids threading params into the
   * inversion driver). */
  PetscCall(PetscStrncpy(iparams.bundleFile, params.inputFile,
                         sizeof(iparams.bundleFile)));

  /* Pull case-property defaults out of the bundle (error_level,
   * fixed_materials) — CLI overrides applied by readInversionParams
   * already take precedence via the *FromCLI provenance flags. */
  PetscCall(loadInversionMetaFromBundle(iparams.bundleFile, &iparams));

  /* ---------------------------------------------------------------- */
  /* Load unified PETGEM input: mesh + sigma + materials_id +          */
  /* receivers from the bundle. The bundle's inv_sources group and     */
  /* observed Ex dataset are read below via setupInversionSources and  */
  /* (later, once numReceivers is known) loadObservedData inside       */
  /* runCsemInversion. The forward /sources group is skipped (NULL) —  */
  /* the inverse kernel uses the multi-frequency inv_sources records.  */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 5);
#endif

  PetscCall(PetscTime(&start_timer));
  PetscCall(loadCsemInputs(&params, &dm, &resistivity, &materials_id,
                           NULL,           /* skip /sources; we read /inv_sources next */
                           &receivers));
  /* Sync the basis order into invParams: the bundle's /nord (read by
   * loadCsemInputs into params.nord) is the source of truth, unless the
   * user supplied -nord on the command line (handled in readInversionParams). */
  if (iparams.nord == 0) iparams.nord = params.nord;
  PetscCall(PetscTime(&end_timer));
  timers[2] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* ---------------------------------------------------------------- */
  /* Load multi-frequency inversion sources from bundle /inv_sources   */
  /* ---------------------------------------------------------------- */
#ifdef USE_EXTRAE
  Extrae_event(1000, 4);
#endif

  PetscCall(PetscTime(&start_timer));
  PetscCall(setupInversionSources(params.inputFile, &iparams));
  PetscCall(PetscTime(&end_timer));
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

  PetscCall(PetscTime(&start_timer));
  PetscCall(setupCsemGrid(params, &dm, &grid));
  PetscCall(PetscTime(&end_timer));
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

  PetscCall(PetscTime(&start_timer));
  PetscCall(runCsemInversion(&iparams,
                              dm, &grid, resistivity, materials_id,
                              receivers));
  PetscCall(PetscTime(&end_timer));
  timers[4] = end_timer - start_timer;

#ifdef USE_EXTRAE
  Extrae_event(1000, 0);
#endif

  /* Unused timer slots */
  timers[5] = 0.0;
  timers[6] = 0.0;

  /* ---------------------------------------------------------------- */
  /* Print timers and footer                                           */
  /* ---------------------------------------------------------------- */
  PetscCall(printTimers(timers));
  PetscCall(printFooter());

  /* ---------------------------------------------------------------- */
  /* Free memory                                                       */
  /* ---------------------------------------------------------------- */
  PetscCall(DMDestroy(&grid.H1dm));
  PetscCall(DMDestroy(&grid.H1dm_Pnord));
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
