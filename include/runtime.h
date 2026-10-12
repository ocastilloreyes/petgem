/*
 * Filename: runtime.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * Run scaffolding shared by the PETGEM kernels: PETSc start-up, timed log
 * stages (with optional Extrae events) and the common run-report sections.
 */

#ifndef RUNTIME_H
#define RUNTIME_H

#include <petsc.h>

#include "io.h"
#include "solver.h"

/** @brief Closing lines of every kernel's -help intro text. */
#define PETGEM_HELP_FOOTER                                                                                                                 \
  "MORE\n"                                                                                                                                 \
  "  -help        Full option database, incl. advanced PETSc flags.\n"                                                                     \
  "  -help intro  This concise usage summary, then exit.\n"                                                                                \
  "  --version    Print the PETGEM version and exit.\n"                                                                                    \
  "\n"                                                                                                                                     \
  " Docs: github.com/ocastilloreyes/petgem\n"                                                                                              \
  "================================================================\n"

/**
 * @brief Phases of a kernel run, in execution order.
 *
 * Each phase is a PETSc log stage and, in Extrae builds, the event value
 * phase + 3 of event type 1000.
 */
typedef enum { PHASE_PARSE, PHASE_LOAD, PHASE_GRID, PHASE_ASSEMBLY, PHASE_SOLVE, PHASE_OUTPUT, NUM_PHASES } KernelPhase;

/**
 * @brief State of one kernel run.
 */
typedef struct {
  const char    *name;                /**< Kernel name, e.g. "fm.csem". */
  const char    *outputStage;         /**< Name of the PHASE_OUTPUT stage. */
  PetscBool      helpRequested;       /**< -help was given: parse options and exit. */
  PetscLogStage  stage[NUM_PHASES];   /**< PETSc log stage per phase. */
  PetscLogDouble elapsed[NUM_PHASES]; /**< Accumulated wall time per phase (s). */
  PetscLogDouble start;               /**< Start time of the open phase. */
} KernelRun;

/**
 * @brief Returns PETSC_TRUE when argv[1] is --version.
 */
PetscBool isVersionRequest(int argc, char **argv);

/**
 * @brief Prints "PETGEM version X.Y.Z" with plain printf (no PETSc needed).
 *
 * @return int 0, to be used as the process exit status.
 */
int printVersion(void);

/**
 * @brief Initializes PETSc, registers the phase stages and prints the header.
 *
 * The header is skipped under -help, so the option listing prints cleanly.
 *
 * @param[in,out] argc         Argument count from main().
 * @param[in,out] argv         Argument vector from main().
 * @param[in]     help         Usage text shown by -help intro.
 * @param[in]     name         Kernel name.
 * @param[in]     outputStage  Name of the PHASE_OUTPUT stage.
 * @param[out]    run          Run state.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode beginKernelRun(int *argc, char ***argv, const char help[], const char *name, const char *outputStage, KernelRun *run);

/**
 * @brief Opens a phase: pushes its log stage and starts its timer.
 */
PetscErrorCode beginPhase(KernelRun *run, KernelPhase phase);

/**
 * @brief Closes a phase: adds the elapsed time and pops its log stage.
 */
PetscErrorCode endPhase(KernelRun *run, KernelPhase phase);

/**
 * @brief Ends a -help run: prints the usage tip and finalizes PETSc.
 */
PetscErrorCode endHelpRun(const KernelRun *run);

/**
 * @brief Prints the timers and the closing banner.
 *
 * @param[in] labels  Timer labels.
 * @param[in] times   Timer values in seconds.
 * @param[in] n       Number of timers.
 */
PetscErrorCode printRunSummary(const char *const labels[], const PetscLogDouble times[], PetscInt n);

/**
 * @brief Time spent reading parameters, loading the bundle and setting up the grid.
 *
 * @param[in] run  Kernel run.
 *
 * @return Sum of the PHASE_PARSE, PHASE_LOAD and PHASE_GRID times (s).
 */
PetscLogDouble setupTime(const KernelRun *run);

/**
 * @brief Prints the timers of a forward kernel (load + grid, assembly, solve, output) and the footer.
 *
 * @param[in] run  Kernel run.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode printForwardSummary(const KernelRun *run);

/**
 * @brief Prints the "Assembly" section: operator type and right-hand sides.
 */
PetscErrorCode logAssembly(MPI_Comm comm, Mat A, PetscInt numRhs);

/**
 * @brief Prints the "Solve" section: solver name and convergence status.
 */
PetscErrorCode logSolve(MPI_Comm comm, const SolveInfo *info);

/**
 * @brief Prints a receiver-output section: receiver count and output file.
 *
 * @param[in] comm       Communicator (rank 0 prints).
 * @param[in] title      Section title.
 * @param[in] params     Parameters carrying the output path.
 * @param[in] receivers  Serial Vec of 3·N_recv receiver coordinates.
 */
PetscErrorCode logReceiverOutput(MPI_Comm comm, const char *title, const PetgemParams *params, Vec receivers);

#endif
