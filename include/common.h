/*
 * Filename: common.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Prototypes for the common utility functions (printing helpers,
 * timers, …) used throughout PETGEM.
 */

#ifndef COMMON_H
#define COMMON_H

#include <petsc.h>

/**
 * @brief Prints the formatted PETGEM header banner.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode printHeader();

/**
 * @brief Prints the PETGEM closing banner with run timestamp and author.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode printFooter();

/**
 * @brief Ensures that a directory exists, creating it if necessary.
 *
 * @param[in] path  Directory path to create (no-op if it already exists).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode createDirectory(const char* path);

/**
 * @brief Prints execution-time statistics for the main PETGEM stages.
 *
 * Reports per-stage absolute time (hh:mm:ss.sss) and percentage of the
 * total runtime, collectively across PETSC_COMM_WORLD.
 *
 * @param[in] timers  Array of length 6 of per-stage elapsed times (s):
 *                     [0] read params, [1] load input, [2] grid setup,
 *                     [3] assembly, [4] solver, [5] postprocessing.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode printTimers(const PetscLogDouble timers[]);

/**
 * @brief Parses the dispatcher mode argument into a numeric code.
 *
 * Accepts the synonyms "modeling"/"forward"/"fm" (mode 0) and
 * "inverse"/"im" (mode 1); anything else yields mode -1 (unknown).
 *
 * @param[in]  s     Mode string from argv (must be non-NULL).
 * @param[out] mode  Receives the numeric mode code (0, 1, or -1).
 *
 * @return PetscErrorCode PETSC_SUCCESS, or PETSC_ERR_ARG_NULL when `s` is NULL.
 */
PetscErrorCode parseModeArg(const char *s, PetscInt *mode);

/**
 * @brief Prints a one-screen CLI usage summary for the `petgem` dispatcher.
 *
 * @param[in] progname  Executable basename (typically argv[0]).
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode printUsage(const char *progname);

/**
 * @brief Formats an integer with space-grouped thousands for readable logs.
 *
 * Renders `value` with a space every three digits (e.g. 3738963 -> "3 738 963")
 * for use as a `%s` argument in log output. Presentation only; values below
 * 1000 are rendered unchanged. Uses rotating static buffers (not thread-safe).
 *
 * @param[in] value  Integer to format.
 *
 * @return Pointer to a NUL-terminated grouped-number string (do not free).
 */
const char *formatGroupedInt(PetscInt value);

/**
 * @brief Structured console-log helpers - the single place that owns the
 *        PETGEM run-report layout (section headers and 24-column key-value
 *        lines), keeping fm.csem and im.csem output consistent.
 *
 * - logSection prints "\n <title>:\n".
 * - logKVStr/Int/Real print "   %-24s = <value>\n" with a string, a
 *   space-grouped integer, or a %g real, respectively.
 * - logKVf prints "   %-24s = <value>\n" with a printf-formatted value for the
 *   few lines that need a custom value layout.
 *
 * @param[in] comm  Communicator to print on (rank 0 emits).
 * @param[in] key   Left-hand label (padded to 24 columns; KV helpers only).
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc error code.
 */
PetscErrorCode logSection(MPI_Comm comm, const char *title);
PetscErrorCode logKVStr(MPI_Comm comm, const char *key, const char *val);
PetscErrorCode logKVInt(MPI_Comm comm, const char *key, PetscInt val);
PetscErrorCode logKVReal(MPI_Comm comm, const char *key, PetscReal val);
PetscErrorCode logKVf(MPI_Comm comm, const char *key, const char *valfmt, ...);

#endif
