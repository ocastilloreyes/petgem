/*
 * Filename: common.h
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Prototypes for the common utility functions (printing helpers,
 * timers, ...) used throughout PETGEM.
 */

#ifndef COMMON_H
#define COMMON_H

#include <petsc.h>

/**
 * @brief Prints the PETGEM banner and the "Run" section (kernel, PETGEM
 *        version and git revision, PETSc version, MPI ranks, start time).
 *
 * @param[in] kernel  Kernel name shown in the header (e.g. "fm.csem").
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode printHeader(const char *kernel);

/**
 * @brief Prints the closing banner with the finish time.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode printFooter(void);

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
 * @brief Prints the "Timers" section: per-stage time (hh:mm:ss.sss) and
 *        percentage of the total, followed by the total.
 *
 * @param[in] labels  Stage names.
 * @param[in] times   Stage times in seconds.
 * @param[in] n       Number of stages.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode printTimers(const char *const labels[], const PetscLogDouble times[], PetscInt n);

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
 * @brief Formats a real for logs, keeping it visibly a real.
 *
 * Renders `value` with %g-like significant-digit behaviour, then guarantees the
 * result still reads as floating point: a whole value gains a ".0" (1 -> "1.0",
 * -4000 -> "-4000.0") so it cannot be mistaken for a count, while values that
 * already carry a fractional part or an exponent are left as they are ("2.1",
 * "0.01", "1e-08").
 *
 * Use this for physical quantities - frequencies, coordinates, tolerances - and
 * formatGroupedInt() for cardinalities such as cell or receiver counts, so the
 * two are distinguishable at a glance in the report.
 *
 * Uses rotating static buffers (not thread-safe), matching formatGroupedInt().
 *
 * @param[in] value  Real to format.
 *
 * @return Pointer to a NUL-terminated string (do not free).
 */
const char *formatReal(PetscReal value);


/**
 * @brief Prints a section header in the PETGEM run-report layout.
 *
 * Emits:
 *
 *     "\n<title>:\n"
 *
 * Only rank 0 of the communicator produces output.
 *
 * @param[in] comm   Communicator to print on (rank 0 emits).
 * @param[in] title  Section title.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode logSection(MPI_Comm comm, const char *title);


/**
 * @brief Formats a real with "%.6g" (whole values print without a point).
 *
 * @param[in] value  Real to format.
 *
 * @return Pointer to a NUL-terminated string (rotating static buffer).
 */
const char *formatCompactReal(PetscReal value);

/**
 * @brief Prints a string key-value line in the PETGEM run-report layout.
 *
 * Emits:
 *
 *     "   %-24s = <value>\n"
 *
 * where the value is a string. Only rank 0 of the communicator produces
 * output.
 *
 * @param[in] comm  Communicator to print on (rank 0 emits).
 * @param[in] key   Left-hand label (padded to 24 columns).
 * @param[in] val   String value.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode logKVStr(MPI_Comm comm, const char *key, const char *val);


/**
 * @brief Prints an integer key-value line in the PETGEM run-report layout.
 *
 * Emits:
 *
 *     "   %-24s = <value>\n"
 *
 * where the value is printed as a space-grouped integer. Only rank 0 of the
 * communicator produces output.
 *
 * @param[in] comm  Communicator to print on (rank 0 emits).
 * @param[in] key   Left-hand label (padded to 24 columns).
 * @param[in] val   Integer value.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode logKVInt(MPI_Comm comm, const char *key, PetscInt val);


/**
 * @brief Prints a real-valued key-value line in the PETGEM run-report layout.
 *
 * Emits:
 *
 *     "   %-24s = <value>\n"
 *
 * where the value is printed using %g. Only rank 0 of the communicator
 * produces output.
 *
 * @param[in] comm  Communicator to print on (rank 0 emits).
 * @param[in] key   Left-hand label (padded to 24 columns).
 * @param[in] val   Real value.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode logKVReal(MPI_Comm comm, const char *key, PetscReal val);


/**
 * @brief Prints a printf-formatted key-value line in the PETGEM run-report
 *        layout.
 *
 * Emits:
 *
 *     "   %-24s = <value>\n"
 *
 * where the value is generated from a printf-style format string. Only rank 0
 * of the communicator produces output.
 *
 * @param[in] comm    Communicator to print on (rank 0 emits).
 * @param[in] key     Left-hand label (padded to 24 columns).
 * @param[in] valfmt  printf-style format string for the value.
 * @param[in] ...     Arguments consumed by valfmt.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success,
 *         or a PETSc error code otherwise.
 */
PetscErrorCode logKVf(MPI_Comm comm, const char *key, const char *valfmt, ...);




















#endif
