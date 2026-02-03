/*
 * Filename: common.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2025-05-28
 *
 * Description:
 * This file contains a collection of functions for common
 * utility functions that are used throughout the PETGEM
 * toolkit. These functions include operations for printing
 * and timers.
 *
 * Usage:
 * Include this file in your source code to utilize the
 * common functions. For example: #include "common.h"
 *
 */

/* C libraries */
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

/* PETSc libraries */
#include <petscsys.h>

/* PETGEM funcions*/
#include "common.h"
#include "version.h"

#define LINE_WIDTH 74

/**
 * @brief Computes the display width of a UTF-8 string in characters.
 *
 * This function calculates the number of printable characters in
 * the input null-terminated string, assuming UTF-8 encoding. It
 * counts only the leading bytes of multi-byte UTF-8 characters,
 * effectively providing the number of characters as they would
 * appear on the console.
 *
 * This function is used by formatting helpers (e.g., printCenteredText)
 * to correctly align text containing multi-byte characters.
 *
 * @param[in]  s      Null-terminated UTF-8 string.
 * @param[out] width  Pointer to an integer where the computed display
 *                    width (in characters) will be stored.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful computation,
 *         or a PETSc error code otherwise.
 */
static PetscErrorCode computeDisplayWidth(const char* s, PetscInt* width) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt len = 0;

  while (*s) {
    unsigned char c = (unsigned char)*s;
    if ((c & 0xC0) != 0x80) { /* Count only start bytes of
                                 UTF-8 characters */
      len++;
    }
    s++;
  }

  *width = len;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints a horizontal separator line.
 *
 * This function prints a line of length LINE_WIDTH consisting
 * of repeated occurrences of the specified character, followed
 * by a newline. It is typically used to visually separate
 * sections of formatted console output.
 *
 * Output is produced using PETSc parallel printing routines on
 * PETSC_COMM_WORLD, ensuring consistent and collective display
 * across all MPI processes.
 *
 * @param[in] c  Character used to fill the separator line.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful
 *         completion, or a PETSc error code otherwise.
 */
static PetscErrorCode printSeparator(const char c) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  char line[LINE_WIDTH + 1]; /* +1 for Null terminator*/

  for (PetscInt i = 0; i < LINE_WIDTH; i++) {
    line[i] = c;
  }
  line[LINE_WIDTH] = '\0'; /* Null terminate */

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s\n", line));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints an empty framed line.
 *
 * This function prints a blank line enclosed by leading and
 * trailing '-' characters, with a total width of LINE_WIDTH.
 * It is intended for spacing within formatted PETGEM output
 * blocks while preserving the visual frame.
 *
 * Output is produced using PETSc parallel printing routines on
 * PETSC_COMM_WORLD, ensuring consistent and collective display
 * across all MPI processes.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful
 *         completion, or a PETSc error code otherwise.
 */
static PetscErrorCode printEmptyLine(void) {

  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-%*s-\n", LINE_WIDTH - 2, ""));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints a line of text centered within a fixed-width frame.
 *
 * This function prints the given text centered within a line of
 * width LINE_WIDTH, enclosed by leading and trailing '-' characters.
 * The centering is computed based on the display width of the text
 * (as returned by computeDisplayWidth()), allowing correct alignment
 * for multi-byte or wide characters.
 *
 * Output is produced using PETSc parallel printing routines on
 * PETSC_COMM_WORLD, ensuring consistent and collective display
 * across all MPI processes.
 *
 * @param[in] text  Null-terminated string to be printed centered
 *                  within the formatted line.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful
 *         completion, or a PETSc error code otherwise.
 */
static PetscErrorCode printCenteredText(const char* text) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt text_len, total_space, left_pad, right_pad;

  /* Compute display width */
  PetscCall(computeDisplayWidth(text, &text_len));

  /* Compute paddings */
  total_space = LINE_WIDTH - 2 - text_len;
  left_pad = total_space / 2;
  right_pad = total_space - left_pad;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-%*s%s%*s-\n", left_pad, "", text, right_pad, ""));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints a formatted timer value in hh:mm:ss.sss format
 *        along with its percentage of the total runtime.
 *
 * This function converts a time interval given in seconds into
 * hours, minutes, and seconds, and prints it together with the
 * percentage that this interval represents relative to a total
 * execution time.
 *
 * The output is formatted as a single line containing a textual
 * label, the elapsed time in hh:mm:ss.sss format, and the
 * corresponding percentage. Printing is performed collectively
 * using PETSc parallel printing routines on PETSC_COMM_WORLD.
 *
 * If the total time is zero or negative, the reported percentage
 * is set to zero to avoid division by zero.
 *
 * @param[in] label  Descriptive label for the timed stage.
 * @param[in] t      Elapsed time for the stage, in seconds.
 * @param[in] total  Total elapsed time used to compute the
 *                   percentage, in seconds.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful
 *         completion, or a PETSc error code otherwise.
 */
static PetscErrorCode PrintTimerHMSPercent(const char* label, PetscLogDouble t, PetscLogDouble total) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscInt hours, minutes;
  PetscLogDouble seconds, percent;

  hours = (PetscInt)(t / 3600.0);
  minutes = (PetscInt)((t - hours * 3600.0) / 60.0);
  seconds = t - hours * 3600.0 - minutes * 60.0;

  percent = (total > 0.0) ? (100.0 * t / total) : 0.0;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   %-16s = %02d:%02d:%06.3f  | %6.2f %% |\n", label, hours, minutes, seconds, percent));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints a formatted PETGEM header banner.
 *
 * This function prints a formatted header to standard output
 * containing basic information about the PETGEM project,
 * including:
 *   - Project name and expanded acronym
 *   - GitHub repository URL
 *   - Developer name
 *   - Institutional affiliations
 *
 * The header is printed using PETSc-based printing utilities
 * and formatting helpers (separators, centered text), ensuring
 * consistent and collective output across all MPI processes
 * associated with PETSC_COMM_WORLD.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful
 *         completion, or a PETSc error code otherwise.
 */
PetscErrorCode printHeader(void) {

  PetscFunctionBeginUser;

  PetscCall(printSeparator('-'));
  PetscCall(printEmptyLine());
  PetscCall(printEmptyLine());
  PetscCall(printCenteredText("PETGEM"));
  PetscCall(printCenteredText("Parallel Exascale Toolkit for "
                              "Geophysical Electromagnetic "
                              "Modeling"));
  PetscCall(printEmptyLine());
  PetscCall(printCenteredText("GitHub repository: "
                              "github.com/ocastilloreyes/petgem"));
  PetscCall(printEmptyLine());
  PetscCall(printSeparator('-'));
  PetscCall(printEmptyLine());
  PetscCall(printCenteredText("Octavio Castillo-Reyes"));
  PetscCall(printCenteredText("Universitat Politècnica de Catalunya (UPC) - 2026"));
  PetscCall(printCenteredText("Barcelona Supercomputing Center (BSC) - 2026"));
  PetscCall(printEmptyLine());
  PetscCall(printSeparator('-'));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode printFooter(void) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  char date[30];

  /* Get date*/
  PetscCall(PetscGetDate(date, 30));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n Finished: %s", date));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n PETGEM version: %d.%d.%d\n", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH));
  PetscCall(printSeparator('-'));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Ensures that a directory exists, creating it if necessary.
 *
 * This function checks whether the specified path exists. If the path
 * already exists and refers to a directory, the function returns
 * successfully. If the path exists but is not a directory, an error
 * is raised.
 *
 * If the path does not exist, the function attempts to create the
 * directory with POSIX permissions 0755. Any errors encountered
 * during directory creation are reported using PETSc error handling
 * mechanisms.
 *
 * @param[in] path  Path to the directory to be checked or created.
 *
 * @return PetscErrorCode PETSC_SUCCESS on success, or a PETSc
 *         error code if the path exists but is not a directory,
 *         or if directory creation fails.
 */
PetscErrorCode createDirectory(const char* path) {

  PetscFunctionBeginUser;

  /* Verify if the directory exists */
  struct stat st;
  if (stat(path, &st) == 0) {
    /* Directory exists */
    if (S_ISDIR(st.st_mode)) {
      PetscFunctionReturn(PETSC_SUCCESS);
    } else {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, " Path exists but is not a directory: %s", path);
    }
  } else {
    /* Directory doesn't exist, create it */
    if (mkdir(path, 0755) && errno != EEXIST) {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, " Error when creating output directory: %s", path);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints execution time statistics for the main
 * computational stages of the PETGEM workflow.
 *
 * This function receives an array of timers containing the
 * elapsed execution times (in seconds) for the different
 * phases of the PETGEM execution. It computes the total
 * elapsed time as the sum of all stages and prints a
 * formatted timing report to standard output, including
 * the absolute time (hh:mm:ss.sss) and the percentage of
 * the total runtime for each stage.
 *
 * The reported stages are:
 *   - Reading user parameters
 *   - Source setup
 *   - Grid import
 *   - Grid setup
 *   - Assembly
 *   - Solver
 *   - Postprocessing
 *
 * Output is produced using PETSc parallel printing routines,
 * ensuring consistent and collective reporting across all
 * MPI processes associated with PETSC_COMM_WORLD.
 *
 * @param[in] timers Array of length 7 containing execution
 *                   times (in seconds) for each stage, in
 *                   the following order:
 *                   timers[0] = Read user parameters
 *                   timers[1] = Setup source
 *                   timers[2] = Import grid
 *                   timers[3] = Setup grid
 *                   timers[4] = Assembly
 *                   timers[5] = Solver
 *                   timers[6] = Postprocessing
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful
 *         completion, or a PETSc error code otherwise.
 */
PetscErrorCode printTimers(const PetscLogDouble timers[]) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscLogDouble elapsed_time = 0.0;

  /* Compute elapsed time */
  for (PetscInt i = 0; i < 7; i++) {
    elapsed_time += timers[i];
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n Timers (hh:mm:ss.sss | %% |):\n"));
  PetscCall(PrintTimerHMSPercent("Read user params", timers[0], elapsed_time));
  PetscCall(PrintTimerHMSPercent("Setup source", timers[1], elapsed_time));
  PetscCall(PrintTimerHMSPercent("Import grid", timers[2], elapsed_time));
  PetscCall(PrintTimerHMSPercent("Setup grid", timers[3], elapsed_time));
  PetscCall(PrintTimerHMSPercent("Assembly", timers[4], elapsed_time));
  PetscCall(PrintTimerHMSPercent("Solver", timers[5], elapsed_time));
  PetscCall(PrintTimerHMSPercent("Postprocessing", timers[6], elapsed_time));
  PetscCall(PrintTimerHMSPercent("Elapsed time", elapsed_time, elapsed_time));

  PetscFunctionReturn(PETSC_SUCCESS);
}
