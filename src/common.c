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

static PetscErrorCode printEmptyLine(void) {

  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-%*s-\n", LINE_WIDTH - 2, ""));

  PetscFunctionReturn(PETSC_SUCCESS);
}

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
 * @brief Prints a header with PETGEM project information
 * and the current year.
 *
 * This function prints a formatted header to the console,
 * containing information about the PETGEM project,
 * including its name, purpose, GitHub repository, website,
 * and the names and affiliations of the developers. It also
 * includes the current year.
 *
 * The header is printed using PETSc's parallel printing
 * functions, ensuring that the output is consistent across
 * all processes in the PETSc communicator.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful
 * completion, or an error code otherwise.
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
  PetscCall(printCenteredText("Universitat Politècnica de Catalunya (UPC) - 2025"));
  PetscCall(printCenteredText("Barcelona Supercomputing Center (BSC) - 2025"));
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
 * @brief Creates a directory if it does not already exist.
 *
 * This function checks if the specified directory exists.
 * If the directory does not exist, it attempts to create it
 * with the specified permissions. The function uses PETSc
 * error handling to report any issues encountered during
 * the creation of the directory.
 *
 * @param[in] path  A constant character pointer to the path
 * of the directory to be created.
 * @return PetscErrorCode PETSC_SUCCESS on successful
 * completion. If the directory creation fails, it returns
 * an appropriate PETSc error code.
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
 * @brief Prints execution time statistics for different
 * computational stages.
 *
 * This function receives an array of timers containing the
 * elapsed execution times of different phases of the PETGEM
 * workflow. It prints to the console the time spent in:
 *   - Reading and setting up the grid
 *   - Assembly
 *   - Solver
 *   - Postprocessing
 *
 * In addition, it computes and displays the total elapsed
 * time as the sum of these stages. The output is generated
 * using PETSc's parallel printing functions, ensuring
 * consistency across all processes in the PETSc
 * communicator.
 *
 * @param[in] timers Array of length 4 containing execution
 * times (in seconds) for each stage of the workflow in the
 * following order: timers[0] = Read/setup grid timers[1] =
 * Assembly timers[2] = Solver timers[3] = Postprocessing
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful
 * completion, or an error code otherwise.
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
