/*
 * Filename: common.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-02-03
 *
 * Description:
 * Common utility functions used throughout the PETGEM toolkit, including printing helpers and timers.
 */

/* C libraries */
#include <errno.h>
#include <stdarg.h>
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
 * PETSC_COMM_WORLD.
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
 * PETSC_COMM_WORLD.
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
 * PETSC_COMM_WORLD.
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
  PetscInt text_len;
  
  /* `*` width specifier requires `int` per C standard; PetscInt would
   * be int64_t on 64-bit indices builds and trip strict format checks. */
  int total_space, left_pad, right_pad;

  /* Compute display width */
  PetscCall(computeDisplayWidth(text, &text_len));

  /* Compute paddings */
  total_space = LINE_WIDTH - 2 - (int)text_len;
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
  /* `%02d` requires `int`; hours/minutes are bounded small (≤ runtime in
   * hours), so plain int is the natural type. */
  int hours, minutes;
  PetscLogDouble seconds, percent;

  hours = (int)(t / 3600.0);
  minutes = (int)((t - hours * 3600.0) / 60.0);
  seconds = t - hours * 3600.0 - minutes * 60.0;

  percent = (total > 0.0) ? (100.0 * t / total) : 0.0;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   %-24s = %02d:%02d:%06.3f  | %6.2f %% |\n", label, hours, minutes, seconds, percent));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Formats an integer with space-grouped thousands for readable logs.
 *
 * Renders `value` with a space every three digits (e.g. 3738963 -> "3 738 963")
 * into one of a few rotating internal buffers, so several formatted numbers can
 * appear in a single printf argument list (e.g. "M x N"). Presentation only:
 * values below 1000 are rendered unchanged. Not thread-safe (static buffers).
 *
 * @param[in] value  Integer to format.
 *
 * @return Pointer to a NUL-terminated grouped-number string (do not free).
 */
const char *formatGroupedInt(PetscInt value) {
  enum { NUM_BUFS = 6, BUF_LEN = 32 };
  static char bufs[NUM_BUFS][BUF_LEN];
  static PetscInt which = 0;
  char *out = bufs[which];
  which = (which + 1) % NUM_BUFS;

  /* Absolute-value decimal digits, least-significant first (INT_MIN-safe). */
  char digits[24];
  PetscInt nd = 0;
  unsigned long long uv = (value < 0) ? (unsigned long long)(-(value + 1)) + 1ULL
                                      : (unsigned long long)value;
  if (uv == 0) {
    digits[nd++] = '0';
  }

  while (uv > 0) { 
    digits[nd++] = (char)('0' + (int)(uv % 10ULL)); uv /= 10ULL; 
  }

  /* Emit most-significant first, a space after every third remaining digit. */
  PetscInt oi = 0;
  if (value < 0) {
    out[oi++] = '-';
  }
  for (PetscInt i = nd - 1; i >= 0; i--) {
    out[oi++] = digits[i];
    if (i > 0 && (i % 3) == 0) {
      out[oi++] = ' ';
    }
  }
  out[oi] = '\0';
  return out;
}


/**
 * @brief Prints a titled section header in the PETGEM run report.
 *
 * Emits a blank line followed by a section title and trailing colon
 * (e.g. "Mesh:" or "Inversion parameters:") using PETSc collective
 * printing. Used as a visual separator between logical groups of
 * runtime information.
 *
 * @param[in] comm   MPI communicator used by PetscPrintf().
 * @param[in] title  Section title to display.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode logSection(MPI_Comm comm, const char *title) {
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(comm, "\n %s:\n", title));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Prints a string-valued key/value entry in the PETGEM run report.
 *
 * Formats a report line using the standard PETGEM layout
 * ("   %-24s = %s") so all modules produce aligned output.
 * Intended for textual values such as filenames, modes, states,
 * or descriptive labels.
 *
 * @param[in] comm  MPI communicator used by PetscPrintf().
 * @param[in] key   Entry label displayed in the left column.
 * @param[in] val   String value displayed in the right column.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode logKVStr(MPI_Comm comm, const char *key, const char *val) {
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(comm, "   %-24s = %s\n", key, val));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Prints an integer-valued key/value entry in the PETGEM run report.
 *
 * Formats the integer through formatGroupedInt() before printing so
 * large values appear with space-grouped thousands (e.g. 44447 ->
 * "44 447"). Uses the standard PETGEM key/value layout to keep
 * console output aligned and consistent across kernels.
 *
 * @param[in] comm  MPI communicator used by PetscPrintf().
 * @param[in] key   Entry label displayed in the left column.
 * @param[in] val   Integer value to display.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode logKVInt(MPI_Comm comm, const char *key, PetscInt val) {
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(comm, "   %-24s = %s\n", key, formatGroupedInt(val)));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Prints a floating-point key/value entry in the PETGEM run report.
 *
 * Renders the supplied value using PETSc's standard console output path
 * and the "%g" numeric format. Intended for tolerances, weights,
 * frequencies, physical parameters, and other scalar quantities.
 *
 * @param[in] comm  MPI communicator used by PetscPrintf().
 * @param[in] key   Entry label displayed in the left column.
 * @param[in] val   Floating-point value to display.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode logKVReal(MPI_Comm comm, const char *key, PetscReal val) {
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(comm, "   %-24s = %g\n", key, (double)val));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Prints a formatted key/value entry in the PETGEM run report.
 *
 * Builds the value string from a printf-style format specification and
 * variable argument list, then emits the result using the standard
 * PETGEM key/value layout. Useful when the displayed value combines
 * multiple quantities or requires custom formatting.
 *
 * The formatted value is written into a fixed-size internal stack buffer
 * before printing. Output longer than the buffer capacity is truncated by
 * vsnprintf().
 *
 * @param[in] comm    MPI communicator used by PetscPrintf().
 * @param[in] key     Entry label displayed in the left column.
 * @param[in] valfmt  printf-style format string used to build the value.
 * @param[in] ...     Arguments consumed by valfmt.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode logKVf(MPI_Comm comm, const char *key, const char *valfmt, ...) {
  PetscFunctionBeginUser;
  char    buf[256];
  va_list ap;
  va_start(ap, valfmt);
  vsnprintf(buf, sizeof(buf), valfmt, ap);
  va_end(ap);
  PetscCall(PetscPrintf(comm, "   %-24s = %s\n", key, buf));
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
  PetscCall(printCenteredText("Parallel Edge-element Toolkit for General Electromagnetic Modeling"));
  PetscCall(printEmptyLine());
  PetscCall(printCenteredText("GitHub repository: github.com/ocastilloreyes/petgem"));
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

/**
 * @brief Prints the PETGEM closing banner with run timestamp + author/affiliation.
 *
 * Called once at the end of every kernel run (fm.csem, im.csem, petgem) to
 * mark the end of stdout output.  Output goes to PETSC_COMM_WORLD via the
 * centered-text helpers, so it is collectively printed by rank 0 only.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful completion, or a
 *         PETSc error code if any of the underlying print helpers fail.
 */
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
 *   - Load input data
 *   - Grid setup
 *   - Assembly
 *   - Solver
 *   - Postprocessing
 *
 * Output is produced using PETSc parallel printing routines,
 * ensuring consistent and collective reporting across all
 * MPI processes associated with PETSC_COMM_WORLD.
 *
 * @param[in] timers Array of length 6 containing execution
 *                   times (in seconds) for each stage, in
 *                   the following order:
 *                   timers[0] = Read user parameters
 *                   timers[1] = Load input data
 *                   timers[2] = Grid setup
 *                   timers[3] = Assembly
 *                   timers[4] = Solver
 *                   timers[5] = Postprocessing
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful
 *         completion, or a PETSc error code otherwise.
 */
PetscErrorCode printTimers(const PetscLogDouble timers[]) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscLogDouble elapsed_time = 0.0;

  /* Compute elapsed time */
  for (PetscInt i = 0; i < 6; i++) {
    elapsed_time += timers[i];
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n Timers (hh:mm:ss.sss |   %% |):\n"));
  PetscCall(PrintTimerHMSPercent("Read parameters",     timers[0], elapsed_time));
  PetscCall(PrintTimerHMSPercent("Load input bundle",   timers[1], elapsed_time));
  PetscCall(PrintTimerHMSPercent("Setup grid",          timers[2], elapsed_time));
  PetscCall(PrintTimerHMSPercent("Assembly",            timers[3], elapsed_time));
  PetscCall(PrintTimerHMSPercent("Linear solve",        timers[4], elapsed_time));
  PetscCall(PrintTimerHMSPercent("Field interpolation", timers[5], elapsed_time));
  PetscCall(PrintTimerHMSPercent("Total",               elapsed_time, elapsed_time));

  PetscFunctionReturn(PETSC_SUCCESS);
}



/**
 * @brief Prints a one-screen CLI usage summary for the unified `petgem` dispatcher.
 *
 * Lists the positional-subcommand form (`petgem modeling | inverse`), the
 * equivalent `-mode <m>` PETSc-option form, and `--version`.  Called when
 * the dispatcher in src/petgem.c receives no recognized subcommand, or
 * when `--help` is requested.
 *
 * @param[in] progname Executable basename (typically argv[0]) used as the
 *                     leading word of each example line.
 * @return PetscErrorCode PETSC_SUCCESS on success.
 */
PetscErrorCode printUsage(const char *progname) {
  PetscFunctionBeginUser;

  /* Plain printf, not PetscPrintf: the dispatcher calls printUsage BEFORE
   * PetscInitialize (for --help and for an unknown subcommand), so MPI is not
   * up yet and PetscPrintf(PETSC_COMM_WORLD) would abort. Matches the
   * --version path, which prints the same way. */
  printf(
    "Usage:\n"
    "  %s fm [petsc options...]      # run forward kernel (fm.csem)\n"
    "  %s im [petsc options...]      # run inverse kernel (im.csem)\n"
    "  %s -mode fm [petsc options]   # equivalent (PETSc-option form)\n"
    "  %s -mode im [petsc options]\n"
    "  %s --version\n"
    "\n"
    "'fm' and 'im' are the canonical simulation tags, used consistently across\n"
    "the interface (binaries, -mode, the -im_* options, and the simulation_type\n"
    "attribute of every output file). 'forward'/'modeling' and 'inverse' are\n"
    "accepted as aliases.\n"
    "\n"
    "Pass -options_file <file.txt> for the usual params input.\n",
    progname, progname, progname, progname, progname);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Parse the dispatcher mode argument into a numeric code.
 *
 * Accepts the user-friendly synonyms for each kernel:
 *   - "modeling", "forward", "fm"  : *mode = 0 (forward)
 *   - "inverse", "im"              : *mode = 1 (inverse)
 *   - anything else                : *mode = -1 (unknown; caller can
 *                                    then call printUsage())
 *
 * @param[in]  s     Mode string from argv. Must be non-NULL.
 * @param[out] mode  Receives the numeric mode code (see above).
 * @return PetscErrorCode PETSC_SUCCESS, or PETSC_ERR_ARG_NULL when `s` is NULL.
 */
PetscErrorCode parseModeArg(const char *s, PetscInt *mode)
{
  PetscFunctionBeginUser;

  PetscCheck(s, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "Mode string cannot be NULL");

  if (strcmp(s, "modeling") == 0 || strcmp(s, "forward") == 0 || strcmp(s, "fm") == 0) { 
    *mode = 0;  /* Forward modeling */
  } else if (strcmp(s, "inverse") == 0 || strcmp(s, "im") == 0) { 
    *mode = 1;  /* Inverse modeling */
  } else {
    *mode = -1; /* Unknown */
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}