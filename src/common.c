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
#include <time.h>

/* PETSc libraries */
#include <petscsys.h>

/* PETGEM functions */
#include "common.h"
#include "git_rev.h"
#include "version.h"

#define LINE_WIDTH 74
#define KEY_WIDTH  24

/* Number of characters (UTF-8 start bytes) in s. */
static int displayWidth(const char *s) {
  int n = 0;
  for (; *s; s++) {
    if (((unsigned char)*s & 0xC0) != 0x80) n++;
  }
  return n;
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
 * (as returned by displayWidth()), allowing correct alignment
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
static PetscErrorCode printCenteredText(const char *text) {
  PetscFunctionBeginUser;
  const int space = LINE_WIDTH - 2 - displayWidth(text);
  const int left  = space / 2;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "-%*s%s%*s-\n", left, "", text, space - left, ""));
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
static PetscErrorCode printTimerLine(const char *label, PetscLogDouble t, PetscLogDouble total) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  /* `%02d` requires `int`; hours/minutes are bounded small (≤ runtime in
   * hours), so plain int is the natural type. */
  int            hours, minutes;
  PetscLogDouble seconds, percent;

  hours   = (int)(t / 3600.0);
  minutes = (int)((t - hours * 3600.0) / 60.0);
  seconds = t - hours * 3600.0 - minutes * 60.0;

  percent = (total > 0.0) ? (100.0 * t / total) : 0.0;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   %-24s = %02d:%02d:%06.3f  | %6.2f %% |\n", label, hours, minutes, seconds, percent));

  PetscFunctionReturn(PETSC_SUCCESS);
}

#define FORMAT_NUM_BUFS 16
#define FORMAT_BUF_LEN  32

/* Rotating buffers, so several formatted values fit in one printf (not thread-safe). */
static char *nextFormatBuffer(void) {
  static char     bufs[FORMAT_NUM_BUFS][FORMAT_BUF_LEN];
  static PetscInt which = 0;
  char           *out   = bufs[which];
  which                 = (which + 1) % FORMAT_NUM_BUFS;
  return out;
}

/**
 * @brief Formats an integer with space-grouped thousands for readable logs.
 *
 * Renders `value` with a space every three digits (e.g. 3738963 -> "3 738 963")
 * into one of a few rotating internal buffers, so several formatted numbers can
 * appear in a single printf argument list (e.g. "M x N"). Presentation only:
 * values below 1000 are rendered unchanged. Not thread-safe (static buffers).
 */
const char *formatGroupedInt(PetscInt value) {
  char *out = nextFormatBuffer();

  /* Absolute-value decimal digits, least-significant first (INT_MIN-safe). */
  char               digits[24];
  PetscInt           nd = 0;
  unsigned long long uv = (value < 0) ? (unsigned long long)(-(value + 1)) + 1ULL : (unsigned long long)value;
  if (uv == 0) {
    digits[nd++] = '0';
  }

  while (uv > 0) {
    digits[nd++] = (char)('0' + (int)(uv % 10ULL));
    uv /= 10ULL;
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
 * @brief Formats a real for logs, keeping it visibly a real ("1" -> "1.0").
 */
const char *formatReal(PetscReal value) {
  char *out = nextFormatBuffer();

  /* snprintf, not PetscSNPrintf: PETSc's %g handling renders a whole value as
   * "1." (bare trailing point). Naming the precision gives plain C behavior,
   * so 1.0 arrives here as "1". */
  snprintf(out, FORMAT_BUF_LEN, "%.6g", (double)value);

  /* Restore the float signal that %g drops. Anything already carrying a point,
   * an exponent, or being nan/inf is left untouched. */
  for (const char *p = out; *p; p++) {
    if (*p == '.' || *p == 'e' || *p == 'E' || *p == 'n' || *p == 'i') return out;
  }
  size_t len = strlen(out);
  if (len + 2 < FORMAT_BUF_LEN) {
    out[len]     = '.';
    out[len + 1] = '0';
    out[len + 2] = '\0';
  }
  return out;
}

/**
 * @brief Formats a real with "%.6g" (whole values print without a point).
 *
 * Renders into one of a few rotating internal buffers. Not thread-safe.
 */
const char *formatCompactReal(PetscReal value) {
  char *out = nextFormatBuffer();
  snprintf(out, FORMAT_BUF_LEN, "%.6g", (double)value);
  return out;
}

/**
 * @brief Prints a titled section header in the PETGEM run report.
 *
 * Emits a blank line followed by a section title and trailing colon
 * (e.g. "Mesh:" or "Inversion parameters:") using PETSc collective
 * printing. Used as a visual separator between logical groups of
 * runtime information.
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
 */
PetscErrorCode logKvStr(MPI_Comm comm, const char *key, const char *val) {
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(comm, "   %-*s = %s\n", KEY_WIDTH, key, val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints an integer-valued key/value entry in the PETGEM run report.
 *
 * Formats the integer through formatGroupedInt() before printing so
 * large values appear with space-grouped thousands (e.g. 44447 ->
 * "44 447"). Uses the standard PETGEM key/value layout to keep
 * console output aligned and consistent across kernels.
 */
PetscErrorCode logKvInt(MPI_Comm comm, const char *key, PetscInt val) {
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(comm, "   %-*s = %s\n", KEY_WIDTH, key, formatGroupedInt(val)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints a floating-point key/value entry in the PETGEM run report.
 *
 * Renders the value through formatReal(), so it stays visibly a real: whole
 * values print as "1.0" rather than "1", keeping them distinct from the counts
 * logKvInt() prints. Intended for tolerances, weights, frequencies, physical
 * parameters, and other scalar quantities.
 */
PetscErrorCode logKvReal(MPI_Comm comm, const char *key, PetscReal val) {
  PetscFunctionBeginUser;
  /* formatReal, not %g: PETSc's %g renders whole values as "1." (bare trailing
   * point), and a plain "%.6g" would render them as "1", which reads like a
   * count. formatReal gives "1.0", keeping reals distinguishable from the
   * integers that logKvInt prints. */
  PetscCall(PetscPrintf(comm, "   %-*s = %s\n", KEY_WIDTH, key, formatReal(val)));
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
 */
PetscErrorCode logKvf(MPI_Comm comm, const char *key, const char *valfmt, ...) {
  PetscFunctionBeginUser;
  char    buf[256];
  va_list ap;
  va_start(ap, valfmt);
  vsnprintf(buf, sizeof(buf), valfmt, ap);
  va_end(ap);
  PetscCall(PetscPrintf(comm, "   %-*s = %s\n", KEY_WIDTH, key, buf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Formats the current local time as "YYYY-MM-DD hh:mm:ss".
 *
 * @param[out] buf  Destination buffer.
 * @param[in]  len  Size of @p buf in bytes.
 */
static void formatNow(char *buf, size_t len) {
  time_t now = time(NULL);
  strftime(buf, len, "%Y-%m-%d %H:%M:%S", localtime(&now));
}

/**
 * @brief Prints the PETGEM banner followed by the "Run" section.
 *
 * The banner carries the project name, repository, developer and
 * affiliations. The "Run" section reports the kernel, the PETGEM version and
 * git revision, the PETSc version, the MPI rank count and the start time.
 */
PetscErrorCode printHeader(const char *kernel) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  MPI_Comm    comm = PETSC_COMM_WORLD;
  char        date[32], petsc[80];
  PetscMPIInt size;

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

  formatNow(date, sizeof(date));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (strcmp(PETSC_VERSION_GIT, "unknown") != 0) {
    PetscCall(PetscStrncpy(petsc, PETSC_VERSION_GIT, sizeof(petsc)));
  } else {
    PetscCall(PetscSNPrintf(petsc, sizeof(petsc), "%d.%d.%d", PETSC_VERSION_MAJOR, PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR));
  }

  PetscCall(logSection(comm, "Run"));
  PetscCall(logKvStr(comm, "Kernel", kernel));
  if (strcmp(PETGEM_GIT_REV, "unknown") != 0) {
    PetscCall(logKvf(comm, "PETGEM version", "%d.%d.%d (git %s)", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, PETGEM_GIT_REV));
  } else {
    PetscCall(logKvf(comm, "PETGEM version", "%d.%d.%d", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH));
  }
  PetscCall(logKvStr(comm, "PETSc version", petsc));
  PetscCall(logKvInt(comm, "MPI ranks", (PetscInt)size));
  PetscCall(logKvStr(comm, "Started", date));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints the closing banner with the finish time.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful completion, or a
 *         PETSc error code otherwise.
 */
PetscErrorCode printFooter(void) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  char date[32];

  formatNow(date, sizeof(date));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n Finished: %s\n", date));
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
 */
PetscErrorCode createDirectory(const char *path) {

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
 * @brief Prints the "Timers" section of the run report.
 *
 * One line per stage with its time in hh:mm:ss.sss and its percentage of the
 * total, followed by the total (the sum of all stages).
 */
PetscErrorCode printTimers(const char *const labels[], const PetscLogDouble times[], PetscInt n) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  PetscLogDouble total = 0.0;

  for (PetscInt i = 0; i < n; i++) total += times[i];

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n Timers (hh:mm:ss.sss |   %% |):\n"));
  for (PetscInt i = 0; i < n; i++) PetscCall(printTimerLine(labels[i], times[i], total));
  PetscCall(printTimerLine("Total", total, total));

  PetscFunctionReturn(PETSC_SUCCESS);
}
