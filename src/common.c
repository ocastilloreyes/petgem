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

/* PETGEM funcions*/
#include "common.h"
#include "git_rev.h"
#include "version.h"

#define KEY_WIDTH   12
#define STAGE_WIDTH 56

/**
 * @brief Computes the display width of a UTF-8 string in characters.
 *
 * This function calculates the number of printable characters in
 * the input null-terminated string, assuming UTF-8 encoding. It
 * counts only the leading bytes of multi-byte UTF-8 characters,
 * effectively providing the number of characters as they would
 * appear on the console.
 *
 * Used by logStage() to align rows containing multi-byte characters.
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


const char *formatReal(PetscReal value) {
  enum { NUM_BUFS = 8, BUF_LEN = 32 };
  static char bufs[NUM_BUFS][BUF_LEN];
  static PetscInt which = 0;
  char *out = bufs[which];
  which = (which + 1) % NUM_BUFS;

  /* snprintf, not PetscSNPrintf: PETSc's %g handling renders a whole value as
   * "1." (bare trailing point). Naming the precision gives plain C behaviour,
   * so 1.0 arrives here as "1". */
  snprintf(out, BUF_LEN, "%.6g", (double)value);

  /* Restore the float signal that %g drops. Anything already carrying a point,
   * an exponent, or being nan/inf is left untouched. */
  for (const char *p = out; *p; p++) {
    if (*p == '.' || *p == 'e' || *p == 'E' || *p == 'n' || *p == 'i') return out;
  }
  size_t len = strlen(out);
  if (len + 2 < BUF_LEN) {
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
 *
 * @param[in] value  Real to format.
 *
 * @return Pointer to a NUL-terminated string (do not free).
 */
const char *formatCompactReal(PetscReal value) {
  enum { NUM_BUFS = 8, BUF_LEN = 32 };
  static char bufs[NUM_BUFS][BUF_LEN];
  static PetscInt which = 0;
  char *out = bufs[which];
  which = (which + 1) % NUM_BUFS;
  snprintf(out, BUF_LEN, "%.6g", (double)value);
  return out;
}


/**
 * @brief Prints a string-valued key/value entry in the PETGEM run report.
 *
 * Formats a report line using the standard PETGEM layout
 * (" %-12s = %s") so all modules produce aligned output.
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
  PetscCall(PetscPrintf(comm, " %-*s = %s\n", KEY_WIDTH, key, val));
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
  PetscCall(PetscPrintf(comm, " %-*s = %s\n", KEY_WIDTH, key, formatGroupedInt(val)));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Prints a floating-point key/value entry in the PETGEM run report.
 *
 * Renders the value through formatReal(), so it stays visibly a real: whole
 * values print as "1.0" rather than "1", keeping them distinct from the counts
 * logKVInt() prints. Intended for tolerances, weights, frequencies, physical
 * parameters, and other scalar quantities.
 *
 * @param[in] comm  MPI communicator used by PetscPrintf().
 * @param[in] key   Entry label displayed in the left column.
 * @param[in] val   Floating-point value to display.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode logKVReal(MPI_Comm comm, const char *key, PetscReal val) {
  PetscFunctionBeginUser;
  /* formatReal, not %g: PETSc's %g renders whole values as "1." (bare trailing
   * point), and a plain "%.6g" would render them as "1", which reads like a
   * count. formatReal gives "1.0", keeping reals distinguishable from the
   * integers that logKVInt prints. */
  PetscCall(PetscPrintf(comm, " %-*s = %s\n", KEY_WIDTH, key, formatReal(val)));
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
  PetscCall(PetscPrintf(comm, " %-*s = %s\n", KEY_WIDTH, key, buf));
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
 * @brief Prints the two-line PETGEM run header.
 *
 * Line 1: PETGEM version, git revision, kernel name and affiliation.
 * Line 2: start time, MPI rank count and PETSc version.
 *
 * @param[in] kernel  Kernel name shown in the header (e.g. "fm.csem").
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful
 *         completion, or a PETSc error code otherwise.
 */
PetscErrorCode printHeader(const char *kernel) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  char        date[32], rev[80] = "", petsc[80];
  PetscMPIInt size;

  formatNow(date, sizeof(date));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  if (strcmp(PETGEM_GIT_REV, "unknown") != 0) {
    PetscCall(PetscSNPrintf(rev, sizeof(rev), " (git %s)", PETGEM_GIT_REV));
  }
  if (strcmp(PETSC_VERSION_GIT, "unknown") != 0) {
    PetscCall(PetscStrncpy(petsc, PETSC_VERSION_GIT, sizeof(petsc)));
  } else {
    PetscCall(PetscSNPrintf(petsc, sizeof(petsc), "%d.%d.%d", PETSC_VERSION_MAJOR, PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, " PETGEM %d.%d.%d%s · %s · UPC / BSC\n",
                        VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, rev, kernel));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Started %s · %d rank%s · PETSc %s\n\n", date, (int)size, size == 1 ? "" : "s", petsc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints the closing line with the finish time.
 *
 * @return PetscErrorCode PETSC_SUCCESS on successful completion, or a
 *         PETSc error code otherwise.
 */
PetscErrorCode printFooter(void) {

  PetscFunctionBeginUser;

  /* Variables declaration */
  char date[32];

  formatNow(date, sizeof(date));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Finished %s\n", date));

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
 * @brief Formats an elapsed time: "1.23 s" below one minute, "h:mm:ss" above.
 *
 * Renders into one of a few rotating internal buffers. Not thread-safe.
 *
 * @param[in] t  Elapsed time in seconds.
 *
 * @return Pointer to a NUL-terminated string (do not free).
 */
const char *formatElapsed(PetscLogDouble t) {
  enum { NUM_BUFS = 4, BUF_LEN = 32 };
  static char bufs[NUM_BUFS][BUF_LEN];
  static PetscInt which = 0;
  char *out = bufs[which];
  which = (which + 1) % NUM_BUFS;

  if (t < 60.0) {
    snprintf(out, BUF_LEN, "%.2f s", (double)t);
  } else {
    long s = (long)(t + 0.5);
    snprintf(out, BUF_LEN, "%ld:%02ld:%02ld", s / 3600, (s / 60) % 60, s % 60);
  }
  return out;
}

/**
 * @brief Prints the header row of the stage/time table.
 *
 * @param[in] comm  MPI communicator used by PetscPrintf().
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode logStageHeader(MPI_Comm comm) {
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(comm, "\n %-*s %10s\n", STAGE_WIDTH, "Stage", "time"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Prints one row of the stage/time table.
 *
 * The row reads "<label> <detail>" padded to the stage column, followed by the
 * elapsed time right-aligned. A NULL @p detail prints the label alone.
 *
 * @param[in] comm    MPI communicator used by PetscPrintf().
 * @param[in] label   Stage name.
 * @param[in] detail  Optional free-text detail (may be NULL).
 * @param[in] t       Elapsed time in seconds.
 *
 * @return PETSC_SUCCESS on success, or a PETSc error code otherwise.
 */
PetscErrorCode logStage(MPI_Comm comm, const char *label, const char *detail, PetscLogDouble t) {
  PetscFunctionBeginUser;

  /* Variables declaration */
  char     text[256];
  PetscInt width;
  int      pad;

  if (detail) {
    PetscCall(PetscSNPrintf(text, sizeof(text), "%-10s %s", label, detail));
  } else {
    PetscCall(PetscStrncpy(text, label, sizeof(text)));
  }
  PetscCall(computeDisplayWidth(text, &width));
  pad = STAGE_WIDTH - (int)width;
  if (pad < 1) pad = 1;
  PetscCall(PetscPrintf(comm, " %s%*s %10s\n", text, pad, "", formatElapsed(t)));

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