/*
 * Filename: petgem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Unified PETGEM entry point. Dispatches to the forward or inverse
 * kernel based on a positional subcommand or the -mode PETSc option.
 */

/*
 * Usage:
 *   - positional subcommand:   ./petgem modeling -options_file ...
 *                              ./petgem inverse  -options_file ...
 *   - PETSc option:            ./petgem -mode modeling -options_file ...
 *                              ./petgem -mode inverse  -options_file ...
 * Both forms are accepted; positional takes precedence if present.
 *
 * The legacy single-purpose binaries (fm.csem, im.csem) keep working
 * unchanged and call the same runForward / runInverse functions.
 */

#include <stdio.h>
#include <string.h>

#include <petsc.h>

#include "kernels.h"
#include "version.h"
#include "common.h" 

/**
 * @brief Unified PETGEM dispatcher entry point.
 *
 * Parses --version / --help / positional-subcommand / -mode and forwards
 * argv (with the mode token stripped) to runForward or runInverse.
 *
 * @param[in] argc  Argument count.
 * @param[in] argv  Argument vector.
 *
 * @return int the kernel's exit status, 0 for --version/--help, or 2 on
 *         invalid CLI usage.
 */
int main(int argc, char **argv) {
  /* --version short-circuit (no PETSc init) */
  if (argc > 1 && strcmp(argv[1], "--version") == 0) {
    printf("PETGEM version %d.%d.%d\n", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH);
    return 0;
  }

  if (argc > 1 && (strcmp(argv[1], "--help") == 0 ||
                   strcmp(argv[1], "-h")     == 0)) {
    printUsage(argv[0]);
    return 0;
  }

  /* (a) positional subcommand: argv[1] is "modeling"/"inverse"/aliases.
   *     We splice it out of argv before forwarding so that the kernel
   *     doesn't see an unrecognized first token.                       */
  PetscInt mode = -1;
  if (argc > 1 && argv[1][0] != '-') {
    mode = parseModeArg(argv[1], &mode);
    if (mode < 0) {
      fprintf(stderr, "PETGEM: unknown subcommand '%s' (expected 'modeling' or 'inverse')\n", argv[1]);
      printUsage(argv[0]);
      return 2;
    }
    /* Shift argv left by one, keeping argv[0] as program name.        */
    for (PetscInt i = 1; i < argc - 1; i++) {
      argv[i] = argv[i + 1];
    }
    argv[argc - 1] = NULL;
    argc -= 1;
  }

  /* (b) PETSc -mode option fallback (works with options_file too).
   *     Done WITHOUT a full PetscInitialize: scan argv directly so we
   *     can dispatch before either kernel takes over the lifecycle.   */
  if (mode < 0) {
    for (PetscInt i = 1; i < argc - 1; i++) {
      if (strcmp(argv[i], "-mode") == 0) {
        mode = parseModeArg(argv[i + 1], &mode);
        if (mode < 0) {
          fprintf(stderr, "PETGEM: -mode value '%s' is invalid (expected 'modeling' or 'inverse')\n", argv[i + 1]);
          return 2;
        }
        /* Strip "-mode <value>" from argv so the kernel doesn't see it.
         * Both kernels' options-file readers tolerate unknown options
         * (PETSc warns rather than errors), but stripping keeps the
         * downstream log clean.                                       */
        for (PetscInt j = i; j < argc - 2; j++) {
          argv[j] = argv[j + 2];
        }
        argv[argc - 2] = NULL;
        argv[argc - 1] = NULL;
        argc -= 2;
        break;
      }
    }
  }

  if (mode < 0) {
    fprintf(stderr, "PETGEM: no mode specified.\n");
    printUsage(argv[0]);
    return 2;
  }

  return (mode == 0) ? runForward(argc, argv)
                     : runInverse(argc, argv);
}
