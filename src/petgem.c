/*
 * Filename: petgem.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Unified PETGEM entry point. Runs the kernel named by a positional
 * subcommand or by the -mode option: fm.csem, im.csem, fm.mt or fm.tdcsem, the same
 * names as the standalone binaries.
 */

/*
 * Usage:
 *   ./petgem fm.csem|im.csem|fm.mt|fm.tdcsem -options_file ...
 *   ./petgem -mode fm.csem|im.csem|fm.mt|fm.tdcsem -options_file ...
 * The positional form takes precedence.
 */

/* C libraries */
#include <stdio.h>
#include <string.h>

/* PETGEM functions */
#include "kernels.h"
#include "runtime.h"

#define NUM_KERNELS ((int)(sizeof(kernels) / sizeof(kernels[0])))

typedef int (*KernelMain)(int argc, char **argv);

static const struct {
  const char *name;
  KernelMain  run;
  const char *summary;
} kernels[] = {
    {"fm.csem", runCsemForward, "CSEM forward modeling"},
    {"im.csem", runCsemInverse, "CSEM inverse modeling"},
    {"fm.mt", runMtForward, "MT forward modeling"},
    {"fm.tdcsem", runTdcsemForward, "Time-domain CSEM forward modeling"},
};

/**
 * @brief Looks up a kernel by name.
 *
 * @param[in] s  Kernel name (fm.csem, im.csem, fm.mt, fm.tdcsem).
 *
 * @return KernelMain Entry point of the kernel, or NULL when s is not a kernel
 *         name.
 */
static KernelMain findKernel(const char *s) {
  for (int i = 0; i < NUM_KERNELS; i++) {
    if (strcmp(s, kernels[i].name) == 0) return kernels[i].run;
  }
  return NULL;
}

/**
 * @brief Prints the usage of the unified binary.
 *
 * @param[in] progname  Program name (argv[0]).
 *
 * @note Uses printf, since it runs before PetscInitialize.
 */
static void printUsage(const char *progname) {
  printf("Usage:\n");
  for (int i = 0; i < NUM_KERNELS; i++) {
    printf("  %s %-9s [petsc options...]   # %s\n", progname, kernels[i].name, kernels[i].summary);
  }
  printf("  %s -mode <kernel> [petsc options...]\n"
         "  %s --version\n"
         "\n"
         "Kernel names match the standalone binaries (fm.csem, im.csem, fm.mt, fm.tdcsem).\n"
         "Pass -options_file <file.txt> for the usual params input.\n",
         progname, progname);
}

/**
 * @brief Reports an unknown kernel name and prints the usage.
 *
 * @param[in] progname  Program name (argv[0]).
 * @param[in] what      Where the name came from ("subcommand" or "-mode
 *                      value").
 * @param[in] name      Name given on the command line.
 *
 * @return int Exit status 2.
 */
static int unknownKernel(const char *progname, const char *what, const char *name) {
  fprintf(stderr, "PETGEM: unknown %s '%s' (expected fm.csem, im.csem, fm.mt or fm.tdcsem)\n", what, name);
  printUsage(progname);
  return 2;
}

/**
 * @brief Removes argv[first .. first + count - 1], keeping argv
 *        NULL-terminated.
 *
 * @param[in,out] argc   Argument count.
 * @param[in,out] argv   Argument vector.
 * @param[in]     first  First argument to remove.
 * @param[in]     count  Number of arguments to remove.
 */
static void removeArgs(int *argc, char **argv, int first, int count) {
  for (int i = first; i + count <= *argc; i++) argv[i] = argv[i + count];
  *argc -= count;
}

/**
 * @brief Entry point of the unified petgem binary.
 *
 * Selects the kernel from the first positional argument or from -mode, removes
 * it from the command line and runs the kernel.
 *
 * @param[in] argc  Argument count.
 * @param[in] argv  Argument vector.
 *
 * @return int Exit status of the kernel; 2 when no valid kernel is given.
 */
int main(int argc, char **argv) {
  KernelMain run = NULL;

  if (isVersionRequest(argc, argv)) return printVersion();
  if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    printUsage(argv[0]);
    return 0;
  }

  if (argc > 1 && argv[1][0] != '-') {
    run = findKernel(argv[1]);
    if (!run) return unknownKernel(argv[0], "subcommand", argv[1]);
    removeArgs(&argc, argv, 1, 1);
  } else {
    for (int i = 1; i < argc - 1; i++) {
      if (strcmp(argv[i], "-mode") == 0) {
        run = findKernel(argv[i + 1]);
        if (!run) return unknownKernel(argv[0], "-mode value", argv[i + 1]);
        removeArgs(&argc, argv, i, 2);
        break;
      }
    }
  }

  if (!run) {
    fprintf(stderr, "PETGEM: no kernel specified.\n");
    printUsage(argv[0]);
    return 2;
  }
  return run(argc, argv);
}
