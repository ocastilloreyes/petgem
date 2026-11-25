/*
 * Filename: im_csem_main.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Entry point of the im.csem binary; the kernel lives in src/im_csem.c.
 */

/* PETGEM functions */
#include "kernels.h"

/**
 * @brief Entry point of the im.csem binary.
 *
 * Forwards the command line to runCsemInverse().
 *
 * @param[in] argc  Argument count.
 * @param[in] argv  Argument vector.
 *
 * @return int Exit status returned by runCsemInverse().
 */
int main(int argc, char **argv) {
  return runCsemInverse(argc, argv);
}
