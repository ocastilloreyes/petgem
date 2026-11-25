/*
 * Filename: fm_csem_main.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Entry point of the fm.csem binary; the kernel lives in src/fm_csem.c.
 */

/* PETGEM functions */
#include "kernels.h"

/**
 * @brief Entry point of the fm.csem binary.
 *
 * Forwards the command line to runCsemForward().
 *
 * @param[in] argc  Argument count.
 * @param[in] argv  Argument vector.
 *
 * @return int Exit status returned by runCsemForward().
 */
int main(int argc, char **argv) {
  return runCsemForward(argc, argv);
}
