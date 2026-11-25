/*
 * Filename: fm_mt_main.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * Entry point of the fm.mt binary; the kernel lives in src/fm_mt.c.
 */

/* PETGEM functions */
#include "kernels.h"

/**
 * @brief Entry point of the fm.mt binary.
 *
 * Forwards the command line to runMtForward().
 *
 * @param[in] argc  Argument count.
 * @param[in] argv  Argument vector.
 *
 * @return int Exit status returned by runMtForward().
 */
int main(int argc, char **argv) {
  return runMtForward(argc, argv);
}
