/*
 * Filename: fm_tdcsem_main.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-12
 *
 * Description:
 * Entry point of the fm.tdcsem binary; the kernel lives in src/fm_tdcsem.c.
 */

/* PETGEM functions */
#include "kernels.h"

/**
 * @brief Entry point of the fm.tdcsem binary.
 *
 * Forwards the command line to runTdcsemForward().
 *
 * @param[in] argc  Argument count.
 * @param[in] argv  Argument vector.
 *
 * @return int Exit status returned by runTdcsemForward().
 */
int main(int argc, char **argv) {
  return runTdcsemForward(argc, argv);
}
