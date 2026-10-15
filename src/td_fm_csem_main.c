/*
 * Filename: td_fm_csem_main.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-12
 *
 * Description:
 * Entry point of the fm.tdcsem binary; the kernel lives in src/td_fm_csem.c.
 */

#include "kernels.h"

int main(int argc, char **argv) {
  return runTdCsemForward(argc, argv);
}
