/*
 * Filename: fm_mt_main.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * Entry point of the fm.mt binary; the kernel lives in src/fm_mt.c.
 */

#include "kernels.h"

int main(int argc, char **argv) {
  return runMtForward(argc, argv);
}
