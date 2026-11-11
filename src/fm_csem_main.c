/*
 * Filename: fm_csem_main.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Entry point of the fm.csem binary; the kernel lives in src/fm_csem.c.
 */

#include "kernels.h"

int main(int argc, char **argv) {
  return runForward(argc, argv);
}
