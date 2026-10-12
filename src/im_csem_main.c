/*
 * Filename: im_csem_main.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Entry point of the im.csem binary; the kernel lives in src/im_csem.c.
 */

#include "kernels.h"

int main(int argc, char **argv) {
  return runInverse(argc, argv);
}
