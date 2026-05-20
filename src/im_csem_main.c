/*
 * Filename: im_csem_main.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Standalone entry point for the legacy im.csem binary.
 */

/*
 * Notes:
 * The full inverse-kernel logic lives in runInverse (src/im_csem.c);
 * this wrapper exists only so the same translation unit can also be
 * linked into the unified petgem binary without symbol clashes.
 */
#include "kernels.h"

int main(int argc, char **argv) {
  return runInverse(argc, argv);
}
