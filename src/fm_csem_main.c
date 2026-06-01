/*
 * Filename: fm_csem_main.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-05-20
 *
 * Description:
 * Standalone entry point for the legacy fm.csem binary.
 */

/*
 * Notes:
 * The full forward-kernel logic lives in runForward (src/fm_csem.c);
 * this wrapper exists only so the same translation unit can also be
 * linked into the unified petgem binary without symbol clashes.
 */
#include "kernels.h"

/**
 * @brief Standalone entry point for fm.csem; delegates to runForward.
 *
 * @param[in] argc  Argument count.
 * @param[in] argv  Argument vector.
 *
 * @return int the PetscErrorCode (cast to int) as the process exit status.
 */
int main(int argc, char **argv) {
  return runForward(argc, argv);
}
