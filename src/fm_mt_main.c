/*
 * Filename: fm_mt_main.c
 * Author: Octavio Castillo Reyes (UPC/BSC)
 * Date: 2026-10-09
 *
 * Description:
 * Standalone entry point for the fm.mt binary.
 */

/*
 * Notes:
 * The full MT forward-kernel logic lives in runMtForward (src/fm_mt.c).
 */
#include "kernels.h"

/**
 * @brief Standalone entry point for fm.mt; delegates to runMtForward.
 *
 * @param[in] argc  Argument count.
 * @param[in] argv  Argument vector.
 *
 * @return int the PetscErrorCode (cast to int) as the process exit status.
 */
int main(int argc, char **argv) {
  return runMtForward(argc, argv);
}
