#!/usr/bin/env bash

# ============================================================================
# fm_csem_unit_cube test dataset: goldens
#
# This script regenerates the golden responses committed in reference/.
# They are exact serial LU solves of the current code at orders 1, 2 and 3.
# Regenerate them only from a trusted build/fm.csem, and only when the
# forward result changes for a legitimate reason (see tests/README.md).
#
# Usage (from the repository root):
#   bash examples/fm_csem_unit_cube/scripts/regen_golden.sh
#
# Required input:
#   outputs/input.h5
#     Built with scripts/build_bundles.sh if it is missing
#   build/fm.csem
#
# Generated output:
#   reference/responses_p1.h5
#   reference/responses_p2.h5
#   reference/responses_p3.h5
# ============================================================================

set -euo pipefail

CASE_DIR=examples/fm_csem_unit_cube
EXECUTABLE=build/fm.csem

# Build the input bundle if it is not there yet.
if [[ ! -f "${CASE_DIR}/outputs/input.h5" ]]; then
  bash "${CASE_DIR}/scripts/build_bundles.sh"
fi

echo "============================================================"
echo "Regenerating the fm_csem_unit_cube reference solutions"
echo "============================================================"
echo "Executable   : ${EXECUTABLE}"
echo "Orders       : 1 2 3"
echo

for N in 1 2 3; do
  "${EXECUTABLE}" \
      -input_filename "${CASE_DIR}/outputs/input.h5" -order "${N}" \
      -output_dir "${CASE_DIR}/reference" -output_filename "responses_p${N}" \
      -dm_mat_type aij -ksp_type preonly -pc_type lu \
      -pc_factor_mat_ordering_type nd -ksp_error_if_not_converged
done

echo
echo "Reference solutions updated in ${CASE_DIR}/reference"
