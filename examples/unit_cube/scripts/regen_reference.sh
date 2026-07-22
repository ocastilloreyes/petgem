#!/usr/bin/env bash
# ===========================================================================
# Regenerate the committed golden references in reference/ (responses_p{1,2,3}.h5).
#
#   bash examples/unit_cube/scripts/regen_reference.sh
#
# The goldens are exact serial LU solves of the current code. Regenerate them
# from a TRUSTED build/fm.csem whenever the forward result legitimately changes
# (see tests/README.md). Builds the input bundle first if it is missing.
# Run from the REPOSITORY ROOT.
# ===========================================================================
set -euo pipefail
C=examples/unit_cube
test -f "$C/outputs/input.h5" || bash "$C/scripts/build_bundles.sh" 1

for N in 1 2 3; do
  build/fm.csem \
    -input_filename "$C/outputs/input.h5" -order "$N" \
    -output_dir "$C/reference" -output_filename "responses_p$N" \
    -dm_mat_type aij -ksp_type preonly -pc_type lu \
    -pc_factor_mat_ordering_type nd -ksp_error_if_not_converged
done

echo "=== goldens refreshed in $C/reference ==="
