#!/usr/bin/env bash

# ============================================================================
# Reduced inversion test: input bundle
#
# This script builds the reduced im_csem_block bundle (order 1, 1 and 10 Hz):
# forward bundles on the true model, fm.csem with MUMPS, noisy observations
# of Ex, Hy and Zxy, and the inverse bundle.
#
# Usage (from the repository root):
#   bash tests/im/build_bundle.sh [OUT]
#
#   OUT  output bundle (default tests/im/input.h5)
#
# Notes:
#   - fm.csem is taken from PETGEM_FM_CSEM, or build/fm.csem.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${1:-$SCRIPT_DIR/input.h5}"
FM="${PETGEM_FM_CSEM:-$ROOT_DIR/build/fm.csem}"
CASE="$ROOT_DIR/examples/im_csem_block"
FREQS="1 10"

command -v python3 >/dev/null || { echo "ERROR: python3 not found."; exit 1; }
[[ -x "$FM" ]] || { echo "ERROR: fm.csem not found at $FM (build it or set PETGEM_FM_CSEM)."; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
cp -r "$CASE/geometry" "$CASE/survey" "$WORK/"
grep '^#' "$CASE/survey/sources_im.txt" > "$WORK/survey/sources_im_test.txt"
for F in $FREQS; do grep -E "^$F[[:space:]]" "$CASE/survey/sources_im.txt" >> "$WORK/survey/sources_im_test.txt"; done

for F in $FREQS; do
  python3 "$ROOT_DIR/utils/preprocess.py" -mode fm.csem -order 1 -case_dir "$WORK" \
      -mesh_filename geometry/mesh_true.msh -receiver_filename survey/receivers.txt \
      -source_filename "survey/sources_f$F.txt" -sigma_file survey/sigmas_true.txt \
      -input_filename "fm_f$F.h5" -params_filename "_pp_fm_f$F.txt" >/dev/null
  mpirun -n 2 "$FM" -input_filename "$WORK/fm_f$F.h5" -output_dir "$WORK/" -output_filename "responses_fm_f${F}_p1" \
      -dm_mat_type aij -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps >/dev/null
done
python3 "$ROOT_DIR/utils/make_observed.py" -case_dir "$WORK" -pattern "responses_fm_f{freq}_p1.h5" \
    -freqs "${FREQS// /,}" -seed 7 -error_level 0.01 -components Ex,Hy,Zxy -output_filename observed.h5 >/dev/null
python3 "$ROOT_DIR/utils/preprocess.py" -mode im.csem -order 1 -case_dir "$WORK" \
    -mesh_filename geometry/mesh.msh -receiver_filename survey/receivers.txt \
    -im_source_filename survey/sources_im_test.txt -observed_filename observed.h5 -error_level 0.01 \
    -sigma_file survey/sigmas_im.txt -input_filename input.h5 -params_filename _pp_im.txt >/dev/null
mv "$WORK/input.h5" "$OUT"
echo "wrote $OUT"
