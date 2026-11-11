#!/usr/bin/env bash

# ============================================================================
# CSEM inversion benchmarks: input bundles (shared)
#
# Called by examples/im_csem_*/scripts/build_bundles.sh with CASE_DIR set.
#
# Usage (from the repository root):
#   CASE_DIR=examples/<case> bash examples/shared/inversion_bundles.sh <fm|im>
#
#   fm  one forward bundle per frequency on the true model (geometry/mesh_true.msh)
#   im  the inverse bundle: starting model (geometry/mesh.msh) and observed data
#
# Notes:
#   - The polynomial order is fixed to 2.
#   - OBSERVED=<path relative to the case> selects another observations file.
# ============================================================================

set -euo pipefail

STAGE="${1:?stage: fm|im}"
CASE_DIR="${CASE_DIR:?CASE_DIR not set}"
CASE="$(basename "${CASE_DIR}")"
ORDER=2
FREQS="1 10 50 100 300 800 1500"
OBSERVED="${OBSERVED:-outputs/observed.h5}"

mkdir -p "${CASE_DIR}/outputs"

echo "============================================================"
echo "Building the ${CASE} input bundles (stage: ${STAGE})"
echo "============================================================"
echo "Case directory : ${CASE_DIR}"
echo "Order          : ${ORDER}"
echo

case "${STAGE}" in
  fm)
    for F in ${FREQS}; do
      python3 utils/preprocess.py -mode fm.csem -order "${ORDER}" -case_dir "${CASE_DIR}" \
          -mesh_filename     geometry/mesh_true.msh \
          -receiver_filename survey/receivers.txt \
          -source_filename   "survey/sources_f${F}.txt" \
          -sigma_file        survey/sigmas_true.txt \
          -input_filename    "outputs/input_fm_f${F}.h5" \
          -params_filename   "outputs/_pp_fm_f${F}.txt"
    done
    echo
    echo "Forward bundles ready: ${CASE_DIR}/outputs/input_fm_f*.h5"
    ;;
  im)
    if [[ ! -f "${CASE_DIR}/${OBSERVED}" ]]; then
      echo "ERROR: missing ${CASE_DIR}/${OBSERVED}"
      echo "Run the pipeline in order:"
      echo "  build_bundles.sh fm -> run_fm.slurm -> make_observations.sh"
      exit 1
    fi
    python3 utils/preprocess.py -mode im.csem -order "${ORDER}" -case_dir "${CASE_DIR}" \
        -mesh_filename      geometry/mesh.msh \
        -receiver_filename  survey/receivers.txt \
        -im_source_filename survey/sources_im.txt \
        -observed_filename  "${OBSERVED}" \
        -error_level        0.01 \
        -sigma_file         survey/sigmas_im.txt \
        -input_filename     outputs/input_im.h5 \
        -params_filename    outputs/_pp_im.txt
    echo
    echo "Inverse bundle ready: ${CASE_DIR}/outputs/input_im.h5"
    ;;
  *)
    echo "ERROR: stage must be fm or im"
    exit 1
    ;;
esac
