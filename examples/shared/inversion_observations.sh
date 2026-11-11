#!/usr/bin/env bash

# ============================================================================
# CSEM inversion benchmarks: synthetic observations (shared)
#
# Adds Gaussian noise to the forward responses with utils/make_observed.py.
# Called by examples/im_csem_*/scripts/make_observations.sh with CASE_DIR and
# SEED set.
#
# Usage (from the repository root):
#   CASE_DIR=examples/<case> SEED=<n> bash examples/shared/inversion_observations.sh
# ============================================================================

set -euo pipefail

CASE_DIR="${CASE_DIR:?CASE_DIR not set}"
SEED="${SEED:?SEED not set}"
FREQS=1,10,50,100,300,800,1500
ERROR_LEVEL=0.01

missing=0
for F in ${FREQS//,/ }; do
  if [[ ! -f "${CASE_DIR}/outputs/responses_fm_f${F}_p2.h5" ]]; then
    echo "ERROR: missing ${CASE_DIR}/outputs/responses_fm_f${F}_p2.h5"
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "Run the forward stage first:"
  echo "  build_bundles.sh fm -> run_fm.slurm"
  exit 1
fi

echo "============================================================"
echo "Generating synthetic observations"
echo "============================================================"
echo "Case directory : ${CASE_DIR}"
echo "Frequencies    : ${FREQS}"
echo "Seed           : ${SEED}"
echo "Noise level    : ${ERROR_LEVEL}"
echo

python3 utils/make_observed.py \
    -case_dir        "${CASE_DIR}" \
    -pattern         "outputs/responses_fm_f{freq}_p2.h5" \
    -freqs           "${FREQS}" \
    -seed            "${SEED}" \
    -error_level     "${ERROR_LEVEL}" \
    -output_filename outputs/observed.h5

echo
echo "Observations ready: ${CASE_DIR}/outputs/observed.h5"
