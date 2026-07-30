#!/usr/bin/env bash
# ===========================================================================
# Pipeline stage 3: turn the forward responses into the synthetic observations
# the inversion targets, by adding Gaussian noise.
#
#   true model -> forward modelling -> [ synthetic observations + noise ] ->
#   inversion -> recovered model
#
# Drives the general utility utils/make_observed.py with this benchmark's fixed
# parameters, so the noise realisation is reproducible: the seed is pinned here
# and recorded in the output file, and reference/reference_metrics.json quotes
# it. Changing SEED or ERROR_LEVEL invalidates the reference metrics.
#
# Requires the petgem Python package. Run from the REPOSITORY ROOT:
#
#   docker run --rm -v "$PWD":/workspace -w /workspace petgem-env:latest \
#       bash examples/im/scripts/make_observations.sh
#
# Consumes examples/im/outputs/responses_fm_f<F>_p2.h5  (run_forward.slurm)
# Produces examples/im/outputs/observed.h5              (-> build_bundles.sh im)
# ===========================================================================
set -euo pipefail
C=examples/im
FREQS=1,10,50,100,300,800,1500
SEED=20260720
ERROR_LEVEL=0.01

missing=0
for F in ${FREQS//,/ }; do
  [ -f "$C/outputs/responses_fm_f${F}_p2.h5" ] || { echo "missing $C/outputs/responses_fm_f${F}_p2.h5"; missing=1; }
done
[ "$missing" -eq 0 ] || {
  echo "Run the forward stage first: build_bundles.sh fm -> run_forward.slurm"; exit 1; }

python3 utils/make_observed.py \
    -case_dir "$C" \
    -pattern  "outputs/responses_fm_f{freq}_p2.h5" \
    -freqs    "$FREQS" \
    -seed     "$SEED" -error_level "$ERROR_LEVEL" \
    -out      outputs/observed.h5

echo "=== observations ready: $C/outputs/observed.h5 (seed $SEED, noise $ERROR_LEVEL) ==="
