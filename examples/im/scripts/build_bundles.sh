#!/usr/bin/env bash
# ===========================================================================
# Build the HDF5 input bundles the PETGEM solvers consume, by driving the
# general preprocessor utils/preprocess.py with this benchmark's mesh, survey
# and conductivity files.
#
# Requires the petgem Python package + meshio. Run from the REPOSITORY ROOT.
# With the petgem-env image, mount the repo at /workspace:
#
#   docker run --rm -v "$PWD":/workspace -w /workspace petgem-env:latest \
#       bash examples/im/scripts/build_bundles.sh <stage>
#
#   stage = fm  -> 7 forward bundles: true model (mesh_true.msh), one per frequency
#   stage = im  -> 1 inverse bundle : starting model (mesh.msh) + observed data
#
# The inverse stage consumes the observations produced earlier in the pipeline
# by scripts/make_observations.sh (stage 3). No observed dataset is shipped:
# the benchmark is meant to be run end to end, so the observations are always
# the ones this run generated. Override the location with OBSERVED=<path>
# relative to the case directory.
# ===========================================================================
set -euo pipefail
STAGE="${1:?stage: fm|im}"
C=examples/im
ORDER=2
FREQS="1 10 50 100 300 800 1500"
OBSERVED="${OBSERVED:-outputs/observed.h5}"        # relative to $C
mkdir -p "$C/outputs"

case "$STAGE" in
  fm)
    for F in $FREQS; do
      python3 utils/preprocess.py -mode fm -order $ORDER -case_dir $C \
        -mesh_filename geometry/mesh_true.msh \
        -receiver_filename survey/receivers.txt \
        -source_filename survey/sources_f${F}.txt \
        -sigma_file survey/sigmas_true.txt \
        -input_filename outputs/input_fm_f${F}.h5 \
        -params_filename outputs/_pp_fm_f${F}.txt
    done
    echo "=== forward bundles ready: $C/outputs/input_fm_f*.h5 ==="
    ;;
  im)
    test -f "$C/$OBSERVED" || {
      echo "missing $C/$OBSERVED"
      echo "Run the pipeline in order: build_bundles.sh fm -> run_forward.slurm"
      echo "-> make_observations.sh, which produces it."; exit 1; }
    python3 utils/preprocess.py -mode im -order $ORDER -case_dir $C \
      -mesh_filename geometry/mesh.msh \
      -receiver_filename survey/receivers.txt \
      -im_source_filename survey/sources_im.txt \
      -observed_filename $OBSERVED \
      -error_level 0.01 \
      -sigma_file survey/sigmas_im.txt \
      -input_filename outputs/input_im.h5 \
      -params_filename outputs/_pp_im.txt
    echo "=== inverse bundle ready: $C/outputs/input_im.h5 ==="
    ;;
  *) echo "stage must be fm or im"; exit 1 ;;
esac
