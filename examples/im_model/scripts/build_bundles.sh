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
#       bash examples/im_model/scripts/build_bundles.sh <stage>
#
#   stage = fm  -> 7 forward bundles: true model (im_true.msh), one per frequency
#   stage = im  -> 1 inverse bundle : starting model (im_model.msh) + observed data
#
# The inverse stage needs the observed dataset. By default it uses the shipped
# reference/observed.h5, so the inversion can be built without re-running the
# forward stage; override with OBSERVED=outputs/observed.h5 for a from-scratch
# run (see utils/make_observed.py).
# ===========================================================================
set -euo pipefail
STAGE="${1:?stage: fm|im}"
C=examples/im_model
ORDER=2
FREQS="1 10 50 100 300 800 1500"
OBSERVED="${OBSERVED:-reference/observed.h5}"      # relative to $C
mkdir -p "$C/outputs"

case "$STAGE" in
  fm)
    for F in $FREQS; do
      python3 utils/preprocess.py -mode fm -order $ORDER -case_dir $C \
        -mesh_filename geometry/im_true.msh \
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
      echo "missing $C/$OBSERVED - generate it with utils/make_observed.py,"
      echo "or use the shipped reference/observed.h5 (the default)."; exit 1; }
    python3 utils/preprocess.py -mode im -order $ORDER -case_dir $C \
      -mesh_filename geometry/im_model.msh \
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
