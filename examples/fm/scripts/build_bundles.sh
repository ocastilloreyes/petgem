#!/usr/bin/env bash
# ===========================================================================
# Build the HDF5 input bundle fm.csem consumes, by meshing geometry/mesh.geo
# with gmsh and driving the general preprocessor utils/preprocess.py with this
# benchmark's mesh, survey and conductivity files.
#
# Requires gmsh + the petgem Python package (meshio). Run from the REPOSITORY
# ROOT. With the petgem-env image, mount the repo at /workspace:
#
#   docker run --rm -v "$PWD":/workspace -w /workspace petgem-env:latest \
#       bash examples/fm/scripts/build_bundles.sh [order]
#
# Produces examples/fm/outputs/input.h5 (+ a throwaway preprocess params
# file). The committed solver options live in configs/params.txt; run the solve
# with scripts/run_forward.slurm (or -options_file configs/params.txt).
# ===========================================================================
set -euo pipefail
C=examples/fm
ORDER="${1:-1}"
mkdir -p "$C/outputs"

gmsh -3 "$C/geometry/mesh.geo" -o "$C/outputs/mesh.msh"

python3 utils/preprocess.py -mode fm -order "$ORDER" -case_dir "$C" \
  -mesh_filename     outputs/mesh.msh \
  -source_filename   survey/sources.txt \
  -receiver_filename survey/receivers.txt \
  -sigma_file        survey/sigmas.txt \
  -input_filename    outputs/input.h5 \
  -params_filename   outputs/_pp_p${ORDER}.txt

echo "=== fm bundle ready: $C/outputs/input.h5 ==="
