#!/usr/bin/env bash

# ============================================================================
# Marine CSEM benchmark: input bundle
#
# This script meshes geometry/mesh.geo with gmsh and then calls the general
# preprocessor utils/preprocess.py with the mesh, survey and conductivity
# files of this benchmark. The result is the HDF5 bundle read by fm.csem.
#
# Usage (from the repository root):
#   bash examples/fm/scripts/build_bundles.sh [order]
#
#   With the petgem-env image:
#   docker run --rm -u $(id -u):$(id -g) -v "$PWD":/workspace -w /workspace \
#       petgem-env:latest bash examples/fm/scripts/build_bundles.sh [order]
#
# Required input:
#   geometry/mesh.geo
#   survey/sources.txt, survey/receivers.txt, survey/sigmas.txt
#
# Generated output:
#   outputs/input.h5
#   outputs/_pp_p<order>.txt (temporary preprocessor parameters)
#
# Next step:
#   sbatch scripts/run_fm.slurm   (from examples/fm)
#
# Notes:
#   - Requires gmsh and the petgem Python package (meshio).
#   - The polynomial order defaults to 1.
#   - The solver options used in the simulation live in configs/params.txt.
# ============================================================================

set -euo pipefail

CASE_DIR=examples/fm
ORDER="${1:-1}"

mkdir -p "${CASE_DIR}/outputs"

echo "============================================================"
echo "Building the marine CSEM input bundle"
echo "============================================================"
echo "Case directory : ${CASE_DIR}"
echo "Order          : ${ORDER}"
echo

# Generate the mesh.
gmsh -3 "${CASE_DIR}/geometry/mesh.geo" -o "${CASE_DIR}/outputs/mesh.msh"

# Assemble mesh, survey and conductivities into the HDF5 bundle.
python3 utils/preprocess.py -mode fm -order "${ORDER}" -case_dir "${CASE_DIR}" \
    -mesh_filename     outputs/mesh.msh \
    -source_filename   survey/sources.txt \
    -receiver_filename survey/receivers.txt \
    -sigma_file        survey/sigmas.txt \
    -input_filename    outputs/input.h5 \
    -params_filename   "outputs/_pp_p${ORDER}.txt"

echo
echo "Input bundle ready: ${CASE_DIR}/outputs/input.h5"
