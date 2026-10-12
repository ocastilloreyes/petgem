#!/usr/bin/env bash

# ============================================================================
# fm_csem_unit_cube test dataset: input bundle
#
# This script meshes geometry/mesh.geo with gmsh and then calls
# utils/preprocess.py to assemble the mesh and survey into the HDF5 bundle
# read by fm.csem. CI runs exactly these steps before the e2e and Extrae
# jobs (see .github/workflows/tests-fm-csem.yml and tests/README.md).
#
# Usage (from the repository root):
#   bash examples/fm_csem_unit_cube/scripts/build_bundles.sh
#
#   With the petgem-env image:
#   docker run --rm -u $(id -u):$(id -g) -v "$PWD":/workspace -w /workspace \
#       petgem-env:latest bash examples/fm_csem_unit_cube/scripts/build_bundles.sh
#
# Required input:
#   geometry/mesh.geo
#   survey/sources.txt, survey/receivers.txt, survey/sigmas.txt
#
# Generated output:
#   outputs/input.h5
#   outputs/_pp.txt (temporary preprocessor parameters)
#
# Notes:
#   - Requires gmsh, numpy, meshio, h5py and petsc4py (all in petgem-env).
#   - fm.csem takes the polynomial order at run time (-order N), so one
#     bundle serves every order.
# ============================================================================

set -euo pipefail

CASE_DIR=examples/fm_csem_unit_cube

mkdir -p "${CASE_DIR}/outputs"

echo "============================================================"
echo "Building the fm_csem_unit_cube input bundle"
echo "============================================================"
echo "Case directory : ${CASE_DIR}"
echo

# Generate the mesh.
gmsh -3 "${CASE_DIR}/geometry/mesh.geo" -o "${CASE_DIR}/outputs/mesh.msh"

# Assemble mesh and survey into the HDF5 bundle.
python3 utils/preprocess.py -mode fm.csem -order 1 -case_dir "${CASE_DIR}" \
    -mesh_filename     outputs/mesh.msh \
    -source_filename   survey/sources.txt \
    -receiver_filename survey/receivers.txt \
    -sigma_file        survey/sigmas.txt \
    -input_filename    outputs/input.h5 \
    -params_filename   outputs/_pp.txt

echo
echo "Input bundle ready: ${CASE_DIR}/outputs/input.h5"
