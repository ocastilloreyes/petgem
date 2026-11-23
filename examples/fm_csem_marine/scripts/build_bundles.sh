#!/usr/bin/env bash

# ============================================================================
# Marine CSEM model: input bundle
#
# This script meshes geometry/mesh.geo with gmsh and then calls the general
# preprocessor utils/preprocess.py with the mesh, survey and conductivity
# files of this model. The result is the HDF5 bundle read by fm.csem.
#
# Usage (from the repository root):
#   bash examples/fm_csem_marine/scripts/build_bundles.sh [order] [secondary]
#
#   With the petgem-env image:
#   docker run --rm -u $(id -u):$(id -g) -v "$PWD":/workspace -w /workspace \
#       petgem-env:latest bash examples/fm_csem_marine/scripts/build_bundles.sh [order] [secondary]
#
# Required input:
#   geometry/mesh.geo
#   survey/sources.txt, survey/receivers.txt, survey/sigmas.txt
#   survey/background_layers.txt, survey/background_sigmas.txt (secondary)
#
# Generated output:
#   outputs/input.h5
#   outputs/_pp_p<order>.txt (temporary preprocessor parameters)
#
# Next step:
#   sbatch scripts/run_fm.slurm   (from examples/fm_csem_marine)
#
# Notes:
#   - Requires gmsh, numpy, meshio, h5py and petsc4py (all in petgem-env).
#   - The polynomial order defaults to 1.
#   - secondary embeds the background model (/background) for
#     configs/params_secondary.txt.
#   - The solver options used in the simulation live in configs/params.txt.
# ============================================================================

set -euo pipefail

CASE_DIR=examples/fm_csem_marine
ORDER="${1:-1}"
BACKGROUND=()
case "${2:-}" in
  "") ;;
  secondary) BACKGROUND=(-background_layers_file survey/background_layers.txt -background_sigma_file survey/background_sigmas.txt) ;;
  *) echo "ERROR: unknown option ${2} (expected: secondary)" >&2; exit 1 ;;
esac

mkdir -p "${CASE_DIR}/outputs"

echo "============================================================"
echo "Building the marine CSEM input bundle"
echo "============================================================"
echo "Case directory : ${CASE_DIR}"
echo "Order          : ${ORDER}"
echo "Background     : ${2:-none}"
echo

# Generate the mesh.
gmsh -3 "${CASE_DIR}/geometry/mesh.geo" -o "${CASE_DIR}/outputs/mesh.msh"

# Assemble mesh, survey and conductivities into the HDF5 bundle.
python3 utils/preprocess.py -mode fm.csem -order "${ORDER}" -case_dir "${CASE_DIR}" \
    -mesh_filename     outputs/mesh.msh \
    -source_filename   survey/sources.txt \
    -receiver_filename survey/receivers.txt \
    -sigma_file        survey/sigmas.txt \
    -input_filename    outputs/input.h5 \
    -params_filename   "outputs/_pp_p${ORDER}.txt" \
    "${BACKGROUND[@]}"

echo
echo "Input bundle ready: ${CASE_DIR}/outputs/input.h5"
