#!/usr/bin/env bash

# ============================================================================
# Time-domain CSEM half-space model: input bundle
#
# This script meshes geometry/mesh.geo with gmsh and then calls the general
# preprocessor utils/preprocess.py with the mesh, survey, waveform and
# conductivity files of this model. The result is the HDF5 bundle read by
# fm.tdcsem.
#
# Usage (from the repository root):
#   bash examples/fm_tdcsem_halfspace/scripts/build_bundles.sh [order]
#
#   With the petgem-env image:
#   docker run --rm -u $(id -u):$(id -g) -v "$PWD":/workspace -w /workspace \
#       petgem-env:latest bash examples/fm_tdcsem_halfspace/scripts/build_bundles.sh [order]
#
# Required input:
#   geometry/mesh.geo
#   survey/sources.txt, survey/waveform.txt, survey/receivers.txt, survey/sigmas.txt
#
# Generated output:
#   outputs/mesh.msh
#   outputs/input_p<order>.h5
#   outputs/_pp_p<order>.txt (temporary preprocessor parameters)
#
# Next step:
#   sbatch scripts/run_fm.slurm   (from examples/fm_tdcsem_halfspace)
#
# Notes:
#   - Requires gmsh, numpy, meshio, h5py and petsc4py (all in petgem-env).
#   - The polynomial order defaults to 2, the order run by scripts/run_fm.slurm.
#   - The time window and the solver options live in configs/params.txt.
# ============================================================================

set -euo pipefail

CASE_DIR=examples/fm_tdcsem_halfspace
ORDER="${1:-2}"

mkdir -p "${CASE_DIR}/outputs"

echo "============================================================"
echo "Building the time-domain CSEM half-space input bundle"
echo "============================================================"
echo "Case directory : ${CASE_DIR}"
echo "Order          : ${ORDER}"
echo

# Generate the mesh.
gmsh -3 "${CASE_DIR}/geometry/mesh.geo" -o "${CASE_DIR}/outputs/mesh.msh"

# Assemble mesh, survey, waveform and conductivities into the HDF5 bundle.
python3 utils/preprocess.py -mode fm.tdcsem -order "${ORDER}" -case_dir "${CASE_DIR}" \
    -mesh_filename            outputs/mesh.msh \
    -source_filename          survey/sources.txt \
    -tdcsem_waveform_filename survey/waveform.txt \
    -tdcsem_t_end             1.0 \
    -receiver_filename        survey/receivers.txt \
    -sigma_file               survey/sigmas.txt \
    -input_filename           "outputs/input_p${ORDER}.h5" \
    -params_filename          "outputs/_pp_p${ORDER}.txt"

echo
echo "Input bundle ready: ${CASE_DIR}/outputs/input_p${ORDER}.h5"
