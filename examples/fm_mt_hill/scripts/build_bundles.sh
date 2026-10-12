#!/usr/bin/env bash

# ============================================================================
# Trapezoidal hill MT model: input bundle
#
# This script meshes geometry/mesh.geo with gmsh, or takes an existing Gmsh
# mesh, and then calls utils/preprocess.py with the survey and conductivity
# files of this model. The result is the HDF5 bundle read by fm.mt.
#
# Usage (from the repository root):
#   bash examples/fm_mt_hill/scripts/build_bundles.sh <order> <nskin> [hmin | mesh.msh]
#
#   order  Nedelec order written into the bundle (1..6)
#   nskin  skin depths between the survey and the boundaries (1, 2, 4, 6, 8, 10)
#   hmin   element size on the hill and the survey line, in m
#          (default: 35 for order 1, 42 for order 2+)
#   mesh   an existing Gmsh mesh (physical 1 = air, 2 = earth) used instead of
#          geometry/mesh.geo, e.g. the meshes of Castillo-Reyes et al. (2022)
#
#   With the petgem-env image:
#   docker run --rm -u $(id -u):$(id -g) -v "$PWD":/workspace -w /workspace \
#       petgem-env:latest bash examples/fm_mt_hill/scripts/build_bundles.sh 2 4
#
# Required input:
#   geometry/mesh.geo (or the mesh given as third argument)
#   survey/mt_frequency.txt, survey/receivers.txt, survey/sigmas.txt
#
# Generated output:
#   outputs/input_p<order>_n<nskin>.h5
#   outputs/_pp_p<order>_n<nskin>.txt (temporary preprocessor parameters)
#
# Next step:
#   sbatch --export=ALL,ORDER=<order>,NSKIN=<nskin> scripts/run_fm.slurm   (from examples/fm_mt_hill)
#
# Notes:
#   - Requires gmsh, numpy, meshio, h5py and petsc4py (all in petgem-env).
# ============================================================================

set -euo pipefail

CASE_DIR=examples/fm_mt_hill
ORDER="${1:?usage: build_bundles.sh <order> <nskin> [hmin | mesh.msh]}"
NSKIN="${2:?usage: build_bundles.sh <order> <nskin> [hmin | mesh.msh]}"
THIRD="${3:-}"
TAG="p${ORDER}_n${NSKIN}"

mkdir -p "${CASE_DIR}/outputs"

echo "============================================================"
echo "Building the trapezoidal hill MT input bundle"
echo "============================================================"
echo "Case directory : ${CASE_DIR}"
echo "Order          : ${ORDER}"
echo "Skin depths    : ${NSKIN}"

if [[ -n "${THIRD}" && -f "${THIRD}" ]]; then
    echo "Mesh           : ${THIRD}"
    cp "${THIRD}" "${CASE_DIR}/outputs/mesh_${TAG}.msh"
else
    HMIN="${THIRD:-$([[ "${ORDER}" == 1 ]] && echo 35 || echo 42)}"
    echo "Mesh           : geometry/mesh.geo (hmin = ${HMIN} m)"
    gmsh -3 -setnumber nskin "${NSKIN}" -setnumber hmin "${HMIN}" \
        "${CASE_DIR}/geometry/mesh.geo" -o "${CASE_DIR}/outputs/mesh_${TAG}.msh"
fi
echo

python3 utils/preprocess.py -mode fm.mt -order "${ORDER}" -case_dir "${CASE_DIR}" \
    -mesh_filename          "outputs/mesh_${TAG}.msh" \
    -mt_frequency_filename  survey/mt_frequency.txt \
    -receiver_filename      survey/receivers.txt \
    -sigma_file             survey/sigmas.txt \
    -input_filename         "outputs/input_${TAG}.h5" \
    -params_filename        "outputs/_pp_${TAG}.txt"

echo
echo "Input bundle ready: ${CASE_DIR}/outputs/input_${TAG}.h5"
