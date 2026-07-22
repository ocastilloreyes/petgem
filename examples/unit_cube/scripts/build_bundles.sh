#!/usr/bin/env bash
# ===========================================================================
# Build the fm.csem input bundle for the unit_cube FM-CSEM test fixture: gmsh
# meshes geometry/mesh.geo, then utils/preprocess.py assembles the mesh + survey
# into outputs/input.h5. This is exactly what CI does before the e2e / Extrae
# jobs (see .github/workflows/tests-fm-csem.yml and tests/README.md).
#
# Requires gmsh + the petgem Python package (meshio). Run from the REPOSITORY
# ROOT. With the petgem-env image, mount the repo at /workspace:
#
#   docker run --rm -v "$PWD":/workspace -w /workspace petgem-env:latest \
#       bash examples/unit_cube/scripts/build_bundles.sh [order]
#
# The polynomial order is chosen at run time with `-order N`, so one bundle
# serves every order; the argument only labels the throwaway preprocess params.
# ===========================================================================
set -euo pipefail
C=examples/unit_cube
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

echo "=== unit_cube bundle ready: $C/outputs/input.h5 ==="
