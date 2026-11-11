#!/usr/bin/env bash

# ============================================================================
# MT half-space test: input bundle
#
# This script meshes mesh.geo with gmsh and calls utils/preprocess.py
# -mode fm.mt.
#
# Usage (from the repository root):
#   bash tests/mt/build_bundle.sh [OUT]
#
#   OUT  output bundle (default tests/mt/input.h5)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${1:-$SCRIPT_DIR/input.h5}"

command -v gmsh    >/dev/null || { echo "ERROR: gmsh not found."; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 not found."; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
cp "$SCRIPT_DIR"/mesh.geo "$SCRIPT_DIR"/receivers.txt "$SCRIPT_DIR"/sigmas.txt "$SCRIPT_DIR"/mt_frequency.txt "$WORK/"
gmsh -3 -o "$WORK/mesh.msh" "$WORK/mesh.geo" >/dev/null
python3 "$ROOT_DIR/utils/preprocess.py" -mode fm.mt -order 1 \
    -case_dir "$WORK" -mesh_filename mesh.msh \
    -receiver_filename receivers.txt -mt_frequency_filename mt_frequency.txt \
    -sigma_file sigmas.txt -params_filename params.txt >/dev/null
mv "$WORK/input.h5" "$OUT"
echo "wrote $OUT"
