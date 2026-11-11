#!/usr/bin/env bash

# ============================================================================
# MMS test: input bundle
#
# This script meshes mesh.geo with gmsh and calls utils/preprocess.py
# -mode fm.csem. One bundle serves every order, since mms takes -order at
# run time.
#
# Usage (from the repository root):
#   bash tests/mms/build_bundle.sh [N] [OUT]
#
#   N    cells per cube edge (default 4)
#   OUT  output bundle (default tests/mms/input.h5)
# ============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
N="${1:-4}"
OUT="${2:-$SCRIPT_DIR/input.h5}"

command -v gmsh    >/dev/null || { echo "ERROR: gmsh not found."; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 not found."; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
cp "$SCRIPT_DIR"/sources.txt "$SCRIPT_DIR"/receivers.txt "$SCRIPT_DIR"/sigmas.txt "$SCRIPT_DIR"/mesh.geo "$WORK/"
gmsh -3 -setnumber N "$N" -o "$WORK/mesh.msh" "$WORK/mesh.geo" >/dev/null
python3 "$ROOT_DIR/utils/preprocess.py" -mode fm.csem -order 1 \
    -case_dir "$WORK" -mesh_filename mesh.msh \
    -source_filename sources.txt -receiver_filename receivers.txt \
    -sigma_file sigmas.txt -params_filename params.txt >/dev/null
mv "$WORK/input.h5" "$OUT"
echo "wrote $OUT   (N=$N, 6*${N}^3 tetrahedra)"
