#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# make_bundle.sh - build the MMS input bundle for the CI regression test.
#
# Self-contained: uses ONLY this directory's sources (mesh.geo + the sigma /
# source / receiver stubs) and the core preprocessor utils/preprocess.py.
# It does NOT read paper/ - the CI test must run without any paper/ dependence.
#
# Needs gmsh + the PETGEM python layer (present in petgem-env / the CI image).
# A single bundle serves every order: fm.csem picks the basis order at run time
# with -order p, so this is one bundle per MESH, not per (order, mesh).
#
# Usage:  bash tests/mms/make_bundle.sh [N] [OUT]
#         bash tests/mms/make_bundle.sh 4                 # -> tests/mms/input.h5
#         bash tests/mms/make_bundle.sh 8 /tmp/n8.h5
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
N="${1:-4}"
OUT="${2:-$HERE/input.h5}"

command -v gmsh    >/dev/null || { echo "ERROR: gmsh not found."; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 not found."; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
cp "$HERE"/sources.txt "$HERE"/receivers.txt "$HERE"/sigmas.txt "$HERE"/mesh.geo "$WORK/"
gmsh -3 -setnumber N "$N" -o "$WORK/mesh.msh" "$WORK/mesh.geo" >/dev/null
python3 "$ROOT/utils/preprocess.py" -mode forward -order 1 \
    -case_dir "$WORK" -mesh_filename mesh.msh \
    -source_filename sources.txt -receiver_filename receivers.txt \
    -sigma_file sigmas.txt -params_filename params.txt >/dev/null
mv "$WORK/input.h5" "$OUT"
echo "wrote $OUT   (N=$N, 6*${N}^3 tetrahedra)"
