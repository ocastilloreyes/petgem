#!/usr/bin/env bash
# Build the MT half-space bundle: mesh.geo -> mesh.msh -> input.h5 (gmsh + preprocess -mode mt).
# Usage: bash tests/mt/make_bundle.sh [OUT]   (default tests/mt/input.h5)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${1:-$HERE/input.h5}"

command -v gmsh    >/dev/null || { echo "ERROR: gmsh not found."; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 not found."; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
cp "$HERE"/mesh.geo "$HERE"/receivers.txt "$HERE"/sigmas.txt "$HERE"/mt_frequency.txt "$WORK/"
gmsh -3 -o "$WORK/mesh.msh" "$WORK/mesh.geo" >/dev/null

# In-tree utils package as `petgem`
python3 - "$ROOT" "$WORK" <<'PY' >/dev/null
import runpy, sys
root, work = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
import utils
sys.modules["petgem"] = utils
sys.argv = ["preprocess.py", "-mode", "mt", "-order", "1", "-case_dir", work,
            "-mesh_filename", "mesh.msh", "-receiver_filename", "receivers.txt",
            "-mt_frequency_filename", "mt_frequency.txt", "-sigma_file", "sigmas.txt",
            "-params_filename", "params.txt"]
runpy.run_path(f"{root}/utils/preprocess.py", run_name="__main__")
PY

mv "$WORK/input.h5" "$OUT"
echo "wrote $OUT"
