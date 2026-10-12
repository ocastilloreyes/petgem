#!/usr/bin/env bash

# ============================================================================
# MMS test: golden errors
#
# This script regenerates tests/mms/reference/mms_golden.json. It builds the
# bundle, runs build/mms with a direct LU solve for each order and writes the
# error norms. Run it only after an intentional change of the discretization
# and review the new goldens.
#
# Usage (from the repository root):
#   bash tests/mms/regen_golden.sh [N]
#
#   N  cells per cube edge (default 4)
#
#   With the petgem-env image:
#   docker run --rm -u $(id -u):$(id -g) -v "$PWD":/workspace -w /workspace \
#       petgem-env:latest bash tests/mms/regen_golden.sh
#
# Notes:
#   - ORDERS="1 3 6" (default) selects the orders.
#   - mms is taken from PETGEM_MMS, or build/mms.
# ============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ORDERS="${ORDERS:-1 3 6}"
N="${1:-4}"
MMS="${PETGEM_MMS:-$ROOT_DIR/build/mms}"
SOLVER="-dm_mat_type aij -ksp_type preonly -pc_type lu \
        -pc_factor_mat_solver_type mumps -ksp_error_if_not_converged"

[ -x "$MMS" ] || { echo "ERROR: mms not found at $MMS (run make mms, or set PETGEM_MMS)."; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
bash "$SCRIPT_DIR/build_bundle.sh" "$N" "$WORK/input.h5"
for p in $ORDERS; do
  "$MMS" -order "$p" -input_filename "$WORK/input.h5" \
      -output_dir "$WORK" -output_filename "golden_o${p}" $SOLVER >/dev/null
done

python3 - "$WORK" "$SCRIPT_DIR/reference/mms_golden.json" "$N" "$ORDERS" <<'PY'
import h5py, json, sys
work, out, N, orders = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split()
d = {}
for p in orders:
    with h5py.File(f"{work}/golden_o{p}.h5", "r") as f:
        a = f.attrs
        d[str(p)] = {"dofs": int(a["dofs"]),
                     "solve_L2": float(a["solve_L2"]), "solve_energy": float(a["solve_energy"]),
                     "proj_L2": float(a["proj_L2"]),   "proj_energy": float(a["proj_energy"]),
                     "residual": float(a["residual"])}
doc = {"_comment": "Golden MMS relative errors for the CI (tests/mms/test_level6_mms.py). "
                   "Regenerate with tests/mms/regen_golden.sh only after an intentional "
                   "discretization change.",
       "mesh": f"cube [0,1000]^3 at N={N} (6*{N}^3 tetrahedra)",
       "frequency_hz": 10.0, "sigma_S_per_m": [0.5, 1.0, 2.0],
       "solver": "MUMPS direct LU (ksp preonly, pc lu)",
       "rtol": 1e-3, "atol": 1e-8, "residual_max": 1e-10, "orders": d}
open(out, "w").write(json.dumps(doc, indent=2) + "\n")
print("wrote", out)
PY
