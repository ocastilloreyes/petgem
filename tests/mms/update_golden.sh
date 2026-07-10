#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# update_golden.sh - regenerate tests/mms/reference/mms_golden.json.
#
# Self-contained (no paper/ dependence): builds the N=4 bundle from this
# directory's sources, runs fm.csem -mms for the CI orders with a direct LU
# solve, and writes the golden relative errors. Run this ONLY after an
# INTENTIONAL discretisation change (basis, DOF layout, assembly, quadrature,
# error norm); the new goldens must then be code-reviewed.
#
#     docker run --rm -v "$PWD":/workspace -w /workspace petgem-env \
#         bash tests/mms/update_golden.sh
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ORDERS="${ORDERS:-1 3 6}"
N="${N:-4}"
FMCSEM="${PETGEM_FM_CSEM:-$ROOT/build/fm.csem}"
SOLVER="-dm_mat_type aij -ksp_type preonly -pc_type lu \
        -pc_factor_mat_solver_type mumps -ksp_error_if_not_converged"

[ -x "$FMCSEM" ] || { echo "ERROR: fm.csem not found at $FMCSEM (build it, or set PETGEM_FM_CSEM)."; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
bash "$HERE/make_bundle.sh" "$N" "$WORK/input.h5"
for p in $ORDERS; do
  "$FMCSEM" -mms -order "$p" -input_filename "$WORK/input.h5" \
      -output_dir "$WORK" -output_filename "golden_o${p}" $SOLVER >/dev/null
done

python3 - "$WORK" "$HERE/reference/mms_golden.json" "$N" "$ORDERS" <<'PY'
import h5py, json, sys
work, out, N, orders = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split()
d = {}
for p in orders:
    with h5py.File(f"{work}/golden_o{p}.h5", "r") as f:
        a = f.attrs
        d[str(p)] = {"dofs": int(a["dofs"]),
                     "solve_L2": float(a["solve_L2"]), "solve_Hcurl": float(a["solve_Hcurl"]),
                     "proj_L2": float(a["proj_L2"]),   "proj_Hcurl": float(a["proj_Hcurl"]),
                     "residual": float(a["residual"])}
doc = {"_comment": "Golden MMS relative errors for fm.csem CI (tests/mms/test_level6_mms.py). "
                   "Regenerate with tests/mms/update_golden.sh only after an intentional "
                   "discretisation change.",
       "mesh": f"unit cube [0,1]^3 at N={N} (6*{N}^3 tetrahedra)",
       "frequency_hz": 2.0, "sigma_S_per_m": 1.0,
       "solver": "MUMPS direct LU (ksp preonly, pc lu)",
       "rtol": 1e-3, "atol": 1e-8, "residual_max": 1e-10, "orders": d}
open(out, "w").write(json.dumps(doc, indent=2) + "\n")
print("wrote", out)
PY
