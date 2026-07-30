#!/bin/bash
# ===========================================================================
# Inverse stage: im.csem. L-BFGS with an adjoint-state gradient.
#
# The linear solver is user-selectable, exactly as in run_forward.slurm: the
# case options live in configs/params_im.txt and the solver is added as a
# preset. Both solves (forward and adjoint) share one KSP, so the preset governs
# both.
#
#   sbatch examples/im/scripts/run_inversion.slurm               # PCBDDC iterative (default)
#   SOLVER=mumps  sbatch examples/im/scripts/run_inversion.slurm # MUMPS direct
#
# Consumes examples/im/outputs/input_im.h5  (build_bundles.sh im)
# Produces examples/im/outputs/responses_im_p2.h5 (recovered conductivity,
#          log_perturbation, rms_history) + VTU model snapshots.
#
#
# Site-specific: set --account and --qos/--partition for your cluster, and load
# the PETGEM runtime environment (module load / set_environment.sh) as needed.
# ===========================================================================
#SBATCH --job-name=im_inv
#SBATCH --output=examples/im/outputs/im_%j.out
#SBATCH --error=examples/im/outputs/im_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=01:59:00

set -euo pipefail
C=examples/im
SOLVER="${SOLVER:-bddc}"
echo "=== im.csem (solver: ${SOLVER}, ${SLURM_NTASKS} ranks) ==="
srun --cpu-bind=core build/im.csem \
    -options_file "$C/configs/params_im.txt" \
    -options_file "$C/configs/solver_${SOLVER}.txt"
echo "=== inversion complete -> $C/outputs/responses_im_p2.h5 ==="
