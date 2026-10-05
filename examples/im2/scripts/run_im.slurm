#!/bin/bash
# ===========================================================================
# Inverse stage: im.csem. L-BFGS with an adjoint-state gradient.
#
# The linear solver is user-selectable, exactly as in run_fm.slurm: the
# case options live in configs/params_im.txt and the solver is added as a
# preset. Both solves (forward and adjoint) share one KSP, so the preset governs
# both.
#
#   sbatch examples/im2/scripts/run_im.slurm               # PCBDDC iterative (default)
#   SOLVER=mumps  sbatch examples/im2/scripts/run_im.slurm # MUMPS direct
#
# Consumes examples/im2/outputs/input_im.h5  (build_bundles.sh im)
# Produces examples/im2/outputs/responses_im_p2.h5 (recovered conductivity,
#          log_perturbation, rms_history) + VTU model snapshots.
#
# WALLTIME. The comparable examples/im1 inversion took 01:40:06 on 112 ranks for
# 94 L-BFGS steps on a 58022-cell mesh. This case has a mesh of the same size
# but a 80-step budget (Table 2 of the paper), so the request below is 06:00:00
# with margin rather than the 2 h that fits a debug QoS. No --qos line is set
# on purpose: 2 h debug queues cannot hold this job, so the account default
# applies unless you name your own production QoS here.
#
# Site-specific: set --account and --qos/--partition for your cluster, and load
# the PETGEM runtime environment (module load / set_environment.sh) as needed.
# ===========================================================================
#SBATCH --job-name=im2_inv
#SBATCH --output=examples/im2/outputs/im_%j.out
#SBATCH --error=examples/im2/outputs/im_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=01:59:00

set -euo pipefail
C=examples/im2
SOLVER="${SOLVER:-bddc}"
echo "=== im.csem (solver: ${SOLVER}, ${SLURM_NTASKS} ranks) ==="
srun --cpu-bind=core build/im.csem \
    -options_file "$C/configs/params_im.txt" \
    -options_file "$C/configs/solver_${SOLVER}.txt"
echo "=== inversion complete -> $C/outputs/responses_im_p2.h5 ==="
