#!/bin/bash
# ===========================================================================
# Inverse stage: im.csem. L-BFGS with an adjoint-state gradient.
#
#   sbatch examples/im_model/scripts/run_inversion.slurm
#
# Consumes examples/im_model/outputs/input_im.h5  (build_bundles.sh im)
# Produces examples/im_model/outputs/responses_im_p2.h5 (recovered conductivity,
#          log_perturbation, rms_history) + VTU model snapshots.
#
# Site-specific: set --account and --qos/--partition for your cluster, and load
# the PETGEM runtime environment (module load / set_environment.sh) as needed.
# ===========================================================================
#SBATCH --job-name=im_model_im
#SBATCH --output=examples/im_model/outputs/im_%j.out
#SBATCH --error=examples/im_model/outputs/im_%j.err
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
set -euo pipefail
C=examples/im_model
echo "=== im.csem ==="
srun --cpu-bind=core build/im.csem -options_file "$C/configs/params_im.txt"
echo "=== inversion complete -> $C/outputs/responses_im_p2.h5 ==="
