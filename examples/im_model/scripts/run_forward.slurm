#!/bin/bash
# ===========================================================================
# Forward stage: fm.csem. One array task per frequency (the forward kernel is
# monochromatic). Uses the iterative PCBDDC solver (configs/params_fm_f<F>.txt).
#
#   sbatch examples/im_model/scripts/run_forward.slurm
#
# Consumes examples/im_model/outputs/input_fm_f<F>.h5  (build_bundles.sh fm)
# Produces examples/im_model/outputs/responses_fm_f<F>_p2.h5
#
# Site-specific: set --account and --qos/--partition for your cluster, and load
# the PETGEM runtime environment (module load / set_environment.sh) as needed.
# ===========================================================================
#SBATCH --job-name=im_model_fm
#SBATCH --output=examples/im_model/outputs/fm_%A_%a.out
#SBATCH --error=examples/im_model/outputs/fm_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-6
set -euo pipefail
C=examples/im_model
FREQS=(1 10 50 100 300 800 1500)
F=${FREQS[$SLURM_ARRAY_TASK_ID]}
echo "=== ${F} Hz : fm.csem (PCBDDC, ${SLURM_NTASKS} ranks) ==="
srun --cpu-bind=core build/fm.csem -options_file "$C/configs/params_fm_f${F}.txt"
