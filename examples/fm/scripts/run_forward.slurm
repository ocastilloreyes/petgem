#!/bin/bash
# ===========================================================================
# Forward stage: fm.csem on the marine CSEM benchmark. Uses the iterative
# PCBDDC solver (configs/params.txt).
#
#   sbatch examples/fm/scripts/run_forward.slurm
#
# Consumes examples/fm/outputs/input.h5   (scripts/build_bundles.sh)
# Produces examples/fm/outputs/responses_p1.h5
#
# Validate the result afterwards with:
#   python3 examples/fm/scripts/postprocess.py
#
# Site-specific: set --account and --qos/--partition for your cluster, and load
# the PETGEM runtime environment (module load / set_environment.sh) as needed.
# ===========================================================================
#SBATCH --job-name=fm_fwd
#SBATCH --output=examples/fm/outputs/fm_%j.out
#SBATCH --error=examples/fm/outputs/fm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

set -euo pipefail
C=examples/fm
echo "=== fm.csem (PCBDDC, ${SLURM_NTASKS} ranks) ==="
srun --cpu-bind=core build/fm.csem -options_file "$C/configs/params.txt" -order 1
