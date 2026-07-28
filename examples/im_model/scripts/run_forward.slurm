#!/bin/bash
# ===========================================================================
# Forward stage: fm.csem. One array task per frequency (the forward kernel is
# monochromatic). The linear solver is user-selectable: the case options live in
# configs/params_fm_f<F>.txt and the solver is added as a preset. Choose it with
# the SOLVER variable:
#
#   sbatch examples/im_model/scripts/run_forward.slurm              # MUMPS direct (default)
#   SOLVER=bddc  sbatch examples/im_model/scripts/run_forward.slurm # PCBDDC iterative
#
# The forward stage generates the reference/"truth" data, so it defaults to the
# MUMPS DIRECT solver: exact and robust. The PCBDDC preset (SOLVER=bddc) is the
# memory-scalable iterative path and runs at order 2 with the full Nedelec
# coarse space: the "SIZE OF EDGE > EXTCOL SECOND PASS" abort came from BDDC
# reading the Plex field IS block size as a field count, and is fixed by
# prescribing a single field (src/solver.c, plus -pc_bddc_monolithic in
# configs/solver_bddc.txt). Verified at orders 1 and 2 over several rank counts.
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
SOLVER="${SOLVER:-mumps}"
FREQS=(1 10 50 100 300 800 1500)
F=${FREQS[$SLURM_ARRAY_TASK_ID]}
echo "=== ${F} Hz : fm.csem (solver: ${SOLVER}, ${SLURM_NTASKS} ranks) ==="
srun --cpu-bind=core build/fm.csem \
    -options_file "$C/configs/params_fm_f${F}.txt" \
    -options_file "$C/configs/solver_${SOLVER}.txt"
