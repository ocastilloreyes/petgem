#!/bin/bash
# ===========================================================================
# Pipeline stage 2: fm.csem on the TRUE model. One array task per frequency
# (the forward kernel is monochromatic). The linear solver is user-selectable
# through a preset; choose it with the SOLVER variable:
#
#   sbatch examples/im/scripts/run_forward.slurm               # PCBDDC iterative (default)
#   SOLVER=mumps  sbatch examples/im/scripts/run_forward.slurm # MUMPS direct
#
# Defaults to the PCBDDC ITERATIVE solver: memory-scalable.
#
# SOLVER=mumps keeps the direct solver available as an exact cross-check, which
# is worth running once per frequency when regenerating reference data.
#
# Consumes examples/im/outputs/input_fm_f<F>.h5  (build_bundles.sh fm)
# Produces examples/im/outputs/responses_fm_f<F>_p2.h5
#
# Site-specific: set --account and --qos/--partition for your cluster, and load
# the PETGEM runtime environment (module load / set_environment.sh) as needed.
# ===========================================================================
#SBATCH --job-name=im_fwd
#SBATCH --output=examples/im/outputs/fm_%A_%a.out
#SBATCH --error=examples/im/outputs/fm_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=01:59:00
#SBATCH --array=0-6

set -euo pipefail
C=examples/im
SOLVER="${SOLVER:-bddc}"
FREQS=(1 10 50 100 300 800 1500)
F=${FREQS[$SLURM_ARRAY_TASK_ID]}
echo "=== ${F} Hz : fm.csem (solver: ${SOLVER}, ${SLURM_NTASKS} ranks) ==="
# The per-frequency case options are three I/O paths that differ only in ${F},
# so they are set here rather than in seven near-identical config files. The
# solver still comes from a preset, exactly as in run_inversion.slurm.
srun --cpu-bind=core build/fm.csem \
    -input_filename   "$C/outputs/input_fm_f${F}.h5" \
    -output_dir       "$C/outputs/" \
    -output_filename  "responses_fm_f${F}_p2" \
    -options_file     "$C/configs/solver_${SOLVER}.txt"
