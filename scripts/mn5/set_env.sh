# ============================================================================
# PETGEM runtime environment on MareNostrum 5
#
# This file loads the modules and PETSC_DIR written by compile_petsc.sh.
# It must be sourced, not executed. The job scripts in examples/ source it
# unless PETGEM_ENV points to another file.
#
# Usage:
#   source set_env.sh                                  # petsc-current
#   PETSC_VERSION=3.26.0 source set_env.sh             # a specific version
#   PETSC_PREFIX=/path/to/install source set_env.sh    # a specific install
# ============================================================================

# Locate the petgem_env.sh to load.
_root="${PETSC_BASE:-/gpfs/scratch/bsc115/$USER}"
if [ -n "${PETSC_PREFIX:-}" ]; then
  _env="$PETSC_PREFIX/petgem_env.sh"
elif [ -n "${PETSC_VERSION:-}" ]; then
  _env="$_root/petsc-v$PETSC_VERSION/install/petgem_env.sh"
else
  _env="$_root/petsc-current/petgem_env.sh"
fi

if [ -r "$_env" ]; then
  . "$_env"
  echo "PETSC_DIR=$PETSC_DIR"
else
  echo "set_env.sh: $_env not found; run 'bash compile_petsc.sh env'" >&2
  unset _root _env
  return 1 2>/dev/null || exit 1
fi
unset _root _env

# Uncomment to trace runs with Extrae.
#export EXTRAE_HOME=$HOME/PETGEM/extrae/build/install
#export LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:$LD_LIBRARY_PATH
