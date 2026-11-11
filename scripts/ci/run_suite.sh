#!/usr/bin/env bash

# ============================================================================
# Test suite
#
# This script builds the input of one test suite and runs it, as the CI does.
#
# Usage:
#   bash scripts/ci/run_suite.sh <unit|e2e|mms|mt|im|tdcsem>
#
# Required input:
#   build/ binaries (make all mms)
#
# Generated output:
#   build/reports/<suite>.xml
#
# Notes:
#   - A skipped test is reported as failed.
# ============================================================================

set -euo pipefail

suite="${1:?usage: run_suite.sh <unit|e2e|mms|mt|im|tdcsem>}"

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CI="${CI:-true}"

case "${suite}" in
  unit)   ;;
  e2e)    bash examples/fm_csem_unit_cube/scripts/build_bundles.sh; export PETGEM_NP="${PETGEM_NP:-2}" ;;
  mms)    bash tests/mms/build_bundle.sh 4; export PETGEM_NP="${PETGEM_NP:-2}" ;;
  mt)     bash tests/mt/build_bundle.sh ;;
  im)     bash tests/im/build_bundle.sh ;;
  tdcsem) bash tests/tdcsem/build_bundle.sh ;;
  *)      echo "ERROR: unknown suite ${suite}" >&2; exit 2 ;;
esac

pytest "tests/${suite}" --junitxml="build/reports/${suite}.xml"
