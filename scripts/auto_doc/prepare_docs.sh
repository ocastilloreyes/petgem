#!/usr/bin/env bash
#
# prepare_docs.sh - generate the Doxygen XML + the Sphinx API .rst pages.
#
# Single source of truth for the documentation-prep steps, shared by:
#   - the Makefile `docs` target (local builds), and
#   - the Read the Docs `pre_build` job (.readthedocs.yaml).
#
# Deliberately PETSc-free: it must run in the RTD build environment, which
# has no PETSc. (The repo Makefile `include`s PETSc's conf at the top, so
# RTD cannot call `make docs` directly - hence this standalone script.)
#
# It does NOT run the final Sphinx HTML build: locally that is the
# Makefile `sphinx_html` target; on RTD it is the native `sphinx:` step.
set -euo pipefail

# Resolve the repo root from this script's location (robust to CWD).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo ">>> [DOC] Cleaning generated doc artifacts"
rm -rf docs/doxygen/* docs/source/api/* docs/source/readme/* 2>/dev/null || true

echo ">>> [DOC] Generating Doxygen XML"
doxygen Doxyfile

echo ">>> [DOC] Generating Sphinx API .rst pages"
python3 scripts/auto_doc/api_rst_generator.py

echo ">>> [DOC] Doc prep complete (api pages: $(ls docs/source/api/*.rst | wc -l))"
