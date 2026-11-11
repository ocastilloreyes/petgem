#!/usr/bin/env bash

# ============================================================================
# Documentation preparation
#
# This script generates the Doxygen XML and the Sphinx API .rst pages. It is
# the single place where these steps are defined, and both the Makefile
# `docs` target (local builds) and the Read the Docs `pre_build` job
# (.readthedocs.yaml) call it.
#
# Usage:
#   bash scripts/auto_doc/prepare_docs.sh
#
# Generated output:
#   docs/doxygen/
#   docs/source/api/*.rst
#
# Notes:
#   - It does not need PETSc. Read the Docs has no PETSc, and the Makefile
#     includes PETSc's configuration at the top, so RTD cannot run
#     `make docs` directly.
#   - It does not build the final HTML. Locally that is the Makefile
#     `sphinx_html` target; on RTD it is the native `sphinx:` step.
#   - It can be run from any directory.
# ============================================================================

set -euo pipefail

# Work from the repository root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo ">>> [DOC] Cleaning generated doc artifacts"
rm -rf docs/doxygen/* docs/source/api/* docs/source/readme/* 2>/dev/null || true

echo ">>> [DOC] Generating Doxygen XML"
doxygen Doxyfile

echo ">>> [DOC] Generating Sphinx API .rst pages"
python3 scripts/auto_doc/api_rst_generator.py

echo ">>> [DOC] Doc prep complete (api pages: $(ls docs/source/api/*.rst | wc -l))"
