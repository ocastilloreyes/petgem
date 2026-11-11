#!/usr/bin/env bash

# ============================================================================
# Static checks
#
# This script checks the format of the C sources with clang-format and the
# syntax of the Python files.
#
# Usage:
#   bash scripts/ci/check_static.sh
# ============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

find src include tests/csrc tests/mms -name '*.[ch]' -print0 \
  | xargs -0 clang-format --dry-run --Werror

find utils tests examples scripts docs/source/conf.py -name '*.py' -print0 \
  | xargs -0 -n 1 python3 -c 'import ast, sys; ast.parse(open(sys.argv[1], encoding="utf-8").read(), sys.argv[1])'

echo "check_static: PASS"
