#!/usr/bin/env bash

# ============================================================================
# Release tag
#
# This script prints the release tag of the version in include/version.h
# (vMAJOR.MINOR.PATCH).
#
# Usage:
#   bash scripts/ci/version_tag.sh
# ============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

part() {
  local value
  value="$(awk -v key="VERSION_$1" '$1 == "#define" && $2 == key { print $3 }' include/version.h)"
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid VERSION_$1 in include/version.h" >&2; exit 1; }
  echo "${value}"
}

major="$(part MAJOR)"
minor="$(part MINOR)"
patch="$(part PATCH)"
echo "v${major}.${minor}.${patch}"
