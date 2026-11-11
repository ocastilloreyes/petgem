#!/usr/bin/env bash

# ============================================================================
# Release version check
#
# This script checks that the version in include/version.h is greater than
# every tag of the origin remote.
#
# Usage:
#   bash scripts/ci/check_release.sh
#
# Notes:
#   - Tags with and without a leading v are compared.
# ============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

version="$(bash scripts/ci/version_tag.sh)"
version="${version#v}"
latest="$(git ls-remote --tags --refs origin | sed -E 's#.*refs/tags/v?##' | sort -V | tail -n 1)"
highest="$(printf '%s\n%s\n' "${latest}" "${version}" | sort -V | tail -n 1)"

if [[ "${version}" == "${latest}" || "${highest}" != "${version}" ]]; then
  echo "ERROR: version ${version} must be greater than the latest tag ${latest} (see include/version.h)" >&2
  exit 1
fi

echo "check_release: ${version} (latest tag: ${latest:-none})"
