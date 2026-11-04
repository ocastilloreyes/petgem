#!/usr/bin/env bash

# Pull the CI image, with up to 3 attempts 30 s apart.
#
# Usage:
#   bash scripts/ci/pull_image.sh <image>

set -uo pipefail

image="${1:?usage: pull_image.sh <image>}"

for attempt in 1 2 3; do
  docker pull "${image}" && exit 0
  echo "docker pull failed (attempt ${attempt}/3)" >&2
  [[ "${attempt}" -lt 3 ]] && sleep 30
done
exit 1
