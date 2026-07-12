#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_COMMIT="$(tr -d '[:space:]' < "${SCRIPT_DIR}/BASE_COMMIT")"
DESTINATION="${1:-${SCRIPT_DIR}/../../../../PIDSMaker-orthrus-no-leakage}"

if ! test -d "${DESTINATION}/.git"; then
  git clone https://github.com/ubc-provenance/PIDSMaker.git "${DESTINATION}"
fi

if test -n "$(git -C "${DESTINATION}" status --porcelain --ignored)"; then
  printf 'ERROR: refusing to overwrite dirty repository: %s\n' "${DESTINATION}" >&2
  exit 2
fi

git -C "${DESTINATION}" checkout --detach "${BASE_COMMIT}"
rsync -a "${SCRIPT_DIR}/overlay/" "${DESTINATION}/"

printf 'Materialized strict no-leakage Orthrus at %s\n' "${DESTINATION}"
printf 'Base commit: %s\n' "${BASE_COMMIT}"
