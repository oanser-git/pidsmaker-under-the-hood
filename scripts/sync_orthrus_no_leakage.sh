#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${ORTHRUS_IRIS_REMOTE:-iris-cluster}"
SOURCE="${ORTHRUS_NO_LEAKAGE_SOURCE:-/scratch/users/oanser/orthrus_t5/no_leakage/orthrus_no_leakage/THEIA_E3/artifacts}"
DESTINATION="${1:-${ORTHRUS_NO_LEAKAGE_ARTIFACTS:-${REPO_ROOT}/artifacts/orthrus/no_leakage/THEIA_E3/t5-5524660}}"

command -v rsync >/dev/null 2>&1 || {
  printf 'ERROR: rsync is required.\n' >&2
  exit 2
}

mkdir -p "${DESTINATION}"
rsync --archive --checksum --partial --info=progress2 \
  "${REMOTE}:${SOURCE%/}/" "${DESTINATION}/"

printf 'Artifacts synchronized to %s\n' "${DESTINATION}"
printf 'Run: python scripts/verify_orthrus_no_leakage.py %q\n' "${DESTINATION}"
