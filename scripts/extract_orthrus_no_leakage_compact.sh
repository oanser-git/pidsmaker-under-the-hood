#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OBJECT_DIR="${ORTHRUS_OBJECT_DIR:-${REPO_ROOT}/artifacts/orthrus/objects/no_leakage/THEIA_E3}"
DEFAULT_ARCHIVE="${OBJECT_DIR}/orthrus-no-leakage-THEIA_E3-t5-5524660.tar.zst"
ARCHIVE="${1:-${ORTHRUS_NO_LEAKAGE_ARCHIVE:-${DEFAULT_ARCHIVE}}}"
DESTINATION="${2:-${REPO_ROOT}/artifacts/orthrus/extracted/no_leakage/THEIA_E3}"

test -f "${ARCHIVE}" || {
  printf 'ERROR: archive not found: %s\n' "${ARCHIVE}" >&2
  exit 2
}
test -f "${ARCHIVE}.sha256" || {
  printf 'ERROR: checksum not found: %s.sha256\n' "${ARCHIVE}" >&2
  exit 2
}
(cd "$(dirname "${ARCHIVE}")" && sha256sum -c "$(basename "${ARCHIVE}.sha256")")

if test -d "${DESTINATION}" \
  && test -n "$(find "${DESTINATION}" -mindepth 1 -print -quit)"; then
  printf 'ERROR: destination is not empty: %s\n' "${DESTINATION}" >&2
  exit 2
fi
mkdir -p "${DESTINATION}"

tar -I zstd -xf "${ARCHIVE}" -C "${DESTINATION}" \
  --exclude='artifacts/feat_inference'

printf 'Compact Orthrus artifacts extracted to %s\n' "${DESTINATION}/artifacts"
printf 'Excluded the 53.07 GB feat_inference tree.\n'
