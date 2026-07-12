#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${ORTHRUS_IRIS_REMOTE:-iris-cluster}"
OBJECT_SOURCE="${ORTHRUS_NO_LEAKAGE_OBJECT_SOURCE:-/scratch/users/oanser/pidsmaker_objects/orthrus/no_leakage/THEIA_E3/orthrus-no-leakage-THEIA_E3-t5-5524660.tar.zst}"
OBJECT_DIR="${ORTHRUS_OBJECT_DIR:-${REPO_ROOT}/artifacts/objects}"
EXTRACT_ROOT="${1:-${ORTHRUS_NO_LEAKAGE_ARTIFACTS:-${REPO_ROOT}/artifacts/orthrus/no_leakage/THEIA_E3/t5-5524660}}"

command -v rsync >/dev/null 2>&1 || {
  printf 'ERROR: rsync is required.\n' >&2
  exit 2
}

mkdir -p "${OBJECT_DIR}"
archive="${OBJECT_DIR}/$(basename "${OBJECT_SOURCE}")"
checksum="${archive}.sha256"
rsync --archive --partial --info=progress2 \
  "${REMOTE}:${OBJECT_SOURCE}" "${archive}"
rsync --archive \
  "${REMOTE}:${OBJECT_SOURCE}.sha256" "${checksum}"
(cd "${OBJECT_DIR}" && sha256sum -c "$(basename "${checksum}")")

if test -d "${EXTRACT_ROOT}" \
  && test -n "$(find "${EXTRACT_ROOT}" -mindepth 1 -print -quit)"; then
  printf 'ERROR: extraction destination is not empty: %s\n' "${EXTRACT_ROOT}" >&2
  exit 2
fi
mkdir -p "${EXTRACT_ROOT}"
tar -I zstd -xf "${archive}" -C "${EXTRACT_ROOT}"

printf 'Verified object: %s\n' "${archive}"
printf 'Artifacts extracted to %s\n' "${EXTRACT_ROOT}/artifacts"
printf 'Run: python scripts/verify_orthrus_no_leakage.py %q\n' \
  "${EXTRACT_ROOT}/artifacts"
