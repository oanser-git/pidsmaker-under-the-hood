#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${ORTHRUS_IRIS_REMOTE:-iris-cluster}"
OBJECT_SOURCE="${ORTHRUS_NO_LEAKAGE_OBJECT_SOURCE:-/scratch/users/oanser/pidsmaker_objects/orthrus/no_leakage/THEIA_E3/orthrus-no-leakage-THEIA_E3-t5-5524660.tar.zst}"
OBJECT_DIR="${ORTHRUS_OBJECT_DIR:-${REPO_ROOT}/artifacts/orthrus/objects/no_leakage/THEIA_E3}"

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

printf 'Verified object: %s\n' "${archive}"
printf 'Run scripts/extract_orthrus_no_leakage_compact.sh to extract it.\n'
