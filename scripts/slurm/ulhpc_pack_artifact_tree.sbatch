#!/bin/bash -l

#SBATCH --job-name=pids-pack-artifacts
#SBATCH --clusters=iris
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${SOURCE_DIR:?Set SOURCE_DIR to the artifact tree to seal}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the allowed parent of SOURCE_DIR}"
: "${OBJECT_ID:?Set OBJECT_ID to the stable bundle identifier}"
: "${COMPLETION_MARKER:?Set COMPLETION_MARKER written after the producer closes the tree}"

OBJECT_DIR="${OBJECT_DIR:-/scratch/users/oanser/pidsmaker_objects}"
DELETE_SOURCE_AFTER_VERIFY="${DELETE_SOURCE_AFTER_VERIFY:-0}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-3}"

SOURCE_ROOT="$(realpath -e "${SOURCE_ROOT}")"
SOURCE_DIR="$(realpath -e "${SOURCE_DIR}")"
OBJECT_DIR="$(realpath -m "${OBJECT_DIR}")"
COMPLETION_MARKER="$(realpath -e "${COMPLETION_MARKER}")"
test -f "${COMPLETION_MARKER}" || {
  printf 'ERROR: completion marker is not a file: %s\n' "${COMPLETION_MARKER}" >&2
  exit 2
}
case "${COMPLETION_MARKER}" in
  "${SOURCE_DIR}"/*) ;;
  *)
    printf 'ERROR: COMPLETION_MARKER must be inside SOURCE_DIR.\n' >&2
    exit 2
    ;;
esac
completion_marker_relative="${COMPLETION_MARKER#${SOURCE_DIR}/}"

case "${SOURCE_DIR}" in
  "${SOURCE_ROOT}"/*) ;;
  *)
    printf 'ERROR: SOURCE_DIR must be below SOURCE_ROOT: %s\n' "${SOURCE_DIR}" >&2
    exit 2
    ;;
esac
case "${OBJECT_DIR}" in
  "${SOURCE_DIR}"|"${SOURCE_DIR}"/*)
    printf 'ERROR: OBJECT_DIR must be outside SOURCE_DIR: %s\n' "${OBJECT_DIR}" >&2
    exit 2
    ;;
esac
case "${OBJECT_ID}" in
  *[!A-Za-z0-9._-]*|'')
    printf 'ERROR: invalid OBJECT_ID: %s\n' "${OBJECT_ID}" >&2
    exit 2
    ;;
esac

mkdir -p "${OBJECT_DIR}"
exec 9>"${OBJECT_DIR}/.${OBJECT_ID}.lock"
flock -n 9 || {
  printf 'ERROR: another job is packing %s.\n' "${OBJECT_ID}" >&2
  exit 2
}

archive="${OBJECT_DIR}/${OBJECT_ID}.tar.zst"
temporary_archive="${archive}.tmp-${SLURM_JOB_ID}"
checksum="${archive}.sha256"
temporary_checksum=""
metadata_dir=""
sealed_source_active=0
original_source_dir="${SOURCE_DIR}"
source_name="$(basename "${SOURCE_DIR}")"
source_parent="$(dirname "${SOURCE_DIR}")"
sealed_source_dir="${source_parent}/.${source_name}.sealing-${SLURM_JOB_ID}"

cleanup() {
  if test -n "${metadata_dir}"; then
    rm -rf -- "${metadata_dir}"
  fi
  rm -f -- "${temporary_archive}"
  if test -n "${temporary_checksum}"; then
    rm -f -- "${temporary_checksum}"
  fi
  if test "${sealed_source_active}" = "1"; then
    if ! test -e "${original_source_dir}"; then
      mv "${sealed_source_dir}" "${original_source_dir}"
    else
      printf 'WARNING: preserving sealed source because original path was recreated: %s\n' \
        "${sealed_source_dir}" >&2
    fi
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

exec 8>"${source_parent}/.${source_name}.pack.lock"
flock -n 8 || {
  printf 'ERROR: another job is sealing %s.\n' "${original_source_dir}" >&2
  exit 2
}
test ! -e "${sealed_source_dir}" || {
  printf 'ERROR: sealed source path already exists: %s\n' "${sealed_source_dir}" >&2
  exit 2
}
mv "${original_source_dir}" "${sealed_source_dir}"
SOURCE_DIR="${sealed_source_dir}"
COMPLETION_MARKER="${SOURCE_DIR}/${completion_marker_relative}"
sealed_source_active=1

metadata_dir="$(mktemp -d "${TMPDIR:-/tmp}/${USER}-${OBJECT_ID}-XXXXXX")"

source_files="$(find "${SOURCE_DIR}" -type f | wc -l)"
source_bytes="$(du -sb "${SOURCE_DIR}" | cut -f1)"
source_tree_digest="$(tar --sort=name --format=gnu --mtime=@0 \
  --owner=0 --group=0 --numeric-owner -cf - -C "${SOURCE_DIR}" . \
  | sha256sum | cut -d ' ' -f 1)"

cat > "${metadata_dir}/manifest.json" <<EOF
{
  "schema_version": 1,
  "object_id": "${OBJECT_ID}",
  "source_name": "${source_name}",
  "source_file_count": ${source_files},
  "source_bytes": ${source_bytes},
  "source_tree_digest": "${source_tree_digest}",
  "completion_marker": "${completion_marker_relative}",
  "completion_marker_sha256": "$(sha256sum "${COMPLETION_MARKER}" | cut -d ' ' -f 1)",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "compression": "zstd-${COMPRESSION_LEVEL}"
}
EOF

if test -f "${archive}"; then
  printf 'Archive already exists; verifying %s\n' "${archive}"
  test -f "${checksum}" || {
    printf 'ERROR: existing archive has no trusted checksum: %s\n' "${checksum}" >&2
    exit 2
  }
  (cd "${OBJECT_DIR}" && sha256sum -c "$(basename "${checksum}")")
  embedded_manifest="$(tar -I zstd -xOf "${archive}" manifest.json)"
  test "$(jq -r '.object_id' <<<"${embedded_manifest}")" = "${OBJECT_ID}"
  test "$(jq -r '.source_name' <<<"${embedded_manifest}")" = "${source_name}"
  test "$(jq -r '.source_file_count' <<<"${embedded_manifest}")" = "${source_files}"
  test "$(jq -r '.source_bytes' <<<"${embedded_manifest}")" = "${source_bytes}"
  test "$(jq -r '.source_tree_digest' <<<"${embedded_manifest}")" = "${source_tree_digest}"
  test "$(jq -r '.completion_marker' <<<"${embedded_manifest}")" = "${completion_marker_relative}"
  test "$(jq -r '.completion_marker_sha256' <<<"${embedded_manifest}")" \
    = "$(sha256sum "${COMPLETION_MARKER}" | cut -d ' ' -f 1)"
else
  tar -I "zstd -T${SLURM_CPUS_PER_TASK:-1} -${COMPRESSION_LEVEL}" \
    -cf "${temporary_archive}" \
    --transform="s|^\.$|${source_name}|" \
    --transform="s|^\./|${source_name}/|" \
    -C "${SOURCE_DIR}" . \
    -C "${metadata_dir}" manifest.json
  tar -I zstd -tf "${temporary_archive}" >/dev/null
  mv "${temporary_archive}" "${archive}"
fi

tar -I zstd -tf "${archive}" >/dev/null
archive_sha256="$(sha256sum "${archive}" | cut -d ' ' -f 1)"
temporary_checksum="${checksum}.tmp-${SLURM_JOB_ID}"
printf '%s  %s\n' "${archive_sha256}" "$(basename "${archive}")" > "${temporary_checksum}"
mv "${temporary_checksum}" "${checksum}"

printf 'Verified archive: %s\n' "${archive}"
printf 'SHA-256: %s\n' "${archive_sha256}"
printf 'Source files: %s\nSource bytes: %s\n' "${source_files}" "${source_bytes}"

final_source_digest="$(tar --sort=name --format=gnu --mtime=@0 \
  --owner=0 --group=0 --numeric-owner -cf - -C "${SOURCE_DIR}" . \
  | sha256sum | cut -d ' ' -f 1)"
test "${final_source_digest}" = "${source_tree_digest}" || {
  printf 'ERROR: sealed source changed while packing; refusing deletion.\n' >&2
  exit 2
}

if test "${DELETE_SOURCE_AFTER_VERIFY}" = "1"; then
  rm -rf -- "${SOURCE_DIR}"
  sealed_source_active=0
  printf 'Deleted verified source tree: %s\n' "${original_source_dir}"
else
  mv "${SOURCE_DIR}" "${original_source_dir}"
  sealed_source_active=0
fi
