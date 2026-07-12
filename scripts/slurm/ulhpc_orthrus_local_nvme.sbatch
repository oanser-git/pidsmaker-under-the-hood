#!/bin/bash -l

#SBATCH --job-name=orthrus-local
#SBATCH --clusters=iris
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the deployed PIDSMaker repository}"
: "${MODEL:?Set MODEL to orthrus or orthrus_no_leakage}"

DATASET="${DATASET:-optc_h201}"
RUN_MODE="${RUN_MODE:-smoke}"
DUMP_FILE="${DUMP_FILE:-/scratch/users/oanser/pidsmaker_dumps/optc_h201.dump}"
DB_NAME="${DB_NAME:-optc_201}"
PIDS_IMAGE="${PIDS_IMAGE:-/scratch/users/oanser/p-edr-hpo/containers/pidsmaker-pids.sif}"
POSTGRES_IMAGE_SOURCE="${POSTGRES_IMAGE_SOURCE:-${PROJECT_ROOT}/scripts/apptainer/postgres.sif}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/users/oanser/.wandb_api_key}"
OBJECT_ROOT="${OBJECT_ROOT:-/scratch/users/oanser/pidsmaker_objects}"
MIN_LOCAL_FREE_BYTES="${MIN_LOCAL_FREE_BYTES:-268435456000}"
MAX_LOCAL_JOB_BYTES="${MAX_LOCAL_JOB_BYTES:-268435456000}"
TUNING_FILE_PATH="${TUNING_FILE_PATH:-}"
WANDB_PROJECT="${WANDB_PROJECT:-PIDSMaker-${MODEL}-${DATASET}-${RUN_MODE}}"
HPO_EXP="${HPO_EXP:-${MODEL}_${DATASET}_${RUN_MODE}}"

case "${RUN_MODE}" in
  smoke|hpo) ;;
  *)
    printf 'ERROR: RUN_MODE must be smoke or hpo: %s\n' "${RUN_MODE}" >&2
    exit 2
    ;;
esac
case "${MODEL}" in
  *[!A-Za-z0-9._-]*|'')
    printf 'ERROR: invalid MODEL identifier: %s\n' "${MODEL}" >&2
    exit 2
    ;;
esac
case "${DATASET}" in
  *[!A-Za-z0-9._-]*|'')
    printf 'ERROR: invalid DATASET identifier: %s\n' "${DATASET}" >&2
    exit 2
    ;;
esac
PROJECT_ROOT="$(realpath -e "${PROJECT_ROOT}")"
POSTGRES_IMAGE_SOURCE="$(realpath -e "${POSTGRES_IMAGE_SOURCE}")"
mkdir -p "${OBJECT_ROOT}"
OBJECT_ROOT="$(realpath -e "${OBJECT_ROOT}")"
test -f "${DUMP_FILE}" || {
  printf 'ERROR: dump not found: %s\n' "${DUMP_FILE}" >&2
  exit 2
}
test -f "${PIDS_IMAGE}" || {
  printf 'ERROR: PIDSMaker image not found: %s\n' "${PIDS_IMAGE}" >&2
  exit 2
}
test -f "${POSTGRES_IMAGE_SOURCE}" || {
  printf 'ERROR: PostgreSQL image not found: %s\n' "${POSTGRES_IMAGE_SOURCE}" >&2
  exit 2
}
test -f "${PROJECT_ROOT}/config/${MODEL}.yml" || {
  printf 'ERROR: model configuration not found: config/%s.yml\n' "${MODEL}" >&2
  exit 2
}
if test "${RUN_MODE}" = "hpo" && test -z "${TUNING_FILE_PATH}"; then
  printf 'ERROR: HPO requires an explicit TUNING_FILE_PATH.\n' >&2
  exit 2
fi
if test -n "${TUNING_FILE_PATH}"; then
  test -f "${PROJECT_ROOT}/config/experiments/tuning/${TUNING_FILE_PATH}.yml" || {
    printf 'ERROR: tuning configuration not found: %s\n' "${TUNING_FILE_PATH}" >&2
    exit 2
  }
fi

if ! command -v apptainer >/dev/null 2>&1; then
  module load tools/Apptainer
fi
command -v apptainer >/dev/null 2>&1 || {
  printf 'ERROR: apptainer is unavailable.\n' >&2
  exit 2
}

nvme_slot_dir="/tmp/${USER}/pidsmaker-nvme-slots"
mkdir -p "${nvme_slot_dir}"
NVME_SLOT=""
for slot in 0 1 2 3; do
  exec {slot_lock_fd}>"${nvme_slot_dir}/${slot}.lock"
  if flock -n "${slot_lock_fd}"; then
    NVME_SLOT="${slot}"
    NVME_SLOT_LOCK_FD="${slot_lock_fd}"
    break
  fi
  eval "exec ${slot_lock_fd}>&-"
done
test -n "${NVME_SLOT}" || {
  printf 'ERROR: all four PIDSMaker NVMe capacity slots are occupied.\n' >&2
  exit 2
}

local_free_bytes="$(df -PB1 /tmp | awk 'NR == 2 {print $4}')"
if test "${local_free_bytes}" -lt "${MIN_LOCAL_FREE_BYTES}"; then
  printf 'ERROR: /tmp has %s bytes free; require %s.\n' \
    "${local_free_bytes}" "${MIN_LOCAL_FREE_BYTES}" >&2
  exit 2
fi

source_digest="$(
  cd "${PROJECT_ROOT}"
  find pidsmaker config scripts/apptainer -type f \
    ! -name '*.pyc' \
    ! -name 'postgres.sif' \
    ! -path 'scripts/apptainer/postgres_data/*' \
    ! -path 'scripts/apptainer/postgres_run/*' \
    ! -path 'scripts/apptainer/postgres_log/*' \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sha256sum \
    | cut -d ' ' -f 1
)"
runner_digest="$(sha256sum "${BASH_SOURCE[0]}" | cut -d ' ' -f 1)"
container_digest="$(sha256sum "${PIDS_IMAGE}" | cut -d ' ' -f 1)"
postgres_digest="$(sha256sum "${POSTGRES_IMAGE_SOURCE}" | cut -d ' ' -f 1)"
dump_digest="$(sha256sum "${DUMP_FILE}" | cut -d ' ' -f 1)"
run_spec="${MODEL}|${DATASET}|${RUN_MODE}|${DB_NAME}|${source_digest}|${runner_digest}|${container_digest}|${postgres_digest}|${dump_digest}|${TUNING_FILE_PATH}|${WANDB_PROJECT}|${HPO_EXP}"
run_digest="$(printf '%s' "${run_spec}" | sha256sum | cut -d ' ' -f 1)"
run_id="${MODEL}-${DATASET}-${RUN_MODE}-${run_digest:0:20}"

object_dir="${OBJECT_ROOT}/${MODEL}/${DATASET}/${RUN_MODE}"
object_dir="$(realpath -m "${object_dir}")"
case "${object_dir}" in
  "${OBJECT_ROOT}"/*) ;;
  *)
    printf 'ERROR: object directory escaped OBJECT_ROOT: %s\n' "${object_dir}" >&2
    exit 2
    ;;
esac
archive="${object_dir}/${run_id}.tar.zst"
checksum="${archive}.sha256"
mkdir -p "${object_dir}"
exec 9>"${object_dir}/.${run_id}.lock"
flock -n 9 || {
  printf 'ERROR: another job is running object %s.\n' "${run_id}" >&2
  exit 2
}

if test -f "${archive}"; then
  tar -I zstd -tf "${archive}" >/dev/null
  if ! test -f "${checksum}"; then
    recovery_checksum="${checksum}.recovery-${SLURM_JOB_ID}"
    printf '%s  %s\n' "$(sha256sum "${archive}" | cut -d ' ' -f 1)" \
      "$(basename "${archive}")" > "${recovery_checksum}"
    mv "${recovery_checksum}" "${checksum}"
  fi
  (cd "${object_dir}" && sha256sum -c "$(basename "${checksum}")")
  printf 'Verified existing object; skipping rerun: %s\n' "${archive}"
  exit 0
fi

local_root="/tmp/${USER}/pidsmaker/${SLURM_JOB_ID}-${run_id}"
local_artifacts="${local_root}/artifacts"
local_apptainer="${local_root}/apptainer"
local_home="${local_root}/home"
case "${local_root}" in
  "/tmp/${USER}/pidsmaker/"*) ;;
  *)
    printf 'ERROR: unsafe local root: %s\n' "${local_root}" >&2
    exit 2
    ;;
esac

temporary_archive=""
temporary_checksum=""
postgres_started=0
storage_monitor_pid=""
active_pid_file="${local_root}/active-process.pid"

terminate_active_process() {
  if test -s "${active_pid_file}"; then
    active_pid="$(cat "${active_pid_file}")"
    if [[ "${active_pid}" =~ ^[0-9]+$ ]]; then
      kill -TERM -- "-${active_pid}" >/dev/null 2>&1 || true
      sleep 5
      kill -KILL -- "-${active_pid}" >/dev/null 2>&1 || true
    fi
  fi
}

run_active() {
  local child_pid
  local child_status=0
  setsid "$@" &
  child_pid="$!"
  printf '%s\n' "${child_pid}" > "${active_pid_file}"
  wait "${child_pid}" || child_status="$?"
  rm -f -- "${active_pid_file}"
  return "${child_status}"
}

cleanup() {
  stop_succeeded=1
  terminate_active_process
  if test "${postgres_started}" = "1"; then
    pushd "${local_apptainer}" >/dev/null 2>&1 || stop_succeeded=0
    if test "${stop_succeeded}" = "1"; then
      ./postgres-stop.sh >/dev/null 2>&1 || stop_succeeded=0
      popd >/dev/null 2>&1 || true
    fi
  fi
  if test -n "${temporary_archive}"; then
    rm -f -- "${temporary_archive}"
  fi
  if test -n "${temporary_checksum}"; then
    rm -f -- "${temporary_checksum}"
  fi
  if test -n "${storage_monitor_pid}"; then
    kill "${storage_monitor_pid}" >/dev/null 2>&1 || true
  fi
  if test "${stop_succeeded}" = "1"; then
    case "${local_root}" in
      "/tmp/${USER}/pidsmaker/"*) rm -rf -- "${local_root}" ;;
    esac
  else
    printf 'WARNING: PostgreSQL did not stop; preserving local root: %s\n' \
      "${local_root}" >&2
  fi
}
trap cleanup EXIT
trap 'exit 70' INT TERM

mkdir -p "${local_artifacts}" "${local_apptainer}" "${local_home}"
runner_pid="$$"
(
  while kill -0 "${runner_pid}" >/dev/null 2>&1; do
    local_bytes="$(du -sb "${local_root}" | cut -f1)"
    node_free_bytes="$(df -PB1 /tmp | awk 'NR == 2 {print $4}')"
    if test "${local_bytes}" -gt "${MAX_LOCAL_JOB_BYTES}" \
      || test "${node_free_bytes}" -lt 21474836480; then
      printf 'ERROR: local NVMe safety limit reached: job_bytes=%s free_bytes=%s\n' \
        "${local_bytes}" "${node_free_bytes}" >&2
      touch "${local_root}/nvme-limit-exceeded"
      terminate_active_process
      kill -TERM "${runner_pid}" >/dev/null 2>&1 || true
      break
    fi
    sleep 60
  done
) &
storage_monitor_pid="$!"

run_active rsync -a \
  --exclude postgres_data \
  --exclude postgres_run \
  --exclude postgres_log \
  --exclude 'postgres-*.pid' \
  --exclude postgres.pid \
  "${PROJECT_ROOT}/scripts/apptainer/" "${local_apptainer}/"
run_active cp "${POSTGRES_IMAGE_SOURCE}" "${local_apptainer}/postgres.sif"

export APPTAINER_CACHEDIR="${local_root}/apptainer-cache"
export APPTAINER_TMPDIR="${local_root}/apptainer-tmp"
export APPTAINER_SESSIONDIR="${local_root}/apptainer-sessions"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_SESSIONDIR}"

port_lock_dir="/tmp/${USER}/pidsmaker-port-locks"
mkdir -p "${port_lock_dir}"
POSTGRES_PORT=""
for offset in $(seq 0 9999); do
  candidate_port="$((20000 + (SLURM_JOB_ID + offset) % 10000))"
  candidate_lock="${port_lock_dir}/${candidate_port}.lock"
  exec {candidate_lock_fd}>"${candidate_lock}"
  if flock -n "${candidate_lock_fd}"; then
    if ! ss -ltnH "sport = :${candidate_port}" | grep -q .; then
      POSTGRES_PORT="${candidate_port}"
      POSTGRES_PORT_LOCK_FD="${candidate_lock_fd}"
      break
    fi
  fi
  eval "exec ${candidate_lock_fd}>&-"
done
test -n "${POSTGRES_PORT}" || {
  printf 'ERROR: no free PostgreSQL port available.\n' >&2
  exit 2
}
export POSTGRES_INSTANCE="pids_pg_${SLURM_JOB_ID}"
export POSTGRES_PORT
export POSTGRES_IMAGE="${local_apptainer}/postgres.sif"
export POSTGRES_DATA_DIR="${local_root}/postgres-data"
export POSTGRES_RUN_DIR="${local_root}/postgres-run"
export POSTGRES_LOG_DIR="${local_root}/postgres-log"
export POSTGRES_PID_FILE="${local_root}/postgres.pid"

postgres_started=1
run_active env INPUT_DIR="$(dirname "${DUMP_FILE}")" \
  bash -c 'cd "$1" && exec ./postgres-start.sh' _ "${local_apptainer}"

run_active apptainer exec "instance://${POSTGRES_INSTANCE}" \
  psql -U postgres -h localhost -p "${POSTGRES_PORT}" -d postgres \
  -c "CREATE DATABASE \"${DB_NAME}\";"
run_active apptainer exec "instance://${POSTGRES_INSTANCE}" \
  pg_restore -U postgres -h localhost -p "${POSTGRES_PORT}" \
  --no-owner --no-privileges --exit-on-error \
  -d "${DB_NAME}" "/data/$(basename "${DUMP_FILE}")"
table_count="$(apptainer exec "instance://${POSTGRES_INSTANCE}" \
  psql -U postgres -h localhost -p "${POSTGRES_PORT}" -d "${DB_NAME}" \
  -tAc "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")"
test "${table_count}" -gt 0 || {
  printf 'ERROR: restored database has no public tables.\n' >&2
  exit 2
}

export APPTAINERENV_WANDB_API_KEY
APPTAINERENV_WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_API_KEY_FILE}")"
export APPTAINERENV_WANDB_DIR=/home/artifacts/wandb
export APPTAINERENV_WANDB_MODE=online
export APPTAINERENV_NLTK_DATA=/home/artifacts/nltk_data
export APPTAINERENV_MPLCONFIGDIR=/home/artifacts/matplotlib
export APPTAINERENV_XDG_CACHE_HOME=/home/artifacts/cache
export APPTAINERENV_PYTHONHASHSEED=0
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_PYTHONPYCACHEPREFIX=/home/artifacts/pycache
mkdir -p "${local_artifacts}/wandb" "${local_artifacts}/nltk_data" \
  "${local_artifacts}/matplotlib" "${local_artifacts}/cache"

run_args=(
  "${MODEL}" "${DATASET}"
  --artifact_dir /home/artifacts
  --database_host localhost
  --database_port "${POSTGRES_PORT}"
  --database_user postgres
  --database_password postgres
  --exp "${HPO_EXP}"
  --wandb
  --project "${WANDB_PROJECT}"
)
if test "${RUN_MODE}" = "smoke"; then
  run_args+=(
    --force_restart=construction
    --featurization.epochs=1
    --training.num_epochs=1
    --tags "smoke,local-nvme,${MODEL},${DATASET}"
  )
else
  run_args+=(--tuning_mode=hyperparameters)
  if test -n "${TUNING_FILE_PATH}"; then
    run_args+=(--tuning_file_path "${TUNING_FILE_PATH}")
  fi
fi

printf 'run_id=%s\nlocal_root=%s\npostgres_port=%s\n' \
  "${run_id}" "${local_root}" "${POSTGRES_PORT}"
set +e
run_active apptainer exec --nv --no-mount home \
  --home "${local_home}:/home/users/${USER}" \
  --bind "${PROJECT_ROOT}:/home/pids:ro" \
  --bind "${local_artifacts}:/home/artifacts" \
  --pwd /home/pids \
  "${PIDS_IMAGE}" \
  python -u -m pidsmaker.main "${run_args[@]}"
run_status="$?"
kill "${storage_monitor_pid}" >/dev/null 2>&1 || true
wait "${storage_monitor_pid}" >/dev/null 2>&1 || true
storage_monitor_pid=""
if test -f "${local_root}/nvme-limit-exceeded"; then
  run_status=70
fi
set -e

cat > "${local_root}/manifest.json" <<EOF
{
  "schema_version": 1,
  "run_id": "${run_id}",
  "model": "${MODEL}",
  "dataset": "${DATASET}",
  "mode": "${RUN_MODE}",
  "source_digest": "${source_digest}",
  "container_sha256": "${container_digest}",
  "postgres_sha256": "${postgres_digest}",
  "dump_sha256": "${dump_digest}",
  "runner_sha256": "${runner_digest}",
  "database": "${DB_NAME}",
  "wandb_project": "${WANDB_PROJECT}",
  "experiment": "${HPO_EXP}",
  "tuning_file_path": "${TUNING_FILE_PATH}",
  "nvme_slot": "${NVME_SLOT}",
  "slurm_job_id": "${SLURM_JOB_ID}",
  "run_status": ${run_status}
}
EOF

publish_archive="${archive}"
archive_members=(manifest.json artifacts)
if test "${run_status}" != "0"; then
  publish_archive="${object_dir}/${run_id}.failed-${SLURM_JOB_ID}.tar.zst"
  archive_members=(manifest.json artifacts/wandb)
fi
publish_checksum="${publish_archive}.sha256"
temporary_archive="${publish_archive}.tmp-${SLURM_JOB_ID}"
tar -I "zstd -T${SLURM_CPUS_PER_TASK:-1} -3" -cf "${temporary_archive}" \
  -C "${local_root}" "${archive_members[@]}"
tar -I zstd -tf "${temporary_archive}" >/dev/null
mv "${temporary_archive}" "${publish_archive}"

archive_sha256="$(sha256sum "${publish_archive}" | cut -d ' ' -f 1)"
temporary_checksum="${publish_checksum}.tmp-${SLURM_JOB_ID}"
printf '%s  %s\n' "${archive_sha256}" "$(basename "${publish_archive}")" \
  > "${temporary_checksum}"
mv "${temporary_checksum}" "${publish_checksum}"
printf 'Published object: %s\nSHA-256: %s\n' "${publish_archive}" "${archive_sha256}"

exit "${run_status}"
