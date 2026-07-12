#!/bin/bash -l

#SBATCH --job-name=orthrus-no-leak-t5
#SBATCH --clusters=iris
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/scratch/users/oanser/orthrus_t5/no_leakage/orthrus_no_leakage/THEIA_E3/artifacts}"
DUMP_DIR="${DUMP_DIR:-/scratch/users/oanser/pidsmaker_dumps}"
PIDS_IMAGE="${PIDS_IMAGE:-/scratch/users/oanser/p-edr-hpo/containers/pidsmaker-pids.sif}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/users/oanser/.wandb_api_key}"
APPTAINER_DIR="${PROJECT_ROOT}/scripts/apptainer"
MEASURE_DIR="${ARTIFACT_DIR%/artifacts}/measurements/t5-${SLURM_JOB_ID}"
WANDB_PROJECT="${WANDB_PROJECT:-PIDSMaker-Orthrus-NoLeakage-THEIA-E3-T5}"

if ! command -v apptainer >/dev/null 2>&1; then
  module load tools/Apptainer
fi
command -v apptainer >/dev/null 2>&1 || {
  printf 'ERROR: apptainer is unavailable.\n' >&2
  exit 2
}
CONTAINER_CMD=apptainer

mkdir -p "${ARTIFACT_DIR}/wandb" "${ARTIFACT_DIR}/nltk_data" \
  "${ARTIFACT_DIR}/matplotlib" "${ARTIFACT_DIR}/cache" "${MEASURE_DIR}/apptainer_home"

export APPTAINERENV_WANDB_API_KEY
APPTAINERENV_WANDB_API_KEY="$(tr -d '\r\n' < "${WANDB_API_KEY_FILE}")"
export APPTAINERENV_WANDB_DIR=/home/artifacts/wandb
export APPTAINERENV_WANDB_MODE=online
export APPTAINERENV_NLTK_DATA=/home/artifacts/nltk_data
export APPTAINERENV_MPLCONFIGDIR=/home/artifacts/matplotlib
export APPTAINERENV_XDG_CACHE_HOME=/home/artifacts/cache

cleanup() {
  pushd "${APPTAINER_DIR}" >/dev/null || return
  ./postgres-stop.sh >/dev/null 2>&1 || true
  popd >/dev/null || true
}
trap cleanup EXIT

pushd "${APPTAINER_DIR}" >/dev/null
INPUT_DIR="${DUMP_DIR}" ./postgres-start.sh
popd >/dev/null

run_pipeline() {
  local seed="$1"
  local phase="$2"
  shift 2

  "${CONTAINER_CMD}" exec --nv --no-mount home \
    --home "${MEASURE_DIR}/apptainer_home:/home/users/oanser" \
    --bind "${PROJECT_ROOT}:/home/pids" \
    --bind "${ARTIFACT_DIR}:/home/artifacts" \
    --pwd /home/pids \
    "${PIDS_IMAGE}" \
    python -u -m pidsmaker.main orthrus_no_leakage THEIA_E3 \
      --artifact_dir /home/artifacts \
      --database_host localhost \
      --database_port 5432 \
      --database_user postgres \
      --database_password postgres \
      --exp "orthrus_THEIA_E3_no_leakage_best_hpo_seed_${seed}_${phase}" \
      --tags "best-hpo,t5,seed-${seed},no-leakage,${phase}" \
      --wandb \
      --project "${WANDB_PROJECT}" \
      --construction.time_window_size=5 \
      --featurization.seed="${seed}" \
      --training.seed="${seed}" \
      --training.lr=0.0001 \
      --training.node_hid_dim=128 \
      --training.node_out_dim=128 \
      --training.encoder.dropout=0.3 \
      "$@"
}

for seed in 0 1 2 3 4; do
  export APPTAINERENV_PYTHONHASHSEED="${seed}"
  printf 'Selecting no-leakage checkpoint for seed %s/4\n' "${seed}"
  run_pipeline "${seed}" selection --force_restart=featurization

  selection_summary="${ARTIFACT_DIR}/wandb/wandb/latest-run/files/wandb-summary.json"
  test -s "${selection_summary}"
  cp "${selection_summary}" "${MEASURE_DIR}/seed-${seed}-selection-summary.json"
  selected_epoch="$(jq -er '.best_val_loss_epoch | numbers' "${selection_summary}")"
  checkpoint_sha="$(jq -er \
    '.best_val_loss_model_sha256 | select(type == "string" and test("^[0-9a-f]{64}$"))' \
    "${selection_summary}")"
  printf '%s\t%s\t%s\n' "${seed}" "${selected_epoch}" "${checkpoint_sha}" \
    >> "${MEASURE_DIR}/selected-checkpoints.tsv"

  printf 'Evaluating frozen no-leakage checkpoint for seed %s/4\n' "${seed}"
  run_pipeline "${seed}" test \
    --evaluation.selected_epoch="${selected_epoch}" \
    --evaluation.expected_checkpoint_sha256="${checkpoint_sha}"

  final_summary="${ARTIFACT_DIR}/wandb/wandb/latest-run/files/wandb-summary.json"
  test -s "${final_summary}"
  cp "${final_summary}" "${MEASURE_DIR}/seed-${seed}-test-summary.json"
done

printf 'Completed all five no-leakage seeds.\n'
