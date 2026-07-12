#!/bin/bash -l

#SBATCH --job-name=orthrus-no-leak-final
#SBATCH --clusters=iris
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ANALYSIS_ROOT="${ANALYSIS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROJECT_ROOT="${PIDSMAKER_ROOT:-${ANALYSIS_ROOT}/external/PIDSMaker}"
FINAL_TUNING_FILE_PATH="${FINAL_TUNING_FILE_PATH:-systems/theia_e3/tuning_orthrus_no_leakage_final}"
if [ ! -f "${PROJECT_ROOT}/config/experiments/tuning/${FINAL_TUNING_FILE_PATH}.yml" ]; then
  printf 'ERROR: final tuning file not found: %s\n' "${FINAL_TUNING_FILE_PATH}.yml" >&2
  exit 2
fi

export MODEL=orthrus_no_leakage
export DATASET=THEIA_E3
export MODE=hpo
export DUMP_DIR="${DUMP_DIR:-/scratch/users/oanser/pidsmaker_dumps}"
export ARTIFACT_BASE="${ARTIFACT_BASE:-/scratch/users/oanser/orthrus_hpo/no_leakage}"
export WANDB=1
export WANDB_PROJECT="${WANDB_PROJECT:-PIDSMaker-Orthrus-NoLeakage-THEIA-E3-Final}"
export HPO_EXP="${HPO_EXP:-orthrus_THEIA_E3_no_leakage_final}"
export TUNING_FILE_PATH="${FINAL_TUNING_FILE_PATH}"

CALIBRATION_SCRIPT="${CALIBRATION_SCRIPT:-/scratch/users/oanser/p-edr-hpo/PIDSMaker-meluxina-calibration/scripts/slurm/ulhpc_pidsmaker_calibrate.sbatch}"
test -x "${CALIBRATION_SCRIPT}" || {
  printf 'ERROR: set CALIBRATION_SCRIPT to the calibration launcher.\n' >&2
  exit 2
}
exec "${CALIBRATION_SCRIPT}"
