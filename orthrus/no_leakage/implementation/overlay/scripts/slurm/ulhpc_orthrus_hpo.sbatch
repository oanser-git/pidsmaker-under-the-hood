#!/bin/bash -l

#SBATCH --job-name=orthrus-no-leak-hpo
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

export MODEL=orthrus_no_leakage
export DATASET=THEIA_E3
export MODE=hpo
export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export DUMP_DIR="${DUMP_DIR:-/scratch/users/oanser/pidsmaker_dumps}"
export ARTIFACT_BASE="${ARTIFACT_BASE:-/scratch/users/oanser/orthrus_hpo/no_leakage}"
export WANDB=1
export WANDB_PROJECT="${WANDB_PROJECT:-PIDSMaker-Orthrus-NoLeakage-THEIA-E3}"
export HPO_EXP="${HPO_EXP:-orthrus_THEIA_E3_no_leakage}"

CALIBRATION_SCRIPT="${CALIBRATION_SCRIPT:-/scratch/users/oanser/p-edr-hpo/PIDSMaker-meluxina-calibration/scripts/slurm/ulhpc_pidsmaker_calibrate.sbatch}"
test -x "${CALIBRATION_SCRIPT}" || {
  printf 'ERROR: set CALIBRATION_SCRIPT to the calibration launcher.\n' >&2
  exit 2
}
exec "${CALIBRATION_SCRIPT}"
