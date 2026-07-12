#!/bin/bash -l

#SBATCH --job-name=orthrus-nl-tests
#SBATCH --clusters=iris
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PIDS_IMAGE="${PIDS_IMAGE:-/scratch/users/oanser/p-edr-hpo/containers/pidsmaker-pids.sif}"
TEST_CACHE="${PROJECT_ROOT}/.test-cache"

mkdir -p "${TEST_CACHE}/nltk" "${TEST_CACHE}/matplotlib" "${TEST_CACHE}/xdg"
export APPTAINERENV_NLTK_DATA=/home/pids/.test-cache/nltk
export APPTAINERENV_MPLCONFIGDIR=/home/pids/.test-cache/matplotlib
export APPTAINERENV_XDG_CACHE_HOME=/home/pids/.test-cache/xdg

module purge
module load tools/Apptainer

apptainer exec --no-mount home \
  --bind "${PROJECT_ROOT}:/home/pids" \
  --pwd /home/pids \
  "${PIDS_IMAGE}" \
  python -m pytest tests/test_orthrus_no_leakage.py -q
