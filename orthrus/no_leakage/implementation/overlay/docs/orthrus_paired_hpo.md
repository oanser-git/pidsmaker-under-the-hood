# Paired Orthrus HPO Protocol

## Worktrees

Both tracks start from PIDSMaker `3260273` (`2.1.1`).

```text
/home/oanser/Documents/work/coctel/PIDSMaker-orthrus-as-is
/home/oanser/Documents/work/coctel/PIDSMaker-orthrus-no-leakage
```

ULHPC deployments:

```text
/scratch/users/oanser/p-edr-hpo/orthrus-hpo/as-is/repo
/scratch/users/oanser/p-edr-hpo/orthrus-hpo/no-leakage/repo
```

## Controlled Difference

Both tracks use the same eight-point grid, model architecture, THEIA_E3 dates, and seed for HPO.

The as-is track fits Word2Vec on all splits, adapts K-means to test scores, and selects configurations and epochs by test ADP.

The no-leakage track fits Word2Vec on training only, disables test K-means, orders temporal events causally, selects configurations and epochs by validation loss, and evaluates one frozen checkpoint on test.

## Verification

Run the focused tests on Iris CPU nodes:

```bash
sbatch --clusters=iris --partition=batch -A radu.state \
  --chdir=/scratch/users/oanser/p-edr-hpo/orthrus-hpo/no-leakage/repo \
  --export=ALL,PROJECT_ROOT=/scratch/users/oanser/p-edr-hpo/orthrus-hpo/no-leakage/repo \
  scripts/slurm/ulhpc_orthrus_no_leakage_tests.sbatch
```

## Dry Runs

```bash
sbatch --test-only -A radu.state \
  --chdir=/scratch/users/oanser/p-edr-hpo/orthrus-hpo/as-is/repo \
  scripts/slurm/ulhpc_orthrus_hpo.sbatch

sbatch --test-only -A radu.state \
  --chdir=/scratch/users/oanser/p-edr-hpo/orthrus-hpo/no-leakage/repo \
  scripts/slurm/ulhpc_orthrus_hpo.sbatch
```

## HPO

Submit the two sweeps only after the focused tests pass:

```bash
sbatch -A radu.state \
  --chdir=/scratch/users/oanser/p-edr-hpo/orthrus-hpo/as-is/repo \
  scripts/slurm/ulhpc_orthrus_hpo.sbatch

sbatch -A radu.state \
  --chdir=/scratch/users/oanser/p-edr-hpo/orthrus-hpo/no-leakage/repo \
  scripts/slurm/ulhpc_orthrus_hpo.sbatch
```

The as-is sweep maximizes `adp_score`. The clean sweep minimizes `best_val_loss` and logs `best_val_loss_epoch`, `best_val_loss_model_path`, and `best_val_loss_model_sha256`.

## Frozen Final Evaluation

Before looking at test metrics, record the clean sweep's selected hyperparameters, epoch, and checkpoint SHA-256. Generate a one-run final sweep:

```bash
python scripts/analysis/create_orthrus_final_sweep.py \
  --window 15 \
  --lr 0.0001 \
  --hidden 128 \
  --dropout 0.3 \
  --epoch 3 \
  --sha256 CHECKPOINT_SHA256
```

Synchronize that generated YAML to the ULHPC no-leakage deployment, then submit:

```bash
sbatch -A radu.state \
  --chdir=/scratch/users/oanser/p-edr-hpo/orthrus-hpo/no-leakage/repo \
  scripts/slurm/ulhpc_orthrus_final_eval.sbatch
```

Final evaluation fails if the checkpoint is absent or its digest differs. It loads the checkpoint without running training, generates validation and test losses for only the selected epoch, and evaluates that epoch once.
