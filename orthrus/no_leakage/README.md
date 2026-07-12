# Orthrus no-leakage

This directory is the portable entry point for the strict no-leakage Orthrus
implementation and its five-seed THEIA E3 results. It is intentionally separate
from the as-is Orthrus implementation.

## Experiment

- Dataset: `THEIA_E3`
- Seeds: featurization and training seeds `0..4`, paired per run
- Window: `5` minutes
- Learning rate: `0.0001`
- Hidden/output dimensions: `128`
- Dropout: `0.3`
- Selection: minimum validation loss only
- Test: one frozen evaluation of the selected checkpoint
- W&B: <https://wandb.ai/omar-anser-university-of-luxembourg/PIDSMaker-Orthrus-NoLeakage-THEIA-E3-T5>

The selected epochs are `7, 7, 5, 3, 11`. Mean ADP is `0.0164`, with a
minimum of `0.003`, maximum of `0.045`, and population standard deviation of
`0.0147`.

Compact metrics are in `t5-5524660/results.csv`. The original W&B selection and
frozen-test summaries are in `t5-5524660/summaries/`.

## Materialize The Implementation

The implementation is stored as a complete overlay against PIDSMaker commit
`32602734bc9f896be5fc0f03f0a185c967cd6624`.

```bash
./orthrus/no_leakage/implementation/materialize.sh ../PIDSMaker-orthrus-no-leakage
export PIDSMAKER_SRC="$PWD/../PIDSMaker-orthrus-no-leakage"
```

The materializer refuses to modify a dirty destination repository.

## Retrieve The Artifacts

The checkpoints, scores, feature outputs, and edge losses occupy 59.96 GB when
extracted. They are sealed into one 5.4 GB immutable object on Iris instead of
being stored as 45,612 loose scratch files. Pull and extract it after cloning
this repository on the GPU machine:

```bash
./scripts/sync_orthrus_no_leakage.sh
python scripts/verify_orthrus_no_leakage.py
```

The default extracted artifact root is
`artifacts/orthrus/no_leakage/THEIA_E3/t5-5524660/artifacts`. Override the SSH
alias, remote object, or destination when needed:

```bash
ORTHRUS_IRIS_REMOTE=user@access-iris.uni.lu \
ORTHRUS_NO_LEAKAGE_OBJECT_SOURCE=/scratch/users/oanser/pidsmaker_objects/orthrus/no_leakage/THEIA_E3/orthrus-no-leakage-THEIA_E3-t5-5524660.tar.zst \
./scripts/sync_orthrus_no_leakage.sh /data/orthrus-no-leakage
```

The destination is ignored by Git. The verification script checks each selected
checkpoint and score SHA-256, non-empty selected validation/test edge-loss paths,
and file counts and byte totals for every synchronized artifact tree.

Object SHA-256:
`5586ae6b032990f177590e1c19cf391ab3b002fa3e23a513fa409a2215f49407`.

The historical Iris HPO launchers use a separate calibration wrapper. Set
`CALIBRATION_SCRIPT` when using those launchers outside their original checkout.
