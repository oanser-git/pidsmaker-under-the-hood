# Orthrus Paired Analysis Plan

## Scope

Compare the original Orthrus protocol with the strict no-leakage protocol on
THEIA_E3. Treat this as a bundled protocol contrast. The variants differ in
featurization scope, temporal ordering, model selection, test-time clustering,
and frozen evaluation, so this comparison cannot by itself identify one change
as the cause of the performance difference.

The primary unit of replication is a matched seed. Report individual seeds,
paired seed differences, and uncertainty across seeds. Do not report only the
best run.

## Interpretation Constraints

- An edge loss is cross-entropy for the observed event type, not a malicious
  probability.
- A node score is the maximum loss of an incident test edge.
- Time controls causal neighbor selection but is not encoded as a numerical
  model input.
- Orthrus uses one-hop, last-20, undirected historical neighborhoods.
- Current edges are added to history after each 1024-edge batch, so earlier
  events in the same batch are not context for later events in that batch.
- Fused edges can represent repeated raw events, and saved losses do not retain
  a unique event UUID.
- Internal explanations describe the sampled temporal neighborhood, not the
  complete provenance graph.

## Phase 0: Artifact Gate

Require the following before using a seed in a result table:

- Archive SHA-256 matches its published checksum.
- Selected checkpoint and score SHA-256 match the seed manifest.
- Selected epoch, hyperparameters, and selection policy are recorded.
- Validation and test edge-loss directories are present and non-empty.
- Score node IDs, labels, and attack mappings have consistent lengths.
- Reconstructed node maxima match saved node scores within `1e-6`.

Keep as-is and strict manifests separate. Never infer one variant's threshold or
selection rule from the other variant.

## Phase 1: All-Seed Output Analysis

This phase uses compact artifacts and does not require `feat_inference`.

For every seed, produce:

- ADP, AP, ROC AUC, discrimination, precision, recall, F1, TP, FP, FN, and TN.
- Rank of every malicious node and rank required to cover each attack.
- Top-k precision and attack coverage for fixed analyst budgets.
- Score quantiles for malicious and benign nodes.
- Validation maximum edge loss and the saved decision-policy result.
- The highest-loss incident edge for every malicious node, TP, FP, and selected
  high-scoring benign control.
- Responsible edge type, endpoint roles, timestamp, attack-window membership,
  and incident-edge count.

Use threshold-independent ranking and attack-coverage metrics as the primary
paired comparison because the original and strict variants use different
decision policies. Report policy-specific confusion matrices as secondary
operational results.

Across seeds, report:

- Paired per-seed differences with median, range, mean, and bootstrap interval.
- Malicious-node rank stability and top-k overlap.
- Responsible event-type and endpoint-role stability.
- False-positive node and responsible-edge concentration.
- Score correlation only after joining by node ID; never correlate array
  positions directly.

## Phase 2: Explanation Cohort

Pre-register seed 1 for internal explanation. It is the median strict seed by
ADP and avoids selecting the most favorable run. Use seed 1 in both variants.

Build a fixed cohort before inspecting internal attributions:

- Every malicious node.
- Every true positive.
- The highest-ranked false negatives for each attack.
- Up to 20 highest-ranked false positives.
- Up to 20 benign nodes immediately below the decision boundary.
- Matched benign controls with similar node type and incident-edge count.

For every cohort node, select its exact maximum-loss incident edge and preserve
ties. If `(src, dst, time, edge_type)` is non-unique, reconstruct deterministic
`event_order` from the transformed graph rather than claiming a unique raw UUID.

## Phase 3: Exact Replay And Ablation

Strict seed 1 uses feature tree
`4b34c10eff96d271b5aba564008412b8c62d863f6ab3bf4fffd5ddfade13b8a4`.

Before attribution, reconstruct each target batch chronologically and require
the replayed edge loss to match the saved loss within `1e-6`. Reject the target
if exact replay fails.

Run these counterfactuals on a frozen batch and checkpoint:

- Zero the 128 Word2Vec node-label dimensions.
- Zero the three node-type dimensions.
- Zero historical edge-type attributes while retaining the current event type
  as the prediction target.
- Remove each historical edge independently for small neighborhoods.
- Remove temporal buckets of historical edges for larger neighborhoods.
- Reverse source/destination roles only as a sensitivity test, not a realistic
  intervention.

Measure change in observed-event cross-entropy and predicted event-type rank.
Report absolute effects and effects relative to the unmodified loss.

## Phase 4: GNN Attribution

Wrap the graph encoder and decoder in a differentiable scalar objective for one
target edge. Do not use the inference path because it disables gradients.

Run gradient edge masks first. Use GNNExplainer only after gradient results pass
sanity checks. Compare attribution rankings against leave-one-edge-out loss
changes and random edge masks.

Required sanity checks:

- Parameter randomization reduces attribution agreement.
- Label-target randomization changes the attribution.
- Important-edge removal changes the target loss more than random removal.
- Results are stable across at least three explainer initializations.

Report source/destination IDs, event type, relative age, endpoint role, and mask
weight for each attributed historical edge.

## Phase 5: Mechanism Isolation

Only use causal language about an individual leakage mechanism after factorial
experiments that change one mechanism at a time. At minimum isolate:

- Train-only versus all-split Word2Vec.
- Causal versus non-causal split ordering.
- Validation-loss versus test-ADP epoch selection.
- Validation threshold versus test-driven k-means.

Use identical seeds, architecture, windows, and checkpoints where the mechanism
allows it. Otherwise describe the result as an ablation association.

## Storage And Runtime Strategy

The compact strict tree occupies about 6.5 GB. The complete strict
`feat_inference` tree is 53.07 GB and does not fit on the current disk.

- Keep all-seed score, checkpoint, and edge-loss artifacts compact.
- Keep completed as-is seed archives compressed and extract only `training` and
  `evaluation` for Phase 1.
- Expand only seed 1 feature data for Phase 3.
- Process strict and as-is internal replay sequentially if both feature trees do
  not fit simultaneously.
- Preserve remote immutable objects and checksums before deleting any local
  compressed copy to make room.
- Use the original pinned runtime for exact replay. The local Torch 2.12/PyG
  2.8 environment is suitable for compact analysis but differs from the
  original Torch 1.13 runtime.

## Deliverables

- `seed_metrics.csv`: one row per variant and seed.
- `malicious_node_ranks.csv`: all malicious-node ranks and attack labels.
- `responsible_edges.csv`: score decomposition with exact edge-loss provenance.
- `paired_seed_deltas.csv`: matched as-is minus strict metrics.
- `rank_stability.csv`: cross-seed and cross-variant node-rank comparisons.
- `explanation_cohort.csv`: frozen seed-1 targets and controls.
- `replay_validation.csv`: saved versus replayed edge losses.
- `feature_ablations.csv`: counterfactual loss changes.
- `temporal_edge_attributions.csv`: edge-mask and deletion effects.
- A methods note separating output decomposition, model attribution, and
  mechanism-isolation evidence.
