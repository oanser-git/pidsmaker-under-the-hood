import argparse
import csv
import math
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from poc_paths import artifacts_root, dataset_inputs, first_existing


ROOT = artifacts_root()
OBS_ROOT = ROOT / "observations"

RUNS = {
    "THEIA_E3": {
        "slug": "theia_e3",
        "obs_id": "2026-07-01_theia_e3_velox",
        "db": "theia_e3",
        "log": "artifacts/velox_THEIA_E3_from_weights.log",
        "paper_best": "1.00",
    },
    "CADETS_E3": {
        "slug": "cadets_e3",
        "obs_id": "2026-07-01_cadets_e3_velox",
        "db": "cadets_e3",
        "log": "artifacts/velox_CADETS_E3_from_weights_sanity.log",
        "paper_best": "1.00",
    },
    "optc_h201": {
        "slug": "optc_h201",
        "obs_id": "2026-07-01_optc_h201_velox",
        "db": "optc_201",
        "log": "artifacts/velox_optc_h201_from_weights.log",
        "paper_best": "1.00",
    },
    "optc_h501": {
        "slug": "optc_h501",
        "obs_id": "2026-07-01_optc_h501_velox",
        "db": "optc_501",
        "log": "artifacts/velox_optc_h501_from_weights.log",
        "paper_best": "0.50",
    },
    "optc_h051": {
        "slug": "optc_h051",
        "obs_id": "2026-07-01_optc_h051_velox",
        "db": "optc_051",
        "log": "artifacts/velox_optc_h051_from_weights.log",
        "paper_best": "1.00",
        "caveat": "Local ADP does not match the paper/reference best ADP for optc_h051.",
    },
}

TYPE_COLORS = {
    "subject": "#2563eb",
    "file": "#e11d48",
    "netflow": "#059669",
    "": "#64748b",
}

ATTACK_PALETTE = ["#e11d48", "#2563eb", "#059669", "#f97316", "#7c3aed", "#0891b2", "#be123c"]


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#0f172a",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "font.size": 10.5,
        "savefig.facecolor": "white",
        "legend.frameon": False,
    })


def save(fig, obs_dir, stem):
    slug = obs_dir.name.split("_", 1)[1].removesuffix("_velox")
    plot_dir = obs_dir / "plots" / "from_saved_node_scores"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"{slug}_produced_{stem}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)
    plt.close(fig)


def confusion(labels, pred):
    labels = np.asarray(labels, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def ranked_order(scores, nodes):
    # Match PIDSMaker's all-attacks ranking tie-break: sorted(zip(scores, nodes), reverse=True).
    return np.asarray(sorted(range(len(scores)), key=lambda i: (float(scores[i]), int(nodes[i])), reverse=True), dtype=int)


def load_run(run):
    input_dir = dataset_inputs(run["slug"])
    score_path = first_existing(
        input_dir / "scores_model_epoch_0.pkl",
        input_dir / "precision_recall_dir" / "scores_model_epoch_0.pkl",
    )
    result_path = first_existing(
        input_dir / "results.pth",
        input_dir / "results" / "results.pth",
    )
    stats_path = first_existing(
        input_dir / "stats_model_epoch_0.pth",
        input_dir / "precision_recall_dir" / "stats_model_epoch_0.pth",
    )
    score_blob = torch.load(score_path)
    results = torch.load(result_path)
    stats = torch.load(stats_path) if stats_path.exists() else {}
    scores = np.asarray(score_blob["pred_scores"], dtype=float)
    labels = np.asarray(score_blob["y_truth"], dtype=int)
    nodes = np.asarray(score_blob["nodes"], dtype=int)
    node2attacks = {int(k): set(v) for k, v in score_blob["node2attacks"].items()}
    yhat = np.asarray([int(results[int(node)]["y_hat"]) for node in nodes], dtype=int)
    return score_path, result_path, stats_path, scores, labels, nodes, node2attacks, yhat, stats


def scan_max_loss(path):
    threshold = None
    for csv_path in path.glob("*.csv"):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                loss = float(row["loss"])
                if threshold is None or loss > threshold:
                    threshold = loss
    return threshold


def find_threshold(dataset, scores, yhat, input_dir):
    lower = float(scores[yhat == 0].max()) if (yhat == 0).any() else float("nan")
    upper = float(scores[yhat == 1].min()) if (yhat == 1).any() else float("nan")
    candidates = []
    path = input_dir / "edge_losses" / "val" / "model_epoch_0"
    if path.exists():
        max_loss = scan_max_loss(path)
        if max_loss is None:
            max_loss = None
        elif np.array_equal((scores > max_loss).astype(int), yhat):
            candidates.append((max_loss, path))

    if candidates:
        candidates.sort(key=lambda item: abs(item[0] - lower) if not math.isnan(lower) else 0)
        return {
            "threshold": float(candidates[0][0]),
            "kind": "validation max edge loss",
            "source": candidates[0][1],
            "lower": lower,
            "upper": upper,
        }

    if (yhat == 1).any() and (yhat == 0).any():
        return {
            "threshold": (lower + upper) / 2,
            "kind": "implied saved-prediction boundary midpoint",
            "source": "not found",
            "lower": lower,
            "upper": upper,
        }

    return {"threshold": float("nan"), "kind": "unavailable", "source": "not found", "lower": lower, "upper": upper}


def find_all_attacks_cutoff(scores, labels, nodes, node2attacks, order):
    all_attacks = set()
    for attacks in node2attacks.values():
        all_attacks.update(attacks)
    if not all_attacks:
        return None

    detected_attacks = set()
    tps_before, fps_before = 0, 0
    sorted_nodes = nodes[order]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    for rank, (node, score, label) in enumerate(zip(sorted_nodes, sorted_scores, sorted_labels), start=1):
        attacks = set(node2attacks.get(int(node), set()))
        newly_detected = detected_attacks | attacks
        if attacks and newly_detected == all_attacks:
            score_mask = scores >= score
            return {
                "rank": rank,
                "score": float(score),
                "node": int(node),
                "attacks": sorted(attacks),
                "analyst_tps_before": tps_before,
                "analyst_fps_before": fps_before,
                "rank_prefix_tp": int(sorted_labels[:rank].sum()),
                "rank_prefix_fp": int(rank - sorted_labels[:rank].sum()),
                "score_cutoff_tp": int(((score_mask == 1) & (labels == 1)).sum()),
                "score_cutoff_fp": int(((score_mask == 1) & (labels == 0)).sum()),
                "score_cutoff_fn": int(((score_mask == 0) & (labels == 1)).sum()),
            }
        if attacks:
            tps_before += 1
        else:
            fps_before += 1
        detected_attacks = newly_detected
    return None


def find_pidmaker_all_attacks_cutoff(scores, nodes, node2attacks):
    """Match PIDSMaker get_metrics_if_all_attacks_detected ordering exactly.

    PIDSMaker uses sorted(zip(scores, nodes), reverse=True), so score ties are
    broken by node id rather than by numpy argsort/index order.
    """
    all_attacks = set()
    for attacks in node2attacks.values():
        all_attacks.update(attacks)
    if not all_attacks:
        return None

    detected_attacks = set()
    tps_before, fps_before = 0, 0
    for rank, (score, node) in enumerate(sorted(zip(scores, nodes), reverse=True), start=1):
        attacks = set(node2attacks.get(int(node), set()))
        newly_detected = detected_attacks | attacks
        if newly_detected == all_attacks:
            return {
                "pidmaker_rank": rank,
                "pidmaker_score": float(score),
                "pidmaker_node": int(node),
                "pidmaker_tps_before": tps_before,
                "pidmaker_fps_before": fps_before,
            }
        if attacks:
            tps_before += 1
        else:
            fps_before += 1
        detected_attacks = newly_detected
    return None


def adp_curve(scores, labels, nodes, node2attacks):
    sorted_indices = np.argsort(scores)[::-1]
    total_attacks = len(set(a for attacks in node2attacks.values() for a in attacks))
    if total_attacks == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0]), 0.0
    detected_attacks = set()
    detected_percentages = [0.0]
    precisions = [0.0]
    tp = 0
    fp = 0
    for idx in sorted_indices:
        node = int(nodes[idx])
        label = int(labels[idx])
        if label:
            tp += 1
        else:
            fp += 1
        if node in node2attacks:
            detected_attacks.update(node2attacks[node])
        precisions.append(tp / (tp + fp))
        detected_percentages.append((len(detected_attacks) / total_attacks) * 100)

    precision_to_attacks = {}
    for precision, detected_percentage in zip(precisions, detected_percentages):
        precision_to_attacks[precision] = max(precision_to_attacks.get(precision, 0.0), detected_percentage)
    xs = np.asarray(sorted(precision_to_attacks), dtype=float)
    ys = np.asarray([precision_to_attacks[x] for x in xs], dtype=float)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return xs, ys, trapz(ys, xs) / 100


def downsample(x, y, max_points=5000):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) <= max_points:
        return x, y
    idx = np.unique(np.r_[0, np.linspace(0, len(x) - 1, max_points).astype(int), len(x) - 1])
    return x[idx], y[idx]


def load_metadata(obs_dir):
    path = obs_dir / "tables/attack_node_metadata.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {int(row["node_id"]): row for row in csv.DictReader(f)}


def node_label(node, metadata):
    meta = metadata.get(int(node), {})
    node_type = meta.get("node_type", "")
    if node_type == "netflow":
        return f"netflow {meta.get('src_addr', '')}:{meta.get('src_port', '')}->{meta.get('dst_addr', '')}:{meta.get('dst_port', '')}"
    path = meta.get("path") or ""
    cmd = meta.get("cmd") or ""
    if path and cmd and cmd != path:
        return f"{node_type}: {path} | {cmd}"
    if path:
        return f"{node_type}: {path}"
    return f"node {node}"


def attack_color(attack):
    return ATTACK_PALETTE[int(attack) % len(ATTACK_PALETTE)]


def plot_distribution(obs_dir, dataset, scores, labels, threshold_info, metrics, all_attacks):
    benign = labels == 0
    malicious = labels == 1
    fig, ax = plt.subplots(figsize=(9.6, 5.45))
    score_min = float(np.nanmin(scores))
    score_max = float(np.nanmax(scores))
    bins = np.linspace(score_min, score_max, 120)
    ax.hist(scores[benign], bins=bins, color="#94a3b8", alpha=0.58, label=f"Benign ({benign.sum():,})")
    ax.hist(scores[malicious], bins=bins, color="#f97316", alpha=0.92, label=f"Malicious ({malicious.sum():,})")
    if not math.isnan(threshold_info["threshold"]):
        ax.axvline(threshold_info["threshold"], color="#111827", lw=2, ls="--", label=f"VELOX threshold = {threshold_info['threshold']:.6f}")
    if all_attacks:
        ax.axvline(all_attacks["score"], color="#059669", lw=2, ls=":", label=f"Coverage score = {all_attacks['score']:.6f}")
    ax.set_yscale("log")
    ax.set_xlabel("VELOX node anomaly score")
    ax.set_ylabel("Node count (log scale)")
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    all_attack_text = ""
    if all_attacks:
        all_attack_text = f" | Coverage rank={all_attacks['rank']} cutoff TP={all_attacks['score_cutoff_tp']} FP={all_attacks['score_cutoff_fp']} FN={all_attacks['score_cutoff_fn']}"
    ax.text(
        0.0,
        1.045,
        f"{dataset}: threshold TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}" + all_attack_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1"),
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    save(fig, obs_dir, "score_distribution")


def plot_metric_curves(obs_dir, scores, labels, nodes, node2attacks):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    x, y = downsample(fpr, tpr)
    axes[0].plot(x, y, color="#2563eb", lw=2, label=f"AUC = {roc_auc:.5f}")
    axes[0].plot([0, 1], [0, 1], color="#94a3b8", lw=1.2, ls="--", label="Random")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(color="#e2e8f0", lw=0.8)
    axes[0].legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    x, y = downsample(recall, precision)
    axes[1].plot(x, y, color="#e11d48", lw=2, label=f"AP = {ap:.5f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(color="#e2e8f0", lw=0.8)
    axes[1].legend(loc="upper right")

    adp_x, adp_y, adp = adp_curve(scores, labels, nodes, node2attacks)
    x, y = downsample(adp_x, adp_y)
    axes[2].plot(x, y, color="#059669", lw=2, label=f"ADP = {adp:.5f}")
    axes[2].fill_between(x, y, color="#059669", alpha=0.16)
    axes[2].set_xlabel("Precision")
    axes[2].set_ylabel("Detected attacks (%)")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 100.5)
    axes[2].grid(color="#e2e8f0", lw=0.8)
    axes[2].legend(loc="lower right")
    save(fig, obs_dir, "metric_curves")
    return roc_auc, ap, adp


def plot_ranked(obs_dir, scores, labels, nodes, node2attacks, threshold_info, all_attacks, order):
    sorted_scores = scores[order]
    sorted_nodes = nodes[order]
    ranks = np.arange(1, len(sorted_scores) + 1)
    fig, ax = plt.subplots(figsize=(9.8, 5.45))
    ax.plot(ranks, sorted_scores, color="#94a3b8", lw=1.2, label="All test nodes")
    if not math.isnan(threshold_info["threshold"]):
        ax.axhline(threshold_info["threshold"], color="#111827", lw=1.6, ls="--", label="VELOX threshold")
    if all_attacks:
        ax.axhline(all_attacks["score"], color="#059669", lw=1.8, ls=":", label="Coverage score")
        ax.axvline(all_attacks["rank"], color="#059669", lw=1.8, ls=":", label="Coverage rank")

    attack_ids = sorted(set(a for attacks in node2attacks.values() for a in attacks))
    for attack in attack_ids:
        idx = np.asarray([i for i, node in enumerate(sorted_nodes) if attack in node2attacks.get(int(node), set())])
        ax.scatter(ranks[idx], sorted_scores[idx], s=38, color=attack_color(attack), edgecolor="white", linewidth=0.4, zorder=5, label=f"attack {attack}")
        if len(idx):
            first = ranks[idx[0]]
            ax.axvline(first, color=attack_color(attack), lw=1.0, alpha=0.7)

    if all_attacks:
        ax.annotate(
            f"All attack types covered\nCoverage rank = {all_attacks['rank']}\nCoverage score = {all_attacks['score']:.3f}",
            xy=(all_attacks["rank"], all_attacks["score"]),
            xytext=(max(20, all_attacks["rank"] * 3), np.nanmax(sorted_scores) * 0.84),
            arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.1),
            fontsize=10,
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1"),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Rank after sorting by score (log scale)")
    ax.set_ylabel("VELOX node anomaly score")
    ax.set_xlim(1, len(sorted_scores))
    ax.set_ylim(0, np.nanmax(sorted_scores) * 1.04)
    ax.grid(color="#e2e8f0", lw=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    save(fig, obs_dir, "ranked_scores")


def plot_all_attack_nodes(obs_dir, scores, nodes, node2attacks, threshold_info, all_attacks, order):
    attack_ids = sorted(set(a for attacks in node2attacks.values() for a in attacks))
    if not attack_ids:
        return
    rank_by_node = {int(node): rank for rank, node in enumerate(nodes[order], start=1)}
    score_by_node = {int(node): float(score) for node, score in zip(nodes, scores)}
    fig, axes = plt.subplots(len(attack_ids), 1, figsize=(10.4, max(4.2, 3.1 * len(attack_ids))), sharex=True, sharey=True)
    if len(attack_ids) == 1:
        axes = [axes]
    for ax, attack in zip(axes, attack_ids):
        attack_nodes = sorted([int(node) for node, attacks in node2attacks.items() if attack in attacks and int(node) in rank_by_node], key=lambda n: rank_by_node[n])
        attack_ranks = np.asarray([rank_by_node[node] for node in attack_nodes])
        attack_scores = np.asarray([score_by_node[node] for node in attack_nodes])
        ax.scatter(attack_ranks, attack_scores, s=42, color=attack_color(attack), edgecolor="white", linewidth=0.45, zorder=4, label=f"attack {attack} ({len(attack_nodes)} labeled nodes)")
        if not math.isnan(threshold_info["threshold"]):
            ax.axhline(threshold_info["threshold"], color="#111827", lw=1.3, ls="--", label="VELOX threshold")
        if all_attacks:
            ax.axhline(all_attacks["score"], color="#059669", lw=1.4, ls=":", label="Coverage score")
            ax.axvline(all_attacks["rank"], color="#059669", lw=1.2, ls=":", label="Coverage rank")
        ax.text(0.012, 0.9, f"attack {attack}\n{len(attack_nodes)} labeled nodes", transform=ax.transAxes, ha="left", va="top", fontsize=10, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1"))
        ax.set_xscale("log")
        ax.set_ylabel("Score")
        ax.grid(color="#e2e8f0", lw=0.8)
    axes[-1].set_xlabel("Rank in full test-node list (log scale)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    save(fig, obs_dir, "all_attack_nodes_ranked")


def wrap(text, width=56):
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def plot_top_nodes(obs_dir, scores, labels, nodes, node2attacks, order, metadata, top_n=20):
    top_idx = order[:top_n]
    top_nodes = nodes[top_idx]
    top_scores = scores[top_idx]
    y = np.arange(top_n)[::-1]
    colors = []
    labels_out = []
    for rank, node, label in zip(range(1, top_n + 1), top_nodes, labels[top_idx]):
        attacks = sorted(node2attacks.get(int(node), set()))
        if label and attacks:
            colors.append(attack_color(attacks[0]))
            tag = "attacks " + ",".join(map(str, attacks))
        elif label:
            colors.append("#f97316")
            tag = "malicious"
        else:
            colors.append("#64748b")
            tag = "benign"
        labels_out.append(wrap(f"#{rank} | {tag} | {node_label(int(node), metadata)}", 58))
    fig, ax = plt.subplots(figsize=(11.4, 8.0))
    ax.barh(y, top_scores, color=colors, edgecolor="white", linewidth=1.0)
    for rank, score in zip(range(1, top_n + 1), top_scores):
        ax.text(score + max(top_scores) * 0.006, top_n - rank, f"{score:.3f}", va="center", fontsize=8.5, color="#334155")
    ax.set_yticks(y)
    ax.set_yticklabels(labels_out, fontsize=8.4)
    ax.set_xlabel("VELOX node anomaly score")
    ax.set_xlim(0, max(top_scores) * 1.10)
    ax.grid(axis="x", color="#e2e8f0", lw=0.8)
    legend = [Line2D([0], [0], marker="s", color="w", label="benign", markerfacecolor="#64748b", markersize=9)]
    for attack in sorted(set(a for attacks in node2attacks.values() for a in attacks)):
        legend.append(Line2D([0], [0], marker="s", color="w", label=f"attack {attack}", markerfacecolor=attack_color(attack), markersize=9))
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    save(fig, obs_dir, "top_ranked_nodes")


def export_attack_nodes_csv(obs_dir, scores, labels, nodes, node2attacks, threshold_info, all_attacks, order, metadata):
    tables_dir = obs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    rank_by_node = {int(node): rank for rank, node in enumerate(nodes[order], start=1)}
    score_by_node = {int(node): float(score) for node, score in zip(nodes, scores)}
    label_by_node = {int(node): int(label) for node, label in zip(nodes, labels)}
    out = tables_dir / "all_attack_nodes_ranked.csv"
    fieldnames = ["node_id", "attack_id", "rank", "score", "label", "pred_velox_threshold", "in_all_attacks_rank_prefix", "in_all_attacks_score_cutoff", "node_type", "path", "cmd", "src_addr", "src_port", "dst_addr", "dst_port"]
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        scored_attack_items = [(node, attacks) for node, attacks in node2attacks.items() if int(node) in rank_by_node]
        for node, attacks in sorted(scored_attack_items, key=lambda item: rank_by_node[int(item[0])]):
            for attack in sorted(attacks):
                score = score_by_node[int(node)]
                meta = metadata.get(int(node), {})
                writer.writerow([
                    int(node),
                    attack,
                    rank_by_node[int(node)],
                    f"{score:.9f}",
                    label_by_node[int(node)],
                    int(score > threshold_info["threshold"]) if not math.isnan(threshold_info["threshold"]) else "",
                    int(all_attacks is not None and rank_by_node[int(node)] <= all_attacks["rank"]),
                    int(all_attacks is not None and score >= all_attacks["score"]),
                    meta.get("node_type", ""),
                    meta.get("path", ""),
                    meta.get("cmd", ""),
                    meta.get("src_addr", ""),
                    meta.get("src_port", ""),
                    meta.get("dst_addr", ""),
                    meta.get("dst_port", ""),
                ])
    print(out)


def write_summary_metrics(obs_dir, dataset, run, metrics, threshold_info, all_attacks, curves, labels, yhat):
    obs_dir.mkdir(parents=True, exist_ok=True)
    benign_count = int((labels == 0).sum())
    malicious_count = int((labels == 1).sum())
    predicted_positive = int(yhat.sum())
    all_attack_lines = "N/A"
    if all_attacks:
        all_attack_lines = (
            f"coverage rank {all_attacks['rank']}; "
            f"coverage score {all_attacks['score']:.9f}; "
            f"prefix TP/FP {all_attacks['rank_prefix_tp']} / {all_attacks['rank_prefix_fp']}; "
            f"tie-inclusive TP/FP/FN {all_attacks['score_cutoff_tp']} / {all_attacks['score_cutoff_fp']} / {all_attacks['score_cutoff_fn']}"
        )
    tables_dir = obs_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "summary_metrics.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        rows = [
            ("dataset", dataset),
            ("mode", "--tuned --from_weights --restart_from_scratch"),
            ("total_scored_nodes", len(labels)),
            ("benign_nodes", benign_count),
            ("malicious_nodes", malicious_count),
            ("predicted_positives", predicted_positive),
            ("velox_threshold", f"{threshold_info['threshold']:.9f}"),
            ("threshold_tp", metrics["tp"]),
            ("threshold_fp", metrics["fp"]),
            ("threshold_fn", metrics["fn"]),
            ("precision", f"{metrics['precision']:.6f}"),
            ("recall", f"{metrics['recall']:.6f}"),
            ("f1", f"{metrics['f1']:.6f}"),
            ("auc", f"{curves['auc']:.5f}"),
            ("ap", f"{curves['ap']:.5f}"),
            ("adp", f"{curves['adp']:.5f}"),
            ("paper_reference_best_adp", run.get("paper_best", "")),
            ("coverage", all_attack_lines),
        ]
        if run.get("caveat"):
            rows.append(("caveat", run["caveat"]))
        for metric, value in rows:
            writer.writerow({"metric": metric, "value": value})
    print(out)


def generate(dataset):
    run = RUNS[dataset]
    input_dir = dataset_inputs(run["slug"])
    obs_dir = OBS_ROOT / run["obs_id"]
    (obs_dir / "plots").mkdir(parents=True, exist_ok=True)
    (obs_dir / "tables").mkdir(parents=True, exist_ok=True)
    setup_style()
    _score_path, _result_path, _stats_path, scores, labels, nodes, node2attacks, yhat, _stats = load_run(run)
    threshold_info = find_threshold(dataset, scores, yhat, input_dir)
    metrics = confusion(labels, yhat)
    order = ranked_order(scores, nodes)
    all_attacks = find_all_attacks_cutoff(scores, labels, nodes, node2attacks, order)
    pidmaker_all_attacks = find_pidmaker_all_attacks_cutoff(scores, nodes, node2attacks)
    if all_attacks and pidmaker_all_attacks:
        all_attacks.update(pidmaker_all_attacks)
    metadata = load_metadata(obs_dir)
    export_attack_nodes_csv(obs_dir, scores, labels, nodes, node2attacks, threshold_info, all_attacks, order, metadata)
    plot_distribution(obs_dir, dataset, scores, labels, threshold_info, metrics, all_attacks)
    auc_value, ap_value, adp_value = plot_metric_curves(obs_dir, scores, labels, nodes, node2attacks)
    plot_ranked(obs_dir, scores, labels, nodes, node2attacks, threshold_info, all_attacks, order)
    plot_all_attack_nodes(obs_dir, scores, nodes, node2attacks, threshold_info, all_attacks, order)
    plot_top_nodes(obs_dir, scores, labels, nodes, node2attacks, order, metadata)
    write_summary_metrics(
        obs_dir,
        dataset,
        run,
        metrics,
        threshold_info,
        all_attacks,
        {"auc": auc_value, "ap": ap_value, "adp": adp_value},
        labels,
        yhat,
    )
    print(obs_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(RUNS))
    args = parser.parse_args()
    generate(args.dataset)


if __name__ == "__main__":
    main()
