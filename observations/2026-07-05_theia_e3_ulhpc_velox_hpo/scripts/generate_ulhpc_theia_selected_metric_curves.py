import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve


OBS_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = OBS_DIR / "tables"
PLOT_ROOT = OBS_DIR / "plots"
SELECTED_SUMMARY_PATH = TABLE_DIR / "ulhpc_theia_e3_selected_score_distribution_summary.csv"


def setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "font.size": 10.8,
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def slugify(text):
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug.replace("non_leaky", "nonleaky")


def plot_group(model_label):
    return "non-leaky" if "Non-leaky" in model_label else "leaky"


def downsample(x, y, max_points=5000):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) <= max_points:
        return x, y
    idx = np.unique(np.r_[0, np.linspace(0, len(x) - 1, max_points).astype(int), len(x) - 1])
    return x[idx], y[idx]


def adp_curve(scores, labels, nodes, node2attacks):
    sorted_indices = np.argsort(scores)[::-1]
    total_attacks = len(set(attack for attacks in node2attacks.values() for attack in attacks))
    if total_attacks == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0]), 0.0, 0

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
        detected_attacks.update(node2attacks.get(node, set()))
        precisions.append(tp / (tp + fp))
        detected_percentages.append((len(detected_attacks) / total_attacks) * 100)

    precision_to_attacks = {}
    for precision, detected_percentage in zip(precisions, detected_percentages):
        precision_to_attacks[precision] = max(precision_to_attacks.get(precision, 0.0), detected_percentage)

    x = np.asarray(sorted(precision_to_attacks), dtype=float)
    y = np.asarray([precision_to_attacks[p] for p in x], dtype=float)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return x, y, float(trapz(y, x) / 100), total_attacks


def load_selected_runs():
    with SELECTED_SUMMARY_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def load_scores(score_path):
    blob = torch_load(score_path)
    scores = np.asarray(blob["pred_scores"], dtype=float)
    labels = np.asarray(blob["y_truth"], dtype=int)
    nodes = np.asarray(blob["nodes"], dtype=int)
    node2attacks = {int(node): set(int(attack) for attack in attacks) for node, attacks in blob.get("node2attacks", {}).items()}
    return scores, labels, nodes, node2attacks


def plot_curves(run, scores, labels, nodes, node2attacks):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    pr_precision, pr_recall, _ = precision_recall_curve(labels, scores)
    ap = float(average_precision_score(labels, scores))
    adp_precision, detected_attacks, adp, total_attacks = adp_curve(scores, labels, nodes, node2attacks)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    x, y = downsample(fpr, tpr)
    axes[0].plot(x, y, color="#2563eb", lw=2.1, label=f"AUC = {roc_auc:.5f}")
    axes[0].plot([0, 1], [0, 1], color="#94a3b8", lw=1.3, ls="--", label="Random")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(color="#e2e8f0", lw=0.8)
    axes[0].legend(loc="lower right")

    x, y = downsample(pr_recall, pr_precision)
    axes[1].plot(x, y, color="#e11d48", lw=2.1, label=f"AP = {ap:.5f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(color="#e2e8f0", lw=0.8)
    axes[1].legend(loc="upper right")

    x, y = downsample(adp_precision, detected_attacks)
    axes[2].plot(x, y, color="#059669", lw=2.1, label=f"ADP = {adp:.5f}")
    axes[2].fill_between(x, y, color="#059669", alpha=0.16)
    axes[2].set_xlabel("Precision")
    axes[2].set_ylabel("Detected attacks (%)")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 100.5)
    axes[2].grid(color="#e2e8f0", lw=0.8)
    axes[2].legend(loc="lower right")

    fig.tight_layout()
    plot_dir = PLOT_ROOT / plot_group(run["model_label"]) / "from_saved_node_scores"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stem = f"theia_e3_ulhpc_{slugify(run['model_label'])}_metric_curves"
    out = plot_dir / f"{stem}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)
    plt.close(fig)

    return {
        "model_label": run["model_label"],
        "exp": run["exp"],
        "score_epoch": run["score_epoch"],
        "auc": f"{roc_auc:.10f}",
        "ap": f"{ap:.10f}",
        "adp": f"{adp:.10f}",
        "attack_ids": total_attacks,
        "nodes": len(scores),
        "malicious_nodes": int((labels == 1).sum()),
        "benign_nodes": int((labels == 0).sum()),
        "score_path": run["score_path"],
        "plot_png": str(out),
    }


def write_summary(rows):
    out = TABLE_DIR / "ulhpc_theia_e3_selected_metric_curves_summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


def main():
    setup_style()
    rows = []
    for run in load_selected_runs():
        scores, labels, nodes, node2attacks = load_scores(Path(run["score_path"]))
        rows.append(plot_curves(run, scores, labels, nodes, node2attacks))
    write_summary(rows)


if __name__ == "__main__":
    main()
