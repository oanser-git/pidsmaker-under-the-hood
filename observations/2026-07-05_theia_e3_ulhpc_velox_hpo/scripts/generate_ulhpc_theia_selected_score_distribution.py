import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


OBS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = Path(os.environ.get("ULHPC_INPUTS_DIR") or os.environ.get("INPUTS_DIR") or REPO_ROOT / "inputs") / "ulhpc_theia_e3"
THRESHOLD_PATH = INPUT_ROOT / "thresholds.csv"
PLOT_ROOT = OBS_DIR / "plots"
TABLE_DIR = OBS_DIR / "tables"


RUNS = [
    {
        "model_label": "Leaky selected",
        "threshold_key": "leaky-selected",
        "plot_slug": "leaky_selected",
        "plot_group": "leaky",
        "exp": "velox_THEIA_E3_selected_w5_lr0001_h64_seed0",
        "selection_rule": "leaky ADP-selected configuration, seed 0",
        "score_epoch": 0,
        "score_path": INPUT_ROOT / "leaky_selected" / "scores_model_epoch_0.pkl",
        "state_path": INPUT_ROOT / "leaky_selected" / "state_dict.pkl",
    },
    {
        "model_label": "Non-leaky selected",
        "threshold_key": "no-leaky-selected",
        "plot_slug": "nonleaky_selected",
        "plot_group": "non-leaky",
        "exp": "velox_THEIA_E3_nonleaky_selected_eval_epoch3",
        "selection_rule": "validation-loss-selected configuration, seed 0",
        "score_epoch": 3,
        "score_path": INPUT_ROOT / "nonleaky_selected" / "scores_model_epoch_3.pkl",
        "state_path": INPUT_ROOT / "nonleaky_selected" / "state_dict.pkl",
    },
]


def setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "font.size": 10.5,
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_thresholds():
    thresholds = {}
    with THRESHOLD_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            for key in ("threshold", "max_val_loss", "mean_val_loss", "percentile_90_val_loss"):
                row[key] = float(row[key])
            for key in ("val_file_count", "val_row_count"):
                row[key] = int(row[key])
            thresholds[row["model"]] = row
    return thresholds


def confusion(labels, preds):
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, tn, precision, recall, f1


def prediction_boundary(scores, preds):
    below = scores[preds == 0]
    above = scores[preds == 1]
    if len(below) == 0 or len(above) == 0:
        return float("nan"), float("nan"), float("nan")
    lower = float(below.max())
    upper = float(above.min())
    return lower, upper, (lower + upper) / 2


def quantiles(values):
    return {
        "min": float(np.min(values)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q99": float(np.quantile(values, 0.99)),
        "q999": float(np.quantile(values, 0.999)),
        "max": float(np.max(values)),
    }


def load_run(run, thresholds):
    score_blob = torch_load(run["score_path"])
    scores = np.asarray(score_blob["pred_scores"], dtype=float)
    labels = np.asarray(score_blob["y_truth"], dtype=int)
    preds = np.asarray(score_blob["y_preds"], dtype=int)
    nodes = np.asarray(score_blob["nodes"], dtype=int)
    node2attacks = score_blob.get("node2attacks", {})
    return {
        **run,
        "threshold_info": thresholds[run["threshold_key"]],
        "scores": scores,
        "labels": labels,
        "preds": preds,
        "nodes": nodes,
        "node2attacks": node2attacks,
    }


def make_summary(loaded_runs):
    rows = []
    for run in loaded_runs:
        scores = run["scores"]
        labels = run["labels"]
        preds = run["preds"]
        tp, fp, fn, tn, precision, recall, f1 = confusion(labels, preds)
        lower, upper, midpoint = prediction_boundary(scores, preds)
        all_q = quantiles(scores)
        benign_q = quantiles(scores[labels == 0])
        malicious_q = quantiles(scores[labels == 1])
        rows.append(
            {
                "model_label": run["model_label"],
                "exp": run["exp"],
                "selection_rule": run["selection_rule"],
                "score_epoch": run["score_epoch"],
                "threshold_method": run["threshold_info"]["threshold_method"],
                "threshold": run["threshold_info"]["threshold"],
                "max_val_loss": run["threshold_info"]["max_val_loss"],
                "mean_val_loss": run["threshold_info"]["mean_val_loss"],
                "percentile_90_val_loss": run["threshold_info"]["percentile_90_val_loss"],
                "val_file_count": run["threshold_info"]["val_file_count"],
                "val_row_count": run["threshold_info"]["val_row_count"],
                "nodes": len(scores),
                "benign_nodes": int((labels == 0).sum()),
                "malicious_nodes": int((labels == 1).sum()),
                "attack_nodes": len(run["node2attacks"]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": float(roc_auc_score(labels, scores)),
                "average_precision": float(average_precision_score(labels, scores)),
                "alert_boundary_lower": lower,
                "alert_boundary_upper": upper,
                "alert_boundary_midpoint": midpoint,
                "score_min": all_q["min"],
                "score_q50": all_q["q50"],
                "score_q90": all_q["q90"],
                "score_q99": all_q["q99"],
                "score_q999": all_q["q999"],
                "score_max": all_q["max"],
                "benign_q99": benign_q["q99"],
                "benign_max": benign_q["max"],
                "malicious_q50": malicious_q["q50"],
                "malicious_q90": malicious_q["q90"],
                "malicious_max": malicious_q["max"],
                "score_path": run["score_path"],
                "state_path": run["state_path"],
                "val_edge_loss_dir": run["threshold_info"]["val_dir"],
                "test_edge_loss_dir": run["threshold_info"]["test_dir"],
            }
        )
    return rows


def write_summary(rows):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / "ulhpc_theia_e3_selected_score_distribution_summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


def plot_distributions(loaded_runs, summary_rows):
    colors = {0: "#94a3b8", 1: "#f97316"}
    names = {0: "Benign", 1: "Malicious"}

    for run, row in zip(loaded_runs, summary_rows):
        plot_dir = PLOT_ROOT / run["plot_group"] / "from_saved_node_scores"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        scores = run["scores"]
        labels = run["labels"]
        preds = run["preds"]
        bins = np.linspace(float(scores.min()), float(scores.max()), 120)
        for label in (0, 1):
            mask = labels == label
            ax.hist(
                scores[mask],
                bins=bins,
                color=colors[label],
                alpha=0.62 if label == 0 else 0.90,
                label=f"{names[label]} ({int(mask.sum()):,})",
            )

        threshold = float(row["threshold"])
        ax.axvline(threshold, color="#111827", lw=1.7, ls="--", label="Validation threshold")

        ax.set_yscale("log")
        ax.set_xlabel("VELOX node anomaly score")
        ax.set_ylabel("Node count (log scale)")
        ax.grid(axis="y", color="#e2e8f0", lw=0.8)
        ax.set_title(f"{run['model_label']} (score epoch {run['score_epoch']})", loc="left", fontsize=11.5)
        ax.text(
            0.02,
            0.94,
            f"TP={row['tp']}  FP={row['fp']}  FN={row['fn']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.2,
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#cbd5e1"),
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        out = plot_dir / f"theia_e3_ulhpc_{run['plot_slug']}_score_distribution.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print(out)
        plt.close(fig)


def main():
    setup_style()
    thresholds = load_thresholds()
    loaded_runs = [load_run(run, thresholds) for run in RUNS]
    rows = make_summary(loaded_runs)
    write_summary(rows)
    plot_distributions(loaded_runs, rows)


if __name__ == "__main__":
    main()
