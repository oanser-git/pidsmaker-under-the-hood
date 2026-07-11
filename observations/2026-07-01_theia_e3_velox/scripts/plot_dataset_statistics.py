import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


OBS_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = OBS_DIR / "plots" / "from_dataset_counts"


RAW_COUNTS = {
    "Events": 44_366_117,
    "File nodes": 793_899,
    "Subject nodes": 278_363,
    "Netflow nodes": 186_100,
}

GRAPH_COUNTS = {
    "Transformed graph nodes": 1_248_267,
    "Transformed graph edges": 11_651_551,
    "Scored test nodes": 700_902,
    "GT malicious nodes": 118,
}

ATTACKS = [
    {
        "name": "Firefox backdoor\nDrakon in-memory",
        "day": 10,
        "window": "2018-04-10\n14:30-15:00",
        "gt_nodes": 58,
        "total_events": 223_029,
        "incident_gt_events": 80_997,
        "both_gt_events": 25_215,
    },
    {
        "name": "Browser extension\nDrakon dropper",
        "day": 12,
        "window": "2018-04-12\n12:40-13:30",
        "gt_nodes": 61,
        "total_events": 259_680,
        "incident_gt_events": 66_028,
        "both_gt_events": 843,
    },
]

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


def save(fig, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = OUT_DIR / f"theia_e3_produced_{stem}.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(out)
    plt.close(fig)


def human(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def plot_raw_counts(ax):
    labels = list(RAW_COUNTS)
    values = np.asarray(list(RAW_COUNTS.values()))
    colors = ["#0f172a", "#e11d48", "#2563eb", "#059669"]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Count (log scale)")
    ax.set_title("Raw THEIA_E3 database")
    ax.grid(axis="x", color="#e2e8f0", lw=0.8)
    for yi, value in zip(y, values):
        ax.text(value * 1.08, yi, human(int(value)), va="center", fontsize=9, color="#334155")


def plot_node_composition(ax):
    labels = ["File", "Subject/process", "Netflow"]
    values = np.asarray([RAW_COUNTS["File nodes"], RAW_COUNTS["Subject nodes"], RAW_COUNTS["Netflow nodes"]])
    colors = ["#e11d48", "#2563eb", "#059669"]
    total = values.sum()
    ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor="white"),
        autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=0.78,
    )
    ax.text(0, 0, f"{human(int(total))}\nnodes", ha="center", va="center", fontsize=11, color="#0f172a")
    ax.set_title("Node-type composition")
    ax.legend(
        [Patch(facecolor=c) for c in colors],
        [f"{label}: {human(int(value))}" for label, value in zip(labels, values)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=1,
        handlelength=1.0,
        handletextpad=0.5,
    )


def plot_attack_windows(ax):
    labels = [attack["name"] for attack in ATTACKS]
    x = np.arange(len(labels))
    width = 0.24
    series = [
        ("All events", [a["total_events"] for a in ATTACKS], "#94a3b8"),
        ("Incident to GT node", [a["incident_gt_events"] for a in ATTACKS], "#f97316"),
        ("Both endpoints GT", [a["both_gt_events"] for a in ATTACKS], "#7c3aed"),
    ]
    for offset, (name, values, color) in zip((-width, 0, width), series):
        bars = ax.bar(x + offset, values, width, label=name, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.16, human(value), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Events in attack window (log scale)")
    ax.set_title("Attack windows mixed with background activity")
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)


def plot_attack_labels(ax):
    labels = [attack["name"] for attack in ATTACKS]
    values = [attack["gt_nodes"] for attack in ATTACKS]
    bars = ax.bar(labels, values, color=["#2563eb", "#e11d48"], width=0.55)
    ax.axhline(118, color="#0f172a", lw=1.4, ls="--", label="Unique GT nodes: 118")
    ax.text(1.02, 118, "118 unique\n1 overlap", va="center", ha="left", fontsize=9, color="#0f172a")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, str(value), ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 132)
    ax.set_ylabel("Ground-truth labeled nodes")
    ax.set_title("Configured attack labels")
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    ax.legend(loc="upper left")


def plot_graph_counts(ax):
    labels = list(GRAPH_COUNTS)
    values = np.asarray(list(GRAPH_COUNTS.values()))
    colors = ["#2563eb", "#7c3aed", "#059669", "#dc2626"]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Count (log scale)")
    ax.set_title("Graph and evaluation scale")
    ax.grid(axis="x", color="#e2e8f0", lw=0.8)
    for yi, value in zip(y, values):
        ax.text(value * 1.08, yi, human(int(value)), va="center", fontsize=9, color="#334155")


def main():
    setup_style()
    fig = plt.figure(figsize=(16.8, 8.8))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.32],
        width_ratios=[1.15, 1.0, 1.15],
        hspace=0.66,
        wspace=0.42,
    )

    plot_raw_counts(fig.add_subplot(gs[0, 0]))
    plot_node_composition(fig.add_subplot(gs[0, 1]))
    plot_attack_labels(fig.add_subplot(gs[0, 2]))
    plot_attack_windows(fig.add_subplot(gs[1, 0:2]))
    plot_graph_counts(fig.add_subplot(gs[1, 2]))

    save(fig, "dataset_statistics")


if __name__ == "__main__":
    main()
