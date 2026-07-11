import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D


OBS_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = OBS_DIR / "tables"
GRAPHS_DIR = OBS_DIR / "graphs"
PLOTS_DIR = OBS_DIR / "plots" / "from_ground_truth_db_exports"
SCORE_CSV = TABLES_DIR / "all_attack_nodes_ranked.csv"

ATTACKS = [
    {
        "slug": "browser_extension_drakon_dropper",
        "name": "Browser Extension Drakon Dropper",
        "window": "2018-04-12 12:40-13:30 local",
        "seed": 11,
    },
    {
        "slug": "firefox_backdoor_drakon_in_memory",
        "name": "Firefox Backdoor Drakon In-Memory",
        "window": "2018-04-10 14:30-15:00 local",
        "seed": 23,
    },
]

TYPE_COLORS = {
    "subject": "#2563eb",
    "file": "#e11d48",
    "netflow": "#059669",
    "unknown": "#64748b",
}

TYPE_MARKERS = {
    "subject": "o",
    "file": "s",
    "netflow": "^",
    "unknown": "o",
}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#0f172a",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "font.size": 10,
        "savefig.facecolor": "white",
        "legend.frameon": False,
    })


def save(fig, stem):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"theia_e3_produced_{stem}.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(out)
    plt.close(fig)


def load_scores():
    if not SCORE_CSV.exists():
        return {}
    scores = {}
    with open(SCORE_CSV, newline="") as f:
        for row in csv.DictReader(f):
            node = int(row["node_id"])
            current = scores.get(node)
            rank = int(row["rank"])
            if current is None or rank < current["rank"]:
                scores[node] = {
                    "rank": rank,
                    "score": float(row["score"]),
                    "pred_velox_threshold": int(row["pred_velox_threshold"]),
                }
    return scores


def load_metadata(slug):
    df = pd.read_csv(TABLES_DIR / f"{slug}_metadata.csv")
    metadata = {}
    for row in df.fillna("").to_dict(orient="records"):
        node_id = int(row["node_id"])
        metadata[node_id] = row
    return metadata


def short_name(meta):
    node_type = meta.get("node_type") or "unknown"
    if node_type == "netflow":
        src = f"{meta.get('src_addr', '')}:{meta.get('src_port', '')}"
        dst = f"{meta.get('dst_addr', '')}:{meta.get('dst_port', '')}"
        return f"netflow\n{src}\n-> {dst}"

    path = meta.get("path") or "unknown"
    base = path.rstrip("/").split("/")[-1] or path
    if len(base) > 24:
        base = base[:21] + "..."
    return f"{node_type}\n{base}"


def build_graph(slug, scores):
    edges = pd.read_csv(TABLES_DIR / f"{slug}_gt_gt_edges.csv")
    metadata = load_metadata(slug)
    gt_nodes = pd.read_csv(TABLES_DIR / f"{slug}_gt_nodes.csv")

    graph = nx.DiGraph()
    for node in sorted(gt_nodes["node_id"].astype(int).unique()):
        meta = metadata.get(node, {})
        score_info = scores.get(node, {})
        graph.add_node(
            node,
            node_type=meta.get("node_type", "unknown") or "unknown",
            label=short_name(meta),
            path=meta.get("path", ""),
            cmd=meta.get("cmd", ""),
            src_addr=meta.get("src_addr", ""),
            src_port=meta.get("src_port", ""),
            dst_addr=meta.get("dst_addr", ""),
            dst_port=meta.get("dst_port", ""),
            velox_score=score_info.get("score", float("nan")),
            velox_rank=score_info.get("rank", -1),
            velox_pred=score_info.get("pred_velox_threshold", -1),
        )

    for (src, dst), group in edges.groupby(["src", "dst"]):
        total = int(group["event_count"].sum())
        op_counts = group.groupby("operation")["event_count"].sum().sort_values(ascending=False)
        top_ops = "; ".join(f"{op}:{int(count)}" for op, count in op_counts.head(4).items())
        graph.add_edge(
            int(src),
            int(dst),
            event_count=total,
            operation_count=len(op_counts),
            top_operations=top_ops,
            first_ts_ns=int(group["first_ts_ns"].min()),
            last_ts_ns=int(group["last_ts_ns"].max()),
        )
    return graph, edges


def label_nodes(graph):
    labels = {}
    for node, data in graph.nodes(data=True):
        rank = data.get("velox_rank", -1)
        score = data.get("velox_score", float("nan"))
        degree = graph.degree(node, weight="event_count")
        should_label = (0 < rank <= 25) or degree >= 50 or data.get("node_type") == "netflow"
        if should_label:
            score_text = f"\nscore {score:.3f}" if not math.isnan(score) else ""
            rank_text = f"#{rank} " if rank > 0 else ""
            labels[node] = f"{rank_text}{node}\n{data['label']}{score_text}"
    return labels


def draw_graph(attack, graph, raw_edges):
    fig, ax = plt.subplots(figsize=(17.5, 12.0))
    ax.set_title(f"{attack['name']} GT-Induced Provenance Subgraph\n{attack['window']}", fontsize=15, pad=16)

    # Spring layout is readable enough for these 58-61 node attack-induced graphs.
    weights = {(u, v): max(1.0, math.log1p(data["event_count"])) for u, v, data in graph.edges(data=True)}
    nx.set_edge_attributes(graph, weights, "layout_weight")
    pos = nx.spring_layout(graph, seed=attack["seed"], k=0.85, iterations=300, weight="layout_weight")

    edge_counts = [data["event_count"] for _, _, data in graph.edges(data=True)]
    edge_widths = [0.45 + 2.4 * math.log1p(count) / math.log1p(max(edge_counts)) for count in edge_counts]
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=10,
        width=edge_widths,
        edge_color="#64748b",
        alpha=0.44,
        connectionstyle="arc3,rad=0.06",
    )

    for node_type, marker in TYPE_MARKERS.items():
        nodes = [node for node, data in graph.nodes(data=True) if data.get("node_type") == node_type]
        if not nodes:
            continue
        sizes = []
        for node in nodes:
            score = graph.nodes[node].get("velox_score", float("nan"))
            if math.isnan(score):
                sizes.append(320)
            else:
                sizes.append(260 + 70 * max(score, 0))
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_shape=marker,
            node_color=TYPE_COLORS[node_type],
            node_size=sizes,
            linewidths=1.1,
            edgecolors="white",
            alpha=0.92,
            ax=ax,
        )

    labels = label_nodes(graph)
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=7.6,
        font_color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cbd5e1", alpha=0.82),
        ax=ax,
    )

    top_edges = sorted(graph.edges(data=True), key=lambda item: item[2]["event_count"], reverse=True)[:8]
    edge_labels = {(u, v): f"{data['event_count']}" for u, v, data in top_edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, font_color="#334155", ax=ax)

    stats = pd.read_csv(TABLES_DIR / f"{attack['slug']}_window_stats.csv").iloc[0]
    summary = (
        f"GT nodes: {graph.number_of_nodes()}\n"
        f"GT-GT aggregated edges: {graph.number_of_edges()}\n"
        f"GT-GT raw events: {int(raw_edges['event_count'].sum()):,}\n"
        f"Window events: {int(stats['total_window_events']):,}\n"
        f"Events incident to GT: {int(stats['events_incident_to_gt']):,}"
    )
    ax.text(
        0.01,
        0.01,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#cbd5e1"),
    )

    legend_items = [
        Line2D([0], [0], marker=TYPE_MARKERS[t], color="w", label=t, markerfacecolor=c, markersize=10)
        for t, c in TYPE_COLORS.items()
        if any(data.get("node_type") == t for _, data in graph.nodes(data=True))
    ]
    legend_items.append(Line2D([0], [0], color="#64748b", lw=2, label="edge width = log event count"))
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    ax.set_axis_off()
    save(fig, f"{attack['slug']}_attack_graph")


def export_graphml(attack, graph):
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    out = GRAPHS_DIR / f"{attack['slug']}_attack_graph.graphml"
    # GraphML cannot serialize NaN cleanly across all tools.
    graph_copy = graph.copy()
    for _, data in graph_copy.nodes(data=True):
        if isinstance(data.get("velox_score"), float) and math.isnan(data["velox_score"]):
            data["velox_score"] = -1.0
    nx.write_graphml(graph_copy, out)
    print(out)


def export_summary(rows):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "attack_graph_summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(out)


def main():
    setup_style()
    scores = load_scores()
    summaries = []
    for attack in ATTACKS:
        graph, raw_edges = build_graph(attack["slug"], scores)
        draw_graph(attack, graph, raw_edges)
        export_graphml(attack, graph)
        stats = pd.read_csv(TABLES_DIR / f"{attack['slug']}_window_stats.csv").iloc[0]
        summaries.append({
            "attack_slug": attack["slug"],
            "attack_name": attack["name"],
            "gt_nodes": graph.number_of_nodes(),
            "gt_gt_aggregated_edges": graph.number_of_edges(),
            "gt_gt_raw_events": int(raw_edges["event_count"].sum()),
            "total_window_events": int(stats["total_window_events"]),
            "events_incident_to_gt": int(stats["events_incident_to_gt"]),
        })
    export_summary(summaries)


if __name__ == "__main__":
    main()
