import argparse
import csv
import heapq
import math
import os
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback for older Python images
    ZoneInfo = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from poc_paths import artifacts_root


OBS_ID = "2026-07-01_theia_e3_velox"
DATASET = "THEIA_E3"
GNN_RUN_ID = "b697daa7af550a0e650e8603203036a4ea5450d572f20f2c6ba0aafde1c8b5fc"
EVAL_RUN_ID = "720c98a89d0bb855ac3dd7ae738b360441ca346e5ac7fc97680d12d2e1884108"
ATTACK_NAMES = {
    0: "Browser extension Drakon dropper",
    1: "Firefox backdoor Drakon in-memory",
}
EASTERN = ZoneInfo("US/Eastern") if ZoneInfo is not None else timezone.utc


ROOT = artifacts_root()
OBS_ROOT = ROOT / "observations"
OBS_DIR = OBS_ROOT / OBS_ID
SCORE_PATH = ROOT / f"detection/evaluation/{EVAL_RUN_ID}/{DATASET}/precision_recall_dir/scores_model_epoch_0.pkl"
RESULT_PATH = ROOT / f"detection/evaluation/{EVAL_RUN_ID}/{DATASET}/results/results.pth"
EDGE_ROOT = ROOT / f"detection/gnn_training/{GNN_RUN_ID}/{DATASET}/edge_losses"
VAL_DIR = EDGE_ROOT / "val/model_epoch_0"
TEST_DIR = EDGE_ROOT / "test/model_epoch_0"


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


def rel(path):
    return str(path).replace(str(ROOT) + "/", "artifacts/")


def ns_to_eastern(ns):
    try:
        return datetime.fromtimestamp(int(ns) / 1_000_000_000, tz=timezone.utc).astimezone(EASTERN).strftime("%Y-%m-%d %H:%M:%S.%f %Z")
    except Exception:
        return ""


def fmt_float(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{float(value):.{digits}f}"


def wrap(text, width=68):
    if not text:
        return ""
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def ranked_order(scores, nodes):
    # Match PIDSMaker's all-attacks ranking tie-break: sorted(zip(scores, nodes), reverse=True).
    return np.asarray(sorted(range(len(scores)), key=lambda i: (float(scores[i]), int(nodes[i])), reverse=True), dtype=int)


def save_plot(fig, stem):
    plot_dir = OBS_DIR / "plots" / "from_saved_edge_losses_and_db"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"{DATASET.lower()}_produced_{stem}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)
    plt.close(fig)


def load_scores():
    score_blob = torch.load(SCORE_PATH)
    scores = np.asarray(score_blob["pred_scores"], dtype=float)
    labels = np.asarray(score_blob["y_truth"], dtype=int)
    nodes = np.asarray(score_blob["nodes"], dtype=int)
    node2attacks = {int(k): set(v) for k, v in score_blob["node2attacks"].items()}
    return scores, labels, nodes, node2attacks


def quantile_row(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"count": 0, "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan"), "p999": float("nan"), "max": float("nan")}
    return {
        "count": int(len(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "p999": float(np.quantile(values, 0.999)),
        "max": float(np.max(values)),
    }


def percentile(sorted_values, value):
    if len(sorted_values) == 0:
        return float("nan")
    return float(np.searchsorted(sorted_values, value, side="right") / len(sorted_values) * 100.0)


def find_coverage(scores, labels, nodes, node2attacks, order):
    all_attacks = set()
    for attacks in node2attacks.values():
        all_attacks.update(attacks)
    detected = set()
    for rank, idx in enumerate(order, start=1):
        node = int(nodes[idx])
        attacks = set(node2attacks.get(node, set()))
        if attacks:
            detected.update(attacks)
            if detected == all_attacks:
                return {
                    "rank": rank,
                    "node": node,
                    "score": float(scores[idx]),
                    "label": int(labels[idx]),
                    "attacks": sorted(attacks),
                    "all_attacks": sorted(all_attacks),
                }
    return None


def push_heap(heap, limit, item, key):
    push_heap.counter += 1
    entry = (key, push_heap.counter, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
        return
    if key > heap[0][0]:
        heapq.heapreplace(heap, entry)


push_heap.counter = 0


def iter_edge_csvs(split_dir):
    for path in sorted(split_dir.glob("*.csv")):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield path, {
                    "loss": float(row["loss"]),
                    "srcnode": int(row["srcnode"]),
                    "dstnode": int(row["dstnode"]),
                    "time": int(row["time"]),
                }


def scan_edges(target_nodes, edges_per_node, global_top_n):
    val_losses = []
    test_losses = []
    target_heaps = {node: [] for node in target_nodes}
    target_counts = defaultdict(int)
    global_heap = []

    split_counts = {}
    split_files = {}
    for split, split_dir, sink in (("val", VAL_DIR, val_losses), ("test", TEST_DIR, test_losses)):
        file_count = 0
        row_count = 0
        for path, row in iter_edge_csvs(split_dir):
            file_count += int(row_count == 0 or path.name != getattr(scan_edges, "_last_file", None))
            scan_edges._last_file = path.name
            row_count += 1
            loss = row["loss"]
            sink.append(loss)
            if split != "test":
                continue
            row_with_source = dict(row)
            row_with_source["window_file"] = path.name
            row_with_source["window"] = path.name[:-4]
            push_heap(global_heap, global_top_n, row_with_source, loss)
            src = row["srcnode"]
            dst = row["dstnode"]
            if src in target_nodes:
                target_counts[src] += 1
                item = dict(row_with_source)
                item["target_node"] = src
                item["target_role"] = "src" if dst != src else "self"
                push_heap(target_heaps[src], edges_per_node, item, loss)
            if dst in target_nodes and dst != src:
                target_counts[dst] += 1
                item = dict(row_with_source)
                item["target_node"] = dst
                item["target_role"] = "dst"
                push_heap(target_heaps[dst], edges_per_node, item, loss)
        split_counts[split] = row_count
        split_files[split] = len(list(split_dir.glob("*.csv")))

    target_edges = {}
    for node, heap in target_heaps.items():
        target_edges[node] = [entry[2] for entry in sorted(heap, key=lambda x: x[0], reverse=True)]
    global_edges = [entry[2] for entry in sorted(global_heap, key=lambda x: x[0], reverse=True)]
    return {
        "val_losses": np.asarray(val_losses, dtype=float),
        "test_losses": np.asarray(test_losses, dtype=float),
        "target_edges": target_edges,
        "target_counts": dict(target_counts),
        "global_edges": global_edges,
        "split_counts": split_counts,
        "split_files": split_files,
    }


def db_connection():
    try:
        import psycopg2
    except Exception as exc:
        return None, f"psycopg2 unavailable: {exc}"
    try:
        conn = psycopg2.connect(
            database=os.environ.get("DB_NAME", "theia_e3"),
            host=os.environ.get("DB_HOST", "postgres"),
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", "postgres"),
            port=os.environ.get("DB_PORT", os.environ.get("DOCKER_PORT", "5432")),
        )
        return conn, "connected"
    except Exception as exc:
        return None, f"DB unavailable: {exc}"


def fetch_node_metadata(conn, node_ids):
    metadata = {}
    if conn is None or not node_ids:
        return metadata
    node_ids = sorted({int(n) for n in node_ids})
    with conn.cursor() as cur:
        try:
            cur.execute("SELECT index_id, path FROM file_node_table WHERE index_id = ANY(%s);", (node_ids,))
            for node_id, path in cur.fetchall():
                metadata[int(node_id)] = {"node_type": "file", "path": str(path or "")}
        except Exception:
            conn.rollback()
        try:
            cur.execute("SELECT index_id, path, cmd FROM subject_node_table WHERE index_id = ANY(%s);", (node_ids,))
            for node_id, path, cmd in cur.fetchall():
                metadata[int(node_id)] = {"node_type": "subject", "path": str(path or ""), "cmd": str(cmd or "")}
        except Exception:
            conn.rollback()
        try:
            cur.execute("SELECT index_id, src_addr, src_port, dst_addr, dst_port FROM netflow_node_table WHERE index_id = ANY(%s);", (node_ids,))
            for node_id, src_addr, src_port, dst_addr, dst_port in cur.fetchall():
                metadata[int(node_id)] = {
                    "node_type": "netflow",
                    "src_addr": str(src_addr or ""),
                    "src_port": str(src_port or ""),
                    "dst_addr": str(dst_addr or ""),
                    "dst_port": str(dst_port or ""),
                }
        except Exception:
            conn.rollback()
    return metadata


def fetch_event_metadata(conn, edges):
    event_meta = {}
    if conn is None or not edges:
        return event_meta
    keys = []
    for edge in edges:
        key = (int(edge["srcnode"]), int(edge["dstnode"]), int(edge["time"]))
        if key not in keys:
            keys.append(key)
    with conn.cursor() as cur:
        for src, dst, ts in keys:
            try:
                cur.execute(
                    """
                    SELECT operation, event_uuid
                    FROM event_table
                    WHERE src_index_id::text = %s AND dst_index_id::text = %s AND timestamp_rec::text = %s
                    ORDER BY event_uuid
                    LIMIT 1;
                    """,
                    (str(src), str(dst), str(ts)),
                )
                row = cur.fetchone()
                if row is not None:
                    event_meta[(src, dst, ts)] = {"operation": str(row[0] or ""), "event_uuid": str(row[1] or "")}
            except Exception:
                conn.rollback()
    return event_meta


def node_label(node_id, metadata):
    meta = metadata.get(int(node_id), {})
    node_type = meta.get("node_type", "")
    if node_type == "netflow":
        return f"netflow {meta.get('src_addr', '')}:{meta.get('src_port', '')}->{meta.get('dst_addr', '')}:{meta.get('dst_port', '')}"
    path = meta.get("path", "")
    cmd = meta.get("cmd", "")
    if path and cmd and cmd != path:
        return f"{node_type}: {path} | {cmd}"
    if path:
        return f"{node_type}: {path}"
    if node_type:
        return f"{node_type}: {node_id}"
    return f"node {node_id}"


def attack_label(attacks):
    if not attacks:
        return ""
    return "; ".join(ATTACK_NAMES.get(int(a), f"attack {a}") for a in sorted(attacks))


def enrich_edge(edge, metadata, event_meta, val_sorted, test_sorted, threshold):
    out = dict(edge)
    key = (int(edge["srcnode"]), int(edge["dstnode"]), int(edge["time"]))
    out.update(event_meta.get(key, {}))
    out["src_label"] = node_label(edge["srcnode"], metadata)
    out["dst_label"] = node_label(edge["dstnode"], metadata)
    out["src_type"] = metadata.get(int(edge["srcnode"]), {}).get("node_type", "")
    out["dst_type"] = metadata.get(int(edge["dstnode"]), {}).get("node_type", "")
    out["time_eastern"] = ns_to_eastern(edge["time"])
    out["above_val_threshold"] = int(float(edge["loss"]) > threshold)
    out["val_loss_percentile"] = percentile(val_sorted, float(edge["loss"]))
    out["test_loss_percentile"] = percentile(test_sorted, float(edge["loss"]))
    return out


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(path)


def plot_responsible_edges(target_rows, threshold):
    rows = target_rows[:20]
    labels = []
    scores = []
    losses = []
    colors = []
    for row in rows:
        labels.append(wrap(f"#{row['rank']} node {row['node_id']} | {row['node_label']}", 58))
        scores.append(float(row["node_score"]))
        losses.append(float(row["top_edge_loss"]) if row["top_edge_loss"] != "" else 0.0)
        colors.append("#e11d48" if int(row["label"]) else "#64748b")
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(11.6, max(6.8, 0.45 * len(rows) + 1.8)))
    ax.barh(y + 0.17, scores, height=0.30, color="#93c5fd", label="Node score")
    ax.barh(y - 0.17, losses, height=0.30, color=colors, label="Max incident edge loss")
    ax.axvline(threshold, color="#111827", lw=1.8, ls="--", label="VELOX threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Loss / score")
    ax.grid(axis="x", color="#e2e8f0", lw=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    save_plot(fig, "top_node_responsible_edges")


def plot_loss_distributions(val_losses, test_losses, threshold):
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    max_loss = float(max(np.max(val_losses), np.max(test_losses)))
    bins = np.linspace(0.0, max_loss, 150)
    ax.hist(val_losses, bins=bins, color="#2563eb", alpha=0.58, label=f"Validation edges ({len(val_losses):,})")
    ax.hist(test_losses, bins=bins, color="#f97316", alpha=0.46, label=f"Test edges ({len(test_losses):,})")
    ax.axvline(threshold, color="#111827", lw=2.0, ls="--", label=f"VELOX threshold = {threshold:.6f}")
    ax.set_yscale("log")
    ax.set_xlabel("Per-edge cross-entropy loss")
    ax.set_ylabel("Edge count (log scale)")
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    save_plot(fig, "edge_loss_distribution_validation_vs_test")


def write_loss_summary(val_losses, test_losses):
    rows = []
    for split, values in (("val", val_losses), ("test", test_losses)):
        stats = quantile_row(values)
        rows.append({"split": split, **{k: fmt_float(v, 9) if k != "count" else v for k, v in stats.items()}})
    write_csv(OBS_DIR / "tables/loss_distribution_summary.csv", ["split", "count", "p50", "p90", "p95", "p99", "p999", "max"], rows)
    return {row["split"]: row for row in rows}


def build_target_rows(target_nodes, target_flags, rank_by_node, score_by_node, label_by_node, node2attacks, target_edges, target_counts, metadata, threshold, val_sorted, test_sorted, event_meta):
    rows = []
    for node in sorted(target_nodes, key=lambda n: rank_by_node.get(n, 10**18)):
        edges = target_edges.get(node, [])
        top_edge = enrich_edge(edges[0], metadata, event_meta, val_sorted, test_sorted, threshold) if edges else {}
        node_score = float(score_by_node[node])
        top_loss = float(top_edge["loss"]) if top_edge else float("nan")
        rows.append({
            "rank": rank_by_node.get(node, ""),
            "node_id": node,
            "target_reason": "; ".join(sorted(target_flags[node])),
            "node_score": fmt_float(node_score, 9),
            "label": int(label_by_node[node]),
            "attacks": attack_label(node2attacks.get(node, set())),
            "pred_velox_threshold": int(node_score > threshold),
            "incident_edges_scanned": target_counts.get(node, 0),
            "top_edge_loss": fmt_float(top_loss, 9),
            "score_minus_top_edge_loss": fmt_float(node_score - top_loss, 9) if top_edge else "",
            "top_edge_src": top_edge.get("srcnode", ""),
            "top_edge_dst": top_edge.get("dstnode", ""),
            "top_edge_role": top_edge.get("target_role", ""),
            "top_edge_operation": top_edge.get("operation", ""),
            "top_edge_time_eastern": top_edge.get("time_eastern", ""),
            "top_edge_window": top_edge.get("window", ""),
            "top_edge_val_loss_percentile": fmt_float(top_edge.get("val_loss_percentile", float("nan")), 6) if top_edge else "",
            "top_edge_test_loss_percentile": fmt_float(top_edge.get("test_loss_percentile", float("nan")), 6) if top_edge else "",
            "node_type": metadata.get(node, {}).get("node_type", ""),
            "node_label": node_label(node, metadata),
            "src_label": top_edge.get("src_label", ""),
            "dst_label": top_edge.get("dst_label", ""),
        })
    return rows


def build_responsible_edge_rows(target_nodes, rank_by_node, score_by_node, label_by_node, node2attacks, target_edges, metadata, threshold, val_sorted, test_sorted, event_meta):
    rows = []
    for node in sorted(target_nodes, key=lambda n: rank_by_node.get(n, 10**18)):
        for edge_rank, edge in enumerate(target_edges.get(node, []), start=1):
            enriched = enrich_edge(edge, metadata, event_meta, val_sorted, test_sorted, threshold)
            rows.append({
                "node_rank": rank_by_node.get(node, ""),
                "node_id": node,
                "edge_rank_for_node": edge_rank,
                "node_score": fmt_float(score_by_node[node], 9),
                "label": int(label_by_node[node]),
                "node_label_text": node_label(node, metadata),
                "node_attacks": attack_label(node2attacks.get(node, set())),
                "target_role": enriched.get("target_role", ""),
                "edge_loss": fmt_float(enriched["loss"], 9),
                "above_val_threshold": enriched["above_val_threshold"],
                "val_loss_percentile": fmt_float(enriched["val_loss_percentile"], 6),
                "test_loss_percentile": fmt_float(enriched["test_loss_percentile"], 6),
                "operation": enriched.get("operation", ""),
                "event_uuid": enriched.get("event_uuid", ""),
                "time": enriched["time"],
                "time_eastern": enriched["time_eastern"],
                "window": enriched["window"],
                "srcnode": enriched["srcnode"],
                "dstnode": enriched["dstnode"],
                "src_type": enriched["src_type"],
                "dst_type": enriched["dst_type"],
                "src_label": enriched["src_label"],
                "dst_label": enriched["dst_label"],
            })
    return rows


def write_docs(args, db_status, threshold, coverage, val_stats, test_stats, target_rows, global_rows, exact_matches):
    return
    coverage_text = "not found"
    if coverage:
        coverage_text = f"rank `{coverage['rank']}`, node `{coverage['node']}`, score `{coverage['score']:.9f}`"
    top_rows = target_rows[:15]
    top_table = "\n".join(
        f"| `{row['rank']}` | `{row['node_id']}` | `{row['node_score']}` | `{row['top_edge_loss']}` | `{row['top_edge_operation']}` | {row['attacks'] or 'benign'} | {row['node_label']} |"
        for row in top_rows
    )
    if not top_table:
        top_table = "| | | | | | | |"

    coverage_row = None
    if coverage:
        coverage_row = next((row for row in target_rows if int(row["node_id"]) == int(coverage["node"])), None)
    first_benign = next((row for row in target_rows if int(row["label"]) == 0), None)
    pattern_lines = []
    if target_rows:
        first = target_rows[0]
        pattern_lines.append(f"The highest-ranked node is explained by `{first['top_edge_operation']}` from `{first['src_label']}` to `{first['dst_label']}`.")
    if coverage_row:
        pattern_lines.append(f"The coverage node for the second attack is explained by `{coverage_row['top_edge_operation']}` from `{coverage_row['src_label']}` to `{coverage_row['dst_label']}`.")
    if first_benign:
        pattern_lines.append(f"The first high-scoring benign node is rank `{first_benign['rank']}` and is also driven by `{first_benign['top_edge_operation']}` from `{first_benign['src_label']}` to `{first_benign['dst_label']}`.")
    pattern_text = " ".join(pattern_lines)

    highest_global = global_rows[0] if global_rows else {}
    readme = f"""# THEIA_E3 VELOX Edge-Loss Explanations

## Question

Which incident edges actually give the top-ranked THEIA_E3 VELOX nodes their anomaly scores?

## Scope

| Field | Value |
|---|---|
| Dataset | `{DATASET}` |
| System | `velox` |
| Mode | saved `--tuned --from_weights --restart_from_scratch` artifacts |
| Score file | `{rel(SCORE_PATH)}` |
| Result file | `{rel(RESULT_PATH)}` |
| Test edge losses | `{rel(TEST_DIR)}` |
| Validation edge losses | `{rel(VAL_DIR)}` |
| DB enrichment | `{db_status}` |

## Main Finding

VELOX node scores are directly explainable as the maximum incident edge loss. For the `{len(target_rows)}` target nodes inspected here, `{exact_matches}` have a top incident edge loss matching the saved node score within `1e-6`.

The fixed VELOX threshold is the maximum validation edge loss: `{threshold:.9f}`. The coverage point from the score-ranking observation is {coverage_text}.

{pattern_text}

## Top Target Nodes

| Rank | Node | Node score | Top edge loss | Operation | Label | Node label |
|---:|---:|---:|---:|---|---|---|
{top_table}

## Edge-Loss Distribution

| Split | Edges | p50 | p90 | p95 | p99 | p99.9 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | `{int(val_stats['count']):,}` | `{float(val_stats['p50']):.6f}` | `{float(val_stats['p90']):.6f}` | `{float(val_stats['p95']):.6f}` | `{float(val_stats['p99']):.6f}` | `{float(val_stats['p999']):.6f}` | `{float(val_stats['max']):.6f}` |
| Test | `{int(test_stats['count']):,}` | `{float(test_stats['p50']):.6f}` | `{float(test_stats['p90']):.6f}` | `{float(test_stats['p95']):.6f}` | `{float(test_stats['p99']):.6f}` | `{float(test_stats['p999']):.6f}` | `{float(test_stats['max']):.6f}` |

Highest test edge seen in the scan: loss `{highest_global.get('edge_loss', '')}`, operation `{highest_global.get('operation', '')}`, `src={highest_global.get('srcnode', '')}`, `dst={highest_global.get('dstnode', '')}`.

## Plots

| Plot | PNG | SVG | Purpose |
|---|---|---|---|
| Top-node responsible edges | `plots/top_node_responsible_edges.png` | `plots/top_node_responsible_edges.svg` | Compares saved node scores with max incident edge losses for top/coverage nodes. |
| Validation vs test edge losses | `plots/edge_loss_distribution_validation_vs_test.png` | `plots/edge_loss_distribution_validation_vs_test.svg` | Shows why the validation-max threshold creates the automatic classifier boundary. |

## Data Tables

| File | Purpose |
|---|---|
| `tables/top_node_explanations.csv` | One row per inspected node, including rank, score, label, responsible max-loss edge, endpoint labels, operation, and percentile context. |
| `tables/responsible_edges.csv` | Top `{args.edges_per_node}` incident edges for each inspected node. |
| `tables/high_loss_global_edges.csv` | Top `{args.global_top_n}` test edges by loss, independent of target nodes. |
| `tables/loss_distribution_summary.csv` | Validation/test edge-loss quantiles. |

## Interpretation

This is a lightweight explanation method, not a separate explainer model. It follows VELOX's actual scoring rule: per-edge cross-entropy loss is computed for edge-type prediction, and each node receives the maximum loss among incident test edges.

The `operation` column is joined back from `event_table` using exact `(src_index_id, dst_index_id, timestamp_rec)`. Operation labels are useful context, but global operation rarity is not inferred here because PIDSMaker graph construction collapses consecutive same-operation events per `(src,dst)` pair before writing graph edges.

## Regeneration

Run inside the existing PIDSMaker container:

```shell
docker exec pidsmaker-pids python /home/artifacts/observations/shared_scripts/generate_velox_edge_loss_observation.py --top-n {args.top_n} --edges-per-node {args.edges_per_node} --global-top-n {args.global_top_n}
```
"""
    print(OBS_DIR)


def generate(args):
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    (OBS_DIR / "plots").mkdir(exist_ok=True)
    (OBS_DIR / "tables").mkdir(exist_ok=True)
    setup_style()

    scores, labels, nodes, node2attacks = load_scores()
    order = ranked_order(scores, nodes)
    rank_by_node = {int(node): rank for rank, node in enumerate(nodes[order], start=1)}
    score_by_node = {int(node): float(score) for node, score in zip(nodes, scores)}
    label_by_node = {int(node): int(label) for node, label in zip(nodes, labels)}
    coverage = find_coverage(scores, labels, nodes, node2attacks, order)

    target_flags = defaultdict(set)
    for idx in order[: args.top_n]:
        target_flags[int(nodes[idx])].add(f"top_{args.top_n}")
    if coverage:
        for idx in order[: coverage["rank"]]:
            target_flags[int(nodes[idx])].add("coverage_prefix")
        target_flags[int(coverage["node"])].add("coverage_node")
    for attack in sorted({a for attacks in node2attacks.values() for a in attacks}):
        attack_nodes = [int(n) for n, attacks in node2attacks.items() if attack in attacks and int(n) in rank_by_node]
        if attack_nodes:
            first = min(attack_nodes, key=lambda n: rank_by_node[n])
            target_flags[first].add(f"first_{ATTACK_NAMES.get(int(attack), f'attack_{attack}').lower().replace(' ', '_')}")
    target_nodes = set(target_flags)

    scan = scan_edges(target_nodes, args.edges_per_node, args.global_top_n)
    val_losses = scan["val_losses"]
    test_losses = scan["test_losses"]
    threshold = float(np.max(val_losses))
    val_sorted = np.sort(val_losses)
    test_sorted = np.sort(test_losses)

    all_edges_for_enrichment = []
    endpoint_nodes = set(target_nodes)
    for edges in scan["target_edges"].values():
        all_edges_for_enrichment.extend(edges)
    all_edges_for_enrichment.extend(scan["global_edges"])
    for edge in all_edges_for_enrichment:
        endpoint_nodes.add(int(edge["srcnode"]))
        endpoint_nodes.add(int(edge["dstnode"]))

    conn, db_status = db_connection()
    metadata = fetch_node_metadata(conn, endpoint_nodes)
    event_meta = fetch_event_metadata(conn, all_edges_for_enrichment)
    if conn is not None:
        conn.close()

    target_rows = build_target_rows(
        target_nodes,
        target_flags,
        rank_by_node,
        score_by_node,
        label_by_node,
        node2attacks,
        scan["target_edges"],
        scan["target_counts"],
        metadata,
        threshold,
        val_sorted,
        test_sorted,
        event_meta,
    )
    responsible_rows = build_responsible_edge_rows(
        target_nodes,
        rank_by_node,
        score_by_node,
        label_by_node,
        node2attacks,
        scan["target_edges"],
        metadata,
        threshold,
        val_sorted,
        test_sorted,
        event_meta,
    )
    global_rows = []
    for rank, edge in enumerate(scan["global_edges"], start=1):
        enriched = enrich_edge(edge, metadata, event_meta, val_sorted, test_sorted, threshold)
        global_rows.append({
            "edge_rank": rank,
            "edge_loss": fmt_float(enriched["loss"], 9),
            "above_val_threshold": enriched["above_val_threshold"],
            "val_loss_percentile": fmt_float(enriched["val_loss_percentile"], 6),
            "test_loss_percentile": fmt_float(enriched["test_loss_percentile"], 6),
            "operation": enriched.get("operation", ""),
            "event_uuid": enriched.get("event_uuid", ""),
            "time": enriched["time"],
            "time_eastern": enriched["time_eastern"],
            "window": enriched["window"],
            "srcnode": enriched["srcnode"],
            "dstnode": enriched["dstnode"],
            "src_type": enriched["src_type"],
            "dst_type": enriched["dst_type"],
            "src_label": enriched["src_label"],
            "dst_label": enriched["dst_label"],
        })

    target_fields = [
        "rank", "node_id", "target_reason", "node_score", "label", "attacks", "pred_velox_threshold",
        "incident_edges_scanned", "top_edge_loss", "score_minus_top_edge_loss", "top_edge_src", "top_edge_dst",
        "top_edge_role", "top_edge_operation", "top_edge_time_eastern", "top_edge_window",
        "top_edge_val_loss_percentile", "top_edge_test_loss_percentile", "node_type", "node_label", "src_label", "dst_label",
    ]
    edge_fields = [
        "node_rank", "node_id", "edge_rank_for_node", "node_score", "label", "node_label_text", "node_attacks", "target_role",
        "edge_loss", "above_val_threshold", "val_loss_percentile", "test_loss_percentile", "operation", "event_uuid",
        "time", "time_eastern", "window", "srcnode", "dstnode", "src_type", "dst_type", "src_label", "dst_label",
    ]
    global_fields = [
        "edge_rank", "edge_loss", "above_val_threshold", "val_loss_percentile", "test_loss_percentile", "operation",
            "event_uuid", "time", "time_eastern", "window", "srcnode", "dstnode", "src_type", "dst_type", "src_label", "dst_label",
    ]
    write_csv(OBS_DIR / "tables/top_node_explanations.csv", target_fields, target_rows)
    write_csv(OBS_DIR / "tables/responsible_edges.csv", edge_fields, responsible_rows)
    write_csv(OBS_DIR / "tables/high_loss_global_edges.csv", global_fields, global_rows)
    summary_rows = write_loss_summary(val_losses, test_losses)

    exact_matches = 0
    for row in target_rows:
        if row["top_edge_loss"] and abs(float(row["node_score"]) - float(row["top_edge_loss"])) <= 1e-6:
            exact_matches += 1
    plot_responsible_edges(target_rows, threshold)
    plot_loss_distributions(val_losses, test_losses, threshold)
    write_docs(
        args,
        db_status,
        threshold,
        coverage,
        {k: float(v) if k != "count" else int(v) for k, v in summary_rows["val"].items() if k != "split"},
        {k: float(v) if k != "count" else int(v) for k, v in summary_rows["test"].items() if k != "split"},
        target_rows,
        global_rows,
        exact_matches,
    )
    print(OBS_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--edges-per-node", type=int, default=5)
    parser.add_argument("--global-top-n", type=int, default=50)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
