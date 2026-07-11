import csv
import math
import os
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


OBS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = Path(os.environ.get("ULHPC_INPUTS_DIR") or os.environ.get("INPUTS_DIR") or REPO_ROOT / "inputs") / "ulhpc_theia_e3"
THRESHOLD_PATH = INPUT_ROOT / "thresholds.csv"
PLOT_ROOT = OBS_DIR / "plots"
TABLE_DIR = OBS_DIR / "tables"
try:
    EASTERN = ZoneInfo("US/Eastern") if ZoneInfo is not None else timezone.utc
except Exception:
    EASTERN = timezone(timedelta(hours=-4), "EDT")

ATTACK_NAMES = {
    0: "Browser extension Drakon dropper",
    1: "Firefox backdoor Drakon in-memory",
}

NODE_GROUP_LABELS = {
    "TP": "TP",
    "FP": "FP",
    "FN": "FN",
    "TN_near_threshold": "TN (benign) near threshold",
}
NODE_GROUP_ORDER = ["TP", "FP", "FN", "TN_near_threshold"]
NODE_GROUP_COLORS = {
    "TP": "#dc2626",
    "FP": "#64748b",
    "FN": "#f97316",
    "TN_near_threshold": "#94a3b8",
}
OPERATION_COLORS = {
    "EVENT_EXECUTE": "#2563eb",
    "EVENT_SENDTO": "#e11d48",
    "EVENT_READ": "#059669",
    "EVENT_OPEN": "#f97316",
    "EVENT_RECVFROM": "#7c3aed",
    "EVENT_SENDMSG": "#0891b2",
    "EVENT_RECVMSG": "#c026d3",
    "EVENT_CLONE": "#a16207",
    "EVENT_CONNECT": "#be123c",
    "EVENT_WRITE": "#16a34a",
    "unknown": "#64748b",
}
TYPE_PAIR_HATCHES = {
    "file->subject": "",
    "subject->netflow": "///",
    "netflow->subject": "\\\\",
    "subject->subject": "xx",
    "subject->file": "..",
}
ATTACK_COLORS = {
    "Attack 0": "#f97316",
    "Attack 1": "#7c3aed",
    "Multiple attacks": "#dc2626",
    "benign": "#64748b",
}


def ranked_order(scores, nodes):
    # Match PIDSMaker's all-attacks ranking tie-break: sorted(zip(scores, nodes), reverse=True).
    return np.asarray(sorted(range(len(scores)), key=lambda i: (float(scores[i]), int(nodes[i])), reverse=True), dtype=int)

RUNS = [
    {
        "model_key": "leaky-selected",
        "model_label": "Leaky selected",
        "plot_slug": "leaky_selected",
        "score_epoch": 0,
        "score_path": INPUT_ROOT / "leaky_selected" / "scores_model_epoch_0.pkl",
        "edge_root": INPUT_ROOT / "leaky_selected" / "edge_losses",
    },
    {
        "model_key": "no-leaky-selected",
        "model_label": "Non-leaky selected",
        "plot_slug": "nonleaky_selected",
        "score_epoch": 3,
        "score_path": INPUT_ROOT / "nonleaky_selected" / "scores_model_epoch_3.pkl",
        "edge_root": INPUT_ROOT / "nonleaky_selected" / "edge_losses",
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


def fmt(value, digits=9):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def wrap(text, width=42):
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def ns_to_eastern(ns):
    try:
        return datetime.fromtimestamp(int(ns) / 1_000_000_000, tz=timezone.utc).astimezone(EASTERN).strftime("%Y-%m-%d %H:%M:%S.%f %Z")
    except Exception:
        return ""


def short_label(label):
    text = str(label)
    for prefix in ("file: ", "subject: "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if " | " in text:
        text = text.split(" | ", 1)[0]
    if text.startswith("netflow "):
        return text
    if not text.startswith("/"):
        return text
    parts = [part for part in text.split("/") if part]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[0] if parts else text


def short_attack_label(attacks):
    if not attacks or attacks == "benign":
        return "benign"
    parts = [part.strip() for part in str(attacks).split(";") if part.strip()]
    if len(parts) > 1:
        return "Multiple attacks"
    for attack_id, name in ATTACK_NAMES.items():
        if parts and parts[0] == name:
            return f"Attack {attack_id}"
    return parts[0] if parts else "benign"


def endpoint_pair_label(row):
    type_pair = row.get("type_pair", "source->destination")
    if "->" in type_pair:
        src_type, dst_type = type_pair.split("->", 1)
    else:
        src_type, dst_type = "source", "destination"
    readable = {"file": "file", "subject": "process", "netflow": "netflow"}
    src_role = readable.get(src_type, src_type)
    dst_role = readable.get(dst_type, dst_type)
    target_role = row.get("target_role", "")
    src_prefix = f"{src_role}:"
    dst_prefix = f"{dst_role}:"
    if target_role == "src":
        src_prefix = f"{src_prefix}*"
    elif target_role == "dst":
        dst_prefix = f"{dst_prefix}*"
    return f"{src_prefix} {short_label(row.get('src_label', ''))}\n-> {dst_prefix} {short_label(row.get('dst_label', ''))}"


def load_thresholds():
    rows = {}
    with THRESHOLD_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            for key in ("threshold", "max_val_loss", "mean_val_loss", "percentile_90_val_loss"):
                row[key] = float(row[key])
            for key in ("val_file_count", "val_row_count"):
                row[key] = int(row[key])
            rows[row["model"]] = row
    return rows


def load_scores(path):
    blob = torch_load(path)
    scores = np.asarray(blob["pred_scores"], dtype=float)
    labels = np.asarray(blob["y_truth"], dtype=int)
    preds = np.asarray(blob["y_preds"], dtype=int)
    nodes = np.asarray(blob["nodes"], dtype=int)
    node2attacks = {int(k): set(int(a) for a in v) for k, v in blob.get("node2attacks", {}).items()}
    return scores, labels, preds, nodes, node2attacks


def iter_edge_csvs(split_dir):
    for path in sorted(split_dir.glob("*.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                yield path, int(row["srcnode"]), int(row["dstnode"]), int(row["time"]), float(row["loss"])


def scan_losses(val_dir, test_dir, scored_nodes):
    val_losses = []
    for _, _, _, _, loss in iter_edge_csvs(val_dir):
        val_losses.append(loss)

    scored_set = set(int(node) for node in scored_nodes)
    test_losses = []
    max_edges = {}
    incident_counts = defaultdict(int)
    for path, src, dst, ts, loss in iter_edge_csvs(test_dir):
        test_losses.append(loss)
        window = path.name[:-4]
        if src in scored_set:
            incident_counts[src] += 1
            prev = max_edges.get(src)
            if prev is None or loss > prev["loss"]:
                max_edges[src] = {"loss": loss, "srcnode": src, "dstnode": dst, "time": ts, "target_role": "src", "window": window}
        if dst in scored_set and dst != src:
            incident_counts[dst] += 1
            prev = max_edges.get(dst)
            if prev is None or loss > prev["loss"]:
                max_edges[dst] = {"loss": loss, "srcnode": src, "dstnode": dst, "time": ts, "target_role": "dst", "window": window}
    return np.asarray(val_losses, dtype=float), np.asarray(test_losses, dtype=float), max_edges, dict(incident_counts)


def quantiles(values):
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


def attack_label(attacks):
    if not attacks:
        return "benign"
    return "; ".join(ATTACK_NAMES.get(int(attack), f"attack {attack}") for attack in sorted(attacks))


def db_connection():
    try:
        import psycopg2
    except Exception as exc:
        return None, f"psycopg2 unavailable: {exc}"

    host_candidates = [os.environ.get("DB_HOST")]
    host_candidates.extend(["localhost", "postgres"])
    seen = set()
    for host in [h for h in host_candidates if h and h not in seen and not seen.add(h)]:
        try:
            conn = psycopg2.connect(
                database=os.environ.get("DB_NAME", "theia_e3"),
                host=host,
                user=os.environ.get("DB_USER", "postgres"),
                password=os.environ.get("DB_PASSWORD", "postgres"),
                port=os.environ.get("DB_PORT", os.environ.get("DOCKER_PORT", "5432")),
            )
            return conn, f"connected:{host}"
        except Exception:
            continue
    return None, "DB unavailable"


def fetch_node_metadata(conn, node_ids):
    metadata = {}
    if conn is None or not node_ids:
        return metadata
    node_ids = sorted({int(n) for n in node_ids})
    with conn.cursor() as cur:
        for node_type, query in (
            ("file", "SELECT index_id, path FROM file_node_table WHERE index_id = ANY(%s);"),
            ("subject", "SELECT index_id, path, cmd FROM subject_node_table WHERE index_id = ANY(%s);"),
            ("netflow", "SELECT index_id, src_addr, src_port, dst_addr, dst_port FROM netflow_node_table WHERE index_id = ANY(%s);"),
        ):
            try:
                cur.execute(query, (node_ids,))
                for row in cur.fetchall():
                    node_id = int(row[0])
                    if node_type == "file":
                        metadata[node_id] = {"node_type": "file", "path": str(row[1] or "")}
                    elif node_type == "subject":
                        metadata[node_id] = {"node_type": "subject", "path": str(row[1] or ""), "cmd": str(row[2] or "")}
                    else:
                        metadata[node_id] = {
                            "node_type": "netflow",
                            "src_addr": str(row[1] or ""),
                            "src_port": str(row[2] or ""),
                            "dst_addr": str(row[3] or ""),
                            "dst_port": str(row[4] or ""),
                        }
            except Exception:
                conn.rollback()
    return metadata


def fetch_event_metadata(conn, edges):
    event_meta = {}
    if conn is None or not edges:
        return event_meta
    try:
        from psycopg2.extras import execute_values
    except Exception:
        execute_values = None
    keys = sorted({(str(int(edge["srcnode"])), str(int(edge["dstnode"])), int(edge["time"])) for edge in edges})
    with conn.cursor() as cur:
        if execute_values is not None:
            try:
                query = """
                    SELECT k.src_index_id, k.dst_index_id, k.timestamp_rec, e.operation, e.event_uuid
                    FROM (VALUES %s) AS k(src_index_id, dst_index_id, timestamp_rec)
                    LEFT JOIN event_table e
                      ON e.src_index_id = k.src_index_id
                     AND e.dst_index_id = k.dst_index_id
                     AND e.timestamp_rec = k.timestamp_rec;
                    """
                for i in range(0, len(keys), 1000):
                    chunk = keys[i : i + 1000]
                    execute_values(cur, query, chunk, template="(%s::varchar, %s::varchar, %s::bigint)", page_size=len(chunk))
                    for src, dst, ts, operation, event_uuid in cur.fetchall():
                        if operation is not None:
                            event_meta[(int(src), int(dst), int(ts))] = {"operation": str(operation or ""), "event_uuid": str(event_uuid or "")}
                return event_meta
            except Exception:
                conn.rollback()

        for src, dst, ts in keys[:200]:
            try:
                cur.execute(
                    """
                    SELECT operation, event_uuid
                    FROM event_table
                    WHERE src_index_id = %s AND dst_index_id = %s AND timestamp_rec = %s
                    ORDER BY event_uuid
                    LIMIT 1;
                    """,
                    (src, dst, ts),
                )
                row = cur.fetchone()
                if row is not None:
                    event_meta[(int(src), int(dst), int(ts))] = {"operation": str(row[0] or ""), "event_uuid": str(row[1] or "")}
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
    return f"node {node_id}"


def node_type(node_id, metadata):
    return metadata.get(int(node_id), {}).get("node_type", "unknown")


def selected_node_groups(scores, labels, nodes, threshold, order, high_below_n):
    pred = scores > threshold
    high_below = []
    for idx in order:
        if labels[idx] == 0 and not pred[idx] and len(high_below) < high_below_n:
            high_below.append(int(nodes[idx]))
    high_below = set(high_below)

    groups = {}
    for score, label, node, is_pred in zip(scores, labels, nodes, pred):
        node = int(node)
        if label == 1 and is_pred:
            groups[node] = "TP"
        elif label == 0 and is_pred:
            groups[node] = "FP"
        elif label == 1 and not is_pred:
            groups[node] = "FN"
        elif node in high_below:
            groups[node] = "TN_near_threshold"
    return groups, pred


def build_selected_rows(run, scores, labels, saved_preds, nodes, node2attacks, threshold, max_edges, incident_counts, metadata, event_meta, high_below_n):
    order = ranked_order(scores, nodes)
    rank_by_node = {int(nodes[idx]): rank for rank, idx in enumerate(order, start=1)}
    score_by_node = {int(node): float(score) for node, score in zip(nodes, scores)}
    label_by_node = {int(node): int(label) for node, label in zip(nodes, labels)}
    groups, pred = selected_node_groups(scores, labels, nodes, threshold, order, high_below_n)

    rows = []
    for node in sorted(groups, key=lambda n: rank_by_node[n]):
        edge = max_edges.get(node)
        src = int(edge["srcnode"]) if edge else ""
        dst = int(edge["dstnode"]) if edge else ""
        ts = int(edge["time"]) if edge else ""
        key = (src, dst, ts) if edge else None
        event = event_meta.get(key, {}) if key else {}
        src_type = node_type(src, metadata) if edge else ""
        dst_type = node_type(dst, metadata) if edge else ""
        type_pair = f"{src_type}->{dst_type}" if edge else ""
        responsible_loss = float(edge["loss"]) if edge else float("nan")
        score = score_by_node[node]
        group = groups[node]
        rows.append(
            {
                "model_label": run["model_label"],
                "score_epoch": run["score_epoch"],
                "rank": rank_by_node[node],
                "node_id": node,
                "node_group": group,
                "node_group_label": NODE_GROUP_LABELS[group],
                "score": fmt(score),
                "label": label_by_node[node],
                "pred_above_threshold": int(score > threshold),
                "saved_pred": int(saved_preds[np.where(nodes == node)[0][0]]),
                "attacks": attack_label(node2attacks.get(node, set())),
                "incident_edges": incident_counts.get(node, 0),
                "responsible_loss": fmt(responsible_loss),
                "score_minus_responsible_loss": fmt(score - responsible_loss) if edge else "",
                "operation": event.get("operation", "unknown"),
                "event_uuid": event.get("event_uuid", ""),
                "time_raw": ts,
                "time_eastern": ns_to_eastern(ts) if edge else "",
                "window": edge.get("window", "") if edge else "",
                "target_role": edge.get("target_role", "") if edge else "",
                "srcnode": src,
                "dstnode": dst,
                "src_type": src_type,
                "dst_type": dst_type,
                "type_pair": type_pair,
                "node_label_text": node_label(node, metadata),
                "src_label": node_label(src, metadata) if edge else "",
                "dst_label": node_label(dst, metadata) if edge else "",
                "short_endpoint_pair": f"{short_label(node_label(src, metadata))} -> {short_label(node_label(dst, metadata))}" if edge else "",
            }
        )
    return rows, pred, groups


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def save_plot(fig, stem):
    group = "non-leaky" if "nonleaky" in stem else "leaky"
    plot_dir = PLOT_ROOT / group / "from_saved_edge_losses_and_db"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"{stem}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)
    plt.close(fig)


def plot_loss_distribution(run, val_losses, test_losses, threshold):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    values = np.concatenate([val_losses, test_losses])
    bins = np.linspace(float(values.min()), float(values.max()), 140)
    ax.hist(val_losses, bins=bins, color="#2563eb", alpha=0.48, label=f"Validation ({len(val_losses):,})")
    ax.hist(test_losses, bins=bins, color="#f97316", alpha=0.58, label=f"Test ({len(test_losses):,})")
    ax.axvline(threshold, color="#111827", lw=1.7, ls="--", label="Validation threshold")
    ax.set_yscale("log")
    ax.set_xlabel("VELOX edge loss")
    ax.set_ylabel("Edge count (log scale)")
    ax.set_title(f"{run['model_label']} edge-loss distribution", loc="left", fontsize=11.5)
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_plot(fig, f"theia_e3_ulhpc_{run['plot_slug']}_edge_loss_distribution_validation_vs_test")


def plot_score_vs_responsible_loss(run, selected_rows, threshold):
    rows = [row for row in selected_rows if row["responsible_loss"]]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    max_value = threshold
    for group in NODE_GROUP_ORDER:
        group_rows = [row for row in rows if row["node_group"] == group]
        if not group_rows:
            continue
        x = np.asarray([float(row["score"]) for row in group_rows], dtype=float)
        y = np.asarray([float(row["responsible_loss"]) for row in group_rows], dtype=float)
        max_value = max(max_value, float(np.max(x)), float(np.max(y)))
        ax.scatter(x, y, s=18, alpha=0.82, color=NODE_GROUP_COLORS[group], label=f"{NODE_GROUP_LABELS[group]} ({len(group_rows):,})")
    ax.plot([0, max_value], [0, max_value], color="#0f172a", lw=1.0, alpha=0.75)
    ax.axvline(threshold, color="#111827", lw=1.4, ls="--")
    ax.axhline(threshold, color="#111827", lw=1.4, ls="--")
    ax.set_xlim(left=0, right=max_value * 1.03)
    ax.set_ylim(bottom=0, top=max_value * 1.03)
    ax.set_xlabel("Saved node score")
    ax.set_ylabel("Max incident test edge loss")
    ax.set_title(f"{run['model_label']} responsible-edge check", loc="left", fontsize=11.5)
    ax.grid(color="#e2e8f0", lw=0.8)
    ax.legend(loc="upper left", fontsize=8.8)
    fig.tight_layout()
    save_plot(fig, f"theia_e3_ulhpc_{run['plot_slug']}_score_vs_responsible_edge_loss")


def plot_top_node_responsible_edges(run, selected_rows, threshold, top_n=25):
    rows = [row for row in selected_rows if row["responsible_loss"]][:top_n]
    if not rows:
        return
    y = np.arange(len(rows))
    scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
    losses = np.asarray([float(row["responsible_loss"]) for row in rows], dtype=float)
    colors = [NODE_GROUP_COLORS[row["node_group"]] for row in rows]
    labels = [wrap(f"#{row['rank']} {row['node_group_label']} node {row['node_id']}", 32) for row in rows]

    fig, ax = plt.subplots(figsize=(9.2, max(5.2, len(rows) * 0.36)))
    ax.barh(y, scores, color="#e2e8f0", edgecolor="none", label="Saved node score")
    ax.scatter(losses, y, color=colors, s=34, zorder=3, label="Responsible edge loss")
    ax.axvline(threshold, color="#111827", lw=1.5, ls="--", label="Validation threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlabel("Loss / node score")
    ax.set_title(f"{run['model_label']} top responsible edges", loc="left", fontsize=11.5)
    ax.grid(axis="x", color="#e2e8f0", lw=0.8)
    ax.legend(loc="lower right", fontsize=9.0)
    fig.tight_layout()
    save_plot(fig, f"theia_e3_ulhpc_{run['plot_slug']}_top_node_responsible_edges")


def plot_operation_mix(run, operation_rows):
    rows = [row for row in operation_rows if row["model_label"] == run["model_label"]]
    if not rows:
        return
    groups = [group for group in NODE_GROUP_ORDER if any(row["node_group"] == group for row in rows)]
    totals = {group: sum(int(row["count"]) for row in rows if row["node_group"] == group) for group in groups}
    operation_totals = Counter()
    counts = defaultdict(int)
    for row in rows:
        count = int(row["count"])
        operation = row["operation"]
        operation_totals[operation] += count
        counts[(row["node_group"], operation)] += count
    operations = [operation for operation, _ in operation_totals.most_common()]

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for operation in operations:
        vals = np.asarray([counts.get((group, operation), 0) / totals[group] if totals[group] else 0.0 for group in groups])
        if not vals.any():
            continue
        label = "event type unavailable" if operation == "unknown" else operation
        ax.bar(x, vals, bottom=bottom, color=OPERATION_COLORS.get(operation, OPERATION_COLORS["unknown"]), label=label)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([wrap(NODE_GROUP_LABELS[group], 14) for group in groups], fontsize=8.8)
    ax.set_ylabel("Fraction within node group")
    ax.set_ylim(0, 1)
    ax.set_title(f"{run['model_label']} responsible-edge event mix", loc="left", fontsize=11.5)
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.8)
    fig.tight_layout()
    save_plot(fig, f"theia_e3_ulhpc_{run['plot_slug']}_node_group_operation_mix")


def plot_incident_edge_counts(run, selected_rows):
    groups = [group for group in NODE_GROUP_ORDER if any(row["node_group"] == group for row in selected_rows)]
    values = [[int(row["incident_edges"]) for row in selected_rows if row["node_group"] == group] for group in groups]
    if not any(values):
        return

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    box = ax.boxplot(values, patch_artist=True, tick_labels=[wrap(NODE_GROUP_LABELS[group], 14) for group in groups], showfliers=False)
    for patch, group in zip(box["boxes"], groups):
        patch.set_facecolor(NODE_GROUP_COLORS[group])
        patch.set_alpha(0.75)
        patch.set_edgecolor("#0f172a")
    for key in ("whiskers", "caps", "medians"):
        for artist in box[key]:
            artist.set_color("#0f172a")
            artist.set_linewidth(1.1)
    ax.set_ylabel("Incident edge count")
    ax.set_title(f"{run['model_label']} incident edges by node group", loc="left", fontsize=11.5)
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    fig.tight_layout()
    save_plot(fig, f"theia_e3_ulhpc_{run['plot_slug']}_node_group_incident_edges")


def plot_operation_type_pair_mix(run, operation_rows):
    rows = [row for row in operation_rows if row["model_label"] == run["model_label"]]
    if not rows:
        return
    groups = [group for group in NODE_GROUP_ORDER if any(row["node_group"] == group for row in rows)]
    totals = {group: sum(int(row["count"]) for row in rows if row["node_group"] == group) for group in groups}
    operations = [operation for operation, _ in Counter({row["operation"]: 0 for row in rows}).items()]
    operations = [operation for operation, _ in Counter(row["operation"] for row in rows).most_common()]
    type_pairs = [type_pair for type_pair, _ in Counter(row["type_pair"] for row in rows).most_common()]
    combos = [(operation, type_pair) for operation in operations for type_pair in type_pairs]
    counts = defaultdict(int)
    for row in rows:
        counts[(row["node_group"], row["operation"], row["type_pair"])] += int(row["count"])

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for operation, type_pair in combos:
        vals = np.asarray([counts.get((group, operation, type_pair), 0) / totals[group] if totals[group] else 0.0 for group in groups])
        if not vals.any():
            continue
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=OPERATION_COLORS.get(operation, OPERATION_COLORS["unknown"]),
            hatch=TYPE_PAIR_HATCHES.get(type_pair, "xx"),
            edgecolor="#0f172a",
            linewidth=0.35,
        )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([wrap(NODE_GROUP_LABELS[group], 14) for group in groups], fontsize=8.8)
    ax.set_ylabel("Fraction within node group")
    ax.set_ylim(0, 1)
    ax.set_title(f"{run['model_label']} event and endpoint-type mix", loc="left", fontsize=11.5)
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    operation_handles = [plt.Rectangle((0, 0), 1, 1, color=OPERATION_COLORS.get(op, OPERATION_COLORS["unknown"]), label=op) for op in operations]
    type_handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#0f172a", hatch=TYPE_PAIR_HATCHES.get(tp, "xx"), label=tp) for tp in type_pairs]
    operation_legend = ax.legend(handles=operation_handles, title="Event type", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5, title_fontsize=9.0)
    ax.add_artist(operation_legend)
    ax.legend(handles=type_handles, title="Endpoint type", loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=8.5, title_fontsize=9.0)
    fig.tight_layout()
    save_plot(fig, f"theia_e3_ulhpc_{run['plot_slug']}_node_group_operation_type_pair_mix")


def aggregate_endpoint_pairs(run, selected_rows):
    rows = [row for row in selected_rows if row["model_label"] == run["model_label"]]
    grouped = defaultdict(int)
    examples = {}
    for row in rows:
        key = (
            row["node_group"],
            row["operation"],
            row["type_pair"],
            row["target_role"],
            row["src_label"],
            row["dst_label"],
            row["attacks"],
        )
        grouped[key] += 1
        examples.setdefault(key, row)

    out = []
    for key, count in grouped.items():
        row = examples[key]
        out.append(
            {
                "model_label": run["model_label"],
                "node_group": key[0],
                "node_group_label": NODE_GROUP_LABELS[key[0]],
                "operation": key[1],
                "type_pair": key[2],
                "target_role": key[3],
                "src_label": key[4],
                "dst_label": key[5],
                "attack": key[6],
                "short_attack": short_attack_label(key[6]),
                "endpoint_pair": f"{key[4]} -> {key[5]}",
                "short_endpoint_pair": row["short_endpoint_pair"],
                "plot_label": endpoint_pair_label(row),
                "count": count,
            }
        )
    out.sort(key=lambda row: (row["model_label"], row["node_group"], -int(row["count"]), row["operation"], row["short_endpoint_pair"]))
    return out


def endpoint_panel_rows(rows, max_rows=None):
    rows = sorted(rows, key=lambda row: int(row["count"]), reverse=True)
    return rows[:max_rows] if max_rows else rows


def endpoint_row_gap(label):
    line_count = max(1, len(str(label).splitlines()))
    return 1.35 + max(0, line_count - 1) * 0.58


def endpoint_panel_height(rows, max_rows=None):
    rows = endpoint_panel_rows(rows, max_rows)
    if not rows:
        return 3.2
    labels = [wrap(row["plot_label"], 52) for row in rows]
    units = sum(endpoint_row_gap(label) for label in labels)
    return max(3.4, units * 0.42 + 1.2)


def plot_endpoint_panel(ax, rows, title, max_rows=None):
    rows = endpoint_panel_rows(rows, max_rows)
    if not rows:
        ax.text(0.5, 0.5, "No rows", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    labels = [wrap(row["plot_label"], 52) for row in rows]
    gaps = np.asarray([endpoint_row_gap(label) for label in labels], dtype=float)
    y = np.r_[0.0, np.cumsum(gaps[:-1])]
    counts = np.asarray([int(row["count"]) for row in rows], dtype=int)
    colors = [OPERATION_COLORS.get(row["operation"], OPERATION_COLORS["unknown"]) for row in rows]
    ax.barh(y, counts, height=0.62, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.2)
    ax.set_ylim(y[-1] + gaps[-1] * 0.72, -gaps[0] * 0.72)
    ax.set_xlabel("Node count", fontsize=10.2)
    ax.set_title(title, loc="left", fontsize=12.2, weight="bold", pad=10)
    ax.grid(axis="x", color="#e2e8f0", lw=0.8)
    max_count = int(max(counts))
    for yi, count, row in zip(y, counts, rows):
        ax.text(count + max_count * 0.015, yi, str(count), va="center", ha="left", fontsize=9.0, color="#334155")
        ax.scatter(max_count * 1.13, yi, color=ATTACK_COLORS.get(row["short_attack"], "#64748b"), s=24, zorder=3)
        ax.text(max_count * 1.17, yi, row["short_attack"], va="center", ha="left", fontsize=8.8, color="#334155")
    ax.set_xlim(0, max_count * 1.48 + 1)


def add_endpoint_legends(fig, rows):
    operations = [operation for operation, _ in Counter(row["operation"] for row in rows).most_common()]
    attacks = [attack for attack in ("Attack 0", "Attack 1", "Multiple attacks", "benign") if any(row["short_attack"] == attack for row in rows)]
    operation_handles = [plt.Rectangle((0, 0), 1, 1, color=OPERATION_COLORS.get(op, OPERATION_COLORS["unknown"]), label=op) for op in operations]
    attack_handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ATTACK_COLORS[attack], markeredgecolor="white", markersize=7, label=attack) for attack in attacks]
    if operation_handles:
        fig.legend(handles=operation_handles, title="Event type", loc="upper right", bbox_to_anchor=(0.985, 0.94), frameon=False, fontsize=9.2, title_fontsize=10.0)
    if attack_handles:
        fig.legend(handles=attack_handles, title="Attack label", loc="lower right", bbox_to_anchor=(0.985, 0.08), frameon=False, fontsize=9.2, title_fontsize=10.0)


def plot_fn_endpoint_pairs(run, endpoint_rows):
    rows = [row for row in endpoint_rows if row["model_label"] == run["model_label"] and row["node_group"] == "FN"]
    if not rows:
        return
    type_pairs = [tp for tp, _ in Counter(row["type_pair"] for row in rows).most_common()]
    panel_heights = [endpoint_panel_height([row for row in rows if row["type_pair"] == type_pair]) for type_pair in type_pairs]
    fig, axes = plt.subplots(len(type_pairs), 1, figsize=(14.2, sum(panel_heights) + 0.6), gridspec_kw={"height_ratios": panel_heights})
    if len(type_pairs) == 1:
        axes = [axes]
    for ax, type_pair in zip(axes, type_pairs):
        panel_rows = [row for row in rows if row["type_pair"] == type_pair]
        plot_endpoint_panel(ax, panel_rows, f"FN: {type_pair} ({sum(int(row['count']) for row in panel_rows)} nodes, {len(panel_rows)} pairs)")
    add_endpoint_legends(fig, rows)
    fig.tight_layout(rect=(0, 0, 0.84, 1), h_pad=2.0)
    save_plot(fig, f"theia_e3_ulhpc_{run['plot_slug']}_fn_responsible_endpoint_pairs")


def plot_tp_fp_endpoint_pairs(run, endpoint_rows):
    rows = [row for row in endpoint_rows if row["model_label"] == run["model_label"] and row["node_group"] in {"TP", "FP"}]
    if not rows:
        return
    panel_sets = [[row for row in rows if row["node_group"] == group] for group in ("TP", "FP")]
    panel_heights = [endpoint_panel_height(panel_rows) for panel_rows in panel_sets]
    fig, axes = plt.subplots(2, 1, figsize=(14.2, sum(panel_heights) + 0.8), gridspec_kw={"height_ratios": panel_heights})
    for ax, group in zip(axes, ("TP", "FP")):
        panel_rows = [row for row in rows if row["node_group"] == group]
        label = "malicious" if group == "TP" else "benign"
        plot_endpoint_panel(ax, panel_rows, f"{group}: {label} ({sum(int(row['count']) for row in panel_rows)} nodes, {len(panel_rows)} pairs)")
    add_endpoint_legends(fig, rows)
    fig.tight_layout(rect=(0, 0, 0.84, 1), h_pad=2.0)
    save_plot(fig, f"theia_e3_ulhpc_{run['plot_slug']}_tp_fp_responsible_endpoint_pairs")


def summarize_node_groups(run, selected_rows):
    rows = []
    for group in NODE_GROUP_ORDER:
        group_rows = [row for row in selected_rows if row["node_group"] == group]
        if not group_rows:
            rows.append(
                {
                    "model_label": run["model_label"],
                    "node_group": group,
                    "node_group_label": NODE_GROUP_LABELS[group],
                    "nodes": 0,
                    "score_p50": "",
                    "score_p90": "",
                    "score_max": "",
                    "responsible_loss_p50": "",
                    "responsible_loss_p90": "",
                    "responsible_loss_max": "",
                    "incident_edges_p50": "",
                    "incident_edges_p90": "",
                }
            )
            continue
        scores = np.asarray([float(row["score"]) for row in group_rows], dtype=float)
        losses = np.asarray([float(row["responsible_loss"]) for row in group_rows if row["responsible_loss"]], dtype=float)
        incident = np.asarray([int(row["incident_edges"]) for row in group_rows], dtype=float)
        rows.append(
            {
                "model_label": run["model_label"],
                "node_group": group,
                "node_group_label": NODE_GROUP_LABELS[group],
                "nodes": len(group_rows),
                "score_p50": fmt(np.quantile(scores, 0.50)),
                "score_p90": fmt(np.quantile(scores, 0.90)),
                "score_max": fmt(np.max(scores)),
                "responsible_loss_p50": fmt(np.quantile(losses, 0.50)) if len(losses) else "",
                "responsible_loss_p90": fmt(np.quantile(losses, 0.90)) if len(losses) else "",
                "responsible_loss_max": fmt(np.max(losses)) if len(losses) else "",
                "incident_edges_p50": fmt(np.quantile(incident, 0.50), 3),
                "incident_edges_p90": fmt(np.quantile(incident, 0.90), 3),
            }
        )
    return rows


def summarize_operations(run, selected_rows):
    grouped = defaultdict(Counter)
    for row in selected_rows:
        grouped[row["node_group"]][(row["operation"] or "unknown", row["type_pair"] or "unknown->unknown")] += 1
    rows = []
    for group in NODE_GROUP_ORDER:
        total = sum(grouped[group].values())
        for (operation, type_pair), count in grouped[group].most_common():
            rows.append(
                {
                    "model_label": run["model_label"],
                    "node_group": group,
                    "node_group_label": NODE_GROUP_LABELS[group],
                    "operation": operation,
                    "type_pair": type_pair,
                    "count": count,
                    "fraction": fmt(count / total if total else 0.0, 6),
                }
            )
    return rows


def process_run(run, threshold_row, high_below_n):
    scores, labels, saved_preds, nodes, node2attacks = load_scores(run["score_path"])
    val_dir = run["edge_root"] / "val"
    test_dir = run["edge_root"] / "test"
    val_losses, test_losses, max_edges, incident_counts = scan_losses(val_dir, test_dir, nodes)
    threshold = float(threshold_row["threshold"])

    endpoint_nodes = set()
    selected_probe_edges = []
    order = ranked_order(scores, nodes)
    provisional_groups, _ = selected_node_groups(scores, labels, nodes, threshold, order, high_below_n)
    for node in provisional_groups:
        edge = max_edges.get(node)
        endpoint_nodes.add(int(node))
        if edge:
            selected_probe_edges.append(edge)
            endpoint_nodes.add(int(edge["srcnode"]))
            endpoint_nodes.add(int(edge["dstnode"]))

    conn, db_status = db_connection()
    metadata = fetch_node_metadata(conn, endpoint_nodes)
    event_meta = fetch_event_metadata(conn, selected_probe_edges)
    if conn is not None:
        conn.close()

    selected_rows, pred, groups = build_selected_rows(
        run,
        scores,
        labels,
        saved_preds,
        nodes,
        node2attacks,
        threshold,
        max_edges,
        incident_counts,
        metadata,
        event_meta,
        high_below_n,
    )

    errors = []
    for node, score in zip(nodes, scores):
        edge = max_edges.get(int(node))
        if edge:
            errors.append(abs(float(score) - float(edge["loss"])))
    errors = np.asarray(errors, dtype=float)
    saved_pred_mismatches = int(np.sum(saved_preds != pred.astype(int)))
    group_counts = Counter(groups.values())
    val_q = quantiles(val_losses)
    test_q = quantiles(test_losses)
    summary = {
        "model_label": run["model_label"],
        "score_epoch": run["score_epoch"],
        "threshold_method": threshold_row["threshold_method"],
        "threshold": fmt(threshold),
        "computed_max_val_loss": fmt(float(np.max(val_losses)) if len(val_losses) else float("nan")),
        "threshold_minus_computed_max_val_loss": fmt(threshold - float(np.max(val_losses)) if len(val_losses) else float("nan")),
        "val_file_count": threshold_row["val_file_count"],
        "val_row_count_threshold_file": threshold_row["val_row_count"],
        "val_row_count_scanned": len(val_losses),
        "test_row_count_scanned": len(test_losses),
        "scored_nodes": len(nodes),
        "nodes_with_incident_test_edges": len(max_edges),
        "score_responsible_loss_exact_matches_1e_6": int(np.sum(errors <= 1e-6)) if len(errors) else 0,
        "score_responsible_loss_exact_matches_1e_5": int(np.sum(errors <= 1e-5)) if len(errors) else 0,
        "max_abs_score_minus_responsible_loss": fmt(float(np.max(errors)) if len(errors) else float("nan"), 12),
        "saved_pred_mismatches_vs_threshold_rule": saved_pred_mismatches,
        "tp": group_counts["TP"],
        "fp": group_counts["FP"],
        "fn": group_counts["FN"],
        "tn_near_threshold_selected": group_counts["TN_near_threshold"],
        "val_loss_p50": fmt(val_q["p50"]),
        "val_loss_p90": fmt(val_q["p90"]),
        "val_loss_p99": fmt(val_q["p99"]),
        "val_loss_max": fmt(val_q["max"]),
        "test_loss_p50": fmt(test_q["p50"]),
        "test_loss_p90": fmt(test_q["p90"]),
        "test_loss_p99": fmt(test_q["p99"]),
        "test_loss_max": fmt(test_q["max"]),
        "db_status": db_status,
        "score_path": str(run["score_path"]),
        "val_edge_loss_dir": str(val_dir),
        "test_edge_loss_dir": str(test_dir),
    }

    plot_loss_distribution(run, val_losses, test_losses, threshold)
    plot_score_vs_responsible_loss(run, selected_rows, threshold)
    plot_top_node_responsible_edges(run, selected_rows, threshold)
    return summary, selected_rows, summarize_node_groups(run, selected_rows), summarize_operations(run, selected_rows)


def main():
    setup_style()
    thresholds = load_thresholds()
    high_below_n = int(os.environ.get("ULHPC_HIGH_BELOW_N", "500"))
    summary_rows = []
    selected_rows = []
    node_group_rows = []
    operation_rows = []
    endpoint_pair_rows = []

    for run in RUNS:
        summary, selected, groups, operations = process_run(run, thresholds[run["model_key"]], high_below_n)
        summary_rows.append(summary)
        selected_rows.extend(selected)
        node_group_rows.extend(groups)
        operation_rows.extend(operations)
        endpoint_pair_rows.extend(aggregate_endpoint_pairs(run, selected))

    write_csv(TABLE_DIR / "ulhpc_theia_e3_selected_edge_loss_summary.csv", list(summary_rows[0]), summary_rows)
    write_csv(TABLE_DIR / "ulhpc_theia_e3_selected_responsible_edges.csv", list(selected_rows[0]), selected_rows)
    write_csv(TABLE_DIR / "ulhpc_theia_e3_selected_node_group_summary.csv", list(node_group_rows[0]), node_group_rows)
    write_csv(TABLE_DIR / "ulhpc_theia_e3_selected_node_group_operation_summary.csv", list(operation_rows[0]), operation_rows)
    write_csv(TABLE_DIR / "ulhpc_theia_e3_selected_responsible_endpoint_pair_summary.csv", list(endpoint_pair_rows[0]), endpoint_pair_rows)

    for run in RUNS:
        plot_operation_mix(run, operation_rows)
        run_selected_rows = [row for row in selected_rows if row["model_label"] == run["model_label"]]
        plot_incident_edge_counts(run, run_selected_rows)
        plot_operation_type_pair_mix(run, operation_rows)
        plot_fn_endpoint_pairs(run, endpoint_pair_rows)
        plot_tp_fp_endpoint_pairs(run, endpoint_pair_rows)


if __name__ == "__main__":
    main()
