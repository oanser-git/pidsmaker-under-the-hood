import argparse
import csv
import math
import os
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch

from poc_paths import artifacts_root


RUNS = {
    "THEIA_E3": {
        "obs_id": "2026-07-01_theia_e3_velox",
        "dataset": "THEIA_E3",
        "db": "theia_e3",
        "gnn_run_id": "b697daa7af550a0e650e8603203036a4ea5450d572f20f2c6ba0aafde1c8b5fc",
        "eval_run_id": "720c98a89d0bb855ac3dd7ae738b360441ca346e5ac7fc97680d12d2e1884108",
        "attack_names": {
            0: "Browser extension Drakon dropper",
            1: "Firefox backdoor Drakon in-memory",
        },
    },
    "CADETS_E3": {
        "obs_id": "2026-07-01_cadets_e3_velox",
        "dataset": "CADETS_E3",
        "db": "cadets_e3",
        "gnn_run_id": "ad1cce59e1f470e42f1c408928d542f03116b80d2fb8f91314206ec3f855fd52",
        "eval_run_id": "4098bf9f7cb64a267341674a27e1074ec7192a4bfd0e07cb2e9af0bbafd58e4c",
        "attack_names": {
            0: "Nginx backdoor 06",
            1: "Nginx backdoor 12",
            2: "Nginx backdoor 13",
        },
    },
    "optc_h201": {
        "obs_id": "2026-07-01_optc_h201_velox",
        "dataset": "optc_h201",
        "db": "optc_201",
        "gnn_run_id": "74a8ea9eb92352cccd45ef6c8ce8a7d853794ad55b3770f751610f2167e6a483",
        "eval_run_id": "13911614d72e54864db72628acc32dbd72687c95aa16358ffa173268494e79e9",
        "attack_names": {0: "OPTC h201 attack"},
    },
    "optc_h501": {
        "obs_id": "2026-07-01_optc_h501_velox",
        "dataset": "optc_h501",
        "db": "optc_501",
        "gnn_run_id": "3913a9c3d68fcdd8485adaed7176a063f188aec97be631e7c4d351718afb8ef7",
        "eval_run_id": "1ed3b5b56aa5e56830706f3a70641c2defa8caec9d57f423e43399cc91db2517",
        "attack_names": {0: "OPTC h501 attack"},
    },
    "optc_h051": {
        "obs_id": "2026-07-01_optc_h051_velox",
        "dataset": "optc_h051",
        "db": "optc_051",
        "gnn_run_id": "6c7bf3e5bcd436b07bcfffaa3a73840330db85c25cc4b179b85b86fbbdf6fb62",
        "eval_run_id": "9c7192a2df477c09b282149c9705682aa82c5d8d4a05876adcc95a568af2a65c",
        "attack_names": {0: "OPTC h051 attack"},
    },
}

OBS_ID = RUNS["THEIA_E3"]["obs_id"]
DATASET = RUNS["THEIA_E3"]["dataset"]
DB_NAME = RUNS["THEIA_E3"]["db"]
GNN_RUN_ID = RUNS["THEIA_E3"]["gnn_run_id"]
EVAL_RUN_ID = RUNS["THEIA_E3"]["eval_run_id"]
ATTACK_NAMES = dict(RUNS["THEIA_E3"]["attack_names"])
ENDPOINT_TOP_N = 0
EASTERN = ZoneInfo("US/Eastern") if ZoneInfo is not None else timezone.utc


ROOT = artifacts_root()
OBS_ROOT = ROOT / "observations"
OBS_DIR = OBS_ROOT / OBS_ID
SCORE_PATH = ROOT / f"detection/evaluation/{EVAL_RUN_ID}/{DATASET}/precision_recall_dir/scores_model_epoch_0.pkl"
RESULT_PATH = ROOT / f"detection/evaluation/{EVAL_RUN_ID}/{DATASET}/results/results.pth"
EDGE_ROOT = ROOT / f"detection/gnn_training/{GNN_RUN_ID}/{DATASET}/edge_losses"
VAL_DIR = EDGE_ROOT / "val/model_epoch_0"
TEST_DIR = EDGE_ROOT / "test/model_epoch_0"


def configure_run(run_key):
    global OBS_ID, DATASET, DB_NAME, GNN_RUN_ID, EVAL_RUN_ID, ATTACK_NAMES
    global OBS_DIR, SCORE_PATH, RESULT_PATH, EDGE_ROOT, VAL_DIR, TEST_DIR
    run = RUNS[run_key]
    OBS_ID = run["obs_id"]
    DATASET = run["dataset"]
    DB_NAME = run["db"]
    GNN_RUN_ID = run["gnn_run_id"]
    EVAL_RUN_ID = run["eval_run_id"]
    ATTACK_NAMES = dict(run["attack_names"])
    OBS_DIR = OBS_ROOT / OBS_ID
    SCORE_PATH = ROOT / f"detection/evaluation/{EVAL_RUN_ID}/{DATASET}/precision_recall_dir/scores_model_epoch_0.pkl"
    RESULT_PATH = ROOT / f"detection/evaluation/{EVAL_RUN_ID}/{DATASET}/results/results.pth"
    EDGE_ROOT = ROOT / f"detection/gnn_training/{GNN_RUN_ID}/{DATASET}/edge_losses"
    VAL_DIR = EDGE_ROOT / "val/model_epoch_0"
    TEST_DIR = EDGE_ROOT / "test/model_epoch_0"


NODE_GROUP_LABELS = {
    "TP_above_threshold": "TP: malicious above threshold",
    "FP_above_threshold": "FP: benign above threshold",
    "FN_below_threshold": "FN: malicious below threshold",
    "TN_high_below_threshold": "Top selected benign nodes just below threshold",
    "TN_other": "Other benign below threshold",
}
NODE_GROUP_COMPACT_LABELS = {
    "TP_above_threshold": "TP\nmalicious",
    "FP_above_threshold": "FP\nbenign",
    "FN_below_threshold": "FN\nmalicious",
    "TN_high_below_threshold": "Top 50k\nbenign below\nthreshold",
    "TN_other": "Other\nbenign below\nthreshold",
}
NODE_GROUP_ORDER = list(NODE_GROUP_LABELS)
NODE_GROUP_COLORS = {
    "TP_above_threshold": "#dc2626",
    "FP_above_threshold": "#64748b",
    "FN_below_threshold": "#f97316",
    "TN_high_below_threshold": "#94a3b8",
    "TN_other": "#cbd5e1",
}
OPERATION_COLORS = {
    # THEIA/CADETS-style event names.
    "EVENT_CONNECT": "#0f766e",
    "EVENT_EXECUTE": "#2563eb",
    "EVENT_CLONE": "#65a30d",
    "EVENT_SENDTO": "#e11d48",
    "EVENT_READ": "#059669",
    "EVENT_OPEN": "#f97316",
    "EVENT_RECVFROM": "#7c3aed",
    "EVENT_SENDMSG": "#0891b2",
    "EVENT_RECVMSG": "#c026d3",
    "EVENT_WRITE": "#a16207",
    # OPTC-style event names.
    "CREATE": "#2563eb",
    "DELETE": "#dc2626",
    "MESSAGE": "#7c3aed",
    "MODIFY": "#f97316",
    "OPEN": "#059669",
    "READ": "#0891b2",
    "RENAME": "#db2777",
    "START": "#16a34a",
    "TERMINATE": "#64748b",
    "WRITE": "#a16207",
    "unknown": "#64748b",
}
ATTACK_COLORS = {
    "Attack 0": "#f97316",
    "Attack 1": "#7c3aed",
    "Attack 2": "#0891b2",
    "Attack 3": "#16a34a",
    "Attack 4": "#be123c",
    "Attack 5": "#a16207",
    "Multiple attacks": "#dc2626",
    "benign": "#64748b",
}
TYPE_PAIR_ORDER = [
    "file->subject",
    "subject->file",
    "subject->netflow",
    "netflow->subject",
    "subject->subject",
    "file->file",
    "netflow->netflow",
    "unknown->unknown",
]
TYPE_PAIR_HATCHES = {
    "file->subject": "",
    "subject->file": "///",
    "subject->netflow": "\\\\",
    "netflow->subject": "---",
    "subject->subject": "xx",
    "file->file": "..",
    "netflow->netflow": "++",
    "unknown->unknown": "xx",
}


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


def fmt(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def wrap(text, width=52):
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def wrap_preserve_lines(text, width=52):
    wrapped_lines = []
    for line in str(text).splitlines():
        wrapped = textwrap.wrap(line, width=width, break_long_words=False)
        wrapped_lines.extend(wrapped if wrapped else [""])
    return "\n".join(wrapped_lines)


def strip_endpoint_prefix(label):
    for prefix in ("file: ", "subject: "):
        if str(label).startswith(prefix):
            return str(label)[len(prefix):]
    return str(label)


def shorten_endpoint(label):
    text = strip_endpoint_prefix(label)
    if text.startswith("netflow "):
        return text.replace("netflow ", "netflow ")
    if " | " in text:
        text = text.split(" | ", 1)[0]
    if text.startswith("./"):
        return text
    if not text.startswith("/"):
        return text

    parts = [part for part in text.split("/") if part]
    if not parts:
        return text
    if "native-messaging-hosts" in parts:
        idx = parts.index("native-messaging-hosts")
        return "/".join(parts[idx:])
    if len(parts) >= 3 and parts[-1] in {"qmake", "firefox", "clean", "profile", "mail", "crashreporter"}:
        return "/".join(parts[-3:])
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1]


def compact_pair_label(src_label, dst_label):
    return f"{shorten_endpoint(src_label)} --> {shorten_endpoint(dst_label)}"


def underline_text(text):
    return "".join(f"{char}\u0332" if char != "\n" else char for char in str(text))


def role_label(role, flagged):
    text = underline_text(role) if flagged else role
    return f"$\\bf{{{text}}}$"


def wrap_endpoint_line(prefix, role, endpoint, flagged, width):
    line = f"{prefix}{role_label(role, flagged)}: {endpoint}"
    return textwrap.wrap(line, width=width, break_long_words=False) or [""]


def plot_endpoint_pair_label(row, width=58):
    type_pair = row.get("type_pair", "source->destination")
    if "->" in type_pair:
        src_type, dst_type = type_pair.split("->", 1)
    else:
        src_type, dst_type = "source", "destination"
    readable_type = {
        "file": "file",
        "subject": "process",
        "netflow": "netflow",
    }
    src_role = readable_type.get(src_type, src_type)
    dst_role = readable_type.get(dst_type, dst_type)
    src = shorten_endpoint(row.get("src_label", ""))
    dst = shorten_endpoint(row.get("dst_label", ""))
    target_role = row.get("target_role", "")
    lines = []
    lines.extend(wrap_endpoint_line("", src_role, src, target_role in {"src", "self"}, width))
    lines.extend(wrap_endpoint_line("----> ", dst_role, dst, target_role in {"dst", "self"}, width))
    return "\n".join(lines)


def label_row_gap(label):
    line_count = max(2, len(str(label).splitlines()))
    return 1.8 + max(0, line_count - 2) * 0.55


def y_positions_for_labels(labels):
    positions = []
    current = 0.0
    for label in labels:
        positions.append(current)
        current -= label_row_gap(label)
    return np.asarray(positions, dtype=float)


def panel_height_for_labels(labels, minimum):
    units = sum(label_row_gap(label) for label in labels)
    return max(minimum, units * 0.32 + 0.9)


def short_attack_label(attack_label):
    if not attack_label:
        return "benign"
    parts = [part.strip() for part in str(attack_label).split(";") if part.strip()]
    if len(parts) > 1:
        return "Multiple attacks"
    text = parts[0] if parts else str(attack_label)
    for attack_id, attack_name in ATTACK_NAMES.items():
        if text == attack_name or text.lower() == f"attack {attack_id}":
            return f"Attack {attack_id}"
    return text


def attack_legend_order():
    return [f"Attack {attack_id}" for attack_id in sorted(ATTACK_NAMES)] + ["Multiple attacks", "benign"]


def display_attack_label(label):
    if label == "benign":
        return "Benign"
    return label


def sorted_seen(values, preferred_order):
    seen = {value for value in values if value}
    ordered = [value for value in preferred_order if value in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def ranked_order(scores, nodes):
    # Match PIDSMaker's all-attacks ranking tie-break: sorted(zip(scores, nodes), reverse=True).
    return np.asarray(sorted(range(len(scores)), key=lambda i: (float(scores[i]), int(nodes[i])), reverse=True), dtype=int)


def add_endpoint_pair_legends(ax, operations, attack_labels, event_loc="upper right", attack_loc="lower right"):
    operation_handles = [
        plt.Rectangle((0, 0), 1, 1, color=OPERATION_COLORS.get(operation, OPERATION_COLORS["unknown"]), label=operation)
        for operation in operations
    ]
    attack_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=ATTACK_COLORS.get(label, ATTACK_COLORS["benign"]),
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=8,
            label=display_attack_label(label),
        )
        for label in attack_labels
    ]
    event_legend = ax.legend(handles=operation_handles, title="Event type", loc=event_loc, frameon=False, fontsize=9.4, title_fontsize=9.6)
    ax.add_artist(event_legend)
    ax.legend(handles=attack_handles, title="Attack label", loc=attack_loc, frameon=False, fontsize=9.4, title_fontsize=9.6)


def md_escape(text):
    return str(text).replace("|", "\\|")


def save_plot(fig, stem):
    plot_dir = OBS_DIR / "plots" / "from_saved_edge_losses_and_db"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"{DATASET.lower()}_produced_{stem}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)
    plt.close(fig)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(path)


def load_scores():
    score_blob = torch.load(SCORE_PATH)
    scores = np.asarray(score_blob["pred_scores"], dtype=float)
    labels = np.asarray(score_blob["y_truth"], dtype=int)
    nodes = np.asarray(score_blob["nodes"], dtype=int)
    node2attacks = {int(k): set(v) for k, v in score_blob["node2attacks"].items()}
    return scores, labels, nodes, node2attacks


def iter_edge_csvs(split_dir):
    for path in sorted(split_dir.glob("*.csv")):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield path, int(row["srcnode"]), int(row["dstnode"]), int(row["time"]), float(row["loss"])


def scan_threshold_and_responsible_edges(scored_nodes):
    val_losses = []
    for _, _, _, _, loss in iter_edge_csvs(VAL_DIR):
        val_losses.append(loss)
    threshold = float(np.max(np.asarray(val_losses, dtype=float)))

    scored_set = set(int(n) for n in scored_nodes)
    max_edges = {}
    incident_counts = defaultdict(int)
    test_losses = []
    scanned_edges = 0
    for path, src, dst, ts, loss in iter_edge_csvs(TEST_DIR):
        scanned_edges += 1
        test_losses.append(loss)
        window = path.name[:-4]
        if src in scored_set:
            incident_counts[src] += 1
            prev = max_edges.get(src)
            if prev is None or loss > prev["loss"]:
                max_edges[src] = {"loss": loss, "srcnode": src, "dstnode": dst, "time": ts, "role": "src", "window": window}
        if dst in scored_set and dst != src:
            incident_counts[dst] += 1
            prev = max_edges.get(dst)
            if prev is None or loss > prev["loss"]:
                max_edges[dst] = {"loss": loss, "srcnode": src, "dstnode": dst, "time": ts, "role": "dst", "window": window}
    return threshold, np.asarray(val_losses, dtype=float), np.asarray(test_losses, dtype=float), max_edges, dict(incident_counts), scanned_edges


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
                return {"rank": rank, "node": node, "score": float(scores[idx])}
    return None


def assign_node_groups(scores, labels, nodes, threshold, order, high_below_n):
    pred = scores > threshold
    node_groups = {}
    high_below = []
    for idx in order:
        node = int(nodes[idx])
        if labels[idx] == 0 and not pred[idx] and len(high_below) < high_below_n:
            high_below.append(node)
    high_below = set(high_below)

    for score, label, node, yhat in zip(scores, labels, nodes, pred):
        node = int(node)
        if label == 1 and yhat:
            node_group = "TP_above_threshold"
        elif label == 0 and yhat:
            node_group = "FP_above_threshold"
        elif label == 1 and not yhat:
            node_group = "FN_below_threshold"
        elif node in high_below:
            node_group = "TN_high_below_threshold"
        else:
            node_group = "TN_other"
        node_groups[node] = node_group
    return node_groups, pred, high_below


def quantiles(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"count": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def loss_quantiles(values):
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


def db_connection():
    try:
        import psycopg2
    except Exception as exc:
        return None, f"psycopg2 unavailable: {exc}"
    try:
        conn = psycopg2.connect(
            database=os.environ.get("DB_NAME", DB_NAME),
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
                execute_values(
                    cur,
                    query,
                    keys,
                    template="(%s::varchar, %s::varchar, %s::bigint)",
                    page_size=len(keys),
                )
                for src, dst, ts, operation, event_uuid in cur.fetchall():
                    key = (int(src), int(dst), int(ts))
                    if key not in event_meta and operation is not None:
                        event_meta[key] = {"operation": str(operation or ""), "event_uuid": str(event_uuid or "")}
                return event_meta
            except Exception:
                conn.rollback()

        # Small fallback only for environments lacking psycopg2.extras.
        for src, dst, ts in keys[:50]:
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


def attack_label(attacks):
    if not attacks:
        return ""
    return "; ".join(ATTACK_NAMES.get(int(a), f"attack {a}") for a in sorted(attacks))


def _same_cut(left, right):
    return abs(float(left) - float(right)) <= 1e-9


def _cut_name(value, threshold, coverage_score):
    if _same_cut(value, threshold):
        return "threshold"
    if _same_cut(value, coverage_score):
        return "coverage score"
    return f"{value:.3f}"


def _band_label(low, high, threshold, coverage_score):
    if math.isinf(low) and high == 1.0:
        return "< 1.000"
    if math.isinf(high):
        if _same_cut(low, coverage_score):
            return f">= coverage score ({coverage_score:.3f})"
        if _same_cut(low, threshold):
            return f">= threshold ({threshold:.3f})"
        return f">= {low:.3f}"
    if _same_cut(low, threshold) and _same_cut(high, coverage_score):
        return f"threshold to coverage ({threshold:.3f}-{coverage_score:.3f})"
    if _same_cut(low, coverage_score) and _same_cut(high, threshold):
        return f"coverage score to threshold ({coverage_score:.3f}-{threshold:.3f})"
    if _same_cut(high, threshold):
        return f"{_cut_name(low, threshold, coverage_score)} to threshold ({threshold:.3f})"
    if _same_cut(high, coverage_score):
        return f"{_cut_name(low, threshold, coverage_score)} to coverage score ({coverage_score:.3f})"
    if _same_cut(low, threshold):
        return f"threshold to {_cut_name(high, threshold, coverage_score)} ({threshold:.3f}-{high:.3f})"
    if _same_cut(low, coverage_score):
        return f"coverage score to {_cut_name(high, threshold, coverage_score)} ({coverage_score:.3f}-{high:.3f})"
    if low in {1.0, 2.0, 3.0, 4.0, 5.0} and high == low + 1.0:
        return f"{low:.3f} to {high - 0.001:.3f}"
    return f"{_cut_name(low, threshold, coverage_score)} to {_cut_name(high, threshold, coverage_score)} ({low:.3f}-{high:.3f})"


def score_band_definitions(threshold, coverage_score):
    cuts = [float("-inf"), 1.0, 2.0, 3.0, 4.0, 5.0]
    special = [float(threshold), float(coverage_score)]
    if max(special) > 6.0:
        cuts.append(6.0)
    cuts.extend(special)
    cuts.append(float("inf"))

    ordered = []
    for cut in sorted(cuts):
        if ordered and not math.isinf(cut) and not math.isinf(ordered[-1]) and _same_cut(cut, ordered[-1]):
            continue
        if ordered and cut == ordered[-1]:
            continue
        ordered.append(cut)

    bands = []
    for low, high in zip(ordered, ordered[1:]):
        if low >= high:
            continue
        bands.append((low, high, _band_label(low, high, threshold, coverage_score)))
    return bands


def score_band(score, threshold, coverage_score):
    for low, high, label in score_band_definitions(threshold, coverage_score):
        if score >= low and score < high:
            return label
    return score_band_definitions(threshold, coverage_score)[-1][2]


def build_rows(args, scores, labels, nodes, node2attacks, pred, node_groups, rank_by_node, score_by_node, max_edges, incident_counts, metadata, event_meta, threshold, coverage_score):
    selected_nodes = [n for n, group_key in node_groups.items() if group_key != "TN_other"]
    selected_nodes.sort(key=lambda n: rank_by_node[n])
    rows = []
    for node in selected_nodes:
        edge = max_edges.get(node, {})
        key = (int(edge.get("srcnode", -1)), int(edge.get("dstnode", -1)), int(edge.get("time", -1)))
        event = event_meta.get(key, {})
        src = int(edge.get("srcnode", -1))
        dst = int(edge.get("dstnode", -1))
        score = score_by_node[node]
        rows.append({
            "rank": rank_by_node[node],
            "node_id": node,
            "node_group": node_groups[node],
            "score": fmt(score, 9),
            "label": int(labels[np.where(nodes == node)[0][0]]),
            "pred_above_threshold": int(score > threshold),
            "attacks": attack_label(node2attacks.get(node, set())),
            "incident_edges": incident_counts.get(node, 0),
            "responsible_loss": fmt(edge.get("loss", float("nan")), 9),
            "score_minus_responsible_loss": fmt(score - edge.get("loss", float("nan")), 9),
            "operation": event.get("operation", ""),
            "event_uuid": event.get("event_uuid", ""),
            "time_raw": edge.get("time", ""),
            "time_eastern": ns_to_eastern(edge.get("time", 0)),
            "window": edge.get("window", ""),
            "target_role": edge.get("role", ""),
            "srcnode": src if src >= 0 else "",
            "dstnode": dst if dst >= 0 else "",
            "src_type": metadata.get(src, {}).get("node_type", ""),
            "dst_type": metadata.get(dst, {}).get("node_type", ""),
            "type_pair": f"{metadata.get(src, {}).get('node_type', 'unknown')}->{metadata.get(dst, {}).get('node_type', 'unknown')}",
            "node_label_text": node_label(node, metadata),
            "src_label": node_label(src, metadata) if src >= 0 else "",
            "dst_label": node_label(dst, metadata) if dst >= 0 else "",
            "score_band": score_band(score, threshold, coverage_score),
        })
    return rows


def node_group_summary(scores, labels, nodes, node_groups, max_edges, incident_counts, threshold):
    score_by_node = {int(n): float(s) for n, s in zip(nodes, scores)}
    label_by_node = {int(n): int(y) for n, y in zip(nodes, labels)}
    rows = []
    for group_key in NODE_GROUP_ORDER:
        group_nodes = [n for n, current_group in node_groups.items() if current_group == group_key]
        group_scores = [score_by_node[n] for n in group_nodes]
        losses = [max_edges[n]["loss"] for n in group_nodes if n in max_edges]
        incident = [incident_counts.get(n, 0) for n in group_nodes]
        score_stats = quantiles(group_scores)
        loss_stats = quantiles(losses)
        incident_stats = quantiles(incident)
        exact = sum(1 for n in group_nodes if n in max_edges and abs(score_by_node[n] - max_edges[n]["loss"]) <= 1e-6)
        rows.append({
            "node_group": group_key,
            "description": NODE_GROUP_LABELS[group_key],
            "nodes": len(group_nodes),
            "malicious": sum(label_by_node[n] for n in group_nodes),
            "benign": len(group_nodes) - sum(label_by_node[n] for n in group_nodes),
            "above_threshold": sum(1 for n in group_nodes if score_by_node[n] > threshold),
            "score_mean": fmt(score_stats["mean"], 9),
            "score_p50": fmt(score_stats["p50"], 9),
            "score_p90": fmt(score_stats["p90"], 9),
            "score_p99": fmt(score_stats["p99"], 9),
            "score_max": fmt(score_stats["max"], 9),
            "responsible_loss_p50": fmt(loss_stats["p50"], 9),
            "responsible_loss_max": fmt(loss_stats["max"], 9),
            "incident_edges_p50": fmt(incident_stats["p50"], 3),
            "incident_edges_p90": fmt(incident_stats["p90"], 3),
            "incident_edges_max": fmt(incident_stats["max"], 3),
            "score_matches_responsible_loss": exact,
        })
    return rows


def score_band_summary(scores, labels, threshold, coverage_score):
    counts = defaultdict(lambda: {"benign": 0, "malicious": 0, "total": 0})
    for score, label in zip(scores, labels):
        band = score_band(float(score), threshold, coverage_score)
        counts[band]["total"] += 1
        counts[band]["malicious" if label else "benign"] += 1
    band_order = [label for _low, _high, label in score_band_definitions(threshold, coverage_score)]
    rows = []
    for band in band_order:
        row = counts[band]
        total = row["total"]
        rows.append({
            "score_band": band,
            "total": total,
            "benign": row["benign"],
            "malicious": row["malicious"],
            "malicious_fraction": fmt(row["malicious"] / total if total else 0.0, 9),
        })
    return rows


def aggregate_selected(rows, key):
    counts = defaultdict(Counter)
    for row in rows:
        counts[row["node_group"]][row.get(key, "") or "unknown"] += 1
    out = []
    for group_key in NODE_GROUP_ORDER:
        total = sum(counts[group_key].values())
        for value, count in counts[group_key].most_common():
            out.append({"node_group": group_key, key: value, "count": count, "fraction": fmt(count / total if total else 0.0, 9)})
    return out


def aggregate_operation_type_pairs(rows):
    counts = defaultdict(Counter)
    for row in rows:
        operation = row.get("operation", "") or "unknown"
        type_pair = row.get("type_pair", "") or "unknown->unknown"
        counts[row["node_group"]][(operation, type_pair)] += 1

    out = []
    for group_key in NODE_GROUP_ORDER:
        total = sum(counts[group_key].values())
        ordered_items = sorted(
            counts[group_key].items(),
            key=lambda item: (
                -item[1],
                list(OPERATION_COLORS).index(item[0][0]) if item[0][0] in OPERATION_COLORS else len(OPERATION_COLORS),
                TYPE_PAIR_ORDER.index(item[0][1]) if item[0][1] in TYPE_PAIR_ORDER else len(TYPE_PAIR_ORDER),
                item[0],
            ),
        )
        for (operation, type_pair), count in ordered_items:
            out.append({
                "node_group": group_key,
                "operation": operation,
                "type_pair": type_pair,
                "count": count,
                "fraction": fmt(count / total if total else 0.0, 9),
            })
    return out


def aggregate_endpoint_pairs(rows):
    counts = Counter()
    group_totals = Counter()
    for row in rows:
        group_key = row["node_group"]
        attack = row.get("attacks", "") or "benign"
        key = (
            group_key,
            row.get("operation", ""),
            row.get("type_pair", ""),
            row.get("target_role", ""),
            row.get("src_label", ""),
            row.get("dst_label", ""),
            attack,
        )
        counts[key] += 1
        group_totals[group_key] += 1

    out = []
    for (group_key, operation, type_pair, target_role, src_label, dst_label, attack), count in counts.most_common():
        total = group_totals[group_key]
        out.append({
            "node_group": group_key,
            "node_group_label": NODE_GROUP_LABELS[group_key],
            "operation": operation,
            "type_pair": type_pair,
            "target_role": target_role,
            "src_label": src_label,
            "dst_label": dst_label,
            "attack": attack,
            "short_attack": short_attack_label(attack),
            "endpoint_pair": f"{src_label} -> {dst_label}",
            "short_endpoint_pair": compact_pair_label(src_label, dst_label),
            "count": count,
            "fraction": fmt(count / total if total else 0.0, 9),
        })
    return out


HIGHLIGHT_STYLES = {
    "tp_primary": {"face": "#fef3c7", "edge": "#f59e0b", "text": "#78350f"},
    "fp_primary": {"face": "#dbeafe", "edge": "#2563eb", "text": "#1e3a8a"},
    "fn_network": {"face": "#ffe4e6", "edge": "#e11d48", "text": "#9f1239"},
}


def concrete_pair_key(row):
    return (
        row.get("node_group", ""),
        row.get("operation", ""),
        row.get("type_pair", ""),
        row.get("src_label", ""),
        row.get("dst_label", ""),
        row.get("attack", ""),
    )


def build_endpoint_highlights(endpoint_pair_rows):
    counts = Counter()
    row_by_key = {}
    for row in endpoint_pair_rows:
        key = concrete_pair_key(row)
        counts[key] += int(row["count"])
        row_by_key[key] = row

    highlights = {}
    for group_key, style_name in (("TP_above_threshold", "tp_primary"), ("FP_above_threshold", "fp_primary")):
        group_keys = [key for key in counts if key[0] == group_key]
        if group_keys:
            top_key = max(group_keys, key=lambda key: (counts[key], key))
            highlights[top_key] = HIGHLIGHT_STYLES[style_name]

    fn_network_keys = [
        key for key in counts
        if key[0] == "FN_below_threshold" and key[1] == "EVENT_SENDTO" and key[2] == "subject->netflow"
    ]
    for key in sorted(fn_network_keys, key=lambda key: counts[key], reverse=True)[:5]:
        highlights[key] = HIGHLIGHT_STYLES["fn_network"]
    return highlights


def endpoint_row_highlight(row, highlights):
    return highlights.get(concrete_pair_key(row))


def highlighted_count(rows, highlights, style):
    return sum(int(row["count"]) for row in rows if endpoint_row_highlight(row, highlights) is style)


def draw_endpoint_bars(ax, y, counts, colors, plot_rows, highlights):
    for yi, count, color, row in zip(y, counts, colors, plot_rows):
        highlight = endpoint_row_highlight(row, highlights)
        ax.barh(
            yi,
            count,
            height=0.72,
            color=color,
            alpha=0.94,
            edgecolor=highlight["edge"] if highlight else "none",
            linewidth=1.45 if highlight else 0.0,
        )


def style_endpoint_ticklabels(ax, plot_rows, highlights):
    for tick_label, row in zip(ax.get_yticklabels(), plot_rows):
        highlight = endpoint_row_highlight(row, highlights)
        if not highlight:
            continue
        tick_label.set_color(highlight["text"])
        tick_label.set_fontweight("bold")
        tick_label.set_bbox({
            "boxstyle": "round,pad=0.16",
            "facecolor": highlight["face"],
            "edgecolor": highlight["edge"],
            "linewidth": 0.8,
            "alpha": 0.95,
        })


def plot_tp_fp_endpoint_pairs(endpoint_pair_rows):
    highlights = build_endpoint_highlights(endpoint_pair_rows)
    tp_rows = [row for row in endpoint_pair_rows if row["node_group"] == "TP_above_threshold"]
    fp_rows = [row for row in endpoint_pair_rows if row["node_group"] == "FP_above_threshold"]
    tp_rows = sorted(tp_rows, key=lambda row: int(row["count"]), reverse=True)
    fp_rows = sorted(fp_rows, key=lambda row: int(row["count"]), reverse=True)
    if ENDPOINT_TOP_N > 0:
        tp_rows = tp_rows[:ENDPOINT_TOP_N]
        fp_rows = fp_rows[:ENDPOINT_TOP_N]

    fp_plot_rows = fp_rows

    tp_labels = [plot_endpoint_pair_label(row, 58) for row in tp_rows]
    fp_labels = [plot_endpoint_pair_label(row, 58) for row in fp_plot_rows]
    tp_panel = panel_height_for_labels(tp_labels, 4.2)
    fp_panel = panel_height_for_labels(fp_labels, 9.5)
    fig_height = tp_panel + fp_panel + 2.6
    fig, axes = plt.subplots(2, 1, figsize=(16.8, fig_height), gridspec_kw={"height_ratios": [tp_panel, fp_panel]})
    fig.subplots_adjust(left=0.37, right=0.97, top=0.97, bottom=0.055, hspace=0.18)
    all_plot_rows = tp_rows + fp_plot_rows
    present_operations = sorted_seen(
        [row.get("operation", "unknown") or "unknown" for row in all_plot_rows],
        list(OPERATION_COLORS),
    )
    present_attack_labels = sorted_seen(
        [row.get("short_attack", "benign") or "benign" for row in all_plot_rows],
        attack_legend_order(),
    )
    tp_highlight_count = highlighted_count(tp_rows, highlights, HIGHLIGHT_STYLES["tp_primary"])
    fp_highlight_count = highlighted_count(fp_plot_rows, highlights, HIGHLIGHT_STYLES["fp_primary"])

    for ax, plot_rows, labels, title in (
        (axes[0], tp_rows, tp_labels, f"TP: malicious | highlighted main attack pair = {tp_highlight_count}/{sum(int(row['count']) for row in tp_rows)} nodes"),
        (axes[1], fp_plot_rows, fp_labels, f"FP: benign | highlighted dominant benign pair = {fp_highlight_count}/{sum(int(row['count']) for row in fp_plot_rows)} nodes"),
    ):
        counts = [int(row["count"]) for row in plot_rows]
        colors = [OPERATION_COLORS.get(row.get("operation", "unknown") or "unknown", OPERATION_COLORS["unknown"]) for row in plot_rows]
        y = y_positions_for_labels(labels)
        draw_endpoint_bars(ax, y, counts, colors, plot_rows, highlights)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11.2, linespacing=1.18)
        style_endpoint_ticklabels(ax, plot_rows, highlights)
        ax.set_xlabel("Node count", fontsize=11.0)
        ax.text(0.0, 1.025, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.6, fontweight="bold", color="#0f172a")
        ax.grid(axis="x", color="#e2e8f0", lw=0.8)
        ax.tick_params(axis="x", labelsize=10.2)
        max_count = max(counts) if counts else 1
        ax.set_xlim(0, max_count * 1.45)
        if len(y):
            ax.set_ylim(float(y.min()) - 1.0, float(y.max()) + 1.0)
        for yi, count, row in zip(y, counts, plot_rows):
            attack_label = row.get("short_attack", "benign") or "benign"
            count_x = count + max_count * 0.025
            dot_x = count + max_count * 0.105
            label_x = count + max_count * 0.135
            ax.text(count_x, yi, f"{count}", va="center", fontsize=9.8, color="#334155")
            ax.scatter([dot_x], [yi], s=46, color=ATTACK_COLORS.get(attack_label, ATTACK_COLORS["benign"]), edgecolors="white", linewidths=0.7, zorder=4)
            ax.text(label_x, yi, display_attack_label(attack_label), va="center", fontsize=9.6, color="#334155")

    add_endpoint_pair_legends(axes[0], present_operations, present_attack_labels, event_loc="upper right", attack_loc="lower right")
    save_plot(fig, "tp_fp_responsible_endpoint_pairs")


def type_pair_title(type_pair):
    readable = {"file": "file", "subject": "process", "netflow": "network", "unknown": "unknown"}
    if "->" not in type_pair:
        return type_pair or "unknown"
    src, dst = type_pair.split("->", 1)
    return f"{readable.get(src, src)} -> {readable.get(dst, dst)}"


def plot_fn_endpoint_pairs(endpoint_pair_rows):
    highlights = build_endpoint_highlights(endpoint_pair_rows)
    fn_rows = [row for row in endpoint_pair_rows if row["node_group"] == "FN_below_threshold"]
    type_counts = Counter(row.get("type_pair", "unknown->unknown") or "unknown->unknown" for row in fn_rows)
    preferred_types = ["subject->netflow", "file->subject", "netflow->subject"]
    type_sections = [(type_pair, f"FN: {type_pair_title(type_pair)}") for type_pair in preferred_types if type_pair in type_counts]
    type_sections.extend(
        (type_pair, f"FN: {type_pair_title(type_pair)}")
        for type_pair, _ in type_counts.most_common()
        if type_pair not in preferred_types
    )
    if not type_sections:
        type_sections = [("unknown->unknown", "FN: no missed malicious nodes")]
    rows_by_type = {
        type_pair: sorted([row for row in fn_rows if row["type_pair"] == type_pair], key=lambda row: int(row["count"]), reverse=True)
        for type_pair, _ in type_sections
    }
    if ENDPOINT_TOP_N > 0:
        rows_by_type = {type_pair: rows[:ENDPOINT_TOP_N] for type_pair, rows in rows_by_type.items()}
    labels_by_type = {
        type_pair: [plot_endpoint_pair_label(row, 58) for row in rows_by_type[type_pair]]
        for type_pair, _ in type_sections
    }
    height_ratios = [panel_height_for_labels(labels_by_type[type_pair], 2.2) for type_pair, _ in type_sections]
    fig_height = max(21.0, sum(height_ratios) + 4.5)
    fig, axes = plt.subplots(len(type_sections), 1, figsize=(18.2, fig_height), gridspec_kw={"height_ratios": height_ratios})
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.42, right=0.97, top=0.98, bottom=0.05, hspace=0.28)

    present_operations = sorted_seen(
        [row.get("operation", "unknown") or "unknown" for row in fn_rows],
        list(OPERATION_COLORS),
    )
    present_attack_labels = sorted_seen(
        [row.get("short_attack", "benign") or "benign" for row in fn_rows],
        attack_legend_order(),
    )
    fn_highlight_count = highlighted_count(fn_rows, highlights, HIGHLIGHT_STYLES["fn_network"])
    fn_total = sum(int(row["count"]) for row in fn_rows)

    for ax, (type_pair, section_title) in zip(axes, type_sections):
        plot_rows = rows_by_type[type_pair]
        labels = labels_by_type[type_pair]
        counts = [int(row["count"]) for row in plot_rows]
        colors = [OPERATION_COLORS.get(row.get("operation", "unknown") or "unknown", OPERATION_COLORS["unknown"]) for row in plot_rows]
        y = y_positions_for_labels(labels)
        draw_endpoint_bars(ax, y, counts, colors, plot_rows, highlights)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10.4, linespacing=1.18)
        style_endpoint_ticklabels(ax, plot_rows, highlights)
        ax.set_xlabel("Node count", fontsize=11.0)
        ax.grid(axis="x", color="#e2e8f0", lw=0.8)
        ax.tick_params(axis="x", labelsize=10.2)
        node_total = sum(counts)
        pair_total = len(plot_rows)
        ax.text(
            0.0,
            1.025,
            f"{section_title} ({node_total} nodes, {pair_total} pairs; highlighted top send pairs = {fn_highlight_count}/{fn_total} FN)" if type_pair == "subject->netflow" else f"{section_title} ({node_total} nodes, {pair_total} pairs)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.6,
            fontweight="bold",
            color="#0f172a",
        )
        max_count = max(counts) if counts else 1
        ax.set_xlim(0, max_count * 1.55)
        if len(y):
            ax.set_ylim(float(y.min()) - 1.0, float(y.max()) + 1.0)
        for yi, count, row in zip(y, counts, plot_rows):
            attack_label = row.get("short_attack", "benign") or "benign"
            count_x = count + max_count * 0.025
            dot_x = count + max_count * 0.125
            label_x = count + max_count * 0.165
            ax.text(count_x, yi, f"{count}", va="center", fontsize=9.6, color="#334155")
            ax.scatter([dot_x], [yi], s=44, color=ATTACK_COLORS.get(attack_label, ATTACK_COLORS["benign"]), edgecolors="white", linewidths=0.7, zorder=4)
            ax.text(
                label_x,
                yi,
                display_attack_label(attack_label),
                va="center",
                fontsize=9.6,
                color="#334155",
            )

    add_endpoint_pair_legends(axes[0], present_operations, present_attack_labels, event_loc="upper right", attack_loc="lower right")
    save_plot(fig, "fn_responsible_endpoint_pairs")


def plot_loss_distributions(val_losses, test_losses, threshold):
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    max_loss = float(max(np.max(val_losses), np.max(test_losses))) if len(val_losses) and len(test_losses) else float(threshold)
    bins = np.linspace(0.0, max(max_loss, threshold) * 1.02, 150)
    ax.hist(val_losses, bins=bins, color="#2563eb", alpha=0.58, label=f"Validation edges ({len(val_losses):,})")
    ax.hist(test_losses, bins=bins, color="#f97316", alpha=0.46, label=f"Test edges ({len(test_losses):,})")
    ax.axvline(threshold, color="#111827", lw=2.0, ls="--", label=f"VELOX threshold = {threshold:.6f}")
    ax.set_yscale("log")
    ax.set_xlabel("Per-edge cross-entropy loss")
    ax.set_ylabel("Edge count (log scale)")
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    save_plot(fig, "edge_loss_distribution_validation_vs_test")


def plot_top_node_responsible_edges(selected_rows, threshold):
    rows = sorted(selected_rows, key=lambda row: int(row["rank"]))[:20]
    labels = []
    scores = []
    losses = []
    colors = []
    for row in rows:
        labels.append(wrap(f"#{row['rank']} node {row['node_id']} | {row['node_label_text']}", 58))
        scores.append(float(row["score"]))
        losses.append(float(row["responsible_loss"]) if row["responsible_loss"] else 0.0)
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


def plot_node_group_boxplots(scores, nodes, node_groups, incident_counts):
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    data_incident = [[incident_counts.get(n, 0) for n, current_group in node_groups.items() if current_group == group_key] for group_key in NODE_GROUP_ORDER]
    labels = [NODE_GROUP_COMPACT_LABELS[group_key] for group_key in NODE_GROUP_ORDER]
    ax.boxplot(data_incident, labels=labels, showfliers=False, patch_artist=True)
    ax.set_ylabel("Incident edge count")
    ax.tick_params(axis="x", labelsize=8.8, pad=7)
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    save_plot(fig, "node_group_score_and_incident_edges")


def plot_mix(summary_rows, value_key, stem, top_n=6):
    groups = NODE_GROUP_ORDER[:-1]
    value_counts = Counter()
    for row in summary_rows:
        if row["node_group"] in groups:
            value_counts[row[value_key]] += int(row["count"])
    values = [value for value, _ in value_counts.most_common(top_n)]
    other_values = set(value_counts) - set(values)
    if other_values:
        values.append("Other")
    if "unknown" in values and len(values) > 1:
        values = [v for v in values if v != "unknown"] + ["unknown"]
    totals = {
        group_key: sum(int(row["count"]) for row in summary_rows if row["node_group"] == group_key)
        for group_key in groups
    }
    matrix = []
    for group_key in groups:
        total = totals[group_key]
        row = []
        for value in values:
            if value == "Other":
                count = sum(int(r["count"]) for r in summary_rows if r["node_group"] == group_key and r[value_key] in other_values)
            else:
                count = sum(int(r["count"]) for r in summary_rows if r["node_group"] == group_key and r[value_key] == value)
            row.append(count / total if total else 0.0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    bottom = np.zeros(len(groups))
    palette = ["#2563eb", "#e11d48", "#059669", "#f97316", "#7c3aed", "#0891b2", "#64748b"]
    x = np.arange(len(groups))
    for i, value in enumerate(values):
        vals = np.asarray([row[i] for row in matrix])
        ax.bar(x, vals, bottom=bottom, color=palette[i % len(palette)], label=value)
        bottom += vals
    for i, group_key in enumerate(groups):
        if totals[group_key] == 0:
            ax.text(i, 0.03, "no nodes", ha="center", va="bottom", fontsize=8.0, color="#64748b", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap(f"{NODE_GROUP_LABELS[group_key]}\nn={totals[group_key]}", 18) for group_key in groups], fontsize=8.5)
    ax.set_ylabel("Fraction within node group")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    save_plot(fig, stem)


def plot_operation_type_pair_mix(summary_rows):
    groups = NODE_GROUP_ORDER[:-1]
    operations = sorted_seen(
        [row["operation"] for row in summary_rows if row["node_group"] in groups],
        list(OPERATION_COLORS),
    )
    type_pairs = sorted_seen(
        [row["type_pair"] for row in summary_rows if row["node_group"] in groups],
        TYPE_PAIR_ORDER,
    )
    combos = [(operation, type_pair) for operation in operations for type_pair in type_pairs]
    combos = [
        combo for combo in combos
        if any(row["operation"] == combo[0] and row["type_pair"] == combo[1] and row["node_group"] in groups for row in summary_rows)
    ]

    totals = {
        group_key: sum(int(row["count"]) for row in summary_rows if row["node_group"] == group_key)
        for group_key in groups
    }
    counts = {
        (row["node_group"], row["operation"], row["type_pair"]): int(row["count"])
        for row in summary_rows
    }

    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for operation, type_pair in combos:
        vals = np.asarray([
            counts.get((group_key, operation, type_pair), 0) / totals[group_key] if totals[group_key] else 0.0
            for group_key in groups
        ])
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
            alpha=0.94,
        )
        bottom += vals

    ax.set_xticks(x)
    for i, group_key in enumerate(groups):
        if totals[group_key] == 0:
            ax.text(i, 0.03, "no nodes", ha="center", va="bottom", fontsize=8.0, color="#64748b", rotation=90)
    ax.set_xticklabels([wrap(f"{NODE_GROUP_LABELS[group_key]}\nn={totals[group_key]}", 18) for group_key in groups], fontsize=8.5)
    ax.set_ylabel("Fraction within node group")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#e2e8f0", lw=0.8)
    operation_handles = [
        Patch(facecolor=OPERATION_COLORS.get(operation, OPERATION_COLORS["unknown"]), edgecolor="none", label=operation)
        for operation in operations
    ]
    type_pair_handles = [
        Patch(facecolor="white", edgecolor="#0f172a", hatch=TYPE_PAIR_HATCHES.get(type_pair, "xx"), label=type_pair)
        for type_pair in type_pairs
    ]
    operation_legend = ax.legend(
        handles=operation_handles,
        title="Event type",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9.3,
        title_fontsize=9.6,
    )
    ax.add_artist(operation_legend)
    ax.legend(
        handles=type_pair_handles,
        title="Endpoint type",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9.3,
        title_fontsize=9.6,
    )
    save_plot(fig, "node_group_operation_type_pair_mix")


def write_docs(args, db_status, metrics, threshold, coverage, node_group_rows, band_rows, operation_rows, type_pair_rows, endpoint_pair_rows, selected_rows, exact_matches, scanned_edges):
    return
    tp = next(row for row in node_group_rows if row["node_group"] == "TP_above_threshold")
    fp = next(row for row in node_group_rows if row["node_group"] == "FP_above_threshold")
    fn = next(row for row in node_group_rows if row["node_group"] == "FN_below_threshold")
    high_benign = next(row for row in node_group_rows if row["node_group"] == "TN_high_below_threshold")
    top_fp_ops = [row for row in operation_rows if row["node_group"] == "FP_above_threshold"][:3]
    top_fn_ops = [row for row in operation_rows if row["node_group"] == "FN_below_threshold"][:3]
    fp_op_text = ", ".join(f"`{row['operation']}` {row['count']}" for row in top_fp_ops) or "N/A"
    fn_op_text = ", ".join(f"`{row['operation']}` {row['count']}" for row in top_fn_ops) or "N/A"
    tp_endpoint_pairs = [row for row in endpoint_pair_rows if row["node_group"] == "TP_above_threshold"]
    fp_endpoint_pairs = [row for row in endpoint_pair_rows if row["node_group"] == "FP_above_threshold"]
    fn_endpoint_pairs = [row for row in endpoint_pair_rows if row["node_group"] == "FN_below_threshold"]

    def pair_key(row):
        return (row["operation"], row["type_pair"], row["src_label"], row["dst_label"], row["attack"])

    def distinct_pair_count(rows):
        return len({pair_key(row) for row in rows})

    def target_role_text(row):
        role = row.get("target_role", "")
        if role == "src":
            return "source flagged"
        if role == "dst":
            return "destination flagged"
        if role == "self":
            return "source/destination flagged"
        return "flagged role unknown"

    fn_type_counts = defaultdict(lambda: {"nodes": 0, "pair_keys": set(), "plot_rows": 0})
    for row in fn_endpoint_pairs:
        fn_type_counts[row["type_pair"]]["nodes"] += int(row["count"])
        fn_type_counts[row["type_pair"]]["pair_keys"].add(pair_key(row))
        fn_type_counts[row["type_pair"]]["plot_rows"] += 1
    top_tp_pair = max(tp_endpoint_pairs, key=lambda row: int(row["count"])) if tp_endpoint_pairs else None
    top_fp_pair = max(fp_endpoint_pairs, key=lambda row: int(row["count"])) if fp_endpoint_pairs else None
    top_tp_pair_text = f"`{top_tp_pair['short_endpoint_pair']}` ({top_tp_pair['count']} nodes, {top_tp_pair['short_attack']}, {target_role_text(top_tp_pair)})" if top_tp_pair else "N/A"
    top_fp_pair_text = f"`{top_fp_pair['short_endpoint_pair']}` ({top_fp_pair['count']} nodes, {target_role_text(top_fp_pair)})" if top_fp_pair else "N/A"
    highlights = build_endpoint_highlights(endpoint_pair_rows)
    tp_highlight_count = highlighted_count(tp_endpoint_pairs, highlights, HIGHLIGHT_STYLES["tp_primary"])
    fp_highlight_count = highlighted_count(fp_endpoint_pairs, highlights, HIGHLIGHT_STYLES["fp_primary"])
    fn_highlight_count = highlighted_count(fn_endpoint_pairs, highlights, HIGHLIGHT_STYLES["fn_network"])

    node_group_table = "\n".join(
        f"| {row['description']} | `{int(row['nodes']):,}` | `{row['score_p50']}` | `{row['score_p90']}` | `{row['score_max']}` | `{row['incident_edges_p50']}` | `{row['incident_edges_p90']}` |"
        for row in node_group_rows
    )
    band_table = "\n".join(
        f"| {row['score_band']} | `{int(row['total']):,}` | `{int(row['benign']):,}` | `{int(row['malicious']):,}` | `{float(row['malicious_fraction']):.6f}` |"
        for row in band_rows
    )
    example_rows = selected_rows[:8]
    example_table = "\n".join(
        f"| `{row['rank']}` | {NODE_GROUP_LABELS[row['node_group']]} | `{row['score']}` | `{row['operation']}` | {md_escape(row['src_label'])} -> {md_escape(row['dst_label'])} |"
        for row in example_rows
    )

    readme = f"""# THEIA_E3 VELOX Score-Overlap Explanation

## Question

Why do benign and malicious THEIA_E3 VELOX node-score distributions overlap?

## Scope

| Field | Value |
|---|---|
| Dataset | `{DATASET}` |
| System | `velox` |
| Mode | saved `--tuned --from_weights --restart_from_scratch` artifacts |
| Score file | `{rel(SCORE_PATH)}` |
| Test edge losses | `{rel(TEST_DIR)}` |
| Validation edge losses | `{rel(VAL_DIR)}` |
| DB enrichment | `{db_status}` |
| Test edges scanned | `{scanned_edges:,}` |

## Main Finding

The overlap is expected from VELOX's scoring mechanism. VELOX scores edges first and assigns each endpoint node the maximum incident edge loss. In this run, `{exact_matches:,}` scored nodes have a saved node score matching their responsible max edge loss within `1e-6`.

At the fixed threshold `{threshold:.9f}`, THEIA_E3 has `TP={metrics['tp']}`, `FP={metrics['fp']}`, and `FN={metrics['fn']}`. The threshold-positive false positives are not random: their responsible edges are mostly the same kind of high-loss endpoint/event pattern as the true positives. Top FP operations: {fp_op_text}. False negatives stay below threshold because their maximum incident edge never exceeds the validation max; top FN responsible operations: {fn_op_text}.

`file->subject EVENT_EXECUTE` reflects PIDSMaker/DARPA graph orientation; `subject` means process. Read it as an execution relationship between a file endpoint and a process endpoint, not as a literal claim that a file actively initiated execution.

The `{NODE_GROUP_LABELS["TN_high_below_threshold"]}` group is an analysis subset, not a new metric. It keeps the plots readable and focuses on the benign nodes closest to becoming false positives. The size comes from the regeneration flag `--high-below-n {args.high_below_n:,}` and can be changed if a wider near-threshold slice is needed.

The endpoint-type plot intentionally collapses concrete paths. Above threshold, TP and FP nodes are all `EVENT_EXECUTE file->subject`, but they are not the same exact file/process pair. TP nodes use `{distinct_pair_count(tp_endpoint_pairs)}` distinct responsible endpoint pairs, shown as `{len(tp_endpoint_pairs)}` source/destination-flagged plot rows. FP nodes use `{distinct_pair_count(fp_endpoint_pairs)}` distinct benign endpoint pairs, shown as `{len(fp_endpoint_pairs)}` source/destination-flagged plot rows. The most common TP plotted row is {top_tp_pair_text}; the most common FP plotted row is {top_fp_pair_text}.

The FN group is different. Its responsible endpoint pairs cover `{distinct_pair_count(fn_endpoint_pairs)}` concrete pairs: `subject->netflow` has `{fn_type_counts['subject->netflow']['nodes']}` nodes across `{len(fn_type_counts['subject->netflow']['pair_keys'])}` pairs, `file->subject` has `{fn_type_counts['file->subject']['nodes']}` nodes across `{len(fn_type_counts['file->subject']['pair_keys'])}` pairs, and `netflow->subject` has `{fn_type_counts['netflow->subject']['nodes']}` nodes across `{len(fn_type_counts['netflow->subject']['pair_keys'])}` pairs. This is why the FN bar is not all `EVENT_EXECUTE`.

In the concrete endpoint-pair plots, bar color encodes the event type and the dot/text tag encodes the attack label or benign status. The underlined y-axis role word (`file`, `process`, or `netflow`) marks which endpoint received the VELOX score for that row. Highlighted y-axis labels mark the dominant visual patterns: TP's primary attack pair accounts for `{tp_highlight_count}/{tp['nodes']}` nodes, FP's dominant benign pair accounts for `{fp_highlight_count}/{fp['nodes']}` nodes, and FN's highlighted process-to-network send pairs account for `{fn_highlight_count}/{fn['nodes']}` missed nodes.

The coverage point remains rank `{coverage['rank']}`, node `{coverage['node']}`, score `{coverage['score']:.9f}`. That explains high ADP despite overlapping node-score distributions: one high-loss edge from each configured attack appears early, while many other malicious nodes have routine-looking max edges.

## Node Group Summary

| Node group | Nodes | Median score | p90 score | Max score | Median incident edges | p90 incident edges |
|---|---:|---:|---:|---:|---:|---:|
{node_group_table}

## Score Bands

| Score band | Total | Benign | Malicious | Malicious fraction |
|---|---:|---:|---:|---:|
{band_table}

## Top Responsible-Edge Examples

| Rank | Node group | Score | Operation | Responsible edge |
|---:|---|---:|---|---|
{example_table}

## Plots

| Plot | PNG | SVG | Purpose |
|---|---|---|---|
| Node-group incident edges | `plots/node_group_score_and_incident_edges.png` | `plots/node_group_score_and_incident_edges.svg` | Shows incident-edge-count distributions by node group. |
| Operation mix | `plots/node_group_operation_mix.png` | `plots/node_group_operation_mix.svg` | Shows which event operations explain selected TP/FP/FN/high-benign node groups. |
| Endpoint type-pair mix | `plots/node_group_type_pair_mix.png` | `plots/node_group_type_pair_mix.svg` | Shows whether responsible edges are file->subject, file->file, etc. |
| Operation + endpoint type-pair mix | `plots/node_group_operation_type_pair_mix.png` | `plots/node_group_operation_type_pair_mix.svg` | Combines the previous two summaries in one view: color is event type, hatch is source/destination endpoint type. |
| TP/FP responsible endpoint pairs | `plots/tp_fp_responsible_endpoint_pairs.png` | `plots/tp_fp_responsible_endpoint_pairs.svg` | Expands the above-threshold `EVENT_EXECUTE file->subject` result into concrete file/process pairs; highlighted y-axis labels mark the primary TP and FP patterns. |
| FN responsible endpoint pairs | `plots/fn_responsible_endpoint_pairs.png` | `plots/fn_responsible_endpoint_pairs.svg` | Expands the missed malicious nodes into concrete process/network and file/process responsible pairs; highlighted y-axis labels mark the dominant process-to-network send patterns. |
| Internal feature ablation | `plots/internal_feature_ablation.png` | `plots/internal_feature_ablation.svg` | Diverging bar plot of Captum/direct ablation over all TP/FP/FN responsible edges plus the top 500 near-threshold benign responsible edges, summarized by node group, event type, and endpoint type using the saved post-epoch model state. |

## Data Tables

| File | Purpose |
|---|---|
| `tables/node_group_summary.csv` | Counts and quantiles for TP, FP, FN, high benign below threshold, and remaining benign nodes. |
| `tables/score_band_summary.csv` | Benign/malicious counts across score bands. |
| `tables/selected_responsible_edges.csv` | Responsible-edge rows for all TP/FP/FN nodes plus top `{args.high_below_n:,}` benign nodes below threshold. |
| `tables/node_group_operation_summary.csv` | Operation counts within selected node groups. |
| `tables/node_group_type_pair_summary.csv` | Endpoint type-pair counts within selected node groups. |
| `tables/node_group_operation_type_pair_summary.csv` | Joint operation and endpoint type-pair counts within selected node groups. |
| `tables/node_group_source_label_summary.csv` | Top source endpoint labels within selected node groups. |
| `tables/node_group_destination_label_summary.csv` | Top destination endpoint labels within selected node groups. |
| `tables/responsible_endpoint_pair_summary.csv` | Concrete responsible endpoint-pair counts by node group, event type, flagged endpoint role, and attack label. |
| `tables/internal_edge_logits.csv` | Edge-type logits/probabilities for all TP/FP/FN responsible edges plus the top 500 near-threshold benign responsible edges using the saved post-epoch model state. |
| `tables/internal_feature_ablation.csv` | Per-edge Captum `FeatureAblation` and direct zero-baseline ablation by feature group. |
| `tables/internal_feature_ablation_summary.csv` | Median/mean/quantile loss drops by node group, event type, endpoint type, and feature group. |

## Scientific Interpretation

VELOX learned an edge-type expectation model, not a complete malicious-node separator. The distribution overlaps for three reasons visible in these summaries: benign endpoints can share high-loss attack-adjacent edges, malicious nodes can have only routine low-loss incident edges, and max aggregation gives high-degree benign nodes more chances to inherit a rare edge loss.

Captum `0.7.0` is installed in the current `pids` runtime. The post-epoch model state was regenerated with `save_theia_velox_post_epoch_model.py`, then used for internal attribution. Computed losses for all 605 TP/FP/FN responsible edges plus the top 500 near-threshold benign responsible edges match the saved edge-loss CSV values within `2e-6`, so these logits/ablations are exact for those evaluated edges.

The internal attribution plot shows median loss change as bars. Bars to the right mean the feature group drives high loss; bars to the left mean the feature group helps predict the true edge type. The high-loss above-threshold `EVENT_EXECUTE file->subject` groups are driven mostly by source endpoint type, while the below-threshold process/network sends have small or negative loss changes. The `Benign` row is the top 500 near-threshold benign slice from the selected responsible-edge table, not all benign nodes.

## Regeneration

```shell
docker exec pidsmaker-pids bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate pids && PYTHONPATH=/home/src python /home/artifacts/observations/shared_scripts/generate_theia_score_overlap_observation.py --high-below-n {args.high_below_n}'
```

Regenerate internal attribution with the activated `pids` environment:

```shell
docker exec pidsmaker-pids bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate pids && PYTHONPATH=/home/src python /home/artifacts/observations/shared_scripts/generate_theia_internal_attribution.py'
```

Regenerate the post-epoch model state without rerunning evaluation:

```shell
docker exec pidsmaker-pids bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate pids && PYTHONPATH=/home/src python /home/artifacts/observations/shared_scripts/save_theia_velox_post_epoch_model.py'
```
"""
    print(OBS_DIR)


def generate(args, run_key):
    global ENDPOINT_TOP_N
    configure_run(run_key)
    ENDPOINT_TOP_N = args.endpoint_top_n
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    (OBS_DIR / "plots").mkdir(exist_ok=True)
    (OBS_DIR / "tables").mkdir(exist_ok=True)
    NODE_GROUP_LABELS["TN_high_below_threshold"] = f"Top {args.high_below_n:,} benign nodes just below threshold"
    setup_style()

    scores, labels, nodes, node2attacks = load_scores()
    order = ranked_order(scores, nodes)
    rank_by_node = {int(node): rank for rank, node in enumerate(nodes[order], start=1)}
    score_by_node = {int(node): float(score) for node, score in zip(nodes, scores)}
    threshold, val_losses, test_losses, max_edges, incident_counts, scanned_edges = scan_threshold_and_responsible_edges(nodes)
    coverage = find_coverage(scores, labels, nodes, node2attacks, order)
    coverage_score = coverage["score"] if coverage else threshold
    node_groups, pred, high_below = assign_node_groups(scores, labels, nodes, threshold, order, args.high_below_n)

    selected_nodes = [n for n, group_key in node_groups.items() if group_key != "TN_other"]
    selected_edges = [max_edges[n] for n in selected_nodes if n in max_edges]
    endpoint_nodes = set(selected_nodes)
    for edge in selected_edges:
        endpoint_nodes.add(int(edge["srcnode"]))
        endpoint_nodes.add(int(edge["dstnode"]))

    conn, db_status = db_connection()
    metadata = fetch_node_metadata(conn, endpoint_nodes)
    event_meta = fetch_event_metadata(conn, selected_edges)
    if conn is not None:
        conn.close()

    selected_rows = build_rows(args, scores, labels, nodes, node2attacks, pred, node_groups, rank_by_node, score_by_node, max_edges, incident_counts, metadata, event_meta, threshold, coverage_score)
    node_group_rows = node_group_summary(scores, labels, nodes, node_groups, max_edges, incident_counts, threshold)
    band_rows = score_band_summary(scores, labels, threshold, coverage_score)
    operation_rows = aggregate_selected(selected_rows, "operation")
    type_pair_rows = aggregate_selected(selected_rows, "type_pair")
    operation_type_pair_rows = aggregate_operation_type_pairs(selected_rows)
    source_rows = aggregate_selected(selected_rows, "src_label")
    destination_rows = aggregate_selected(selected_rows, "dst_label")
    endpoint_pair_rows = aggregate_endpoint_pairs(selected_rows)

    metrics = {
        "tp": int(((pred == 1) & (labels == 1)).sum()),
        "fp": int(((pred == 1) & (labels == 0)).sum()),
        "fn": int(((pred == 0) & (labels == 1)).sum()),
        "tn": int(((pred == 0) & (labels == 0)).sum()),
    }
    exact_matches = sum(1 for node, score in score_by_node.items() if node in max_edges and abs(score - max_edges[node]["loss"]) <= 1e-6)

    write_csv(OBS_DIR / "tables/node_group_summary.csv", list(node_group_rows[0]), node_group_rows)
    write_csv(OBS_DIR / "tables/score_band_summary.csv", list(band_rows[0]), band_rows)
    loss_rows = []
    for split, values in (("val", val_losses), ("test", test_losses)):
        stats = loss_quantiles(values)
        loss_rows.append({"split": split, **{key: fmt(value, 9) if key != "count" else value for key, value in stats.items()}})
    write_csv(OBS_DIR / "tables/loss_distribution_summary.csv", ["split", "count", "p50", "p90", "p95", "p99", "p999", "max"], loss_rows)
    selected_fields = [
        "rank", "node_id", "node_group", "score", "label", "pred_above_threshold", "attacks", "incident_edges",
        "responsible_loss", "score_minus_responsible_loss", "operation", "event_uuid", "time_raw", "time_eastern", "window",
        "target_role", "srcnode", "dstnode", "src_type", "dst_type", "type_pair", "node_label_text", "src_label", "dst_label", "score_band",
    ]
    write_csv(OBS_DIR / "tables/selected_responsible_edges.csv", selected_fields, selected_rows)
    write_csv(OBS_DIR / "tables/node_group_operation_summary.csv", ["node_group", "operation", "count", "fraction"], operation_rows)
    write_csv(OBS_DIR / "tables/node_group_type_pair_summary.csv", ["node_group", "type_pair", "count", "fraction"], type_pair_rows)
    write_csv(OBS_DIR / "tables/node_group_operation_type_pair_summary.csv", ["node_group", "operation", "type_pair", "count", "fraction"], operation_type_pair_rows)
    write_csv(OBS_DIR / "tables/node_group_source_label_summary.csv", ["node_group", "src_label", "count", "fraction"], source_rows)
    write_csv(OBS_DIR / "tables/node_group_destination_label_summary.csv", ["node_group", "dst_label", "count", "fraction"], destination_rows)
    write_csv(
        OBS_DIR / "tables/responsible_endpoint_pair_summary.csv",
        ["node_group", "node_group_label", "operation", "type_pair", "target_role", "src_label", "dst_label", "attack", "short_attack", "endpoint_pair", "short_endpoint_pair", "count", "fraction"],
        endpoint_pair_rows,
    )
    plot_loss_distributions(val_losses, test_losses, threshold)
    plot_top_node_responsible_edges(selected_rows, threshold)
    plot_node_group_boxplots(scores, nodes, node_groups, incident_counts)
    plot_mix(operation_rows, "operation", "node_group_operation_mix", top_n=10)
    plot_mix(type_pair_rows, "type_pair", "node_group_type_pair_mix")
    plot_operation_type_pair_mix(operation_type_pair_rows)
    plot_tp_fp_endpoint_pairs(endpoint_pair_rows)
    plot_fn_endpoint_pairs(endpoint_pair_rows)

    write_docs(args, db_status, metrics, threshold, coverage, node_group_rows, band_rows, operation_rows, type_pair_rows, endpoint_pair_rows, selected_rows, exact_matches, scanned_edges)
    print(OBS_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="THEIA_E3", choices=list(RUNS) + ["all"])
    parser.add_argument("--high-below-n", type=int, default=500)
    parser.add_argument("--endpoint-top-n", type=int, default=40, help="Limit endpoint-pair plot rows per node group/type pair; 0 keeps all rows.")
    args = parser.parse_args()
    run_keys = [key for key in RUNS if key != "THEIA_E3"] if args.dataset == "all" else [args.dataset]
    for run_key in run_keys:
        generate(args, run_key)


if __name__ == "__main__":
    main()
