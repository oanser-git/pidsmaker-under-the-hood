import argparse
import csv
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import FeatureAblation


from poc_paths import add_pidsmaker_src, artifacts_root  # noqa: E402


add_pidsmaker_src()


RUNS = {
    "THEIA_E3": {
        "obs_id": "2026-07-01_theia_e3_velox",
        "dataset": "THEIA_E3",
        "model_slug": "theia_e3",
        "edge_dir_candidates": [
            "499b3a9d2ad282bab5e39ca1a680d479cb2804fbb35d8759f2e3e84fbdd30d1a",
            "1c3c32bde5673f589fc13db31945a83a617cd549e7b05c07a72d8a56211215ec",
        ],
    },
    "CADETS_E3": {
        "obs_id": "2026-07-01_cadets_e3_velox",
        "dataset": "CADETS_E3",
        "model_slug": "cadets_e3",
        "edge_dir_candidates": ["eae5106c3e9b47e33137c97ef008f51a1ef2c69d85c7398510d91586b7bda848"],
    },
    "optc_h201": {
        "obs_id": "2026-07-01_optc_h201_velox",
        "dataset": "optc_h201",
        "model_slug": "optc_h201",
        "edge_dir_candidates": ["6a3db7ae1e2232a0d7450f8d450b249ef5a448a63ebc4ef5a4bc633db1fb755d"],
    },
    "optc_h501": {
        "obs_id": "2026-07-01_optc_h501_velox",
        "dataset": "optc_h501",
        "model_slug": "optc_h501",
        "edge_dir_candidates": ["fcc39f462d34d8d0a2b9dd02365aa38717cb247d6d5d9d4a208cc961270d3d2f"],
    },
    "optc_h051": {
        "obs_id": "2026-07-01_optc_h051_velox",
        "dataset": "optc_h051",
        "model_slug": "optc_h051",
        "edge_dir_candidates": ["7f75250cc14901cf70bde015a0bab8f5d5ecc3513462f5d9e7d16ae8b4eb157b"],
    },
}
OBS_ID = RUNS["THEIA_E3"]["obs_id"]
DATASET = RUNS["THEIA_E3"]["dataset"]
MODEL_SLUG = RUNS["THEIA_E3"]["model_slug"]
EDGE_DIR_CANDIDATES = list(RUNS["THEIA_E3"]["edge_dir_candidates"])
DARPA_EDGE_TYPE_NAMES = {
    0: "EVENT_CONNECT",
    1: "EVENT_EXECUTE",
    2: "EVENT_OPEN",
    3: "EVENT_READ",
    4: "EVENT_RECVFROM",
    5: "EVENT_RECVMSG",
    6: "EVENT_SENDMSG",
    7: "EVENT_SENDTO",
    8: "EVENT_WRITE",
    9: "EVENT_CLONE",
}
OPTC_EDGE_TYPE_NAMES = {
    0: "OPEN",
    1: "READ",
    2: "CREATE",
    3: "MESSAGE",
    4: "MODIFY",
    5: "START",
    6: "RENAME",
    7: "DELETE",
    8: "TERMINATE",
    9: "WRITE",
}
EDGE_TYPE_NAMES = dict(DARPA_EDGE_TYPE_NAMES)
GROUPS = {
    "src_embedding": (0, 64),
    "src_type": (64, 67),
    "dst_embedding": (67, 131),
    "dst_type": (131, 134),
}
GROUP_LABELS = {
    "src_embedding": "source embedding",
    "src_type": "source type",
    "dst_embedding": "destination embedding",
    "dst_type": "destination type",
}
GROUP_COLORS = {
    "src_embedding": "#2563eb",
    "src_type": "#f97316",
    "dst_embedding": "#14b8a6",
    "dst_type": "#a855f7",
}
LOGIT_FIELDS = [
    "case",
    "rank",
    "node_id",
    "node_group",
    "operation_from_db",
    "type_pair",
    "target_class",
    "target_edge_type",
    "predicted_class",
    "predicted_edge_type",
    "true_probability",
    "predicted_probability",
    "computed_loss",
    "saved_loss",
    "loss_error",
    "edge_dir_id",
    "weights_path",
    "feature_layout",
    "edge_index_in_window",
    "edge_candidate_count",
    "edge_count_same_endpoints",
    "src_label",
    "dst_label",
]
BENIGN_BELOW_ABLATION_LIMIT = 500
NODE_GROUP_ORDER = [
    "TP_above_threshold",
    "FP_above_threshold",
    "FN_below_threshold",
    "TN_high_below_threshold",
]
NODE_GROUP_SHORT = {
    "TP_above_threshold": "TP",
    "FP_above_threshold": "FP",
    "FN_below_threshold": "FN",
    "TN_high_below_threshold": "TN (benign)\nnear threshold",
}


ROOT = artifacts_root()
OBS_DIR = ROOT / "observations" / OBS_ID
SELECTED_PATH = OBS_DIR / "tables/selected_responsible_edges.csv"
POST_EPOCH_WEIGHTS_PATH = OBS_DIR / f"models/{MODEL_SLUG}_velox_post_epoch_model_epoch_0/state_dict.pkl"
AUTHOR_WEIGHTS_PATH = Path(f"/home/weights/encoder/{DATASET}.pkl") if Path(f"/home/weights/encoder/{DATASET}.pkl").exists() else Path(f"weights/encoder/{DATASET}.pkl")
WEIGHTS_PATH = POST_EPOCH_WEIGHTS_PATH if POST_EPOCH_WEIGHTS_PATH.exists() else AUTHOR_WEIGHTS_PATH


def configure_run(run_key):
    global OBS_ID, DATASET, MODEL_SLUG, EDGE_DIR_CANDIDATES, EDGE_TYPE_NAMES
    global OBS_DIR, SELECTED_PATH, POST_EPOCH_WEIGHTS_PATH, AUTHOR_WEIGHTS_PATH, WEIGHTS_PATH
    run = RUNS[run_key]
    OBS_ID = run["obs_id"]
    DATASET = run["dataset"]
    MODEL_SLUG = run["model_slug"]
    EDGE_DIR_CANDIDATES = list(run["edge_dir_candidates"])
    EDGE_TYPE_NAMES = dict(OPTC_EDGE_TYPE_NAMES if DATASET.startswith("optc_") else DARPA_EDGE_TYPE_NAMES)
    OBS_DIR = ROOT / "observations" / OBS_ID
    SELECTED_PATH = OBS_DIR / "tables/selected_responsible_edges.csv"
    POST_EPOCH_WEIGHTS_PATH = OBS_DIR / f"models/{MODEL_SLUG}_velox_post_epoch_model_epoch_0/state_dict.pkl"
    AUTHOR_WEIGHTS_PATH = Path(f"/home/weights/encoder/{DATASET}.pkl") if Path(f"/home/weights/encoder/{DATASET}.pkl").exists() else Path(f"weights/encoder/{DATASET}.pkl")
    WEIGHTS_PATH = POST_EPOCH_WEIGHTS_PATH if POST_EPOCH_WEIGHTS_PATH.exists() else AUTHOR_WEIGHTS_PATH


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(path)


def fmt(value, digits=9):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{float(value):.{digits}f}"


class VeloxEdgeModel(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.state = state_dict
        self.node_feature_dim = int(self.state["encoder.lin1.weight"].shape[1])
        self.embedding_dim = self.node_feature_dim - 3

    def logits_from_pair(self, x_src, x_dst):
        h_src = F.linear(x_src, self.state["encoder.lin1.weight"], self.state["encoder.lin1.bias"])
        h_dst = F.linear(x_dst, self.state["encoder.lin1.weight"], self.state["encoder.lin1.bias"])
        z_src = F.linear(h_src, self.state["decoders.0.objective.decoder.lin_src.weight"], self.state["decoders.0.objective.decoder.lin_src.bias"])
        z_dst = F.linear(h_dst, self.state["decoders.0.objective.decoder.lin_dst.weight"], self.state["decoders.0.objective.decoder.lin_dst.bias"])
        h = torch.cat([z_src, z_dst], dim=-1)
        h = F.linear(h, self.state["decoders.0.objective.decoder.mlp.0.weight"], self.state["decoders.0.objective.decoder.mlp.0.bias"])
        h = F.relu(h)
        return F.linear(h, self.state["decoders.0.objective.decoder.mlp.2.weight"], self.state["decoders.0.objective.decoder.mlp.2.bias"])

    def forward(self, x_pair, target_classes):
        x_src = x_pair[:, :self.node_feature_dim]
        x_dst = x_pair[:, self.node_feature_dim:]
        logits = self.logits_from_pair(x_src, x_dst)
        return F.cross_entropy(logits, target_classes, reduction="none")


def set_feature_groups(embedding_dim):
    global GROUPS
    node_dim = embedding_dim + 3
    GROUPS = {
        "src_embedding": (0, embedding_dim),
        "src_type": (embedding_dim, node_dim),
        "dst_embedding": (node_dim, node_dim + embedding_dim),
        "dst_type": (node_dim + embedding_dim, node_dim * 2),
    }


def parse_temporal_data(data):
    msg = data.msg
    emb_dim = (msg.size(1) - 16) // 2
    src_type = msg[:, 0:3]
    src_emb = msg[:, 3:3 + emb_dim]
    edge_type_start = 3 + emb_dim
    edge_type = msg[:, edge_type_start:edge_type_start + 10]
    dst_type_start = edge_type_start + 10
    dst_type = msg[:, dst_type_start:dst_type_start + 3]
    dst_emb = msg[:, dst_type_start + 3:dst_type_start + 3 + emb_dim]
    variants = {
        "emb_type": (torch.cat([src_emb, src_type], dim=-1), torch.cat([dst_emb, dst_type], dim=-1)),
        "type_emb": (torch.cat([src_type, src_emb], dim=-1), torch.cat([dst_type, dst_emb], dim=-1)),
        "emb_type_srcdst_swap": (torch.cat([dst_emb, dst_type], dim=-1), torch.cat([src_emb, src_type], dim=-1)),
        "type_emb_srcdst_swap": (torch.cat([dst_type, dst_emb], dim=-1), torch.cat([src_type, src_emb], dim=-1)),
    }
    return variants, edge_type


def parse_file_time(value):
    value = value.strip()
    if "." in value:
        base, frac = value.split(".", 1)
        frac = "".join(ch for ch in frac if ch.isdigit())[:6].ljust(6, "0")
        return datetime.strptime(f"{base}.{frac}", "%Y-%m-%d %H:%M:%S.%f")
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def parse_row_display_time(row):
    value = row.get("time_eastern", "")
    if not value:
        return None
    value = value.replace(" EDT", "").replace(" EST", "")
    return parse_file_time(value)


def parse_row_raw_time(row):
    raw = row.get("time_raw", "")
    if not raw:
        return None
    return datetime.fromtimestamp(int(raw) / 1_000_000_000, tz=timezone.utc).replace(tzinfo=None)


def find_window_path(edge_dir_id, row):
    base = ROOT / f"featurization/{DATASET}/embed_edges/{edge_dir_id}/edge_embeds/test"
    exact = base / (row["window"] + ".TemporalData.simple")
    if exact.exists():
        return exact
    if "~" in row.get("window", ""):
        start, _end = row["window"].split("~", 1)
        matches = sorted(base.glob(start + "~*.TemporalData.simple"))
        if matches:
            return matches[0]
    row_time = parse_row_display_time(row)
    if row_time is None:
        row_time = parse_row_raw_time(row)
    if row_time is None:
        return None
    for path in sorted(base.glob("*.TemporalData.simple")):
        stem = path.name.replace(".TemporalData.simple", "")
        if "~" not in stem:
            continue
        start, end = stem.split("~", 1)
        try:
            if parse_file_time(start) <= row_time <= parse_file_time(end):
                return path
        except Exception:
            continue
    return None


@lru_cache(maxsize=16)
def load_window(path):
    data = torch.load(path, map_location="cpu").to("cpu")
    variants, edge_type = parse_temporal_data(data)
    return data, variants, edge_type


def load_edges_from_candidate(edge_dir_id, row):
    path = find_window_path(edge_dir_id, row)
    if path is None or not path.exists():
        return []
    data, variants, edge_type = load_window(str(path))
    src = int(row["srcnode"])
    dst = int(row["dstnode"])
    time = int(row["time_raw"]) if "time_raw" in row and row["time_raw"] else None
    if time is None:
        mask = (data.src == src) & (data.dst == dst)
    else:
        mask = (data.src == src) & (data.dst == dst) & (data.t == time)
    idx = mask.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return []

    # Keep all duplicate endpoint/time candidates. The display operation can be
    # ambiguous when several events share the same key, so saved loss is the
    # authoritative disambiguator.
    idx_values = [int(item) for item in idx.tolist()]

    loaded = []
    for chosen in idx_values:
        loaded.append({
            "edge_dir_id": edge_dir_id,
            "path": str(path),
            "x_pair_variants": {
                name: torch.cat([x_src[chosen], x_dst[chosen]], dim=-1).unsqueeze(0).float()
                for name, (x_src, x_dst) in variants.items()
            },
            "target_class": int(edge_type[chosen].argmax().item()),
            "edge_index_in_window": chosen,
            "edge_candidate_count": int(idx.numel()),
            "edge_count_same_endpoints": int(idx.numel()),
        })
    return loaded


def choose_rows():
    with open(SELECTED_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    chosen = []
    benign_below_count = 0
    for row in rows:
        if row["node_group"] not in NODE_GROUP_SHORT:
            continue
        if row["node_group"] == "TN_high_below_threshold":
            if benign_below_count >= BENIGN_BELOW_ABLATION_LIMIT:
                continue
            benign_below_count += 1
        out = dict(row)
        out["case"] = f"{row['node_group']}:{row['rank']}:{row['node_id']}"
        chosen.append(out)
    return chosen


def readable_type_pair(type_pair):
    return str(type_pair).replace("subject", "process").replace("netflow", "network")


def pattern_key(row):
    return row["node_group"], row["operation"], row["type_pair"]


def pattern_label(node_group, operation, type_pair, count):
    op = str(operation).replace("EVENT_", "")
    return f"{NODE_GROUP_SHORT.get(node_group, node_group)}\n{op}, {readable_type_pair(type_pair)}\nn={count}"


def percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize_ablation_rows(rows):
    counts_by_pattern = Counter((row["node_group"], row["operation"], row["type_pair"]) for row in rows if row["group"] == "src_embedding")
    pattern_order = sorted(
        counts_by_pattern,
        key=lambda key: (
            NODE_GROUP_ORDER.index(key[0]),
            -counts_by_pattern[key],
            key[1],
            key[2],
        ),
    )
    out = []
    for node_group, operation, type_pair in pattern_order:
        for group in GROUPS:
            vals = [
                float(row["loss_drop_when_zeroed"])
                for row in rows
                if row["node_group"] == node_group and row["operation"] == operation and row["type_pair"] == type_pair and row["group"] == group
            ]
            out.append({
                "node_group": node_group,
                "operation": operation,
                "type_pair": type_pair,
                "feature_group": group,
                "edge_count": len(vals),
                "loss_drop_mean": fmt(float(np.mean(vals)) if vals else float("nan")),
                "loss_drop_median": fmt(float(np.median(vals)) if vals else float("nan")),
                "loss_drop_p10": fmt(percentile(vals, 10)),
                "loss_drop_p90": fmt(percentile(vals, 90)),
            })
    return out


def find_matching_edge(model, row):
    candidates = []
    expected_loss = float(row["responsible_loss"])
    for edge_dir_id in EDGE_DIR_CANDIDATES:
        for loaded in load_edges_from_candidate(edge_dir_id, row):
            target = torch.tensor([loaded["target_class"]], dtype=torch.long)
            for variant_name, x_pair in loaded["x_pair_variants"].items():
                candidate = dict(loaded)
                candidate.pop("x_pair_variants")
                candidate["x_pair"] = x_pair
                candidate["feature_layout"] = variant_name
                loss = float(model(x_pair, target).item())
                candidate["loss"] = loss
                candidate["loss_error"] = abs(loss - expected_loss)
                candidates.append(candidate)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item["loss_error"])[0]


def manual_ablation(model, x_pair, target):
    original = float(model(x_pair, target).item())
    rows = []
    for group, (start, end) in GROUPS.items():
        ablated = x_pair.clone()
        ablated[:, start:end] = 0.0
        loss = float(model(ablated, target).item())
        rows.append({
            "group": group,
            "original_loss": original,
            "ablated_loss": loss,
            "loss_drop_when_zeroed": original - loss,
        })
    return rows


def captum_ablation(model, x_pair, target):
    feature_mask = torch.empty_like(x_pair, dtype=torch.long)
    for group_id, (_group, (start, end)) in enumerate(GROUPS.items()):
        feature_mask[:, start:end] = group_id
    ablator = FeatureAblation(lambda x: model(x, target))
    attrs = ablator.attribute(x_pair, baselines=torch.zeros_like(x_pair), feature_mask=feature_mask)
    rows = []
    for group, (start, end) in GROUPS.items():
        vals = attrs[:, start:end]
        rows.append({
            "group": group,
            "captum_attr_sum": float(vals.sum().item()),
            "captum_attr_mean": float(vals.mean().item()),
        })
    return rows


def plot_feature_ablation(summary_rows):
    pattern_keys = []
    pattern_counts = {}
    for row in summary_rows:
        key = (row["node_group"], row["operation"], row["type_pair"])
        if key not in pattern_keys:
            pattern_keys.append(key)
        pattern_counts[key] = int(row["edge_count"])

    groups = list(GROUPS)
    values = np.zeros((len(pattern_keys), len(groups)))
    for row in summary_rows:
        key = (row["node_group"], row["operation"], row["type_pair"])
        values[pattern_keys.index(key), groups.index(row["feature_group"])] = float(row["loss_drop_median"])

    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    left = min(-0.35, min_value * 1.2)
    right = max(0.35, max_value * 1.12)
    fig, ax = plt.subplots(figsize=(11.4, max(6.2, len(pattern_keys) * 0.86 + 2.0)))
    ax.axvspan(left, 0, color="#eff6ff", zorder=0)
    ax.axvspan(0, right, color="#f0fdf4", zorder=0)
    ax.axvline(0, color="#334155", linewidth=1.1, zorder=1)
    offsets = np.linspace(-0.27, 0.27, len(groups))
    bar_height = 0.13
    y_base = np.arange(len(pattern_keys))
    for group_index, group in enumerate(groups):
        ax.barh(
            y_base + offsets[group_index],
            values[:, group_index],
            height=bar_height,
            color=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.7,
            label=GROUP_LABELS[group],
            zorder=3,
        )
    for i in range(len(pattern_keys) + 1):
        ax.axhline(i - 0.5, color="#e2e8f0", linewidth=0.8, zorder=1)
    ax.set_xlim(left, right)
    ax.set_yticks(y_base)
    ax.set_yticklabels([pattern_label(*key, pattern_counts[key]) for key in pattern_keys], fontsize=8.7)
    ax.invert_yaxis()
    ax.set_xlabel("Median loss change when this feature group is removed")
    ax.set_ylabel("Node group, event type, endpoint type")
    ax.grid(axis="x", color="#cbd5e1", linewidth=0.7, alpha=0.65, zorder=1)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
        fontsize=9.2,
        handlelength=1.2,
        columnspacing=1.5,
    )
    plot_dir = OBS_DIR / "plots" / "from_reproduced_post_epoch_model"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"{MODEL_SLUG}_produced_internal_feature_ablation.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)
    plt.close(fig)


def generate(max_loss_error=1e-5, replay_only=False):
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights file not found: {WEIGHTS_PATH}")
    if WEIGHTS_PATH != POST_EPOCH_WEIGHTS_PATH:
        raise FileNotFoundError(
            f"Exact post-epoch model is required for internal ablation: {POST_EPOCH_WEIGHTS_PATH}"
        )
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model = VeloxEdgeModel(state)
    set_feature_groups(model.embedding_dim)
    model.eval()
    cases = choose_rows()
    logit_rows = []
    ablation_rows = []
    unmatched_cases = []
    for row in cases:
        loaded = find_matching_edge(model, row)
        if loaded is None:
            unmatched_cases.append(row["case"])
            continue
        x_pair = loaded["x_pair"]
        target = torch.tensor([loaded["target_class"]], dtype=torch.long)
        logits = model.logits_from_pair(x_pair[:, :model.node_feature_dim], x_pair[:, model.node_feature_dim:])
        probs = F.softmax(logits, dim=-1)[0]
        pred_class = int(probs.argmax().item())
        true_class = int(target.item())
        true_prob = float(probs[true_class].item())
        pred_prob = float(probs[pred_class].item())
        loss = float(model(x_pair, target).item())
        logit_rows.append({
            "case": row["case"],
            "rank": row["rank"],
            "node_id": row["node_id"],
            "node_group": row["node_group"],
            "operation_from_db": row["operation"],
            "type_pair": row["type_pair"],
            "target_class": true_class,
            "target_edge_type": EDGE_TYPE_NAMES.get(true_class, str(true_class)),
            "predicted_class": pred_class,
            "predicted_edge_type": EDGE_TYPE_NAMES.get(pred_class, str(pred_class)),
            "true_probability": fmt(true_prob, 12),
            "predicted_probability": fmt(pred_prob, 12),
            "computed_loss": fmt(loss),
            "saved_loss": row["responsible_loss"],
            "loss_error": fmt(abs(loss - float(row["responsible_loss"])), 12),
            "edge_dir_id": loaded["edge_dir_id"],
            "weights_path": str(WEIGHTS_PATH),
            "feature_layout": loaded["feature_layout"],
            "edge_index_in_window": loaded["edge_index_in_window"],
            "edge_candidate_count": loaded["edge_candidate_count"],
            "edge_count_same_endpoints": loaded["edge_count_same_endpoints"],
            "src_label": row["src_label"],
            "dst_label": row["dst_label"],
        })
        if replay_only:
            continue
        manual = manual_ablation(model, x_pair, target)
        captum = captum_ablation(model, x_pair, target)
        captum_by_group = {r["group"]: r for r in captum}
        for item in manual:
            group = item["group"]
            out = {
                "case": row["case"],
                "rank": row["rank"],
                "node_id": row["node_id"],
                "node_group": row["node_group"],
                "operation": row["operation"],
                "type_pair": row["type_pair"],
                "group": group,
                "original_loss": fmt(item["original_loss"]),
                "ablated_loss": fmt(item["ablated_loss"]),
                "loss_drop_when_zeroed": fmt(item["loss_drop_when_zeroed"]),
                "captum_attr_sum": fmt(captum_by_group[group]["captum_attr_sum"]),
                "captum_attr_mean": fmt(captum_by_group[group]["captum_attr_mean"]),
            }
            ablation_rows.append(out)
    if unmatched_cases:
        preview = ", ".join(unmatched_cases[:5])
        raise RuntimeError(f"Could not match {len(unmatched_cases)} selected edges in TemporalData windows. First unmatched: {preview}")
    if logit_rows:
        max_error = max(float(row["loss_error"]) for row in logit_rows)
        if max_error > max_loss_error:
            worst = max(logit_rows, key=lambda row: float(row["loss_error"]))
            raise RuntimeError(
                f"Loss reproduction failed for {DATASET}: max loss_error={max_error:.12f} "
                f"> {max_loss_error:.12f}; worst case={worst['case']} weights={WEIGHTS_PATH}"
            )
        print(f"max_loss_error={max_error:.12f}")
    if replay_only:
        write_csv(OBS_DIR / "tables/internal_edge_logits.csv", LOGIT_FIELDS, logit_rows)
        return
    summary_rows = summarize_ablation_rows(ablation_rows)
    write_csv(
        OBS_DIR / "tables/internal_edge_logits.csv",
        LOGIT_FIELDS,
        logit_rows,
    )
    write_csv(
        OBS_DIR / "tables/internal_feature_ablation.csv",
        ["case", "rank", "node_id", "node_group", "operation", "type_pair", "group", "original_loss", "ablated_loss", "loss_drop_when_zeroed", "captum_attr_sum", "captum_attr_mean"],
        ablation_rows,
    )
    write_csv(
        OBS_DIR / "tables/internal_feature_ablation_summary.csv",
        ["node_group", "operation", "type_pair", "feature_group", "edge_count", "loss_drop_mean", "loss_drop_median", "loss_drop_p10", "loss_drop_p90"],
        summary_rows,
    )
    plot_feature_ablation(summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="THEIA_E3", choices=list(RUNS))
    parser.add_argument("--max-loss-error", type=float, default=1e-5)
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    configure_run(args.dataset)
    generate(max_loss_error=args.max_loss_error, replay_only=args.replay_only)


if __name__ == "__main__":
    main()
