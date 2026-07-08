import argparse
import csv
import math
import os
import sys
import types
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


from poc_paths import add_pidsmaker_src, artifacts_root  # noqa: E402


add_pidsmaker_src()

try:
    import data_utils as _data_utils

    pidsmaker_pkg = sys.modules.setdefault("pidsmaker", types.ModuleType("pidsmaker"))
    pidsmaker_pkg.__path__ = []
    utils_pkg = sys.modules.setdefault("pidsmaker.utils", types.ModuleType("pidsmaker.utils"))
    utils_pkg.__path__ = []
    sys.modules.setdefault("pidsmaker.data_utils", _data_utils)
    sys.modules.setdefault("pidsmaker.utils.data_utils", _data_utils)
except Exception:
    # The alias is only needed when unpickling TemporalData files produced by
    # the packaged ULHPC runtime. Let torch.load report the concrete failure.
    pass


DEFAULT_OBS_ID = "2026-07-05_theia_e3_ulhpc_velox_hpo"
ARTIFACTS_ROOT = artifacts_root()
OBS_DIR = Path(os.environ.get("OBS_DIR") or ARTIFACTS_ROOT / "observations" / DEFAULT_OBS_ID)
ULHPC_ROOT = ARTIFACTS_ROOT / "ULHPC-reporduction"
REQUESTED_ROOT = ULHPC_ROOT / "requested-data"
FEATURE_ROOT = REQUESTED_ROOT / "selected-edge-features"
TABLE_DIR = OBS_DIR / "tables"
SELECTED_PATH = TABLE_DIR / "ulhpc_theia_e3_selected_responsible_edges.csv"

EDGE_TYPE_NAMES = {
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
EDGE_TYPE_IDS = {value: key for key, value in EDGE_TYPE_NAMES.items()}

RUNS = {
    "leaky-selected": {
        "model_label": "Leaky selected",
        "score_epoch": "0",
        "epoch": "model_epoch_0",
        "training_hash": "e7584cf3b37ac038613cb4118afa5119a158cf216ce2a57cb5c35ade8988e0ff",
        "weights_path": ULHPC_ROOT
        / "leaky-HPO"
        / "seed-stability"
        / "training"
        / "training"
        / "e7584cf3b37ac038613cb4118afa5119a158cf216ce2a57cb5c35ade8988e0ff"
        / "THEIA_E3"
        / "trained_models"
        / "model_epoch_0"
        / "state_dict.pkl",
    },
    "no-leaky-selected": {
        "model_label": "Non-leaky selected",
        "score_epoch": "3",
        "epoch": "model_epoch_3",
        "training_hash": "372703b49da8dd8caad886a6ce4469d921d95cd85272386dc2380afc523ed1f2",
        "weights_path": ULHPC_ROOT
        / "no-leaky-HPO"
        / "validation-loss-HPO-and-selected-test"
        / "training"
        / "training"
        / "372703b49da8dd8caad886a6ce4469d921d95cd85272386dc2380afc523ed1f2"
        / "THEIA_E3"
        / "trained_models"
        / "model_epoch_3"
        / "state_dict.pkl",
    },
}


class VeloxEdgeModel(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.state = state_dict
        self.node_feature_dim = int(self.state["encoder.lin1.weight"].shape[1])
        self.embedding_dim = self.node_feature_dim - 3
        self.keys = {
            "lin_src_weight": self.resolve_key("decoders.0.objective.decoder.lin_src.weight", "objectives.0.objective.decoder.lin_src.weight"),
            "lin_src_bias": self.resolve_key("decoders.0.objective.decoder.lin_src.bias", "objectives.0.objective.decoder.lin_src.bias"),
            "lin_dst_weight": self.resolve_key("decoders.0.objective.decoder.lin_dst.weight", "objectives.0.objective.decoder.lin_dst.weight"),
            "lin_dst_bias": self.resolve_key("decoders.0.objective.decoder.lin_dst.bias", "objectives.0.objective.decoder.lin_dst.bias"),
            "mlp0_weight": self.resolve_key("decoders.0.objective.decoder.mlp.0.weight", "objectives.0.objective.decoder.mlp.mlp.0.weight"),
            "mlp0_bias": self.resolve_key("decoders.0.objective.decoder.mlp.0.bias", "objectives.0.objective.decoder.mlp.mlp.0.bias"),
            "mlp2_weight": self.resolve_key("decoders.0.objective.decoder.mlp.2.weight", "objectives.0.objective.decoder.mlp.mlp.2.weight"),
            "mlp2_bias": self.resolve_key("decoders.0.objective.decoder.mlp.2.bias", "objectives.0.objective.decoder.mlp.mlp.2.bias"),
        }

    def resolve_key(self, *candidates):
        for key in candidates:
            if key in self.state:
                return key
        raise KeyError(f"None of these checkpoint keys exist: {', '.join(candidates)}")

    def param(self, name):
        return self.state[self.keys[name]]

    def logits_from_pair(self, x_src, x_dst):
        h_src = F.linear(x_src, self.state["encoder.lin1.weight"], self.state["encoder.lin1.bias"])
        h_dst = F.linear(x_dst, self.state["encoder.lin1.weight"], self.state["encoder.lin1.bias"])
        z_src = F.linear(h_src, self.param("lin_src_weight"), self.param("lin_src_bias"))
        z_dst = F.linear(h_dst, self.param("lin_dst_weight"), self.param("lin_dst_bias"))
        h = torch.cat([z_src, z_dst], dim=-1)
        h = F.linear(h, self.param("mlp0_weight"), self.param("mlp0_bias"))
        h = F.relu(h)
        return F.linear(h, self.param("mlp2_weight"), self.param("mlp2_bias"))

    def forward(self, x_pair, target_classes):
        x_src = x_pair[:, : self.node_feature_dim]
        x_dst = x_pair[:, self.node_feature_dim :]
        logits = self.logits_from_pair(x_src, x_dst)
        return F.cross_entropy(logits, target_classes, reduction="none")


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_state_dict(state):
    if isinstance(state, dict) and "encoder.lin1.weight" in state:
        return state
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            nested = state.get(key)
            if isinstance(nested, dict) and "encoder.lin1.weight" in nested:
                return nested
    raise KeyError("Could not find encoder.lin1.weight in loaded checkpoint")


def parse_temporal_data(data):
    msg = data.msg
    emb_dim = (msg.size(1) - 16) // 2
    src_type = msg[:, 0:3]
    src_emb = msg[:, 3 : 3 + emb_dim]
    edge_type_start = 3 + emb_dim
    edge_type = msg[:, edge_type_start : edge_type_start + 10]
    dst_type_start = edge_type_start + 10
    dst_type = msg[:, dst_type_start : dst_type_start + 3]
    dst_emb = msg[:, dst_type_start + 3 : dst_type_start + 3 + emb_dim]
    variants = {
        "emb_type": (torch.cat([src_emb, src_type], dim=-1), torch.cat([dst_emb, dst_type], dim=-1)),
        "type_emb": (torch.cat([src_type, src_emb], dim=-1), torch.cat([dst_type, dst_emb], dim=-1)),
        "emb_type_srcdst_swap": (torch.cat([dst_emb, dst_type], dim=-1), torch.cat([src_emb, src_type], dim=-1)),
        "type_emb_srcdst_swap": (torch.cat([dst_type, dst_emb], dim=-1), torch.cat([src_type, src_emb], dim=-1)),
    }
    return data, variants, edge_type


@lru_cache(maxsize=2)
def load_window(path):
    data = torch_load(path).to("cpu")
    return parse_temporal_data(data)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(path)


def fmt(value, digits=12):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{float(value):.{digits}f}"


def percentile(values, q):
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_mappings(model_key):
    mapping_path = FEATURE_ROOT / model_key / "mapping_test.csv"
    rows = read_csv(mapping_path)
    by_window = {}
    for row in rows:
        window = row["edge_loss_csv"].removesuffix(".csv")
        if window in by_window:
            raise RuntimeError(f"Duplicate mapping for {model_key} window {window}")
        by_window[window] = row
    return by_window


def load_selected_rows(model_keys, limit=None):
    rows = read_csv(SELECTED_PATH)
    by_label = {run["model_label"]: key for key, run in RUNS.items()}
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        model_key = by_label.get(row["model_label"])
        if model_key not in model_keys:
            continue
        row = dict(row)
        row["original_order"] = index
        row["model_key"] = model_key
        grouped[model_key].append(row)

    selected = []
    for model_key in model_keys:
        model_rows = grouped[model_key]
        if limit is not None:
            model_rows = model_rows[:limit]
        selected.extend(model_rows)
    return selected


def prepare_rows(rows):
    mappings = {model_key: load_mappings(model_key) for model_key in {row["model_key"] for row in rows}}
    prepared = []
    for row in rows:
        mapping = mappings[row["model_key"]].get(row["window"])
        out = dict(row)
        if mapping is None:
            out["status"] = "missing_mapping"
            out["temporaldata_path"] = ""
            out["temporaldata_file"] = ""
            out["match_type"] = ""
        else:
            temporaldata_path = FEATURE_ROOT / mapping["staged_temporaldata_path"]
            out["status"] = "prepared"
            out["temporaldata_path"] = str(temporaldata_path)
            out["temporaldata_file"] = mapping["temporaldata_file"]
            out["match_type"] = mapping["match_type"]
            out["edge_loss_csv"] = mapping["edge_loss_csv"]
            out["edge_loss_start"] = mapping["edge_loss_start"]
            out["edge_loss_end"] = mapping["edge_loss_end"]
        prepared.append(out)
    return prepared


def candidate_indices(data, edge_type, row):
    src = int(row["srcnode"])
    dst = int(row["dstnode"])
    time_raw = int(row["time_raw"])
    mask = (data.src == src) & (data.dst == dst) & (data.t == time_raw)
    indices = mask.nonzero(as_tuple=False).flatten().tolist()
    expected_class = EDGE_TYPE_IDS.get(row.get("operation", ""))
    if expected_class is None or not indices:
        return indices, len(indices)
    matching = [index for index in indices if int(edge_type[index].argmax().item()) == expected_class]
    return matching or indices, len(indices)


def replay_row(model, row):
    if row["status"] != "prepared":
        return {"status": row["status"]}
    path = row["temporaldata_path"]
    if not Path(path).exists():
        return {"status": "missing_temporaldata"}

    data, variants, edge_type = load_window(path)
    indices, same_endpoint_count = candidate_indices(data, edge_type, row)
    if not indices:
        return {"status": "missing_edge_in_temporaldata", "edge_count_same_endpoints": "0"}

    expected_loss = float(row["responsible_loss"])
    candidates = []
    with torch.no_grad():
        for index in indices:
            target_class = int(edge_type[index].argmax().item())
            target = torch.tensor([target_class], dtype=torch.long)
            for feature_layout, (x_src, x_dst) in variants.items():
                x_pair = torch.cat([x_src[index], x_dst[index]], dim=-1).unsqueeze(0).float()
                logits = model.logits_from_pair(x_pair[:, : model.node_feature_dim], x_pair[:, model.node_feature_dim :])
                loss = float(F.cross_entropy(logits, target, reduction="none").item())
                probs = F.softmax(logits, dim=-1)[0]
                pred_class = int(probs.argmax().item())
                candidates.append(
                    {
                        "status": "matched",
                        "computed_loss": loss,
                        "loss_error": abs(loss - expected_loss),
                        "target_class": target_class,
                        "target_edge_type": EDGE_TYPE_NAMES.get(target_class, str(target_class)),
                        "predicted_class": pred_class,
                        "predicted_edge_type": EDGE_TYPE_NAMES.get(pred_class, str(pred_class)),
                        "true_probability": float(probs[target_class].item()),
                        "predicted_probability": float(probs[pred_class].item()),
                        "feature_layout": feature_layout,
                        "edge_index_in_window": index,
                        "edge_count_same_endpoints": same_endpoint_count,
                    }
                )
    return min(candidates, key=lambda item: item["loss_error"])


def build_base_output(row):
    case = f"{row['model_key']}:{row['node_group']}:{row['rank']}:{row['node_id']}"
    return {
        "case": case,
        "model_key": row["model_key"],
        "model_label": row["model_label"],
        "score_epoch": row["score_epoch"],
        "rank": row["rank"],
        "node_id": row["node_id"],
        "node_group": row["node_group"],
        "score": row["score"],
        "label": row["label"],
        "responsible_loss": row["responsible_loss"],
        "operation_from_db": row["operation"],
        "window": row["window"],
        "srcnode": row["srcnode"],
        "dstnode": row["dstnode"],
        "time_raw": row["time_raw"],
        "type_pair": row["type_pair"],
        "src_label": row["src_label"],
        "dst_label": row["dst_label"],
        "temporaldata_file": row.get("temporaldata_file", ""),
        "temporaldata_path": row.get("temporaldata_path", ""),
        "mapping_match_type": row.get("match_type", ""),
        "edge_loss_csv": row.get("edge_loss_csv", ""),
        "original_order": row["original_order"],
    }


def replay_model(model_key, rows):
    run = RUNS[model_key]
    weights_path = run["weights_path"]
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    state = normalize_state_dict(torch_load(weights_path))
    model = VeloxEdgeModel(state)
    model.eval()

    out_rows = []
    for row in rows:
        base = build_base_output(row)
        result = replay_row(model, row)
        base.update(result)
        base["weights_path"] = str(weights_path)
        out_rows.append(base)
    return out_rows


def summarize(rows, max_loss_error):
    summary = []
    for model_key in RUNS:
        model_rows = [row for row in rows if row["model_key"] == model_key]
        if not model_rows:
            continue
        matched = [row for row in model_rows if row.get("status") == "matched"]
        errors = [float(row["loss_error"]) for row in matched]
        unmatched = [row for row in model_rows if row.get("status") != "matched"]
        max_error = max(errors) if errors else float("nan")
        worst = max(matched, key=lambda row: float(row["loss_error"])) if matched else {}
        replay_passed = len(unmatched) == 0 and bool(errors) and max_error <= max_loss_error
        layouts = sorted({row.get("feature_layout", "") for row in matched if row.get("feature_layout")})
        summary.append(
            {
                "model_key": model_key,
                "model_label": RUNS[model_key]["model_label"],
                "epoch": RUNS[model_key]["epoch"],
                "selected_rows": len(model_rows),
                "matched_rows": len(matched),
                "unmatched_rows": len(unmatched),
                "max_loss_error": fmt(max_error),
                "p50_loss_error": fmt(percentile(errors, 50)),
                "p90_loss_error": fmt(percentile(errors, 90)),
                "p99_loss_error": fmt(percentile(errors, 99)),
                "within_1e_6": sum(1 for error in errors if error <= 1e-6),
                "within_1e_5": sum(1 for error in errors if error <= 1e-5),
                "max_loss_error_allowed": fmt(max_loss_error),
                "replay_passed": "yes" if replay_passed else "no",
                "ablation_allowed": "yes" if replay_passed else "no",
                "worst_case": worst.get("case", ""),
                "worst_saved_loss": worst.get("responsible_loss", ""),
                "worst_computed_loss": fmt(worst.get("computed_loss")) if worst else "",
                "worst_temporaldata_file": worst.get("temporaldata_file", ""),
                "feature_layouts": ";".join(layouts),
                "weights_path": str(RUNS[model_key]["weights_path"]),
            }
        )
    return summary


def format_output_rows(rows):
    float_columns = {"computed_loss", "loss_error", "true_probability", "predicted_probability"}
    formatted = []
    for row in rows:
        out = dict(row)
        for column in float_columns:
            out[column] = fmt(out.get(column))
        for column in ("target_class", "predicted_class", "edge_index_in_window", "edge_count_same_endpoints"):
            out[column] = fmt(out.get(column), digits=0)
        formatted.append(out)
    return formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all"] + list(RUNS), default="all")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected rows per model for smoke checks.")
    parser.add_argument("--max-loss-error", type=float, default=1e-5)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    parser.add_argument("--allow-mismatch", action="store_true", help="Write diagnostics but exit 0 even when exact replay fails.")
    args = parser.parse_args()

    model_keys = list(RUNS) if args.model == "all" else [args.model]
    selected_rows = load_selected_rows(model_keys, limit=args.limit)
    prepared_rows = prepare_rows(selected_rows)
    prepared_rows.sort(key=lambda row: (row["model_key"], row.get("temporaldata_path", ""), int(row["original_order"])))

    all_rows = []
    for model_key in model_keys:
        model_rows = [row for row in prepared_rows if row["model_key"] == model_key]
        all_rows.extend(replay_model(model_key, model_rows))
        load_window.cache_clear()

    all_rows.sort(key=lambda row: int(row["original_order"]))
    summary_rows = summarize(all_rows, args.max_loss_error)

    row_fields = [
        "case",
        "model_key",
        "model_label",
        "score_epoch",
        "rank",
        "node_id",
        "node_group",
        "score",
        "label",
        "responsible_loss",
        "computed_loss",
        "loss_error",
        "status",
        "operation_from_db",
        "target_class",
        "target_edge_type",
        "predicted_class",
        "predicted_edge_type",
        "true_probability",
        "predicted_probability",
        "feature_layout",
        "edge_index_in_window",
        "edge_count_same_endpoints",
        "mapping_match_type",
        "temporaldata_file",
        "window",
        "edge_loss_csv",
        "srcnode",
        "dstnode",
        "time_raw",
        "type_pair",
        "src_label",
        "dst_label",
        "weights_path",
        "temporaldata_path",
    ]
    summary_fields = [
        "model_key",
        "model_label",
        "epoch",
        "selected_rows",
        "matched_rows",
        "unmatched_rows",
        "max_loss_error",
        "p50_loss_error",
        "p90_loss_error",
        "p99_loss_error",
        "within_1e_6",
        "within_1e_5",
        "max_loss_error_allowed",
        "replay_passed",
        "ablation_allowed",
        "worst_case",
        "worst_saved_loss",
        "worst_computed_loss",
        "worst_temporaldata_file",
        "feature_layouts",
        "weights_path",
    ]
    suffix = "sample" if args.limit is not None else "full"
    write_csv(TABLE_DIR / f"ulhpc_theia_e3_selected_loss_replay_{suffix}.csv", row_fields, format_output_rows(all_rows))
    write_csv(TABLE_DIR / f"ulhpc_theia_e3_selected_loss_replay_{suffix}_summary.csv", summary_fields, summary_rows)

    for row in summary_rows:
        print(
            f"{row['model_key']}: matched={row['matched_rows']}/{row['selected_rows']} "
            f"max_loss_error={row['max_loss_error']} replay_passed={row['replay_passed']}"
        )
    if any(row["replay_passed"] != "yes" for row in summary_rows) and not args.allow_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
