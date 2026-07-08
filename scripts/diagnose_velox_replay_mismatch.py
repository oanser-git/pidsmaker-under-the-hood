import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


from poc_paths import add_pidsmaker_src, artifacts_root  # noqa: E402


add_pidsmaker_src()

from config import get_runtime_required_args, get_yml_cfg  # noqa: E402
from data_utils import extract_msg_from_data, load_data_set  # noqa: E402
from factory import batch_loader_factory, build_model  # noqa: E402


RUNS = {
    "CADETS_E3": {
        "obs_id": "2026-07-01_cadets_e3_velox",
        "model_slug": "cadets_e3",
        "edge_id": "eae5106c3e9b47e33137c97ef008f51a1ef2c69d85c7398510d91586b7bda848",
    },
    "optc_h051": {
        "obs_id": "2026-07-01_optc_h051_velox",
        "model_slug": "optc_h051",
        "edge_id": "7f75250cc14901cf70bde015a0bab8f5d5ecc3513462f5d9e7d16ae8b4eb157b",
    },
    "optc_h201": {
        "obs_id": "2026-07-01_optc_h201_velox",
        "model_slug": "optc_h201",
        "edge_id": "6a3db7ae1e2232a0d7450f8d450b249ef5a448a63ebc4ef5a4bc633db1fb755d",
    },
    "optc_h501": {
        "obs_id": "2026-07-01_optc_h501_velox",
        "model_slug": "optc_h501",
        "edge_id": "fcc39f462d34d8d0a2b9dd02365aa38717cb247d6d5d9d4a208cc961270d3d2f",
    },
    "THEIA_E3": {
        "obs_id": "2026-07-01_theia_e3_velox",
        "model_slug": "theia_e3",
        "edge_id": "499b3a9d2ad282bab5e39ca1a680d479cb2804fbb35d8759f2e3e84fbdd30d1a",
    },
}


ROOT = artifacts_root()


def parse_file_time(value):
    value = str(value).strip()
    if "." in value:
        base, frac = value.split(".", 1)
        frac = "".join(ch for ch in frac if ch.isdigit())[:6].ljust(6, "0")
        return datetime.strptime(f"{base}.{frac}", "%Y-%m-%d %H:%M:%S.%f")
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def find_window_path(base, window):
    exact = base / f"{window}.TemporalData.simple"
    if exact.exists():
        return exact, "exact"
    if "~" in window:
        start, end = window.split("~", 1)
        matches = sorted(base.glob(start + "~*.TemporalData.simple"))
        if matches:
            return matches[0], "same_start"
        try:
            want_start = parse_file_time(start)
            want_end = parse_file_time(end)
        except Exception:
            return None, "bad_window_time"
        for path in sorted(base.glob("*.TemporalData.simple")):
            stem = path.name.replace(".TemporalData.simple", "")
            if "~" not in stem:
                continue
            have_start, have_end = stem.split("~", 1)
            try:
                if parse_file_time(have_start) <= want_start and want_end <= parse_file_time(have_end):
                    return path, "parent_contains"
            except Exception:
                continue
    return None, "missing"


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


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(path)


@lru_cache(maxsize=8)
def load_raw_window(path):
    return torch.load(path, map_location="cpu")


def build_runtime(dataset, edge_id, weights_path):
    args = get_runtime_required_args(args=["velox", dataset, "--tuned", "--from_weights"])
    cfg = get_yml_cfg(args)
    edge_dir = ROOT / "featurization" / dataset / "embed_edges" / edge_id / "edge_embeds"
    cfg.featurization.embed_edges._edge_embeds_dir = str(edge_dir)
    train_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="train")
    val_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="val")
    test_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="test")
    max_node_num = max(
        int(torch.cat([graph.src, graph.dst]).max().item())
        for split_data in (train_data, val_data, test_data)
        for graph in split_data
    ) + 1
    model = build_model(data_sample=train_data[0], device=torch.device("cpu"), cfg=cfg, max_node_num=max_node_num)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return cfg, model, edge_dir


def actual_losses_for_window(cfg, model, path):
    raw = load_raw_window(str(path))
    data = extract_msg_from_data([raw.clone()], cfg)[0]
    losses_by_key = defaultdict(list)
    for batch in batch_loader_factory(cfg, data, model.graph_reindexer, test_mode=True):
        with torch.no_grad():
            losses = model(batch, None, inference=True, validation=False)["loss"].detach().cpu()
        edge_index = batch.original_edge_index if hasattr(batch, "original_edge_index") else batch.edge_index
        edge_index = edge_index.detach().cpu()
        times = batch.t.detach().cpu()
        for idx, loss in enumerate(losses.tolist()):
            key = (int(edge_index[0, idx].item()), int(edge_index[1, idx].item()), int(times[idx].item()))
            losses_by_key[key].append(float(loss))
    return losses_by_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(RUNS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run = RUNS[args.dataset]
    obs_dir = ROOT / "observations" / run["obs_id"]
    selected_path = obs_dir / "tables" / "selected_responsible_edges.csv"
    weights_path = obs_dir / "models" / f"{run['model_slug']}_velox_post_epoch_model_epoch_0" / "state_dict.pkl"
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    cfg, model, edge_dir = build_runtime(args.dataset, run["edge_id"], weights_path)
    base = edge_dir / "test"
    rows = []
    with selected_path.open(newline="") as f:
        selected_rows = list(csv.DictReader(f))
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    window_cache = {}
    for row in selected_rows:
        path, match_type = find_window_path(base, row["window"])
        out = {
            "case": f"{row['node_group']}:{row['rank']}:{row['node_id']}",
            "node_group": row["node_group"],
            "rank": row["rank"],
            "node_id": row["node_id"],
            "srcnode": row["srcnode"],
            "dstnode": row["dstnode"],
            "time_raw": row.get("time_raw", ""),
            "window": row["window"],
            "saved_loss": row["responsible_loss"],
            "temporaldata_match_type": match_type,
            "temporaldata_path": str(path) if path else "",
        }
        if path is None:
            out.update({"status": "missing_window", "actual_loss": "", "actual_loss_error": ""})
            rows.append(out)
            continue
        if path not in window_cache:
            window_cache[path] = actual_losses_for_window(cfg, model, path)
        if row.get("time_raw"):
            key = (int(row["srcnode"]), int(row["dstnode"]), int(row["time_raw"]))
            candidates = window_cache[path].get(key, [])
        else:
            src = int(row["srcnode"])
            dst = int(row["dstnode"])
            candidates = [
                loss
                for (candidate_src, candidate_dst, _candidate_time), losses in window_cache[path].items()
                if candidate_src == src and candidate_dst == dst
                for loss in losses
            ]
        if not candidates:
            out.update({"status": "missing_edge", "actual_loss": "", "actual_loss_error": "", "candidate_count": 0})
        else:
            saved_loss = float(row["responsible_loss"])
            actual_loss = min(candidates, key=lambda value: abs(value - saved_loss))
            out.update(
                {
                    "status": "matched",
                    "actual_loss": fmt(actual_loss),
                    "actual_loss_error": fmt(abs(actual_loss - saved_loss)),
                    "candidate_count": len(candidates),
                }
            )
        rows.append(out)

    matched_errors = [float(row["actual_loss_error"]) for row in rows if row.get("status") == "matched"]
    summary_rows = [
        {
            "dataset": args.dataset,
            "rows": len(rows),
            "matched_rows": sum(1 for row in rows if row.get("status") == "matched"),
            "missing_rows": sum(1 for row in rows if row.get("status") != "matched"),
            "unique_temporaldata_files": len(window_cache),
            "max_actual_loss_error": fmt(max(matched_errors) if matched_errors else float("nan")),
            "p50_actual_loss_error": fmt(float(np.percentile(matched_errors, 50)) if matched_errors else float("nan")),
            "p90_actual_loss_error": fmt(float(np.percentile(matched_errors, 90)) if matched_errors else float("nan")),
            "within_1e_5": sum(1 for error in matched_errors if error <= 1e-5),
            "replay_passed": "yes" if matched_errors and max(matched_errors) <= 1e-5 and all(row.get("status") == "matched" for row in rows) else "no",
            "weights_path": str(weights_path),
        }
    ]
    out_prefix = obs_dir / "tables" / "actual_pipeline_replay_diagnostic"
    write_csv(
        out_prefix.with_suffix(".csv"),
        [
            "case",
            "node_group",
            "rank",
            "node_id",
            "srcnode",
            "dstnode",
            "time_raw",
            "window",
            "saved_loss",
            "actual_loss",
            "actual_loss_error",
            "status",
            "candidate_count",
            "temporaldata_match_type",
            "temporaldata_path",
        ],
        rows,
    )
    write_csv(
        obs_dir / "tables" / "actual_pipeline_replay_diagnostic_summary.csv",
        [
            "dataset",
            "rows",
            "matched_rows",
            "missing_rows",
            "unique_temporaldata_files",
            "max_actual_loss_error",
            "p50_actual_loss_error",
            "p90_actual_loss_error",
            "within_1e_5",
            "replay_passed",
            "weights_path",
        ],
        summary_rows,
    )
    print(summary_rows[0])


if __name__ == "__main__":
    main()
