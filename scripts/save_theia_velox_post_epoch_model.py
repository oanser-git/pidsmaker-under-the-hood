import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch


from poc_paths import add_pidsmaker_src, artifacts_root  # noqa: E402


add_pidsmaker_src()

from config import get_runtime_required_args, get_yml_cfg  # noqa: E402
from data_utils import load_data_set, save_model  # noqa: E402
from detection.training_methods.orthrus_gnn_training import train  # noqa: E402
from factory import build_model, optimizer_factory  # noqa: E402
from provnet_utils import get_device, log_tqdm  # noqa: E402


RUNS = {
    "THEIA_E3": {
        "obs_id": "2026-07-01_theia_e3_velox",
        "edge_embed_id": "499b3a9d2ad282bab5e39ca1a680d479cb2804fbb35d8759f2e3e84fbdd30d1a",
        "model_slug": "theia_e3",
    },
    "CADETS_E3": {
        "obs_id": "2026-07-01_cadets_e3_velox",
        "edge_embed_id": "eae5106c3e9b47e33137c97ef008f51a1ef2c69d85c7398510d91586b7bda848",
        "model_slug": "cadets_e3",
    },
    "optc_h201": {
        "obs_id": "2026-07-01_optc_h201_velox",
        "edge_embed_id": "6a3db7ae1e2232a0d7450f8d450b249ef5a448a63ebc4ef5a4bc633db1fb755d",
        "model_slug": "optc_h201",
    },
    "optc_h501": {
        "obs_id": "2026-07-01_optc_h501_velox",
        "edge_embed_id": "fcc39f462d34d8d0a2b9dd02365aa38717cb247d6d5d9d4a208cc961270d3d2f",
        "model_slug": "optc_h501",
    },
    "optc_h051": {
        "obs_id": "2026-07-01_optc_h051_velox",
        "edge_embed_id": "7f75250cc14901cf70bde015a0bab8f5d5ecc3513462f5d9e7d16ae8b4eb157b",
        "model_slug": "optc_h051",
    },
}


ARTIFACTS_ROOT = artifacts_root()


def max_node_from_graphs(graphs):
    max_node = 0
    for graph in graphs:
        max_node = max(max_node, int(torch.cat([graph.src, graph.dst]).max().item()))
    return max_node + 1


def seed_like_pipeline(cfg):
    if not cfg.detection.gnn_training.use_seed:
        return
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_for_dataset(dataset):
    run = RUNS[dataset]
    args = get_runtime_required_args(args=["velox", dataset, "--tuned", "--from_weights"])
    cfg = get_yml_cfg(args)
    seed_like_pipeline(cfg)
    device = get_device(cfg)

    edge_embed_id = os.environ.get(f"{dataset.upper()}_EDGE_EMBED_ID", run["edge_embed_id"])
    edge_embed_dir = ARTIFACTS_ROOT / "featurization" / dataset / "embed_edges" / edge_embed_id / "edge_embeds"
    if not edge_embed_dir.exists():
        raise FileNotFoundError(f"Completed edge embedding directory not found: {edge_embed_dir}")
    cfg.featurization.embed_edges._edge_embeds_dir = str(edge_embed_dir)

    train_data = load_data_set(cfg, path=cfg.featurization.embed_edges._edge_embeds_dir, split="train")
    max_node_num = max_node_from_graphs(train_data)
    model = build_model(data_sample=train_data[0], device=device, cfg=cfg, max_node_num=max_node_num)
    weights_path = os.path.join(cfg._from_weights_path, "encoder", f"{cfg.dataset.name}.pkl")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    optimizer = optimizer_factory(cfg, parameters=set(model.parameters()))

    total_loss = 0.0
    for graph in log_tqdm(train_data, f"Training one {dataset} from-weights epoch for attribution"):
        graph.to(device=device)
        total_loss += float(train(data=graph, full_data=None, model=model, optimizer=optimizer, cfg=cfg))
        graph.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    mean_loss = total_loss / len(train_data)

    obs_dir = ARTIFACTS_ROOT / "observations" / run["obs_id"]
    model_dir = obs_dir / f"models/{run['model_slug']}_velox_post_epoch_model_epoch_0"
    save_model(model, str(model_dir), cfg)
    print(model_dir)
    print(f"mean_train_loss={mean_loss:.9f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="THEIA_E3", choices=list(RUNS) + ["all"])
    args = parser.parse_args()
    datasets = [key for key in RUNS if key != "THEIA_E3"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        save_for_dataset(dataset)


if __name__ == "__main__":
    main()
