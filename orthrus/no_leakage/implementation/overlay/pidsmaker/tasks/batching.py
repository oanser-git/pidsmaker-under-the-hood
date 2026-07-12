import os

import torch

from pidsmaker.utils.data_utils import load_all_datasets
from pidsmaker.utils.utils import get_device, log, log_start, set_seed


def get_preprocessed_graphs(cfg, include_test=True):
    file_suffix = "" if include_test else "_validation_only"
    if cfg.batching.save_on_disk:
        log("Loading preprocessed graphs...")
        out_dir = cfg.batching._preprocessed_graphs_dir
        out_file = os.path.join(out_dir, f"torch_graphs{file_suffix}.pkl")
        train_data, val_data, test_data, max_node_num = torch.load(out_file)

    else:
        log("Computing graphs...")
        device = get_device(cfg)
        train_data, val_data, test_data, max_node_num = load_all_datasets(
            cfg, device, include_test=include_test
        )

    return train_data, val_data, test_data, max_node_num


def main(cfg):
    set_seed(cfg)
    log_start(__file__)

    if cfg.batching.save_on_disk:
        device = get_device(cfg)
        include_test = not cfg.training.validation_only
        train_data, val_data, test_data, max_node_num = load_all_datasets(
            cfg, device, include_test=include_test
        )

        out_dir = cfg.batching._preprocessed_graphs_dir
        file_suffix = "" if include_test else "_validation_only"
        out_file = os.path.join(out_dir, f"torch_graphs{file_suffix}.pkl")
        os.makedirs(out_dir, exist_ok=True)
        log(f"Saving preprocessed graphs to {out_file}...")
        torch.save((train_data, val_data, test_data, max_node_num), out_file)

    else:
        log("Not saving to disk, skipping this task.")
