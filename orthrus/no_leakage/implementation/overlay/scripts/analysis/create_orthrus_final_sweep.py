#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the one-run sweep used for frozen Orthrus test evaluation."
    )
    parser.add_argument("--window", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "config/experiments/tuning/systems/theia_e3/"
            "tuning_orthrus_no_leakage_final.yml"
        ),
    )
    return parser.parse_args()


def build_sweep(args):
    if args.epoch < 0:
        raise ValueError("Selected epoch must be non-negative.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.sha256):
        raise ValueError("Checkpoint SHA-256 must contain exactly 64 hexadecimal characters.")

    return {
        "method": "grid",
        "metric": {"name": "adp_score", "goal": "maximize"},
        "parameters": {
            "construction.time_window_size": {"values": [args.window]},
            "training.lr": {"values": [args.lr]},
            "training.node_hid_dim": {"values": [args.hidden]},
            "training.node_out_dim": {"values": [-1]},
            "training.encoder.dropout": {"values": [args.dropout]},
            "evaluation.selected_epoch": {"values": [args.epoch]},
            "evaluation.expected_checkpoint_sha256": {
                "values": [args.sha256.lower()]
            },
        },
    }


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        yaml.safe_dump(build_sweep(args), file, sort_keys=False)
    print(args.output)


if __name__ == "__main__":
    main()
