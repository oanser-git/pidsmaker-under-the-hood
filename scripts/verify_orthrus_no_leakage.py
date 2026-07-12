#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_artifacts = (
        repo_root / "artifacts/orthrus/extracted/no_leakage/THEIA_E3/artifacts"
    )
    parser = argparse.ArgumentParser(
        description="Verify the five strict no-leakage Orthrus artifact sets."
    )
    parser.add_argument(
        "artifact_dir",
        nargs="?",
        type=Path,
        default=Path(os.environ.get("ORTHRUS_NO_LEAKAGE_ARTIFACTS", default_artifacts)),
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Verify selected checkpoints/scores/losses without requiring feat_inference.",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(root, container_path):
    prefix = "/home/artifacts/"
    if not container_path.startswith(prefix):
        raise ValueError(f"Unexpected container artifact path: {container_path}")
    return root / container_path[len(prefix) :]


def tree_stats(path):
    file_count = 0
    total_bytes = 0
    for directory, _, files in os.walk(str(path)):
        for name in files:
            file_path = Path(directory) / name
            if file_path.is_file():
                file_count += 1
                total_bytes += file_path.stat().st_size
    return {"file_count": file_count, "total_bytes": total_bytes}


def main():
    args = parse_args()
    artifact_root = args.artifact_dir.resolve()
    summary_dir = (
        Path(__file__).resolve().parents[1]
        / "orthrus/no_leakage/t5-5524660/summaries"
    )
    inventory_path = summary_dir.parent / "artifact-inventory.json"
    with inventory_path.open() as file:
        inventory = json.load(file)

    if not args.compact:
        actual_tree_names = {path.name for path in artifact_root.iterdir() if path.is_dir()}
        expected_tree_names = set(inventory["trees"])
        if actual_tree_names != expected_tree_names:
            raise ValueError(
                f"Artifact tree mismatch: {actual_tree_names} != {expected_tree_names}"
            )
        for tree_name, expected_stats in inventory["trees"].items():
            actual_stats = tree_stats(artifact_root / tree_name)
            if actual_stats != expected_stats:
                raise ValueError(
                    f"Artifact inventory mismatch for {tree_name}: "
                    f"{actual_stats} != {expected_stats}"
                )

    for seed in range(5):
        summary_path = summary_dir / f"seed-{seed}-test-summary.json"
        with summary_path.open() as file:
            summary = json.load(file)

        epoch = int(summary["selected_epoch"])
        checkpoint_dir = artifact_path(artifact_root, summary["selected_checkpoint"])
        state_dict = checkpoint_dir / "state_dict.pkl"
        score_file = artifact_path(artifact_root, summary["scores_file"])
        training_root = checkpoint_dir.parents[1]
        val_losses = training_root / f"edge_losses/val/model_epoch_{epoch}"
        test_losses = training_root / f"edge_losses/test/model_epoch_{epoch}"

        required_paths = [state_dict, score_file, val_losses, test_losses]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Seed {seed} is missing: {missing}")

        actual_sha256 = sha256(state_dict)
        expected_sha256 = summary["selected_checkpoint_sha256"]
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Seed {seed} checkpoint SHA-256 mismatch: "
                f"{actual_sha256} != {expected_sha256}"
            )

        expected_score = inventory["selected_scores"][str(seed)]
        if score_file.stat().st_size != expected_score["size"]:
            raise ValueError(f"Seed {seed} score file size mismatch.")
        score_sha256 = sha256(score_file)
        if score_sha256 != expected_score["sha256"]:
            raise ValueError(f"Seed {seed} score file SHA-256 mismatch.")
        if not any(val_losses.iterdir()) or not any(test_losses.iterdir()):
            raise ValueError(f"Seed {seed} has an empty edge-loss directory.")

        print(
            f"seed={seed} epoch={epoch} adp={summary['adp_score']} "
            f"checkpoint_sha256={actual_sha256}"
        )

    if args.compact:
        print("All five compact no-leakage Orthrus selections are valid.")
    else:
        print("Artifact inventory and all five no-leakage Orthrus selections are valid.")


if __name__ == "__main__":
    main()
