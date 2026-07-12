#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build the strict no-leakage Orthrus artifact inventory."
    )
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=repo_root / "orthrus/no_leakage/t5-5524660/summaries",
    )
    parser.add_argument("--output", type=Path, required=True)
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
    trees = {}
    for path in sorted(artifact_root.iterdir()):
        if path.is_dir():
            trees[path.name] = tree_stats(path)

    selected_scores = {}
    for seed in range(5):
        with (args.summary_dir / f"seed-{seed}-test-summary.json").open() as file:
            summary = json.load(file)
        score_file = artifact_path(artifact_root, summary["scores_file"])
        selected_scores[str(seed)] = {
            "path": str(score_file.relative_to(artifact_root)),
            "sha256": sha256(score_file),
            "size": score_file.stat().st_size,
        }

    inventory = {
        "schema_version": 1,
        "dataset": "THEIA_E3",
        "variant": "no-leakage",
        "trees": trees,
        "selected_scores": selected_scores,
    }
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary_output), str(args.output))


if __name__ == "__main__":
    main()
