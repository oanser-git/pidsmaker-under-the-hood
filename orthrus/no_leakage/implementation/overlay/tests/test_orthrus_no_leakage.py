import os

import pytest
import torch

from pidsmaker import main
from pidsmaker.config import get_runtime_required_args, get_yml_cfg, set_task_paths
from pidsmaker.experiments.tuning import fuse_cfg_with_sweep_cfg, get_tuning_sweep_cfg
from pidsmaker.featurization.featurization_utils import get_splits_to_train_featurization
from pidsmaker.tasks import batching, evaluation
from pidsmaker.tasks.feat_inference import temporal_edge_sort_key
from pidsmaker.utils.data_utils import assert_causal_neighbor_times, file_sha256


def load_cfg(model, dataset, artifact_dir, extra_args=None):
    args, unknown = get_runtime_required_args(
        return_unknown_args=True,
        args=[model, dataset],
    )
    assert unknown == []
    args.artifact_dir = str(artifact_dir)
    for key, value in (extra_args or {}).items():
        args.__dict__[key] = value
    return get_yml_cfg(args)


def test_no_leakage_config_preserves_orthrus_architecture(tmp_path):
    original = load_cfg("orthrus", "THEIA_E3", tmp_path / "original")
    clean = load_cfg("orthrus_no_leakage", "THEIA_E3", tmp_path / "clean")

    assert clean.training.encoder.used_methods == original.training.encoder.used_methods
    assert clean.training.decoder.used_methods == original.training.decoder.used_methods
    assert clean.training.node_hid_dim == original.training.node_hid_dim
    assert clean.training.node_out_dim == original.training.node_out_dim
    assert clean.training.num_epochs == original.training.num_epochs

    assert original.featurization.training_split == "all"
    assert clean.featurization.training_split == "train"
    assert original.evaluation.node_evaluation.use_kmeans is True
    assert clean.evaluation.node_evaluation.use_kmeans is False
    assert clean.training.validation_only is True
    assert clean.construction.enforce_chronological_splits is True
    assert clean.feat_inference.sort_edges_by_time is True
    assert clean.batching.intra_graph_batching.tgn_last_neighbor.assert_causal_neighbors is True
    assert get_splits_to_train_featurization(clean) == ["train"]


def test_theia_e3_is_accepted_as_chronological(tmp_path):
    cfg = load_cfg("orthrus_no_leakage", "THEIA_E3", tmp_path)
    assert max(cfg.dataset.train_dates) < min(cfg.dataset.val_dates)
    assert max(cfg.dataset.val_dates) < min(cfg.dataset.test_dates)


def test_cadets_e3_is_rejected_as_non_chronological(tmp_path):
    with pytest.raises(ValueError, match="Strict chronological mode"):
        load_cfg("orthrus_no_leakage", "CADETS_E3", tmp_path)


def test_clean_hpo_uses_validation_metric_and_same_grid(tmp_path):
    cfg = load_cfg(
        "orthrus_no_leakage",
        "THEIA_E3",
        tmp_path,
        extra_args={"tuning_mode": "hyperparameters"},
    )
    sweep = get_tuning_sweep_cfg(cfg)

    assert sweep["metric"] == {"name": "best_val_loss", "goal": "minimize"}
    assert sweep["parameters"]["construction.time_window_size"]["values"] == [5.0, 15.0]
    assert sweep["parameters"]["training.lr"]["values"] == [0.001, 0.0001]
    assert sweep["parameters"]["training.node_hid_dim"]["values"] == [64, 128]


def test_sweep_and_final_replay_resolve_to_same_training_path(tmp_path):
    sweep_values = {
        "construction.time_window_size": 15.0,
        "training.lr": 0.0001,
        "training.node_hid_dim": 128,
        "training.node_out_dim": -1,
        "training.encoder.dropout": 0.3,
    }
    base = load_cfg("orthrus_no_leakage", "THEIA_E3", tmp_path)
    swept = fuse_cfg_with_sweep_cfg(base, sweep_values)
    replayed = load_cfg(
        "orthrus_no_leakage",
        "THEIA_E3",
        tmp_path,
        extra_args=sweep_values,
    )

    assert swept.training.node_out_dim == 128
    assert replayed.training.node_out_dim == 128
    assert swept.training._task_path == replayed.training._task_path


def test_split_boundaries_are_part_of_construction_hash(tmp_path):
    cfg = load_cfg("orthrus_no_leakage", "THEIA_E3", tmp_path)
    cfg.dataset.train_dates = ["2018-04-02", "2018-04-03"]
    cfg.dataset.val_dates = ["2018-04-04"]
    cfg.dataset.test_dates = ["2018-04-05"]
    set_task_paths(cfg)
    first_path = cfg.construction._task_path

    cfg.dataset.train_dates = ["2018-04-02"]
    cfg.dataset.val_dates = ["2018-04-03", "2018-04-04"]
    set_task_paths(cfg)

    assert cfg.construction._task_path != first_path


def test_pipeline_modes_do_not_mix_hpo_and_final_evaluation(tmp_path):
    cfg = load_cfg("orthrus_no_leakage", "THEIA_E3", tmp_path)
    assert "training" in main.get_pipeline_tasks(cfg)
    assert "evaluation" not in main.get_pipeline_tasks(cfg)
    assert "triage" not in main.get_pipeline_tasks(cfg)

    cfg.evaluation.selected_epoch = 3
    assert main.get_pipeline_tasks(cfg) == ["evaluation"]


def test_validation_only_graph_loading_excludes_test(tmp_path, monkeypatch):
    cfg = load_cfg("orthrus_no_leakage", "THEIA_E3", tmp_path)
    calls = []
    expected = ("train", "val", [], 5)

    monkeypatch.setattr(batching, "get_device", lambda cfg: "cpu")
    monkeypatch.setattr(
        batching,
        "load_all_datasets",
        lambda cfg, device, include_test: calls.append(include_test) or expected,
    )

    assert batching.get_preprocessed_graphs(cfg, include_test=False) == expected
    assert calls == [False]


def test_temporal_edge_sort_is_deterministic():
    edges = [
        (2, 3, 0, {"time": 20, "event_uuid": "b", "label": "write"}),
        (1, 4, 0, {"time": 10, "event_uuid": "c", "label": "read"}),
        (1, 2, 0, {"time": 10, "event_uuid": "a", "label": "read"}),
    ]

    ordered = sorted(edges, key=temporal_edge_sort_key)
    assert [(edge[3]["time"], edge[3]["event_uuid"]) for edge in ordered] == [
        (10, "a"),
        (10, "c"),
        (20, "b"),
    ]


def test_causal_neighbor_assertion_rejects_future_context():
    assert_causal_neighbor_times(
        torch.tensor([10, 20]),
        torch.tensor([0, 1]),
        torch.tensor([20, 30]),
        torch.tensor([2, 3]),
    )

    with pytest.raises(ValueError, match="Non-causal TGN context"):
        assert_causal_neighbor_times(
            torch.tensor([10, 21]),
            torch.tensor([0, 1]),
            torch.tensor([20, 30]),
            torch.tensor([2, 3]),
        )

    with pytest.raises(ValueError, match="Non-causal TGN context"):
        assert_causal_neighbor_times(
            torch.tensor([20]),
            torch.tensor([2]),
            torch.tensor([20]),
            torch.tensor([2]),
        )


def test_final_evaluation_refuses_to_retrain_missing_checkpoint(tmp_path):
    cfg = load_cfg(
        "orthrus_no_leakage",
        "THEIA_E3",
        tmp_path,
        extra_args={
            "evaluation.selected_epoch": 3,
            "evaluation.expected_checkpoint_sha256": "0" * 64,
        },
    )

    with pytest.raises(FileNotFoundError, match="never retrains"):
        evaluation.infer_selected_checkpoint(cfg)


def test_final_evaluation_rejects_checkpoint_digest_mismatch(tmp_path):
    cfg = load_cfg("orthrus_no_leakage", "THEIA_E3", tmp_path)
    checkpoint = os.path.join(cfg.training._trained_models_dir, "model_epoch_3")
    os.makedirs(checkpoint, exist_ok=True)
    with open(os.path.join(checkpoint, "state_dict.pkl"), "wb") as file:
        file.write(b"wrong-checkpoint")
    cfg.evaluation.selected_epoch = 3
    cfg.evaluation.expected_checkpoint_sha256 = "0" * 64

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluation.infer_selected_checkpoint(cfg)


def test_final_evaluation_loads_one_frozen_checkpoint(tmp_path, monkeypatch):
    cfg = load_cfg("orthrus_no_leakage", "THEIA_E3", tmp_path)
    checkpoint = os.path.join(cfg.training._trained_models_dir, "model_epoch_3")
    os.makedirs(checkpoint, exist_ok=True)
    with open(os.path.join(checkpoint, "state_dict.pkl"), "wb") as file:
        file.write(b"frozen-checkpoint")
    cfg.evaluation.selected_epoch = 3
    cfg.evaluation.expected_checkpoint_sha256 = file_sha256(
        os.path.join(checkpoint, "state_dict.pkl")
    )

    sample = object()
    train_data, val_data, test_data = [[sample]], [[object()]], [[object()]]
    model = object()
    calls = []

    monkeypatch.setattr(
        evaluation,
        "get_preprocessed_graphs",
        lambda cfg: (train_data, val_data, test_data, 5),
    )
    monkeypatch.setattr(evaluation, "get_device", lambda cfg: "cpu")
    monkeypatch.setattr(evaluation, "build_model", lambda **kwargs: model)
    monkeypatch.setattr(
        evaluation,
        "load_model",
        lambda loaded_model, path, cfg, map_location=None: loaded_model,
    )
    monkeypatch.setattr(
        evaluation.inference_loop,
        "main",
        lambda **kwargs: calls.append(kwargs) or {},
    )
    monkeypatch.setattr(evaluation.wandb, "log", lambda payload: None)

    evaluation.infer_selected_checkpoint(cfg)

    assert len(calls) == 1
    assert calls[0]["model"] is model
    assert calls[0]["epoch"] == 3
    assert calls[0]["split"] == "all"
