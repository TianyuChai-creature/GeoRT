from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from geort.contact.auto_label_contacts import PAIR_NAMES
from geort.contact.contact_model import (
    build_arg_parser,
    create_pair_trainers,
    fit_pair_scaler,
    load_contact_dataset,
    train_contact_models,
)


def _write_labels(path: Path, *, held_out_offset: float = 0.0) -> Path:
    rows = 160
    labels = np.tile((np.arange(rows) % 2)[:, None], (1, 4)).astype(np.int8)
    labels[10:15] = -1
    held_out = np.zeros((rows, 4), dtype=bool)
    held_out[120:] = labels[120:] >= 0
    signs = np.where(labels == 1, 1.0, -1.0)
    features = np.repeat(signs[:, :, None], 6, axis=2).astype(np.float32)
    features[held_out] += held_out_offset
    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        held_out=held_out,
        pair_names=np.asarray(PAIR_NAMES),
        pair_landmarks=np.asarray(((4, 8), (4, 12), (4, 16), (4, 20))),
        frame_indices=np.arange(rows),
    )
    return path


def test_load_contact_dataset_preserves_pair_and_time_split_contract(tmp_path: Path) -> None:
    dataset = load_contact_dataset(_write_labels(tmp_path / "labels.npz"))

    assert dataset.features.shape == (160, 4, 6)
    assert dataset.labels.shape == (160, 4)
    assert dataset.pair_names == PAIR_NAMES
    assert dataset.frame_indices[dataset.held_out[:, 0]][0] == 120
    assert np.all(np.diff(dataset.frame_indices[dataset.held_out[:, 0]]) == 1)


def test_pair_scaler_fits_clear_training_frames_only(tmp_path: Path) -> None:
    dataset = load_contact_dataset(
        _write_labels(tmp_path / "labels.npz", held_out_offset=100.0)
    )

    scaler = fit_pair_scaler(dataset, pair_index=0)

    np.testing.assert_allclose(scaler.mean, np.zeros(6), atol=0.05)
    np.testing.assert_allclose(scaler.scale, np.ones(6), atol=0.05)


def test_each_pair_has_a_distinct_model_and_optimizer() -> None:
    models, optimizers = create_pair_trainers(hidden_dims=(8, 4), learning_rate=1e-3)

    assert tuple(models) == PAIR_NAMES
    assert tuple(optimizers) == PAIR_NAMES
    parameter_ids = [
        {id(parameter) for group in optimizers[name].param_groups for parameter in group["params"]}
        for name in PAIR_NAMES
    ]
    assert all(parameter_ids[index].isdisjoint(parameter_ids[other]) for index in range(4) for other in range(index + 1, 4))


def test_training_writes_four_models_scalers_and_temporal_holdout_report(tmp_path: Path) -> None:
    labels_path = _write_labels(tmp_path / "labels.npz")

    artifacts = train_contact_models(
        labels_path=labels_path,
        output_dir=tmp_path / "contact_checkpoint",
        epochs=5,
        learning_rate=1e-2,
        batch_size=32,
        hidden_dims=(8, 4),
        seed=3,
        device="cpu",
    )

    assert artifacts.checkpoint_path.exists()
    assert artifacts.report_path.exists()
    checkpoint = torch.load(artifacts.checkpoint_path, map_location="cpu", weights_only=True)
    assert tuple(checkpoint["pairs"]) == PAIR_NAMES
    assert all(checkpoint["pairs"][name]["scaler_mean"].shape == (6,) for name in PAIR_NAMES)
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert [pair["name"] for pair in report["pairs"]] == list(PAIR_NAMES)
    assert all(pair["history"]["train_bce"][-1] < pair["history"]["train_bce"][0] for pair in report["pairs"])
    assert all(pair["held_out"]["positive_count"] > 0 for pair in report["pairs"])
    assert all(pair["held_out"]["negative_count"] > 0 for pair in report["pairs"])
    assert all("f1" in pair["held_out"] for pair in report["pairs"])


def test_training_cli_uses_the_approved_step7_defaults() -> None:
    args = build_arg_parser().parse_args(["--labels", "data/contact_labels_right.npz", "--tag", "right_step7"])

    assert args.epochs == 20
    assert args.learning_rate == 1e-4
    assert args.batch_size == 2048
    assert args.hidden_dims == (64, 32)
