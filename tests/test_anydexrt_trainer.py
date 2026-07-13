from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import torch

from geort import trainer
from geort.utils import config_utils


def _function(module, name: str):
    assert hasattr(module, name), f"{module.__name__}.{name} is not implemented"
    return getattr(module, name)


def test_tip_finger_groups_use_one_3d_input_per_finger() -> None:
    info = {
        "tip_indices": [1, 3],
        "finger": ["thumb", "thumb", "index", "index"],
        "finger_groups": [
            {"finger": "thumb", "keypoint_indices": [0, 1], "joint_indices": [0, 1, 2, 3]},
            {"finger": "index", "keypoint_indices": [2, 3], "joint_indices": [4, 5, 6, 7]},
        ],
    }

    groups = _function(config_utils, "build_tip_finger_groups")(info)

    assert groups == [
        {"finger": "thumb", "keypoint_indices": [0], "joint_indices": [0, 1, 2, 3]},
        {"finger": "index", "keypoint_indices": [1], "joint_indices": [4, 5, 6, 7]},
    ]


def test_anydexrt_loss_bundle_is_optimal_for_identity_tip_mapping() -> None:
    human = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    delta = torch.tensor(
        [
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        ]
    )

    losses = _function(trainer, "compute_anydexrt_losses")(
        human_points=human,
        mapped_points=human,
        jittered_human_points=human + delta,
        jittered_mapped_points=human + delta,
        robot_cloud=human,
        n_pairs=8,
    )

    assert losses["partial_chamfer"].item() == 0.0
    assert losses["distance"].item() == 0.0
    assert torch.allclose(losses["motion"], torch.tensor(-1.0), atol=1e-6)


def test_load_prepared_training_data_checks_tip_contract(tmp_path: Path) -> None:
    np.savez(
        tmp_path / "right_prepared.npz",
        human_points=np.zeros((4, 2, 3), dtype=np.float32),
        robot_points=np.zeros((5, 2, 3), dtype=np.float32),
        keypoint_names=np.array(["thumb_tip", "index_tip"]),
        finger_names=np.array(["thumb", "index"]),
        human_ids=np.array([4, 8]),
    )
    manifest = {
        "schema_version": 1,
        "prepared_data": "right_prepared.npz",
        "config": "custom_right",
        "keypoint_names": ["thumb_tip", "index_tip"],
        "finger_names": ["thumb", "index"],
        "human": {"normalization": {
            "thumb": {"center": [0, 0, 0], "scale": 1.0},
            "index": {"center": [0, 0, 0], "scale": 1.0},
        }},
        "robot": {"normalization": {
            "thumb": {"center": [0, 0, 0], "scale": 1.0},
            "index": {"center": [0, 0, 0], "scale": 1.0},
        }},
        "anchors": None,
        "contact": None,
    }
    manifest_path = tmp_path / "right_prepared.json"
    manifest_path.write_text(json.dumps(manifest))

    with warnings.catch_warnings(record=True) as caught:
        data = _function(trainer, "load_prepared_training_data")(
            manifest_path, expected_config="custom_right"
        )

    assert data.human_points.shape == (4, 2, 3)
    assert data.robot_points.shape == (5, 2, 3)
    assert data.human_ids == [4, 8]
    assert data.anchor_points is None
    assert any("anchors" in str(item.message).lower() for item in caught)


def test_step5_cli_and_defaults_match_manual() -> None:
    parser = _function(trainer, "build_arg_parser")()
    args = parser.parse_args(
        ["-hand", "custom_right", "-human_data", "hts_right_prepared", "-ckpt_tag", "step5"]
    )

    assert set(vars(args)) == {"hand", "human_data", "ckpt_tag", "save_every"}
    assert trainer.DEFAULT_EPOCHS == 20
    assert trainer.DEFAULT_BATCH_SIZE == 2048
    assert trainer.DEFAULT_LEARNING_RATE == 1e-4


def test_manifest_anchor_points_are_normalized_when_enabled(tmp_path: Path) -> None:
    np.savez(
        tmp_path / "prepared.npz",
        human_points=np.zeros((2, 1, 3), dtype=np.float32),
        robot_points=np.zeros((2, 1, 3), dtype=np.float32),
        keypoint_names=np.array(["thumb_tip"]),
        finger_names=np.array(["thumb"]),
        human_ids=np.array([4]),
    )
    np.savez(
        tmp_path / "anchors.npz",
        human_points=np.array([[[3.0, 3.0, 3.0]]], dtype=np.float32),
        robot_points=np.array([[[5.0, 5.0, 5.0]]], dtype=np.float32),
    )
    manifest = {
        "schema_version": 1,
        "prepared_data": "prepared.npz",
        "config": "custom_right",
        "keypoint_names": ["thumb_tip"],
        "finger_names": ["thumb"],
        "human": {"normalization": {
            "thumb": {"center": [1, 1, 1], "scale": 2.0}
        }},
        "robot": {"normalization": {
            "thumb": {"center": [1, 1, 1], "scale": 4.0}
        }},
        "anchors": {"path": "anchors.npz", "normalized": False},
        "contact": None,
    }
    manifest_path = tmp_path / "prepared.json"
    manifest_path.write_text(json.dumps(manifest))

    data = trainer.load_prepared_training_data(
        manifest_path, expected_config="custom_right"
    )

    assert data.anchor_points is not None
    human, robot = data.anchor_points
    assert np.allclose(human, 1.0)
    assert np.allclose(robot, 1.0)


def test_trainer_source_does_not_retain_old_objectives() -> None:
    source = Path(trainer.__file__).read_text()
    for forbidden in (
        "chamfer_distance",
        "curvature_loss",
        "w_chamfer",
        "w_curvature",
    ):
        assert forbidden not in source
