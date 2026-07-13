from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geort.data.prepare import (
    denormalize_finger_points,
    fit_finger_normalization,
    normalize_finger_points,
    prepare_dataset,
)


FINGERS = ["thumb", "thumb", "index", "index"]


def sample_points() -> np.ndarray:
    return np.array(
        [
            [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [10.0, -2.0, 5.0], [14.0, 0.0, 7.0]],
            [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0], [12.0, -1.0, 6.0], [13.0, -2.0, 5.0]],
        ],
        dtype=np.float32,
    )


def test_each_finger_is_inside_unit_cube_and_longest_axis_reaches_bounds() -> None:
    points = sample_points()
    stats = fit_finger_normalization(points, FINGERS)

    normalized = normalize_finger_points(points, FINGERS, stats)

    for finger in ("thumb", "index"):
        selected = normalized[:, [i for i, name in enumerate(FINGERS) if name == finger]]
        assert selected.min() >= -1.0 - 1e-6
        assert selected.max() <= 1.0 + 1e-6
        axis_min = selected.min(axis=(0, 1))
        axis_max = selected.max(axis=(0, 1))
        assert np.any(np.isclose(axis_min, -1.0, atol=1e-6))
        assert np.any(np.isclose(axis_max, 1.0, atol=1e-6))


def test_finger_normalization_uses_one_isotropic_scale() -> None:
    stats = fit_finger_normalization(sample_points(), FINGERS)

    assert isinstance(stats["thumb"]["scale"], float)
    assert stats["thumb"]["scale"] == pytest.approx(1.0)
    assert stats["index"]["scale"] == pytest.approx(2.0)


def test_normalization_round_trip_error_is_below_tolerance() -> None:
    points = sample_points()
    stats = fit_finger_normalization(points, FINGERS)

    restored = denormalize_finger_points(
        normalize_finger_points(points, FINGERS, stats),
        FINGERS,
        stats,
    )

    assert np.max(np.abs(restored - points)) < 1e-6


def test_prepare_dataset_writes_compact_npz_and_manifest(tmp_path: Path) -> None:
    human_path = tmp_path / "hts_right.npy"
    robot_path = tmp_path / "custom_right.npz"
    output_path = tmp_path / "hts_right_prepared.npz"
    manifest_path = tmp_path / "hts_right_prepared.json"
    np.save(human_path, sample_points())
    robot_keypoints = {
        "F1-PIP": sample_points()[:, 0],
        "F1-DIP": sample_points()[:, 1],
        "F2-PIP": sample_points()[:, 2],
        "F2-DIP": sample_points()[:, 3],
    }
    np.savez(robot_path, qpos=np.zeros((2, 2)), keypoint=robot_keypoints)
    config = {
        "name": "custom_right",
        "fingertip_link": [
            {"name": "thumb_pip", "link": "F1-PIP", "joint": [], "center_offset": [0, 0, 0], "human_hand_id": 0, "finger": "thumb", "keypoint_type": "pip"},
            {"name": "thumb_tip", "link": "F1-DIP", "joint": [], "center_offset": [0, 0, 0], "human_hand_id": 1, "finger": "thumb", "keypoint_type": "tip"},
            {"name": "index_pip", "link": "F2-PIP", "joint": [], "center_offset": [0, 0, 0], "human_hand_id": 2, "finger": "index", "keypoint_type": "pip"},
            {"name": "index_tip", "link": "F2-DIP", "joint": [], "center_offset": [0, 0, 0], "human_hand_id": 3, "finger": "index", "keypoint_type": "tip"},
        ],
        "joint_order": [],
    }

    prepare_dataset(
        human_path=human_path,
        robot_path=robot_path,
        config=config,
        output_path=output_path,
        manifest_path=manifest_path,
    )

    prepared = np.load(output_path)
    assert prepared["human_points"].shape == (2, 4, 3)
    assert prepared["robot_points"].shape == (2, 4, 3)
    assert prepared["keypoint_names"].tolist() == [
        "thumb_pip", "thumb_tip", "index_pip", "index_tip"
    ]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["prepared_data"] == output_path.name
    assert manifest["anchors"] is None
    assert manifest["contact"] is None
    assert set(manifest["human"]["normalization"]) == {"thumb", "index"}
    assert set(manifest["robot"]["normalization"]) == {"thumb", "index"}
