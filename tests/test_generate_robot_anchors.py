from __future__ import annotations

import numpy as np
import pytest

from geort.anchor.generate_robot_anchors import build_paired_anchors, main
from geort.anchor.mining import LEVEL_FRACTIONS, MinedHumanAnchors


def _human_anchors() -> MinedHumanAnchors:
    rows = 50
    human_points = np.column_stack(
        (np.arange(rows, dtype=np.float64), np.zeros(rows), np.ones(rows))
    )
    frames = np.broadcast_to(human_points[:, None, :], (rows, 21, 3)).copy()
    finger_indices = np.repeat(np.arange(5, dtype=np.int64), 10)
    finger_names = np.repeat(
        np.array(["thumb", "index", "middle", "ring", "pinky"]), 10
    )
    anchor_types = np.tile(np.repeat(np.array(["lateral", "bending"]), 5), 5)
    return MinedHumanAnchors(
        human_frames=frames,
        human_points=human_points,
        source_indices=np.arange(rows, dtype=np.int64),
        finger_indices=finger_indices,
        finger_names=finger_names,
        anchor_types=anchor_types,
        levels=np.tile(np.arange(5, dtype=np.int64), 10),
        trajectory_t=np.tile(LEVEL_FRACTIONS, 10),
        target_parameters=np.arange(rows, dtype=np.float64),
        observed_parameters=np.arange(rows, dtype=np.float64),
        candidate_counts=np.repeat(5, rows),
        support_counts=np.repeat(5, rows),
        group_metadata={},
    )


def test_paired_builder_interpolates_by_group_to_exactly_750_rows() -> None:
    human = _human_anchors()
    robot_knots = np.zeros((50, 20), dtype=np.float64)
    robot_knots[:, 0] = np.arange(50, dtype=np.float64)

    def fk(qpos: np.ndarray, finger_index: int) -> np.ndarray:
        return np.array([qpos[0], float(finger_index), 0.0])

    paired = build_paired_anchors(human, robot_knots, fk)

    assert paired.human_tip_contexts.shape == (750, 5, 3)
    assert paired.human_points.shape == (750, 3)
    assert paired.robot_points.shape == (750, 3)
    assert paired.robot_qpos.shape == (750, 20)
    assert paired.finger_indices.shape == (750,)
    assert paired.anchor_types.shape == (750,)
    assert np.count_nonzero(paired.anchor_types == "lateral") == 250
    assert np.count_nonzero(paired.anchor_types == "bending") == 500
    assert paired.human_points[0, 0] == 0.0
    assert paired.human_tip_contexts[0, 0, 0] == 0.0
    assert paired.human_points[49, 0] == 4.0
    assert paired.human_points[50, 0] == 5.0
    assert paired.human_points[149, 0] == 9.0
    assert paired.robot_points[0].tolist() == [0.0, 0.0, 0.0]
    assert paired.robot_points[149].tolist() == [9.0, 0.0, 0.0]
    assert np.all(paired.finger_indices[:150] == 0)


def test_paired_builder_keeps_human_robot_match_by_finger_type_level_only() -> None:
    human = _human_anchors()
    robot_knots = np.zeros((50, 4), dtype=np.float64)
    robot_knots[:, 0] = np.arange(50, dtype=np.float64)

    paired = build_paired_anchors(
        human,
        robot_knots,
        lambda qpos, finger_index: np.array([qpos[0], finger_index, 0.0]),
    )

    group_starts = [0, 50, 150, 200, 300, 350, 450, 500, 600, 650]
    assert paired.human_points[group_starts, 0].tolist() == [
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        45.0,
    ]


def test_main_rejects_missing_manifest_before_loading_fk(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="run geort.data.prepare first"):
        main(
            [
                "--hand",
                "custom_right",
                "--human-anchors",
                str(tmp_path / "human.npz"),
                "--output",
                str(tmp_path / "paired.npz"),
                "--manifest",
                str(tmp_path / "missing.json"),
            ]
        )
