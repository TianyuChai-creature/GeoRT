from __future__ import annotations

import numpy as np
import pytest

from geort.dataset import RobotKinematicsDataset


def test_robot_rotation_cloud_requires_new_rotation_field(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, qpos=np.zeros((2, 20)), keypoint={"F1-R-DIP": np.zeros((2, 3))})
    dataset = RobotKinematicsDataset(path, ["F1-R-DIP"])
    with pytest.raises(ValueError, match="link_rotation"):
        dataset.export_robot_link_rotations(["F1-R-DIP"])


def test_robot_rotation_cloud_reads_link_rotation_field(tmp_path):
    path = tmp_path / "rotations.npz"
    rotations = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    np.savez(
        path,
        qpos=np.zeros((2, 20)),
        keypoint={"F1-R-DIP": np.zeros((2, 3))},
        link_rotation={"F1-R-DIP": rotations},
    )
    dataset = RobotKinematicsDataset(path, ["F1-R-DIP"])
    np.testing.assert_array_equal(
        dataset.export_robot_link_rotations(["F1-R-DIP"]), rotations[None, ...]
    )
