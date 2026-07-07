from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "geort" / "mocap" / "build_target_cloud.py"
spec = importlib.util.spec_from_file_location("build_target_cloud", MODULE_PATH)
assert spec is not None and spec.loader is not None
build_target_cloud = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = build_target_cloud
spec.loader.exec_module(build_target_cloud)

AngleProxySpec = build_target_cloud.AngleProxySpec
MoldJoint = build_target_cloud.MoldJoint
apply_mold = build_target_cloud.apply_mold
build_angle_proxy_specs = build_target_cloud.build_angle_proxy_specs
build_mold = build_target_cloud.build_mold
density_cap_qpos = build_target_cloud.density_cap_qpos
extract_angles = build_target_cloud.extract_angles
save_robot_kinematics_npz = build_target_cloud.save_robot_kinematics_npz
build_target_cloud_file = build_target_cloud.build_target_cloud
boost_fist_mcp1_qpos = build_target_cloud.boost_fist_mcp1_qpos


def make_frames(count: int) -> np.ndarray:
    frames = np.zeros((count, 21, 3), dtype=np.float32)
    for t in range(count):
        frames[t, :, 0] = np.linspace(0.0, 0.2, 21)
        frames[t, :, 1] = float(t) * 0.001
        frames[t, :, 2] = np.linspace(0.0, 0.1, 21)
    return frames


def test_build_angle_proxy_specs_matches_custom_right_dof() -> None:
    config = {
        "joint_order": [
            "F1-R-MCP2",
            "F1-R-MCP1",
            "F1-R-PIP",
            "F1-R-DIP",
            "F2-R-MCP2",
            "F2-R-MCP1",
            "F2-R-PIP",
            "F2-R-DIP",
        ]
    }

    specs = build_angle_proxy_specs(config)

    assert [spec.joint_name for spec in specs] == config["joint_order"]
    assert [spec.finger for spec in specs[:4]] == ["thumb"] * 4
    assert [spec.kind for spec in specs[:4]] == ["thumb_azimuth", "thumb_elevation", "flexion", "flexion"]
    assert [spec.finger for spec in specs[4:]] == ["index"] * 4
    assert [spec.kind for spec in specs[4:]] == ["aa", "flexion", "flexion", "flexion"]


def test_extract_angles_returns_one_column_per_proxy() -> None:
    frames = make_frames(4)
    specs = [
        AngleProxySpec("index_aa", "index", "aa", (5, 6, 8)),
        AngleProxySpec("index_flex", "index", "flexion", (5, 6, 8)),
    ]

    angles = extract_angles(frames, specs)

    assert angles.shape == (4, 2)
    assert np.isfinite(angles).all()


def test_build_mold_marks_noise_side_as_invalid_and_maps_rest_to_default() -> None:
    rest = np.array(
        [
            [0.0],
            [0.001],
            [-0.001],
            [0.0],
        ],
        dtype=np.float32,
    )
    motion = np.array(
        [
            [-0.0005],
            [0.0],
            [0.5],
            [1.0],
        ],
        dtype=np.float32,
    )

    mold = build_mold(
        rest_angles=rest,
        motion_angles=motion,
        joint_names=["joint_a"],
        q_low=np.array([-1.0], dtype=np.float32),
        q_high=np.array([2.0], dtype=np.float32),
        q_default=np.array([0.25], dtype=np.float32),
        pin_k=2.0,
    )

    joint = mold.joints[0]
    assert joint.left_valid is False
    assert joint.right_valid is True
    assert apply_mold(np.array([[joint.theta_rest]], dtype=np.float32), mold)[0, 0] == pytest.approx(0.25)
    assert apply_mold(np.array([[joint.theta_hi]], dtype=np.float32), mold)[0, 0] == pytest.approx(2.0)


def test_default_qpos_prefers_urdf_zero_when_within_joint_limits() -> None:
    q_low = np.array([-0.61, 0.0, 1.0], dtype=np.float32)
    q_high = np.array([0.61, 1.92, 2.0], dtype=np.float32)

    q_default = build_target_cloud.default_qpos_from_limits(q_low, q_high)

    assert np.allclose(q_default, [0.0, 0.0, 1.5])


def test_flexion_mold_maps_smaller_human_angle_to_higher_robot_curl() -> None:
    rest = np.array([[3.0], [3.02], [2.98], [3.01]], dtype=np.float32)
    motion = np.array([[1.2], [1.4], [1.6], [2.8], [3.0]], dtype=np.float32)

    mold = build_mold(
        rest_angles=rest,
        motion_angles=motion,
        joint_names=["F3-R-PIP"],
        q_low=np.array([0.0], dtype=np.float32),
        q_high=np.array([1.92], dtype=np.float32),
        q_default=np.array([0.0], dtype=np.float32),
        pin_k=2.0,
    )

    joint = mold.joints[0]
    qpos = apply_mold(np.array([[joint.theta_rest], [joint.theta_lo]], dtype=np.float32), mold)[:, 0]

    assert qpos[0] == pytest.approx(0.0)
    assert qpos[1] == pytest.approx(1.92)


def test_apply_mold_keeps_double_invalid_joint_at_default() -> None:
    mold = build_target_cloud.Mold(
        joints=[
            MoldJoint(
                joint_name="static",
                theta_rest=0.0,
                theta_lo=-0.01,
                theta_hi=0.01,
                sigma=0.1,
                q_low=-1.0,
                q_high=1.0,
                q_default=0.2,
                left_valid=False,
                right_valid=False,
            )
        ]
    )

    qpos = apply_mold(np.array([[-10.0], [0.0], [10.0]], dtype=np.float32), mold)

    assert np.allclose(qpos, 0.2)


def test_density_cap_qpos_limits_repeated_voxels_without_dropping_sparse_tail() -> None:
    cluster = np.zeros((10, 2), dtype=np.float32)
    tail = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    qpos = np.concatenate([cluster, tail], axis=0)

    capped, indices = density_cap_qpos(qpos, q_low=np.zeros(2), q_high=np.ones(2) * 2.0, voxel_size=0.1, cap=3)

    assert capped.shape == (5, 2)
    assert indices.tolist() == [0, 1, 2, 10, 11]



def test_boost_fist_mcp1_qpos_only_pushes_selected_non_thumb_mcp1_toward_high() -> None:
    joint_names = [
        "F1-R-MCP1",
        "F2-R-MCP1",
        "F2-R-PIP",
        "F3-R-MCP1",
        "F4-R-MCP1",
        "F5-R-MCP1",
    ]
    qpos = np.array(
        [
            [0.2, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.2, 1.0, 0.6, 1.1, 1.2, 1.3],
            [0.2, 1.5, 0.6, 1.5, 1.5, 1.5],
        ],
        dtype=np.float32,
    )
    q_high = np.array([0.8, 1.6, 1.9, 1.6, 1.6, 1.6], dtype=np.float32)

    boosted = boost_fist_mcp1_qpos(
        qpos,
        joint_names=joint_names,
        q_high=q_high,
        fist_indices=np.array([0, 2], dtype=np.int64),
        boost_alpha=0.25,
    )

    assert boosted[0, 0] == pytest.approx(qpos[0, 0])
    assert boosted[0, 2] == pytest.approx(qpos[0, 2])
    assert boosted[1].tolist() == pytest.approx(qpos[1].tolist())
    assert boosted[0, 1] == pytest.approx(0.5 + 0.25 * (1.6 - 0.5))
    assert boosted[0, 3] == pytest.approx(0.7 + 0.25 * (1.6 - 0.7))
    assert boosted[2, 1] == pytest.approx(1.5 + 0.25 * (1.6 - 1.5))
    assert boosted[2, 5] == pytest.approx(1.5 + 0.25 * (1.6 - 1.5))

def test_save_robot_kinematics_npz_writes_dataset_compatible_fields(tmp_path: Path) -> None:
    qpos = np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32)
    keypoints = {
        "tip": np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        "pip": np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
    }
    path = tmp_path / "target.npz"

    save_robot_kinematics_npz(path, qpos=qpos, keypoints=keypoints)

    data = np.load(path, allow_pickle=True)
    assert data["qpos"].shape == (2, 2)
    assert data["keypoint"].item()["tip"].shape == (2, 3)


def test_build_target_cloud_file_uses_mold_density_cap_and_hand_fk(tmp_path: Path) -> None:
    rest = make_frames(6)
    motion = make_frames(12)
    rest_path = tmp_path / "rest.npy"
    motion_path = tmp_path / "motion.npy"
    np.save(rest_path, rest)
    np.save(motion_path, motion)

    class FakeHand:
        def __init__(self) -> None:
            self.calls = []

        def get_joint_limit(self):
            return np.array([-1.0, -1.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)

        def keypoint_from_qpos(self, qpos):
            self.calls.append(np.asarray(qpos, dtype=np.float32))
            value = float(np.sum(qpos))
            return {
                "tip": np.array([value, 0.0, 0.0], dtype=np.float32),
                "pip": np.array([0.0, value, 0.0], dtype=np.float32),
            }

    hand = FakeHand()
    config = {"name": "fake_hand", "joint_order": ["F2-R-MCP2", "F2-R-MCP1"]}
    output = tmp_path / "fake_humanshaped.npz"
    debug_dir = tmp_path / "debug"

    result = build_target_cloud_file(
        config=config,
        hand=hand,
        motion_path=motion_path,
        rest_path=rest_path,
        output_path=output,
        debug_dir=debug_dir,
        keypoint_names=["tip", "pip"],
        voxel_size=0.05,
        cap=5,
    )

    assert result == output
    data = np.load(output, allow_pickle=True)
    assert data["qpos"].ndim == 2
    assert data["qpos"].shape[1] == 2
    assert set(data["keypoint"].item()) == {"tip", "pip"}
    assert (debug_dir / "mold.json").is_file()
    assert (debug_dir / "angles_rest.npy").is_file()
    assert (debug_dir / "angles_motion.npy").is_file()
