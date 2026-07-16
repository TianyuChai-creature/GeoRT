from __future__ import annotations

import numpy as np
import pytest

from geort.anchor.anchor_spec import (
    RobotFingerJoints,
    build_lateral_knots,
    build_non_thumb_bending_knots,
    build_thumb_arc_knots,
    derive_finger_joint_layouts,
)


def _limits() -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(20, -1.0)
    upper = np.full(20, 1.0)
    lower[5:8] = np.array([-0.3, -0.2, -0.4])
    upper[5:8] = np.array([0.5, 0.6, 0.8])
    return lower, upper


def test_lateral_knots_span_only_target_mcp2_range() -> None:
    lower, upper = _limits()
    joints = RobotFingerJoints(mcp2=4, mcp1=5, pip=6, dip=7)

    qpos = build_lateral_knots(lower, upper, joints)

    assert qpos.shape == (5, 20)
    assert np.allclose(qpos[:, 4], np.linspace(lower[4], upper[4], 5))
    assert np.allclose(qpos[:, 5:], 0.0)
    assert np.allclose(qpos[:, :4], 0.0)


def test_non_thumb_bending_uses_feasible_coupled_b_range() -> None:
    lower, upper = _limits()
    joints = RobotFingerJoints(mcp2=4, mcp1=5, pip=6, dip=7)

    qpos = build_non_thumb_bending_knots(lower, upper, joints)

    expected_low = max(lower[5], lower[6], 2.0 * lower[7])
    expected_high = min(upper[5], upper[6], 2.0 * upper[7])
    bend = np.linspace(expected_low, expected_high, 5)
    assert np.allclose(qpos[:, 4], 0.0)
    assert np.allclose(qpos[:, 5], bend)
    assert np.allclose(qpos[:, 6], bend)
    assert np.allclose(qpos[:, 7], bend / 2.0)
    assert np.all(qpos >= lower)
    assert np.all(qpos <= upper)


def test_thumb_knots_follow_fk_tip_arc_with_exact_endpoints() -> None:
    lower, upper = _limits()
    joints = RobotFingerJoints(mcp2=0, mcp1=1, pip=2, dip=3)

    def fake_fk(qpos: np.ndarray) -> np.ndarray:
        bend = qpos[1]
        return np.array([bend, bend * bend, 0.0])

    qpos = build_thumb_arc_knots(lower, upper, joints, fake_fk, dense_count=101)

    assert qpos.shape == (5, 20)
    assert np.all(np.diff(qpos[:, 1]) > 0.0)
    assert qpos[0, 1] == pytest.approx(-1.0)
    assert qpos[-1, 1] == pytest.approx(1.0)
    assert np.allclose(qpos[:, 2], qpos[:, 1])
    assert np.allclose(qpos[:, 3], qpos[:, 1] / 2.0)


def test_coupled_bending_rejects_empty_feasible_interval() -> None:
    lower, upper = _limits()
    lower[5:8] = np.array([0.8, 0.8, -0.1])
    upper[5:8] = np.array([1.0, 1.0, 0.1])
    joints = RobotFingerJoints(mcp2=4, mcp1=5, pip=6, dip=7)

    with pytest.raises(ValueError, match="feasible"):
        build_non_thumb_bending_knots(lower, upper, joints)


def test_derives_custom_hand_joint_blocks_in_thumb_to_pinky_order() -> None:
    joint_order = [
        f"F{finger}-R-{component}"
        for finger in range(1, 6)
        for component in ("MCP2", "MCP1", "PIP", "DIP")
    ]

    layouts = derive_finger_joint_layouts(joint_order)

    assert layouts == (
        RobotFingerJoints(0, 1, 2, 3),
        RobotFingerJoints(4, 5, 6, 7),
        RobotFingerJoints(8, 9, 10, 11),
        RobotFingerJoints(12, 13, 14, 15),
        RobotFingerJoints(16, 17, 18, 19),
    )
