import numpy as np


def test_robot_bending_arc_knots_use_tip_arc_length_not_joint_spacing() -> None:
    from geort.anchor.arc_bending_v2 import build_arc_length_coupled_knots
    from geort.anchor.anchor_spec import RobotFingerJoints

    lower = np.zeros(4)
    upper = np.full(4, 4.0)
    joints = RobotFingerJoints(mcp2=0, mcp1=1, pip=2, dip=3)

    def curved_tip(qpos: np.ndarray) -> np.ndarray:
        b = qpos[1]
        return np.array((b, b * b, 0.0))

    knots = build_arc_length_coupled_knots(
        lower, upper, joints, curved_tip, dense_count=401
    )

    assert knots.shape == (5, 4)
    assert np.allclose(knots[:, 1], knots[:, 2])
    assert np.allclose(knots[:, 3], knots[:, 1] / 2.0)
    # A curved TIP path is deliberately not uniformly spaced in b.
    assert not np.allclose(np.diff(knots[:, 1]), np.diff(knots[:, 1])[0])
