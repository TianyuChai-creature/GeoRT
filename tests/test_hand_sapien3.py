import unittest
from unittest import mock

from geort.env.hand import HandKinematicModel, HandViewerEnv


class _FakeLink:
    def get_name(self):
        return "base_link"


class _FakePinocchioModel:
    pass


class _FakeJoint:
    def __init__(self, name):
        self.name = name
        self.drive_target = None

    def get_limits(self):
        return [[-1.0, 1.0]]

    def set_drive_property(self, **kwargs):
        self.drive_property = kwargs

    def set_drive_target(self, target):
        self.drive_target = target


class _FakeHand:
    def __init__(self, active_joint_names=None):
        self._links = [_FakeLink()]
        self._active_joints = [
            _FakeJoint(name) for name in (active_joint_names or [])
        ]

    def create_pinocchio_model(self):
        return _FakePinocchioModel()

    def get_links(self):
        return self._links

    def get_active_joints(self):
        return self._active_joints

    def set_qpos(self, qpos):
        self.qpos = qpos

    def set_qvel(self, qvel):
        self.qvel = qvel


class _ClosedViewer:
    closed = True

    def render(self):
        raise AssertionError("render should not be called for a closed viewer")


class _SceneThatShouldNotStep:
    def step(self):
        raise AssertionError("closed viewer updates should not step the scene")

    def update_render(self):
        raise AssertionError("closed viewer updates should not update render")


class HandSapien3ConfigTest(unittest.TestCase):
    def test_constructs_scene_with_current_sapien_physx_config(self):
        with mock.patch("geort.env.hand.sapien.Scene") as scene_cls:
            scene = scene_cls.return_value

            model = HandKinematicModel(
                hand=_FakeHand(),
                base_link="base_link",
                joint_names=[],
            )

        self.assertIs(model.scene, scene)

    def test_viewer_update_stops_when_viewer_is_closed(self):
        env = HandViewerEnv.__new__(HandViewerEnv)
        env.scene = _SceneThatShouldNotStep()
        env.viewer = _ClosedViewer()

        self.assertFalse(env.update())

    def test_omitted_active_joints_are_fixed_at_zero(self):
        hand = _FakeHand(["wrist_a", "wrist_b", "finger_a", "finger_b"])

        model = HandKinematicModel(
            scene=object(),
            hand=hand,
            base_link="base_link",
            joint_names=["finger_a", "finger_b"],
        )

        self.assertEqual(model.get_n_dof(), 2)
        self.assertEqual(hand.qpos.tolist(), [0.0, 0.0, 0.0, 0.0])

        full_qpos = model.convert_user_order_to_sim_order([0.25, -0.5])

        self.assertEqual(full_qpos.tolist(), [0.0, 0.0, 0.25, -0.5])

        model.set_qpos_target([0.25, -0.5])

        self.assertEqual(hand.get_active_joints()[0].drive_target, 0.0)
        self.assertEqual(hand.get_active_joints()[1].drive_target, 0.0)
        self.assertEqual(hand.get_active_joints()[2].drive_target, 0.25)
        self.assertEqual(hand.get_active_joints()[3].drive_target, -0.5)


if __name__ == "__main__":
    unittest.main()
