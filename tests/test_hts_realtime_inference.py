import unittest

import numpy as np

from geort.mocap.hts_realtime_inference import (
    LatestPointBuffer,
    build_arg_parser,
    infer_hand_side,
    run_realtime_inference,
    run_realtime_viewer_loop,
)


class DummyModel:
    def __init__(self):
        self.inputs = []

    def forward(self, points):
        self.inputs.append(points.copy())
        return np.arange(20, dtype=np.float32)


class DummyHand:
    def __init__(self):
        self.targets = []

    def set_qpos_target(self, qpos):
        self.targets.append(np.asarray(qpos, dtype=np.float32).copy())


class DummyContactVisualizer:
    def __init__(self):
        self.calls = []

    def update(self, qpos, *, frame_id):
        self.calls.append((np.asarray(qpos, dtype=np.float32).copy(), frame_id))


class DummyViewerEnv:
    def __init__(self, updates):
        self.updates = updates
        self.calls = 0

    def update(self):
        self.calls += 1
        return self.calls <= self.updates


class HTSRealtimeInferenceTest(unittest.TestCase):
    def test_default_cli_listens_on_hts_udp_broadcast_port_and_custom_right(self):
        args = build_arg_parser().parse_args([])

        self.assertEqual(args.hand, "custom_right")
        self.assertEqual(args.ckpt_tag, "custom_right_last")
        self.assertEqual(args.transport, "udp")
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)
        self.assertEqual(args.hand_side, "auto")
        self.assertFalse(args.contact_visual)
        self.assertEqual(args.contact_threshold, 0.015)

    def test_infer_hand_side_from_hand_name(self):
        self.assertEqual(infer_hand_side("custom_left", "auto"), "left")
        self.assertEqual(infer_hand_side("custom_right", "auto"), "right")
        self.assertEqual(infer_hand_side("custom_left", "right"), "right")

    def test_infer_hand_side_rejects_ambiguous_auto_hand_name(self):
        with self.assertRaisesRegex(ValueError, "--hand-side"):
            infer_hand_side("custom", "auto")

    def test_contact_visual_cli_can_be_enabled(self):
        args = build_arg_parser().parse_args([
            "--contact-visual",
            "--contact-threshold",
            "0.02",
            "--contact-report-interval",
            "5",
        ])

        self.assertTrue(args.contact_visual)
        self.assertEqual(args.contact_threshold, 0.02)
        self.assertEqual(args.contact_report_interval, 5)

    def test_run_realtime_inference_drives_hand_from_valid_points(self):
        points = np.ones((21, 3), dtype=np.float32)
        model = DummyModel()
        hand = DummyHand()
        viewer_env = DummyViewerEnv(updates=10)

        processed = run_realtime_inference(
            model=model,
            hand=hand,
            viewer_env=viewer_env,
            points_iter=iter([points]),
            viewer_updates_per_frame=2,
            max_frames=1,
        )

        self.assertEqual(processed, 1)
        self.assertEqual(viewer_env.calls, 2)
        self.assertEqual(len(model.inputs), 1)
        np.testing.assert_allclose(model.inputs[0], points)
        self.assertEqual(len(hand.targets), 1)
        np.testing.assert_allclose(hand.targets[0], np.arange(20, dtype=np.float32))

    def test_run_realtime_inference_updates_contact_visualizer(self):
        points = np.ones((21, 3), dtype=np.float32)
        model = DummyModel()
        hand = DummyHand()
        viewer_env = DummyViewerEnv(updates=10)
        contact_visualizer = DummyContactVisualizer()

        processed = run_realtime_inference(
            model=model,
            hand=hand,
            viewer_env=viewer_env,
            points_iter=iter([points]),
            viewer_updates_per_frame=1,
            max_frames=1,
            contact_visualizer=contact_visualizer,
        )

        self.assertEqual(processed, 1)
        self.assertEqual(len(contact_visualizer.calls), 1)
        qpos, frame_id = contact_visualizer.calls[0]
        np.testing.assert_allclose(qpos, np.arange(20, dtype=np.float32))
        self.assertEqual(frame_id, 1)

    def test_run_realtime_inference_skips_nonfinite_points(self):
        bad = np.ones((21, 3), dtype=np.float32)
        bad[0, 0] = np.nan
        good = np.zeros((21, 3), dtype=np.float32)
        model = DummyModel()
        hand = DummyHand()
        viewer_env = DummyViewerEnv(updates=10)

        processed = run_realtime_inference(
            model=model,
            hand=hand,
            viewer_env=viewer_env,
            points_iter=iter([bad, good]),
            viewer_updates_per_frame=1,
            max_frames=1,
        )

        self.assertEqual(processed, 1)
        self.assertEqual(len(model.inputs), 1)
        np.testing.assert_allclose(model.inputs[0], good)
        self.assertEqual(len(hand.targets), 1)

    def test_run_realtime_viewer_loop_updates_without_frames(self):
        model = DummyModel()
        hand = DummyHand()
        viewer_env = DummyViewerEnv(updates=3)
        buffer = LatestPointBuffer()

        processed = run_realtime_viewer_loop(
            model=model,
            hand=hand,
            viewer_env=viewer_env,
            point_buffer=buffer,
            max_frames=None,
            fps_interval=0,
        )

        self.assertEqual(processed, 0)
        self.assertEqual(viewer_env.calls, 4)
        self.assertEqual(model.inputs, [])
        self.assertEqual(hand.targets, [])

    def test_latest_point_buffer_keeps_newest_frame(self):
        buffer = LatestPointBuffer()
        first = np.zeros((21, 3), dtype=np.float32)
        second = np.ones((21, 3), dtype=np.float32)

        buffer.put(first)
        buffer.put(second)

        np.testing.assert_allclose(buffer.get_latest(), second)
        self.assertIsNone(buffer.get_latest())


if __name__ == "__main__":
    unittest.main()
