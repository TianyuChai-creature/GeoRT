import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from geort.mocap.hts_right_mocap import (
    build_arg_parser,
    frame_to_geort_points,
    make_output_path,
    make_right_output_path,
    save_human_data,
    save_right_human_data,
)


class HTSRightMocapTest(unittest.TestCase):
    def test_make_output_path_adds_selected_hand_suffix_and_npy_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            left = make_output_path("quest3_custom", hand_side="left", data_dir=Path(tmpdir))
            right = make_output_path("quest3_custom.npy", hand_side="right", data_dir=Path(tmpdir))

        self.assertEqual(left.name, "quest3_custom_left.npy")
        self.assertEqual(right.name, "quest3_custom_right.npy")

    def test_make_right_output_path_adds_right_suffix_and_npy_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = make_right_output_path("quest3_custom", data_dir=Path(tmpdir))

        self.assertEqual(out.name, "quest3_custom_right.npy")

    def test_make_output_path_rejects_unknown_hand_side(self):
        with self.assertRaisesRegex(ValueError, "hand_side"):
            make_output_path("quest3_custom", hand_side="middle")

    def test_default_cli_listens_on_hts_udp_broadcast_port(self):
        args = build_arg_parser().parse_args([])

        self.assertEqual(args.hand_side, "right")
        self.assertEqual(args.name, "hts")
        self.assertEqual(args.transport, "udp")
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)

    def test_cli_accepts_left_hand_capture(self):
        args = build_arg_parser().parse_args(["--hand-side", "left"])

        self.assertEqual(args.hand_side, "left")

    def test_frame_to_geort_points_converts_frame_and_returns_float32_points(self):
        raw_points = [(float(i), float(i + 1), float(i + 2)) for i in range(21)]
        converted_points = [(x + 10.0, -y, z) for x, y, z in raw_points]
        raw_frame = SimpleNamespace(landmarks=SimpleNamespace(points=raw_points))
        converted_frame = SimpleNamespace(landmarks=SimpleNamespace(points=converted_points))

        calls = []

        def converter(frame):
            calls.append(frame)
            return converted_frame

        points = frame_to_geort_points(raw_frame, converter=converter)

        self.assertEqual(calls, [raw_frame])
        self.assertEqual(points.dtype, np.float32)
        self.assertEqual(points.shape, (21, 3))
        np.testing.assert_allclose(points, np.asarray(converted_points, dtype=np.float32))

    def test_frame_to_geort_points_rejects_wrong_landmark_count(self):
        frame = SimpleNamespace(landmarks=SimpleNamespace(points=[(0.0, 0.0, 0.0)] * 20))

        with self.assertRaisesRegex(ValueError, "Expected 21 HTS landmarks"):
            frame_to_geort_points(frame, converter=lambda f: f)

    def test_save_right_human_data_writes_geort_compatible_npy(self):
        frames = [np.ones((21, 3), dtype=np.float32), np.full((21, 3), 2.0, dtype=np.float32)]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_right_human_data(frames, name="quest3_custom", data_dir=Path(tmpdir))
            loaded = np.load(out)

        self.assertEqual(out.name, "quest3_custom_right.npy")
        self.assertEqual(loaded.dtype, np.float32)
        self.assertEqual(loaded.shape, (2, 21, 3))
        np.testing.assert_allclose(loaded[0], 1.0)
        np.testing.assert_allclose(loaded[1], 2.0)

    def test_save_human_data_writes_left_hand_npy(self):
        frames = [np.ones((21, 3), dtype=np.float32)]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_human_data(frames, name="quest3_custom", hand_side="left", data_dir=Path(tmpdir))
            loaded = np.load(out)

        self.assertEqual(out.name, "quest3_custom_left.npy")
        self.assertEqual(loaded.dtype, np.float32)
        self.assertEqual(loaded.shape, (1, 21, 3))


if __name__ == "__main__":
    unittest.main()
