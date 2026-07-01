import unittest

import numpy as np

from geort.mocap.measure_hts_aa_pose import (
    FINGER_SEGMENTS,
    build_arg_parser,
    compute_pose_aa_angles,
    summarize_angles,
)


class HTSAAPoseMeasurementTest(unittest.TestCase):
    def test_default_cli_uses_single_closed_pose_on_hts_udp(self):
        args = build_arg_parser().parse_args(["--state", "neutral"])

        self.assertEqual(args.state, "neutral")
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)
        self.assertEqual(args.max_frames, 120)

    def test_compute_pose_aa_angles_reports_mcp_and_pip_tip_xz_angles(self):
        frames = np.zeros((2, 21, 3), dtype=np.float32)
        for _, ids in FINGER_SEGMENTS.items():
            frames[:, ids["mcp"], :] = [0.0, 0.0, 0.0]
            frames[:, ids["pip"], :] = [1.0, 0.0, 0.0]
            frames[:, ids["tip"], :] = [1.0, 0.0, 1.0]

        angles = compute_pose_aa_angles(frames)

        self.assertEqual(set(angles.keys()), {"index", "middle", "ring", "pinky"})
        for values in angles.values():
            np.testing.assert_allclose(values["mcp_pip"], np.array([0.0, 0.0], dtype=np.float32))
            np.testing.assert_allclose(values["pip_tip"], np.array([np.pi / 2.0, np.pi / 2.0], dtype=np.float32))

    def test_summarize_angles_uses_circular_mean_and_percentiles(self):
        angles = {
            "index": {
                "mcp_pip": np.array([0.0, 0.1, -0.1], dtype=np.float32),
                "pip_tip": np.array([1.0, 1.1, 0.9], dtype=np.float32),
            }
        }

        summary = summarize_angles(angles)

        self.assertAlmostEqual(summary["index"]["mcp_pip_mean_rad"], 0.0, places=6)
        self.assertAlmostEqual(summary["index"]["mcp_pip_median_rad"], 0.0, places=6)
        self.assertAlmostEqual(summary["index"]["pip_tip_median_rad"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
