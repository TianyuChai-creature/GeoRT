import tempfile
import unittest
from pathlib import Path

import numpy as np

from geort.mocap.hts_coverage import (
    FINGER_KEYPOINTS,
    build_stage1_report,
    detect_axis_bracketed_holes,
    extract_finger_features,
    save_stage1_report,
    voxelize_points,
)


class HTSCoverageTest(unittest.TestCase):
    def test_extract_finger_features_uses_pip_tip_joint_pairs(self):
        frames = np.zeros((2, 21, 3), dtype=np.float32)
        frames[:, 6, :] = [1.0, 2.0, 3.0]
        frames[:, 8, :] = [4.0, 5.0, 6.0]

        features = extract_finger_features(frames)

        self.assertIn("index", features)
        np.testing.assert_allclose(features["index"], np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], dtype=np.float32))
        self.assertEqual(FINGER_KEYPOINTS["index"], {"pip": 6, "tip": 8})

    def test_voxelize_points_maps_6d_points_to_integer_voxels(self):
        points = np.array([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06], [0.029, 0.041, 0.051, 0.061, 0.071, 0.081]], dtype=np.float32)

        voxel_indices, occupied = voxelize_points(points, voxel_size=0.01)

        self.assertEqual(voxel_indices.shape, (2, 6))
        self.assertEqual(len(occupied), 2)
        self.assertIn((0, 0, 0, 0, 0, 0), occupied)

    def test_detect_axis_bracketed_holes_finds_center_between_axis_neighbors(self):
        occupied = set()
        center = (0, 0, 0, 0, 0, 0)
        for axis in range(6):
            neg = list(center)
            pos = list(center)
            neg[axis] = -1
            pos[axis] = 1
            occupied.add(tuple(neg))
            occupied.add(tuple(pos))

        holes = detect_axis_bracketed_holes(occupied, max_holes=10)

        self.assertIn(center, holes)

    def test_build_stage1_report_contains_per_finger_6d_coverage(self):
        frames = np.zeros((3, 21, 3), dtype=np.float32)
        frames[:, 6, :] = [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]]
        frames[:, 8, :] = [[0.0, 0.01, 0.0], [0.01, 0.01, 0.0], [0.02, 0.01, 0.0]]

        report = build_stage1_report(frames, voxel_size=0.01, max_holes_per_finger=5)

        self.assertEqual(report["stage"], 1)
        self.assertEqual(report["num_frames"], 3)
        self.assertEqual(report["voxel_space"], "pip_tip_6d")
        self.assertIn("index", report["fingers"])
        self.assertGreaterEqual(report["fingers"]["index"]["occupied_voxels"], 1)
        self.assertIn("hole_voxels", report["fingers"]["index"])

    def test_save_stage1_report_writes_json(self):
        report = {"stage": 1, "fingers": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_stage1_report(report, Path(tmpdir) / "coverage.json")
            text = path.read_text()

        self.assertIn("stage", text)


if __name__ == "__main__":
    unittest.main()
