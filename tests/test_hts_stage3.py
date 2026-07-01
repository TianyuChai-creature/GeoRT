import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geort.mocap.hts_stage3 import (
    CONTACT_PAIRS,
    build_stage3_report,
    compute_frame_weights,
    detect_contact_frames,
    save_stage3_outputs,
)


class HTSStage3Test(unittest.TestCase):
    def test_detect_contact_frames_marks_thumb_tip_to_index_tip_pinch(self):
        frames = np.zeros((3, 21, 3), dtype=np.float32)
        frames[:, 4, :] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        frames[:, 8, :] = [[0.01, 0.0, 0.0], [0.04, 0.0, 0.0], [1.1, 0.0, 0.0]]
        frames[:, 12, :] = [2.0, 0.0, 0.0]
        frames[:, 16, :] = [3.0, 0.0, 0.0]
        frames[:, 20, :] = [4.0, 0.0, 0.0]

        masks, min_dist = detect_contact_frames(frames, threshold=0.025)

        self.assertIn("thumb_tip__index_tip", masks)
        np.testing.assert_array_equal(masks["thumb_tip__index_tip"], np.array([True, False, False]))
        self.assertAlmostEqual(float(min_dist[0]), 0.01, places=6)
        self.assertTrue(any(pair[0] == 4 and pair[1] == 8 for pair in CONTACT_PAIRS))

    def test_compute_frame_weights_adds_bonus_for_contact_frames(self):
        frames = np.zeros((2, 21, 3), dtype=np.float32)
        frames[:, 4, :] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        frames[:, 8, :] = [[0.01, 0.0, 0.0], [0.05, 0.0, 0.0]]
        frames[:, 12, :] = [2.0, 0.0, 0.0]
        frames[:, 16, :] = [3.0, 0.0, 0.0]
        frames[:, 20, :] = [4.0, 0.0, 0.0]

        weights, masks, _ = compute_frame_weights(frames, threshold=0.025, contact_bonus=2.0, max_weight=5.0)

        self.assertEqual(weights.dtype, np.float32)
        self.assertEqual(weights.tolist(), [3.0, 1.0])
        self.assertTrue(masks["thumb_tip__index_tip"][0])

    def test_build_stage3_report_summarizes_contacts_and_weights(self):
        weights = np.array([3.0, 1.0, 3.0], dtype=np.float32)
        masks = {"thumb_tip__index_tip": np.array([True, False, True])}
        min_dist = np.array([0.01, 0.04, 0.02], dtype=np.float32)

        report = build_stage3_report(weights, masks, min_dist, threshold=0.025, contact_bonus=2.0, max_weight=5.0)

        self.assertEqual(report["stage"], 3)
        self.assertEqual(report["num_frames"], 3)
        self.assertEqual(report["contact_frames"], 2)
        self.assertEqual(report["contacts"]["thumb_tip__index_tip"]["count"], 2)
        self.assertAlmostEqual(report["weight_mean"], 7.0 / 3.0, places=6)

    def test_save_stage3_outputs_writes_weights_report_and_manifest(self):
        weights = np.array([1.0, 3.0], dtype=np.float32)
        report = {"stage": 3}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "balanced.npy"
            data_path.write_bytes(b"dataset")

            weight_path, report_path, manifest_path = save_stage3_outputs(
                weights,
                report,
                data_path=data_path,
                weights_path=root / "frame_importance.npy",
                report_path=root / "stage3_report.json",
                manifest_path=root / "training_manifest.json",
                dataset_id="sample_stage3",
            )
            loaded = np.load(weight_path)
            text = report_path.read_text()
            manifest = json.loads(manifest_path.read_text())

        np.testing.assert_allclose(loaded, weights)
        self.assertIn("stage", text)
        self.assertEqual(manifest["id"], "sample_stage3")
        self.assertEqual(manifest["data_path"], "balanced.npy")
        self.assertEqual(manifest["weights_path"], "frame_importance.npy")
        self.assertEqual(manifest["reports"], {"stage3": "stage3_report.json"})


if __name__ == "__main__":
    unittest.main()
