import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geort.dataset import FramePointDataset
from geort.trainer import find_human_weight_path, prepare_human_training_dataset


class Stage3TrainingDataTest(unittest.TestCase):
    def test_frame_point_dataset_preserves_full_keypoint_frame(self):
        points = np.arange(4 * 10 * 3, dtype=np.float32).reshape(4, 10, 3)
        dataset = FramePointDataset(points)

        self.assertEqual(len(dataset), 4)
        np.testing.assert_allclose(dataset[2], points[2])

    def test_find_human_weight_path_only_uses_manifest_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "sample.npy"
            sidecar_path = root / "sample_weights.npy"
            manifest_path = root / "manifest.json"
            data_path.write_bytes(b"data")
            sidecar_path.write_bytes(b"legacy")
            manifest_path.write_text(json.dumps({
                "id": "sample",
                "data_path": "sample.npy",
                "weights_path": "explicit_weights.npy",
            }))
            expected = root / "explicit_weights.npy"
            expected.write_bytes(b"weights")

            self.assertIsNone(find_human_weight_path(data_path))
            result = find_human_weight_path(manifest_path)

        self.assertEqual(result, expected)

    def test_prepare_human_training_dataset_ignores_sidecar_weights_for_npy(self):
        frames = np.zeros((3, 21, 3), dtype=np.float32)
        frames[:, 4, :] = [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
        frames[:, 8, :] = [[0, 1, 0], [0, 2, 0], [0, 3, 0]]
        weights = np.array([1.0, 3.0, 1.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sample.npy"
            weight_path = Path(tmpdir) / "sample_weights.npy"
            np.save(data_path, frames)
            np.save(weight_path, weights)

            dataset, loaded_weights = prepare_human_training_dataset(data_path, human_ids=[4, 8])

        self.assertEqual(len(dataset), 3)
        self.assertIsNone(loaded_weights)
        np.testing.assert_allclose(dataset[1], np.array([[2, 0, 0], [0, 2, 0]], dtype=np.float32))

    def test_prepare_human_training_dataset_rejects_manifest_weight_length_mismatch(self):
        frames = np.zeros((3, 21, 3), dtype=np.float32)
        weights = np.ones((2,), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "sample.npy"
            weight_path = root / "importance.npy"
            manifest_path = root / "manifest.json"
            np.save(data_path, frames)
            np.save(weight_path, weights)
            manifest_path.write_text(json.dumps({
                "id": "sample",
                "data_path": "sample.npy",
                "weights_path": "importance.npy",
            }))

            with self.assertRaisesRegex(ValueError, "weights length"):
                prepare_human_training_dataset(manifest_path, human_ids=[4, 8])


if __name__ == "__main__":
    unittest.main()
