import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geort.dataset_manifest import load_dataset_manifest
from geort.trainer import prepare_human_training_dataset


class DatasetManifestTest(unittest.TestCase):
    def test_load_dataset_manifest_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "raw.npy"
            weights_path = root / "weights.npy"
            data_path.write_bytes(b"data")
            weights_path.write_bytes(b"weights")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({
                "id": "sample",
                "data_path": "raw.npy",
                "weights_path": "weights.npy",
                "reports": {"stage2": "stage2.json"},
            }))

            manifest = load_dataset_manifest(manifest_path)

        self.assertEqual(manifest.dataset_id, "sample")
        self.assertEqual(manifest.data_path, data_path)
        self.assertEqual(manifest.weights_path, weights_path)
        self.assertEqual(manifest.reports["stage2"], root / "stage2.json")

    def test_prepare_human_training_dataset_uses_manifest_weights(self):
        frames = np.zeros((3, 21, 3), dtype=np.float32)
        frames[:, 4, :] = [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
        weights = np.array([1.0, 2.0, 4.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "frames.npy"
            weights_path = root / "importance.npy"
            manifest_path = root / "manifest.json"
            np.save(data_path, frames)
            np.save(weights_path, weights)
            manifest_path.write_text(json.dumps({
                "id": "weighted_sample",
                "data_path": "frames.npy",
                "weights_path": "importance.npy",
            }))

            dataset, loaded_weights = prepare_human_training_dataset(manifest_path, human_ids=[4])

        self.assertEqual(len(dataset), 3)
        np.testing.assert_allclose(loaded_weights, weights)
        np.testing.assert_allclose(dataset[2], np.array([[3, 0, 0]], dtype=np.float32))

    def test_prepare_human_training_dataset_uses_inline_manifest_weights(self):
        frames = np.zeros((3, 21, 3), dtype=np.float32)
        frames[:, 4, :] = [[1, 0, 0], [2, 0, 0], [3, 0, 0]]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "frames.npy"
            manifest_path = root / "train.json"
            np.save(data_path, frames)
            manifest_path.write_text(json.dumps({
                "id": "inline_weighted_sample",
                "data_path": "frames.npy",
                "weights": [1.0, 2.0, 4.0],
            }))

            dataset, loaded_weights = prepare_human_training_dataset(manifest_path, human_ids=[4])

        self.assertEqual(len(dataset), 3)
        np.testing.assert_allclose(loaded_weights, np.array([1.0, 2.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(dataset[2], np.array([[3, 0, 0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
