import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from geort.mocap.replay_mocap import ReplayMocap


class ReplayMocapTest(unittest.TestCase):
    def test_replay_uses_manifest_weights_when_requested(self):
        frames = np.arange(4 * 21 * 3, dtype=np.float32).reshape(4, 21, 3)
        weights = np.array([1.0, 5.0, 1.0, 1.0], dtype=np.float32)

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

            mocap = ReplayMocap(manifest_path, use_weights=True, seed=0)

        self.assertEqual(mocap.weights_path, weight_path)
        self.assertEqual(mocap.human_points.shape, frames.shape)
        self.assertEqual(mocap.replay_indices.shape, (4,))
        self.assertTrue(set(mocap.replay_indices.tolist()).issubset({0, 1, 2, 3}))


    def test_replay_uses_inline_manifest_weights_when_requested(self):
        frames = np.arange(4 * 21 * 3, dtype=np.float32).reshape(4, 21, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "sample.npy"
            manifest_path = root / "train.json"
            np.save(data_path, frames)
            manifest_path.write_text(json.dumps({
                "id": "sample_train",
                "data_path": "sample.npy",
                "weights": [1.0, 5.0, 1.0, 1.0],
            }))

            mocap = ReplayMocap(manifest_path, use_weights=True, seed=0)

        self.assertIsNone(mocap.weights_path)
        self.assertEqual(mocap.human_points.shape, frames.shape)
        self.assertEqual(mocap.replay_indices.shape, (4,))
        self.assertTrue(set(mocap.replay_indices.tolist()).issubset({0, 1, 2, 3}))

    def test_replay_ignores_sidecar_weights_for_npy(self):
        frames = np.arange(4 * 21 * 3, dtype=np.float32).reshape(4, 21, 3)
        weights = np.array([1.0, 5.0, 1.0, 1.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "sample.npy"
            weight_path = root / "sample_weights.npy"
            np.save(data_path, frames)
            np.save(weight_path, weights)

            with patch("geort.mocap.replay_mocap.get_human_data", return_value=data_path):
                mocap = ReplayMocap("sample", use_weights=True, seed=0)

        self.assertIsNone(mocap.weights_path)
        np.testing.assert_array_equal(mocap.replay_indices, np.array([0, 1, 2, 3]))

    def test_replay_without_weights_uses_sequential_order(self):
        frames = np.arange(3 * 21 * 3, dtype=np.float32).reshape(3, 21, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sample.npy"
            np.save(data_path, frames)

            with patch("geort.mocap.replay_mocap.get_human_data", return_value=data_path):
                mocap = ReplayMocap("sample", use_weights=False)

        np.testing.assert_array_equal(mocap.replay_indices, np.array([0, 1, 2]))
        np.testing.assert_allclose(mocap.get()["result"], frames[0])
        np.testing.assert_allclose(mocap.get()["result"], frames[1])


if __name__ == "__main__":
    unittest.main()
