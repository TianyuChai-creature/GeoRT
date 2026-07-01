import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geort.mocap.hts_prepare_training import prepare_training_dataset
from geort.mocap.hts_prepare_training import build_arg_parser


class HTSTrainingDatasetTest(unittest.TestCase):
    def test_default_thresholds_keep_dense_fist_frames(self):
        frames = np.zeros((20, 21, 3), dtype=np.float32)
        for idx in range(16):
            frames[idx, 4, :] = [1.0, 1.0, 1.0]
            frames[idx, 5, :] = [0.0, 0.0, 0.0]
            frames[idx, 6, :] = [0.01, 0.0, 0.0]
            frames[idx, 7, :] = [0.012, 0.0, 0.0]
            frames[idx, 8, :] = [0.006, 0.0, 0.0]
            frames[idx, 9, :] = frames[idx, 5, :]
            frames[idx, 10, :] = frames[idx, 6, :]
            frames[idx, 11, :] = frames[idx, 7, :]
            frames[idx, 12, :] = frames[idx, 8, :]
            frames[idx, 13, :] = frames[idx, 5, :]
            frames[idx, 14, :] = frames[idx, 6, :]
            frames[idx, 15, :] = frames[idx, 7, :]
            frames[idx, 16, :] = frames[idx, 8, :]
            frames[idx, 17, :] = frames[idx, 5, :]
            frames[idx, 18, :] = frames[idx, 6, :]
            frames[idx, 19, :] = frames[idx, 7, :]
            frames[idx, 20, :] = frames[idx, 8, :]
        for idx in range(16, 20):
            frames[idx, :, 0] = np.linspace(0.0, 0.02 + idx * 0.001, 21)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "hts_right.npy"
            np.save(source_path, frames)

            output_data, _ = prepare_training_dataset(source_path=source_path)
            train_frames = np.load(output_data)

        dense_fist_count = int(np.isclose(train_frames[:, 8, 0], 0.006).sum())
        self.assertGreaterEqual(dense_fist_count, 16)

    def test_cli_defaults_use_looser_density_thresholds(self):
        args = build_arg_parser().parse_args([])

        self.assertEqual(args.voxel_size, 0.003)
        self.assertEqual(args.max_per_voxel, 24)

    def test_prepare_training_dataset_writes_train_npy_and_json_only(self):
        frames = np.zeros((6, 21, 3), dtype=np.float32)
        frames[:, 4, :] = 0.0
        frames[:, 8, :] = [0.01, 0.0, 0.0]
        frames[:, 12, :] = [0.05, 0.0, 0.0]
        frames[:, 16, :] = [0.06, 0.0, 0.0]
        frames[:, 20, :] = [0.07, 0.0, 0.0]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "hts_right.npy"
            train_path = root / "hts_right_train.npy"
            metadata_path = root / "hts_right_train.json"
            np.save(source_path, frames)

            output_data, output_metadata = prepare_training_dataset(
                source_path=source_path,
                output_path=train_path,
                metadata_path=metadata_path,
                dataset_id="hts_right_train",
                voxel_size=0.005,
                max_per_voxel=8,
                preserve_contact_pairs="all",
                contact_threshold=0.025,
                contact_bonus=2.0,
                max_weight=5.0,
            )

            files = sorted(path.name for path in root.iterdir() if path.name.startswith("hts_right_train"))
            train_frames = np.load(output_data)
            metadata = json.loads(output_metadata.read_text())

        self.assertEqual(files, ["hts_right_train.json", "hts_right_train.npy"])
        self.assertEqual(output_data, train_path)
        self.assertEqual(output_metadata, metadata_path)
        self.assertEqual(metadata["id"], "hts_right_train")
        self.assertEqual(metadata["data_path"], "hts_right_train.npy")
        self.assertEqual(len(metadata["weights"]), train_frames.shape[0])
        self.assertEqual(metadata["processing"]["source"], "hts_right.npy")
        self.assertIn("stage2", metadata["processing"])
        self.assertIn("stage3", metadata["processing"])


if __name__ == "__main__":
    unittest.main()
