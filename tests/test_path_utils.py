import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from geort.utils.path import get_human_data


class PathUtilsTest(unittest.TestCase):
    def test_get_human_data_prefers_exact_npy_over_sidecar_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "hts_right_balanced.npy"
            dataset.write_bytes(b"dataset")
            (root / "hts_right_balanced_weights.npy").write_bytes(b"weights")
            (root / "hts_right_stage2_balance.json").write_text("{}")

            with patch("geort.utils.path.get_data_root", return_value=root):
                result = get_human_data("hts_right_balanced")

        self.assertEqual(result, dataset)

    def test_get_human_data_ignores_json_when_matching_dataset_stem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "hts_right.npy"
            dataset.write_bytes(b"dataset")
            (root / "hts_right_stage2_balance.json").write_text("{}")

            with patch("geort.utils.path.get_data_root", return_value=root):
                result = get_human_data("hts_right")

        self.assertEqual(result, dataset)

    def test_get_human_data_rejects_ambiguous_partial_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "hts_right.npy").write_bytes(b"raw")
            (root / "hts_right_balanced.npy").write_bytes(b"balanced")

            with patch("geort.utils.path.get_data_root", return_value=root):
                with self.assertRaisesRegex(FileNotFoundError, "No exact human dataset"):
                    get_human_data("right")


if __name__ == "__main__":
    unittest.main()
