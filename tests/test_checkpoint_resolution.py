import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from geort.export import resolve_checkpoint_dir
from geort.trainer import should_save_epoch_checkpoint


class CheckpointResolutionTest(unittest.TestCase):
    def test_resolve_checkpoint_dir_requires_exact_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "custom_right_last").mkdir()
            (root / "custom_right_2026-06-19_run").mkdir()

            with patch("geort.export.get_checkpoint_root", return_value=root):
                with self.assertRaisesRegex(FileNotFoundError, "No exact checkpoint"):
                    resolve_checkpoint_dir("custom_right")

    def test_resolve_checkpoint_dir_loads_exact_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected = root / "custom_right_last"
            expected.mkdir()

            with patch("geort.export.get_checkpoint_root", return_value=root):
                result = resolve_checkpoint_dir("custom_right_last")

        self.assertEqual(result, expected)

    def test_default_checkpoint_policy_saves_only_last_file(self):
        self.assertFalse(should_save_epoch_checkpoint(epoch=0, n_epoch=3, save_every=0))
        self.assertFalse(should_save_epoch_checkpoint(epoch=2, n_epoch=3, save_every=0))

    def test_checkpoint_policy_can_save_periodic_epochs(self):
        self.assertTrue(should_save_epoch_checkpoint(epoch=1, n_epoch=5, save_every=2))
        self.assertFalse(should_save_epoch_checkpoint(epoch=2, n_epoch=5, save_every=2))


if __name__ == "__main__":
    unittest.main()
