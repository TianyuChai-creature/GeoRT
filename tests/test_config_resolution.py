import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from geort.utils.config_utils import get_config


class ConfigResolutionTest(unittest.TestCase):
    def test_get_config_requires_exact_config_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "custom_right.json").write_text(json.dumps({"name": "custom_right"}))
            (root / "custom_left.json").write_text(json.dumps({"name": "custom_left"}))

            with patch("geort.utils.config_utils.get_package_root", return_value=root.parent):
                config_dir = root.parent / "geort" / "config"
                config_dir.mkdir(parents=True, exist_ok=True)
                for source in root.iterdir():
                    source.replace(config_dir / source.name)

                with self.assertRaisesRegex(FileNotFoundError, "No exact config"):
                    get_config("custom")

    def test_get_config_loads_exact_stem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_root = Path(tmpdir)
            config_dir = package_root / "geort" / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "custom_right.json").write_text(json.dumps({"name": "custom_right"}))

            with patch("geort.utils.config_utils.get_package_root", return_value=package_root):
                config = get_config("custom_right")

        self.assertEqual(config["name"], "custom_right")


if __name__ == "__main__":
    unittest.main()
