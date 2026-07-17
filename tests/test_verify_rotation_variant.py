import json
import subprocess
import sys

import numpy as np


def test_rotation_variant_reports_only_added_rotation(tmp_path):
    base = tmp_path / "base.npz"
    variant = tmp_path / "variant.npz"
    qpos = np.arange(12, dtype=np.float64).reshape(3, 4)
    keypoint = np.empty((), dtype=object)
    keypoint[()] = {"tip": np.arange(9, dtype=np.float32).reshape(3, 3)}
    rotation = np.empty((), dtype=object)
    rotation[()] = {"tip": np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))}
    np.savez(base, qpos=qpos, keypoint=keypoint)
    np.savez(variant, qpos=qpos.copy(), keypoint=keypoint, link_rotation=rotation)

    completed = subprocess.run(
        [sys.executable, "scripts/verify_rotation_variant.py", "--base", str(base), "--variant", str(variant)],
        check=True, capture_output=True, text=True,
    )
    report = json.loads(completed.stdout)

    assert report["added_fields"] == ["link_rotation"]
    assert report["removed_fields"] == []
    assert report["shared_fields_equivalent"] is True
    assert report["shared_max_abs_diff"] == 0.0
