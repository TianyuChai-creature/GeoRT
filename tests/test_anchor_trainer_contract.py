import json

import numpy as np
import pytest


def _stats():
    return {name: {"center": [0.0, 0.0, 0.0], "scale": 1.0} for name in ("thumb", "index")}


def _write_bundle(path, source):
    np.savez(
        path,
        human_tip_contexts=np.zeros((2, 2, 3), dtype=np.float32),
        robot_points=np.zeros((2, 3), dtype=np.float32),
        finger_indices=np.array([0, 1], dtype=np.int64),
        metadata_json=np.asarray(json.dumps({
            "coordinate_space": {
                "human_coordinate_frame": "hand_base",
                "robot_coordinate_frame": "hand_base",
                "units": "m",
            },
            "human_data_source": str(source),
        })),
    )


def test_anchor_contract_requires_current_run_normalization_and_matching_source(tmp_path):
    from geort.anchor.trainer_contract import load_anchor_points_for_current_run

    human = tmp_path / "hts_right.npy"
    human.write_bytes(b"D1")
    bundle = tmp_path / "anchors.npz"
    _write_bundle(bundle, human)
    missing = tmp_path / "normalization.json"

    with pytest.raises(FileNotFoundError, match="归一化契约尚未写入"):
        load_anchor_points_for_current_run(bundle, missing, ["thumb", "index"])

    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({
        "finger_names": ["thumb", "index"],
        "human_data_source": str(human),
        "human": _stats(),
        "robot": _stats(),
    }))
    points = load_anchor_points_for_current_run(bundle, normalization, ["thumb", "index"])
    assert points.human_contexts.shape == (2, 2, 3)

    _write_bundle(bundle, tmp_path / "other.npy")
    with pytest.raises(ValueError, match="human_data_source mismatch"):
        load_anchor_points_for_current_run(bundle, normalization, ["thumb", "index"])
