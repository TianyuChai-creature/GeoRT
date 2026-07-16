import json
import numpy as np
import pytest


def test_runtime_loader_accepts_hand_base_shorthand_and_checks_current_contract(tmp_path) -> None:
    from geort.anchor.anchor_runtime_loader import load_anchor_points_for_current_run

    anchor = tmp_path / "anchor.npz"
    normalization = tmp_path / "normalization.json"
    np.savez_compressed(
        anchor,
        human_tip_contexts=np.zeros((1, 5, 3), np.float32),
        robot_points=np.zeros((1, 3), np.float32),
        finger_indices=np.array([0], np.int64),
        metadata_json=np.asarray(json.dumps({"coordinate_frame": "hand_base", "units": "m", "human_data_source": "data/hts_right.npy"})),
    )
    normalization.write_text(json.dumps({"finger_names": ["thumb", "index", "middle", "ring", "pinky"], "human_data_source": "data/hts_right.npy", "human": {name: {"center": [0, 0, 0], "scale": 1} for name in ["thumb", "index", "middle", "ring", "pinky"]}, "robot": {name: {"center": [0, 0, 0], "scale": 1} for name in ["thumb", "index", "middle", "ring", "pinky"]}}))
    loaded = load_anchor_points_for_current_run(anchor, normalization, ["thumb", "index", "middle", "ring", "pinky"])
    assert loaded.human_contexts.shape == (1, 5, 3)
    normalization.unlink()
    with pytest.raises(FileNotFoundError, match="归一化契约尚未写入"):
        load_anchor_points_for_current_run(anchor, normalization, ["thumb", "index", "middle", "ring", "pinky"])
