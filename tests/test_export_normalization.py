from __future__ import annotations

import numpy as np

from geort import export


def test_inference_selects_and_normalizes_tip_points() -> None:
    assert hasattr(export, "normalize_selected_human_keypoints")
    keypoints = np.zeros((21, 3), dtype=np.float32)
    keypoints[4] = [3.0, 4.0, 5.0]
    keypoints[8] = [12.0, 22.0, 32.0]
    stats = {
        "thumb": {"center": [1.0, 2.0, 3.0], "scale": 2.0},
        "index": {"center": [10.0, 20.0, 30.0], "scale": 2.0},
    }

    normalized = export.normalize_selected_human_keypoints(
        keypoints,
        human_ids=[4, 8],
        finger_names=["thumb", "index"],
        stats=stats,
    )

    assert normalized.shape == (2, 3)
    assert np.allclose(normalized, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
