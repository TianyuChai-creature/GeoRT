from __future__ import annotations

import numpy as np
import pytest
import torch


def test_analytic_anchor_fk_uses_current_limits_and_config_tip_offsets(monkeypatch) -> None:
    from geort.anchor.generate_robot_anchors import evaluate_analytic_tip_fk
    from geort.anchor.compat import get_joint_limits
    from geort.utils.config_utils import get_config, parse_config_keypoint_info, select_keypoint_types

    captured: dict[str, object] = {}

    class FakeAnalyticFK:
        def __init__(self, _urdf_path, _lower, _upper, *, tip_offsets):
            captured["tip_offsets"] = tip_offsets

        def __call__(self, qpos, *, return_link_rotations=False):
            tips = torch.zeros((qpos.shape[0], 5, 3), dtype=torch.float32)
            if return_link_rotations:
                return tips, torch.eye(3).repeat(qpos.shape[0], 5, 1, 1)
            return tips

    monkeypatch.setattr("geort.analytic_fk.AnalyticFK", FakeAnalyticFK)

    config = get_config("custom_right")
    lower, upper = get_joint_limits(config)
    qpos = np.asarray([(np.asarray(lower) + np.asarray(upper)) / 2.0])

    tips = evaluate_analytic_tip_fk(qpos, config)
    expected_offsets = select_keypoint_types(
        parse_config_keypoint_info(config), allowed_types=("tip",)
    )["offset"]

    assert tips.shape == (1, 5, 3)
    assert np.isfinite(tips).all()
    assert captured["tip_offsets"] == expected_offsets


def test_parse_config_joint_limit_rejects_source_hand_config() -> None:
    from geort.utils.config_utils import get_config, parse_config_joint_limit

    with pytest.raises(ValueError, match="checkpoint export config"):
        parse_config_joint_limit(get_config("custom_right"))
