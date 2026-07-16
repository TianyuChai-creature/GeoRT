from __future__ import annotations

import numpy as np


def test_analytic_anchor_fk_uses_config_tip_offsets() -> None:
    from geort.anchor.generate_robot_anchors import evaluate_analytic_tip_fk
    from geort.utils.config_utils import get_config, parse_config_joint_limit

    config = get_config("custom_right")
    lower, upper = parse_config_joint_limit(config)
    qpos = np.asarray([(np.asarray(lower) + np.asarray(upper)) / 2.0])

    tips = evaluate_analytic_tip_fk(qpos, config)

    assert tips.shape == (1, 5, 3)
    assert np.isfinite(tips).all()
