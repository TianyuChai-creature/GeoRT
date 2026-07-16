from __future__ import annotations

import numpy as np
import torch

from geort.analytic_fk import AnalyticFK
from geort.trainer import GeoRTTrainer
from geort.utils.config_utils import get_config, parse_config_keypoint_info, select_keypoint_types


def test_analytic_fk_optional_link_rotations_preserve_tip_output():
    config = get_config("custom_right")
    trainer = GeoRTTrainer(config)
    lower, upper = trainer.hand.get_joint_limit()
    keypoints = select_keypoint_types(
        parse_config_keypoint_info(config), allowed_types=("tip",)
    )
    fk = AnalyticFK(
        "assets/custom_right/URDF_R.urdf", lower, upper, tip_offsets=keypoints["offset"]
    )
    q = torch.linspace(-0.8, 0.8, 40, dtype=torch.float32).reshape(2, 20)

    positions = fk(q)
    positions_with_rotation, rotations = fk(q, return_link_rotations=True)

    torch.testing.assert_close(positions_with_rotation, positions, rtol=0.0, atol=0.0)
    assert rotations.shape == (2, 5, 3, 3)
    torch.testing.assert_close(
        rotations.transpose(-1, -2) @ rotations,
        torch.eye(3).expand_as(rotations),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(torch.linalg.det(rotations).detach().numpy(), 1.0, atol=1e-5)
