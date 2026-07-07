from __future__ import annotations

import pytest
import torch

from geort.trainer import compute_mcp1_fist_prior_loss, non_thumb_mcp1_joint_indices


def test_non_thumb_mcp1_joint_indices_skips_thumb_and_other_joints() -> None:
    joint_order = [
        "F1-R-MCP1",
        "F2-R-MCP2",
        "F2-R-MCP1",
        "F2-R-PIP",
        "F3-R-MCP1",
        "F4-R-MCP1",
        "F5-R-MCP1",
    ]

    assert non_thumb_mcp1_joint_indices(joint_order) == [2, 4, 5, 6]


def test_compute_mcp1_fist_prior_loss_only_pushes_masked_mcp1_toward_upper_limit() -> None:
    joint = torch.tensor(
        [
            [0.0, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=torch.float32,
    )
    fist_mask = torch.tensor([1.0, 0.0], dtype=torch.float32)

    loss = compute_mcp1_fist_prior_loss(
        joint,
        fist_mask=fist_mask,
        mcp1_indices=[1, 3],
        target_alpha=0.5,
    )

    target_1 = 0.2 + 0.5 * (1.0 - 0.2)
    target_3 = 0.4 + 0.5 * (1.0 - 0.4)
    expected = ((0.2 - target_1) ** 2 + (0.4 - target_3) ** 2) / 2.0
    assert loss.item() == pytest.approx(expected)


def test_compute_mcp1_fist_prior_loss_is_zero_without_masked_frames() -> None:
    joint = torch.tensor([[0.0, 0.2, 0.3, 0.4]], dtype=torch.float32)
    fist_mask = torch.tensor([0.0], dtype=torch.float32)

    loss = compute_mcp1_fist_prior_loss(
        joint,
        fist_mask=fist_mask,
        mcp1_indices=[1, 3],
        target_alpha=0.5,
    )

    assert loss.item() == 0.0
