from __future__ import annotations

import torch

from geort import loss as loss_module


def _loss(name: str):
    assert hasattr(loss_module, name), f"geort.loss.{name} is not implemented"
    return getattr(loss_module, name)


def test_partial_chamfer_ignores_unused_target_regions() -> None:
    mapped = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    target = mapped.clone()
    target_with_extra_region = torch.cat(
        [target, torch.tensor([[[100.0, 100.0, 100.0]]])], dim=1
    )

    partial_chamfer = _loss("partial_chamfer")
    assert partial_chamfer(mapped, target).item() == 0.0
    assert partial_chamfer(mapped, target_with_extra_region).item() == 0.0


def test_distance_preservation_is_zero_for_rigid_transform() -> None:
    points = torch.tensor(
        [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]]
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mapped = points @ rotation.T + torch.tensor([3.0, -2.0, 5.0])

    loss = _loss("distance_preservation")(points, mapped, n_pairs=64)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_motion_direction_loss_is_optimal_for_identity_mapping() -> None:
    x = torch.zeros((1, 2, 3))
    fx = x.clone()
    delta = torch.tensor([[[1.0, 2.0, 0.0], [0.0, -1.0, 1.0]]])

    loss = _loss("motion_direction_loss")(x, fx, x + delta, fx + delta)

    assert torch.allclose(loss, torch.tensor(-1.0), atol=1e-6)


def test_anchor_align_loss_is_zero_for_matching_anchors() -> None:
    anchors = torch.randn(4, 5, 3)

    assert _loss("anchor_align_loss")(anchors, anchors).item() == 0.0


def test_partial_chamfer_uses_l2_distance_not_squared_distance() -> None:
    mapped = torch.tensor([[[0.0, 0.0, 0.0]]])
    target = torch.tensor([[[3.0, 4.0, 0.0]]])

    assert _loss("partial_chamfer")(mapped, target).item() == 5.0


def test_distance_preservation_samples_different_batch_items() -> None:
    points = torch.tensor([[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
    collapsed = torch.zeros_like(points)

    loss = _loss("distance_preservation")(points, collapsed, n_pairs=8)

    assert loss.item() == 4.0
