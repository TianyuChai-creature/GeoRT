# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch 

def chamfer_distance(input_points, target_points):
    """
    Args:
    - input_points (torch.Tensor): Input point cloud tensor of shape [B, N, 3].
    - target_points (torch.Tensor): Target point cloud tensor of shape [B, M, 3].
    
    Returns:
    - chamfer_dist (torch.Tensor): Chamfer distance.
    """
    B, N, _ = input_points.size()
    _, M, _ = target_points.size()
    
    input_points = input_points.clone()
    target_points = target_points.clone()
    input_points[..., 1] = input_points[..., 1] 
    target_points[..., 1] = target_points[..., 1]

    input_points = input_points.unsqueeze(2)    # [B, N, 1, 3]
    target_points = target_points.unsqueeze(1)  # [B, 1, M, 3]
    
    input_points_repeat = input_points.repeat(1, 1, M, 1)    # [B, N, M, 3]
    target_points_repeat = target_points.repeat(1, N, 1, 1)  # [B, N, M, 3]
    

    dist_matrix = torch.sum((input_points_repeat - target_points_repeat)**2, dim=-1)  # [B, N, M]
    
    min_dist_a, _ = torch.min(dist_matrix, dim=2)  # [B, N]
    min_dist_b, _ = torch.min(dist_matrix, dim=1)  # [B, M]
    
    chamfer_dist = torch.mean(min_dist_a, dim=1) + torch.mean(min_dist_b, dim=1)
    
    return chamfer_dist.mean()


def _validate_point_clouds(*point_clouds):
    for points in point_clouds:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                f"Expected point cloud shape [B, N, 3], got {tuple(points.shape)}"
            )


def partial_chamfer(mapped_human, robot_cloud):
    """Return the one-sided squared Chamfer distance from human to robot."""
    _validate_point_clouds(mapped_human, robot_cloud)
    if mapped_human.shape[0] != robot_cloud.shape[0]:
        raise ValueError("Point cloud batch sizes must match")
    distances = torch.cdist(mapped_human, robot_cloud).square()
    return distances.min(dim=-1).values.mean()


def distance_preservation(points, mapped_points, n_pairs):
    """Penalize pairwise distance changes introduced by the mapping."""
    _validate_point_clouds(points, mapped_points)
    if points.shape != mapped_points.shape:
        raise ValueError("Input and mapped point clouds must have the same shape")
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")

    n_points = points.shape[1]
    first = torch.randint(n_points, (n_pairs,), device=points.device)
    second = torch.randint(n_points, (n_pairs,), device=points.device)
    source_distance = torch.linalg.vector_norm(
        points[:, first] - points[:, second], dim=-1
    )
    mapped_distance = torch.linalg.vector_norm(
        mapped_points[:, first] - mapped_points[:, second], dim=-1
    )
    return (source_distance - mapped_distance).square().mean()


def motion_direction_loss(x, fx, dx, dfx):
    """Compare motion in the shared right-handed global hand frame.

    dx and dfx are perturbed positions corresponding to base positions x and
    fx. No per-finger coordinate transform is applied.
    """
    _validate_point_clouds(x, fx, dx, dfx)
    source_motion = torch.nn.functional.normalize(dx - x, dim=-1, eps=1e-8)
    mapped_motion = torch.nn.functional.normalize(dfx - fx, dim=-1, eps=1e-8)
    return -(source_motion * mapped_motion).sum(dim=-1).mean()


def anchor_align_loss(mapped_human_anchor, robot_anchor):
    """Return mean paired L2 distance between mapped and robot anchors."""
    if mapped_human_anchor.shape != robot_anchor.shape:
        raise ValueError("Mapped and robot anchors must have the same shape")
    return torch.linalg.vector_norm(
        mapped_human_anchor - robot_anchor, dim=-1
    ).mean()
