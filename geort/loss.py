# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np

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


def partial_chamfer_distance(input_points, target_points):
    """One-way mean nearest-neighbor L2 distance from input to target.

    L_P-Chamfer = (1/|C_H|) * sum_j min_k ||f_m(x_j^H) - x_k^R||

    Args:
        input_points: Mapped human points [B, N, 3].
        target_points: Robot target points [B, M, 3].

    Returns:
        Scalar partial Chamfer distance.
    """
    pairwise_distance = torch.cdist(input_points, target_points, p=2.0)  # [B, N, M]
    min_distance, _ = pairwise_distance.min(dim=2)  # [B, N]
    return min_distance.mean()


def distance_preservation(points, mapped_points):
    """Per-finger intra-workspace isometry constraint.

    For each finger i, compare pairwise distances among B sampled positions
    before and after mapping.  Prevents collapse, stretch, or local distortion
    of the fingertip workspace.

    L_dist^i = 1/(B*(B-1)) * sum_{j1!=j2} (||f_m(x_j1)-f_m(x_j2)|| - ||x_j1-x_j2||)^2

    Args:
        points: Original human tip positions [B, K, 3].
        mapped_points: Mapped robot tip positions [B, K, 3].

    Returns:
        Scalar loss averaged over fingers.
    """
    B, K, _ = points.shape
    if B < 2:
        return torch.zeros((), device=points.device, dtype=points.dtype)
    mask = ~torch.eye(B, dtype=torch.bool, device=points.device)
    losses = []
    for i in range(K):
        orig_dists = torch.cdist(points[:, i, :], points[:, i, :])       # [B, B]
        mapped_dists = torch.cdist(mapped_points[:, i, :], mapped_points[:, i, :])  # [B, B]
        squared_diff = (mapped_dists - orig_dists).square()
        losses.append(squared_diff[mask].mean())
    return torch.stack(losses).mean()


def local_motion_loss(d_human, d_robot):
    """Local motion preservation via negative cosine similarity.

    L_motion = -1/|C| * sum_j ⟨T⁻¹(x_j)·Δx/‖Δx‖, T⁻¹(f_m(x_j))·Δf_m/‖Δf_m‖⟩

    Masks out samples where mapped displacement ‖Δf_m‖ < 1e-6 (joint
    saturation, mapping flat-regions) to avoid NaN from near-zero division.

    Args:
        d_human: Raw perturbation vectors in human space [B, K, 3].
        d_robot: Raw perturbation vectors in robot space [B, K, 3].

    Returns:
        (loss, invalid_fraction) tuple.
    """
    d_human_hat = F.normalize(d_human, dim=-1, p=2, eps=1e-8)
    norm_r = d_robot.norm(dim=-1, keepdim=True)  # [B, K, 1]
    valid = (norm_r.squeeze(-1) > 1e-6)  # [B, K]
    total = valid.numel()
    invalid_count = total - valid.sum().item() if total > 0 else 0
    invalid_frac = invalid_count / total if total > 0 else 0.0
    if not valid.any():
        return torch.zeros((), device=d_robot.device, dtype=d_robot.dtype), 1.0
    cos = (d_human_hat * d_robot / norm_r.clamp(min=1e-8)).sum(-1)  # [B, K]
    return -cos[valid].mean(), invalid_frac


# F2–F5 bending joint indices in the 20-DOF config joint_order.
# Per finger: [β1=MCP1, β2=PIP, β3=DIP].  MCP2 (abduction α) excluded.
# Thumb (F1, indices 0–3) excluded — λ=2 synergy does not apply.
_SYNERGY_INDICES: tuple[tuple[int, int, int], ...] = (
    (5, 6, 7),    # F2: MCP1, PIP, DIP
    (9, 10, 11),  # F3
    (13, 14, 15), # F4
    (17, 18, 19), # F5
)


def synergy_loss(joint_physical: torch.Tensor, lam: float = 2.0):
    """Bending-joint synergy regularisation for F2–F5.

    Penalises deviation from β1=β2=λ·β3 on the four non-thumb fingers:
      L_syn = mean_over_fingers (β1-β2)² + (β1-λ·β3)²

    Args:
        joint_physical: Joint angles in physical radians [B, 20].
        lam: Synergy ratio λ (default 2.0).

    Returns:
        (loss, residual_dict) tuple.
    """
    if joint_physical.shape[1] != 20:
        raise ValueError(f"Expected [B, 20], got {tuple(joint_physical.shape)}")
    losses = []
    beta12_res = []
    beta13_res = []
    for beta1_idx, beta2_idx, beta3_idx in _SYNERGY_INDICES:
        b1 = joint_physical[:, beta1_idx]
        b2 = joint_physical[:, beta2_idx]
        b3 = joint_physical[:, beta3_idx]
        losses.append((b1 - b2).square() + (b1 - lam * b3).square())
        beta12_res.append((b1 - b2).abs().mean().item())
        beta13_res.append((b1 - lam * b3).abs().mean().item())
    loss = torch.stack(losses, dim=1).mean()
    residuals = {
        "beta1_beta2_mean_abs": float(np.mean(beta12_res)),
        "beta1_lambda_beta3_mean_abs": float(np.mean(beta13_res)),
    }
    return loss, residuals
