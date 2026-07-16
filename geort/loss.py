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


def synergy_loss(
    joint_physical: torch.Tensor,
    lam: float = 2.0,
    pca_params: dict | None = None,
):
    """Bending-joint synergy regularisation for F2–F5.

    Two modes:
      - pca_params=None: hand-crafted  (β1-β2)² + (β1-λ·β3)²
      - pca_params given: PCA deviation  ||β - proj_PCA(β)||²

    Args:
        joint_physical: Joint angles in physical radians [B, 20].
        lam: Synergy ratio λ (only used in hand-crafted mode).
        pca_params: Optional dict mapping finger_name → {mu, pc}
            where mu [3] = mean of (β1,β2,β3) and pc [3] = principal component.

    Returns:
        (loss, residual_dict) tuple.
    """
    if joint_physical.shape[1] != 20:
        raise ValueError(f"Expected [B, 20], got {tuple(joint_physical.shape)}")

    _FINGER_NAMES = ("index", "middle", "ring", "pinky")

    losses = []
    residuals = {}
    for fi, (beta1_idx, beta2_idx, beta3_idx) in enumerate(_SYNERGY_INDICES):
        b1 = joint_physical[:, beta1_idx]
        b2 = joint_physical[:, beta2_idx]
        b3 = joint_physical[:, beta3_idx]
        B = torch.stack([b1, b2, b3], dim=1)  # [B, 3]

        if pca_params is not None:
            finger_name = _FINGER_NAMES[fi]
            mu = torch.tensor(
                pca_params[finger_name]["mu"],
                device=joint_physical.device,
                dtype=joint_physical.dtype,
            )
            pc = torch.tensor(
                pca_params[finger_name]["pc"],
                device=joint_physical.device,
                dtype=joint_physical.dtype,
            )
            Bc = B - mu.unsqueeze(0)  # [B, 3]
            t = (Bc * pc.unsqueeze(0)).sum(dim=1)  # [B]
            proj = mu.unsqueeze(0) + t.unsqueeze(1) * pc.unsqueeze(0)  # [B, 3]
            losses.append(((B - proj) ** 2).sum(dim=1))  # [B]
            residuals[f"{finger_name}_dev"] = (
                (B - proj).norm(dim=1).mean().item()
            )
        else:
            losses.append((b1 - b2).square() + (b1 - lam * b3).square())
            residuals["beta1_beta2_mean_abs"] = float(
                (b1 - b2).abs().mean().item()
            )
            residuals["beta1_lambda_beta3_mean_abs"] = float(
                (b1 - lam * b3).abs().mean().item()
            )

    loss = torch.stack(losses, dim=1).mean()
    return loss, residuals


def null_space_loss(
    joint_phys: torch.Tensor,
    q_mid: torch.Tensor,
    finger_chains: list,
    finger_chain_joint_idx: list[list[int]],
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
) -> torch.Tensor:
    """Per-finger kinematic null-space regularisation (physical joint space).

    Uses pytorch_kinematics SerialChain Jacobian (analytical, not autograd)
    to compute the 3×4 tip-position Jacobian J for each finger.  SVD gives
    the null-space direction n_phys.  Penalises  (n_phys · (q_phys − q_mid))².

    CRITICAL: ALL quantities (n_phys, q_finger, q_mid_finger) are in the SAME
    physical radian space — n_phys comes from pytorch_kinematics native output
    (NO normalisation applied to joint inputs before chain.jacobian), and
    q_finger / q_mid_finger both derived from joint_phys / q_mid which are
    unnormalised physical radians.

    The entire Jacobian + SVD pipeline runs inside torch.no_grad() — n is a
    kinematic constant and does not need gradient.  Only the q_phys deviation
    term receives gradient.

    Args:
        joint_phys: Physical joint angles [B, 20] in radians.
        q_mid: Mid-range joint angles [20] in radians.
        finger_chains: List of 5 pytorch_kinematics SerialChain objects,
            one per finger, from base_link to the DIP link.
        finger_chain_joint_idx: List of 5 lists of indices into the 20-DOF
            vector that map to the chain's finger-joint columns (columns
            after fixed base joints).  Each inner list has 4 entries.
        joint_lower: Lower joint limits [20] in physical radians.
        joint_upper: Upper joint limits [20] in physical radians.

    Returns:
        Scalar null-space loss averaged over samples and fingers.
    """
    B = joint_phys.shape[0]
    device = joint_phys.device
    D = joint_phys.shape[1]  # 20

    # ── Sanity assertions ──────────────────────────────────────────────
    assert D == 20, f"Expected 20-DOF joint vector, got {D}"
    assert q_mid.shape == (D,), f"Expected q_mid [{D}], got {tuple(q_mid.shape)}"
    assert joint_lower.shape == (D,), f"Expected joint_lower [{D}]"
    assert joint_upper.shape == (D,), f"Expected joint_upper [{D}]"
    # joint_phys must live within [lower, upper] (physical radian range).
    lo_viol = (joint_phys < joint_lower.unsqueeze(0) - 1e-4).any().item()
    hi_viol = (joint_phys > joint_upper.unsqueeze(0) + 1e-4).any().item()
    if lo_viol or hi_viol:
        raise ValueError(
            f"joint_phys out of physical limits: "
            f"min={(joint_phys.min().item()):.3f} vs lower={joint_lower.min().item():.3f}, "
            f"max={(joint_phys.max().item()):.3f} vs upper={joint_upper.max().item():.3f}"
        )
    # q_mid must be within limits too.
    mid_lo = (q_mid < joint_lower - 1e-4).any().item()
    mid_hi = (q_mid > joint_upper + 1e-4).any().item()
    if mid_lo or mid_hi:
        raise ValueError(f"q_mid out of physical limits")
    # ───────────────────────────────────────────────────────────────────

    losses = []

    for fi, chain in enumerate(finger_chains):
        fj_idx = finger_chain_joint_idx[fi]  # 4 indices into 20-DOF

        # Build the [B, nj] joint tensor for this chain.
        # Fixed base joints (WRIST, PALM) → 0; finger joints from joint_phys.
        nj = len(chain.get_joint_parameter_names())
        n_fixed = nj - 4  # wrist + palm
        th = torch.zeros(B, nj, device=device)
        th[:, n_fixed:] = joint_phys[:, fj_idx]  # finger joints (physical rad)

        # Analytical Jacobian (no graph — n is detached below).
        with torch.no_grad():
            J_full = chain.jacobian(th)  # [B, 6, nj] — physical space
            J_lin = J_full[:, :3, n_fixed:]  # [B, 3, 4] — finger joints only

            # SVD → null-space direction n_phys (detached).
            _U, _S, Vh = torch.linalg.svd(J_lin, full_matrices=False)
            n_phys = Vh[:, -1, :]  # [B, 4] — PHYSICAL radian space

            # Assert n_phys is unit vector.
            n_norm = n_phys.norm(dim=-1)  # [B]
            ok = (n_norm > 0.999) & (n_norm < 1.001)
            if not ok.all():
                raise RuntimeError(
                    f"nullspace unit-vector check failed: "
                    f"norm min={n_norm.min().item():.4f} max={n_norm.max().item():.4f}"
                )

        # Deviation from mid-range along null-space direction (gradient flows here).
        # BOTH q_finger and q_mid_finger are in physical radian space, SAME as n_phys.
        q_finger = joint_phys[:, fj_idx]  # [B, 4] — physical rad
        q_mid_finger = q_mid[fj_idx].unsqueeze(0)  # [1, 4] — physical rad
        delta = q_finger - q_mid_finger  # [B, 4] — physical rad
        dev = (delta * n_phys).sum(dim=1)  # [B] — physical rad · unitless = physical rad
        losses.append(dev.square())  # (physical rad)²

    return torch.stack(losses, dim=1).mean()
