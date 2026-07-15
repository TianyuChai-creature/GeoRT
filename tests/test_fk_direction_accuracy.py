"""Verify neural FK directional accuracy for motion loss perturbations."""
from __future__ import annotations

import numpy as np
import torch

from geort.formatter import HandFormatter
from geort.trainer import GeoRTTrainer
from geort.utils.config_utils import get_config, parse_config_keypoint_info, select_keypoint_types


def main():
    config = get_config("custom_right")
    trainer = GeoRTTrainer(config)

    # Train neural FK
    print("Training neural FK ...")
    fk = trainer.get_robot_neural_fk_model(force_train=True)
    print("FK training done.\n")

    # Setup analytical FK keypoints (tip-only)
    info = select_keypoint_types(
        parse_config_keypoint_info(config), allowed_types=("tip",)
    )
    trainer.hand.initialize_keypoint(
        keypoint_link_names=info["link"], keypoint_offsets=info["offset"]
    )

    # Sample random joint pairs
    n_samples = 2000
    joint_low, joint_high = trainer.hand.get_joint_limit()
    joint_low = np.array(joint_low)
    joint_high = np.array(joint_high)

    rng = np.random.default_rng(42)
    q = rng.uniform(joint_low, joint_high, (n_samples, len(joint_low)))

    # Small perturbation in normalized joint space (~ dq_norm of full range)
    dq_norm = 0.001
    dq = rng.standard_normal((n_samples, len(joint_low)))
    dq = dq_norm * dq / (np.linalg.norm(dq, axis=1, keepdims=True) + 1e-8)
    q2 = np.clip(q + dq, joint_low, joint_high)

    # Analytical FK (SAPIEN)
    print("Computing analytical FK ...")
    tip_a = np.array(
        [trainer.hand.keypoint_from_qpos(qi, ret_vec=True) for qi in q]
    )
    tip2_a = np.array(
        [trainer.hand.keypoint_from_qpos(qi, ret_vec=True) for qi in q2]
    )
    d_analytical = tip2_a - tip_a  # [N, K, 3]

    # Neural FK
    print("Computing neural FK ...")
    normalizer = HandFormatter(joint_low, joint_high)
    q_t = normalizer.normalize_torch(torch.from_numpy(q).float()).cuda()
    q2_t = normalizer.normalize_torch(torch.from_numpy(q2).float()).cuda()

    with torch.no_grad():
        tip_n = fk(q_t).cpu().numpy()
        tip2_n = fk(q2_t).cpu().numpy()
    d_neural = tip2_n - tip_n

    # Cosine similarity
    eps = 1e-8
    d_a_norm = d_analytical / (
        np.linalg.norm(d_analytical, axis=2, keepdims=True) + eps
    )
    d_n_norm = d_neural / (
        np.linalg.norm(d_neural, axis=2, keepdims=True) + eps
    )
    cos_sim = (d_a_norm * d_n_norm).sum(axis=2)

    # Filter tiny displacements
    valid = np.linalg.norm(d_analytical, axis=2) > 1e-6
    cos_valid = cos_sim[valid]

    print()
    print("=== FK Directional Accuracy Report ===")
    print(f"Samples: {n_samples}, Valid displacements: {valid.sum()}/{valid.size}")
    print(f"Mean  cos similarity: {cos_valid.mean():.6f}")
    print(f"Median cos similarity: {np.median(cos_valid):.6f}")
    print(f"Min   cos similarity: {cos_valid.min():.6f}")
    print(f"Std   cos similarity: {cos_valid.std():.6f}")
    thresholds = [0.999, 0.99, 0.95, 0.90]
    for t in thresholds:
        pct = (cos_valid < t).mean() * 100
        print(f"% cos < {t}:           {pct:.1f}%")

    print()
    print("--- Per-finger breakdown ---")
    for fi, finger in enumerate(info["finger"]):
        v = np.linalg.norm(d_analytical[:, fi], axis=1) > 1e-6
        cv = cos_sim[:, fi][v]
        print(
            f"  {finger:>8s}: mean={cv.mean():.6f}  "
            f"median={np.median(cv):.6f}  min={cv.min():.6f}"
        )

    verdict = (
        "PASS ✅"
        if cos_valid.mean() >= 0.99
        else "FAIL ⚠️  — increase --motion_delta or use analytical FK"
    )
    print(f"\nVerdict (threshold mean_cos >= 0.99): {verdict}")
    return cos_valid.mean()


if __name__ == "__main__":
    main()
