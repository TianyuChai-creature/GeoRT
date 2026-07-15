"""Position and direction accuracy tests for AnalyticFK vs SAPIEN (ground truth).

Runs two comparisons:
  Position: 1000 random joint configs → analytic FK vs SAPIEN FK.
  Direction: 200 random (q, q+δq) pairs → compare directional accuracy of
             neural FK and analytic FK against SAPIEN ground truth.
"""

from __future__ import annotations

import numpy as np
import torch

from geort.analytic_fk import AnalyticFK
from geort.formatter import HandFormatter
from geort.trainer import GeoRTTrainer
from geort.utils.config_utils import get_config, parse_config_keypoint_info, select_keypoint_types

# Suppress URDF parse warnings about unknown attributes.
import warnings
warnings.filterwarnings("ignore")


def _get_hand_and_limits():
    config = get_config("custom_right")
    trainer = GeoRTTrainer(config)
    jl, jh = trainer.hand.get_joint_limit()
    return trainer, np.array(jl), np.array(jh)


def test_position_accuracy():
    """Compare AnalyticFK vs SAPIEN FK on 1000 random joint configs."""
    print("=" * 60)
    print("POSITION ACCURACY: AnalyticFK vs SAPIEN")
    print("=" * 60)

    trainer, jl, jh = _get_hand_and_limits()
    info = select_keypoint_types(
        parse_config_keypoint_info(trainer.config), allowed_types=("tip",)
    )
    trainer.hand.initialize_keypoint(
        keypoint_link_names=info["link"], keypoint_offsets=info["offset"]
    )

    afk = AnalyticFK("assets/custom_right/URDF_R.urdf", jl, jh,
                     tip_offsets=info["offset"])
    normalizer = HandFormatter(jl, jh)

    n = 1000
    rng = np.random.default_rng(42)
    q = rng.uniform(jl, jh, (n, len(jl)))

    # SAPIEN FK
    print(f"Computing SAPIEN FK for {n} samples ...")
    tips_sapien = np.array([
        trainer.hand.keypoint_from_qpos(qi, ret_vec=True) for qi in q
    ])  # [N, 5, 3]

    # Analytic FK
    print("Computing Analytic FK ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q_norm = normalizer.normalize_torch(torch.from_numpy(q).float().to(device))
    with torch.no_grad():
        tips_analytic = afk.forward(q_norm).cpu().numpy()  # [N, 5, 3]

    # Error
    errors = np.linalg.norm(tips_analytic - tips_sapien, axis=2)  # [N, 5]
    finger_names = info["finger"]

    print()
    print(f"{'Finger':>10s}  {'Max error (m)':>14s}  {'Mean error (m)':>15s}")
    print("-" * 45)
    overall_max = 0.0
    overall_mean = 0.0
    for fi, finger in enumerate(finger_names):
        emax = errors[:, fi].max()
        emean = errors[:, fi].mean()
        overall_max = max(overall_max, emax)
        overall_mean += emean
        print(f"{finger:>10s}  {emax:>14.6e}  {emean:>15.6e}")

    overall_mean /= len(finger_names)
    print("-" * 45)
    print(f"{'OVERALL':>10s}  {overall_max:>14.6e}  {overall_mean:>15.6e}")

    threshold = 1e-4
    passed = overall_max < threshold
    verdict = "PASS" if passed else "FAIL"
    print(f"\nVerdict (max error < {threshold:.0e} m): {verdict}")
    if not passed:
        print("\nDiagnostics:")
        # Which fingers/joint-regions are worst?
        worst_finger = np.argmax(errors.max(axis=0))
        worst_sample = np.argmax(errors.max(axis=1))
        print(f"  Worst finger: {finger_names[worst_finger]} "
              f"(max error = {errors[:, worst_finger].max():.4e} m)")
        print(f"  Worst sample: #{worst_sample} "
              f"(joint angles near limits? q_min={q[worst_sample].min():.3f}, "
              f"q_max={q[worst_sample].max():.3f})")
        # Per-axis breakdown for worst finger
        diff = tips_analytic - tips_sapien
        for axis, name in enumerate(["X", "Y", "Z"]):
            ax_err = np.abs(diff[:, worst_finger, axis])
            print(f"  {name}-axis bias: mean={ax_err.mean():.4e}  max={ax_err.max():.4e}")

    return passed


def test_direction_accuracy():
    """Compare directional accuracy of neural FK and analytic FK vs SAPIEN."""
    print("\n" + "=" * 60)
    print("DIRECTION ACCURACY: Neural FK vs Analytic FK vs SAPIEN")
    print("=" * 60)

    trainer, jl, jh = _get_hand_and_limits()
    info = select_keypoint_types(
        parse_config_keypoint_info(trainer.config), allowed_types=("tip",)
    )
    trainer.hand.initialize_keypoint(
        keypoint_link_names=info["link"], keypoint_offsets=info["offset"]
    )

    afk = AnalyticFK("assets/custom_right/URDF_R.urdf", jl, jh,
                     tip_offsets=info["offset"])
    normalizer = HandFormatter(jl, jh)

    n = 200
    rng = np.random.default_rng(123)
    q = rng.uniform(jl, jh, (n, len(jl)))

    # δq: target ~0.5 mm tip displacement.
    # dq_norm is the L2 norm (rad) of the joint perturbation vector.
    # Calibrated: dq_norm ≈ 0.03 rad → ~0.5 mm tip displacement.
    dq_norm = 0.03
    dq_raw = rng.standard_normal((n, len(jl)))
    dq = dq_norm * dq_raw / (np.linalg.norm(dq_raw, axis=1, keepdims=True) + 1e-12)
    q2 = np.clip(q + dq, jl, jh)

    # SAPIEN (ground truth) — run FIRST so neural FK training does not
    # interfere with GPU state / memory.
    print(f"Computing SAPIEN FK for {n}×(q,q+δq) pairs ...")
    ta = np.array([trainer.hand.keypoint_from_qpos(qi, ret_vec=True) for qi in q])
    t2a = np.array([trainer.hand.keypoint_from_qpos(qi, ret_vec=True) for qi in q2])
    da = t2a - ta  # [N, 5, 3]

    # Displacement magnitude guard: prevent silent regression of dq scaling.
    tip_disp_mm = np.linalg.norm(da, axis=2).mean() * 1000
    print(f"Measured tip displacement: {tip_disp_mm:.3f} mm (mean)")
    assert 0.3 < tip_disp_mm < 0.8, (
        f"Tip displacement {tip_disp_mm:.3f} mm out of [0.3, 0.8] mm band. "
        f"Check dq_norm ({dq_norm}) / dq generation logic."
    )

    # Analytic FK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qn = normalizer.normalize_torch(torch.from_numpy(q).float().to(device))
    q2n = normalizer.normalize_torch(torch.from_numpy(q2).float().to(device))
    with torch.no_grad():
        tn_analytic = afk.forward(qn).cpu().numpy()
        t2n_analytic = afk.forward(q2n).cpu().numpy()
    dn_analytic = t2n_analytic - tn_analytic

    # Neural FK — train/load AFTER SAPIEN and analytic FK are done.
    fk_neural = trainer.get_robot_neural_fk_model()
    fk_neural.to(device)
    with torch.no_grad():
        tn_neural = fk_neural(qn).cpu().numpy()
        t2n_neural = fk_neural(q2n).cpu().numpy()
    dn_neural = t2n_neural - tn_neural

    def cos_sim(d_ref, d_test):
        eps = 1e-8
        ref_n = d_ref / (np.linalg.norm(d_ref, axis=2, keepdims=True) + eps)
        test_n = d_test / (np.linalg.norm(d_test, axis=2, keepdims=True) + eps)
        return (ref_n * test_n).sum(axis=2)

    cs_analytic = cos_sim(da, dn_analytic)
    cs_neural = cos_sim(da, dn_neural)

    valid = np.linalg.norm(da, axis=2) > 1e-6

    finger_names = info["finger"]
    print()
    print(f"{'Finger':>10s}  {'Analytic mean cos':>16s}  {'Neural mean cos':>16s}")
    print("-" * 50)
    for fi, finger in enumerate(finger_names):
        v = valid[:, fi]
        ca = cs_analytic[:, fi][v].mean()
        cn = cs_neural[:, fi][v].mean()
        print(f"{finger:>10s}  {ca:>16.6f}  {cn:>16.6f}")

    ca_all = cs_analytic[valid].mean()
    cn_all = cs_neural[valid].mean()
    print("-" * 50)
    print(f"{'OVERALL':>10s}  {ca_all:>16.6f}  {cn_all:>16.6f}")

    print(f"\nAnalytic FK — % cos < 0.9999: {(cs_analytic[valid] < 0.9999).mean()*100:.1f}%")
    print(f"Analytic FK — % cos < 0.999:  {(cs_analytic[valid] < 0.999).mean()*100:.1f}%")
    print(f"Neural  FK — % cos < 0.99:   {(cs_neural[valid] < 0.99).mean()*100:.1f}%")
    print(f"Neural  FK — % cos < 0.95:   {(cs_neural[valid] < 0.95).mean()*100:.1f}%")

    improvement = ca_all - cn_all
    print(f"\nAnalytic FK improves mean cosine by: {improvement:+.6f} "
          f"({'↑' if improvement > 0 else '↓'} over neural FK)")


if __name__ == "__main__":
    pos_ok = test_position_accuracy()
    test_direction_accuracy()
    if not pos_ok:
        print("\n⚠️  Position accuracy check FAILED — see diagnostics above.")
