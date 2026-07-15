# FK Backend Ablation

Evaluation: 1000 random frames (seed=42) from `data/hts_right.npy` (234114 total).
Coverage verified against full dataset: 88–100% per axis.
GeoRT baseline (main branch) is pending — main branch pipeline incompatible with current dataset setup.

| Backend | Signed Gain | Rest Offset | Sat (any) |
|---------|------------|-------------|-----------|
| GeoRT baseline (main) | — | — | — |
| Neural FK | 0.925 | 16.9% | 17.5% |
| Analytic FK (ours) | 0.872 | 17.8% | 15.1% |

- **Gain** = median(qpos_range / joint_range), ~1.0 ideal.
- **Rest offset** = median |qpos - q_default| / joint_range (lower better).
- **Saturation** = fraction of frames at joint limits (0% ideal).

Training configuration (both AnyDexRT checkpoints):
- chamfer=1.0, distance=1.0, curvature=0.1, motion=1.0, pinch=0, collision=0
- fk_backend varies, motion_delta=0.01, 200 epochs, human-shaped target cloud disabled (uniform)
