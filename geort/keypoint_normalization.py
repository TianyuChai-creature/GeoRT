"""Per-finger point normalization shared by training and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


NormalizationStats = dict[str, dict[str, object]]


def _validate_points(points, finger_names: Sequence[str]):
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape [N, K, 3], got {tuple(points.shape)}")
    if points.shape[1] != len(finger_names):
        raise ValueError(
            f"Got {points.shape[1]} keypoints but {len(finger_names)} finger labels"
        )
    return points


def fit_finger_normalization(
    points: np.ndarray,
    finger_names: Sequence[str],
) -> NormalizationStats:
    """Fit one AABB center and one isotropic max-axis scale per finger."""
    points = _validate_points(np.asarray(points), finger_names)
    if not np.isfinite(points).all():
        raise ValueError("Point data contains non-finite values")

    names = np.asarray(finger_names)
    stats: NormalizationStats = {}
    for finger in dict.fromkeys(finger_names):
        finger_points = points[:, names == finger, :].reshape(-1, 3).astype(np.float64)
        # Clip to [0.5, 99.5] percentiles per axis to guard against
        # HTS tracking glitches or lost-frame outliers that would
        # inflate the AABB and compress valid data.
        lo = np.percentile(finger_points, 0.5, axis=0)
        hi = np.percentile(finger_points, 99.5, axis=0)
        center = (lo + hi) / 2.0
        scale = float((hi - lo).max() / 2.0)
        if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
            raise ValueError(f"Finger {finger!r} has a degenerate workspace")
        stats[finger] = {"center": center, "scale": scale}
    return stats


def normalize_finger_points(
    points: np.ndarray,
    finger_names: Sequence[str],
    stats: Mapping[str, Mapping[str, object]],
) -> np.ndarray:
    """Normalize each finger independently: (x - center) / scale  →  [-1, 1]."""
    points = _validate_points(np.asarray(points, dtype=np.float32), finger_names)
    out = points.copy()
    for i, finger in enumerate(finger_names):
        center = np.asarray(stats[finger]["center"], dtype=np.float32)
        scale = float(stats[finger]["scale"])
        out[:, i, :] = (out[:, i, :] - center) / scale
    return out


def normalize_finger_points_torch(
    points: torch.Tensor,
    finger_names: Sequence[str],
    stats: Mapping[str, Mapping[str, object]],
) -> torch.Tensor:
    """Torch equivalent of normalize_finger_points, preserves gradients."""
    _validate_points(points, finger_names)
    out = points.clone()
    for i, finger in enumerate(finger_names):
        center = torch.as_tensor(
            stats[finger]["center"], device=points.device, dtype=points.dtype
        )
        scale = float(stats[finger]["scale"])
        out[:, i, :] = (out[:, i, :] - center) / scale
    return out


def normalization_stats_to_json(
    stats: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Serialize normalization statistics for checkpoint persistence."""
    return {
        finger: {
            "center": np.asarray(v["center"]).tolist(),
            "scale": float(v["scale"]),
        }
        for finger, v in stats.items()
    }
