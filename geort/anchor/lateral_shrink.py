"""Centered robot-lateral range contraction without altering human anchors."""

from __future__ import annotations

import numpy as np


def scale_knots_to_target_ratio(
    knots: np.ndarray, *, current_ratio: float, target_ratio: float = 0.85
) -> tuple[np.ndarray, float]:
    """Scale five robot knots about L3 by ``current_ratio / target_ratio``."""
    values = np.asarray(knots, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 5 or not np.all(np.isfinite(values)):
        raise ValueError("knots must be finite [5, D]")
    if not np.isfinite(current_ratio) or current_ratio <= 0.0:
        raise ValueError("current_ratio must be positive and finite")
    if not np.isfinite(target_ratio) or target_ratio <= 0.0:
        raise ValueError("target_ratio must be positive and finite")
    multiplier = float(current_ratio / target_ratio)
    midpoint = values[2]
    return midpoint + multiplier * (values - midpoint), multiplier
