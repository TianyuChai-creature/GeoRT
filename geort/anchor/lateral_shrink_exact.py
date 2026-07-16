"""Exact sparse-knot recovery from linearly interpolated pair trajectories."""

from __future__ import annotations

import numpy as np


def exact_level_knots(values: np.ndarray, trajectory_t: np.ndarray) -> np.ndarray:
    """Evaluate a linear pair trajectory exactly at its five sparse levels."""
    points = np.asarray(values, dtype=np.float64)
    times = np.asarray(trajectory_t, dtype=np.float64)
    if points.ndim != 2 or times.shape != (points.shape[0],):
        raise ValueError("values must be [N,D] with matching trajectory_t")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("trajectory_t must be strictly increasing")
    return np.column_stack([
        np.interp(np.linspace(0.0, 1.0, 5), times, points[:, dimension])
        for dimension in range(points.shape[1])
    ])
