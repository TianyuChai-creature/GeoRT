from __future__ import annotations

import numpy as np


def interpolate_sparse_trajectory(
    points: np.ndarray,
    output_count: int,
) -> dict[str, np.ndarray]:
    """Interpolate a trajectory from five uniformly spaced sparse knots."""
    sparse_points = np.asarray(points, dtype=float)
    if (
        sparse_points.ndim != 2
        or sparse_points.shape[0] != 5
        or sparse_points.shape[1] == 0
        or not np.all(np.isfinite(sparse_points))
    ):
        raise ValueError("points must have shape [5, D] with D > 0 and be finite")
    if (
        isinstance(output_count, (bool, np.bool_))
        or not isinstance(output_count, (int, np.integer))
        or output_count < 5
    ):
        raise ValueError("output_count must be an integer of at least 5")

    sparse_t = np.linspace(0.0, 1.0, 5)
    trajectory_t = np.linspace(0.0, 1.0, int(output_count))
    interpolated_points = np.column_stack(
        [
            np.interp(trajectory_t, sparse_t, sparse_points[:, dimension])
            for dimension in range(sparse_points.shape[1])
        ]
    )

    upper_indices = np.searchsorted(
        sparse_t,
        trajectory_t,
        side="right",
    ).clip(1, 4)
    source_sparse_indices = np.column_stack(
        (upper_indices - 1, upper_indices)
    )

    return {
        "points": interpolated_points,
        "trajectory_t": trajectory_t,
        "source_sparse_indices": source_sparse_indices,
    }
