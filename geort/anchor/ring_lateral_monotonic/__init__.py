"""Constrained Ring lateral medoid chooser (injected package override)."""

from __future__ import annotations

import numpy as np


def choose_monotonic_pair(
    *,
    level2_order: np.ndarray,
    level3_order: np.ndarray,
    projection: dict[int, float],
    fixed_projections: tuple[float, float, float],
) -> tuple[int, int] | None:
    """Choose a monotonic pair by minimum combined descriptor-rank cost."""
    level1, level4, level5 = fixed_projections
    options: list[tuple[int, int, int, int]] = []
    for rank2, row2 in enumerate(np.asarray(level2_order, dtype=np.int64)):
        p2 = projection[int(row2)]
        if not level1 < p2 < level4:
            continue
        for rank3, row3 in enumerate(np.asarray(level3_order, dtype=np.int64)):
            p3 = projection[int(row3)]
            if p2 < p3 < level4 < level5:
                options.append((rank2 + rank3, rank3, rank2, int(row2), int(row3)))
    if not options:
        return None
    _, _, _, row2, row3 = min(options)
    return row2, row3
