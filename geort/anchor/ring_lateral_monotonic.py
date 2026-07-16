"""Minimal constrained L2/L3 medoid choice for Ring lateral anchors."""

from __future__ import annotations

import numpy as np


def choose_monotonic_pair(
    *,
    level2_order: np.ndarray,
    level3_order: np.ndarray,
    projection: dict[int, float],
    fixed_projections: tuple[float, float, float],
) -> tuple[int, int] | None:
    """Return the first descriptor-ranked L2/L3 pair with ordered projections."""
    level1, level4, level5 = fixed_projections
    for row2 in np.asarray(level2_order, dtype=np.int64):
        p2 = projection[int(row2)]
        if not level1 < p2 < level4:
            continue
        for row3 in np.asarray(level3_order, dtype=np.int64):
            p3 = projection[int(row3)]
            if p2 < p3 < level4 < level5:
                return int(row2), int(row3)
    return None
