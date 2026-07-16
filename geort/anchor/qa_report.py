"""CPU-only metrics shared by the custom_right anchor QA report."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
ANCHOR_TYPES = ("lateral", "bending")


def validate_parity_composition(bundle: Mapping[str, object]) -> None:
    """Require the canonical five-finger 250 lateral + 500 bending layout."""
    finger_indices = np.asarray(bundle["finger_indices"], dtype=np.int64)
    anchor_types = np.asarray(bundle["anchor_types"]).astype(str)
    if finger_indices.shape != (750,) or anchor_types.shape != (750,):
        raise ValueError("parity bundle must contain exactly 750 rows")
    for finger_index in range(len(FINGER_NAMES)):
        for anchor_type, expected in (("lateral", 50), ("bending", 100)):
            count = int(
                np.count_nonzero(
                    (finger_indices == finger_index) & (anchor_types == anchor_type)
                )
            )
            if count != expected:
                raise ValueError(
                    f"finger {finger_index} {anchor_type} rows must be {expected}, got {count}"
                )


def trajectory_quality(points: np.ndarray) -> dict[str, float | bool]:
    """Measure interval uniformity and consecutive-step direction agreement."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 2:
        raise ValueError("trajectory points must have shape [N>=2, 3]")
    steps = np.diff(values, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    minimum = float(lengths.min())
    dots = np.einsum("ij,ij->i", steps[:-1], steps[1:])
    return {
        "step_ratio_max_min": float(lengths.max() / minimum)
        if minimum > 1e-12
        else float("inf"),
        "all_direction_dots_positive": bool(np.all(dots > 0.0)),
        "min_direction_dot": float(dots.min()) if dots.size else float("inf"),
    }


def _normalize(points: np.ndarray, stats: Mapping[str, object]) -> np.ndarray:
    center = np.asarray(stats["center"], dtype=np.float64)
    scale = float(stats["scale"])
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization stats must have finite [3] center and positive scale")
    return (np.asarray(points, dtype=np.float64) - center) / scale


def normalized_span_ratio(
    human_points: np.ndarray,
    robot_points: np.ndarray,
    human_stats: Mapping[str, object],
    robot_stats: Mapping[str, object],
) -> float:
    """Return L1→L5 normalized TIP span ratio human / robot."""
    human = _normalize(human_points, human_stats)
    robot = _normalize(robot_points, robot_stats)
    if human.shape != (2, 3) or robot.shape != (2, 3):
        raise ValueError("span endpoints must both have shape [2, 3]")
    robot_span = float(np.linalg.norm(robot[1] - robot[0]))
    if robot_span <= 1e-12:
        raise ValueError("robot normalized span is degenerate")
    return float(np.linalg.norm(human[1] - human[0]) / robot_span)

