"""Robot-side sparse anchor trajectories independent of human angle values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from geort.anchor.mining import LEVEL_FRACTIONS


@dataclass(frozen=True, slots=True)
class RobotFingerJoints:
    """User-qpos indices for one robot finger's four controlled joints."""

    mcp2: int
    mcp1: int
    pip: int
    dip: int


def _limits(lower: object, upper: object) -> tuple[np.ndarray, np.ndarray]:
    try:
        low = np.asarray(lower, dtype=np.float64)
        high = np.asarray(upper, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("joint limits must be finite one-dimensional arrays") from error
    if (
        low.ndim != 1
        or high.shape != low.shape
        or low.size == 0
        or not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or np.any(low > high)
    ):
        raise ValueError("joint limits must be finite matching arrays with lower <= upper")
    return low, high


def _indices(joints: RobotFingerJoints, dof: int) -> tuple[int, int, int, int]:
    values = (joints.mcp2, joints.mcp1, joints.pip, joints.dip)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
        raise ValueError("finger joint indices must be integers")
    indices = tuple(int(value) for value in values)
    if len(set(indices)) != 4 or any(value < 0 or value >= dof for value in indices):
        raise ValueError("finger joint indices must be four distinct in-range values")
    return indices


def neutral_qpos(lower: object, upper: object) -> np.ndarray:
    """Use q=0 where feasible, otherwise the nearest mechanical endpoint."""
    low, high = _limits(lower, upper)
    return np.clip(np.zeros(low.shape, dtype=np.float64), low, high)


def _base_knots(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.repeat(neutral_qpos(lower, upper)[None, :], 5, axis=0)


def build_lateral_knots(
    lower: object,
    upper: object,
    joints: RobotFingerJoints,
) -> np.ndarray:
    """Span the target MCP2's own feasible range in five geometric levels."""
    low, high = _limits(lower, upper)
    mcp2, _, _, _ = _indices(joints, low.size)
    if not high[mcp2] > low[mcp2]:
        raise ValueError("target MCP2 has a degenerate feasible range")
    qpos = _base_knots(low, high)
    qpos[:, mcp2] = low[mcp2] + LEVEL_FRACTIONS * (high[mcp2] - low[mcp2])
    return qpos


def coupled_bending_interval(
    lower: object,
    upper: object,
    joints: RobotFingerJoints,
) -> tuple[float, float]:
    """Find b where ``[MCP1, PIP, DIP] = [b, b, b/2]`` is feasible."""
    low, high = _limits(lower, upper)
    _, mcp1, pip, dip = _indices(joints, low.size)
    bend_low = float(max(low[mcp1], low[pip], 2.0 * low[dip]))
    bend_high = float(min(high[mcp1], high[pip], 2.0 * high[dip]))
    if not bend_high > bend_low:
        raise ValueError("coupled bending trajectory has no feasible interval")
    return bend_low, bend_high


def _coupled_bending_qpos(
    lower: object,
    upper: object,
    joints: RobotFingerJoints,
    bend_values: np.ndarray,
) -> np.ndarray:
    low, high = _limits(lower, upper)
    mcp2, mcp1, pip, dip = _indices(joints, low.size)
    qpos = np.repeat(neutral_qpos(low, high)[None, :], bend_values.size, axis=0)
    qpos[:, mcp2] = neutral_qpos(low, high)[mcp2]
    qpos[:, mcp1] = bend_values
    qpos[:, pip] = bend_values
    qpos[:, dip] = bend_values / 2.0
    if np.any(qpos < low) or np.any(qpos > high):  # defensive against future changes
        raise ValueError("coupled bending qpos violates mechanical limits")
    return qpos


def build_non_thumb_bending_knots(
    lower: object,
    upper: object,
    joints: RobotFingerJoints,
) -> np.ndarray:
    """Construct five non-thumb bending knots from the robot's own b interval."""
    bend_low, bend_high = coupled_bending_interval(lower, upper, joints)
    return _coupled_bending_qpos(
        lower,
        upper,
        joints,
        np.linspace(bend_low, bend_high, 5),
    )


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 5
    ):
        raise ValueError(f"{name} must be an integer of at least 5")
    return int(value)


def build_thumb_arc_knots(
    lower: object,
    upper: object,
    joints: RobotFingerJoints,
    tip_fk: Callable[[np.ndarray], np.ndarray],
    *,
    dense_count: int = 201,
) -> np.ndarray:
    """Sample five thumb bends by uniform arc length of exact-FK TIP motion."""
    count = _positive_integer(dense_count, name="dense_count")
    bend_low, bend_high = coupled_bending_interval(lower, upper, joints)
    dense_qpos = _coupled_bending_qpos(
        lower,
        upper,
        joints,
        np.linspace(bend_low, bend_high, count),
    )
    try:
        tips = np.asarray([tip_fk(qpos) for qpos in dense_qpos], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("tip_fk must return finite [3] points") from error
    if tips.shape != (count, 3) or not np.all(np.isfinite(tips)):
        raise ValueError("tip_fk must return finite [3] points")
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(tips, axis=0), axis=1))))
    if not arc[-1] > 0.0 or not np.isfinite(arc[-1]):
        raise ValueError("thumb exact-FK trajectory has degenerate arc length")
    targets = LEVEL_FRACTIONS * arc[-1]
    selected = np.searchsorted(arc, targets, side="left")
    selected[0] = 0
    selected[-1] = count - 1
    selected = np.maximum.accumulate(selected)
    if np.unique(selected).size != 5:
        raise ValueError("thumb exact-FK trajectory cannot provide five distinct arc levels")
    return dense_qpos[selected]


def derive_finger_joint_layouts(
    joint_order: Sequence[str],
) -> tuple[RobotFingerJoints, ...]:
    """Derive thumb-to-pinky MCP2/MCP1/PIP/DIP indices from custom joint names."""
    names = tuple(str(name) for name in joint_order)
    layouts: list[RobotFingerJoints] = []
    for finger_number in range(1, 6):
        prefix = f"F{finger_number}-"
        matching = {
            component: next(
                (
                    index
                    for index, name in enumerate(names)
                    if name.startswith(prefix) and name.endswith(component)
                ),
                None,
            )
            for component in ("MCP2", "MCP1", "PIP", "DIP")
        }
        if any(index is None for index in matching.values()):
            raise ValueError(
                f"joint_order lacks a complete MCP2/MCP1/PIP/DIP block for F{finger_number}"
            )
        layouts.append(
            RobotFingerJoints(
                mcp2=int(matching["MCP2"]),
                mcp1=int(matching["MCP1"]),
                pip=int(matching["PIP"]),
                dip=int(matching["DIP"]),
            )
        )
    return tuple(layouts)
