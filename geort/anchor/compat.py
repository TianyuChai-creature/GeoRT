"""Current-branch adapters injected around the unchanged anchor modules."""

from __future__ import annotations

from typing import Any

import numpy as np

import torch

def get_joint_limits(hand_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Read user-order physical joint limits from the current hand model."""
    from geort.env.hand import HandKinematicModel

    hand = HandKinematicModel.build_from_config(hand_config, render=False)
    lower, upper = hand.get_joint_limit()
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

def make_analytic_tip_callback(
    hand_config: dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
    tip_offsets: object,
    *,
    side: str = "R",
):
    """Inject current AnalyticFK behind the legacy physical-qpos callback.

    AnalyticFK receives normalized qpos and internally converts it back to
    physical float32 angles. This physical→normalized→physical round-trip adds
    micrometre-scale TIP noise, harmless for the 1 mm parity gate but not a
    reusable path for future micrometre-sensitive applications.
    """
    from geort.analytic_fk import AnalyticFK

    fk = AnalyticFK(
        hand_config["urdf_path"], lower, upper, tip_offsets=tip_offsets, side=side
    )
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)

    def evaluate(qpos: np.ndarray, finger_index: int) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float32)
        normalised = 2.0 * (qpos - lower) / (upper - lower) - 1.0
        with torch.no_grad():
            tips = fk(torch.from_numpy(normalised[None, :]))
        return tips[0, finger_index].cpu().numpy().astype(np.float64)

    return evaluate
