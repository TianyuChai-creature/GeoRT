"""Raw sparse-anchor loading under the current-run normalization contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geort.keypoint_normalization import normalize_finger_points


@dataclass(frozen=True)
class AnchorTrainingPoints:
    human_contexts: np.ndarray
    robot_targets: np.ndarray
    finger_indices: np.ndarray


def load_raw_anchor_training_points(anchor_path: Path | str, normalization_path: Path | str, finger_names: list[str]) -> AnchorTrainingPoints:
    """Load metric anchors, enforce shared hand-base/metre contract, then normalize."""
    with np.load(anchor_path, allow_pickle=False) as bundle:
        contexts = np.asarray(bundle["human_tip_contexts"], dtype=np.float32)
        targets = np.asarray(bundle["robot_points"], dtype=np.float32)
        indices = np.asarray(bundle["finger_indices"], dtype=np.int64)
        metadata = json.loads(str(bundle["metadata_json"]))
    space = metadata.get("coordinate_space", metadata)
    if space.get("human_coordinate_frame") != "hand_base" or space.get("robot_coordinate_frame") != "hand_base" or space.get("units") != "m":
        raise ValueError("anchors must declare human and robot hand_base coordinates in m")
    contract = json.loads(Path(normalization_path).read_text())
    if contract.get("finger_names") != finger_names:
        raise ValueError("anchor finger ordering differs from normalization contract")
    if contexts.ndim != 3 or contexts.shape[1:] != (len(finger_names), 3) or targets.shape != (len(contexts), 3) or indices.shape != (len(contexts),):
        raise ValueError("invalid raw anchor bundle shapes")
    if np.any(indices < 0) or np.any(indices >= len(finger_names)):
        raise ValueError("anchor finger indices are out of range")
    human = normalize_finger_points(contexts, finger_names, contract["human"])
    robot = np.empty_like(targets)
    for index, finger in enumerate(finger_names):
        rows = indices == index
        if np.any(rows):
            robot[rows] = normalize_finger_points(targets[rows, None], [finger], contract["robot"])[..., 0, :]
    print("Anchor normalized ranges: human min/max", human.min(axis=(0, 1)), human.max(axis=(0, 1)), "robot min/max", robot.min(axis=0), robot.max(axis=0))
    return AnchorTrainingPoints(human, robot, indices)
