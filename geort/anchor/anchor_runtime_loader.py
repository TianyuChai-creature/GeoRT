"""Current-run raw-anchor loader for the finalized hand-base bundle schema."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geort.anchor.training import AnchorTrainingPoints
from geort.keypoint_normalization import normalize_finger_points


def _source(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("human_data_source is missing from anchor or normalization metadata")
    return Path(value).resolve()


def load_anchor_points_for_current_run(anchor_path, normalization_path, finger_names):
    """Load final raw anchors only after validating this run's normalization contract."""
    anchor_path, normalization_path = Path(anchor_path), Path(normalization_path)
    if not normalization_path.is_file():
        raise FileNotFoundError(f"归一化契约尚未写入: expected current-run normalization.json at {normalization_path}")
    with np.load(anchor_path, allow_pickle=False) as bundle:
        contexts = np.asarray(bundle["human_tip_contexts"], dtype=np.float32)
        targets = np.asarray(bundle["robot_points"], dtype=np.float32)
        indices = np.asarray(bundle["finger_indices"], dtype=np.int64)
        metadata = json.loads(str(bundle["metadata_json"].item()))
    contract = json.loads(normalization_path.read_text(encoding="utf-8"))
    if _source(metadata.get("human_data_source")) != _source(contract.get("human_data_source")):
        raise ValueError(f"human_data_source mismatch: anchor={metadata.get('human_data_source')} normalization={contract.get('human_data_source')}")
    if metadata.get("coordinate_frame") != "hand_base" or metadata.get("units") != "m":
        raise ValueError("anchors must declare hand_base coordinates in m")
    if contract.get("finger_names") != finger_names:
        raise ValueError("anchor finger ordering differs from normalization contract")
    if contexts.ndim != 3 or contexts.shape[1:] != (len(finger_names), 3) or targets.shape != (contexts.shape[0], 3) or indices.shape != (contexts.shape[0],):
        raise ValueError("invalid raw anchor bundle shapes")
    if np.any(indices < 0) or np.any(indices >= len(finger_names)):
        raise ValueError("anchor finger indices are out of range")
    human = normalize_finger_points(contexts, finger_names, contract["human"])
    robot = np.empty_like(targets)
    for index, finger in enumerate(finger_names):
        rows = indices == index
        if np.any(rows):
            robot[rows] = normalize_finger_points(targets[rows, None], [finger], contract["robot"])[..., 0, :]
    print("Anchor normalized ranges: human min/max", human.min(axis=(0, 1)), human.max(axis=(0, 1)), "robot min/max", robot.min(axis=0), robot.max(axis=0), flush=True)
    return AnchorTrainingPoints(human, robot, indices)
