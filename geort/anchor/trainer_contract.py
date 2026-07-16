"""Current-run contract checks around the raw sparse-anchor loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geort.anchor.training import AnchorTrainingPoints, load_raw_anchor_training_points


def _resolved_source(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("human_data_source is missing from anchor or normalization metadata")
    return Path(value).resolve()


def load_anchor_points_for_current_run(
    anchor_path: Path | str,
    normalization_path: Path | str,
    finger_names: list[str],
) -> AnchorTrainingPoints:
    """Read anchors only after this run has written its normalization contract."""
    anchor_path = Path(anchor_path)
    normalization_path = Path(normalization_path)
    if not normalization_path.is_file():
        raise FileNotFoundError(
            f"归一化契约尚未写入: expected current-run normalization.json at {normalization_path}"
        )
    with np.load(anchor_path, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata_json"]))
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    anchor_source = _resolved_source(metadata.get("human_data_source"))
    normalization_source = _resolved_source(normalization.get("human_data_source"))
    if anchor_source != normalization_source:
        raise ValueError(
            "human_data_source mismatch: "
            f"anchor={anchor_source} normalization={normalization_source}"
        )
    return load_raw_anchor_training_points(anchor_path, normalization_path, finger_names)
