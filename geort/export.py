# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Load GeoRT checkpoints for realtime retargeting."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from geort.formatter import HandFormatter
from geort.model import IKModel
from geort.utils.config_utils import (
    build_tip_finger_groups,
    load_json,
    parse_config_joint_limit,
    parse_config_keypoint_info,
)
from geort.utils.path import get_checkpoint_root


def normalize_selected_human_keypoints(
    keypoints: np.ndarray,
    *,
    human_ids: Sequence[int],
    finger_names: Sequence[str],
    stats: Mapping[str, Mapping[str, object]],
) -> np.ndarray:
    """Select raw landmarks and apply checkpoint TIP normalization."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    selected = keypoints[list(human_ids), :3].astype(np.float64)
    if selected.shape != (len(finger_names), 3):
        raise ValueError(
            f"Expected {len(finger_names)} selected TIP points, got {selected.shape}"
        )
    for index, finger in enumerate(finger_names):
        center = np.asarray(stats[finger]["center"], dtype=np.float64)
        scale = float(stats[finger]["scale"])
        selected[index] = (selected[index] - center) / scale
    return selected.astype(np.float32)


class GeoRTRetargetingModel:
    """Retarget raw mocap landmarks with a saved GeoRT checkpoint."""

    def __init__(self, model_path, config_path, normalization_path=None):
        config = load_json(config_path)
        keypoint_info = parse_config_keypoint_info(config)
        joint_lower_limit, joint_upper_limit = parse_config_joint_limit(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        normalization_file = (
            Path(normalization_path) if normalization_path is not None else None
        )
        if normalization_file is not None and normalization_file.exists():
            self.normalization = load_json(normalization_file)
            self.human_ids = list(self.normalization["human_ids"])
            self.finger_names = list(self.normalization["finger_names"])
            self.human_stats = self.normalization["human"]
            finger_groups = build_tip_finger_groups(keypoint_info)
        else:
            self.normalization = None
            self.human_ids = keypoint_info["human_id"]
            self.finger_names = keypoint_info["finger"]
            self.human_stats = None
            finger_groups = keypoint_info["finger_groups"]

        self.model = IKModel(
            finger_groups=finger_groups,
            n_total_joint=len(config["joint_order"]),
        ).to(self.device)
        try:
            state_dict = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
        except TypeError:  # pragma: no cover - compatibility with older torch.
            state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.qpos_normalizer = HandFormatter(
            joint_lower_limit, joint_upper_limit
        )

    def forward(self, keypoints):
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if self.normalization is None:
            selected = keypoints[self.human_ids, :3]
        else:
            selected = normalize_selected_human_keypoints(
                keypoints,
                human_ids=self.human_ids,
                finger_names=self.finger_names,
                stats=self.human_stats,
            )
        input_tensor = torch.from_numpy(selected).unsqueeze(0).to(self.device)
        with torch.no_grad():
            joint_normalized = self.model(input_tensor)
        joint_raw = self.qpos_normalizer.unnormalize(
            joint_normalized.cpu().numpy()
        )
        return joint_raw[0]


def resolve_checkpoint_dir(tag=""):
    checkpoint_root = Path(get_checkpoint_root())
    requested = Path(tag)
    if requested.is_dir():
        return requested.resolve()

    exact_path = checkpoint_root / requested.name
    if exact_path.is_dir():
        return exact_path.resolve()

    partial_matches = (
        sorted(
            path.name
            for path in checkpoint_root.iterdir()
            if path.is_dir() and str(tag) in path.name
        )
        if checkpoint_root.exists()
        else []
    )
    hint = f" Partial matches: {partial_matches}" if partial_matches else ""
    raise FileNotFoundError(
        f"No exact checkpoint {tag!r} found in {checkpoint_root}.{hint}"
    )


def load_model(tag="", epoch=0):
    """Load a checkpoint using TIP normalization when metadata is available."""
    checkpoint_dir = resolve_checkpoint_dir(tag)
    model_path = (
        checkpoint_dir / f"epoch_{epoch}.pth"
        if epoch > 0
        else checkpoint_dir / "last.pth"
    )
    return GeoRTRetargetingModel(
        model_path=model_path,
        config_path=checkpoint_dir / "config.json",
        normalization_path=checkpoint_dir / "normalization.json",
    )


if __name__ == "__main__":
    load_model(tag="allegro_last")
