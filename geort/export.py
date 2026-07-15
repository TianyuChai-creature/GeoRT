# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pathlib import Path
from geort.formatter import HandFormatter
from geort.keypoint_normalization import normalize_finger_points
from geort.model import IKModel
from geort.utils.path import get_checkpoint_root
from geort.utils.config_utils import load_json, parse_config_keypoint_info, parse_config_joint_limit, select_keypoint_types


def _validate_normalization_contract(contract, keypoint_info):
    if contract.get("schema_version") != 1:
        raise ValueError("Unsupported normalization schema_version")
    if contract.get("keypoint_type") != "tip":
        raise ValueError("Checkpoint normalization must describe tip keypoints")
    expected = {
        "keypoint_names": keypoint_info["name"],
        "keypoint_links": keypoint_info["link"],
        "human_ids": keypoint_info["human_id"],
        "finger_names": keypoint_info["finger"],
    }
    for field, values in expected.items():
        if contract.get(field) != values:
            raise ValueError(
                f"Checkpoint normalization {field} does not match config: "
                f"{contract.get(field)!r} != {values!r}"
            )
    if "human" not in contract or "robot" not in contract:
        raise ValueError("Checkpoint normalization is missing human or robot statistics")


def _select_and_normalize_tips(keypoints, human_ids, finger_names, human_stats):
    if keypoints.ndim != 2 or keypoints.shape[1] < 3:
        raise ValueError(f"Expected human keypoints [N, 3], got {keypoints.shape}")
    if max(human_ids) >= keypoints.shape[0]:
        raise ValueError("Checkpoint human_ids exceed the input landmark count")
    selected = keypoints[human_ids, :3].astype("float32", copy=False)
    return normalize_finger_points(selected[None, ...], finger_names, human_stats)[0]


class GeoRTRetargetingModel:
    '''
        Used by external programs.
    '''
    def __init__(self, model_path, config_path, normalization_path=None):
        config = load_json(config_path)
        keypoint_info = select_keypoint_types(
            parse_config_keypoint_info(config), allowed_types=("tip",)
        )
        normalization_path = normalization_path or Path(config_path).with_name("normalization.json")
        normalization = load_json(normalization_path)
        _validate_normalization_contract(normalization, keypoint_info)

        joint_lower_limit, joint_upper_limit = parse_config_joint_limit(config)
        print(keypoint_info["joint"])
        self.human_ids = keypoint_info["human_id"]
        self.finger_names = keypoint_info["finger"]
        self.human_normalization = normalization["human"]
        self.model = IKModel(
            finger_groups=keypoint_info["finger_groups"],
            n_total_joint=len(config["joint_order"]),
        ).cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit)

        # Report FK backend used during training (if metadata is available).
        metadata_path = Path(model_path).parent / "training_metadata.json"
        if metadata_path.exists():
            meta = load_json(metadata_path)
            backend = meta.get("cli_args", {}).get("fk_backend", "unknown")
            print(f"FK backend (from training metadata): {backend}")

    def forward(self, keypoints):
        # keypoints: [N, 3] — raw HTS landmarks in metric space.
        keypoints = _select_and_normalize_tips(
            keypoints, self.human_ids, self.finger_names, self.human_normalization
        )
        joint_normalized = self.model.forward(
            torch.from_numpy(keypoints).unsqueeze(0).float().cuda()
        )
        joint_raw = self.qpos_normalizer.unnormalize(joint_normalized.detach().cpu().numpy())
        return joint_raw[0]


def resolve_checkpoint_dir(tag=''):
    checkpoint_root = Path(get_checkpoint_root())
    requested = Path(tag)
    if requested.is_dir():
        return requested.resolve()

    exact_path = checkpoint_root / requested.name
    if exact_path.is_dir():
        return exact_path.resolve()

    partial_matches = sorted(
        path.name
        for path in checkpoint_root.iterdir()
        if path.is_dir() and str(tag) in path.name
    ) if checkpoint_root.exists() else []
    hint = f" Partial matches: {partial_matches}" if partial_matches else ""
    raise FileNotFoundError(f"No exact checkpoint {tag!r} found in {checkpoint_root}.{hint}")


def load_model(tag='', epoch=0):
    '''
        Loading API.
    '''
    checkpoint_root = resolve_checkpoint_dir(tag)
    if epoch > 0:
        model_path = checkpoint_root / f"epoch_{epoch}.pth"
    else:
        model_path = checkpoint_root / "last.pth"

    config_path = checkpoint_root / "config.json"
    return GeoRTRetargetingModel(model_path=model_path, config_path=config_path)

if __name__ == '__main__':
    # load the model in one line.
    load_model(tag="allegro_last")