# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pathlib import Path
from geort.formatter import HandFormatter
from geort.model import IKModel
from geort.utils.path import get_checkpoint_root
from geort.utils.config_utils import load_json, parse_config_keypoint_info, parse_config_joint_limit


class GeoRTRetargetingModel:
    '''
        Used by external programs.
    '''
    def __init__(self, model_path, config_path):
        config = load_json(config_path)
        keypoint_info = parse_config_keypoint_info(config)
        joint_lower_limit, joint_upper_limit = parse_config_joint_limit(config)
        print(keypoint_info["joint"])
        self.human_ids = keypoint_info["human_id"]
        self.model = IKModel(
            finger_groups=keypoint_info["finger_groups"],
            n_total_joint=len(config["joint_order"]),
        ).cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit) # GeoRT will do normalization.

    def forward(self, keypoints):
        # keypoints: [N, 3]
        keypoints = keypoints[self.human_ids] # extract.
        joint_normalized = self.model.forward(torch.from_numpy(keypoints).unsqueeze(0).reshape(1, -1, 3).float().cuda())
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