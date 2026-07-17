# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from pathlib import Path

import numpy as np
import torch
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
    def __init__(
        self,
        model_path,
        config_path,
        normalization_path=None,
        *,
        contact_refine="off",
        contact_model_path=None,
        contact_p_lo=0.5,
        contact_p_hi=0.8,
        contact_target_dist=0.0,
        contact_lambda=1e-3,
        contact_refine_steps=40,
    ):
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
        if contact_refine not in {"off", "on"}:
            raise ValueError("contact_refine must be 'off' or 'on'")
        self.contact_refiner = None
        self.contact_p_lo = float(contact_p_lo)
        self.contact_p_hi = float(contact_p_hi)
        if not 0.0 <= self.contact_p_lo < self.contact_p_hi <= 1.0:
            raise ValueError("contact trigger thresholds must satisfy 0 <= p_lo < p_hi <= 1")
        if contact_refine == "on":
            if contact_model_path is None:
                raise ValueError("contact_refine='on' requires contact_model_path")
            from geort.contact.runtime import ContactRefiner

            self.contact_refiner = ContactRefiner.load(
                contact_model_path,
                hand_config=config,
                target_distance=float(contact_target_dist),
                regularization=float(contact_lambda),
                steps=int(contact_refine_steps),
            )
            print(
                "contact_refine=on "
                f"model={contact_model_path} p_lo={self.contact_p_lo} p_hi={self.contact_p_hi} "
                f"target_dist={contact_target_dist} lambda={contact_lambda} steps={contact_refine_steps}"
            )
        else:
            print("contact_refine=off")
        self.last_contact_refinement = None
        self.last_normalized_tips = None
        self.last_mapped_qpos = None
        self.last_refined_qpos = None
        self.last_timings_ms = {}

        # Report FK backend used during training (if metadata is available).
        metadata_path = Path(model_path).parent / "training_metadata.json"
        if metadata_path.exists():
            meta = load_json(metadata_path)
            backend = meta.get("cli_args", {}).get("fk_backend", "unknown")
            print(f"FK backend (from training metadata): {backend}")

    def _apply_contact_refinement(self, q_map, keypoints):
        """Preserve the mapper output exactly unless opt-in contact refinement is active."""
        if self.contact_refiner is None:
            return q_map
        result = self.contact_refiner.refine_from_keypoints(
            q_map, keypoints, p_lo=self.contact_p_lo, p_hi=self.contact_p_hi
        )
        self.last_contact_refinement = result
        return result.q_out

    def forward(self, keypoints):
        # keypoints: [N, 3] — raw HTS landmarks in metric hand-base space.
        raw_keypoints = keypoints
        start = time.perf_counter()
        normalized_tips = _select_and_normalize_tips(
            raw_keypoints, self.human_ids, self.finger_names, self.human_normalization
        )
        normalized_done = time.perf_counter()
        joint_normalized = self.model.forward(
            torch.from_numpy(normalized_tips).unsqueeze(0).float().cuda()
        )
        joint_raw = self.qpos_normalizer.unnormalize(joint_normalized.detach().cpu().numpy())
        q_map = joint_raw[0]
        mapped_done = time.perf_counter()
        q_out = self._apply_contact_refinement(q_map, raw_keypoints)
        refined_done = time.perf_counter()
        self.last_normalized_tips = np.asarray(normalized_tips, dtype=np.float32).copy()
        self.last_mapped_qpos = np.asarray(q_map, dtype=np.float32).copy()
        self.last_refined_qpos = np.asarray(q_out, dtype=np.float32).copy()
        self.last_timings_ms = {
            "normalization": (normalized_done - start) * 1000.0,
            "mapping": (mapped_done - normalized_done) * 1000.0,
            "contact": (refined_done - mapped_done) * 1000.0,
            "forward": (refined_done - start) * 1000.0,
        }
        return q_out


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


def load_model(
    tag='', epoch=0, *, contact_refine="off", contact_model_path=None,
    contact_p_lo=0.5, contact_p_hi=0.8, contact_target_dist=0.0,
    contact_lambda=1e-3, contact_refine_steps=40,
):
    '''
        Loading API.
    '''
    checkpoint_root = resolve_checkpoint_dir(tag)
    if epoch > 0:
        model_path = checkpoint_root / f"epoch_{epoch}.pth"
    else:
        model_path = checkpoint_root / "last.pth"

    config_path = checkpoint_root / "config.json"
    return GeoRTRetargetingModel(
        model_path=model_path, config_path=config_path,
        contact_refine=contact_refine, contact_model_path=contact_model_path,
        contact_p_lo=contact_p_lo, contact_p_hi=contact_p_hi,
        contact_target_dist=contact_target_dist, contact_lambda=contact_lambda,
        contact_refine_steps=contact_refine_steps,
    )

if __name__ == '__main__':
    # load the model in one line.
    load_model(tag="allegro_last")