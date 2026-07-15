# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json 
from geort.utils.path import get_package_root
from pathlib import Path 
import os 
import numpy as np 


def save_json(data, filename):
    """
    Save a Python dictionary to a JSON file.
    
    Parameters:
    - data (dict): The data to be saved.
    - filename (str): Path to the file where data will be saved.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_json(filename):
    """
    Load data from a JSON file into a Python dictionary.
    
    Parameters:
    - filename (str): Path to the JSON file to be loaded.
    
    Returns:
    - dict: The loaded data.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_config(config_name):
    config_root = Path(get_package_root()) / "geort" / "config"
    requested = Path(config_name)
    if requested.suffix == ".json" and requested.is_file():
        return load_json(requested)

    exact_name = requested.name if requested.suffix == ".json" else f"{config_name}.json"
    exact_path = config_root / exact_name
    if exact_path.exists():
        return load_json(exact_path)

    stem = requested.stem if requested.suffix == ".json" else str(config_name)
    partial_matches = sorted(
        path.name
        for path in config_root.glob("*.json")
        if stem in path.stem
    )
    hint = f" Partial matches: {partial_matches}" if partial_matches else ""
    raise FileNotFoundError(f"No exact config {exact_name!r} found in {config_root}.{hint}")

def _infer_keypoint_type(info):
    if "keypoint_type" in info:
        return info["keypoint_type"]

    name = info.get("name", "").lower()
    if "pip" in name:
        return "pip"
    return "tip"


def _infer_finger(info, fallback_idx):
    if "finger" in info:
        return info["finger"]

    name = info.get("name", "").lower()
    for suffix in ("_pip", "_tip"):
        if name.endswith(suffix):
            return name[: -len(suffix)]

    known_fingers = ["thumb", "index", "middle", "ring", "pinky"]
    for finger in known_fingers:
        if finger in name:
            return finger

    return f"finger_{fallback_idx}"


def _build_finger_groups(keypoint_fingers, keypoint_joints, joint_order):
    groups_by_finger = {}
    for keypoint_idx, (finger, joints) in enumerate(zip(keypoint_fingers, keypoint_joints)):
        group = groups_by_finger.setdefault(
            finger,
            {"finger": finger, "keypoint_indices": [], "joint_indices": []},
        )
        group["keypoint_indices"].append(keypoint_idx)

        joint_set = set(group["joint_indices"])
        for joint_idx in joints:
            if joint_idx not in joint_set:
                group["joint_indices"].append(joint_idx)
                joint_set.add(joint_idx)

    # Keep each finger's joint block in user joint_order for stable IK writes.
    for group in groups_by_finger.values():
        group["joint_indices"] = [
            joint_idx for joint_idx in range(len(joint_order))
            if joint_idx in set(group["joint_indices"])
        ]

    return list(groups_by_finger.values())


def _build_pinch_pairs(keypoint_fingers, keypoint_types):
    thumb_tip_indices = [
        idx for idx, (finger, keypoint_type) in enumerate(zip(keypoint_fingers, keypoint_types))
        if finger == "thumb" and keypoint_type == "tip"
    ]
    if not thumb_tip_indices:
        return []

    thumb_tip_idx = thumb_tip_indices[0]
    return [
        (thumb_tip_idx, idx)
        for idx, (finger, keypoint_type) in enumerate(zip(keypoint_fingers, keypoint_types))
        if finger != "thumb" and keypoint_type == "tip"
    ]


def _build_segment_pairs(keypoint_fingers, keypoint_types):
    pairs = []
    for finger in dict.fromkeys(keypoint_fingers):
        pip_indices = [
            idx for idx, (keypoint_finger, keypoint_type) in enumerate(zip(keypoint_fingers, keypoint_types))
            if keypoint_finger == finger and keypoint_type == "pip"
        ]
        tip_indices = [
            idx for idx, (keypoint_finger, keypoint_type) in enumerate(zip(keypoint_fingers, keypoint_types))
            if keypoint_finger == finger and keypoint_type == "tip"
        ]
        if pip_indices and tip_indices:
            pairs.append((pip_indices[0], tip_indices[0]))
    return pairs


def parse_config_keypoint_info(config):
    keypoint_names = []
    keypoint_links = []
    keypoint_offsets = []
    keypoint_joints = []
    keypoint_human_ids = []
    keypoint_fingers = []
    keypoint_types = []
    keypoint_weights = []

    joint_order = config["joint_order"]

    for idx, info in enumerate(config["fingertip_link"]):
        keypoint_name = info.get("name", info["link"])
        keypoint_type = _infer_keypoint_type(info)
        keypoint_finger = _infer_finger(info, idx)
        keypoint_weight = info.get(
            "loss_weight",
            0.25 if keypoint_type == "pip" else 1.0,
        )

        keypoint_names.append(keypoint_name)
        keypoint_links.append(info["link"])
        keypoint_offsets.append(info['center_offset'])
        keypoint_human_ids.append(info['human_hand_id'])
        keypoint_fingers.append(keypoint_finger)
        keypoint_types.append(keypoint_type)
        keypoint_weights.append(float(keypoint_weight))
        
        keypoint_joint = []
        for joint in info["joint"]:
            keypoint_joint.append(joint_order.index(joint))

        keypoint_joints.append(keypoint_joint)

    tip_indices = [idx for idx, keypoint_type in enumerate(keypoint_types) if keypoint_type == "tip"]
    pip_indices = [idx for idx, keypoint_type in enumerate(keypoint_types) if keypoint_type == "pip"]

    out = {
        "name": keypoint_names,
        "link": keypoint_links,
        "offset": keypoint_offsets,
        "joint": keypoint_joints,
        "human_id": keypoint_human_ids,
        "finger": keypoint_fingers,
        "type": keypoint_types,
        "weight": keypoint_weights,
        "tip_indices": tip_indices,
        "pip_indices": pip_indices,
        "finger_groups": _build_finger_groups(keypoint_fingers, keypoint_joints, joint_order),
        "pinch_pairs": _build_pinch_pairs(keypoint_fingers, keypoint_types),
        "segment_pairs": _build_segment_pairs(keypoint_fingers, keypoint_types),
    }
    return out 


def select_keypoint_types(keypoint_info, allowed_types=("tip",)):
    """Return a self-consistent keypoint view containing only selected types."""
    allowed_types = set(allowed_types)
    selected = [
        idx for idx, keypoint_type in enumerate(keypoint_info["type"])
        if keypoint_type in allowed_types
    ]
    if not selected:
        raise ValueError(f"No keypoints match types {sorted(allowed_types)}")

    list_fields = ("name", "link", "offset", "joint", "human_id", "finger", "type", "weight")
    out = {
        field: [keypoint_info[field][idx] for idx in selected]
        for field in list_fields
    }
    out["source_indices"] = selected
    out["tip_indices"] = [idx for idx, value in enumerate(out["type"]) if value == "tip"]
    out["pip_indices"] = [idx for idx, value in enumerate(out["type"]) if value == "pip"]

    max_joint_idx = max(joint_idx for joints in out["joint"] for joint_idx in joints)
    joint_order_proxy = range(max_joint_idx + 1)
    out["finger_groups"] = _build_finger_groups(out["finger"], out["joint"], joint_order_proxy)
    out["pinch_pairs"] = _build_pinch_pairs(out["finger"], out["type"])
    out["segment_pairs"] = _build_segment_pairs(out["finger"], out["type"])
    return out


def parse_config_joint_limit(config):
    lower_limit = config["joint"]["lower"]
    upper_limit = config["joint"]["upper"]
    return np.array(lower_limit), np.array(upper_limit)
