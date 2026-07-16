"""Rebuild only four robot laterals using exact sparse qpos levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geort.anchor.arc_bending_v2 import FINGER_NAMES, _atomic_npz
from geort.anchor.compat import get_joint_limits, make_analytic_tip_callback
from geort.anchor.generate_robot_anchors import build_paired_anchors, load_human_anchor_records
from geort.anchor.lateral_shrink import scale_knots_to_target_ratio
from geort.anchor.lateral_shrink_exact import exact_level_knots
from geort.utils.config_utils import get_config, parse_config_keypoint_info, select_keypoint_types


def _tip_offsets(config):
    info = select_keypoint_types(parse_config_keypoint_info(config), allowed_types=("tip",))
    if tuple(info["finger"]) != FINGER_NAMES:
        raise ValueError("TIP config must be thumb-to-pinky")
    return info["offset"]


def _normalise(points, stats):
    return (points - np.asarray(stats["center"], dtype=np.float64)) / float(stats["scale"])


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path, default=Path("data/anchors_human_right_arc_bending_v2_cpu_exact.npz"))
    parser.add_argument("--paired", type=Path, default=Path("data/anchors_custom_right_arc_bending_v2_cpu_exact.npz"))
    parser.add_argument("--normalization", type=Path, default=Path("checkpoint/custom_right_last/normalization.json"))
    parser.add_argument("--output", type=Path, default=Path("data/anchors_custom_right_arc_bending_v2_lateral085_exactknots.npz"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite versioned bundle: {args.output}")
    contract = json.loads(args.normalization.read_text())
    human = load_human_anchor_records(args.human)
    with np.load(args.paired, allow_pickle=False) as bundle:
        old = {key: np.asarray(bundle[key]) for key in bundle.files if key != "metadata_json"}
    config = get_config("custom_right")
    lower, upper = get_joint_limits(config)
    callback = make_analytic_tip_callback(config, lower, upper, _tip_offsets(config))
    sparse, multipliers, old_ratios = [], {}, {}
    for finger_index, finger in enumerate(FINGER_NAMES):
        for kind in ("lateral", "bending"):
            mask = (old["finger_indices"] == finger_index) & (old["anchor_types"].astype(str) == kind)
            qknots = exact_level_knots(old["robot_qpos"][mask], old["trajectory_t"][mask])
            if finger_index and kind == "lateral":
                hmask = (human.finger_indices == finger_index) & (human.anchor_types.astype(str) == kind)
                hnorm = _normalise(human.human_points[hmask], contract["human"][finger])
                rtip = np.asarray([callback(q, finger_index) for q in qknots])
                rnorm = _normalise(rtip, contract["robot"][finger])
                ratio = float(np.linalg.norm(np.diff(hnorm, axis=0), axis=1).sum() / np.linalg.norm(np.diff(rnorm, axis=0), axis=1).sum())
                qknots, multiplier = scale_knots_to_target_ratio(qknots, current_ratio=ratio, target_ratio=0.85)
                old_ratios[finger], multipliers[finger] = ratio, multiplier
            sparse.append(qknots)
    paired = build_paired_anchors(human, np.concatenate(sparse, axis=0), callback)
    metadata = {
        "schema_version": 3, "generation": "arc_bending_v2_lateral085_exactknots",
        "parent_bundle": str(args.paired), "human_anchor_source": str(args.human),
        "human_data_source": "data/hts_right.npy", "coordinate_frame": "hand_base", "units": "m",
        "fk_backend": "analytic", "sparse_qpos_recovery": "linear evaluation at t=[0,0.25,0.5,0.75,1]",
        "robot_lateral_target_human_over_robot": 0.85, "robot_lateral_old_ratio": old_ratios,
        "robot_lateral_multipliers": multipliers, "frozen": ["all_human", "thumb_lateral", "all_bending"],
        "paired_count": int(paired.robot_qpos.shape[0]), "lateral_count_per_finger": 50, "bending_count_per_finger": 100,
    }
    _atomic_npz(args.output, human_tip_contexts=paired.human_tip_contexts, human_points=paired.human_points, robot_points=paired.robot_points, robot_qpos=paired.robot_qpos, finger_indices=paired.finger_indices, finger_names=paired.finger_names, anchor_types=paired.anchor_types, trajectory_t=paired.trajectory_t, source_sparse_indices=paired.source_sparse_indices, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps({"output": str(args.output), "multipliers": multipliers, "old_ratios": old_ratios}, sort_keys=True))
    return args.output


if __name__ == "__main__":
    main()
