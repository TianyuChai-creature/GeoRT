"""CPU-only analytic/SAPIEN parity gate for sparse-anchor robot TIPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geort.anchor.anchor_spec import derive_finger_joint_layouts
from geort.anchor.compat import get_joint_limits, make_analytic_tip_callback
from geort.anchor.generate_robot_anchors import (
    FINGER_NAMES,
    build_paired_anchors,
    build_robot_sparse_knots,
    load_human_anchor_records,
)
from geort.utils.config_utils import get_config, parse_config_keypoint_info, select_keypoint_types


def _sapien_callback(config: dict):
    from geort.env.hand import HandKinematicModel

    tips = select_keypoint_types(parse_config_keypoint_info(config), allowed_types=("tip",))
    if tuple(tips["finger"]) != FINGER_NAMES:
        raise ValueError("TIP config must be thumb-to-pinky")
    hand = HandKinematicModel.build_from_config(config, render=False)
    hand.initialize_keypoint(tips["link"], tips["offset"])

    def evaluate(qpos: np.ndarray, finger_index: int) -> np.ndarray:
        return np.asarray(hand.keypoint_from_qpos(qpos, ret_vec=True)[finger_index], dtype=np.float64)

    return evaluate, tips["offset"]


def write_parity_qpos(path: Path, human_path: Path, config: dict) -> Path:
    """Create the canonical 750-row physical-qpos input exactly once."""
    lower, upper = get_joint_limits(config)
    sapien, offsets = _sapien_callback(config)
    human = load_human_anchor_records(human_path)
    knots = build_robot_sparse_knots(
        lower, upper, derive_finger_joint_layouts(config["joint_order"]), sapien
    )
    paired = build_paired_anchors(human, knots, sapien)
    np.savez_compressed(
        path,
        robot_qpos=paired.robot_qpos,
        finger_indices=paired.finger_indices,
        anchor_types=paired.anchor_types,
        trajectory_t=paired.trajectory_t,
        tip_offsets=np.asarray(offsets, dtype=np.float64),
        metadata_json=np.asarray(json.dumps({"coordinate_frame": "hand_base", "units": "m"})),
    )
    if paired.robot_qpos.shape[0] != 750:
        raise RuntimeError("parity input must contain 750 robot qpos rows")
    return path


def compare_parity_qpos(path: Path, config: dict, *, threshold_m: float = 1e-3) -> dict:
    """Read one saved qpos bundle through analytic and SAPIEN TIP backends."""
    with np.load(path, allow_pickle=False) as bundle:
        qpos = np.asarray(bundle["robot_qpos"], dtype=np.float64)
        fingers = np.asarray(bundle["finger_indices"], dtype=np.int64)
        offsets = np.asarray(bundle["tip_offsets"], dtype=np.float64)
    lower, upper = get_joint_limits(config)
    sapien, config_offsets = _sapien_callback(config)
    if not np.array_equal(offsets, np.asarray(config_offsets, dtype=np.float64)):
        raise RuntimeError("parity file offsets differ from current config offsets")
    analytic = make_analytic_tip_callback(config, lower, upper, config_offsets)
    sapien_points = np.asarray([sapien(q, int(f)) for q, f in zip(qpos, fingers)])
    analytic_points = np.asarray([analytic(q, int(f)) for q, f in zip(qpos, fingers)])
    errors = np.linalg.norm(analytic_points - sapien_points, axis=1)
    report = {"threshold_m": threshold_m, "overall": {"max_m": float(errors.max()), "mean_m": float(errors.mean())}, "fingers": {}}
    for index, name in enumerate(FINGER_NAMES):
        values = errors[fingers == index]
        report["fingers"][name] = {"max_m": float(values.max()), "mean_m": float(values.mean())}
    if report["overall"]["max_m"] >= threshold_m:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="custom_right")
    parser.add_argument("--human-anchors", type=Path, required=True)
    parser.add_argument("--parity-qpos", type=Path, default=Path("outputs/anchors/parity_qpos.npz"))
    parser.add_argument("--report", type=Path, default=Path("outputs/anchors/custom_right_fk_parity.json"))
    args = parser.parse_args(argv)
    args.parity_qpos.parent.mkdir(parents=True, exist_ok=True)
    write_parity_qpos(args.parity_qpos, args.human_anchors, get_config(args.hand))
    report = compare_parity_qpos(args.parity_qpos, get_config(args.hand))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
