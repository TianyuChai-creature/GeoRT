"""Pair mined human anchors with independent robot range trajectories."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

import torch
from geort.anchor.anchor_spec import (
    RobotFingerJoints,
    build_lateral_knots,
    build_non_thumb_bending_knots,
    build_thumb_arc_knots,
    derive_finger_joint_layouts,
)
from geort.anchor.interpolate import interpolate_sparse_trajectory
from geort.anchor.mining import MinedHumanAnchors


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_GROUP_TYPES = ("lateral", "bending")


def evaluate_analytic_tip_fk(
    qpos: object,
    config: dict[str, Any],
    *,
    return_link_rotations: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Evaluate physical qpos with the training analytic FK and config offsets."""
    from geort.analytic_fk import AnalyticFK
    from geort.utils.config_utils import parse_config_joint_limit, parse_config_keypoint_info, select_keypoint_types

    values = np.asarray(qpos, dtype=np.float32)
    lower, upper = parse_config_joint_limit(config)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != lower.size:
        raise ValueError("qpos must have shape [N, config DOF]")
    info = select_keypoint_types(parse_config_keypoint_info(config), allowed_types=("tip",))
    if tuple(info["finger"]) != FINGER_NAMES or len(info["offset"]) != 5:
        raise ValueError("config must define thumb-to-pinky five TIP offsets")
    normalised = 2.0 * (values - lower) / (upper - lower) - 1.0
    fk = AnalyticFK(config["urdf_path"], lower, upper, tip_offsets=info["offset"])
    with torch.no_grad():
        result = fk(torch.from_numpy(normalised), return_link_rotations=return_link_rotations)
    if return_link_rotations:
        tips, rotations = result
        return (
            tips.cpu().numpy().astype(np.float64, copy=False),
            rotations.cpu().numpy().astype(np.float64, copy=False),
        )
    return result.cpu().numpy().astype(np.float64, copy=False)


def analytic_link_rotation_callback(config: dict[str, Any]) -> Callable[[np.ndarray, int], np.ndarray]:
    """Return physical-qpos callback for analytic distal-link rotations."""
    def evaluate(qpos: np.ndarray, finger_index: int) -> np.ndarray:
        _, rotations = evaluate_analytic_tip_fk(
            np.asarray(qpos, dtype=np.float32)[None, :], config, return_link_rotations=True
        )
        return rotations[0, finger_index]
    return evaluate


def analytic_tip_callback(config: dict[str, Any]) -> Callable[[np.ndarray, int], np.ndarray]:
    """Return the physical-qpos target-finger callback required by pair generation."""
    def evaluate(qpos: np.ndarray, finger_index: int) -> np.ndarray:
        return evaluate_analytic_tip_fk(np.asarray(qpos, dtype=np.float32)[None, :], config)[0, finger_index]
    return evaluate


@dataclass(frozen=True, slots=True)
class PairedAnchors:
    """Finger-indexed, interpolated human/robot anchor pairs."""

    human_tip_contexts: np.ndarray
    human_points: np.ndarray
    robot_points: np.ndarray
    robot_qpos: np.ndarray
    finger_indices: np.ndarray
    finger_names: np.ndarray
    anchor_types: np.ndarray
    trajectory_t: np.ndarray
    source_sparse_indices: np.ndarray
    robot_link_rotations: np.ndarray | None = None


def _validate_human_anchors(anchors: MinedHumanAnchors) -> None:
    if np.asarray(anchors.human_points).shape != (50, 3):
        raise ValueError("human anchors must contain exactly 50 [3] sparse points")
    if np.asarray(anchors.finger_indices).shape != (50,):
        raise ValueError("human anchor finger_indices must have shape [50]")
    if np.asarray(anchors.anchor_types).shape != (50,):
        raise ValueError("human anchor anchor_types must have shape [50]")
    for group_index, (finger_index, anchor_type) in enumerate(
        (finger_index, anchor_type)
        for finger_index in range(5)
        for anchor_type in _GROUP_TYPES
    ):
        rows = slice(group_index * 5, group_index * 5 + 5)
        if not np.all(anchors.finger_indices[rows] == finger_index):
            raise ValueError("human anchors are not ordered by finger/type/level")
        if not np.all(anchors.anchor_types[rows] == anchor_type):
            raise ValueError("human anchors are not ordered by finger/type/level")


def build_robot_sparse_knots(
    lower: np.ndarray,
    upper: np.ndarray,
    layouts: tuple[RobotFingerJoints, ...],
    exact_tip_fk: Callable[[np.ndarray, int], np.ndarray],
    *,
    thumb_dense_count: int = 201,
) -> np.ndarray:
    """Build 50 robot knots in the same finger/type/level order as D2."""
    if len(layouts) != 5:
        raise ValueError("layouts must contain exactly five fingers")
    groups: list[np.ndarray] = []
    for finger_index, joints in enumerate(layouts):
        groups.append(build_lateral_knots(lower, upper, joints))
        if finger_index == 0:
            groups.append(
                build_thumb_arc_knots(
                    lower,
                    upper,
                    joints,
                    lambda qpos, index=finger_index: exact_tip_fk(qpos, index),
                    dense_count=thumb_dense_count,
                )
            )
        else:
            groups.append(build_non_thumb_bending_knots(lower, upper, joints))
    return np.concatenate(groups, axis=0)


def build_paired_anchors(
    human_anchors: MinedHumanAnchors,
    robot_sparse_qpos: object,
    exact_tip_fk: Callable[[np.ndarray, int], np.ndarray],
    *,
    exact_link_rotation_fk: Callable[[np.ndarray, int], np.ndarray] | None = None,
) -> PairedAnchors:
    """Interpolate same-level groups to 250 lateral + 500 bending anchor pairs."""
    _validate_human_anchors(human_anchors)
    try:
        sparse_qpos = np.asarray(robot_sparse_qpos, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("robot_sparse_qpos must have shape [50, D] and be finite") from error
    if (
        sparse_qpos.ndim != 2
        or sparse_qpos.shape[0] != 50
        or sparse_qpos.shape[1] == 0
        or not np.all(np.isfinite(sparse_qpos))
    ):
        raise ValueError("robot_sparse_qpos must have shape [50, D] and be finite")

    human_parts: list[np.ndarray] = []
    robot_parts: list[np.ndarray] = []
    qpos_parts: list[np.ndarray] = []
    finger_parts: list[np.ndarray] = []
    name_parts: list[np.ndarray] = []
    type_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    context_parts: list[np.ndarray] = []
    rotation_parts: list[np.ndarray] = []
    human_tip_indices = np.array([4, 8, 12, 16, 20], dtype=np.int64)
    for group_index, (finger_index, anchor_type) in enumerate(
        (finger_index, anchor_type)
        for finger_index in range(5)
        for anchor_type in _GROUP_TYPES
    ):
        rows = slice(group_index * 5, group_index * 5 + 5)
        output_count = 50 if anchor_type == "lateral" else 100
        context = interpolate_sparse_trajectory(
            human_anchors.human_frames[rows, human_tip_indices, :].reshape(5, -1),
            output_count,
        )["points"].reshape(output_count, 5, 3)
        robot = interpolate_sparse_trajectory(sparse_qpos[rows], output_count)
        try:
            robot_points = np.asarray(
                [exact_tip_fk(qpos, finger_index) for qpos in robot["points"]],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("exact_tip_fk must return finite [3] points") from error
        if robot_points.shape != (output_count, 3) or not np.all(np.isfinite(robot_points)):
            raise ValueError("exact_tip_fk must return finite [3] points")
        if exact_link_rotation_fk is not None:
            rotations = np.asarray(
                [exact_link_rotation_fk(qpos, finger_index) for qpos in robot["points"]],
                dtype=np.float64,
            )
            if rotations.shape != (output_count, 3, 3) or not np.all(np.isfinite(rotations)):
                raise ValueError("exact_link_rotation_fk must return finite [3,3] rotations")
            rotation_parts.append(rotations)
        human_parts.append(context[:, finger_index, :])
        context_parts.append(context)
        robot_parts.append(robot_points)
        qpos_parts.append(robot["points"])
        finger_parts.append(np.full(output_count, finger_index, dtype=np.int64))
        name_parts.append(np.full(output_count, FINGER_NAMES[finger_index]))
        type_parts.append(np.full(output_count, anchor_type))
        t_parts.append(robot["trajectory_t"])
        source_parts.append(robot["source_sparse_indices"] + group_index * 5)
    paired = PairedAnchors(
        human_tip_contexts=np.concatenate(context_parts, axis=0),
        human_points=np.concatenate(human_parts, axis=0),
        robot_points=np.concatenate(robot_parts, axis=0),
        robot_qpos=np.concatenate(qpos_parts, axis=0),
        finger_indices=np.concatenate(finger_parts, axis=0),
        finger_names=np.concatenate(name_parts, axis=0),
        anchor_types=np.concatenate(type_parts, axis=0),
        trajectory_t=np.concatenate(t_parts, axis=0),
        source_sparse_indices=np.concatenate(source_parts, axis=0),
        robot_link_rotations=(np.concatenate(rotation_parts, axis=0) if rotation_parts else None),
    )
    if paired.human_points.shape[0] != 750:
        raise RuntimeError("paired anchor construction did not produce 750 rows")
    return paired


def load_human_anchor_records(path: Path | str) -> MinedHumanAnchors:
    """Load the D2 sparse human NPZ emitted by ``mine_human_anchors``."""
    required = (
        "human_frames",
        "human_points",
        "source_indices",
        "finger_indices",
        "finger_names",
        "anchor_types",
        "levels",
        "trajectory_t",
        "target_parameters",
        "observed_parameters",
        "candidate_counts",
        "support_counts",
    )
    with np.load(path, allow_pickle=False) as bundle:
        missing = [key for key in required if key not in bundle]
        if missing:
            raise ValueError(f"human anchor bundle is missing fields: {missing}")
        values = {key: np.asarray(bundle[key]) for key in required}
    return MinedHumanAnchors(**values, group_metadata={})


def _atomic_npz(path: Path, paired: PairedAnchors, *, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as output_file:
            np.savez_compressed(
                output_file,
                human_tip_contexts=paired.human_tip_contexts,
                human_points=paired.human_points,
                robot_points=paired.robot_points,
                robot_qpos=paired.robot_qpos,
                finger_indices=paired.finger_indices,
                finger_names=paired.finger_names,
                anchor_types=paired.anchor_types,
                trajectory_t=paired.trajectory_t,
                source_sparse_indices=paired.source_sparse_indices,
                **({"robot_link_rotations": paired.robot_link_rotations} if paired.robot_link_rotations is not None else {}),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _update_manifest(manifest_path: Path, anchor_path: Path, *, count: int) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["anchors"] = {
        "path": os.path.relpath(anchor_path, manifest_path.parent),
        "normalized": False,
        "finger_indexed": True,
        "count": int(count),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", required=True, help="GeoRT robot config name")
    parser.add_argument("--human-anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--thumb-dense-count", type=int, default=201)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    from geort.utils.config_utils import get_config

    config = get_config(args.hand)
    output = args.output or Path("data") / f"anchors_{config['name']}.npz"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    if not args.manifest.is_file():
        raise FileNotFoundError(
            f"prepared manifest does not exist: {args.manifest}; run geort.data.prepare first"
        )
    from geort.env.hand import HandKinematicModel
    from geort.utils.config_utils import parse_config_keypoint_info

    human = load_human_anchor_records(args.human_anchors)
    hand = HandKinematicModel.build_from_config(config, render=False)
    info = parse_config_keypoint_info(config)
    hand.initialize_keypoint(info["link"], info["offset"])
    tip_lookup = {
        info["finger"][keypoint_index]: keypoint_index
        for keypoint_index in info["tip_indices"]
    }
    if tuple(tip_lookup) != FINGER_NAMES:
        raise ValueError("robot config must provide thumb-to-pinky TIP keypoints")

    exact_tip_fk = analytic_tip_callback(config)
    exact_link_rotation_fk = analytic_link_rotation_callback(config)

    lower, upper = hand.get_joint_limit()
    robot_knots = build_robot_sparse_knots(
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        derive_finger_joint_layouts(config["joint_order"]),
        exact_tip_fk,
        thumb_dense_count=args.thumb_dense_count,
    )
    paired = build_paired_anchors(
        human, robot_knots, exact_tip_fk, exact_link_rotation_fk=exact_link_rotation_fk
    )
    _atomic_npz(
        output,
        paired,
        metadata={
            "schema_version": 2,
            "hand": config["name"],
            "human_anchor_source": str(args.human_anchors),
            "sparse_count": 50,
            "paired_count": 750,
            "lateral_count_per_finger": 50,
            "bending_count_per_finger": 100,
        },
    )
    _update_manifest(args.manifest, output, count=750)
    print(f"Wrote {output}")
    print(f"Updated {args.manifest}")
    return output


if __name__ == "__main__":
    main()
