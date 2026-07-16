"""Versioned custom_right non-thumb bending anchors parameterized by TIP arc length.

This is a new production entry point.  It deliberately leaves the legacy mining,
lateral groups, and thumb groups untouched: only index/middle/ring/pinky bending
groups are rebuilt from D1 valid-frame TIP trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from geort.anchor.anchor_spec import RobotFingerJoints, build_thumb_arc_knots, derive_finger_joint_layouts
from geort.anchor.compat import get_joint_limits, make_analytic_tip_callback
from geort.anchor.generate_robot_anchors import (
    FINGER_NAMES,
    PairedAnchors,
    build_paired_anchors,
    load_human_anchor_records,
)
from geort.anchor.human_geometry import FINGER_LANDMARKS, align_hts_to_palm, estimate_finger_angles
from geort.anchor.mining import LEVEL_FRACTIONS, MinedHumanAnchors, select_thumb_arc_medoids
from geort.utils.config_utils import get_config, parse_config_keypoint_info, select_keypoint_types


ARC_VERSION = "arc_bending_v2"
NON_THUMB = tuple(range(1, 5))
_TIP_LANDMARKS = np.array((4, 8, 12, 16, 20), dtype=np.int64)


def build_arc_length_coupled_knots(
    lower: object,
    upper: object,
    joints: RobotFingerJoints,
    tip_fk,
    *,
    dense_count: int = 201,
) -> np.ndarray:
    """Return five coupled bending qpos knots evenly spaced by analytic TIP arc."""
    return build_thumb_arc_knots(lower, upper, joints, tip_fk, dense_count=dense_count)


def _group_start(finger_index: int, anchor_type: str) -> int:
    return (finger_index * 2 + (anchor_type == "bending")) * 5


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as bundle:
        return json.loads(str(bundle["metadata_json"].item()))


def pc1_explained_variance(points: np.ndarray) -> float:
    """Match the PCA explained-variance definition used by thumb arc mining."""
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variances = np.square(singular)
    return float(variances[0] / np.sum(variances))


def _arc_selection(
    raw_frames: np.ndarray,
    aligned: np.ndarray,
    valid: np.ndarray,
    finger_name: str,
):
    landmarks = FINGER_LANDMARKS[finger_name]
    rows = np.flatnonzero(valid)
    return select_thumb_arc_medoids(
        aligned[rows, landmarks[-1]],
        aligned[rows, landmarks].reshape(rows.size, -1),
        rows,
    )


def rebuild_human_non_thumb_bending(
    raw_frames: np.ndarray,
    legacy_path: Path,
) -> tuple[MinedHumanAnchors, dict[str, Any]]:
    """Replace only four non-thumb bending groups; retain every frozen row verbatim."""
    legacy = load_human_anchor_records(legacy_path)
    legacy_metadata = _load_metadata(legacy_path)
    aligned, palm_valid = align_hts_to_palm(raw_frames)
    angles = estimate_finger_angles(aligned)
    values = {
        field: np.array(getattr(legacy, field), copy=True)
        for field in (
            "human_frames", "human_points", "source_indices", "finger_indices", "finger_names",
            "anchor_types", "levels", "trajectory_t", "target_parameters", "observed_parameters",
            "candidate_counts", "support_counts",
        )
    }
    groups = {f"{group['finger']}:{group['anchor_type']}": group for group in legacy_metadata["groups"]}
    pc1: dict[str, float] = {}
    for finger_index in NON_THUMB:
        finger = FINGER_NAMES[finger_index]
        landmarks = FINGER_LANDMARKS[finger]
        valid = palm_valid & angles.valid[:, finger_index] & np.all(np.isfinite(aligned[:, landmarks]), axis=(1, 2))
        selection = _arc_selection(raw_frames, aligned, valid, finger)
        rows = slice(_group_start(finger_index, "bending"), _group_start(finger_index, "bending") + 5)
        source = selection.source_indices.astype(np.int64)
        values["human_frames"][rows] = raw_frames[source]
        values["human_points"][rows] = raw_frames[source, landmarks[-1]]
        values["source_indices"][rows] = source
        values["trajectory_t"][rows] = LEVEL_FRACTIONS
        values["target_parameters"][rows] = LEVEL_FRACTIONS
        values["observed_parameters"][rows] = selection.observed_arc_fractions
        values["candidate_counts"][rows] = int(np.count_nonzero(valid))
        values["support_counts"][rows] = selection.support_counts
        pc1[finger] = selection.explained_variance
        groups[f"{finger}:bending"] = {
            "finger": finger,
            "anchor_type": "bending",
            "levels": list(range(5)),
            "trajectory_t": LEVEL_FRACTIONS.tolist(),
            "target_parameters": LEVEL_FRACTIONS.tolist(),
            "observed_parameters": selection.observed_arc_fractions.astype(float).tolist(),
            "source_indices": source.astype(int).tolist(),
            "candidate_counts": [int(np.count_nonzero(valid))] * 5,
            "support_counts": selection.support_counts.astype(int).tolist(),
            "distribution_parameter": "tip_arc_fraction",
            "selected_percentiles": [],
            "diagnostics": {
                "distribution_parameter": "tip_arc_fraction",
                "distribution_values": selection.distribution_arc_fractions.astype(float).tolist(),
                "selection": selection.to_metadata(),
                "candidate_count": int(np.count_nonzero(valid)),
                "parameterization_deviation": "beta1 -> tip_arc_fraction",
            },
        }
    metadata = {
        "schema_version": 2,
        "generation": ARC_VERSION,
        "human_data_source": "data/hts_right.npy",
        "legacy_human_anchor_source": str(legacy_path),
        "frozen_groups": ["thumb:lateral", "thumb:bending", "index:lateral", "middle:lateral", "ring:lateral", "pinky:lateral"],
        "bending_parameterization": "non-thumb beta1 -> tip_arc_fraction",
        "groups": [groups[f"{finger}:{kind}"] for finger in FINGER_NAMES for kind in ("lateral", "bending")],
        "pc1_explained_variance": pc1,
    }
    return MinedHumanAnchors(group_metadata={}), (values, metadata)


def _materialize_human(values: dict[str, np.ndarray]) -> MinedHumanAnchors:
    return MinedHumanAnchors(group_metadata={}, **values)


def _frozen_sparse_robot_qpos(parity_path: Path) -> np.ndarray:
    with np.load(parity_path, allow_pickle=False) as bundle:
        qpos = np.asarray(bundle["robot_qpos"], dtype=np.float64)
        fingers = np.asarray(bundle["finger_indices"], dtype=np.int64)
        types = np.asarray(bundle["anchor_types"]).astype(str)
        t = np.asarray(bundle["trajectory_t"], dtype=np.float64)
    sparse: list[np.ndarray] = []
    for finger_index in range(5):
        for anchor_type in ("lateral", "bending"):
            mask = (fingers == finger_index) & (types == anchor_type)
            group_qpos, group_t = qpos[mask], t[mask]
            sparse.append(np.asarray([group_qpos[np.argmin(np.abs(group_t - level))] for level in LEVEL_FRACTIONS]))
    return np.concatenate(sparse, axis=0)


def _tip_offsets(config: dict[str, Any]) -> list[list[float]]:
    tips = select_keypoint_types(parse_config_keypoint_info(config), allowed_types=("tip",))
    if tuple(tips["finger"]) != FINGER_NAMES:
        raise ValueError("custom_right TIP config must be thumb-to-pinky")
    return tips["offset"]


def rebuild_robot_non_thumb_bending(
    config: dict[str, Any],
    legacy_parity_path: Path,
) -> tuple[np.ndarray, Any]:
    """Retain frozen robot groups and replace only four bending groups via analytic arcs."""
    lower, upper = get_joint_limits(config)
    callback = make_analytic_tip_callback(config, lower, upper, _tip_offsets(config))
    sparse = _frozen_sparse_robot_qpos(legacy_parity_path)
    for finger_index, joints in enumerate(derive_finger_joint_layouts(config["joint_order"])):
        if finger_index in NON_THUMB:
            rows = slice(_group_start(finger_index, "bending"), _group_start(finger_index, "bending") + 5)
            sparse[rows] = build_arc_length_coupled_knots(
                lower, upper, joints, lambda qpos, index=finger_index: callback(qpos, index)
            )
    return sparse, callback


def write_versioned_outputs(
    *,
    human_values: dict[str, np.ndarray],
    human_metadata: dict[str, Any],
    paired: PairedAnchors,
    human_output: Path,
    paired_output: Path,
) -> None:
    _atomic_npz(human_output, **human_values, metadata_json=np.asarray(json.dumps(human_metadata, sort_keys=True)))
    paired_metadata = {
        "schema_version": 2,
        "generation": ARC_VERSION,
        "human_data_source": human_metadata["human_data_source"],
        "human_anchor_source": str(human_output),
        "coordinate_frame": "hand_base",
        "units": "m",
        "fk_backend": "analytic",
        "paired_count": int(paired.robot_qpos.shape[0]),
        "lateral_count_per_finger": 50,
        "bending_count_per_finger": 100,
    }
    _atomic_npz(
        paired_output,
        human_tip_contexts=paired.human_tip_contexts,
        human_points=paired.human_points,
        robot_points=paired.robot_points,
        robot_qpos=paired.robot_qpos,
        finger_indices=paired.finger_indices,
        finger_names=paired.finger_names,
        anchor_types=paired.anchor_types,
        trajectory_t=paired.trajectory_t,
        source_sparse_indices=paired.source_sparse_indices,
        metadata_json=np.asarray(json.dumps(paired_metadata, sort_keys=True)),
    )


def main(argv: list[str] | None = None) -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/hts_right.npy"))
    parser.add_argument("--legacy-human", type=Path, default=Path("data/anchors_human_right.npz"))
    parser.add_argument("--legacy-parity", type=Path, default=Path("outputs/anchors/parity_qpos.npz"))
    parser.add_argument("--human-output", type=Path, default=Path("data/anchors_human_right_arc_bending_v2.npz"))
    parser.add_argument("--paired-output", type=Path, default=Path("data/anchors_custom_right_arc_bending_v2.npz"))
    args = parser.parse_args(argv)
    if args.human_output.exists() or args.paired_output.exists():
        raise FileExistsError("v2 output exists; refusing to overwrite versioned anchors")
    raw = np.load(args.input, allow_pickle=False).astype(np.float64)
    _, packed = rebuild_human_non_thumb_bending(raw, args.legacy_human)
    human_values, human_metadata = packed
    human = _materialize_human(human_values)
    robot_sparse, callback = rebuild_robot_non_thumb_bending(get_config("custom_right"), args.legacy_parity)
    paired = build_paired_anchors(human, robot_sparse, callback)
    write_versioned_outputs(
        human_values=human_values,
        human_metadata=human_metadata,
        paired=paired,
        human_output=args.human_output,
        paired_output=args.paired_output,
    )
    print(json.dumps({"human": str(args.human_output), "paired": str(args.paired_output), "rows": int(paired.robot_qpos.shape[0])}))
    return args.human_output, args.paired_output


if __name__ == "__main__":
    main()
