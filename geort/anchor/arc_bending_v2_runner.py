"""Executable v2 runner kept separate from immutable migrated anchor modules."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geort.anchor.arc_bending_v2 import (
    FINGER_LANDMARKS,
    FINGER_NAMES,
    LEVEL_FRACTIONS,
    NON_THUMB,
    _arc_selection,
    _group_start,
    _load_metadata,
    _materialize_human,
    rebuild_robot_non_thumb_bending,
    write_versioned_outputs,
)
from geort.anchor.generate_robot_anchors import build_paired_anchors, load_human_anchor_records
from geort.anchor.human_geometry import align_hts_to_palm, estimate_finger_angles
from geort.utils.config_utils import get_config


def rebuild_human_non_thumb_bending_v2(
    raw_frames: np.ndarray, legacy_path: Path
) -> tuple[dict[str, np.ndarray], dict]:
    """Copy frozen legacy groups and replace only non-thumb bending medoids."""
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
        valid = palm_valid & angles.valid[:, finger_index] & np.all(
            np.isfinite(aligned[:, landmarks]), axis=(1, 2)
        )
        selection = _arc_selection(raw_frames, aligned, valid, finger)
        start = _group_start(finger_index, "bending")
        rows = slice(start, start + 5)
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
        "generation": "arc_bending_v2",
        "human_data_source": "data/hts_right.npy",
        "legacy_human_anchor_source": str(legacy_path),
        "frozen_groups": [
            "thumb:lateral", "thumb:bending", "index:lateral", "middle:lateral",
            "ring:lateral", "pinky:lateral",
        ],
        "bending_parameterization": "non-thumb beta1 -> tip_arc_fraction",
        "groups": [groups[f"{finger}:{kind}"] for finger in FINGER_NAMES for kind in ("lateral", "bending")],
        "pc1_explained_variance": pc1,
    }
    return values, metadata


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
    human_values, human_metadata = rebuild_human_non_thumb_bending_v2(raw, args.legacy_human)
    human = _materialize_human(human_values)
    robot_sparse, callback = rebuild_robot_non_thumb_bending(
        get_config("custom_right"), args.legacy_parity
    )
    paired = build_paired_anchors(human, robot_sparse, callback)
    write_versioned_outputs(
        human_values=human_values,
        human_metadata=human_metadata,
        paired=paired,
        human_output=args.human_output,
        paired_output=args.paired_output,
    )
    print(f"human={args.human_output}")
    print(f"paired={args.paired_output}")
    print(f"rows={paired.robot_qpos.shape[0]}")
    return args.human_output, args.paired_output


if __name__ == "__main__":
    main()
