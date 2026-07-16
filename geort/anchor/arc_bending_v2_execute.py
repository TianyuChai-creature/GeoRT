"""Fixed executable for versioned custom_right non-thumb arc-bending anchors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geort.anchor.arc_bending_v2 import (
    FINGER_LANDMARKS, FINGER_NAMES, LEVEL_FRACTIONS, NON_THUMB, _group_start,
    _load_metadata, _materialize_human, rebuild_robot_non_thumb_bending,
    write_versioned_outputs,
)
from geort.anchor.generate_robot_anchors import build_paired_anchors, load_human_anchor_records
from geort.anchor.human_geometry import align_hts_to_palm, estimate_finger_angles
from geort.anchor.mining import select_thumb_arc_medoids
from geort.utils.config_utils import get_config


def _select_arc(aligned: np.ndarray, valid: np.ndarray, finger: str):
    landmarks = FINGER_LANDMARKS[finger]
    rows = np.flatnonzero(valid)
    finger_points = aligned[rows][:, landmarks]
    return select_thumb_arc_medoids(
        finger_points[:, -1], finger_points.reshape(rows.size, -1), rows
    )


def rebuild_human(raw: np.ndarray, legacy_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    legacy = load_human_anchor_records(legacy_path)
    legacy_metadata = _load_metadata(legacy_path)
    aligned, palm_valid = align_hts_to_palm(raw)
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
    for index in NON_THUMB:
        finger = FINGER_NAMES[index]
        landmarks = FINGER_LANDMARKS[finger]
        valid = palm_valid & angles.valid[:, index] & np.all(np.isfinite(aligned[:, landmarks]), axis=(1, 2))
        chosen = _select_arc(aligned, valid, finger)
        source = chosen.source_indices.astype(np.int64)
        start = _group_start(index, "bending")
        rows = slice(start, start + 5)
        values["human_frames"][rows] = raw[source]
        values["human_points"][rows] = raw[source, landmarks[-1]]
        values["source_indices"][rows] = source
        values["trajectory_t"][rows] = LEVEL_FRACTIONS
        values["target_parameters"][rows] = LEVEL_FRACTIONS
        values["observed_parameters"][rows] = chosen.observed_arc_fractions
        values["candidate_counts"][rows] = int(valid.sum())
        values["support_counts"][rows] = chosen.support_counts
        pc1[finger] = chosen.explained_variance
        groups[f"{finger}:bending"] = {
            "finger": finger, "anchor_type": "bending", "levels": list(range(5)),
            "trajectory_t": LEVEL_FRACTIONS.tolist(), "target_parameters": LEVEL_FRACTIONS.tolist(),
            "observed_parameters": chosen.observed_arc_fractions.astype(float).tolist(),
            "source_indices": source.astype(int).tolist(), "candidate_counts": [int(valid.sum())] * 5,
            "support_counts": chosen.support_counts.astype(int).tolist(),
            "distribution_parameter": "tip_arc_fraction", "selected_percentiles": [],
            "diagnostics": {
                "distribution_parameter": "tip_arc_fraction",
                "distribution_values": chosen.distribution_arc_fractions.astype(float).tolist(),
                "selection": chosen.to_metadata(), "candidate_count": int(valid.sum()),
                "parameterization_deviation": "beta1 -> tip_arc_fraction",
            },
        }
    return values, {
        "schema_version": 2, "generation": "arc_bending_v2", "human_data_source": "data/hts_right.npy",
        "legacy_human_anchor_source": str(legacy_path),
        "frozen_groups": ["thumb:lateral", "thumb:bending", "index:lateral", "middle:lateral", "ring:lateral", "pinky:lateral"],
        "bending_parameterization": "non-thumb beta1 -> tip_arc_fraction",
        "groups": [groups[f"{finger}:{kind}"] for finger in FINGER_NAMES for kind in ("lateral", "bending")],
        "pc1_explained_variance": pc1,
    }


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
    values, metadata = rebuild_human(np.load(args.input, allow_pickle=False).astype(np.float64), args.legacy_human)
    robot_sparse, callback = rebuild_robot_non_thumb_bending(get_config("custom_right"), args.legacy_parity)
    paired = build_paired_anchors(_materialize_human(values), robot_sparse, callback)
    write_versioned_outputs(human_values=values, human_metadata=metadata, paired=paired, human_output=args.human_output, paired_output=args.paired_output)
    print(f"human={args.human_output}\npaired={args.paired_output}\nrows={paired.robot_qpos.shape[0]}")
    return args.human_output, args.paired_output


if __name__ == "__main__":
    main()
