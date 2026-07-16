"""Build versioned custom_right anchors with robust non-thumb arc domains."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geort.anchor.arc_bending_v2 import (
    FINGER_LANDMARKS, FINGER_NAMES, LEVEL_FRACTIONS, NON_THUMB, _group_start,
    _load_metadata, _materialize_human, rebuild_robot_non_thumb_bending,
    write_versioned_outputs,
)
from geort.anchor.arc_bending_v2_robust import select_robust_arc_medoids
from geort.anchor.generate_robot_anchors import build_paired_anchors, load_human_anchor_records
from geort.anchor.human_geometry import align_hts_to_palm, estimate_finger_angles
from geort.utils.config_utils import get_config


def rebuild_human(raw: np.ndarray, legacy_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    legacy = load_human_anchor_records(legacy_path)
    legacy_metadata = _load_metadata(legacy_path)
    aligned, palm_valid = align_hts_to_palm(raw)
    angles = estimate_finger_angles(aligned)
    values = {name: np.array(getattr(legacy, name), copy=True) for name in (
        "human_frames", "human_points", "source_indices", "finger_indices", "finger_names",
        "anchor_types", "levels", "trajectory_t", "target_parameters", "observed_parameters",
        "candidate_counts", "support_counts",
    )}
    groups = {f"{g['finger']}:{g['anchor_type']}": g for g in legacy_metadata["groups"]}
    pc1: dict[str, float] = {}
    for index in NON_THUMB:
        finger, landmarks = FINGER_NAMES[index], FINGER_LANDMARKS[FINGER_NAMES[index]]
        valid = palm_valid & angles.valid[:, index] & np.all(np.isfinite(aligned[:, landmarks]), axis=(1, 2))
        sources = np.flatnonzero(valid)
        hand = aligned[sources][:, landmarks]
        result = select_robust_arc_medoids(hand[:, -1], hand.reshape(sources.size, -1), sources)
        start, selected = _group_start(index, "bending"), result["source_indices"]
        rows = slice(start, start + 5)
        values["human_frames"][rows] = raw[selected]
        values["human_points"][rows] = raw[selected, landmarks[-1]]
        values["source_indices"][rows] = selected
        values["trajectory_t"][rows] = LEVEL_FRACTIONS
        values["target_parameters"][rows] = LEVEL_FRACTIONS
        values["observed_parameters"][rows] = result["observed_arc_fractions"]
        values["candidate_counts"][rows] = result["candidate_count"]
        values["support_counts"][rows] = result["support_counts"]
        pc1[finger] = result["explained_variance"]
        groups[f"{finger}:bending"] = {
            "finger": finger, "anchor_type": "bending", "levels": list(range(5)),
            "trajectory_t": LEVEL_FRACTIONS.tolist(), "target_parameters": LEVEL_FRACTIONS.tolist(),
            "observed_parameters": result["observed_arc_fractions"].astype(float).tolist(),
            "source_indices": selected.astype(int).tolist(),
            "candidate_counts": [result["candidate_count"]] * 5,
            "support_counts": result["support_counts"].astype(int).tolist(),
            "distribution_parameter": "tip_arc_fraction", "selected_percentiles": [],
            "diagnostics": {
                "distribution_parameter": "tip_arc_fraction",
                "distribution_values": result["distribution_arc_fractions"].astype(float).tolist(),
                "selection": result["selection"], "candidate_count": result["candidate_count"],
                "domain_clip": result["domain_clip"], "endpoint_projection": result["endpoint_projection"],
                "populated_bin_count": result["populated_bin_count"],
                "parameterization_deviation": "beta1 -> tip_arc_fraction",
            },
        }
    return values, {
        "schema_version": 2, "generation": "arc_bending_v2", "human_data_source": "data/hts_right.npy",
        "legacy_human_anchor_source": str(legacy_path),
        "frozen_groups": ["thumb:lateral", "thumb:bending", "index:lateral", "middle:lateral", "ring:lateral", "pinky:lateral"],
        "bending_parameterization": "non-thumb beta1 -> tip_arc_fraction",
        "non_thumb_arc_domain": "PC1 projection P2–P98", "manifold_bins": 64,
        "level_band_fraction": 0.025, "band_factors": [1.0, 2.0, 4.0, 8.0], "min_support": 5,
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
    sparse, tip_fk = rebuild_robot_non_thumb_bending(get_config("custom_right"), args.legacy_parity)
    paired = build_paired_anchors(_materialize_human(values), sparse, tip_fk)
    write_versioned_outputs(human_values=values, human_metadata=metadata, paired=paired, human_output=args.human_output, paired_output=args.paired_output)
    print(f"human={args.human_output}\npaired={args.paired_output}\nrows={paired.robot_qpos.shape[0]}")
    return args.human_output, args.paired_output


if __name__ == "__main__":
    main()
