"""Minimal Ring lateral L2/L3 human-medoid repair with all other rows frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geort.anchor.arc_bending_v2 import FINGER_LANDMARKS, FINGER_NAMES, _atomic_npz, _materialize_human
from geort.anchor.compat import get_joint_limits, make_analytic_tip_callback
from geort.anchor.generate_robot_anchors import build_paired_anchors
from geort.anchor.human_geometry import align_hts_to_palm, estimate_finger_angles
from geort.anchor.lateral_shrink_exact import exact_level_knots
from geort.anchor.mining import _medoid_order, filter_motion_candidates, robust_angle_targets
from geort.anchor.ring_lateral_monotonic import choose_monotonic_pair
from geort.utils.config_utils import get_config, parse_config_keypoint_info, select_keypoint_types


def _tip_offsets(config):
    info = select_keypoint_types(parse_config_keypoint_info(config), allowed_types=("tip",))
    return info["offset"]


def _load_human(path):
    with np.load(path, allow_pickle=False) as bundle:
        values = {key: np.asarray(bundle[key]) for key in bundle.files if key != "metadata_json"}
        metadata = json.loads(str(bundle["metadata_json"].item()))
    return values, metadata


def _ring_reselection(raw, values, metadata):
    index, finger = 3, "ring"
    landmarks = FINGER_LANDMARKS[finger]
    aligned, palm_valid = align_hts_to_palm(raw)
    angles = estimate_finger_angles(aligned)
    valid = palm_valid & angles.valid[:, index] & np.all(np.isfinite(aligned[:, landmarks]), axis=(1, 2))
    candidates = filter_motion_candidates(
        angles.alpha[:, index], angles.beta[:, index], valid, "lateral",
        straight_tol=np.deg2rad(15.0), alpha_zero_tol=np.deg2rad(10.0), coupling_tol=np.deg2rad(20.0),
    )
    candidate_sources = np.flatnonzero(candidates.mask)
    parameter = angles.alpha[candidate_sources, index]
    targets = robust_angle_targets(parameter)
    retained_mask = (parameter >= targets.endpoints[0]) & (parameter <= targets.endpoints[1])
    sources = candidate_sources[retained_mask]
    alpha = parameter[retained_mask]
    descriptor = aligned[sources][:, landmarks].reshape(sources.size, -1)
    tips = aligned[sources, landmarks[-1]]
    centered = tips - tips.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    axis = vectors[0]
    projection = centered @ axis
    if np.dot(projection - projection.mean(), alpha - alpha.mean()) < 0.0:
        axis, projection = -axis, -projection
    groups = {f"{group['finger']}:{group['anchor_type']}": group for group in metadata["groups"]}
    group = groups["ring:lateral"]
    histories = group["diagnostics"]["selection"]["expansion_history"]
    pools, orders, distributions = {}, {}, {}
    for level in (1, 2):
        half_width = float(histories[level][-1]["half_width"])
        local = np.flatnonzero(np.abs(alpha - targets.targets[level]) <= half_width)
        ranked = _medoid_order(local, descriptor, alpha, sources, float(targets.targets[level]))
        pools[level] = local
        orders[level] = ranked
        distributions[level] = projection[local]
    source_to_local = {int(source): local for local, source in enumerate(sources)}
    ring_mask = (values["finger_indices"] == index) & (values["anchor_types"].astype(str) == "lateral")
    old_sources = values["source_indices"][ring_mask]
    fixed_local = [source_to_local[int(old_sources[level])] for level in (0, 3, 4)]
    pair = choose_monotonic_pair(
        level2_order=orders[1], level3_order=orders[2],
        projection={int(local): float(projection[local]) for local in range(sources.size)},
        fixed_projections=tuple(float(projection[local]) for local in fixed_local),
    )
    evidence = {
        "axis": axis.astype(float).tolist(), "targets": targets.targets.astype(float).tolist(),
        "fixed_projections": [float(projection[local]) for local in fixed_local],
        "bands": {str(level + 1): {"half_width": float(histories[level][-1]["half_width"]), "candidate_count": int(pools[level].size), "projection_min": float(distributions[level].min()), "projection_max": float(distributions[level].max())} for level in (1, 2)},
    }
    if pair is None:
        return None, evidence
    start = np.flatnonzero(ring_mask)[0]
    new_sources = sources[np.asarray(pair, dtype=np.int64)]
    new_alpha = alpha[np.asarray(pair, dtype=np.int64)]
    values = {key: np.array(value, copy=True) for key, value in values.items()}
    values["human_frames"][start + 1:start + 3] = raw[new_sources]
    values["human_points"][start + 1:start + 3] = raw[new_sources, landmarks[-1]]
    values["source_indices"][start + 1:start + 3] = new_sources
    values["observed_parameters"][start + 1:start + 3] = new_alpha
    group["source_indices"][1:3] = new_sources.astype(int).tolist()
    group["observed_parameters"][1:3] = new_alpha.astype(float).tolist()
    group["diagnostics"]["monotonic_l2_l3_reselection"] = {
        "constraint": "strict increasing TIP PC1 projection L1<L2<L3<L4<L5",
        "source_indices": new_sources.astype(int).tolist(), "projections": [float(projection[local]) for local in pair],
        "candidate_bands": evidence["bands"],
    }
    metadata = dict(metadata)
    metadata["generation"] = "arc_bending_v2_ringmono"
    metadata["ring_lateral_reselection"] = group["diagnostics"]["monotonic_l2_l3_reselection"]
    return (values, metadata), evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/hts_right.npy"))
    parser.add_argument("--human", type=Path, default=Path("data/anchors_human_right_arc_bending_v2_cpu_exact.npz"))
    parser.add_argument("--paired", type=Path, default=Path("data/anchors_custom_right_arc_bending_v2_lateral085_exactknots.npz"))
    parser.add_argument("--human-output", type=Path, default=Path("data/anchors_human_right_arc_bending_v2_ringmono.npz"))
    parser.add_argument("--paired-output", type=Path, default=Path("data/anchors_custom_right_arc_bending_v2_lateral085_ringmono.npz"))
    args = parser.parse_args(argv)
    if args.human_output.exists() or args.paired_output.exists():
        raise FileExistsError("refusing to overwrite versioned outputs")
    values, metadata = _load_human(args.human)
    result, evidence = _ring_reselection(np.load(args.input, allow_pickle=False).astype(np.float64), values, metadata)
    if result is None:
        print(json.dumps({"status": "no_monotonic_pair", "evidence": evidence}, indent=2, sort_keys=True))
        return None
    values, metadata = result
    with np.load(args.paired, allow_pickle=False) as bundle:
        old = {key: np.asarray(bundle[key]) for key in bundle.files if key != "metadata_json"}
    sparse = []
    for finger_index in range(5):
        for kind in ("lateral", "bending"):
            mask = (old["finger_indices"] == finger_index) & (old["anchor_types"].astype(str) == kind)
            sparse.append(exact_level_knots(old["robot_qpos"][mask], old["trajectory_t"][mask]))
    config = get_config("custom_right"); lower, upper = get_joint_limits(config)
    callback = make_analytic_tip_callback(config, lower, upper, _tip_offsets(config))
    paired = build_paired_anchors(_materialize_human(values), np.concatenate(sparse), callback)
    _atomic_npz(args.human_output, **values, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    pmeta = {"schema_version": 4, "generation": "arc_bending_v2_lateral085_ringmono", "parent_bundle": str(args.paired), "human_anchor_source": str(args.human_output), "human_data_source": "data/hts_right.npy", "coordinate_frame": "hand_base", "units": "m", "fk_backend": "analytic", "ring_lateral_reselection": metadata["ring_lateral_reselection"], "paired_count": 750, "lateral_count_per_finger": 50, "bending_count_per_finger": 100}
    _atomic_npz(args.paired_output, human_tip_contexts=paired.human_tip_contexts, human_points=paired.human_points, robot_points=paired.robot_points, robot_qpos=paired.robot_qpos, finger_indices=paired.finger_indices, finger_names=paired.finger_names, anchor_types=paired.anchor_types, trajectory_t=paired.trajectory_t, source_sparse_indices=paired.source_sparse_indices, metadata_json=np.asarray(json.dumps(pmeta, sort_keys=True)))
    print(json.dumps({"status": "written", "human": str(args.human_output), "paired": str(args.paired_output), "evidence": evidence}, indent=2, sort_keys=True))
    return args.human_output, args.paired_output


if __name__ == "__main__":
    main()
