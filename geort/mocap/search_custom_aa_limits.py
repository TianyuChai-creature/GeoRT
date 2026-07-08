"""Read-only search for custom_right four-finger AA joint limit candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from geort.env.hand import HandKinematicModel
from geort.mocap.visualize_tip_workspace import (
    build_workspace_overlap_report,
    extract_dataset_tip_points,
)
from geort.utils.config_utils import get_config, parse_config_keypoint_info
from geort.utils.path import get_human_data, to_package_root

AA_FINGER_NAMES = ("F2", "F3", "F4", "F5")
REFERENCE_AA_LIMIT = (-0.61, 0.61)
REFERENCE_MCP2_LIMIT_BY_FINGER = {
    "F1": (-0.35, 0.61),
    "F2": REFERENCE_AA_LIMIT,
    "F3": REFERENCE_AA_LIMIT,
    "F4": REFERENCE_AA_LIMIT,
    "F5": REFERENCE_AA_LIMIT,
}

RIGHT_MANUAL_CLOSURE_PRIOR_V1 = {
    "F2-R-MCP2": -0.264,
    "F3-R-MCP2": 0.0,
    "F4-R-MCP2": 0.239,
    "F5-R-MCP2": 0.552,
}
LEFT_MANUAL_CLOSURE_PRIOR_V1 = {
    "F2-L-MCP2": -0.264,
    "F3-L-MCP2": 0.0,
    "F4-L-MCP2": 0.239,
    "F5-L-MCP2": 0.552,
}


def aa_joint_names_for_hand(hand: str) -> list[str]:
    hand_name = hand.lower()
    if "left" in hand_name:
        side = "L"
    elif "right" in hand_name:
        side = "R"
    else:
        raise ValueError(f"Cannot infer hand side from {hand!r}; expected name containing left or right")
    return [f"{finger}-{side}-MCP2" for finger in AA_FINGER_NAMES]


def _finger_name_from_mcp2_joint(joint_name: str) -> str:
    parts = joint_name.split("-")
    if len(parts) < 3 or parts[-1] != "MCP2":
        raise ValueError(f"Invalid MCP2 joint name {joint_name!r}")
    return parts[0]


def reference_aa_limits_for_joints(
    joint_names: list[str],
    reference_limit: tuple[float, float] | None = None,
) -> dict[str, tuple[float, float]]:
    if reference_limit is not None:
        lower, upper = reference_limit
        return {joint_name: (float(lower), float(upper)) for joint_name in joint_names}
    limits = {}
    for joint_name in joint_names:
        finger_name = _finger_name_from_mcp2_joint(joint_name)
        lower, upper = REFERENCE_MCP2_LIMIT_BY_FINGER[finger_name]
        limits[joint_name] = (float(lower), float(upper))
    return limits


ADJACENT_PAIR_NAMES = ["index__middle", "middle__ring", "ring__pinky"]
OPPOSITION_PAIR_NAMES = ["thumb__index", "thumb__middle", "thumb__ring", "thumb__pinky"]
OPTIMIZED_PAIR_NAMES = ADJACENT_PAIR_NAMES
OVERLAP_FIELDS = ["iou", "overlap_a_ratio", "overlap_b_ratio"]


def _round_pair(pair: tuple[float, float]) -> list[float]:
    return [round(float(pair[0]), 6), round(float(pair[1]), 6)]


def _round_limit_dict(limits: dict[str, tuple[float, float]]) -> dict[str, list[float]]:
    return {name: _round_pair(limit) for name, limit in limits.items()}


def _search_reference_aa_limit(reference_lower: float, reference_upper: float, min_width: float) -> tuple[float, float]:
    reference = (float(reference_lower), float(reference_upper))
    _limit_bounds(*reference, min_width)
    return reference


def build_limit_comparison(
    current_limits: dict[str, tuple[float, float]],
    candidate_limits: dict[str, tuple[float, float]],
) -> dict[str, dict[str, list[float]]]:
    comparison = {}
    for joint_name in candidate_limits:
        current = current_limits[joint_name]
        candidate = candidate_limits[joint_name]
        comparison[joint_name] = {
            "current": _round_pair(current),
            "candidate": _round_pair(candidate),
            "delta": _round_pair((candidate[0] - current[0], candidate[1] - current[1])),
        }
    return comparison


def _limit_bounds(current_lower: float, current_upper: float, min_width: float) -> tuple[float, float, float, float]:
    max_lower = min(0.0, current_upper - min_width)
    min_upper = max(0.0, current_lower + min_width)
    if max_lower < current_lower or min_upper > current_upper:
        raise ValueError(f"min_width={min_width} is infeasible for limit {(current_lower, current_upper)}")
    return float(current_lower), float(max_lower), float(min_upper), float(current_upper)


def _repair_limit_pair(lower: float, upper: float, reference: tuple[float, float], min_width: float) -> tuple[float, float]:
    ref_lower, ref_upper = reference
    lower = float(np.clip(lower, ref_lower, min(0.0, ref_upper - min_width)))
    upper = float(np.clip(upper, max(0.0, ref_lower + min_width), ref_upper))
    if lower > 0.0:
        lower = 0.0
    if upper < 0.0:
        upper = 0.0
    if upper - lower < min_width:
        center = float(np.clip((lower + upper) / 2.0, ref_lower + min_width / 2.0, ref_upper - min_width / 2.0))
        lower = center - min_width / 2.0
        upper = center + min_width / 2.0
        if lower > 0.0:
            lower = 0.0
            upper = min(ref_upper, min_width)
        if upper < 0.0:
            upper = 0.0
            lower = max(ref_lower, -min_width)
    return float(lower), float(upper)


def _validate_candidate(candidate: dict[str, tuple[float, float]], reference_limits: dict[str, tuple[float, float]], min_width: float) -> None:
    for joint_name, (lower, upper) in candidate.items():
        ref_lower, ref_upper = reference_limits[joint_name]
        if lower > 0.0 or upper < 0.0 or upper - lower < min_width or lower < ref_lower or upper > ref_upper:
            raise ValueError(f"Invalid candidate limit for {joint_name}: {(lower, upper)}")


def generate_aa_limit_candidates(
    reference_limits: dict[str, tuple[float, float]],
    *,
    num_candidates: int,
    min_width: float,
    seed: int,
) -> list[dict[str, tuple[float, float]]]:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    if min_width <= 0:
        raise ValueError("min_width must be positive")

    rng = np.random.default_rng(seed)
    candidates = []
    for _ in range(num_candidates):
        candidate = {}
        for joint_name, (reference_lower, reference_upper) in reference_limits.items():
            lower_min, lower_max, upper_min, upper_max = _limit_bounds(reference_lower, reference_upper, min_width)
            for _attempt in range(100):
                lower = float(rng.uniform(lower_min, lower_max))
                upper = float(rng.uniform(upper_min, upper_max))
                if lower <= 0.0 <= upper and upper - lower >= min_width:
                    candidate[joint_name] = (lower, upper)
                    break
            else:
                candidate[joint_name] = _repair_limit_pair(lower_min, upper_min, (reference_lower, reference_upper), min_width)
        candidates.append(candidate)
    return candidates


def _candidate_to_vector(candidate: dict[str, tuple[float, float]], joint_names: list[str]) -> np.ndarray:
    values = []
    for joint_name in joint_names:
        lower, upper = candidate[joint_name]
        values.extend([lower, upper])
    return np.asarray(values, dtype=np.float64)


def _vector_to_candidate(vector: np.ndarray, reference_limits: dict[str, tuple[float, float]], min_width: float, joint_names: list[str]) -> dict[str, tuple[float, float]]:
    candidate = {}
    idx = 0
    for joint_name in joint_names:
        lower = float(vector[idx])
        upper = float(vector[idx + 1])
        idx += 2
        candidate[joint_name] = _repair_limit_pair(lower, upper, reference_limits[joint_name], min_width)
    _validate_candidate(candidate, reference_limits, min_width)
    return candidate


def generate_lhs_aa_limit_candidates(
    reference_limits: dict[str, tuple[float, float]],
    *,
    joint_names: list[str],
    num_candidates: int,
    min_width: float,
    seed: int,
) -> list[dict]:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    rng = np.random.default_rng(seed)
    dimensions = len(joint_names) * 2
    unit = np.empty((num_candidates, dimensions), dtype=np.float64)
    for dim in range(dimensions):
        strata = (np.arange(num_candidates, dtype=np.float64) + rng.random(num_candidates)) / num_candidates
        rng.shuffle(strata)
        unit[:, dim] = strata

    candidates = []
    for row_idx, row in enumerate(unit):
        candidate = {}
        dim = 0
        for joint_name in joint_names:
            lower_min, lower_max, upper_min, upper_max = _limit_bounds(*reference_limits[joint_name], min_width)
            lower = lower_min + row[dim] * (lower_max - lower_min)
            upper = upper_min + row[dim + 1] * (upper_max - upper_min)
            dim += 2
            candidate[joint_name] = _repair_limit_pair(lower, upper, reference_limits[joint_name], min_width)
        candidates.append({"limits": candidate, "source": "coarse_lhs", "candidate_index": row_idx})
    return candidates


def _limits_from_result(result: dict) -> dict[str, tuple[float, float]]:
    return {
        joint_name: tuple(float(v) for v in comparison["candidate"])
        for joint_name, comparison in result["limit_comparison"].items()
    }


def generate_refined_aa_limit_candidates(
    parent_results: list[dict],
    reference_limits: dict[str, tuple[float, float]],
    *,
    joint_names: list[str],
    num_samples_per_parent: int,
    min_width: float,
    step_size: float,
    round_index: int,
    seed: int,
) -> list[dict]:
    if num_samples_per_parent <= 0 or not parent_results:
        return []
    rng = np.random.default_rng(seed)
    refined = []
    for parent in parent_results:
        parent_limits = _limits_from_result(parent)
        parent_vector = _candidate_to_vector(parent_limits, joint_names)
        for sample_idx in range(num_samples_per_parent):
            delta = rng.normal(loc=0.0, scale=step_size, size=parent_vector.shape)
            candidate = _vector_to_candidate(parent_vector + delta, reference_limits, min_width, joint_names)
            refined.append(
                {
                    "limits": candidate,
                    "source": f"refine_round_{round_index}",
                    "parent_candidate": int(parent.get("candidate_index", -1)),
                    "step_size": float(step_size),
                    "candidate_index": sample_idx,
                }
            )
    return refined


def _p_norm(values: list[float] | np.ndarray, p: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if p <= 0:
        raise ValueError("p_norm must be positive")
    return float(np.mean(np.power(np.abs(values), p)) ** (1.0 / p))


def _split_pair_name(pair_name: str) -> tuple[str, str]:
    parts = pair_name.split("__")
    if len(parts) != 2:
        raise ValueError(f"Invalid pair name {pair_name!r}")
    return parts[0], parts[1]


def _min_reachable_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    points_a = np.asarray(points_a, dtype=np.float32).reshape(-1, 3)
    points_b = np.asarray(points_b, dtype=np.float32).reshape(-1, 3)
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf")
    distances, _ = cKDTree(points_b).query(points_a, k=1)
    return float(np.min(distances))


def manual_closure_prior_for_hand(hand: str, joint_names: list[str], source: str) -> dict[str, float]:
    if source == "none":
        return {}
    if source not in {"auto", "right_manual_v1", "left_manual_v1"}:
        raise ValueError(f"Unsupported manual_closure_prior={source!r}")
    hand_name = hand.lower()
    if source == "right_manual_v1" or "right" in hand_name:
        return {joint: value for joint, value in RIGHT_MANUAL_CLOSURE_PRIOR_V1.items() if joint in joint_names}
    if source == "left_manual_v1" or "left" in hand_name:
        return {joint: value for joint, value in LEFT_MANUAL_CLOSURE_PRIOR_V1.items() if joint in joint_names}
    return {}


def _manual_closure_prior_report(
    candidate_limits: dict[str, tuple[float, float]] | None,
    reference_limits: dict[str, tuple[float, float]] | None,
    manual_closure_prior: dict[str, float] | None,
    *,
    margin: float,
    p_norm: float,
) -> dict:
    margin = float(margin)
    if margin < 0.0:
        raise ValueError("manual_closure_prior_margin must be non-negative")
    empty = {
        "penalty": 0.0,
        "passes": True,
        "skipped": True,
        "margin": margin,
        "joint_metrics": {},
        "worst_joint": None,
    }
    if not candidate_limits or not manual_closure_prior:
        return empty

    scale = max(margin, 1e-9)
    joint_metrics = {}
    violations = []
    for joint_name, target_value in manual_closure_prior.items():
        if joint_name not in candidate_limits:
            continue
        candidate_lower, candidate_upper = candidate_limits[joint_name]
        required_lower = float(target_value) - margin
        required_upper = float(target_value) + margin
        if reference_limits is not None and joint_name in reference_limits:
            reference_lower, reference_upper = reference_limits[joint_name]
            required_lower = max(required_lower, float(reference_lower))
            required_upper = min(required_upper, float(reference_upper))
        lower_violation = max(0.0, float(candidate_lower) - required_lower) / scale
        upper_violation = max(0.0, required_upper - float(candidate_upper)) / scale
        violation = max(lower_violation, upper_violation)
        joint_metrics[joint_name] = {
            "target": float(target_value),
            "required": _round_pair((required_lower, required_upper)),
            "candidate": _round_pair((candidate_lower, candidate_upper)),
            "lower_violation": float(lower_violation),
            "upper_violation": float(upper_violation),
            "violation": float(violation),
            "passes": bool(violation == 0.0),
        }
        violations.append(violation)

    if not joint_metrics:
        return empty
    worst_joint = max(joint_metrics, key=lambda joint: joint_metrics[joint]["violation"])
    penalty = _p_norm(violations, p_norm)
    return {
        "penalty": float(penalty),
        "passes": bool(penalty == 0.0),
        "skipped": False,
        "margin": margin,
        "joint_metrics": joint_metrics,
        "worst_joint": worst_joint,
    }


def compute_limit_regularization(
    candidate_limits: dict[str, tuple[float, float]],
    reference_limits: dict[str, tuple[float, float]],
) -> dict:
    values = []
    per_joint = {}
    for joint_name, candidate in candidate_limits.items():
        reference = reference_limits[joint_name]
        width = max(float(reference[1] - reference[0]), 1e-7)
        lower_delta = float((candidate[0] - reference[0]) / width)
        upper_delta = float((candidate[1] - reference[1]) / width)
        joint_value = float(np.sqrt((lower_delta**2 + upper_delta**2) / 2.0))
        per_joint[joint_name] = {"lower_delta_norm": lower_delta, "upper_delta_norm": upper_delta, "value": joint_value}
        values.extend([lower_delta, upper_delta])
    return {
        "value": float(np.sqrt(np.mean(np.square(values)))) if values else 0.0,
        "per_joint": per_joint,
        "reference_limits": _round_limit_dict(reference_limits),
    }


def _adjacent_iou_hinge_report(
    dataset_overlap: dict[str, dict[str, float]],
    urdf_overlap: dict[str, dict[str, float]],
    *,
    pair_names: list[str],
    iou_tolerance: float,
    iou_floor: float,
    p_norm: float,
) -> dict:
    pair_metrics = {}
    errors = []
    for pair_name in pair_names:
        target_iou = float(dataset_overlap[pair_name]["iou"])
        observed_iou = float(urdf_overlap[pair_name]["iou"])
        allowed_iou = target_iou + float(iou_tolerance)
        excess_iou = max(0.0, observed_iou - allowed_iou)
        scale = max(abs(target_iou), float(iou_floor), 1e-9)
        relative_error = excess_iou / scale
        pair_metrics[pair_name] = {
            "dataset_iou": target_iou,
            "urdf_iou": observed_iou,
            "allowed_iou": allowed_iou,
            "excess_iou": excess_iou,
            "relative_error": relative_error,
            "scale": scale,
        }
        errors.append(relative_error)
    worst_pair = max(pair_metrics, key=lambda pair: pair_metrics[pair]["relative_error"]) if pair_metrics else None
    return {
        "pair_metrics": pair_metrics,
        "p_norm_error": _p_norm(errors, p_norm),
        "mean_error": float(np.mean(errors)) if errors else 0.0,
        "max_error": float(pair_metrics[worst_pair]["relative_error"]) if worst_pair else 0.0,
        "worst_pair": worst_pair,
    }


def _opposition_reach_report(
    urdf_tips: dict[str, np.ndarray],
    *,
    pair_names: list[str],
    contact_threshold: float,
    p_norm: float,
) -> dict:
    pair_metrics = {}
    violations = []
    threshold = float(contact_threshold)
    scale = max(threshold, 1e-9)
    for pair_name in pair_names:
        finger_a, finger_b = _split_pair_name(pair_name)
        min_distance = _min_reachable_distance(
            urdf_tips.get(finger_a, np.empty((0, 3), dtype=np.float32)),
            urdf_tips.get(finger_b, np.empty((0, 3), dtype=np.float32)),
        )
        violation = max(0.0, min_distance - threshold) / scale
        pair_metrics[pair_name] = {
            "min_distance": min_distance,
            "contact_threshold": threshold,
            "violation": violation,
            "passes": bool(violation == 0.0),
        }
        violations.append(violation)
    worst_pair = max(pair_metrics, key=lambda pair: pair_metrics[pair]["violation"]) if pair_metrics else None
    return {
        "pair_metrics": pair_metrics,
        "penalty": _p_norm(violations, p_norm),
        "mean_violation": float(np.mean(violations)) if violations else 0.0,
        "max_violation": float(pair_metrics[worst_pair]["violation"]) if worst_pair else 0.0,
        "worst_pair": worst_pair,
    }


def _adjacent_reach_report(
    urdf_tips: dict[str, np.ndarray],
    *,
    pair_names: list[str],
    contact_threshold: float,
    p_norm: float,
) -> dict:
    pair_metrics = {}
    violations = []
    threshold = float(contact_threshold)
    scale = max(threshold, 1e-9)
    for pair_name in pair_names:
        finger_a, finger_b = _split_pair_name(pair_name)
        min_distance = _min_reachable_distance(
            urdf_tips.get(finger_a, np.empty((0, 3), dtype=np.float32)),
            urdf_tips.get(finger_b, np.empty((0, 3), dtype=np.float32)),
        )
        violation = max(0.0, min_distance - threshold) / scale
        pair_metrics[pair_name] = {
            "min_distance": min_distance,
            "contact_threshold": threshold,
            "violation": violation,
            "passes": bool(violation == 0.0),
        }
        violations.append(violation)
    worst_pair = max(pair_metrics, key=lambda pair: pair_metrics[pair]["violation"]) if pair_metrics else None
    return {
        "pair_metrics": pair_metrics,
        "penalty": _p_norm(violations, p_norm),
        "mean_violation": float(np.mean(violations)) if violations else 0.0,
        "max_violation": float(pair_metrics[worst_pair]["violation"]) if worst_pair else 0.0,
        "worst_pair": worst_pair,
    }


def _closure_mask_from_tip_frames(
    tips: dict[str, np.ndarray],
    *,
    pair_names: list[str],
    contact_threshold: float,
) -> tuple[np.ndarray, dict[str, float], list[str]]:
    valid_pairs = []
    required_fingers = set()
    for pair_name in pair_names:
        finger_a, finger_b = _split_pair_name(pair_name)
        if finger_a in tips and finger_b in tips:
            valid_pairs.append(pair_name)
            required_fingers.update([finger_a, finger_b])
    if not valid_pairs or not required_fingers:
        return np.zeros(0, dtype=bool), {}, []

    frame_count = min(len(np.asarray(tips[finger]).reshape(-1, 3)) for finger in required_fingers)
    if frame_count <= 0:
        return np.zeros(0, dtype=bool), {pair_name: 0.0 for pair_name in valid_pairs}, valid_pairs

    threshold = float(contact_threshold)
    closure_mask = np.ones(frame_count, dtype=bool)
    pair_contact_rates = {}
    for pair_name in valid_pairs:
        finger_a, finger_b = _split_pair_name(pair_name)
        points_a = np.asarray(tips[finger_a], dtype=np.float32).reshape(-1, 3)[:frame_count]
        points_b = np.asarray(tips[finger_b], dtype=np.float32).reshape(-1, 3)[:frame_count]
        distances = np.linalg.norm(points_a - points_b, axis=1)
        pair_contact = distances <= threshold
        closure_mask &= pair_contact
        pair_contact_rates[pair_name] = float(np.mean(pair_contact))
    return closure_mask, pair_contact_rates, valid_pairs


def _closure_coverage_report(
    dataset_tips: dict[str, np.ndarray] | None,
    urdf_closure_tips: dict[str, np.ndarray] | None,
    *,
    pair_names: list[str],
    contact_threshold: float,
    min_coverage_rate: float,
) -> dict:
    threshold = float(contact_threshold)
    min_rate = float(min_coverage_rate)
    if min_rate < 0.0 or min_rate > 1.0:
        raise ValueError("closure_min_coverage_rate must be in [0, 1]")
    if dataset_tips is None or urdf_closure_tips is None:
        return {
            "penalty": 0.0,
            "dataset_frame_count": 0,
            "dataset_total_frames": 0,
            "dataset_coverage_rate": 0.0,
            "urdf_frame_count": 0,
            "urdf_total_frames": 0,
            "urdf_coverage_rate": 0.0,
            "required_urdf_coverage_rate": min_rate,
            "contact_threshold": threshold,
            "pair_names": [],
            "dataset_pair_contact_rates": {},
            "urdf_pair_contact_rates": {},
            "passes": True,
            "skipped": True,
        }

    dataset_mask, dataset_pair_rates, valid_pairs = _closure_mask_from_tip_frames(
        dataset_tips,
        pair_names=pair_names,
        contact_threshold=threshold,
    )
    dataset_total = int(dataset_mask.size)
    dataset_count = int(np.count_nonzero(dataset_mask))
    dataset_rate = float(dataset_count / dataset_total) if dataset_total else 0.0
    if dataset_count == 0:
        return {
            "penalty": 0.0,
            "dataset_frame_count": dataset_count,
            "dataset_total_frames": dataset_total,
            "dataset_coverage_rate": dataset_rate,
            "urdf_frame_count": 0,
            "urdf_total_frames": 0,
            "urdf_coverage_rate": 0.0,
            "required_urdf_coverage_rate": min_rate,
            "contact_threshold": threshold,
            "pair_names": valid_pairs,
            "dataset_pair_contact_rates": dataset_pair_rates,
            "urdf_pair_contact_rates": {},
            "passes": True,
            "skipped": True,
        }

    urdf_mask, urdf_pair_rates, urdf_pairs = _closure_mask_from_tip_frames(
        urdf_closure_tips,
        pair_names=valid_pairs,
        contact_threshold=threshold,
    )
    urdf_total = int(urdf_mask.size)
    urdf_count = int(np.count_nonzero(urdf_mask))
    urdf_rate = float(urdf_count / urdf_total) if urdf_total else 0.0
    required_rate = min_rate
    scale = max(required_rate, 1e-9)
    penalty = max(0.0, required_rate - urdf_rate) / scale
    return {
        "penalty": float(penalty),
        "dataset_frame_count": dataset_count,
        "dataset_total_frames": dataset_total,
        "dataset_coverage_rate": dataset_rate,
        "urdf_frame_count": urdf_count,
        "urdf_total_frames": urdf_total,
        "urdf_coverage_rate": urdf_rate,
        "required_urdf_coverage_rate": required_rate,
        "contact_threshold": threshold,
        "pair_names": urdf_pairs,
        "dataset_pair_contact_rates": dataset_pair_rates,
        "urdf_pair_contact_rates": urdf_pair_rates,
        "passes": bool(penalty == 0.0),
        "skipped": False,
    }


def _adjacent_distance_features_from_tip_frames(
    tips: dict[str, np.ndarray],
    *,
    pair_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    valid_pairs = []
    required_fingers = set()
    for pair_name in pair_names:
        finger_a, finger_b = _split_pair_name(pair_name)
        if finger_a in tips and finger_b in tips:
            valid_pairs.append(pair_name)
            required_fingers.update([finger_a, finger_b])
    if not valid_pairs or not required_fingers:
        return np.empty((0, 0), dtype=np.float32), []

    frame_count = min(len(np.asarray(tips[finger]).reshape(-1, 3)) for finger in required_fingers)
    if frame_count <= 0:
        return np.empty((0, len(valid_pairs)), dtype=np.float32), valid_pairs

    features = []
    for pair_name in valid_pairs:
        finger_a, finger_b = _split_pair_name(pair_name)
        points_a = np.asarray(tips[finger_a], dtype=np.float32).reshape(-1, 3)[:frame_count]
        points_b = np.asarray(tips[finger_b], dtype=np.float32).reshape(-1, 3)[:frame_count]
        features.append(np.linalg.norm(points_a - points_b, axis=1))
    return np.stack(features, axis=1).astype(np.float32), valid_pairs


def _dataset_closure_replay_report(
    dataset_tips: dict[str, np.ndarray] | None,
    urdf_closure_tips: dict[str, np.ndarray] | None,
    *,
    pair_names: list[str],
    contact_threshold: float,
    replay_tolerance: float,
    p_norm: float,
) -> dict:
    threshold = float(contact_threshold)
    tolerance = float(replay_tolerance)
    if tolerance < 0.0:
        raise ValueError("dataset_closure_replay_tolerance must be non-negative")
    empty = {
        "penalty": 0.0,
        "dataset_frame_count": 0,
        "dataset_total_frames": 0,
        "dataset_closure_rate": 0.0,
        "urdf_frame_count": 0,
        "success_rate": 0.0,
        "mean_target_adjacent_distance": 0.0,
        "mean_candidate_adjacent_distance": 0.0,
        "mean_nearest_residual": 0.0,
        "max_nearest_residual": 0.0,
        "contact_threshold": threshold,
        "replay_tolerance": tolerance,
        "pair_names": [],
        "worst_pair": None,
        "skipped": True,
    }
    if dataset_tips is None or urdf_closure_tips is None:
        return empty

    dataset_features, valid_pairs = _adjacent_distance_features_from_tip_frames(dataset_tips, pair_names=pair_names)
    if dataset_features.size == 0:
        return {**empty, "pair_names": valid_pairs}
    dataset_total = int(dataset_features.shape[0])
    closure_mask = np.all(dataset_features <= threshold, axis=1)
    dataset_closure = dataset_features[closure_mask]
    dataset_count = int(dataset_closure.shape[0])
    dataset_rate = float(dataset_count / dataset_total) if dataset_total else 0.0
    if dataset_count == 0:
        return {
            **empty,
            "dataset_total_frames": dataset_total,
            "dataset_closure_rate": dataset_rate,
            "pair_names": valid_pairs,
        }

    urdf_features, urdf_pairs = _adjacent_distance_features_from_tip_frames(urdf_closure_tips, pair_names=valid_pairs)
    if urdf_features.size == 0 or urdf_features.shape[1] != dataset_closure.shape[1]:
        return {
            **empty,
            "penalty": 1.0,
            "dataset_frame_count": dataset_count,
            "dataset_total_frames": dataset_total,
            "dataset_closure_rate": dataset_rate,
            "pair_names": valid_pairs,
            "skipped": False,
        }

    scale = max(threshold, 1e-9)
    tree = cKDTree(urdf_features / scale)
    nearest_distances, nearest_indices = tree.query(dataset_closure / scale, k=1)
    nearest_features = urdf_features[np.asarray(nearest_indices, dtype=np.int64)]
    residuals = np.abs(nearest_features - dataset_closure) / scale
    max_residuals = np.max(residuals, axis=1)
    nearest_max_distances = np.max(nearest_features, axis=1)
    success_mask = (max_residuals <= tolerance) & (nearest_max_distances <= threshold)
    residual_violations = np.maximum(0.0, max_residuals - tolerance) / max(tolerance, 1e-9)
    contact_violations = np.maximum(0.0, nearest_max_distances - threshold) / scale
    per_frame_penalty = np.maximum(residual_violations, contact_violations)
    pair_mean_residuals = np.mean(residuals, axis=0)
    worst_pair_index = int(np.argmax(pair_mean_residuals)) if len(valid_pairs) else -1
    return {
        "penalty": _p_norm(per_frame_penalty, p_norm),
        "dataset_frame_count": dataset_count,
        "dataset_total_frames": dataset_total,
        "dataset_closure_rate": dataset_rate,
        "urdf_frame_count": int(urdf_features.shape[0]),
        "success_rate": float(np.mean(success_mask)) if success_mask.size else 0.0,
        "mean_target_adjacent_distance": float(np.mean(dataset_closure)),
        "mean_candidate_adjacent_distance": float(np.mean(nearest_features)),
        "mean_nearest_residual": float(np.mean(max_residuals)),
        "max_nearest_residual": float(np.max(max_residuals)) if max_residuals.size else 0.0,
        "contact_threshold": threshold,
        "replay_tolerance": tolerance,
        "pair_names": urdf_pairs,
        "per_pair_mean_residual": {pair: float(pair_mean_residuals[idx]) for idx, pair in enumerate(valid_pairs)},
        "worst_pair": valid_pairs[worst_pair_index] if worst_pair_index >= 0 else None,
        "skipped": False,
    }


def score_limit_candidate(
    *,
    dataset_overlap: dict[str, dict[str, float]],
    urdf_overlap: dict[str, dict[str, float]],
    urdf_tips: dict[str, np.ndarray],
    dataset_tips: dict[str, np.ndarray] | None = None,
    urdf_closure_tips: dict[str, np.ndarray] | None = None,
    adjacent_pair_names: list[str],
    opposition_pair_names: list[str],
    candidate_limits: dict[str, tuple[float, float]] | None = None,
    reference_limits: dict[str, tuple[float, float]] | None = None,
    iou_tolerance: float = 0.0,
    iou_floor: float = 0.01,
    p_norm: float = 6.0,
    contact_threshold: float = 0.015,
    reach_penalty_weight: float = 10.0,
    adjacent_contact_threshold: float = 0.02,
    adjacent_reach_penalty_weight: float = 10.0,
    closure_contact_threshold: float = 0.02,
    closure_min_coverage_rate: float = 0.02,
    closure_coverage_weight: float = 0.0,
    dataset_closure_replay_tolerance: float = 0.25,
    dataset_closure_replay_weight: float = 5.0,
    manual_closure_prior: dict[str, float] | None = None,
    manual_closure_prior_margin: float = 0.10,
    manual_closure_prior_weight: float = 10.0,
    regularization_weight: float = 0.02,
) -> dict:
    adjacent = _adjacent_iou_hinge_report(
        dataset_overlap,
        urdf_overlap,
        pair_names=adjacent_pair_names,
        iou_tolerance=iou_tolerance,
        iou_floor=iou_floor,
        p_norm=p_norm,
    )
    adjacent_reach = _adjacent_reach_report(
        urdf_tips,
        pair_names=adjacent_pair_names,
        contact_threshold=adjacent_contact_threshold,
        p_norm=p_norm,
    )
    opposition = _opposition_reach_report(
        urdf_tips,
        pair_names=opposition_pair_names,
        contact_threshold=contact_threshold,
        p_norm=p_norm,
    )
    closure_coverage = _closure_coverage_report(
        dataset_tips,
        urdf_closure_tips,
        pair_names=adjacent_pair_names,
        contact_threshold=closure_contact_threshold,
        min_coverage_rate=closure_min_coverage_rate,
    )
    dataset_closure_replay = _dataset_closure_replay_report(
        dataset_tips,
        urdf_closure_tips,
        pair_names=adjacent_pair_names,
        contact_threshold=closure_contact_threshold,
        replay_tolerance=dataset_closure_replay_tolerance,
        p_norm=p_norm,
    )
    manual_prior = _manual_closure_prior_report(
        candidate_limits,
        reference_limits,
        manual_closure_prior,
        margin=manual_closure_prior_margin,
        p_norm=p_norm,
    )
    regularization = {"value": 0.0, "per_joint": {}, "reference_limits": {}}
    if candidate_limits is not None and reference_limits is not None:
        regularization = compute_limit_regularization(candidate_limits, reference_limits)
    score = (
        adjacent["p_norm_error"]
        + float(adjacent_reach_penalty_weight) * adjacent_reach["penalty"]
        + float(reach_penalty_weight) * opposition["penalty"]
        + float(closure_coverage_weight) * closure_coverage["penalty"]
        + float(dataset_closure_replay_weight) * dataset_closure_replay["penalty"]
        + float(manual_closure_prior_weight) * manual_prior["penalty"]
        + float(regularization_weight) * regularization["value"]
    )
    return {
        "score": float(score),
        "loss": float(score),
        "adjacent": adjacent,
        "adjacent_reach": adjacent_reach,
        "opposition": opposition,
        "closure_coverage": closure_coverage,
        "dataset_closure_replay": dataset_closure_replay,
        "manual_closure_prior": manual_prior,
        "regularization": regularization,
        "weights": {
            "reach_penalty_weight": float(reach_penalty_weight),
            "adjacent_reach_penalty_weight": float(adjacent_reach_penalty_weight),
            "closure_coverage_weight": float(closure_coverage_weight),
            "dataset_closure_replay_weight": float(dataset_closure_replay_weight),
            "manual_closure_prior_weight": float(manual_closure_prior_weight),
            "regularization_weight": float(regularization_weight),
            "p_norm": float(p_norm),
            "iou_tolerance": float(iou_tolerance),
            "iou_floor": float(iou_floor),
            "contact_threshold": float(contact_threshold),
            "adjacent_contact_threshold": float(adjacent_contact_threshold),
            "closure_contact_threshold": float(closure_contact_threshold),
            "closure_min_coverage_rate": float(closure_min_coverage_rate),
            "dataset_closure_replay_tolerance": float(dataset_closure_replay_tolerance),
            "manual_closure_prior_margin": float(manual_closure_prior_margin),
        },
    }


def score_overlap_candidate(
    dataset_overlap: dict[str, dict[str, float]],
    urdf_overlap: dict[str, dict[str, float]],
    *,
    pair_names: list[str],
    field_weights: dict[str, float] | None = None,
) -> float:
    del field_weights
    score = score_limit_candidate(
        dataset_overlap=dataset_overlap,
        urdf_overlap=urdf_overlap,
        urdf_tips={},
        adjacent_pair_names=pair_names,
        opposition_pair_names=[],
        reach_penalty_weight=0.0,
        adjacent_reach_penalty_weight=0.0,
        regularization_weight=0.0,
    )
    return float(score["score"])


def _current_aa_limits_from_hand(hand: HandKinematicModel, joint_names: list[str]) -> dict[str, tuple[float, float]]:
    return {
        joint_name: (
            float(hand.joint_lower_limit[hand.joint_names.index(joint_name)]),
            float(hand.joint_upper_limit[hand.joint_names.index(joint_name)]),
        )
        for joint_name in joint_names
    }


def _sample_urdf_tip_points_with_candidate_limits(
    hand: HandKinematicModel,
    keypoint_info: dict,
    candidate_limits: dict[str, tuple[float, float]],
    *,
    samples_per_finger: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if samples_per_finger <= 0:
        raise ValueError("samples_per_finger must be positive")
    lower = hand.joint_lower_limit.copy()
    upper = hand.joint_upper_limit.copy()
    for joint_name, (candidate_lower, candidate_upper) in candidate_limits.items():
        idx = hand.joint_names.index(joint_name)
        lower[idx] = candidate_lower
        upper[idx] = candidate_upper

    tip_index_by_finger = {}
    for idx, (finger, keypoint_type) in enumerate(zip(keypoint_info["finger"], keypoint_info["type"])):
        if keypoint_type == "tip":
            tip_index_by_finger[finger] = idx
    groups_by_finger = {group["finger"]: group for group in keypoint_info["finger_groups"]}

    rng = np.random.default_rng(seed)
    tips: dict[str, list[np.ndarray]] = {finger: [] for finger in tip_index_by_finger}
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        if finger not in tip_index_by_finger:
            continue
        group = groups_by_finger[finger]
        joint_indices = np.array(group["joint_indices"], dtype=np.int64)
        tip_idx = tip_index_by_finger[finger]
        qpos = np.zeros(hand.get_n_dof(), dtype=np.float32)
        qpos = np.clip(qpos, lower, upper)
        for _ in range(samples_per_finger):
            qpos[joint_indices] = rng.uniform(lower[joint_indices], upper[joint_indices])
            keypoints = hand.keypoint_from_qpos(qpos, ret_vec=True)
            tips[finger].append(keypoints[tip_idx].astype(np.float32))
    return {finger: np.stack(points, axis=0) for finger, points in tips.items()}


def _sample_urdf_closure_tip_points_with_candidate_limits(
    hand: HandKinematicModel,
    keypoint_info: dict,
    candidate_limits: dict[str, tuple[float, float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if samples <= 0:
        raise ValueError("closure samples must be positive")
    lower = hand.joint_lower_limit.copy()
    upper = hand.joint_upper_limit.copy()
    for joint_name, (candidate_lower, candidate_upper) in candidate_limits.items():
        idx = hand.joint_names.index(joint_name)
        lower[idx] = candidate_lower
        upper[idx] = candidate_upper

    tip_index_by_finger = {}
    for idx, (finger, keypoint_type) in enumerate(zip(keypoint_info["finger"], keypoint_info["type"])):
        if keypoint_type == "tip":
            tip_index_by_finger[finger] = idx
    groups_by_finger = {group["finger"]: group for group in keypoint_info["finger_groups"]}
    closure_fingers = [finger for finger in ["index", "middle", "ring", "pinky"] if finger in tip_index_by_finger and finger in groups_by_finger]
    if not closure_fingers:
        return {}

    finger_joint_indices = {
        finger: np.array(groups_by_finger[finger]["joint_indices"], dtype=np.int64)
        for finger in closure_fingers
    }
    aa_joint_indices = [indices[0] for indices in finger_joint_indices.values() if len(indices) > 0]
    max_group_size = max((len(indices) for indices in finger_joint_indices.values()), default=0)
    rng = np.random.default_rng(seed)
    tips: dict[str, list[np.ndarray]] = {finger: [] for finger in closure_fingers}
    qpos = np.zeros(hand.get_n_dof(), dtype=np.float32)

    for _ in range(samples):
        qpos[:] = np.clip(0.0, lower, upper)
        if aa_joint_indices:
            shared_lower = max(float(lower[idx]) for idx in aa_joint_indices)
            shared_upper = min(float(upper[idx]) for idx in aa_joint_indices)
            if shared_lower <= shared_upper:
                aa_center = float(rng.uniform(shared_lower, shared_upper))
            else:
                aa_center = float(np.mean([np.clip(0.0, lower[idx], upper[idx]) for idx in aa_joint_indices]))
            for idx in aa_joint_indices:
                jitter = float(rng.normal(0.0, 0.015))
                qpos[idx] = np.clip(aa_center + jitter, lower[idx], upper[idx])

        for local_joint_offset in range(1, max_group_size):
            shared_alpha = float(rng.uniform(0.0, 1.0))
            for finger, joint_indices in finger_joint_indices.items():
                if local_joint_offset >= len(joint_indices):
                    continue
                idx = joint_indices[local_joint_offset]
                width = float(upper[idx] - lower[idx])
                jitter = float(rng.normal(0.0, 0.03))
                alpha = float(np.clip(shared_alpha + jitter, 0.0, 1.0))
                qpos[idx] = lower[idx] + alpha * width

        keypoints = hand.keypoint_from_qpos(qpos, ret_vec=True)
        for finger in closure_fingers:
            tips[finger].append(keypoints[tip_index_by_finger[finger]].astype(np.float32))
    return {finger: np.stack(points, axis=0) for finger, points in tips.items()}


def evaluate_candidate(
    *,
    hand_model: HandKinematicModel,
    keypoint_info: dict,
    dataset_tips: dict[str, np.ndarray],
    dataset_overlap: dict,
    current_limits: dict[str, tuple[float, float]],
    reference_limits: dict[str, tuple[float, float]],
    candidate_limits: dict[str, tuple[float, float]],
    samples_per_finger: int,
    seed: int,
    overlap_voxel_size: float,
    adjacent_pair_names: list[str],
    opposition_pair_names: list[str],
    iou_tolerance: float,
    iou_floor: float,
    p_norm: float,
    contact_threshold: float,
    reach_penalty_weight: float,
    adjacent_contact_threshold: float,
    adjacent_reach_penalty_weight: float,
    closure_contact_threshold: float,
    closure_min_coverage_rate: float,
    closure_coverage_weight: float,
    dataset_closure_replay_tolerance: float,
    dataset_closure_replay_weight: float,
    manual_closure_prior: dict[str, float] | None,
    manual_closure_prior_margin: float,
    manual_closure_prior_weight: float,
    regularization_weight: float,
    candidate_source: str,
    parent_candidate: int | None = None,
    step_size: float | None = None,
) -> dict:
    urdf_tips = _sample_urdf_tip_points_with_candidate_limits(
        hand_model,
        keypoint_info,
        candidate_limits,
        samples_per_finger=samples_per_finger,
        seed=seed,
    )
    urdf_closure_tips = _sample_urdf_closure_tip_points_with_candidate_limits(
        hand_model,
        keypoint_info,
        candidate_limits,
        samples=samples_per_finger,
        seed=seed + 1_000_000,
    )
    overlap_report = build_workspace_overlap_report(dataset_tips, urdf_tips, voxel_size=overlap_voxel_size)
    score_report = score_limit_candidate(
        dataset_overlap=dataset_overlap["dataset"],
        urdf_overlap=overlap_report["urdf"],
        urdf_tips=urdf_tips,
        dataset_tips=dataset_tips,
        urdf_closure_tips=urdf_closure_tips,
        adjacent_pair_names=adjacent_pair_names,
        opposition_pair_names=opposition_pair_names,
        candidate_limits=candidate_limits,
        reference_limits=reference_limits,
        iou_tolerance=iou_tolerance,
        iou_floor=iou_floor,
        p_norm=p_norm,
        contact_threshold=contact_threshold,
        reach_penalty_weight=reach_penalty_weight,
        adjacent_contact_threshold=adjacent_contact_threshold,
        adjacent_reach_penalty_weight=adjacent_reach_penalty_weight,
        closure_contact_threshold=closure_contact_threshold,
        closure_min_coverage_rate=closure_min_coverage_rate,
        closure_coverage_weight=closure_coverage_weight,
        dataset_closure_replay_tolerance=dataset_closure_replay_tolerance,
        dataset_closure_replay_weight=dataset_closure_replay_weight,
        manual_closure_prior=manual_closure_prior,
        manual_closure_prior_margin=manual_closure_prior_margin,
        manual_closure_prior_weight=manual_closure_prior_weight,
        regularization_weight=regularization_weight,
    )
    candidate_info = {"source": candidate_source}
    if parent_candidate is not None:
        candidate_info["parent_candidate"] = int(parent_candidate)
    if step_size is not None:
        candidate_info["step_size"] = float(step_size)
    return {
        "loss": score_report["score"],
        "score_report": score_report,
        "candidate_info": candidate_info,
        "limit_comparison": build_limit_comparison(reference_limits, candidate_limits),
        "current_urdf_limit_comparison": build_limit_comparison(current_limits, candidate_limits),
        "optimized_pairs": {
            pair_name: {
                "dataset": dataset_overlap["dataset"][pair_name],
                "urdf": overlap_report["urdf"][pair_name],
                "score": score_report["adjacent"]["pair_metrics"].get(pair_name),
            }
            for pair_name in adjacent_pair_names
        },
        "adjacent_reach_constraints": score_report["adjacent_reach"],
        "closure_coverage_constraints": score_report["closure_coverage"],
        "dataset_closure_replay_constraints": score_report["dataset_closure_replay"],
        "manual_closure_prior_constraints": score_report["manual_closure_prior"],
        "opposition_constraints": score_report["opposition"],
    }


def _wrap_random_candidates(candidates: list[dict[str, tuple[float, float]]], source: str) -> list[dict]:
    return [{"limits": candidate, "source": source, "candidate_index": idx} for idx, candidate in enumerate(candidates)]


def _candidate_variable_values(result: dict) -> dict[str, float]:
    values = {}
    for joint_name, comparison in result["limit_comparison"].items():
        lower, upper = comparison["candidate"]
        values[f"{joint_name}.lower"] = float(lower)
        values[f"{joint_name}.upper"] = float(upper)
    return values


def compute_limit_sensitivity(results: list[dict]) -> dict:
    if len(results) < 3:
        return {}
    variables = sorted(_candidate_variable_values(results[0]))
    sensitivity = {}
    metric_getters = {
        "adjacent_iou_error": lambda result, pair: result["score_report"]["adjacent"]["pair_metrics"].get(pair, {}).get("relative_error"),
        "adjacent_reach_violation": lambda result, pair: result["score_report"]["adjacent_reach"]["pair_metrics"].get(pair, {}).get("violation"),
        "opposition_reach_violation": lambda result, pair: result["score_report"]["opposition"]["pair_metrics"].get(pair, {}).get("violation"),
        "closure_coverage_penalty": lambda result, pair: result["score_report"]["closure_coverage"].get("penalty") if pair == "adjacent_closure" else None,
        "dataset_closure_replay_penalty": lambda result, pair: result["score_report"]["dataset_closure_replay"].get("penalty") if pair == "dataset_closure_replay" else None,
        "manual_closure_prior_penalty": lambda result, pair: result["score_report"].get("manual_closure_prior", {}).get("penalty") if pair == "manual_closure_prior" else None,
    }
    all_pairs = set()
    for result in results:
        all_pairs.update(result["score_report"]["adjacent"]["pair_metrics"])
        all_pairs.update(result["score_report"]["adjacent_reach"]["pair_metrics"])
        all_pairs.update(result["score_report"]["opposition"]["pair_metrics"])
        all_pairs.add("adjacent_closure")
        all_pairs.add("dataset_closure_replay")
        all_pairs.add("manual_closure_prior")
    for pair_name in sorted(all_pairs):
        pair_report = {}
        for metric_name, getter in metric_getters.items():
            raw_y = [getter(result, pair_name) for result in results]
            if any(value is None for value in raw_y):
                continue
            y = np.asarray(raw_y, dtype=np.float64)
            finite = np.isfinite(y)
            if finite.sum() < 2 or np.std(y[finite]) <= 1e-12:
                continue
            entries = []
            for variable in variables:
                x = np.asarray([_candidate_variable_values(result)[variable] for result in results], dtype=np.float64)
                finite_xy = finite & np.isfinite(x)
                if finite_xy.sum() < 2 or np.std(x[finite_xy]) <= 1e-12:
                    continue
                corr = float(np.corrcoef(x[finite_xy], y[finite_xy])[0, 1])
                if np.isfinite(corr):
                    entries.append({"variable": variable, "correlation": corr, "abs_correlation": abs(corr)})
            entries.sort(key=lambda item: item["abs_correlation"], reverse=True)
            if entries:
                pair_report[metric_name] = entries[:4]
        if pair_report:
            sensitivity[pair_name] = pair_report
    return sensitivity


def _evaluate_candidate_specs(
    specs: list[dict],
    *,
    start_index: int,
    hand_model: HandKinematicModel,
    keypoint_info: dict,
    dataset_tips: dict[str, np.ndarray],
    dataset_overlap: dict,
    current_limits: dict[str, tuple[float, float]],
    reference_limits: dict[str, tuple[float, float]],
    samples_per_finger: int,
    seed: int,
    overlap_voxel_size: float,
    adjacent_pair_names: list[str],
    opposition_pair_names: list[str],
    iou_tolerance: float,
    iou_floor: float,
    p_norm: float,
    contact_threshold: float,
    reach_penalty_weight: float,
    adjacent_contact_threshold: float,
    adjacent_reach_penalty_weight: float,
    closure_contact_threshold: float,
    closure_min_coverage_rate: float,
    closure_coverage_weight: float,
    dataset_closure_replay_tolerance: float,
    dataset_closure_replay_weight: float,
    manual_closure_prior: dict[str, float] | None,
    manual_closure_prior_margin: float,
    manual_closure_prior_weight: float,
    regularization_weight: float,
) -> list[dict]:
    results = []
    for local_idx, spec in enumerate(specs):
        global_idx = start_index + local_idx
        result = evaluate_candidate(
            hand_model=hand_model,
            keypoint_info=keypoint_info,
            dataset_tips=dataset_tips,
            dataset_overlap=dataset_overlap,
            current_limits=current_limits,
            reference_limits=reference_limits,
            candidate_limits=spec["limits"],
            samples_per_finger=samples_per_finger,
            seed=seed + global_idx,
            overlap_voxel_size=overlap_voxel_size,
            adjacent_pair_names=adjacent_pair_names,
            opposition_pair_names=opposition_pair_names,
            iou_tolerance=iou_tolerance,
            iou_floor=iou_floor,
            p_norm=p_norm,
            contact_threshold=contact_threshold,
            reach_penalty_weight=reach_penalty_weight,
            adjacent_contact_threshold=adjacent_contact_threshold,
            adjacent_reach_penalty_weight=adjacent_reach_penalty_weight,
            closure_contact_threshold=closure_contact_threshold,
            closure_min_coverage_rate=closure_min_coverage_rate,
            closure_coverage_weight=closure_coverage_weight,
            dataset_closure_replay_tolerance=dataset_closure_replay_tolerance,
            dataset_closure_replay_weight=dataset_closure_replay_weight,
            manual_closure_prior=manual_closure_prior,
            manual_closure_prior_margin=manual_closure_prior_margin,
            manual_closure_prior_weight=manual_closure_prior_weight,
            regularization_weight=regularization_weight,
            candidate_source=spec["source"],
            parent_candidate=spec.get("parent_candidate"),
            step_size=spec.get("step_size"),
        )
        result["candidate_index"] = global_idx
        results.append(result)
        adjacent = result["score_report"]["adjacent"]
        adjacent_reach = result["score_report"]["adjacent_reach"]
        closure = result["score_report"]["closure_coverage"]
        replay = result["score_report"]["dataset_closure_replay"]
        manual_prior = result["score_report"]["manual_closure_prior"]
        opposition = result["score_report"]["opposition"]
        print(
            f"candidate {global_idx + 1}: loss={result['loss']:.6f} "
            f"source={spec['source']} adjacent={adjacent['p_norm_error']:.6f} "
            f"adjacent_reach={adjacent_reach['penalty']:.6f} closure={closure['penalty']:.6f} "
            f"replay={replay['penalty']:.6f} replay_success={replay['success_rate']:.3f} "
            f"manual_prior={manual_prior['penalty']:.6f} "
            f"opposition={opposition['penalty']:.6f}"
        )
    return results


def search_aa_limits(
    *,
    hand: str,
    human_data: str,
    num_candidates: int,
    samples_per_finger: int,
    top_k: int,
    min_width: float,
    overlap_voxel_size: float,
    seed: int,
    search_mode: str = "coarse_to_fine",
    refine_rounds: int = 2,
    refine_top_k: int = 5,
    refine_samples_per_parent: int = 8,
    refine_step: float = 0.08,
    refine_step_decay: float = 0.5,
    iou_tolerance: float = 0.0,
    iou_floor: float = 0.01,
    p_norm: float = 6.0,
    contact_threshold: float = 0.015,
    reach_penalty_weight: float = 10.0,
    adjacent_contact_threshold: float = 0.02,
    adjacent_reach_penalty_weight: float = 10.0,
    closure_contact_threshold: float = 0.02,
    closure_min_coverage_rate: float = 0.02,
    closure_coverage_weight: float = 0.0,
    dataset_closure_replay_tolerance: float = 0.25,
    dataset_closure_replay_weight: float = 5.0,
    manual_closure_prior: str = "auto",
    manual_closure_prior_margin: float = 0.10,
    manual_closure_prior_weight: float = 10.0,
    regularization_weight: float = 0.02,
    reference_lower: float | None = None,
    reference_upper: float | None = None,
) -> dict:
    config = get_config(hand)
    keypoint_info = parse_config_keypoint_info(config)
    frames = np.load(get_human_data(human_data))
    dataset_tips = extract_dataset_tip_points(frames, keypoint_info)
    hand_model = HandKinematicModel.build_from_config(config, render=False)
    hand_model.initialize_keypoint(keypoint_link_names=keypoint_info["link"], keypoint_offsets=keypoint_info["offset"])
    aa_joint_names = aa_joint_names_for_hand(hand)
    current_limits = _current_aa_limits_from_hand(hand_model, aa_joint_names)
    search_reference_limit = None
    if reference_lower is not None or reference_upper is not None:
        if reference_lower is None or reference_upper is None:
            raise ValueError("reference_lower and reference_upper must be provided together")
        search_reference_limit = _search_reference_aa_limit(reference_lower, reference_upper, min_width)
    reference_limits = reference_aa_limits_for_joints(aa_joint_names, search_reference_limit)
    manual_prior = manual_closure_prior_for_hand(hand, aa_joint_names, manual_closure_prior)
    dataset_overlap = build_workspace_overlap_report(dataset_tips, dataset_tips, voxel_size=overlap_voxel_size)
    adjacent_pair_names = [pair for pair in ADJACENT_PAIR_NAMES if pair in dataset_overlap["dataset"]]
    opposition_pair_names = [pair for pair in OPPOSITION_PAIR_NAMES if pair in dataset_overlap["dataset"]]

    if search_mode == "random":
        candidate_specs = _wrap_random_candidates(
            generate_aa_limit_candidates(reference_limits, num_candidates=num_candidates, min_width=min_width, seed=seed),
            "coarse_random",
        )
    elif search_mode in {"lhs", "coarse_to_fine"}:
        candidate_specs = generate_lhs_aa_limit_candidates(reference_limits, joint_names=aa_joint_names, num_candidates=num_candidates, min_width=min_width, seed=seed)
    else:
        raise ValueError(f"Unsupported search_mode={search_mode!r}")

    all_results = _evaluate_candidate_specs(
        candidate_specs,
        start_index=0,
        hand_model=hand_model,
        keypoint_info=keypoint_info,
        dataset_tips=dataset_tips,
        dataset_overlap=dataset_overlap,
        current_limits=current_limits,
        reference_limits=reference_limits,
        samples_per_finger=samples_per_finger,
        seed=seed,
        overlap_voxel_size=overlap_voxel_size,
        adjacent_pair_names=adjacent_pair_names,
        opposition_pair_names=opposition_pair_names,
        iou_tolerance=iou_tolerance,
        iou_floor=iou_floor,
        p_norm=p_norm,
        contact_threshold=contact_threshold,
        reach_penalty_weight=reach_penalty_weight,
        adjacent_contact_threshold=adjacent_contact_threshold,
        adjacent_reach_penalty_weight=adjacent_reach_penalty_weight,
        closure_contact_threshold=closure_contact_threshold,
        closure_min_coverage_rate=closure_min_coverage_rate,
        closure_coverage_weight=closure_coverage_weight,
        dataset_closure_replay_tolerance=dataset_closure_replay_tolerance,
        dataset_closure_replay_weight=dataset_closure_replay_weight,
        manual_closure_prior=manual_prior,
        manual_closure_prior_margin=manual_closure_prior_margin,
        manual_closure_prior_weight=manual_closure_prior_weight,
        regularization_weight=regularization_weight,
    )

    if search_mode == "coarse_to_fine":
        for round_index in range(1, refine_rounds + 1):
            all_results.sort(key=lambda item: item["loss"])
            parents = all_results[:refine_top_k]
            step_size = refine_step * (refine_step_decay ** (round_index - 1))
            refined_specs = generate_refined_aa_limit_candidates(
                parents,
                reference_limits,
                joint_names=aa_joint_names,
                num_samples_per_parent=refine_samples_per_parent,
                min_width=min_width,
                step_size=step_size,
                round_index=round_index,
                seed=seed + 1000 * round_index,
            )
            refined_results = _evaluate_candidate_specs(
                refined_specs,
                start_index=len(all_results),
                hand_model=hand_model,
                keypoint_info=keypoint_info,
                dataset_tips=dataset_tips,
                dataset_overlap=dataset_overlap,
                current_limits=current_limits,
                reference_limits=reference_limits,
                samples_per_finger=samples_per_finger,
                seed=seed,
                overlap_voxel_size=overlap_voxel_size,
                adjacent_pair_names=adjacent_pair_names,
                opposition_pair_names=opposition_pair_names,
                iou_tolerance=iou_tolerance,
                iou_floor=iou_floor,
                p_norm=p_norm,
                contact_threshold=contact_threshold,
                reach_penalty_weight=reach_penalty_weight,
                adjacent_contact_threshold=adjacent_contact_threshold,
                adjacent_reach_penalty_weight=adjacent_reach_penalty_weight,
                closure_contact_threshold=closure_contact_threshold,
                closure_min_coverage_rate=closure_min_coverage_rate,
                closure_coverage_weight=closure_coverage_weight,
                dataset_closure_replay_tolerance=dataset_closure_replay_tolerance,
                dataset_closure_replay_weight=dataset_closure_replay_weight,
                manual_closure_prior=manual_prior,
                manual_closure_prior_margin=manual_closure_prior_margin,
                manual_closure_prior_weight=manual_closure_prior_weight,
                regularization_weight=regularization_weight,
            )
            all_results.extend(refined_results)

    sensitivity = compute_limit_sensitivity(all_results)
    all_results.sort(key=lambda item: item["loss"])
    return {
        "hand": hand,
        "human_data": human_data,
        "optimized_joints": aa_joint_names,
        "reference_limits": _round_limit_dict(reference_limits),
        "search_reference_limit": _round_pair(search_reference_limit) if search_reference_limit is not None else None,
        "search_reference_source": "cli_override" if search_reference_limit is not None else "per_finger_original_mcp2_baseline",
        "current_urdf_limits": _round_limit_dict(current_limits),
        "adjacent_iou_pairs": adjacent_pair_names,
        "adjacent_reach_pairs": adjacent_pair_names,
        "opposition_reach_pairs": opposition_pair_names,
        "closure_coverage_pairs": adjacent_pair_names,
        "dataset_closure_replay_pairs": adjacent_pair_names,
        "manual_closure_prior": {joint: float(value) for joint, value in manual_prior.items()},
        "manual_closure_prior_source": manual_closure_prior,
        "num_candidates": int(num_candidates),
        "total_evaluated_candidates": int(len(all_results)),
        "samples_per_finger": int(samples_per_finger),
        "min_width": float(min_width),
        "overlap_voxel_size": float(overlap_voxel_size),
        "seed": int(seed),
        "search": {
            "mode": search_mode,
            "refine_rounds": int(refine_rounds),
            "refine_top_k": int(refine_top_k),
            "refine_samples_per_parent": int(refine_samples_per_parent),
            "refine_step": float(refine_step),
            "refine_step_decay": float(refine_step_decay),
            "candidate_limit_baseline": "reference_limits",
        },
        "scoring": {
            "space": "tip_xyz_only",
            "adjacent_objective": "one_sided_voxel_iou_hinge",
            "adjacent_reach_constraint": "min_adjacent_tip_distance_threshold",
            "opposition_constraint": "min_tip_distance_threshold",
            "closure_coverage_constraint": "simultaneous_adjacent_tip_distance_threshold",
            "dataset_closure_replay_constraint": "nearest_urdf_adjacent_distance_vector_for_dataset_closure_frames",
            "manual_closure_prior_constraint": "candidate_limits_cover_manual_mcp2_closure_pose_with_margin",
            "iou_tolerance": float(iou_tolerance),
            "iou_floor": float(iou_floor),
            "p_norm": float(p_norm),
            "contact_threshold": float(contact_threshold),
            "adjacent_contact_threshold": float(adjacent_contact_threshold),
            "closure_contact_threshold": float(closure_contact_threshold),
            "closure_min_coverage_rate": float(closure_min_coverage_rate),
            "dataset_closure_replay_tolerance": float(dataset_closure_replay_tolerance),
            "manual_closure_prior_margin": float(manual_closure_prior_margin),
            "reach_penalty_weight": float(reach_penalty_weight),
            "adjacent_reach_penalty_weight": float(adjacent_reach_penalty_weight),
            "closure_coverage_weight": float(closure_coverage_weight),
            "dataset_closure_replay_weight": float(dataset_closure_replay_weight),
            "manual_closure_prior_weight": float(manual_closure_prior_weight),
            "regularization_weight": float(regularization_weight),
        },
        "limit_sensitivity": sensitivity,
        "top_candidates": all_results[:top_k],
    }


def save_search_report(report: dict, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _resolve_output(path: str) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = to_package_root(output)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="custom_right")
    parser.add_argument("--human_data", default="hts_right_train")
    parser.add_argument("--num_candidates", type=int, default=100)
    parser.add_argument("--samples_per_finger", type=int, default=2000)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--min_width", type=float, default=0.20)
    parser.add_argument("--overlap_voxel_size", type=float, default=0.005)
    parser.add_argument("--search_mode", choices=["random", "lhs", "coarse_to_fine"], default="coarse_to_fine")
    parser.add_argument("--refine_rounds", type=int, default=2)
    parser.add_argument("--refine_top_k", type=int, default=5)
    parser.add_argument("--refine_samples_per_parent", type=int, default=8)
    parser.add_argument("--refine_step", type=float, default=0.08)
    parser.add_argument("--refine_step_decay", type=float, default=0.5)
    parser.add_argument("--iou_tolerance", type=float, default=0.0)
    parser.add_argument("--iou_floor", type=float, default=0.01)
    parser.add_argument("--p_norm", type=float, default=6.0)
    parser.add_argument("--contact_threshold", type=float, default=0.015)
    parser.add_argument("--reach_penalty_weight", type=float, default=10.0)
    parser.add_argument("--adjacent_contact_threshold", type=float, default=0.02)
    parser.add_argument("--adjacent_reach_penalty_weight", type=float, default=10.0)
    parser.add_argument("--closure_contact_threshold", type=float, default=0.02)
    parser.add_argument("--closure_min_coverage_rate", type=float, default=0.02)
    parser.add_argument("--closure_coverage_weight", type=float, default=0.0)
    parser.add_argument("--dataset_closure_replay_tolerance", type=float, default=0.25)
    parser.add_argument("--dataset_closure_replay_weight", type=float, default=5.0)
    parser.add_argument("--manual_closure_prior", choices=["auto", "none", "right_manual_v1", "left_manual_v1"], default="auto")
    parser.add_argument("--manual_closure_prior_margin", type=float, default=0.10)
    parser.add_argument("--manual_closure_prior_weight", type=float, default=10.0)
    parser.add_argument("--regularization_weight", type=float, default=0.02)
    parser.add_argument("--reference_lower", type=float, default=None)
    parser.add_argument("--reference_upper", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/visualizations/custom_right_aa_limit_search.json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = search_aa_limits(
        hand=args.hand,
        human_data=args.human_data,
        num_candidates=args.num_candidates,
        samples_per_finger=args.samples_per_finger,
        top_k=args.top_k,
        min_width=args.min_width,
        overlap_voxel_size=args.overlap_voxel_size,
        seed=args.seed,
        search_mode=args.search_mode,
        refine_rounds=args.refine_rounds,
        refine_top_k=args.refine_top_k,
        refine_samples_per_parent=args.refine_samples_per_parent,
        refine_step=args.refine_step,
        refine_step_decay=args.refine_step_decay,
        iou_tolerance=args.iou_tolerance,
        iou_floor=args.iou_floor,
        p_norm=args.p_norm,
        contact_threshold=args.contact_threshold,
        reach_penalty_weight=args.reach_penalty_weight,
        adjacent_contact_threshold=args.adjacent_contact_threshold,
        adjacent_reach_penalty_weight=args.adjacent_reach_penalty_weight,
        closure_contact_threshold=args.closure_contact_threshold,
        closure_min_coverage_rate=args.closure_min_coverage_rate,
        closure_coverage_weight=args.closure_coverage_weight,
        dataset_closure_replay_tolerance=args.dataset_closure_replay_tolerance,
        dataset_closure_replay_weight=args.dataset_closure_replay_weight,
        manual_closure_prior=args.manual_closure_prior,
        manual_closure_prior_margin=args.manual_closure_prior_margin,
        manual_closure_prior_weight=args.manual_closure_prior_weight,
        regularization_weight=args.regularization_weight,
        reference_lower=args.reference_lower,
        reference_upper=args.reference_upper,
    )
    output = save_search_report(report, _resolve_output(args.output))
    print(f"AA limit search report saved to {output}")
    for rank, candidate in enumerate(report["top_candidates"], start=1):
        score_report = candidate["score_report"]
        print(
            f"rank {rank}: loss={candidate['loss']:.6f} source_candidate={candidate['candidate_index']} "
            f"source={candidate['candidate_info']['source']} "
            f"worst_adjacent={score_report['adjacent']['worst_pair']} "
            f"worst_adjacent_reach={score_report['adjacent_reach']['worst_pair']} "
            f"closure_penalty={score_report['closure_coverage']['penalty']:.6f} "
            f"replay_penalty={score_report['dataset_closure_replay']['penalty']:.6f} "
            f"replay_success={score_report['dataset_closure_replay']['success_rate']:.3f} "
            f"worst_replay={score_report['dataset_closure_replay']['worst_pair']} "
            f"manual_prior_penalty={score_report['manual_closure_prior']['penalty']:.6f} "
            f"worst_manual_prior={score_report['manual_closure_prior']['worst_joint']} "
            f"worst_opposition={score_report['opposition']['worst_pair']}"
        )
        for joint_name, comparison in candidate["limit_comparison"].items():
            print(
                f"  {joint_name}: reference={comparison['current']} "
                f"candidate={comparison['candidate']} delta={comparison['delta']}"
            )


if __name__ == "__main__":
    main()
