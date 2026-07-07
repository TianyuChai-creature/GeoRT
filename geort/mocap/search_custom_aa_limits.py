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
REFERENCE_AA_LIMIT = (-0.30, 0.35)


def aa_joint_names_for_hand(hand: str) -> list[str]:
    hand_name = hand.lower()
    if "left" in hand_name:
        side = "L"
    elif "right" in hand_name:
        side = "R"
    else:
        raise ValueError(f"Cannot infer hand side from {hand!r}; expected name containing left or right")
    return [f"{finger}-{side}-MCP2" for finger in AA_FINGER_NAMES]


def reference_aa_limits_for_joints(joint_names: list[str]) -> dict[str, tuple[float, float]]:
    return {joint_name: REFERENCE_AA_LIMIT for joint_name in joint_names}
ADJACENT_PAIR_NAMES = ["index__middle", "middle__ring", "ring__pinky"]
OPPOSITION_PAIR_NAMES = ["thumb__index", "thumb__middle", "thumb__ring", "thumb__pinky"]
OPTIMIZED_PAIR_NAMES = ADJACENT_PAIR_NAMES
OVERLAP_FIELDS = ["iou", "overlap_a_ratio", "overlap_b_ratio"]


def _round_pair(pair: tuple[float, float]) -> list[float]:
    return [round(float(pair[0]), 6), round(float(pair[1]), 6)]


def _round_limit_dict(limits: dict[str, tuple[float, float]]) -> dict[str, list[float]]:
    return {name: _round_pair(limit) for name, limit in limits.items()}


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
    current_limits: dict[str, tuple[float, float]],
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
        for joint_name, (current_lower, current_upper) in current_limits.items():
            lower_min, lower_max, upper_min, upper_max = _limit_bounds(current_lower, current_upper, min_width)
            for _attempt in range(100):
                lower = float(rng.uniform(lower_min, lower_max))
                upper = float(rng.uniform(upper_min, upper_max))
                if lower <= 0.0 <= upper and upper - lower >= min_width:
                    candidate[joint_name] = (lower, upper)
                    break
            else:
                candidate[joint_name] = _repair_limit_pair(lower_min, upper_min, (current_lower, current_upper), min_width)
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


def score_limit_candidate(
    *,
    dataset_overlap: dict[str, dict[str, float]],
    urdf_overlap: dict[str, dict[str, float]],
    urdf_tips: dict[str, np.ndarray],
    adjacent_pair_names: list[str],
    opposition_pair_names: list[str],
    candidate_limits: dict[str, tuple[float, float]] | None = None,
    reference_limits: dict[str, tuple[float, float]] | None = None,
    iou_tolerance: float = 0.0,
    iou_floor: float = 0.01,
    p_norm: float = 6.0,
    contact_threshold: float = 0.015,
    reach_penalty_weight: float = 10.0,
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
    opposition = _opposition_reach_report(
        urdf_tips,
        pair_names=opposition_pair_names,
        contact_threshold=contact_threshold,
        p_norm=p_norm,
    )
    regularization = {"value": 0.0, "per_joint": {}, "reference_limits": {}}
    if candidate_limits is not None and reference_limits is not None:
        regularization = compute_limit_regularization(candidate_limits, reference_limits)
    score = adjacent["p_norm_error"] + float(reach_penalty_weight) * opposition["penalty"] + float(regularization_weight) * regularization["value"]
    return {
        "score": float(score),
        "loss": float(score),
        "adjacent": adjacent,
        "opposition": opposition,
        "regularization": regularization,
        "weights": {
            "reach_penalty_weight": float(reach_penalty_weight),
            "regularization_weight": float(regularization_weight),
            "p_norm": float(p_norm),
            "iou_tolerance": float(iou_tolerance),
            "iou_floor": float(iou_floor),
            "contact_threshold": float(contact_threshold),
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
    overlap_report = build_workspace_overlap_report(dataset_tips, urdf_tips, voxel_size=overlap_voxel_size)
    score_report = score_limit_candidate(
        dataset_overlap=dataset_overlap["dataset"],
        urdf_overlap=overlap_report["urdf"],
        urdf_tips=urdf_tips,
        adjacent_pair_names=adjacent_pair_names,
        opposition_pair_names=opposition_pair_names,
        candidate_limits=candidate_limits,
        reference_limits=reference_limits,
        iou_tolerance=iou_tolerance,
        iou_floor=iou_floor,
        p_norm=p_norm,
        contact_threshold=contact_threshold,
        reach_penalty_weight=reach_penalty_weight,
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
        "opposition_reach_violation": lambda result, pair: result["score_report"]["opposition"]["pair_metrics"].get(pair, {}).get("violation"),
    }
    all_pairs = set()
    for result in results:
        all_pairs.update(result["score_report"]["adjacent"]["pair_metrics"])
        all_pairs.update(result["score_report"]["opposition"]["pair_metrics"])
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
            regularization_weight=regularization_weight,
            candidate_source=spec["source"],
            parent_candidate=spec.get("parent_candidate"),
            step_size=spec.get("step_size"),
        )
        result["candidate_index"] = global_idx
        results.append(result)
        adjacent = result["score_report"]["adjacent"]
        opposition = result["score_report"]["opposition"]
        print(
            f"candidate {global_idx + 1}: loss={result['loss']:.6f} "
            f"source={spec['source']} adjacent={adjacent['p_norm_error']:.6f} opposition={opposition['penalty']:.6f}"
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
    regularization_weight: float = 0.02,
) -> dict:
    config = get_config(hand)
    keypoint_info = parse_config_keypoint_info(config)
    frames = np.load(get_human_data(human_data))
    dataset_tips = extract_dataset_tip_points(frames, keypoint_info)
    hand_model = HandKinematicModel.build_from_config(config, render=False)
    hand_model.initialize_keypoint(keypoint_link_names=keypoint_info["link"], keypoint_offsets=keypoint_info["offset"])
    aa_joint_names = aa_joint_names_for_hand(hand)
    current_limits = _current_aa_limits_from_hand(hand_model, aa_joint_names)
    reference_limits = reference_aa_limits_for_joints(aa_joint_names)
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
        "current_urdf_limits": _round_limit_dict(current_limits),
        "adjacent_iou_pairs": adjacent_pair_names,
        "opposition_reach_pairs": opposition_pair_names,
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
        },
        "scoring": {
            "space": "tip_xyz_only",
            "adjacent_objective": "one_sided_voxel_iou_hinge",
            "opposition_constraint": "min_tip_distance_threshold",
            "iou_tolerance": float(iou_tolerance),
            "iou_floor": float(iou_floor),
            "p_norm": float(p_norm),
            "contact_threshold": float(contact_threshold),
            "reach_penalty_weight": float(reach_penalty_weight),
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
    parser.add_argument("--regularization_weight", type=float, default=0.02)
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
        regularization_weight=args.regularization_weight,
    )
    output = save_search_report(report, _resolve_output(args.output))
    print(f"AA limit search report saved to {output}")
    for rank, candidate in enumerate(report["top_candidates"], start=1):
        score_report = candidate["score_report"]
        print(
            f"rank {rank}: loss={candidate['loss']:.6f} source_candidate={candidate['candidate_index']} "
            f"source={candidate['candidate_info']['source']} "
            f"worst_adjacent={score_report['adjacent']['worst_pair']} "
            f"worst_opposition={score_report['opposition']['worst_pair']}"
        )
        for joint_name, comparison in candidate["limit_comparison"].items():
            print(
                f"  {joint_name}: reference={comparison['current']} "
                f"candidate={comparison['candidate']} delta={comparison['delta']}"
            )


if __name__ == "__main__":
    main()
