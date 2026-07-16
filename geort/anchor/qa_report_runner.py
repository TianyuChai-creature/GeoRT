"""CPU-only static QA evidence collection for custom_right sparse anchors.

The module intentionally reads the persisted analytic/SAPIEN parity result rather than
re-running it.  Robot metrics use the current compat analytic FK callback only.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from geort.anchor.compat import get_joint_limits, make_analytic_tip_callback
from geort.anchor.human_geometry import (
    FINGER_LANDMARKS,
    FINGER_NAMES,
    align_hts_to_palm,
    estimate_finger_angles,
)
from geort.anchor.interpolate import interpolate_sparse_trajectory
from geort.anchor.mining import LEVEL_FRACTIONS, mine_human_anchor_records
from geort.utils.config_utils import (
    get_config,
    parse_config_keypoint_info,
    select_keypoint_types,
)

ANCHOR_TYPES = ("lateral", "bending")
EXPECTED_ROWS = {"lateral": 50, "bending": 100}


@dataclass(frozen=True)
class QAInputs:
    hand: str
    human_data: Path
    human_anchors: Path
    parity_qpos: Path
    parity_report: Path
    normalization_path: Path
    robot_data: Path


def _as_path(value: Path | str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _git_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _percentiles(selected: np.ndarray, distribution: np.ndarray) -> list[float]:
    values = np.sort(np.asarray(distribution, dtype=np.float64).reshape(-1))
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("empty valid-frame parameter distribution")
    ranks = np.searchsorted(values, selected, side="right") - 1
    ranks = np.clip(ranks, 0, values.size - 1)
    denominator = max(values.size - 1, 1)
    return (100.0 * ranks / denominator).astype(float).tolist()


def _quality(points: np.ndarray) -> dict[str, Any]:
    values = np.asarray(points, dtype=np.float64)
    steps = np.diff(values, axis=0)
    distances = np.linalg.norm(steps, axis=1)
    duplicate = bool(np.any(distances <= 1e-10))
    if distances.size == 0:
        raise ValueError("trajectory requires at least two points")
    minimum = float(distances.min())
    ratio = float(distances.max() / minimum) if minimum > 1e-10 else float("inf")
    dots = np.einsum("ij,ij->i", steps[:-1], steps[1:])
    all_positive = bool(np.all(dots > 0.0))
    return {
        "interval_m": distances.tolist(),
        "step_ratio_max_min": ratio,
        "all_direction_dots_positive": all_positive,
        "min_direction_dot": float(dots.min()) if dots.size else float("inf"),
        "has_duplicate_or_degenerate_interval": duplicate,
        "status": "PASS"
        if not duplicate and all_positive and ratio <= 3.0
        else "FAIL",
    }


def _normalise(points: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    center = np.asarray(stats["center"], dtype=np.float64)
    scale = float(stats["scale"])
    if center.shape != (3,) or scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("invalid normalization stats")
    return (np.asarray(points, dtype=np.float64) - center) / scale


def _summary(values: np.ndarray, *, radians_to_degrees: bool = False) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if radians_to_degrees:
        array = np.rad2deg(array)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(array.max()),
    }


def _load_sparse_human(path: Path) -> dict[str, np.ndarray]:
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
            raise ValueError(f"human bundle missing fields: {missing}")
        result = {key: np.asarray(bundle[key]) for key in required}
    if result["human_frames"].shape != (50, 21, 3):
        raise ValueError("human bundle must contain 50 sparse 21-point frames")
    return result


def _validate_parity(parity: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    qpos = parity["robot_qpos"]
    fingers = parity["finger_indices"]
    types = parity["anchor_types"].astype(str)
    trajectory_t = parity["trajectory_t"]
    if qpos.shape[0] != 750 or fingers.shape != (750,) or types.shape != (750,):
        raise ValueError("canonical parity qpos must contain exactly 750 rows")
    rows: list[dict[str, Any]] = []
    for finger_index, finger in enumerate(FINGER_NAMES):
        for anchor_type in ANCHOR_TYPES:
            mask = (fingers == finger_index) & (types == anchor_type)
            count = int(mask.sum())
            expected = EXPECTED_ROWS[anchor_type]
            if count != expected:
                raise ValueError(
                    f"{finger} {anchor_type} has {count} parity rows, expected {expected}"
                )
            rows.append(
                {
                    "finger": finger,
                    "anchor_type": anchor_type,
                    "count": count,
                    "trajectory_t_min": float(trajectory_t[mask].min()),
                    "trajectory_t_max": float(trajectory_t[mask].max()),
                }
            )
    return rows


def _five_knot_qpos(
    qpos: np.ndarray, trajectory_t: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    group_qpos = qpos[mask]
    group_t = trajectory_t[mask]
    order = np.argsort(group_t)
    return np.column_stack(
        [
            np.interp(LEVEL_FRACTIONS, group_t[order], group_qpos[order, dimension])
            for dimension in range(group_qpos.shape[1])
        ]
    )


def _group_mask(records: dict[str, np.ndarray], finger_index: int, anchor_type: str) -> np.ndarray:
    return (
        (records["finger_indices"] == finger_index)
        & (records["anchor_types"].astype(str) == anchor_type)
    )


def _selected_tip_rows(records: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask)
    levels = records["levels"][rows]
    order = rows[np.argsort(levels)]
    if not np.array_equal(records["levels"][order], np.arange(5)):
        raise ValueError("human sparse levels must be ordered 0..4")
    return order


def _robot_tip_callback(hand: str):
    config = get_config(hand)
    info = select_keypoint_types(
        parse_config_keypoint_info(config), allowed_types=("tip",)
    )
    if tuple(info["finger"]) != tuple(FINGER_NAMES):
        raise ValueError("custom_right TIP order must be thumb-to-pinky")
    lower, upper = get_joint_limits(config)
    return config, info, make_analytic_tip_callback(config, lower, upper, info["offset"])


def _normalised_range(
    points: np.ndarray, fingers: np.ndarray, stats: dict[str, Any]
) -> dict[str, list[float]]:
    normalized = np.empty_like(points, dtype=np.float64)
    for finger_index, finger in enumerate(FINGER_NAMES):
        rows = fingers == finger_index
        normalized[rows] = _normalise(points[rows], stats[finger])
    return {
        "min": normalized.min(axis=0).tolist(),
        "max": normalized.max(axis=0).tolist(),
    }


def build_report_record(inputs: QAInputs) -> dict[str, Any]:
    """Collect A–D evidence from the persisted custom_right anchor inputs."""
    if inputs.hand != "custom_right":
        raise ValueError("this QA report is intentionally restricted to custom_right")
    paths = {
        name: _as_path(getattr(inputs, name))
        for name in (
            "human_data",
            "human_anchors",
            "parity_qpos",
            "parity_report",
            "normalization_path",
            "robot_data",
        )
    }
    contract = json.loads(paths["normalization_path"].read_text(encoding="utf-8"))
    if tuple(contract.get("finger_names", ())) != tuple(FINGER_NAMES):
        raise ValueError("normalization finger ordering differs from custom_right")
    if Path(contract.get("human_data_source", "")).resolve() != paths["human_data"].resolve():
        raise ValueError(
            "human_data_source mismatch: "
            f"{contract.get('human_data_source')} != {paths['human_data']}"
        )

    raw_frames = np.load(paths["human_data"], allow_pickle=False).astype(np.float64)
    sparse_human = _load_sparse_human(paths["human_anchors"])
    mined = mine_human_anchor_records(raw_frames)
    if not np.array_equal(mined.source_indices, sparse_human["source_indices"]):
        raise ValueError("remined source frames differ from persisted human anchors")

    aligned, palm_valid = align_hts_to_palm(raw_frames)
    angles = estimate_finger_angles(aligned)

    with np.load(paths["parity_qpos"], allow_pickle=False) as bundle:
        parity = {
            "robot_qpos": np.asarray(bundle["robot_qpos"], dtype=np.float64),
            "finger_indices": np.asarray(bundle["finger_indices"], dtype=np.int64),
            "anchor_types": np.asarray(bundle["anchor_types"]),
            "trajectory_t": np.asarray(bundle["trajectory_t"], dtype=np.float64),
            "tip_offsets": np.asarray(bundle["tip_offsets"], dtype=np.float64),
        }
    parity_groups = _validate_parity(parity)
    parity_gate = json.loads(paths["parity_report"].read_text(encoding="utf-8"))
    if float(parity_gate["overall"]["max_m"]) >= float(parity_gate["threshold_m"]):
        raise ValueError("persisted FK parity gate did not pass")

    config, tip_info, analytic_tip = _robot_tip_callback(inputs.hand)
    config_offsets = np.asarray(tip_info["offset"], dtype=np.float64)
    if not np.array_equal(parity["tip_offsets"], config_offsets):
        raise ValueError("parity qpos offsets differ from current custom_right config")

    robot_all_points = np.asarray(
        [
            analytic_tip(qpos, int(finger_index))
            for qpos, finger_index in zip(
                parity["robot_qpos"], parity["finger_indices"], strict=True
            )
        ],
        dtype=np.float64,
    )
    percentile_rows: list[dict[str, Any]] = []
    span_rows: list[dict[str, Any]] = []
    human_groups: list[dict[str, Any]] = []
    robot_groups: list[dict[str, Any]] = []

    for finger_index, finger in enumerate(FINGER_NAMES):
        valid = palm_valid & angles.valid[:, finger_index]
        for anchor_type in ANCHOR_TYPES:
            human_mask = _group_mask(sparse_human, finger_index, anchor_type)
            human_rows = _selected_tip_rows(sparse_human, human_mask)
            human_tips = sparse_human["human_points"][human_rows]
            remine_key = f"{finger}:{anchor_type}"
            diagnostics = mined.group_metadata[remine_key]
            distribution = np.asarray(diagnostics["distribution_values"], dtype=np.float64)
            parameter = str(diagnostics["distribution_parameter"])
            observed = sparse_human["observed_parameters"][human_rows]
            candidate_count = int(sparse_human["candidate_counts"][human_rows][0])
            support_counts = sparse_human["support_counts"][human_rows].astype(int)

            if anchor_type == "bending" and finger_index != 0:
                source = sparse_human["source_indices"][human_rows]
                beta = angles.beta[source, finger_index]
                coupling = {
                    "unit": "deg",
                    "mcp1_minus_pip": _summary(
                        np.abs(beta[:, 0] - beta[:, 1]), radians_to_degrees=True
                    ),
                    "dip_minus_mcp1_over_2": _summary(
                        np.abs(beta[:, 2] - beta[:, 0] / 2.0),
                        radians_to_degrees=True,
                    ),
                }
            else:
                coupling = {"status": "N/A", "reason": "not a non-thumb bending filter"}

            human_quality = _quality(human_tips)
            interpolated = interpolate_sparse_trajectory(
                human_tips, EXPECTED_ROWS[anchor_type]
            )["points"]
            interpolation_quality = _quality(interpolated)
            human_status = (
                "PASS"
                if candidate_count >= 10
                and human_quality["status"] == "PASS"
                and interpolation_quality["all_direction_dots_positive"]
                else "FAIL"
            )
            human_groups.append(
                {
                    "finger": finger,
                    "anchor_type": anchor_type,
                    "candidate_count": candidate_count,
                    "candidate_count_status": "PASS"
                    if candidate_count >= 10
                    else "WARN",
                    "support_counts_by_level": support_counts.tolist(),
                    "coupling_residual": coupling,
                    "sparse_tip_geometry": human_quality,
                    "interpolated_tip_quality": interpolation_quality,
                    "status": human_status,
                }
            )
            percentile_rows.append(
                {
                    "finger": finger,
                    "anchor_type": anchor_type,
                    "parameter": parameter,
                    "level_percentiles": _percentiles(observed, distribution),
                    "selected_parameter": observed.astype(float).tolist(),
                }
            )

            robot_mask = (
                (parity["finger_indices"] == finger_index)
                & (parity["anchor_types"].astype(str) == anchor_type)
            )
            robot_knots = _five_knot_qpos(
                parity["robot_qpos"], parity["trajectory_t"], robot_mask
            )
            robot_tips = np.asarray(
                [analytic_tip(qpos, finger_index) for qpos in robot_knots],
                dtype=np.float64,
            )
            robot_quality = _quality(robot_tips)
            qpos_intervals = np.linalg.norm(np.diff(robot_knots, axis=0), axis=1)
            tip_intervals = np.linalg.norm(np.diff(robot_tips, axis=0), axis=1)
            arc_deviation = None
            if finger_index == 0 and anchor_type == "bending":
                average = float(tip_intervals.mean())
                arc_deviation = float(
                    np.max(np.abs(tip_intervals - average)) / average
                )
            robot_groups.append(
                {
                    "finger": finger,
                    "anchor_type": anchor_type,
                    "joint_interval_l2": qpos_intervals.tolist(),
                    "tip_interval_m": tip_intervals.tolist(),
                    "tip_geometry": robot_quality,
                    "thumb_arc_equal_spacing_max_relative_deviation": arc_deviation,
                    "status": robot_quality["status"],
                }
            )

            human_normalized = _normalise(human_tips, contract["human"][finger])
            robot_normalized = _normalise(robot_tips, contract["robot"][finger])
            human_span = float(
                np.linalg.norm(human_normalized[-1] - human_normalized[0])
            )
            robot_span = float(
                np.linalg.norm(robot_normalized[-1] - robot_normalized[0])
            )
            if robot_span <= 1e-12:
                raise ValueError(f"{finger} {anchor_type} robot span is degenerate")
            span_rows.append(
                {
                    "finger": finger,
                    "anchor_type": anchor_type,
                    "human_tip_span_normalized": human_span,
                    "robot_tip_span_normalized": robot_span,
                    "human_over_robot": human_span / robot_span,
                }
            )

    human_anchor_tips = sparse_human["human_points"]
    human_anchor_fingers = sparse_human["finger_indices"].astype(np.int64)
    record = {
        "decision": {
            "parameter_percentiles": percentile_rows,
            "span_ratios": span_rows,
        },
        "human_self_check": {
            "groups": human_groups,
            "all_groups_pass": all(group["status"] == "PASS" for group in human_groups),
        },
        "robot_and_pairing": {
            "robot_groups": robot_groups,
            "parity_composition": {
                "total": int(parity["robot_qpos"].shape[0]),
                "lateral_total": int(
                    np.count_nonzero(parity["anchor_types"].astype(str) == "lateral")
                ),
                "bending_total": int(
                    np.count_nonzero(parity["anchor_types"].astype(str) == "bending")
                ),
                "groups": parity_groups,
            },
            "parity": parity_gate,
            "fk_backend": "analytic",
            "tip_offsets_m": config_offsets.tolist(),
        },
        "contract": {
            "hand": inputs.hand,
            "human_data_source": contract["human_data_source"],
            "coordinate_space": {
                "human_coordinate_frame": "hand_base",
                "robot_coordinate_frame": "hand_base",
                "units": "m",
            },
            "normalization_path": str(paths["normalization_path"]),
            "normalization_schema_version": contract.get("schema_version"),
            "normalization_ranges": {
                "human_anchor_tips": _normalised_range(
                    human_anchor_tips, human_anchor_fingers, contract["human"]
                ),
                "robot_anchor_trajectory": _normalised_range(
                    robot_all_points, parity["finger_indices"], contract["robot"]
                ),
            },
            "generation_git_hash": _git_hash(),
            "inputs": {name: str(path) for name, path in paths.items()},
            "remined_source_indices_match": True,
        },
    }
    return _jsonable(record)

