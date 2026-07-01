"""Calibrate custom_right non-thumb AA limits from processed right-hand data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geort import get_config
from geort.env.hand import HandKinematicModel
from geort.utils.config_utils import parse_config_keypoint_info

FINGER_JOINT_PREFIXES = {
    "index": "F2",
    "middle": "F3",
    "ring": "F4",
    "pinky": "F5",
}



@dataclass(frozen=True)
class CenteredRange:
    center: float
    low: float
    high: float
    low_delta: float
    high_delta: float


def _normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def segment_xz_projection_angle(vectors: np.ndarray) -> np.ndarray:
    """Return atan2(z, x) for pip->tip vectors projected onto the AA xz plane."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim < 2 or vectors.shape[-1] != 3:
        raise ValueError(f"Expected vectors with final dimension 3, got {vectors.shape}")
    return np.arctan2(vectors[..., 2], vectors[..., 0]).astype(np.float32)


def circular_delta(angle: np.ndarray, center: float) -> np.ndarray:
    """Return per-sample angular difference from center in [-pi, pi]."""
    angle = np.asarray(angle, dtype=np.float32)
    return np.arctan2(np.sin(angle - center), np.cos(angle - center)).astype(np.float32)


def compute_segment_metric(frames: np.ndarray, *, pip_id: int, tip_id: int) -> np.ndarray:
    """Return pip->tip xz projection angle for AA range calibration."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError(f"Expected frames with shape [T, 21, 3], got {frames.shape}")
    return segment_xz_projection_angle(frames[:, tip_id, :3] - frames[:, pip_id, :3])


def robust_centered_range(values: np.ndarray, *, low_percentile: float, high_percentile: float) -> CenteredRange:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Expected a non-empty 1D metric array")
    center = float(np.percentile(values, 50.0))
    delta = circular_delta(values, center)
    low_delta = float(np.percentile(delta, low_percentile))
    high_delta = float(np.percentile(delta, high_percentile))
    return CenteredRange(
        center=center,
        low=float(center + low_delta),
        high=float(center + high_delta),
        low_delta=low_delta,
        high_delta=high_delta,
    )


def interpolate_joint_for_metric_delta(q_values: np.ndarray, metric_delta: np.ndarray, target_delta: float) -> float:
    q_values = np.asarray(q_values, dtype=np.float64)
    metric_delta = np.asarray(metric_delta, dtype=np.float64)
    if q_values.ndim != 1 or metric_delta.ndim != 1 or q_values.shape != metric_delta.shape:
        raise ValueError("q_values and metric_delta must be 1D arrays with the same shape")

    order = np.argsort(metric_delta)
    sorted_metric = metric_delta[order]
    sorted_q = q_values[order]
    unique_metric, unique_indices = np.unique(sorted_metric, return_index=True)
    unique_q = sorted_q[unique_indices]

    target = float(np.clip(target_delta, unique_metric[0], unique_metric[-1]))
    return float(np.interp(target, unique_metric, unique_q))


def suggest_limit_from_metric_range(
    *,
    q_values: np.ndarray,
    robot_metric_delta: np.ndarray,
    human_low_delta: float,
    human_high_delta: float,
    old_lower: float,
    old_upper: float,
    margin_rad: float,
) -> tuple[float, float]:
    q_for_low = interpolate_joint_for_metric_delta(q_values, robot_metric_delta, human_low_delta)
    q_for_high = interpolate_joint_for_metric_delta(q_values, robot_metric_delta, human_high_delta)
    lower = min(q_for_low, q_for_high) - margin_rad
    upper = max(q_for_low, q_for_high) + margin_rad
    return (
        float(np.clip(lower, old_lower, old_upper)),
        float(np.clip(upper, old_lower, old_upper)),
    )


def _segment_pair_by_finger(keypoint_info):
    out = {}
    for pip_idx, tip_idx in keypoint_info["segment_pairs"]:
        finger = keypoint_info["finger"][pip_idx]
        out[finger] = {
            "pip_keypoint_idx": pip_idx,
            "tip_keypoint_idx": tip_idx,
            "human_pip_id": keypoint_info["human_id"][pip_idx],
            "human_tip_id": keypoint_info["human_id"][tip_idx],
        }
    return out


def _sweep_robot_segment_dirs(hand, q0: np.ndarray, joint_idx: int, pip_idx: int, tip_idx: int, q_values: np.ndarray) -> np.ndarray:
    dirs = []
    for q in q_values:
        qpos = q0.copy()
        qpos[joint_idx] = q
        keypoints = hand.keypoint_from_qpos(qpos, ret_vec=True)
        dirs.append(_normalize((keypoints[tip_idx] - keypoints[pip_idx]).reshape(1, 3))[0])
    return np.asarray(dirs, dtype=np.float32)


def calibrate_aa_limits(
    *,
    hand_name: str,
    data_path: Path,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    margin_rad: float = 0.05,
    sweep_samples: int = 401,
) -> dict:
    frames = np.load(data_path).astype(np.float32)
    config = get_config(hand_name)
    keypoint_info = parse_config_keypoint_info(config)

    hand = HandKinematicModel.build_from_config(config, render=False)
    hand.initialize_keypoint(keypoint_link_names=keypoint_info["link"], keypoint_offsets=keypoint_info["offset"])
    joint_lower, joint_upper = hand.get_joint_limit()
    q0 = np.zeros(len(config["joint_order"]), dtype=np.float32)

    pairs_by_finger = _segment_pair_by_finger(keypoint_info)
    results = []
    for finger, prefix in FINGER_JOINT_PREFIXES.items():
        joint_name = f"{prefix}-R-MCP2"
        joint_idx = config["joint_order"].index(joint_name)
        old_lower = float(joint_lower[joint_idx])
        old_upper = float(joint_upper[joint_idx])
        q_values = np.linspace(old_lower, old_upper, sweep_samples, dtype=np.float32)

        pair = pairs_by_finger[finger]
        robot_dirs = _sweep_robot_segment_dirs(
            hand,
            q0,
            joint_idx,
            pair["pip_keypoint_idx"],
            pair["tip_keypoint_idx"],
            q_values,
        )
        robot_metric = segment_xz_projection_angle(robot_dirs)

        human_metric = compute_segment_metric(
            frames,
            pip_id=pair["human_pip_id"],
            tip_id=pair["human_tip_id"],
        )
        human_range = robust_centered_range(
            human_metric,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
        )

        q0_idx = int(np.argmin(np.abs(q_values)))
        robot_metric_delta = circular_delta(robot_metric, float(robot_metric[q0_idx]))
        new_lower, new_upper = suggest_limit_from_metric_range(
            q_values=q_values,
            robot_metric_delta=robot_metric_delta,
            human_low_delta=human_range.low_delta,
            human_high_delta=human_range.high_delta,
            old_lower=old_lower,
            old_upper=old_upper,
            margin_rad=margin_rad,
        )

        results.append({
            "finger": finger,
            "joint": joint_name,
            "plane": "xz",
            "human_pip_id": pair["human_pip_id"],
            "human_tip_id": pair["human_tip_id"],
            "old_lower": old_lower,
            "old_upper": old_upper,
            "human_center": human_range.center,
            "human_low": human_range.low,
            "human_high": human_range.high,
            "human_low_delta": human_range.low_delta,
            "human_high_delta": human_range.high_delta,
            "new_lower": new_lower,
            "new_upper": new_upper,
            "margin_rad": margin_rad,
        })

    return {
        "hand": hand_name,
        "data": str(data_path),
        "metric": "pip->tip xz projection angle atan2(z, x), compared with circular delta",
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "sweep_samples": sweep_samples,
        "results": results,
    }


def write_markdown_report(report: dict, path: Path) -> None:
    lines = [
        "# Custom Right AA Limit Calibration",
        "",
        f"- hand: `{report['hand']}`",
        f"- data: `{report['data']}`",
        f"- metric: {report['metric']}",
        f"- percentile range: {report['low_percentile']} / {report['high_percentile']}",
        f"- sweep samples: {report['sweep_samples']}",
        "",
        "| finger | joint | plane | old lower | old upper | new lower | new upper | human delta low | human delta high |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["results"]:
        lines.append(
            "| {finger} | `{joint}` | {plane} | {old_lower:.4f} | {old_upper:.4f} | "
            "{new_lower:.4f} | {new_upper:.4f} | {human_low_delta:.4f} | {human_high_delta:.4f} |".format(**item)
        )
    lines.extend([
        "",
        "Notes:",
        "- Thumb AA is intentionally excluded.",
        "- `q=0` is treated as the robot natural straight AA pose.",
        "- The AA motion plane is fixed to the global xz plane.",
        "- Angular ranges use circular delta relative to the median/q=0 center, not time-series unwrap.",
        "- Old URDF limits are used only as search bounds, not as percentile targets.",
        "- Apply these limits only after manually reviewing the report, then rebuild robot point cloud, FK, and IK checkpoints.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="custom_right", help="GeoRT hand config name.")
    parser.add_argument("--data", required=True, type=Path, help="Processed right-hand .npy dataset with shape [T, 21, 3].")
    parser.add_argument("--report", type=Path, default=Path("docs/custom_right_aa_limit_calibration.md"))
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.0)
    parser.add_argument("--margin-rad", type=float, default=0.05)
    parser.add_argument("--sweep-samples", type=int, default=401)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = calibrate_aa_limits(
        hand_name=args.hand,
        data_path=args.data,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        margin_rad=args.margin_rad,
        sweep_samples=args.sweep_samples,
    )
    write_markdown_report(report, args.report)
    print(f"AA limit calibration report written to {args.report}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"AA limit calibration JSON written to {args.json}")


if __name__ == "__main__":
    main()
