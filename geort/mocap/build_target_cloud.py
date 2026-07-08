"""Build human-shaped robot chamfer target clouds for GeoRT v3."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


FINGER_LANDMARKS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
CUSTOM_FINGER_PREFIX = {
    "F1": "thumb",
    "F2": "index",
    "F3": "middle",
    "F4": "ring",
    "F5": "pinky",
}


@dataclass(frozen=True)
class AngleProxySpec:
    joint_name: str
    finger: str
    kind: str
    landmark_ids: tuple[int, ...]


@dataclass(frozen=True)
class MoldJoint:
    joint_name: str
    theta_rest: float
    theta_lo: float
    theta_hi: float
    sigma: float
    q_low: float
    q_high: float
    q_default: float
    left_valid: bool
    right_valid: bool
    angle_kind: str = "generic"


@dataclass(frozen=True)
class Mold:
    joints: list[MoldJoint]

    def to_dict(self) -> dict:
        return {"joints": [asdict(joint) for joint in self.joints]}


def _validate_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError(f"Expected HTS frames with shape [T, 21, 3], got {frames.shape}")
    return frames


def _finger_from_joint_name(joint_name: str) -> str:
    prefix = joint_name.split("-", 1)[0]
    if prefix in CUSTOM_FINGER_PREFIX:
        return CUSTOM_FINGER_PREFIX[prefix]
    raise ValueError(f"Cannot infer finger from joint name {joint_name!r}")


def _custom_joint_kind(joint_name: str) -> str:
    if joint_name.startswith("F1-"):
        if joint_name.endswith("MCP2"):
            return "thumb_azimuth"
        if joint_name.endswith("MCP1"):
            return "thumb_elevation"
        return "flexion"
    if joint_name.endswith("MCP2"):
        return "aa"
    return "flexion"


def _allegro_finger_order(config: dict) -> list[str]:
    fingers = []
    for info in config.get("fingertip_link", []):
        name = str(info.get("finger") or info.get("name") or "").lower()
        if "thumb" in name:
            fingers.append("thumb")
        elif "index" in name:
            fingers.append("index")
        elif "middle" in name:
            fingers.append("middle")
        elif "ring" in name:
            fingers.append("ring")
        elif "pinky" in name:
            fingers.append("pinky")
    return fingers


def build_angle_proxy_specs(config: dict) -> list[AngleProxySpec]:
    joint_order = list(config["joint_order"])
    specs: list[AngleProxySpec] = []
    if joint_order and joint_order[0].startswith("F"):
        for joint_name in joint_order:
            finger = _finger_from_joint_name(joint_name)
            ids = FINGER_LANDMARKS[finger]
            specs.append(AngleProxySpec(joint_name, finger, _custom_joint_kind(joint_name), ids))
        return specs

    finger_order = _allegro_finger_order(config)
    if not finger_order:
        finger_order = ["index", "middle", "ring", "thumb"]
    for joint_idx, joint_name in enumerate(joint_order):
        finger = finger_order[min(joint_idx // 4, len(finger_order) - 1)]
        local_idx = joint_idx % 4
        if finger == "thumb" and local_idx == 0:
            kind = "thumb_azimuth"
        elif finger == "thumb" and local_idx == 1:
            kind = "thumb_elevation"
        elif local_idx == 0:
            kind = "aa"
        else:
            kind = "flexion"
        specs.append(AngleProxySpec(joint_name, finger, kind, FINGER_LANDMARKS[finger]))
    return specs


def _safe_normalize(vec: np.ndarray, *, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=axis, keepdims=True)
    return vec / np.maximum(norm, 1e-8)


def _palm_basis(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrist = frames[:, 0, :]
    index_base = frames[:, 5, :]
    pinky_base = frames[:, 17, :]
    x_axis = _safe_normalize(index_base - wrist)
    palm_span = pinky_base - index_base
    normal = _safe_normalize(np.cross(x_axis, palm_span))
    y_axis = _safe_normalize(np.cross(normal, x_axis))
    return x_axis, y_axis, normal


def _signed_projected_angle(vec: np.ndarray, ref: np.ndarray, other_axis: np.ndarray) -> np.ndarray:
    x = np.sum(vec * ref, axis=1)
    y = np.sum(vec * other_axis, axis=1)
    return np.arctan2(y, x).astype(np.float32)


def _flexion_angle(frames: np.ndarray, ids: tuple[int, ...], local_joint: int) -> np.ndarray:
    local_joint = int(np.clip(local_joint, 1, 2))
    a = frames[:, ids[local_joint - 1], :] - frames[:, ids[local_joint], :]
    b = frames[:, ids[local_joint + 1], :] - frames[:, ids[local_joint], :]
    a = _safe_normalize(a)
    b = _safe_normalize(b)
    dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.arccos(dot).astype(np.float32)


def _angle_for_spec(frames: np.ndarray, spec: AngleProxySpec) -> np.ndarray:
    ids = spec.landmark_ids
    x_axis, y_axis, normal = _palm_basis(frames)
    if spec.kind == "aa":
        distal = frames[:, ids[-1], :] - frames[:, ids[0], :]
        projected = distal - np.sum(distal * normal, axis=1, keepdims=True) * normal
        return _signed_projected_angle(projected, x_axis, y_axis)
    if spec.kind == "thumb_azimuth":
        thumb = frames[:, ids[1], :] - frames[:, ids[0], :]
        projected = thumb - np.sum(thumb * normal, axis=1, keepdims=True) * normal
        return _signed_projected_angle(projected, x_axis, y_axis)
    if spec.kind == "thumb_elevation":
        thumb = _safe_normalize(frames[:, ids[1], :] - frames[:, ids[0], :])
        return np.arcsin(np.clip(np.sum(thumb * normal, axis=1), -1.0, 1.0)).astype(np.float32)
    if spec.kind == "flexion":
        local_joint = 1
        if spec.joint_name.endswith("DIP"):
            local_joint = 2
        return _flexion_angle(frames, ids, local_joint)
    raise ValueError(f"Unsupported angle proxy kind {spec.kind!r}")


def extract_angles(frames: np.ndarray, specs: Iterable[AngleProxySpec]) -> np.ndarray:
    frames = _validate_frames(frames)
    columns = [_angle_for_spec(frames, spec) for spec in specs]
    if not columns:
        return np.zeros((frames.shape[0], 0), dtype=np.float32)
    return np.stack(columns, axis=1).astype(np.float32)


def default_qpos_from_limits(q_low: np.ndarray, q_high: np.ndarray) -> np.ndarray:
    q_low = np.asarray(q_low, dtype=np.float32)
    q_high = np.asarray(q_high, dtype=np.float32)
    midpoint = (q_low + q_high) / 2.0
    zero = np.zeros_like(midpoint, dtype=np.float32)
    return np.where((q_low <= 0.0) & (q_high >= 0.0), zero, midpoint).astype(np.float32)


def build_mold(
    *,
    rest_angles: np.ndarray,
    motion_angles: np.ndarray,
    joint_names: list[str],
    q_low: np.ndarray,
    q_high: np.ndarray,
    q_default: np.ndarray,
    pin_k: float = 2.0,
) -> Mold:
    rest_angles = np.asarray(rest_angles, dtype=np.float32)
    motion_angles = np.asarray(motion_angles, dtype=np.float32)
    q_low = np.asarray(q_low, dtype=np.float32)
    q_high = np.asarray(q_high, dtype=np.float32)
    q_default = np.asarray(q_default, dtype=np.float32)
    n_dof = len(joint_names)
    if rest_angles.shape[1] != n_dof or motion_angles.shape[1] != n_dof:
        raise ValueError("Angle arrays must have one column per joint")

    rest_p2 = np.percentile(rest_angles, 2, axis=0)
    rest_p98 = np.percentile(rest_angles, 98, axis=0)
    sigma = (rest_p98 - rest_p2) / 2.0
    theta_rest = np.median(rest_angles, axis=0)
    theta_lo = np.percentile(motion_angles, 2, axis=0)
    theta_hi = np.percentile(motion_angles, 98, axis=0)

    joints = []
    for idx, joint_name in enumerate(joint_names):
        d_minus = theta_rest[idx] - theta_lo[idx]
        d_plus = theta_hi[idx] - theta_rest[idx]
        threshold = pin_k * sigma[idx]
        left_valid = bool(d_minus >= threshold and d_minus > 1e-8)
        right_valid = bool(d_plus >= threshold and d_plus > 1e-8)
        joints.append(
            MoldJoint(
                joint_name=joint_name,
                theta_rest=float(theta_rest[idx]),
                theta_lo=float(theta_lo[idx]),
                theta_hi=float(theta_hi[idx]),
                sigma=float(sigma[idx]),
                q_low=float(q_low[idx]),
                q_high=float(q_high[idx]),
                q_default=float(q_default[idx]),
                left_valid=left_valid,
                right_valid=right_valid,
                angle_kind=_custom_joint_kind(joint_name) if joint_name.startswith("F") else "generic",
            )
        )
    return Mold(joints=joints)


def _map_one_joint(theta: np.ndarray, joint: MoldJoint) -> np.ndarray:
    out = np.full(theta.shape, joint.q_default, dtype=np.float32)
    if not joint.left_valid and not joint.right_valid:
        return out

    if joint.angle_kind == "flexion":
        left_target = joint.q_high
        right_target = joint.q_low
    else:
        left_target = joint.q_low
        right_target = joint.q_high

    left_mask = theta < joint.theta_rest
    right_mask = ~left_mask
    if joint.left_valid:
        denom = max(joint.theta_rest - joint.theta_lo, 1e-8)
        alpha = np.clip((joint.theta_rest - theta[left_mask]) / denom, 0.0, 1.0)
        out[left_mask] = joint.q_default + alpha * (left_target - joint.q_default)
    elif joint.right_valid:
        denom = max(joint.theta_hi - joint.theta_rest, 1e-8)
        slope = (right_target - joint.q_default) / denom
        out[left_mask] = np.clip(
            joint.q_default + (theta[left_mask] - joint.theta_rest) * slope,
            min(joint.q_low, joint.q_high),
            max(joint.q_low, joint.q_high),
        )

    if joint.right_valid:
        denom = max(joint.theta_hi - joint.theta_rest, 1e-8)
        alpha = np.clip((theta[right_mask] - joint.theta_rest) / denom, 0.0, 1.0)
        out[right_mask] = joint.q_default + alpha * (right_target - joint.q_default)
    elif joint.left_valid:
        denom = max(joint.theta_rest - joint.theta_lo, 1e-8)
        slope = (joint.q_default - left_target) / denom
        out[right_mask] = np.clip(
            joint.q_default + (theta[right_mask] - joint.theta_rest) * slope,
            min(joint.q_low, joint.q_high),
            max(joint.q_low, joint.q_high),
        )
    return out.astype(np.float32)


def apply_mold(angles: np.ndarray, mold: Mold) -> np.ndarray:
    angles = np.asarray(angles, dtype=np.float32)
    if angles.ndim != 2 or angles.shape[1] != len(mold.joints):
        raise ValueError(f"Expected angles with shape [T, {len(mold.joints)}], got {angles.shape}")
    columns = [_map_one_joint(angles[:, idx], joint) for idx, joint in enumerate(mold.joints)]
    return np.stack(columns, axis=1).astype(np.float32)


def _non_thumb_mcp1_indices(joint_names: list[str]) -> list[int]:
    return [
        idx
        for idx, joint_name in enumerate(joint_names)
        if joint_name.endswith("MCP1") and not joint_name.startswith("F1-")
    ]


def boost_fist_mcp1_qpos(
    qpos: np.ndarray,
    *,
    joint_names: list[str],
    q_high: np.ndarray,
    fist_indices: np.ndarray,
    boost_alpha: float,
) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32)
    q_high = np.asarray(q_high, dtype=np.float32)
    fist_indices = np.asarray(fist_indices, dtype=np.int64)
    if boost_alpha <= 0.0 or fist_indices.size == 0:
        return qpos.astype(np.float32, copy=True)
    if boost_alpha > 1.0:
        raise ValueError("fist_mcp1_boost_alpha must be <= 1.0")
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos with shape [T, D], got {qpos.shape}")
    if qpos.shape[1] != len(joint_names) or q_high.shape != (len(joint_names),):
        raise ValueError("qpos, q_high, and joint_names dimensions must match")

    boosted = qpos.astype(np.float32, copy=True)
    mcp1_indices = _non_thumb_mcp1_indices(joint_names)
    if not mcp1_indices:
        return boosted
    valid_fist_indices = fist_indices[(0 <= fist_indices) & (fist_indices < boosted.shape[0])]
    if valid_fist_indices.size == 0:
        return boosted
    cols = np.asarray(mcp1_indices, dtype=np.int64)
    current = boosted[np.ix_(valid_fist_indices, cols)]
    target = q_high[cols].reshape(1, -1)
    boosted[np.ix_(valid_fist_indices, cols)] = current + float(boost_alpha) * (target - current)
    return boosted.astype(np.float32)


def density_cap_qpos(
    qpos: np.ndarray,
    *,
    q_low: np.ndarray,
    q_high: np.ndarray,
    voxel_size: float = 0.05,
    cap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float32)
    q_low = np.asarray(q_low, dtype=np.float32)
    q_high = np.asarray(q_high, dtype=np.float32)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos with shape [T, D], got {qpos.shape}")
    span = np.maximum(q_high - q_low, 1e-8)
    normalized = (qpos - q_low.reshape(1, -1)) / span.reshape(1, -1)
    voxels = np.floor(normalized / voxel_size).astype(np.int32)
    _, inverse, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    if cap is None:
        cap = max(1, int(20 * np.median(counts)))
    if cap <= 0:
        raise ValueError("cap must be positive")

    seen: dict[int, int] = {}
    kept = []
    for idx, voxel_id in enumerate(inverse):
        current = seen.get(int(voxel_id), 0)
        if current < cap:
            kept.append(idx)
            seen[int(voxel_id)] = current + 1
    indices = np.asarray(kept, dtype=np.int64)
    return qpos[indices], indices


def save_robot_kinematics_npz(path: Path | str, *, qpos: np.ndarray, keypoints: dict[str, np.ndarray]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, qpos=np.asarray(qpos, dtype=np.float32), keypoint=keypoints)
    return output


def save_mold_json(path: Path | str, mold: Mold) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(mold.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output



def keypoints_from_qpos_sequence(
    hand,
    qpos: np.ndarray,
    keypoint_names: list[str],
) -> dict[str, np.ndarray]:
    collected = {name: [] for name in keypoint_names}
    for row in np.asarray(qpos, dtype=np.float32):
        keypoint = hand.keypoint_from_qpos(row)
        for name in keypoint_names:
            collected[name].append(np.asarray(keypoint[name], dtype=np.float32)[:3])
    return {name: np.asarray(values, dtype=np.float32) for name, values in collected.items()}


def build_target_cloud(
    *,
    config: dict,
    hand,
    motion_path: Path | str,
    rest_path: Path | str,
    output_path: Path | str,
    debug_dir: Path | str,
    keypoint_names: list[str],
    voxel_size: float = 0.05,
    cap: int | None = None,
    pin_k: float = 2.0,
    fist_boost_top_fraction: float = 0.0,
    fist_boost_repeat: int = 0,
    fist_boost_score_mode: str = "curl",
    fist_boost_mcp_weight: float = 2.0,
    fist_boost_pip_weight: float = 1.0,
    fist_boost_dip_weight: float = 0.7,
    fist_mcp1_boost_top_fraction: float = 0.0,
    fist_mcp1_boost_alpha: float = 0.0,
) -> Path:
    rest_frames = _validate_frames(np.load(rest_path))
    motion_frames = _validate_frames(np.load(motion_path))
    if fist_boost_top_fraction > 0.0 and fist_boost_repeat > 0:
        from geort.mocap.hts_prepare_training import append_fist_boost_frames  # deferred to avoid heavy imports at module level

        motion_frames, _fist_boost_report = append_fist_boost_frames(
            motion_frames,
            top_fraction=fist_boost_top_fraction,
            repeat=fist_boost_repeat,
            score_mode=fist_boost_score_mode,
            mcp_weight=fist_boost_mcp_weight,
            pip_weight=fist_boost_pip_weight,
            dip_weight=fist_boost_dip_weight,
        )
    specs = build_angle_proxy_specs(config)
    joint_names = [spec.joint_name for spec in specs]
    rest_angles = extract_angles(rest_frames, specs)
    motion_angles = extract_angles(motion_frames, specs)

    q_low, q_high = hand.get_joint_limit()
    q_low = np.asarray(q_low, dtype=np.float32)
    q_high = np.asarray(q_high, dtype=np.float32)
    q_default = default_qpos_from_limits(q_low, q_high)
    mold = build_mold(
        rest_angles=rest_angles,
        motion_angles=motion_angles,
        joint_names=joint_names,
        q_low=q_low,
        q_high=q_high,
        q_default=q_default,
        pin_k=pin_k,
    )
    qpos_all = apply_mold(motion_angles, mold)
    if fist_mcp1_boost_top_fraction > 0.0 and fist_mcp1_boost_alpha > 0.0:
        from geort.mocap.hts_prepare_training import compute_mcp_weighted_fist_curl_score

        if fist_mcp1_boost_top_fraction > 1.0:
            raise ValueError("fist_mcp1_boost_top_fraction must be <= 1.0")
        score = compute_mcp_weighted_fist_curl_score(motion_frames)
        selected_count = max(1, int(np.ceil(motion_frames.shape[0] * fist_mcp1_boost_top_fraction)))
        fist_indices = np.argsort(score, kind="stable")[:selected_count]
        qpos_all = boost_fist_mcp1_qpos(
            qpos_all,
            joint_names=joint_names,
            q_high=q_high,
            fist_indices=fist_indices,
            boost_alpha=fist_mcp1_boost_alpha,
        )
    qpos, kept_indices = density_cap_qpos(
        qpos_all,
        q_low=q_low,
        q_high=q_high,
        voxel_size=voxel_size,
        cap=cap,
    )
    keypoints = keypoints_from_qpos_sequence(hand, qpos, keypoint_names)

    debug = Path(debug_dir)
    debug.mkdir(parents=True, exist_ok=True)
    np.save(debug / "angles_rest.npy", rest_angles)
    np.save(debug / "angles_motion.npy", motion_angles)
    np.save(debug / "kept_indices.npy", kept_indices)
    save_mold_json(debug / "mold.json", mold)
    return save_robot_kinematics_npz(output_path, qpos=qpos, keypoints=keypoints)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", required=True, help="GeoRT hand config name.")
    parser.add_argument("--motion", required=True, help="Motion HTS .npy file.")
    parser.add_argument("--rest", required=True, help="Rest HTS .npy file.")
    parser.add_argument("--output", default=None, help="Output target cloud .npz path.")
    parser.add_argument("--debug-dir", default=None, help="Directory for mold/debug artifacts.")
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--cap", type=int, default=None)
    parser.add_argument("--pin-k", type=float, default=2.0)
    parser.add_argument(
        "--fist-boost-top-fraction", type=float, default=0.0,
        help="Fraction of strongest fist frames to repeat before building target cloud; 0 disables.",
    )
    parser.add_argument(
        "--fist-boost-repeat", type=int, default=0,
        help="Number of extra repeats for selected strongest fist frames.",
    )
    parser.add_argument(
        "--fist-boost-score-mode", choices=("curl", "mcp_weighted"), default="curl",
        help="Score used to select frames for fist boost.",
    )
    parser.add_argument("--fist-boost-mcp-weight", type=float, default=2.0)
    parser.add_argument("--fist-boost-pip-weight", type=float, default=1.0)
    parser.add_argument("--fist-boost-dip-weight", type=float, default=0.7)
    parser.add_argument(
        "--fist-mcp1-boost-top-fraction",
        type=float,
        default=0.0,
        help="Fraction of strongest fist target frames whose non-thumb MCP1 qpos is pushed toward q_high; 0 disables.",
    )
    parser.add_argument(
        "--fist-mcp1-boost-alpha",
        type=float,
        default=0.0,
        help="Blend factor toward q_high for non-thumb MCP1 on selected fist target frames; 0 disables.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    from geort.env.hand import HandKinematicModel
    from geort.utils.config_utils import get_config, parse_config_keypoint_info

    config = get_config(args.hand)
    keypoint_info = parse_config_keypoint_info(config)
    hand = HandKinematicModel.build_from_config(config, render=False)
    hand.initialize_keypoint(
        keypoint_link_names=keypoint_info["link"],
        keypoint_offsets=keypoint_info["offset"],
    )
    output = args.output or f"data/{config['name']}_humanshaped.npz"
    debug_dir = args.debug_dir or f"data/{config['name']}_humanshaped_debug"
    result = build_target_cloud(
        config=config,
        hand=hand,
        motion_path=args.motion,
        rest_path=args.rest,
        output_path=output,
        debug_dir=debug_dir,
        keypoint_names=keypoint_info["link"],
        voxel_size=args.voxel_size,
        cap=args.cap,
        pin_k=args.pin_k,
        fist_boost_top_fraction=args.fist_boost_top_fraction,
        fist_boost_repeat=args.fist_boost_repeat,
        fist_boost_score_mode=args.fist_boost_score_mode,
        fist_boost_mcp_weight=args.fist_boost_mcp_weight,
        fist_boost_pip_weight=args.fist_boost_pip_weight,
        fist_boost_dip_weight=args.fist_boost_dip_weight,
        fist_mcp1_boost_top_fraction=args.fist_mcp1_boost_top_fraction,
        fist_mcp1_boost_alpha=args.fist_mcp1_boost_alpha,
    )
    print(f"Human-shaped target cloud saved to {result}")
    print(f"Debug artifacts saved to {debug_dir}")


if __name__ == "__main__":
    main()
