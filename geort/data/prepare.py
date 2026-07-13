"""Prepare human and robot keypoints for the AnyDexRT training pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from geort.utils.config_utils import get_config, parse_config_keypoint_info
from geort.utils.path import get_data_root, get_human_data


NormalizationStats = dict[str, dict[str, object]]


def _validate_points(points: np.ndarray, finger_names: Sequence[str]) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape [N, K, 3], got {points.shape}")
    if points.shape[1] != len(finger_names):
        raise ValueError(
            f"Got {points.shape[1]} keypoints but {len(finger_names)} finger labels"
        )
    if not np.isfinite(points).all():
        raise ValueError("Point data contains non-finite values")
    return points


def _ordered_fingers(finger_names: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(finger_names))


def fit_finger_normalization(
    points: np.ndarray,
    finger_names: Sequence[str],
) -> NormalizationStats:
    """Fit one AABB center and one isotropic scale for each finger."""
    points = _validate_points(points, finger_names)
    stats: NormalizationStats = {}
    names = np.asarray(finger_names)
    for finger in _ordered_fingers(finger_names):
        finger_points = points[:, names == finger, :].reshape(-1, 3).astype(np.float64)
        lower = finger_points.min(axis=0)
        upper = finger_points.max(axis=0)
        center = (lower + upper) / 2.0
        scale = float((upper - lower).max() / 2.0)
        if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
            raise ValueError(f"Finger {finger!r} has a degenerate workspace")
        stats[finger] = {"center": center, "scale": scale}
    return stats


def normalize_finger_points(
    points: np.ndarray,
    finger_names: Sequence[str],
    stats: Mapping[str, Mapping[str, object]],
) -> np.ndarray:
    """Apply per-finger translation and positive isotropic scaling."""
    points = _validate_points(points, finger_names)
    normalized = np.empty(points.shape, dtype=np.float64)
    names = np.asarray(finger_names)
    for finger in _ordered_fingers(finger_names):
        center = np.asarray(stats[finger]["center"], dtype=np.float64)
        scale = float(stats[finger]["scale"])
        normalized[:, names == finger, :] = (
            points[:, names == finger, :] - center
        ) / scale
    return normalized.astype(np.float32)


def denormalize_finger_points(
    points: np.ndarray,
    finger_names: Sequence[str],
    stats: Mapping[str, Mapping[str, object]],
) -> np.ndarray:
    """Invert per-finger isotropic normalization."""
    points = _validate_points(points, finger_names)
    restored = np.empty(points.shape, dtype=np.float64)
    names = np.asarray(finger_names)
    for finger in _ordered_fingers(finger_names):
        center = np.asarray(stats[finger]["center"], dtype=np.float64)
        scale = float(stats[finger]["scale"])
        restored[:, names == finger, :] = (
            points[:, names == finger, :] * scale + center
        )
    return restored.astype(np.float32)


def _json_stats(stats: Mapping[str, Mapping[str, object]]) -> dict[str, dict]:
    return {
        finger: {
            "center": np.asarray(values["center"], dtype=np.float64).tolist(),
            "scale": float(values["scale"]),
        }
        for finger, values in stats.items()
    }


def load_robot_keypoints(
    robot_path: Path | str,
    keypoint_links: Sequence[str],
) -> np.ndarray:
    """Load keypoints in config order from a GeoRT robot kinematics NPZ."""
    with np.load(robot_path, allow_pickle=True) as robot_data:
        keypoints = robot_data["keypoint"]
        if keypoints.dtype == object:
            keypoint_map = keypoints.item()
            missing = [name for name in keypoint_links if name not in keypoint_map]
            if missing:
                raise KeyError(f"Robot dataset is missing keypoints: {missing}")
            points = np.stack(
                [np.asarray(keypoint_map[name])[:, :3] for name in keypoint_links],
                axis=1,
            )
        else:
            points = np.asarray(keypoints)
    if points.ndim != 3 or points.shape[1:] != (len(keypoint_links), 3):
        raise ValueError(
            f"Expected robot keypoints [N, {len(keypoint_links)}, 3], got {points.shape}"
        )
    return points.astype(np.float32)


def prepare_dataset(
    *,
    human_path: Path | str,
    robot_path: Path | str,
    config: Mapping,
    output_path: Path | str,
    manifest_path: Path | str,
) -> tuple[Path, Path]:
    """Normalize human and robot keypoints and write NPZ plus JSON manifest."""
    human_path = Path(human_path)
    robot_path = Path(robot_path)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    info = parse_config_keypoint_info(config)
    tip_indices = info["tip_indices"]
    keypoint_names = [info["name"][idx] for idx in tip_indices]
    keypoint_links = [info["link"][idx] for idx in tip_indices]
    finger_names = [info["finger"][idx] for idx in tip_indices]
    human_ids = [info["human_id"][idx] for idx in tip_indices]

    human_frames = np.load(human_path, mmap_mode="r")
    if human_frames.ndim != 3 or human_frames.shape[-1] < 3:
        raise ValueError(f"Expected human frames [N, L, 3], got {human_frames.shape}")
    if max(human_ids) >= human_frames.shape[1]:
        raise ValueError("Config human_hand_id is outside the human landmark array")
    human_points = np.asarray(human_frames[:, human_ids, :3], dtype=np.float32)
    robot_points = load_robot_keypoints(robot_path, keypoint_links)

    human_stats = fit_finger_normalization(human_points, finger_names)
    robot_stats = fit_finger_normalization(robot_points, finger_names)
    human_normalized = normalize_finger_points(
        human_points, finger_names, human_stats
    )
    robot_normalized = normalize_finger_points(
        robot_points, finger_names, robot_stats
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        human_points=human_normalized,
        robot_points=robot_normalized,
        keypoint_names=np.asarray(keypoint_names),
        finger_names=np.asarray(finger_names),
        human_ids=np.asarray(human_ids, dtype=np.int64),
    )
    manifest = {
        "schema_version": 1,
        "prepared_data": output_path.name,
        "config": config["name"],
        "keypoint_names": keypoint_names,
        "finger_names": finger_names,
        "human": {
            "source": human_path.name,
            "normalization": _json_stats(human_stats),
        },
        "robot": {
            "source": robot_path.name,
            "normalization": _json_stats(robot_stats),
        },
        "anchors": None,
        "contact": None,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path, manifest_path


def ensure_robot_dataset(
    config: Mapping,
    robot_path: Path,
    *,
    n_samples: int,
) -> Path:
    """Generate the standard robot cache when it does not exist."""
    if robot_path.exists():
        return robot_path
    default_path = Path(get_data_root()) / f"{config['name']}.npz"
    if robot_path.resolve() != default_path.resolve():
        raise FileNotFoundError(
            f"Cannot generate a non-standard robot cache path: {robot_path}"
        )
    if n_samples <= 0:
        raise ValueError("robot-samples must be positive")

    from geort.trainer import GeoRTTrainer

    trainer = GeoRTTrainer(config)
    trainer.generate_robot_kinematics_dataset(n_total=n_samples, save=True)
    if not robot_path.exists():
        raise RuntimeError(f"Robot dataset generation did not create {robot_path}")
    return robot_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", required=True, help="GeoRT config name")
    parser.add_argument("--human-data", required=True, help="Raw HTS NPY name or path")
    parser.add_argument("--robot-data", type=Path, default=None)
    parser.add_argument("--robot-samples", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = get_config(args.hand)
    human_path = get_human_data(args.human_data)
    data_root = Path(get_data_root())
    robot_path = args.robot_data or data_root / f"{config['name']}.npz"
    output_path = args.output or data_root / f"{human_path.stem}_prepared.npz"
    manifest_path = args.manifest or output_path.with_suffix(".json")

    ensure_robot_dataset(config, robot_path, n_samples=args.robot_samples)
    prepare_dataset(
        human_path=human_path,
        robot_path=robot_path,
        config=config,
        output_path=output_path,
        manifest_path=manifest_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Wrote {output_path}")
    print(f"Wrote {manifest_path}")
    for domain in ("human", "robot"):
        print(f"{domain} normalization:")
        for finger, values in manifest[domain]["normalization"].items():
            print(
                f"  {finger}: center={np.round(values['center'], 6).tolist()} "
                f"scale={values['scale']:.6f} m"
            )


if __name__ == "__main__":
    main()
