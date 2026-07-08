"""Stage 1 coverage analysis for HTS right-hand GeoRT datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FINGER_KEYPOINTS = {
    "thumb": {"pip": 2, "tip": 4},
    "index": {"pip": 6, "tip": 8},
    "middle": {"pip": 10, "tip": 12},
    "ring": {"pip": 14, "tip": 16},
    "pinky": {"pip": 18, "tip": 20},
}


def extract_finger_features(frames: np.ndarray) -> dict[str, np.ndarray]:
    """Extract per-finger ``PIP xyz + TIP xyz`` 6D feature streams."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError(f"Expected HTS frames with shape [T, 21, 3], got {frames.shape}")

    out: dict[str, np.ndarray] = {}
    for finger, ids in FINGER_KEYPOINTS.items():
        pip = frames[:, ids["pip"], :]
        tip = frames[:, ids["tip"], :]
        out[finger] = np.concatenate([pip, tip], axis=1).astype(np.float32)
    return out


def voxelize_points(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, set[tuple[int, ...]]]:
    """Map 6D points to integer voxel coordinates."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError(f"Expected 6D points with shape [N, 6], got {points.shape}")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    origin = points.min(axis=0, keepdims=True)
    voxel_indices = np.floor((points - origin) / voxel_size).astype(np.int32)
    occupied = {tuple(int(v) for v in row) for row in voxel_indices}
    return voxel_indices, occupied


def detect_axis_bracketed_holes(
    occupied: set[tuple[int, ...]],
    *,
    max_holes: int = 200,
    neighbor_radius: int = 1,
) -> list[tuple[int, ...]]:
    """Find empty 6D voxels bracketed by occupied neighbors on every axis.

    A voxel is reported when it is empty and, for each 6D axis, has occupied
    support within ``neighbor_radius`` steps in both negative and positive
    directions. This is a conservative local hole heuristic for sparse 6D data.
    """
    if not occupied:
        return []

    dims = len(next(iter(occupied)))
    candidates: set[tuple[int, ...]] = set()
    for voxel in occupied:
        for axis in range(dims):
            for delta in (-1, 1):
                candidate = list(voxel)
                candidate[axis] += delta
                candidate_tuple = tuple(candidate)
                if candidate_tuple not in occupied:
                    candidates.add(candidate_tuple)

    holes: list[tuple[int, ...]] = []
    for candidate in sorted(candidates):
        if candidate in occupied:
            continue

        bracketed = True
        for axis in range(dims):
            has_neg = False
            has_pos = False
            for step in range(1, neighbor_radius + 1):
                neg = list(candidate)
                pos = list(candidate)
                neg[axis] -= step
                pos[axis] += step
                has_neg = has_neg or tuple(neg) in occupied
                has_pos = has_pos or tuple(pos) in occupied
            if not (has_neg and has_pos):
                bracketed = False
                break

        if bracketed:
            holes.append(candidate)
            if len(holes) >= max_holes:
                break

    return holes


def _axis_ranges(points: np.ndarray) -> list[list[float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return [[float(lo), float(hi)] for lo, hi in zip(mins, maxs)]


def build_stage1_report(
    frames: np.ndarray,
    *,
    voxel_size: float,
    max_holes_per_finger: int = 200,
    neighbor_radius: int = 1,
) -> dict:
    """Build a Stage 1 6D PIP/TIP voxel coverage report."""
    features = extract_finger_features(frames)
    report = {
        "stage": 1,
        "voxel_space": "pip_tip_6d",
        "voxel_size": float(voxel_size),
        "neighbor_radius": int(neighbor_radius),
        "num_frames": int(np.asarray(frames).shape[0]),
        "fingers": {},
    }

    for finger, points in features.items():
        voxel_indices, occupied = voxelize_points(points, voxel_size=voxel_size)
        holes = detect_axis_bracketed_holes(
            occupied,
            max_holes=max_holes_per_finger,
            neighbor_radius=neighbor_radius,
        )
        unique_counts = np.unique(voxel_indices, axis=0, return_counts=True)[1]
        report["fingers"][finger] = {
            "pip_index": FINGER_KEYPOINTS[finger]["pip"],
            "tip_index": FINGER_KEYPOINTS[finger]["tip"],
            "num_samples": int(points.shape[0]),
            "occupied_voxels": int(len(occupied)),
            "mean_samples_per_occupied_voxel": float(points.shape[0] / max(len(occupied), 1)),
            "max_samples_in_voxel": int(unique_counts.max()) if unique_counts.size else 0,
            "axis_ranges": _axis_ranges(points),
            "hole_count_reported": int(len(holes)),
            "hole_voxels": [list(hole) for hole in holes],
        }

    return report


def save_stage1_report(report: dict, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/hts_right_20260703_quest3_v3.npy", help="Input GeoRT-compatible HTS .npy file.")
    parser.add_argument("--output", default="data/hts_right_20260703_quest3_v3_stage1_coverage.json", help="Output JSON coverage report.")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Voxel size in meters for each of the 6 PIP/TIP axes.")
    parser.add_argument("--max-holes-per-finger", type=int, default=200, help="Maximum local hole candidates to report per finger.")
    parser.add_argument("--neighbor-radius", type=int, default=1, help="Local voxel radius used by axis-bracketed hole detection.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    frames = np.load(args.input)
    report = build_stage1_report(
        frames,
        voxel_size=args.voxel_size,
        max_holes_per_finger=args.max_holes_per_finger,
        neighbor_radius=args.neighbor_radius,
    )
    output = save_stage1_report(report, args.output)
    print(f"Stage 1 coverage report saved to {output}")
    for finger, stats in report["fingers"].items():
        print(
            f"{finger}: occupied={stats['occupied_voxels']} "
            f"reported_holes={stats['hole_count_reported']} "
            f"mean_samples_per_voxel={stats['mean_samples_per_occupied_voxel']:.2f}"
        )


if __name__ == "__main__":
    main()
