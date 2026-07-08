"""Stage 2 density balancing for HTS right-hand GeoRT datasets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from geort.mocap.hts_coverage import FINGER_KEYPOINTS, extract_finger_features, voxelize_points

TIP_INDICES = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
CONTACT_FINGERS = ("index", "middle", "ring", "pinky")


def _evenly_spaced_indices(indices: list[int], max_count: int) -> list[int]:
    if len(indices) <= max_count:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, max_count).round().astype(int)
    return [indices[int(pos)] for pos in positions]


def _frame_indices_by_voxel(voxel_indices: np.ndarray) -> dict[tuple[int, ...], list[int]]:
    by_voxel: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for frame_idx, voxel in enumerate(voxel_indices):
        by_voxel[tuple(int(v) for v in voxel)].append(frame_idx)
    return dict(by_voxel)


def _voxelize_with_origin(points: np.ndarray, voxel_size: float, origin: np.ndarray) -> tuple[np.ndarray, set[tuple[int, ...]]]:
    voxel_indices = np.floor((points - origin.reshape(1, -1)) / voxel_size).astype(np.int32)
    occupied = {tuple(int(v) for v in row) for row in voxel_indices}
    return voxel_indices, occupied


def _resolve_contact_fingers(preserve_contact_pairs: str, fingers: Iterable[str]) -> tuple[str, ...]:
    spec = preserve_contact_pairs.lower()
    if spec in ("", "none"):
        return ()
    if spec == "all":
        finger_set = set(fingers)
        return tuple(finger for finger in CONTACT_FINGERS if finger in finger_set)
    requested = tuple(item.strip() for item in spec.split(",") if item.strip())
    invalid = [finger for finger in requested if finger not in CONTACT_FINGERS]
    if invalid:
        raise ValueError(
            "preserve_contact_pairs must be 'none', 'all', or a comma-separated "
            f"subset of {CONTACT_FINGERS}; got invalid fingers {invalid}"
        )
    finger_set = set(fingers)
    return tuple(finger for finger in CONTACT_FINGERS if finger in requested and finger in finger_set)


def _contact_preserve_report(
    frames: np.ndarray,
    *,
    baseline_indices: np.ndarray,
    final_indices: np.ndarray,
    contact_fingers: tuple[str, ...],
    contact_threshold: float,
) -> dict:
    baseline_set = set(int(idx) for idx in baseline_indices)
    final_set = set(int(idx) for idx in final_indices)
    pairs = {}
    for finger in contact_fingers:
        dist = np.linalg.norm(
            frames[:, TIP_INDICES["thumb"], :] - frames[:, TIP_INDICES[finger], :],
            axis=1,
        )
        raw_indices = set(int(idx) for idx in np.flatnonzero(dist < contact_threshold))
        pairs[f"thumb_tip__{finger}_tip"] = {
            "raw_contact_count": int(len(raw_indices)),
            "baseline_contact_count": int(len(raw_indices & baseline_set)),
            "preserved_count": int(len(raw_indices - baseline_set)),
            "final_contact_count": int(len(raw_indices & final_set)),
        }

    return {
        "mode": "tip_tip",
        "threshold": float(contact_threshold),
        "fingers": list(contact_fingers),
        "pairs": pairs,
    }


def select_balanced_frame_indices(
    frames: np.ndarray,
    *,
    voxel_size: float,
    max_per_voxel: int,
    fingers: Iterable[str] = tuple(FINGER_KEYPOINTS.keys()),
    preserve_contact_pairs: str = "none",
    contact_threshold: float = 0.025,
) -> tuple[np.ndarray, dict]:
    """Select full-frame indices with a strict per-finger 6D voxel cap.

    A frame is retained only when every selected finger's PIP/TIP 6D voxel is
    still below ``max_per_voxel``. This preserves complete 21-landmark frames
    while making repeated pauses in any finger configuration unable to dominate
    the balanced dataset.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError(f"Expected HTS frames with shape [T, 21, 3], got {frames.shape}")
    if max_per_voxel <= 0:
        raise ValueError("max_per_voxel must be positive")
    if contact_threshold <= 0:
        raise ValueError("contact_threshold must be positive")

    finger_names = tuple(fingers)
    contact_fingers = _resolve_contact_fingers(preserve_contact_pairs, finger_names)
    features = extract_finger_features(frames)
    per_finger_voxels: dict[str, np.ndarray] = {}
    report = {
        "stage": 2,
        "voxel_space": "pip_tip_6d",
        "voxel_size": float(voxel_size),
        "max_per_voxel": int(max_per_voxel),
        "raw_frames": int(frames.shape[0]),
        "fingers": {},
    }

    for finger in finger_names:
        points = features[finger]
        voxel_indices, occupied = voxelize_points(points, voxel_size=voxel_size)
        per_finger_voxels[finger] = voxel_indices
        report["fingers"][finger] = {
            "raw_frames": int(frames.shape[0]),
            "occupied_voxels": int(len(occupied)),
            "raw_mean_samples_per_voxel": float(frames.shape[0] / max(len(occupied), 1)),
            "selected_by_quota": 0,
        }

    counts = {finger: defaultdict(int) for finger in finger_names}
    selected: list[int] = []
    for frame_idx in range(frames.shape[0]):
        frame_voxels = {
            finger: tuple(int(v) for v in per_finger_voxels[finger][frame_idx])
            for finger in finger_names
        }
        if any(counts[finger][voxel] >= max_per_voxel for finger, voxel in frame_voxels.items()):
            continue

        selected.append(frame_idx)
        for finger, voxel in frame_voxels.items():
            counts[finger][voxel] += 1

    selected_array = np.array(selected, dtype=np.int64)
    baseline_selected_array = selected_array.copy()
    if contact_fingers:
        contact_indices = []
        for finger in contact_fingers:
            dist = np.linalg.norm(
                frames[:, TIP_INDICES["thumb"], :] - frames[:, TIP_INDICES[finger], :],
                axis=1,
            )
            contact_indices.extend(np.flatnonzero(dist < contact_threshold).tolist())
        if contact_indices:
            selected_array = np.array(sorted(set(selected_array.tolist()) | set(contact_indices)), dtype=np.int64)

    for finger in finger_names:
        report["fingers"][finger]["selected_by_quota"] = int(baseline_selected_array.size)

    report["baseline_balanced_frames"] = int(baseline_selected_array.size)
    report["balanced_frames"] = int(selected_array.size)
    report["compression_ratio"] = float(selected_array.size / max(frames.shape[0], 1))
    report["contact_preserve"] = _contact_preserve_report(
        frames,
        baseline_indices=baseline_selected_array,
        final_indices=selected_array,
        contact_fingers=contact_fingers,
        contact_threshold=contact_threshold,
    )
    return selected_array, report


def build_stage2_report(
    frames: np.ndarray,
    selected_indices: np.ndarray,
    *,
    voxel_size: float,
    max_per_voxel: int,
) -> dict:
    """Summarize effective density after strict full-frame selection."""
    frames = np.asarray(frames, dtype=np.float32)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    selected_frames = frames[selected_indices] if selected_indices.size else frames[:0]
    report = {
        "stage": 2,
        "voxel_space": "pip_tip_6d",
        "voxel_size": float(voxel_size),
        "max_per_voxel": int(max_per_voxel),
        "raw_frames": int(frames.shape[0]),
        "balanced_frames": int(selected_frames.shape[0]),
        "compression_ratio": float(selected_frames.shape[0] / max(frames.shape[0], 1)),
        "fingers": {},
    }

    raw_features = extract_finger_features(frames)
    selected_features = extract_finger_features(selected_frames) if selected_frames.size else {
        finger: np.zeros((0, 6), dtype=np.float32) for finger in FINGER_KEYPOINTS
    }

    for finger in FINGER_KEYPOINTS:
        raw_points = raw_features[finger]
        origin = raw_points.min(axis=0)
        _, raw_occupied = _voxelize_with_origin(raw_points, voxel_size, origin)

        if selected_features[finger].shape[0] > 0:
            selected_voxels, selected_occupied = _voxelize_with_origin(selected_features[finger], voxel_size, origin)
            selected_counts = np.unique(selected_voxels, axis=0, return_counts=True)[1]
            effective_max = int(selected_counts.max()) if selected_counts.size else 0
        else:
            selected_occupied = set()
            effective_max = 0

        retained = len(raw_occupied & selected_occupied)
        report["fingers"][finger] = {
            "raw_occupied_voxels": int(len(raw_occupied)),
            "balanced_occupied_voxels": int(len(selected_occupied)),
            "coverage_retained": float(retained / max(len(raw_occupied), 1)),
            "effective_mean_samples_per_voxel": float(selected_frames.shape[0] / max(len(selected_occupied), 1)),
            "effective_max_samples_in_voxel": effective_max,
        }

    return report


def save_balanced_dataset(frames: np.ndarray, selected_indices: np.ndarray, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(frames, dtype=np.float32)[np.asarray(selected_indices, dtype=np.int64)])
    return path


def save_report(report: dict, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/hts_right_20260703_quest3_v3.npy", help="Input GeoRT-compatible HTS .npy file.")
    parser.add_argument("--output", default="data/hts_right_20260703_quest3_v3_balanced.npy", help="Balanced output .npy file.")
    parser.add_argument("--report", default="data/hts_right_20260703_quest3_v3_stage2_balance.json", help="Stage 2 JSON report path.")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="Voxel size in meters for each 6D PIP/TIP axis.")
    parser.add_argument("--max-per-voxel", type=int, default=8, help="Maximum frames retained per occupied voxel per finger before full-frame union.")
    parser.add_argument(
        "--preserve-contact-pairs",
        default="none",
        help="Preserve tip-tip contact frames after balancing: none, all, or comma-separated fingers.",
    )
    parser.add_argument("--contact-threshold", type=float, default=0.025, help="Tip-tip contact threshold in meters.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    frames = np.load(args.input)
    selected, quota_report = select_balanced_frame_indices(
        frames,
        voxel_size=args.voxel_size,
        max_per_voxel=args.max_per_voxel,
        preserve_contact_pairs=args.preserve_contact_pairs,
        contact_threshold=args.contact_threshold,
    )
    output = save_balanced_dataset(frames, selected, args.output)
    report = build_stage2_report(
        frames,
        selected,
        voxel_size=args.voxel_size,
        max_per_voxel=args.max_per_voxel,
    )
    report["quota_selection"] = quota_report["fingers"]
    report["baseline_balanced_frames"] = quota_report["baseline_balanced_frames"]
    report["contact_preserve"] = quota_report["contact_preserve"]
    report_path = save_report(report, args.report)

    print(f"Balanced dataset saved to {output}")
    print(f"Stage 2 report saved to {report_path}")
    print(f"frames: raw={report['raw_frames']} balanced={report['balanced_frames']} compression={report['compression_ratio']:.3f}")
    for finger, stats in report["fingers"].items():
        print(
            f"{finger}: retained_coverage={stats['coverage_retained']:.3f} "
            f"effective_max={stats['effective_max_samples_in_voxel']} "
            f"effective_mean={stats['effective_mean_samples_per_voxel']:.2f}"
        )


if __name__ == "__main__":
    main()
