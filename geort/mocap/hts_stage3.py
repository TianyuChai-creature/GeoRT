"""Stage 3 contact importance weights for balanced HTS datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

JOINT_LABELS = {
    2: "thumb_pip",
    4: "thumb_tip",
    6: "index_pip",
    8: "index_tip",
    10: "middle_pip",
    12: "middle_tip",
    14: "ring_pip",
    16: "ring_tip",
    18: "pinky_pip",
    20: "pinky_tip",
}

TIP_INDICES = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

PIP_INDICES = {
    "thumb": 2,
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}

PINCH_PAIRS = tuple(
    (TIP_INDICES["thumb"], TIP_INDICES[finger])
    for finger in ("index", "middle", "ring", "pinky")
)
AUX_CONTACT_PAIRS = tuple(
    (TIP_INDICES["thumb"], PIP_INDICES[finger])
    for finger in ("index", "middle", "ring", "pinky")
)
CONTACT_PAIRS = PINCH_PAIRS


def _pair_name(i: int, j: int) -> str:
    return f"{JOINT_LABELS.get(i, str(i))}__{JOINT_LABELS.get(j, str(j))}"


def _validate_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError(f"Expected HTS frames with shape [T, 21, 3], got {frames.shape}")
    return frames


def detect_contact_frames(
    frames: np.ndarray,
    *,
    threshold: float,
    contact_pairs: tuple[tuple[int, int], ...] = CONTACT_PAIRS,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Detect contact/pinch frames by geometric joint distances."""
    frames = _validate_frames(frames)
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    masks: dict[str, np.ndarray] = {}
    all_distances = []
    for i, j in contact_pairs:
        dist = np.linalg.norm(frames[:, i, :] - frames[:, j, :], axis=1)
        masks[_pair_name(i, j)] = dist < threshold
        all_distances.append(dist)

    min_dist = np.min(np.stack(all_distances, axis=1), axis=1).astype(np.float32)
    return masks, min_dist


def compute_frame_weights(
    frames: np.ndarray,
    *,
    threshold: float = 0.025,
    contact_bonus: float = 2.0,
    max_weight: float = 5.0,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Return one importance weight per frame based on contact events."""
    frames = _validate_frames(frames)
    masks, min_dist = detect_contact_frames(frames, threshold=threshold)

    contact_count = np.zeros(frames.shape[0], dtype=np.float32)
    for mask in masks.values():
        contact_count += mask.astype(np.float32)

    weights = 1.0 + contact_bonus * np.minimum(contact_count, 1.0)
    weights = np.minimum(weights, max_weight).astype(np.float32)
    return weights, masks, min_dist


def build_stage3_report(
    weights: np.ndarray,
    masks: dict[str, np.ndarray],
    min_dist: np.ndarray,
    *,
    threshold: float,
    contact_bonus: float,
    max_weight: float,
) -> dict:
    weights = np.asarray(weights, dtype=np.float32)
    min_dist = np.asarray(min_dist, dtype=np.float32)
    any_contact = np.zeros(weights.shape[0], dtype=bool)
    contacts = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        any_contact |= mask
        contacts[name] = {
            "count": int(mask.sum()),
            "ratio": float(mask.mean()) if mask.size else 0.0,
        }

    return {
        "stage": 3,
        "source": "stage2_balanced",
        "num_frames": int(weights.shape[0]),
        "threshold": float(threshold),
        "contact_bonus": float(contact_bonus),
        "max_weight": float(max_weight),
        "contact_frames": int(any_contact.sum()),
        "contact_ratio": float(any_contact.mean()) if any_contact.size else 0.0,
        "weight_min": float(weights.min()) if weights.size else 0.0,
        "weight_max": float(weights.max()) if weights.size else 0.0,
        "weight_mean": float(weights.mean()) if weights.size else 0.0,
        "min_contact_distance_mean": float(min_dist.mean()) if min_dist.size else 0.0,
        "min_contact_distance_min": float(min_dist.min()) if min_dist.size else 0.0,
        "contacts": contacts,
    }


def _manifest_relative_path(path: Path, manifest_parent: Path) -> str:
    try:
        return path.resolve().relative_to(manifest_parent.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def save_stage3_outputs(
    weights: np.ndarray,
    report: dict,
    *,
    data_path: Path | str,
    weights_path: Path | str,
    report_path: Path | str,
    manifest_path: Path | str,
    dataset_id: str | None = None,
) -> tuple[Path, Path, Path]:
    data_out = Path(data_path)
    weights_out = Path(weights_path)
    report_out = Path(report_path)
    manifest_out = Path(manifest_path)
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    np.save(weights_out, np.asarray(weights, dtype=np.float32))
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    manifest_parent = manifest_out.parent
    manifest = {
        "id": dataset_id or data_out.stem,
        "data_path": _manifest_relative_path(data_out, manifest_parent),
        "weights_path": _manifest_relative_path(weights_out, manifest_parent),
        "reports": {
            "stage3": _manifest_relative_path(report_out, manifest_parent),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return weights_out, report_out, manifest_out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/hts_right_balanced.npy", help="Stage 2 balanced HTS .npy file.")
    parser.add_argument("--weights", default="data/hts_right_frame_importance.npy", help="Output frame weights .npy path.")
    parser.add_argument("--report", default="data/hts_right_stage3_report.json", help="Output Stage 3 report JSON path.")
    parser.add_argument("--manifest", default="data/hts_right_training_manifest.json", help="Output training manifest JSON path.")
    parser.add_argument("--dataset-id", default="hts_right_stage3", help="Dataset id stored in the manifest.")
    parser.add_argument("--threshold", type=float, default=0.025, help="Contact distance threshold in meters.")
    parser.add_argument("--contact-bonus", type=float, default=2.0, help="Weight bonus added to detected contact frames.")
    parser.add_argument("--max-weight", type=float, default=5.0, help="Maximum frame weight.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    frames = np.load(args.input)
    weights, masks, min_dist = compute_frame_weights(
        frames,
        threshold=args.threshold,
        contact_bonus=args.contact_bonus,
        max_weight=args.max_weight,
    )
    report = build_stage3_report(
        weights,
        masks,
        min_dist,
        threshold=args.threshold,
        contact_bonus=args.contact_bonus,
        max_weight=args.max_weight,
    )
    weights_path, report_path, manifest_path = save_stage3_outputs(
        weights,
        report,
        data_path=args.input,
        weights_path=args.weights,
        report_path=args.report,
        manifest_path=args.manifest,
        dataset_id=args.dataset_id,
    )
    print(f"Stage 3 weights saved to {weights_path}")
    print(f"Stage 3 report saved to {report_path}")
    print(f"Training manifest saved to {manifest_path}")
    print(
        f"frames={report['num_frames']} contact_frames={report['contact_frames']} "
        f"contact_ratio={report['contact_ratio']:.3f} weight_mean={report['weight_mean']:.3f}"
    )
    for name, stats in report["contacts"].items():
        if stats["count"]:
            print(f"{name}: count={stats['count']} ratio={stats['ratio']:.3f}")


if __name__ == "__main__":
    main()
