"""Prepare a final HTS training dataset from a raw acquisition .npy file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geort.mocap.hts_balance import (
    build_stage2_report,
    select_balanced_frame_indices,
)
from geort.mocap.hts_stage3 import (
    build_stage3_report,
    compute_frame_weights,
)


def _relative_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def default_output_paths(source_path: Path | str) -> tuple[Path, Path]:
    source = Path(source_path)
    stem = source.stem
    parent = source.parent if source.parent != Path("") else Path(".")
    return parent / f"{stem}_train.npy", parent / f"{stem}_train.json"


def prepare_training_dataset(
    *,
    source_path: Path | str,
    output_path: Path | str | None = None,
    metadata_path: Path | str | None = None,
    dataset_id: str | None = None,
    voxel_size: float = 0.003,
    max_per_voxel: int = 24,
    preserve_contact_pairs: str = "all",
    contact_threshold: float = 0.025,
    contact_bonus: float = 2.0,
    max_weight: float = 5.0,
) -> tuple[Path, Path]:
    source = Path(source_path)
    default_data, default_metadata = default_output_paths(source)
    output = Path(output_path) if output_path is not None else default_data
    metadata = Path(metadata_path) if metadata_path is not None else default_metadata

    frames = np.load(source)
    selected, quota_report = select_balanced_frame_indices(
        frames,
        voxel_size=voxel_size,
        max_per_voxel=max_per_voxel,
        preserve_contact_pairs=preserve_contact_pairs,
        contact_threshold=contact_threshold,
    )
    train_frames = np.asarray(frames, dtype=np.float32)[selected]

    stage2_report = build_stage2_report(
        frames,
        selected,
        voxel_size=voxel_size,
        max_per_voxel=max_per_voxel,
    )
    stage2_report["quota_selection"] = quota_report["fingers"]
    stage2_report["baseline_balanced_frames"] = quota_report["baseline_balanced_frames"]
    stage2_report["contact_preserve"] = quota_report["contact_preserve"]

    weights, masks, min_dist = compute_frame_weights(
        train_frames,
        threshold=contact_threshold,
        contact_bonus=contact_bonus,
        max_weight=max_weight,
    )
    stage3_report = build_stage3_report(
        weights,
        masks,
        min_dist,
        threshold=contact_threshold,
        contact_bonus=contact_bonus,
        max_weight=max_weight,
    )
    stage3_report["source"] = "final_train_dataset"

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, train_frames)

    metadata_parent = metadata.parent
    doc = {
        "id": dataset_id or output.stem,
        "data_path": _relative_path(output, metadata_parent),
        "weights": weights.astype(float).tolist(),
        "processing": {
            "source": _relative_path(source, metadata_parent),
            "raw_frames": int(frames.shape[0]),
            "train_frames": int(train_frames.shape[0]),
            "stage2": stage2_report,
            "stage3": stage3_report,
        },
    }
    metadata.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return output, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/hts_right.npy", help="Raw HTS .npy acquisition dataset.")
    parser.add_argument("--output", default=None, help="Final training .npy path. Defaults to <input_stem>_train.npy.")
    parser.add_argument("--metadata", default=None, help="Training JSON path. Defaults to <input_stem>_train.json.")
    parser.add_argument("--dataset-id", default=None, help="Dataset id stored in the training JSON.")
    parser.add_argument("--voxel-size", type=float, default=0.003, help="Voxel size in meters for 6D PIP/TIP balancing.")
    parser.add_argument("--max-per-voxel", type=int, default=24, help="Maximum retained frames per occupied voxel per finger.")
    parser.add_argument("--preserve-contact-pairs", default="all", help="Preserve tip-tip contact frames: none, all, or comma-separated fingers.")
    parser.add_argument("--contact-threshold", type=float, default=0.025, help="Tip-tip contact threshold in meters.")
    parser.add_argument("--contact-bonus", type=float, default=2.0, help="Weight bonus added to detected contact frames.")
    parser.add_argument("--max-weight", type=float, default=5.0, help="Maximum frame weight.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output, metadata = prepare_training_dataset(
        source_path=args.input,
        output_path=args.output,
        metadata_path=args.metadata,
        dataset_id=args.dataset_id,
        voxel_size=args.voxel_size,
        max_per_voxel=args.max_per_voxel,
        preserve_contact_pairs=args.preserve_contact_pairs,
        contact_threshold=args.contact_threshold,
        contact_bonus=args.contact_bonus,
        max_weight=args.max_weight,
    )
    doc = json.loads(metadata.read_text(encoding="utf-8"))
    processing = doc["processing"]
    stage3 = processing["stage3"]
    print(f"Training dataset saved to {output}")
    print(f"Training metadata saved to {metadata}")
    print(
        f"frames: raw={processing['raw_frames']} train={processing['train_frames']} "
        f"contact_frames={stage3['contact_frames']} weight_mean={stage3['weight_mean']:.3f}"
    )


if __name__ == "__main__":
    main()
