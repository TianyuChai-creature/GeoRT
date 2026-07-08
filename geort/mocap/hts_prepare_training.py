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


NON_THUMB_FINGER_LANDMARKS = (
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, 1e-8)


def _joint_angle(frames: np.ndarray, ids: tuple[int, int, int, int], local_joint: int) -> np.ndarray:
    a = frames[:, ids[local_joint - 1], :] - frames[:, ids[local_joint], :]
    b = frames[:, ids[local_joint + 1], :] - frames[:, ids[local_joint], :]
    a = _safe_normalize(a)
    b = _safe_normalize(b)
    dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.arccos(dot).astype(np.float32)


def _mcp_angle(frames: np.ndarray, ids: tuple[int, int, int, int]) -> np.ndarray:
    wrist = frames[:, 0, :]
    mcp = frames[:, ids[0], :]
    pip = frames[:, ids[1], :]
    a = _safe_normalize(wrist - mcp)
    b = _safe_normalize(pip - mcp)
    dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.arccos(dot).astype(np.float32)


def compute_fist_curl_score(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError(f"Expected HTS frames with shape [T, 21, 3], got {frames.shape}")
    angles = []
    for ids in NON_THUMB_FINGER_LANDMARKS:
        angles.append(_joint_angle(frames, ids, local_joint=1))
        angles.append(_joint_angle(frames, ids, local_joint=2))
    return np.stack(angles, axis=1).mean(axis=1).astype(np.float32)


def compute_mcp_weighted_fist_curl_score(
    frames: np.ndarray,
    *,
    mcp_weight: float = 2.0,
    pip_weight: float = 1.0,
    dip_weight: float = 0.7,
) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError(f"Expected HTS frames with shape [T, 21, 3], got {frames.shape}")
    if mcp_weight < 0.0 or pip_weight < 0.0 or dip_weight < 0.0:
        raise ValueError("fist boost score weights must be non-negative")
    weight_sum = float(mcp_weight + pip_weight + dip_weight)
    if weight_sum <= 0.0:
        raise ValueError("at least one fist boost score weight must be positive")

    scores = []
    for ids in NON_THUMB_FINGER_LANDMARKS:
        mcp = _mcp_angle(frames, ids)
        pip = _joint_angle(frames, ids, local_joint=1)
        dip = _joint_angle(frames, ids, local_joint=2)
        scores.append((mcp * mcp_weight + pip * pip_weight + dip * dip_weight) / weight_sum)
    return np.stack(scores, axis=1).mean(axis=1).astype(np.float32)


def _compute_fist_boost_score(
    frames: np.ndarray,
    *,
    score_mode: str,
    mcp_weight: float,
    pip_weight: float,
    dip_weight: float,
) -> np.ndarray:
    if score_mode == "curl":
        return compute_fist_curl_score(frames)
    if score_mode == "mcp_weighted":
        return compute_mcp_weighted_fist_curl_score(
            frames,
            mcp_weight=mcp_weight,
            pip_weight=pip_weight,
            dip_weight=dip_weight,
        )
    raise ValueError(f"Unsupported fist boost score mode {score_mode!r}")


def append_fist_boost_frames(
    frames: np.ndarray,
    *,
    top_fraction: float = 0.05,
    repeat: int = 0,
    score_mode: str = "curl",
    mcp_weight: float = 2.0,
    pip_weight: float = 1.0,
    dip_weight: float = 0.7,
) -> tuple[np.ndarray, dict]:
    frames = np.asarray(frames, dtype=np.float32)
    if repeat <= 0 or top_fraction <= 0.0:
        return frames, {
            "enabled": False,
            "top_fraction": float(top_fraction),
            "repeat": int(repeat),
            "score_mode": score_mode,
            "score_weights": {"mcp": float(mcp_weight), "pip": float(pip_weight), "dip": float(dip_weight)},
            "selected_frames": 0,
            "added_frames": 0,
        }
    if top_fraction > 1.0:
        raise ValueError("fist_boost_top_fraction must be <= 1.0")

    score = _compute_fist_boost_score(
        frames,
        score_mode=score_mode,
        mcp_weight=mcp_weight,
        pip_weight=pip_weight,
        dip_weight=dip_weight,
    )
    selected_count = max(1, int(np.ceil(frames.shape[0] * top_fraction)))
    selected = np.argsort(score, kind="stable")[:selected_count]
    repeated = np.repeat(frames[selected], repeat, axis=0)
    boosted = np.concatenate([frames, repeated], axis=0).astype(np.float32)
    return boosted, {
        "enabled": True,
        "top_fraction": float(top_fraction),
        "repeat": int(repeat),
        "score_mode": score_mode,
        "score_weights": {"mcp": float(mcp_weight), "pip": float(pip_weight), "dip": float(dip_weight)},
        "selected_frames": int(selected_count),
        "added_frames": int(repeated.shape[0]),
        "score_p05": float(np.percentile(score, 5)),
        "score_p50": float(np.percentile(score, 50)),
        "score_p95": float(np.percentile(score, 95)),
    }


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
    fist_boost_top_fraction: float = 0.0,
    fist_boost_repeat: int = 0,
    fist_boost_score_mode: str = "curl",
    fist_boost_mcp_weight: float = 2.0,
    fist_boost_pip_weight: float = 1.0,
    fist_boost_dip_weight: float = 0.7,
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
    train_frames, fist_boost_report = append_fist_boost_frames(
        train_frames,
        top_fraction=fist_boost_top_fraction,
        repeat=fist_boost_repeat,
        score_mode=fist_boost_score_mode,
        mcp_weight=fist_boost_mcp_weight,
        pip_weight=fist_boost_pip_weight,
        dip_weight=fist_boost_dip_weight,
    )

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
            "fist_boost": fist_boost_report,
        },
    }
    metadata.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return output, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/hts_right_20260703_quest3_v3.npy", help="Raw HTS .npy acquisition dataset.")
    parser.add_argument("--output", default=None, help="Final training .npy path. Defaults to <input_stem>_train.npy.")
    parser.add_argument("--metadata", default=None, help="Training JSON path. Defaults to <input_stem>_train.json.")
    parser.add_argument("--dataset-id", default=None, help="Dataset id stored in the training JSON.")
    parser.add_argument("--voxel-size", type=float, default=0.003, help="Voxel size in meters for 6D PIP/TIP balancing.")
    parser.add_argument("--max-per-voxel", type=int, default=24, help="Maximum retained frames per occupied voxel per finger.")
    parser.add_argument("--preserve-contact-pairs", default="all", help="Preserve tip-tip contact frames: none, all, or comma-separated fingers.")
    parser.add_argument("--contact-threshold", type=float, default=0.025, help="Tip-tip contact threshold in meters.")
    parser.add_argument("--contact-bonus", type=float, default=2.0, help="Weight bonus added to detected contact frames.")
    parser.add_argument("--max-weight", type=float, default=5.0, help="Maximum frame weight.")
    parser.add_argument("--fist-boost-top-fraction", type=float, default=0.0, help="Fraction of strongest fist frames to repeat after balancing; 0 disables.")
    parser.add_argument("--fist-boost-repeat", type=int, default=0, help="Number of extra repeats for selected strongest fist frames.")
    parser.add_argument("--fist-boost-score-mode", choices=("curl", "mcp_weighted"), default="curl", help="Score used to select frames for fist boost.")
    parser.add_argument("--fist-boost-mcp-weight", type=float, default=2.0, help="MCP proxy weight for mcp_weighted fist boost selection.")
    parser.add_argument("--fist-boost-pip-weight", type=float, default=1.0, help="PIP proxy weight for mcp_weighted fist boost selection.")
    parser.add_argument("--fist-boost-dip-weight", type=float, default=0.7, help="DIP proxy weight for mcp_weighted fist boost selection.")
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
        fist_boost_top_fraction=args.fist_boost_top_fraction,
        fist_boost_repeat=args.fist_boost_repeat,
        fist_boost_score_mode=args.fist_boost_score_mode,
        fist_boost_mcp_weight=args.fist_boost_mcp_weight,
        fist_boost_pip_weight=args.fist_boost_pip_weight,
        fist_boost_dip_weight=args.fist_boost_dip_weight,
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
