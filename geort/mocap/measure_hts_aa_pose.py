"""Measure one Quest3 HTS right-hand AA pose state without saving data."""

from __future__ import annotations

import argparse
import math

import numpy as np

from geort.mocap.hts_right_mocap import EXPECTED_HTS_LANDMARKS, iter_right_hts_points

FINGER_SEGMENTS = {
    "index": {"mcp": 5, "pip": 6, "tip": 8},
    "middle": {"mcp": 9, "pip": 10, "tip": 12},
    "ring": {"mcp": 13, "pip": 14, "tip": 16},
    "pinky": {"mcp": 17, "pip": 18, "tip": 20},
}


def xz_projection_angle(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[-1] != 3:
        raise ValueError(f"Expected vectors with final dimension 3, got {vectors.shape}")
    return np.arctan2(vectors[..., 2], vectors[..., 0]).astype(np.float32)


def circular_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    return float(math.atan2(float(np.sin(values).mean()), float(np.cos(values).mean())))


def compute_pose_aa_angles(frames: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[1:] != (EXPECTED_HTS_LANDMARKS, 3):
        raise ValueError(f"Expected frames with shape [T, 21, 3], got {frames.shape}")

    out = {}
    for finger, ids in FINGER_SEGMENTS.items():
        mcp_pip = frames[:, ids["pip"], :3] - frames[:, ids["mcp"], :3]
        pip_tip = frames[:, ids["tip"], :3] - frames[:, ids["pip"], :3]
        out[finger] = {
            "mcp_pip": xz_projection_angle(mcp_pip),
            "pip_tip": xz_projection_angle(pip_tip),
        }
    return out


def summarize_angles(angles: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    summary = {}
    for finger, segments in angles.items():
        item = {}
        for segment_name, values in segments.items():
            values = np.asarray(values, dtype=np.float32)
            item[f"{segment_name}_mean_rad"] = circular_mean(values)
            item[f"{segment_name}_median_rad"] = float(np.median(values))
            item[f"{segment_name}_p05_rad"] = float(np.percentile(values, 5.0))
            item[f"{segment_name}_p95_rad"] = float(np.percentile(values, 95.0))
            item[f"{segment_name}_std_rad"] = float(np.std(values))
        summary[finger] = item
    return summary


def collect_pose_frames(*, transport: str, host: str, port: int, timeout_s: float, max_frames: int) -> np.ndarray:
    frames = []
    for points in iter_right_hts_points(transport=transport, host=host, port=port, timeout_s=timeout_s):
        if np.isfinite(points).all():
            frames.append(points.astype(np.float32))
            print(f"Frames collected: {len(frames)}/{max_frames}", end="\r", flush=True)
        if len(frames) >= max_frames:
            break
    if not frames:
        raise RuntimeError("No finite right-hand HTS frames were collected.")
    print()
    return np.asarray(frames, dtype=np.float32)


def print_summary(*, state: str, frames: np.ndarray, summary: dict[str, dict[str, float]]) -> None:
    print(f"HTS AA pose measurement state={state} frames={frames.shape[0]}")
    print("Angles are xz projection atan2(z, x), radians.")
    print("| finger | MCP->PIP mean | MCP->PIP median | MCP->PIP p05..p95 | PIP->TIP mean | PIP->TIP median | PIP->TIP p05..p95 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for finger in ("index", "middle", "ring", "pinky"):
        item = summary[finger]
        print(
            f"| {finger} | "
            f"{item['mcp_pip_mean_rad']:.4f} | {item['mcp_pip_median_rad']:.4f} | "
            f"{item['mcp_pip_p05_rad']:.4f}..{item['mcp_pip_p95_rad']:.4f} | "
            f"{item['pip_tip_mean_rad']:.4f} | {item['pip_tip_median_rad']:.4f} | "
            f"{item['pip_tip_p05_rad']:.4f}..{item['pip_tip_p95_rad']:.4f} |"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, choices=("closed", "neutral", "spread"), help="Pose label for this single-state capture.")
    parser.add_argument("--transport", choices=("udp", "tcp_server", "tcp_client"), default="udp")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout-s", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=120, help="Number of finite frames to average for this pose.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(f"Hold pose '{args.state}' now. Listening on {args.transport}://{args.host}:{args.port}")
    frames = collect_pose_frames(
        transport=args.transport,
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
        max_frames=args.max_frames,
    )
    summary = summarize_angles(compute_pose_aa_angles(frames))
    print_summary(state=args.state, frames=frames, summary=summary)


if __name__ == "__main__":
    main()
