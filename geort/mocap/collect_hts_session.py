"""Collect a Quest3/HTS rest+motion session."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable, Iterable

import numpy as np

try:
    from geort.mocap.hts_right_mocap import (
        EXPECTED_HTS_LANDMARKS,
        iter_hts_points,
        normalize_hand_side,
    )
    from geort.utils.path import get_data_root
except ModuleNotFoundError:  # pragma: no cover - supports direct file loading in lightweight tests.
    EXPECTED_HTS_LANDMARKS = 21

    def normalize_hand_side(hand_side: str) -> str:
        side = hand_side.lower()
        if side not in ("left", "right"):
            raise ValueError(f"Expected hand_side to be 'left' or 'right', got {hand_side!r}")
        return side

    def get_data_root() -> str:
        return "data"


REST_INSTRUCTIONS = """REST segment:
- Keep four fingers vertical and together.
- Keep the thumb in its natural rest position.
- Keep the hand centered in view.
- Stay still; avoid occlusion and strong backlight.
"""

MOTION_INSTRUCTIONS = """MOTION segment:
- Move each finger through the range needed by the downstream task.
- Keep the hand centered and visible throughout the recording.
- Continue until the timed segment ends.
"""


@dataclass(frozen=True)
class SessionPaths:
    rest: Path
    motion: Path
    metadata: Path


@dataclass(frozen=True)
class SegmentSummary:
    name: str
    requested_duration_s: float
    actual_duration_s: float
    frame_count: int
    estimated_fps: float
    finite: bool
    started_monotonic_s: float
    ended_monotonic_s: float


@dataclass(frozen=True)
class TimedSegment:
    name: str
    frames: np.ndarray
    summary: SegmentSummary

    @classmethod
    def from_frames(
        cls,
        *,
        name: str,
        frames: Iterable[np.ndarray],
        requested_duration_s: float,
        started_monotonic_s: float,
        ended_monotonic_s: float,
    ) -> "TimedSegment":
        frame_array = validate_frame_array(list(frames), label=name)
        actual_duration_s = max(float(ended_monotonic_s) - float(started_monotonic_s), 0.0)
        frame_count = int(frame_array.shape[0])
        estimated_fps = frame_count / actual_duration_s if actual_duration_s > 0 else 0.0
        summary = SegmentSummary(
            name=name,
            requested_duration_s=float(requested_duration_s),
            actual_duration_s=actual_duration_s,
            frame_count=frame_count,
            estimated_fps=float(estimated_fps),
            finite=bool(np.isfinite(frame_array).all()),
            started_monotonic_s=float(started_monotonic_s),
            ended_monotonic_s=float(ended_monotonic_s),
        )
        return cls(name=name, frames=frame_array, summary=summary)


def validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    expected = (EXPECTED_HTS_LANDMARKS, 3)
    if points.shape != expected:
        raise ValueError(f"Expected HTS frame shape {expected}, got {points.shape}")
    return points


def validate_frame_array(frames: Iterable[np.ndarray], *, label: str) -> np.ndarray:
    frame_list = [validate_points(frame) for frame in frames]
    if not frame_list:
        raise RuntimeError(f"No HTS frames were collected for {label}; nothing to save.")
    return np.asarray(frame_list, dtype=np.float32)


def build_session_paths(
    *,
    data_dir: Path | str | None,
    name: str,
    hand_side: str,
    session_id: str,
) -> SessionPaths:
    side = normalize_hand_side(hand_side)
    root = Path(data_dir) if data_dir is not None else Path(get_data_root())
    stem = f"{name}_{side}_{session_id}"
    return SessionPaths(
        rest=root / f"{stem}_rest.npy",
        motion=root / f"{stem}.npy",
        metadata=root / f"{stem}.json",
    )


def collect_timed_segment(
    *,
    segment_name: str,
    points_iter: Iterable[np.ndarray],
    duration_s: float,
    now: Callable[[], float] = time.monotonic,
) -> TimedSegment:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    started = float(now())
    frames: list[np.ndarray] = []
    ended = started
    for points in points_iter:
        ended = float(now())
        frames.append(validate_points(points))
        if ended - started >= duration_s:
            break

    if not frames:
        ended = float(now())

    return TimedSegment.from_frames(
        name=segment_name,
        frames=frames,
        requested_duration_s=duration_s,
        started_monotonic_s=started,
        ended_monotonic_s=ended,
    )


def save_session_outputs(
    *,
    paths: SessionPaths,
    rest: TimedSegment,
    motion: TimedSegment,
    session_id: str,
    name: str,
    hand_side: str,
    transport: str,
    host: str,
    port: int,
    timeout_s: float,
    operator: str,
    device: str,
    firmware: str,
    notes: str,
) -> None:
    paths.rest.parent.mkdir(parents=True, exist_ok=True)
    paths.motion.parent.mkdir(parents=True, exist_ok=True)
    paths.metadata.parent.mkdir(parents=True, exist_ok=True)

    np.save(paths.rest, rest.frames)
    np.save(paths.motion, motion.frames)

    metadata = {
        "id": session_id,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operator": operator,
        "device": device,
        "firmware": firmware,
        "hand_side": normalize_hand_side(hand_side),
        "transport": {
            "mode": transport,
            "host": host,
            "port": int(port),
            "timeout_s": float(timeout_s),
        },
        "outputs": {
            "rest": paths.rest.as_posix(),
            "motion": paths.motion.as_posix(),
        },
        "segments": {
            "rest": asdict(rest.summary),
            "motion": asdict(motion.summary),
        },
        "notes": notes,
    }
    paths.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _wait_for_enter(prompt: str) -> None:
    input(prompt)


def _print_segment_result(segment: TimedSegment) -> None:
    summary = segment.summary
    print(
        f"{summary.name}: frames={summary.frame_count} "
        f"duration={summary.actual_duration_s:.1f}s fps={summary.estimated_fps:.1f} "
        f"finite={summary.finite}"
    )


def collect_session(args: argparse.Namespace) -> SessionPaths:
    paths = build_session_paths(
        data_dir=args.data_dir,
        name=args.name,
        hand_side=args.hand_side,
        session_id=args.session_id,
    )
    print(f"Session id: {args.session_id}")
    print(f"Rest output: {paths.rest}")
    print(f"Motion output: {paths.motion}")
    print(f"Metadata output: {paths.metadata}")

    print()
    print(REST_INSTRUCTIONS)
    _wait_for_enter("Press Enter to start timed REST capture.")
    rest = collect_timed_segment(
        segment_name="rest",
        points_iter=iter_hts_points(
            hand_side=args.hand_side,
            transport=args.transport,
            host=args.host,
            port=args.port,
            timeout_s=args.timeout_s,
        ),
        duration_s=args.rest_duration_s,
    )
    _print_segment_result(rest)

    print()
    print(MOTION_INSTRUCTIONS)
    _wait_for_enter("Press Enter to start timed MOTION capture.")
    motion = collect_timed_segment(
        segment_name="motion",
        points_iter=iter_hts_points(
            hand_side=args.hand_side,
            transport=args.transport,
            host=args.host,
            port=args.port,
            timeout_s=args.timeout_s,
        ),
        duration_s=args.motion_duration_s,
    )
    _print_segment_result(motion)

    save_session_outputs(
        paths=paths,
        rest=rest,
        motion=motion,
        session_id=args.session_id,
        name=args.name,
        hand_side=args.hand_side,
        transport=args.transport,
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
        operator=args.operator,
        device=args.device,
        firmware=args.firmware,
        notes=args.notes,
    )
    print("Saved HTS v3 session outputs.")
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="hts", help="Dataset prefix.")
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--session-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--rest-duration-s", type=float, default=30.0)
    parser.add_argument("--motion-duration-s", type=float, default=900.0)
    parser.add_argument("--transport", choices=("udp", "tcp_server", "tcp_client"), default="udp")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout-s", type=float, default=1.0)
    parser.add_argument("--operator", default="")
    parser.add_argument("--device", default="Quest 3")
    parser.add_argument("--firmware", default="")
    parser.add_argument("--notes", default="")
    return parser


def main() -> None:
    collect_session(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
