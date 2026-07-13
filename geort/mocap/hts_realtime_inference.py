"""Run realtime GeoRT inference from HTS UDP frames."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from collections.abc import Iterable

import numpy as np

from geort import get_config, load_model
from geort.env.hand import HandKinematicModel
from geort.mocap.hts_right_mocap import EXPECTED_HTS_LANDMARKS, iter_hts_points


class LatestPointBuffer:
    """Thread-safe single-slot buffer that keeps only the newest HTS frame."""

    def __init__(self):
        self._queue = queue.Queue(maxsize=1)

    def put(self, points: np.ndarray) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(points)

    def get_latest(self) -> np.ndarray | None:
        latest = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                return latest


def start_point_receiver(
    points_iter: Iterable[np.ndarray],
    point_buffer: LatestPointBuffer,
    *,
    hand_side: str = "unknown",
) -> threading.Thread:
    """Start a daemon receiver so the viewer loop never blocks on UDP input."""

    def receive() -> None:
        try:
            for points in points_iter:
                point_buffer.put(points)
        except Exception as exc:  # pragma: no cover - surfaced in live terminal output.
            print(f"[HTSRealtime] Receiver stopped: {exc}")

    thread = threading.Thread(target=receive, name=f"hts-{hand_side}-point-receiver", daemon=True)
    thread.start()
    return thread


def validate_live_points(points: np.ndarray) -> np.ndarray | None:
    """Return GeoRT-ready points or ``None`` when a live frame should be skipped."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (EXPECTED_HTS_LANDMARKS, 3):
        print(f"[HTSRealtime] Skipping frame with shape {points.shape}")
        return None
    if not np.isfinite(points).all():
        print("[HTSRealtime] Skipping non-finite HTS frame")
        return None
    return points


def smooth_live_points(points: np.ndarray, previous: np.ndarray | None, alpha: float | None) -> np.ndarray:
    """Apply optional exponential smoothing to reduce live HTS jitter."""
    if alpha is None:
        return points
    if previous is None:
        return points
    return (alpha * points + (1.0 - alpha) * previous).astype(np.float32)


def scale_and_clamp_qpos(qpos: np.ndarray, hand, qpos_scale: float) -> np.ndarray:
    """Scale realtime qpos targets and keep them inside URDF joint limits."""
    qpos_array = np.asarray(qpos, dtype=np.float32)
    if qpos_scale == 1.0:
        return qpos_array
    lower, upper = hand.get_joint_limit()
    return np.clip(
        qpos_array * float(qpos_scale),
        np.asarray(lower, dtype=np.float32),
        np.asarray(upper, dtype=np.float32),
    ).astype(np.float32)



def run_realtime_viewer_loop(
    *,
    model,
    hand,
    viewer_env,
    point_buffer: LatestPointBuffer,
    max_frames: int | None = None,
    smoothing_alpha: float | None = None,
    fps_interval: int = 60,
    qpos_scale: float = 1.0,
) -> int:
    """Refresh the viewer continuously and consume the newest available HTS frame."""
    processed = 0
    last_points = None
    start_time = time.monotonic()

    while True:
        if viewer_env.update() is False:
            return processed

        raw_points = point_buffer.get_latest()
        if raw_points is None:
            continue

        points = validate_live_points(raw_points)
        if points is None:
            continue

        points = smooth_live_points(points, last_points, smoothing_alpha)
        last_points = points

        qpos = model.forward(points)
        qpos = scale_and_clamp_qpos(qpos, hand, qpos_scale)
        hand.set_qpos_target(qpos)
        processed += 1

        if fps_interval > 0 and processed % fps_interval == 0:
            elapsed = max(time.monotonic() - start_time, 1e-6)
            print(f"[HTSRealtime] processed={processed} fps={processed / elapsed:.1f}")

        if max_frames is not None and processed >= max_frames:
            return processed


def run_realtime_inference(
    *,
    model,
    hand,
    viewer_env,
    points_iter: Iterable[np.ndarray],
    viewer_updates_per_frame: int = 10,
    max_frames: int | None = None,
    smoothing_alpha: float | None = None,
    fps_interval: int = 60,
    qpos_scale: float = 1.0,
) -> int:
    """Drive ``hand`` from a finite stream of GeoRT-ready points. Used by tests."""
    processed = 0
    last_points = None
    start_time = time.monotonic()

    for raw_points in points_iter:
        for _ in range(viewer_updates_per_frame):
            if viewer_env.update() is False:
                return processed

        points = validate_live_points(raw_points)
        if points is None:
            continue

        points = smooth_live_points(points, last_points, smoothing_alpha)
        last_points = points

        qpos = model.forward(points)
        qpos = scale_and_clamp_qpos(qpos, hand, qpos_scale)
        hand.set_qpos_target(qpos)
        processed += 1

        if fps_interval > 0 and processed % fps_interval == 0:
            elapsed = max(time.monotonic() - start_time, 1e-6)
            print(f"[HTSRealtime] processed={processed} fps={processed / elapsed:.1f}")

        if max_frames is not None and processed >= max_frames:
            return processed

    return processed


def infer_hand_side(hand: str, hand_side: str) -> str:
    """Resolve realtime HTS hand side from CLI input."""
    side = hand_side.lower()
    if side in ("left", "right"):
        return side
    if side != "auto":
        raise ValueError(f"--hand-side must be one of auto, left, right; got {hand_side!r}")

    hand_name = hand.lower()
    if "left" in hand_name:
        return "left"
    if "right" in hand_name:
        return "right"
    raise ValueError(f"Cannot infer HTS hand side from --hand {hand!r}; pass --hand-side left or right.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-hand", "--hand", default="custom_right", help="GeoRT hand config name.")
    parser.add_argument("-ckpt_tag", "--ckpt_tag", default="custom_right_last", help="GeoRT checkpoint tag.")
    parser.add_argument(
        "--hand-side",
        choices=("auto", "left", "right"),
        default="auto",
        help="HTS hand stream to consume. Auto infers from --hand name.",
    )
    parser.add_argument("--epoch", type=int, default=0, help="Checkpoint epoch; 0 loads last.pth.")
    parser.add_argument(
        "--transport",
        choices=("udp", "tcp_server", "tcp_client"),
        default="udp",
        help="HTS transport mode. Defaults to UDP broadcast listening.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind/connect host for the selected transport.")
    parser.add_argument("--port", type=int, default=9000, help="Bind/connect port for HTS streaming.")
    parser.add_argument("--timeout-s", type=float, default=1.0, help="Socket receive timeout in seconds.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for smoke tests.")
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=None,
        help="Optional EMA smoothing alpha in (0, 1]; omit to disable smoothing.",
    )
    parser.add_argument("--fps-interval", type=int, default=60, help="Print FPS every N processed frames; 0 disables.")
    parser.add_argument(
        "--qpos-scale",
        type=float,
        default=1.2,
        help="Scale realtime qpos targets before clamping to URDF joint limits.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.smoothing_alpha is not None and not 0.0 < args.smoothing_alpha <= 1.0:
        raise ValueError("--smoothing-alpha must be in (0, 1]")
    if args.qpos_scale <= 0.0:
        raise ValueError("--qpos-scale must be positive")
    hand_side = infer_hand_side(args.hand, args.hand_side)

    print(f"[HTSRealtime] Loading checkpoint tag={args.ckpt_tag} epoch={args.epoch}")
    model = load_model(args.ckpt_tag, epoch=args.epoch)

    config = get_config(args.hand)
    hand = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = hand.get_viewer_env()
    point_buffer = LatestPointBuffer()
    points_iter = iter_hts_points(
        hand_side=hand_side,
        transport=args.transport,
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
    )
    start_point_receiver(points_iter, point_buffer, hand_side=hand_side)

    print(f"[HTSRealtime] Listening for {hand_side}-hand HTS frames on {args.transport}://{args.host}:{args.port}")
    print("[HTSRealtime] Press Ctrl-C or close the viewer to stop.")

    try:
        processed = run_realtime_viewer_loop(
            model=model,
            hand=hand,
            viewer_env=viewer_env,
            point_buffer=point_buffer,
            max_frames=args.max_frames,
            smoothing_alpha=args.smoothing_alpha,
            fps_interval=args.fps_interval,
            qpos_scale=args.qpos_scale,
        )
    except KeyboardInterrupt:
        print("\n[HTSRealtime] Stopped by user.")
    else:
        print(f"[HTSRealtime] Stopped after {processed} processed frames.")


if __name__ == "__main__":
    main()
