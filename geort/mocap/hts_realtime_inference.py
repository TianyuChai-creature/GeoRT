"""Run realtime GeoRT inference from HTS UDP frames."""

from __future__ import annotations

import argparse
import queue
from dataclasses import dataclass
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import sapien

from geort import get_config, load_model
from geort.export import resolve_checkpoint_dir
from geort.env.hand import HandKinematicModel
from geort.mocap.hts_right_mocap import EXPECTED_HTS_LANDMARKS, iter_hts_points
from geort.mocap.realtime_provenance import verify_archived_checkpoint
from geort.mocap.realtime_runtime import RealtimeSafetyController, SessionRecorder, scale_and_clamp_qpos as _scale_and_clamp
from geort.utils.config_utils import parse_config_keypoint_info


PINCH_FINGERS = ("index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class ReceivedPoints:
    """One latest-only realtime input with a receiver-domain timestamp."""

    points: np.ndarray
    recv_ts_s: float
    sender_ts_ns: int | None = None


class HeadlessViewerEnv:
    """Stage-1 diagnostic viewer substitute that leaves mapping and safety unchanged."""

    viewer = None

    def update(self) -> bool:
        return True
DEFAULT_C2B_S42_CHECKPOINT = "checkpoint/custom_right_2026-07-17_12-21-39_c2b_s42"
C2B_S42_LAST_PTH_SHA256 = "dc9c2cc36e20bffe28736ec6111b4401631ee683c6021afe7c816768a4743e73"


def require_c2b_s42_sha(actual_sha256: str) -> str:
    """Reject a realtime startup whose checkpoint is not the audited C2b seed-42 weights."""
    if actual_sha256 != C2B_S42_LAST_PTH_SHA256:
        raise ValueError(
            "C2b seed-42 checkpoint SHA256 mismatch: "
            f"{actual_sha256} != {C2B_S42_LAST_PTH_SHA256}"
        )
    return actual_sha256


def _runtime_git_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _runtime_command() -> str:
    return shlex.join([sys.executable, *sys.argv])


class LatestPointBuffer:
    """Thread-safe latest-frame buffer, with ordered mode reserved for recorded replay."""

    def __init__(self, *, preserve_order: bool = False):
        self.preserve_order = bool(preserve_order)
        self._queue = queue.Queue() if self.preserve_order else queue.Queue(maxsize=1)

    @staticmethod
    def _coerce(
        points: np.ndarray | ReceivedPoints | object,
        *,
        recv_ts_s: float | None,
        sender_ts_ns: int | None,
    ) -> ReceivedPoints:
        if isinstance(points, ReceivedPoints):
            return points
        if hasattr(points, "points"):
            return ReceivedPoints(
                points=np.asarray(points.points, dtype=np.float32),
                recv_ts_s=time.monotonic() if recv_ts_s is None else float(recv_ts_s),
                sender_ts_ns=getattr(points, "source_ts_ns", None) if sender_ts_ns is None else sender_ts_ns,
            )
        return ReceivedPoints(
            points=np.asarray(points, dtype=np.float32),
            recv_ts_s=time.monotonic() if recv_ts_s is None else float(recv_ts_s),
            sender_ts_ns=sender_ts_ns,
        )

    def put(
        self,
        points: np.ndarray | ReceivedPoints | object,
        *,
        recv_ts_s: float | None = None,
        sender_ts_ns: int | None = None,
    ) -> None:
        frame = self._coerce(points, recv_ts_s=recv_ts_s, sender_ts_ns=sender_ts_ns)
        if not self.preserve_order and self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(frame)

    def get_latest(self) -> ReceivedPoints | None:
        if self.preserve_order:
            try:
                return self._queue.get_nowait()
            except queue.Empty:
                return None
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
                point_buffer.put(
                    points,
                    recv_ts_s=time.monotonic(),
                    sender_ts_ns=getattr(points, "source_ts_ns", None),
                )
        except Exception as exc:  # pragma: no cover - surfaced in live terminal output.
            print(f"[HTSRealtime] Receiver stopped: {exc}")

    thread = threading.Thread(target=receive, name=f"hts-{hand_side}-point-receiver", daemon=True)
    thread.start()
    return thread


def iter_recorded_replay(
    session_path: Path | str,
    *,
    sleep_fn=time.sleep,
) -> Iterable[np.ndarray]:
    """Replay raw recorded points with their recorded receiver-domain cadence."""
    frames_path = Path(session_path) / "frames.npz"
    with np.load(frames_path) as frames:
        if "raw_points" not in frames or "t_recv_s" not in frames:
            raise ValueError(f"Replay session lacks raw_points/t_recv_s: {frames_path}")
        raw_points = np.asarray(frames["raw_points"], dtype=np.float32)
        recv_ts_s = np.asarray(frames["t_recv_s"], dtype=np.float64)
    if raw_points.shape[0] != recv_ts_s.shape[0]:
        raise ValueError("Replay raw_points and t_recv_s length mismatch")
    if raw_points.ndim != 3 or raw_points.shape[1:] != (EXPECTED_HTS_LANDMARKS, 3):
        raise ValueError(f"Replay points must have shape (N, {EXPECTED_HTS_LANDMARKS}, 3)")
    if not np.isfinite(recv_ts_s).all():
        raise ValueError("Replay t_recv_s contains non-finite values")
    for index, points in enumerate(raw_points):
        if index:
            sleep_fn(max(0.0, float(recv_ts_s[index] - recv_ts_s[index - 1])))
        yield points


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
    """Scale realtime qpos targets and always keep them inside URDF joint limits."""
    lower, upper = hand.get_joint_limit()
    return _scale_and_clamp(qpos, lower, upper, qpos_scale)


def map_realtime_frame(model, raw_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map one already converted Quest frame through the evaluation model API."""
    points = validate_live_points(raw_points)
    if points is None:
        raise ValueError("realtime frame is not a finite GeoRT [21, 3] landmark array")
    return points, model.forward(points)


def _record_frame(recorder, model, *, timestamp_s, raw_points, mapped, output, timepoints_s=None, sender_ts_ns=None):
    """Record diagnostics exported by the same model API used by evaluation."""
    if recorder is None:
        return
    normalized = getattr(model, "last_normalized_tips", None)
    mapped_qpos = getattr(model, "last_mapped_qpos", mapped)
    refined_qpos = getattr(model, "last_refined_qpos", mapped)
    timings = dict(getattr(model, "last_timings_ms", {}))
    contact_result = getattr(model, "last_contact_refinement", None)
    selection = None if contact_result is None else contact_result.selection
    contact = {} if contact_result is None else {
        "pair": None if selection is None else selection.pair_name,
        "probabilities": np.asarray(contact_result.probabilities, dtype=np.float32).tolist(),
        "weight": 0.0 if selection is None else float(selection.weight),
        "ignored_pairs": [] if selection is None else list(selection.ignored_pair_names),
    }
    recorder.append(
        timestamp_s=timestamp_s, raw_points=raw_points, normalized_tips=normalized,
        mapped_qpos=mapped_qpos, refined_qpos=refined_qpos, output_qpos=output,
        timings_ms=timings, contact=contact,
        timepoints_s=timepoints_s, sender_ts_ns=sender_ts_ns,
    )


class TipContactVisualizer:
    """Visual-only fingertip proximity markers for collision-free URDFs."""

    def __init__(
        self,
        hand,
        keypoint_info: dict,
        *,
        threshold: float = 0.015,
        report_interval: int = 15,
    ):
        self.hand = hand
        self.scene = hand.get_scene()
        self.threshold = float(threshold)
        self.report_interval = int(report_interval)
        self.tip_indices = self._tip_indices_by_finger(keypoint_info)
        missing = {"thumb", *PINCH_FINGERS} - set(self.tip_indices)
        if missing:
            raise ValueError(f"Missing tip keypoints for contact visualization: {sorted(missing)}")
        self.neutral_markers = self._build_markers(
            radius=0.0075,
            material=(0.05, 0.45, 1.0, 0.85),
            name_prefix="tip_marker",
        )
        self.contact_markers = self._build_markers(
            radius=0.014,
            material=(1.0, 0.04, 0.0, 1.0),
            name_prefix="contact_marker",
        )
        self._hidden_pose = sapien.Pose([0.0, 0.0, -10.0])
        self._last_contact_pairs: tuple[str, ...] = ()

    @staticmethod
    def _tip_indices_by_finger(keypoint_info: dict) -> dict[str, int]:
        tips = {}
        for idx, (finger, keypoint_type) in enumerate(zip(keypoint_info["finger"], keypoint_info["type"])):
            if keypoint_type == "tip":
                tips[finger] = idx
        return tips

    def _build_markers(self, *, radius: float, material: tuple[float, float, float, float], name_prefix: str):
        markers = {}
        for finger in ("thumb", *PINCH_FINGERS):
            builder = self.scene.create_actor_builder()
            builder.add_sphere_visual(radius=radius, material=material, name=f"{name_prefix}_{finger}_visual")
            marker = builder.build_kinematic(name=f"{name_prefix}_{finger}")
            marker.set_pose(self._hidden_pose if hasattr(self, "_hidden_pose") else sapien.Pose([0.0, 0.0, -10.0]))
            markers[finger] = marker
        return markers

    def update(self, qpos: np.ndarray, *, frame_id: int) -> None:
        tip_points = self._tip_points_in_world(qpos)
        for finger, point in tip_points.items():
            self.neutral_markers[finger].set_pose(sapien.Pose(point))
            self.contact_markers[finger].set_pose(self._hidden_pose)

        active_pairs = []
        distances = {}
        thumb = tip_points["thumb"]
        for finger in PINCH_FINGERS:
            dist = float(np.linalg.norm(thumb - tip_points[finger]))
            distances[finger] = dist
            if dist <= self.threshold:
                active_pairs.append(f"thumb__{finger}:{dist * 1000.0:.1f}mm")
                self.contact_markers["thumb"].set_pose(sapien.Pose(thumb))
                self.contact_markers[finger].set_pose(sapien.Pose(tip_points[finger]))

        active_pairs_tuple = tuple(active_pairs)
        should_report = self.report_interval > 0 and frame_id % self.report_interval == 0
        if active_pairs_tuple != self._last_contact_pairs or should_report:
            nearest = min(distances.items(), key=lambda item: item[1])
            status = ", ".join(active_pairs) if active_pairs else "none"
            print(
                "[HTSRealtime] contact_visual "
                f"active={status} nearest=thumb__{nearest[0]}:{nearest[1] * 1000.0:.1f}mm"
            )
            self._last_contact_pairs = active_pairs_tuple

    def _tip_points_in_world(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        sim_qpos = self.hand.convert_user_order_to_sim_order(qpos)
        self.hand.pmodel.compute_forward_kinematics(sim_qpos)
        points = {}
        links = self.hand.hand.get_links()
        for finger, keypoint_idx in self.tip_indices.items():
            link = self.hand.keypoint_links[keypoint_idx]
            link_idx = links.index(link)
            pose = self.hand.pmodel.get_link_pose(link_idx)
            offset = self.hand.keypoint_offsets[keypoint_idx].reshape(3, 1)
            point = pose.p + (pose.to_transformation_matrix()[:3, :3] @ offset).reshape(-1)
            points[finger] = point.astype(np.float32)
        return points


def run_realtime_viewer_loop(
    *,
    model,
    hand,
    viewer_env,
    point_buffer: LatestPointBuffer,
    max_frames: int | None = None,
    max_duration_s: float | None = None,
    smoothing_alpha: float | None = None,
    fps_interval: int = 60,
    contact_visualizer: TipContactVisualizer | None = None,
    qpos_scale: float = 1.0,
    safety_controller: RealtimeSafetyController | None = None,
    session_recorder: SessionRecorder | None = None,
    estop_key: str = "space",
    freeze_key: str = "f",
    diagnostic_rate_limit_bypass: bool = False,
    render_hz: float = 30.0,
) -> int:
    """Map every accepted input while rendering only at the requested cadence."""
    if render_hz < 0.0:
        raise ValueError("render_hz must be non-negative")
    processed = 0
    last_points = None
    start_time = time.monotonic()
    render_period_s = float("inf") if render_hz == 0.0 else 1.0 / float(render_hz)
    next_render_s = start_time
    freeze_was_down = False
    pending_records = []

    def flush_render(*, force: bool = False) -> bool:
        nonlocal next_render_s
        now_s = time.monotonic()
        if not force and (render_hz == 0.0 or now_s < next_render_s):
            return True
        if render_hz > 0.0:
            if viewer_env.update() is False:
                return False
            rendered_s = time.monotonic()
            next_render_s = rendered_s + render_period_s
        else:
            rendered_s = now_s
        for pending_record in pending_records:
            pending_record["timepoints_s"]["t_render"] = rendered_s
            _record_frame(session_recorder, model, **pending_record)
        pending_records.clear()
        return True

    def finish() -> int:
        flush_render(force=True)
        return processed

    while True:
        if max_duration_s is not None and time.monotonic() - start_time >= max_duration_s:
            return finish()
        window = getattr(getattr(viewer_env, "viewer", None), "window", None)
        freeze_requested = False
        if window is not None and hasattr(window, "key_down"):
            if safety_controller is not None and window.key_down(estop_key):
                safety_controller.set_estop(True)
                hand.set_qpos_target(safety_controller.last_qpos)
            freeze_down = bool(window.key_down(freeze_key))
            freeze_requested = freeze_down and not freeze_was_down
            freeze_was_down = freeze_down

        received = point_buffer.get_latest()
        if received is None:
            if safety_controller is not None:
                hand.set_qpos_target(safety_controller.watchdog(now_s=time.monotonic()))
            if not flush_render():
                return processed
            continue

        t_start = time.monotonic()
        raw_points = received.points
        points = validate_live_points(raw_points)
        if points is None:
            if not flush_render():
                return processed
            continue
        points = smooth_live_points(points, last_points, smoothing_alpha)
        last_points = points
        _, mapped_qpos = map_realtime_frame(model, points)
        t_map = time.monotonic()
        qpos = scale_and_clamp_qpos(mapped_qpos, hand, qpos_scale)
        if safety_controller is not None:
            qpos = safety_controller.accept(
                qpos,
                now_s=time.monotonic(),
                recv_ts_s=received.recv_ts_s,
                bypass_rate_limit=diagnostic_rate_limit_bypass,
            )
        hand.set_qpos_target(qpos)
        t_out = time.monotonic()
        timestamp_s = time.time()
        if freeze_requested and session_recorder is not None:
            session_recorder.freeze_frame(
                timestamp_s=timestamp_s, raw_points=raw_points,
                normalized_tips=getattr(model, "last_normalized_tips", None),
                mapped_qpos=getattr(model, "last_mapped_qpos", mapped_qpos), output_qpos=qpos,
            )
            print(f"[HTSRealtime] frozen_frame={len(session_recorder._frozen_frames)}")
        if contact_visualizer is not None:
            contact_visualizer.update(qpos, frame_id=processed + 1)
        pending_records.append({
            "timestamp_s": timestamp_s, "raw_points": points, "mapped": mapped_qpos, "output": qpos,
            "timepoints_s": {"t_recv": received.recv_ts_s, "t_start": t_start, "t_map": t_map, "t_out": t_out},
            "sender_ts_ns": received.sender_ts_ns,
        })
        processed += 1
        if fps_interval > 0 and processed % fps_interval == 0:
            elapsed = max(time.monotonic() - start_time, 1e-6)
            print(f"[HTSRealtime] processed={processed} fps={processed / elapsed:.1f}")
        if max_frames is not None and processed >= max_frames:
            return finish()
        if not flush_render():
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
    contact_visualizer: TipContactVisualizer | None = None,
    qpos_scale: float = 1.0,
    safety_controller: RealtimeSafetyController | None = None,
    session_recorder: SessionRecorder | None = None,
) -> int:
    """Drive ``hand`` from a finite stream of GeoRT-ready points. Used by tests."""
    processed = 0
    last_points = None
    start_time = time.monotonic()

    for raw_points in points_iter:
        recv_ts_s = time.monotonic()
        for _ in range(viewer_updates_per_frame):
            if viewer_env.update() is False:
                return processed

        points = validate_live_points(raw_points)
        if points is None:
            continue

        points = smooth_live_points(points, last_points, smoothing_alpha)
        last_points = points

        _, mapped_qpos = map_realtime_frame(model, points)
        qpos = scale_and_clamp_qpos(mapped_qpos, hand, qpos_scale)
        if safety_controller is not None:
            qpos = safety_controller.accept(qpos, now_s=time.monotonic(), recv_ts_s=recv_ts_s)
        hand.set_qpos_target(qpos)
        _record_frame(session_recorder, model, timestamp_s=time.time(), raw_points=points, mapped=mapped_qpos, output=qpos)
        if contact_visualizer is not None:
            contact_visualizer.update(qpos, frame_id=processed + 1)
        processed += 1

        if fps_interval > 0 and processed % fps_interval == 0:
            elapsed = max(time.monotonic() - start_time, 1e-6)
            print(f"[HTSRealtime] processed={processed} fps={processed / elapsed:.1f}")

        if max_frames is not None and processed >= max_frames:
            return processed

    return processed


def validate_stage_contact_mode(stage: str, contact_refine: str) -> str:
    """Fix contact mode for the SAPIEN-as-actuator rollout stages."""
    required = {"1": "off", "2": "off", "3": "on"}
    if stage not in required:
        raise ValueError(f"Unknown realtime stage: {stage!r}")
    expected = required[stage]
    if contact_refine != expected:
        raise ValueError(f"Stage {stage} requires --contact_refine {expected}")
    return expected


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
    parser.add_argument(
        "--checkpoint", "-ckpt_tag", "--ckpt_tag", dest="checkpoint",
        default=DEFAULT_C2B_S42_CHECKPOINT,
        help="Audited C2b seed-42 checkpoint directory.",
    )
    parser.add_argument("--archive-root", default="outputs/final_matrix", help="Final-matrix provenance archive.")
    parser.add_argument("--session-root", default="outputs/realtime_sessions", help="Directory for one auditable session per run.")
    parser.add_argument("--render-mode", choices=("inline", "headless"), default="inline", help="Inline SAPIEN render or Stage-1 diagnostic headless loop.")
    parser.add_argument("--render-hz", type=float, default=30.0, help="Maximum SAPIEN render cadence; 0 disables drawing for diagnostics.")
    parser.add_argument("--diagnostic-rate-limit-bypass", action="store_true", help="Stage-1-only diagnostic: bypass rate limiting while retaining clamp, ramp, watchdog and estop.")
    parser.add_argument("--stage", choices=("1", "2", "3"), default="1", help="1=SAPIEN mirror, 2=robot contact off, 3=robot contact on.")
    parser.add_argument("--watchdog-ms", type=float, default=200.0, help="Freeze after this much input silence.")
    parser.add_argument("--ramp-frames", type=int, default=100, help="Frames used after startup/watchdog recovery.")
    parser.add_argument("--max-joint-speed", type=float, default=None, help="Required non-diagnostic speed limit in rad/s.")
    parser.add_argument("--estop-key", default="space", help="SAPIEN viewer key that latches immediate emergency stop.")
    parser.add_argument("--freeze-key", default="f", help="SAPIEN viewer key that stores the current evidence frame.")
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
    parser.add_argument("--max-duration-s", type=float, default=None, help="Optional wall-clock run duration for clean diagnostic capture.")
    parser.add_argument("--replay-session", default=None, help="Recorded session directory; replays raw points at recorded receive cadence.")
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=None,
        help="Optional EMA smoothing alpha in (0, 1]; omit to disable smoothing.",
    )
    parser.add_argument("--fps-interval", type=int, default=60, help="Print FPS every N processed frames; 0 disables.")
    parser.add_argument(
        "--contact-visual",
        action="store_true",
        help="Show fingertip proximity markers for thumb-to-finger contact in the SAPIEN viewer.",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.015,
        help="Tip distance threshold in meters for contact proximity highlighting.",
    )
    parser.add_argument(
        "--contact-report-interval",
        type=int,
        default=15,
        help="Print contact proximity status every N processed frames; 0 only prints on state changes.",
    )
    parser.add_argument(
        "--qpos-scale",
        type=float,
        default=1.0,
        help="Scale realtime qpos targets before always clamping to URDF joint limits.",
    )
    parser.add_argument(
        "--contact_refine", "--contact-refine",
        choices=("off", "on"),
        default="off",
        help="Enable probability-triggered analytic-FK pinch refinement.",
    )
    parser.add_argument(
        "--contact-model-path",
        default="checkpoint/contact_right_d1_full/contact_models.pth",
        help="D1 custom-right four-MLP contact checkpoint.",
    )
    parser.add_argument("--contact_p_lo", "--contact-p-lo", type=float, default=0.5)
    parser.add_argument("--contact_p_hi", "--contact-p-hi", type=float, default=0.8)
    parser.add_argument("--contact-target-dist", type=float, default=0.0, help="Target thumb/finger distance in metres.")
    parser.add_argument("--contact-lambda", type=float, default=1e-3, help="Physical-qpos proximity regularisation.")
    parser.add_argument("--contact-refine-steps", type=int, default=40, help="Fixed CPU projected-Adam iterations.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.smoothing_alpha is not None and not 0.0 < args.smoothing_alpha <= 1.0:
        raise ValueError("--smoothing-alpha must be in (0, 1]")
    if args.qpos_scale <= 0.0:
        raise ValueError("--qpos-scale must be positive")
    if not 0.0 <= args.contact_p_lo < args.contact_p_hi <= 1.0:
        raise ValueError("contact thresholds must satisfy 0 <= --contact-p-lo < --contact-p-hi <= 1")
    if args.contact_target_dist < 0.0 or args.contact_lambda < 0.0:
        raise ValueError("--contact-target-dist and --contact-lambda must be non-negative")
    if args.contact_refine_steps <= 0:
        raise ValueError("--contact-refine-steps must be positive")
    hand_side = infer_hand_side(args.hand, args.hand_side)

    validate_stage_contact_mode(args.stage, args.contact_refine)
    if args.render_mode == "headless" and args.stage != "1":
        raise ValueError("--render-mode headless is only valid for Stage 1 diagnostics")
    if args.diagnostic_rate_limit_bypass and args.stage != "1":
        raise ValueError("--diagnostic-rate-limit-bypass is only valid for Stage 1 diagnostics")
    if args.watchdog_ms <= 0.0 or args.ramp_frames <= 0:
        raise ValueError("watchdog and ramp must be positive")
    if args.max_joint_speed is not None and args.max_joint_speed <= 0.0:
        raise ValueError("max joint speed must be positive when provided")
    if args.max_duration_s is not None and args.max_duration_s < 0.0:
        raise ValueError("max duration must be non-negative when provided")
    if args.render_hz < 0.0:
        raise ValueError("--render-hz must be non-negative")
    if args.max_joint_speed is None and not args.diagnostic_rate_limit_bypass:
        raise ValueError("--max-joint-speed is required unless --diagnostic-rate-limit-bypass is set")
    checkpoint = resolve_checkpoint_dir(args.checkpoint)
    provenance = verify_archived_checkpoint(checkpoint, args.archive_root, repo_root=Path.cwd())
    require_c2b_s42_sha(provenance.last_pth_sha256)
    print(
        "[HTSRealtime] checkpoint="
        f"{checkpoint} sha256={provenance.last_pth_sha256} motion_frame={provenance.motion_frame} "
        f"anchor: {provenance.anchor.get('count', 0)} pairs from {provenance.anchor.get('path', '')} "
        f"render_hz={args.render_hz}"
    )
    print(f"[HTSRealtime] Loading checkpoint={checkpoint} epoch={args.epoch}")
    model = load_model(
        str(checkpoint), epoch=args.epoch,
        contact_refine=args.contact_refine, contact_model_path=args.contact_model_path,
        contact_p_lo=args.contact_p_lo, contact_p_hi=args.contact_p_hi,
        contact_target_dist=args.contact_target_dist, contact_lambda=args.contact_lambda,
        contact_refine_steps=args.contact_refine_steps,
    )
    print(
        "[HTSRealtime] contact_refine="
        f"{args.contact_refine} model={args.contact_model_path} p_lo={args.contact_p_lo} "
        f"p_hi={args.contact_p_hi} target_dist={args.contact_target_dist} "
        f"lambda={args.contact_lambda} steps={args.contact_refine_steps}"
    )

    config = get_config(args.hand)
    hand = HandKinematicModel.build_from_config(config, render=args.render_mode == "inline")
    lower, upper = hand.get_joint_limit()
    safety_controller = RealtimeSafetyController(
        lower=lower, upper=upper, initial_qpos=(np.asarray(lower) + np.asarray(upper)) / 2.0,
        ramp_frames=args.ramp_frames, max_joint_speed=args.max_joint_speed, watchdog_s=args.watchdog_ms / 1000.0,
    )
    session_recorder = SessionRecorder(args.session_root)
    viewer_env = hand.get_viewer_env() if args.render_mode == "inline" else HeadlessViewerEnv()
    contact_visualizer = None
    if args.contact_visual:
        keypoint_info = parse_config_keypoint_info(config)
        hand.initialize_keypoint(keypoint_link_names=keypoint_info["link"], keypoint_offsets=keypoint_info["offset"])
        contact_visualizer = TipContactVisualizer(
            hand,
            keypoint_info,
            threshold=args.contact_threshold,
            report_interval=args.contact_report_interval,
        )

    point_buffer = LatestPointBuffer(preserve_order=bool(args.replay_session))
    if args.replay_session:
        points_iter = iter_recorded_replay(args.replay_session)
        receiver_name = "recorded-replay"
        print(f"[HTSRealtime] Replaying receive cadence from {args.replay_session}")
    else:
        points_iter = iter_hts_points(
            hand_side=hand_side,
            transport=args.transport,
            host=args.host,
            port=args.port,
            timeout_s=args.timeout_s,
            include_timestamps=True,
        )
        receiver_name = hand_side
        print(f"[HTSRealtime] Listening for {hand_side}-hand HTS frames on {args.transport}://{args.host}:{args.port}")
    start_point_receiver(points_iter, point_buffer, hand_side=receiver_name)

    print("[HTSRealtime] Press Ctrl-C or close the viewer to stop.")

    try:
        processed = run_realtime_viewer_loop(
            model=model,
            hand=hand,
            viewer_env=viewer_env,
            point_buffer=point_buffer,
            max_frames=args.max_frames,
            max_duration_s=args.max_duration_s,
            smoothing_alpha=args.smoothing_alpha,
            fps_interval=args.fps_interval,
            contact_visualizer=contact_visualizer,
            qpos_scale=args.qpos_scale,
            safety_controller=safety_controller,
            session_recorder=session_recorder,
            estop_key=args.estop_key,
            freeze_key=args.freeze_key,
            diagnostic_rate_limit_bypass=args.diagnostic_rate_limit_bypass,
            render_hz=args.render_hz,
        )
    except KeyboardInterrupt:
        safety_controller.set_estop(True)
        print("\n[HTSRealtime] Stopped by user; emergency stop latched.")
    else:
        print(f"[HTSRealtime] Stopped after {processed} processed frames.")
    finally:
        session_path = session_recorder.close(
            counters=safety_controller.counters,
            extra_summary={
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": provenance.last_pth_sha256,
                "git_hash": _runtime_git_hash(),
                "command": _runtime_command(),
                "smoothing_alpha": args.smoothing_alpha,
                "stage": args.stage,
                "motion_frame": provenance.motion_frame,
                "render_mode": args.render_mode,
                "render_hz": args.render_hz,
                "diagnostic_rate_limit_bypass": args.diagnostic_rate_limit_bypass,
                "ramp_frames": args.ramp_frames,
                "max_joint_speed_rad_s": args.max_joint_speed,
                "max_joint_step_cap_rad": RealtimeSafetyController.MAX_JOINT_STEP_RAD,
                "rate_dt_cap_ms": RealtimeSafetyController.MAX_RATE_DT_S * 1000.0,
                "watchdog_ms": args.watchdog_ms,
                "max_duration_s": args.max_duration_s,
            },
        )
        print(f"[HTSRealtime] session={session_path}")


if __name__ == "__main__":
    main()
