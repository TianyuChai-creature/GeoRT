"""Run realtime GeoRT inference from HTS UDP frames."""

from __future__ import annotations

import argparse
import queue
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
    """Scale realtime qpos targets and always keep them inside URDF joint limits."""
    lower, upper = hand.get_joint_limit()
    return _scale_and_clamp(qpos, lower, upper, qpos_scale)


def map_realtime_frame(model, raw_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map one already converted Quest frame through the evaluation model API."""
    points = validate_live_points(raw_points)
    if points is None:
        raise ValueError("realtime frame is not a finite GeoRT [21, 3] landmark array")
    return points, model.forward(points)


def _record_frame(recorder, model, *, timestamp_s, raw_points, mapped, output):
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
    smoothing_alpha: float | None = None,
    fps_interval: int = 60,
    contact_visualizer: TipContactVisualizer | None = None,
    qpos_scale: float = 1.0,
    safety_controller: RealtimeSafetyController | None = None,
    session_recorder: SessionRecorder | None = None,
    estop_key: str = "space",
) -> int:
    """Refresh the viewer continuously and consume the newest available HTS frame."""
    processed = 0
    last_points = None
    start_time = time.monotonic()

    while True:
        if viewer_env.update() is False:
            return processed
        window = getattr(getattr(viewer_env, "viewer", None), "window", None)
        if safety_controller is not None and window is not None and hasattr(window, "key_down"):
            if window.key_down(estop_key):
                safety_controller.set_estop(True)
                hand.set_qpos_target(safety_controller.last_qpos)

        raw_points = point_buffer.get_latest()
        if raw_points is None:
            if safety_controller is not None:
                hand.set_qpos_target(safety_controller.watchdog(now_s=time.monotonic()))
            continue

        points = validate_live_points(raw_points)
        if points is None:
            continue

        points = smooth_live_points(points, last_points, smoothing_alpha)
        last_points = points

        _, mapped_qpos = map_realtime_frame(model, points)
        qpos = scale_and_clamp_qpos(mapped_qpos, hand, qpos_scale)
        if safety_controller is not None:
            qpos = safety_controller.accept(qpos, now_s=time.monotonic())
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
            qpos = safety_controller.accept(qpos, now_s=time.monotonic())
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
        default="checkpoint/custom_right_2026-07-16_22-04-19_c2_s42",
        help="Final-matrix registered checkpoint directory (default: C2 seed 42).",
    )
    parser.add_argument("--archive-root", default="outputs/final_matrix", help="Final-matrix provenance archive.")
    parser.add_argument("--session-root", default="outputs/realtime_sessions", help="Directory for one auditable session per run.")
    parser.add_argument("--stage", choices=("1", "2", "3"), default="1", help="1=SAPIEN mirror, 2=robot contact off, 3=robot contact on.")
    parser.add_argument("--watchdog-ms", type=float, default=200.0, help="Freeze after this much input silence.")
    parser.add_argument("--ramp-frames", type=int, default=100, help="Frames used after startup/watchdog recovery.")
    parser.add_argument("--max-joint-step", type=float, default=0.05, help="Per-joint qpos limit in rad/frame.")
    parser.add_argument("--estop-key", default="space", help="SAPIEN viewer key that latches immediate emergency stop.")
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

    if args.stage == "2" and args.contact_refine != "off":
        raise ValueError("Stage 2 requires --contact_refine off")
    if args.stage == "3" and args.contact_refine != "on":
        raise ValueError("Stage 3 requires --contact_refine on")
    if args.stage in {"2", "3"}:
        raise RuntimeError("未产出+原因: repository has no real-robot actuator adapter; Stage 2/3 are intentionally blocked")
    if args.watchdog_ms <= 0.0 or args.ramp_frames <= 0 or args.max_joint_step <= 0.0:
        raise ValueError("watchdog, ramp and max joint step must be positive")
    checkpoint = resolve_checkpoint_dir(args.checkpoint)
    provenance = verify_archived_checkpoint(checkpoint, args.archive_root, repo_root=Path.cwd())
    print(
        "[HTSRealtime] checkpoint="
        f"{checkpoint} sha256={provenance.last_pth_sha256} motion_frame={provenance.motion_frame} "
        f"anchor: {provenance.anchor.get('count', 0)} pairs from {provenance.anchor.get('path', '')}"
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
    hand = HandKinematicModel.build_from_config(config, render=True)
    lower, upper = hand.get_joint_limit()
    safety_controller = RealtimeSafetyController(
        lower=lower, upper=upper, initial_qpos=(np.asarray(lower) + np.asarray(upper)) / 2.0,
        ramp_frames=args.ramp_frames, max_joint_step=args.max_joint_step, watchdog_s=args.watchdog_ms / 1000.0,
    )
    session_recorder = SessionRecorder(args.session_root)
    viewer_env = hand.get_viewer_env()
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
            contact_visualizer=contact_visualizer,
            qpos_scale=args.qpos_scale,
            safety_controller=safety_controller,
            session_recorder=session_recorder,
            estop_key=args.estop_key,
        )
    except KeyboardInterrupt:
        safety_controller.set_estop(True)
        print("\n[HTSRealtime] Stopped by user; emergency stop latched.")
    else:
        print(f"[HTSRealtime] Stopped after {processed} processed frames.")
    finally:
        session_path = session_recorder.close(
            counters=safety_controller.counters,
            extra_summary={"checkpoint": str(checkpoint), "stage": args.stage, "motion_frame": provenance.motion_frame},
        )
        print(f"[HTSRealtime] session={session_path}")


if __name__ == "__main__":
    main()
