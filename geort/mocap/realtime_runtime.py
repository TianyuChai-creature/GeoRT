"""Pure, CPU-only safety and recording primitives for realtime retargeting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RealtimeCounters:
    """Runtime safety event counters, suitable for a session summary."""

    nan_input: int = 0
    rate_limited: int = 0
    watchdog: int = 0
    estop: int = 0


def scale_and_clamp_qpos(
    qpos: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    qpos_scale: float,
) -> np.ndarray:
    """Scale a physical qpos target and always clamp it to hard joint limits."""
    return np.clip(
        np.asarray(qpos, dtype=np.float32) * float(qpos_scale),
        np.asarray(lower, dtype=np.float32),
        np.asarray(upper, dtype=np.float32),
    ).astype(np.float32)


class RealtimeSafetyController:
    """Clamp, ramp and freeze realtime qpos commands without mutating the mapper."""

    def __init__(
        self,
        *,
        lower: np.ndarray,
        upper: np.ndarray,
        initial_qpos: np.ndarray,
        ramp_frames: int = 100,
        max_joint_step: float = 0.05,
        watchdog_s: float = 0.2,
    ) -> None:
        if ramp_frames <= 0:
            raise ValueError("ramp_frames must be positive")
        if max_joint_step <= 0.0:
            raise ValueError("max_joint_step must be positive")
        if watchdog_s <= 0.0:
            raise ValueError("watchdog_s must be positive")
        self.lower = np.asarray(lower, dtype=np.float32)
        self.upper = np.asarray(upper, dtype=np.float32)
        self.last_qpos = np.clip(np.asarray(initial_qpos, dtype=np.float32), self.lower, self.upper)
        if self.last_qpos.shape != self.lower.shape or self.lower.shape != self.upper.shape:
            raise ValueError("lower, upper and initial_qpos must share the same shape")
        self.ramp_frames = int(ramp_frames)
        self.max_joint_step = float(max_joint_step)
        self.watchdog_s = float(watchdog_s)
        self.counters = RealtimeCounters()
        self._ramp_start = self.last_qpos.copy()
        self._ramp_index = 0
        self._last_input_s: float | None = None
        self._estop_latched = False
        self._watchdog_latched = False

    @property
    def estop_latched(self) -> bool:
        return self._estop_latched

    def set_estop(self, enabled: bool) -> None:
        """Latch/unlatch emergency stop; resuming always starts a fresh ramp."""
        enabled = bool(enabled)
        if enabled and not self._estop_latched:
            self.counters.estop += 1
        if self._estop_latched and not enabled:
            self._start_ramp()
        self._estop_latched = enabled

    def _start_ramp(self) -> None:
        self._ramp_start = self.last_qpos.copy()
        self._ramp_index = 0

    def watchdog(self, *, now_s: float) -> np.ndarray:
        """Freeze when the last accepted input is older than the watchdog deadline."""
        timed_out = self._last_input_s is None or now_s - self._last_input_s > self.watchdog_s
        if timed_out and not self._watchdog_latched:
            self.counters.watchdog += 1
            self._watchdog_latched = True
            self._start_ramp()
        return self.last_qpos.copy()

    def accept(self, qpos: np.ndarray, *, now_s: float, bypass_rate_limit: bool = False) -> np.ndarray:
        """Return the safe target for one candidate qpos at monotonic time ``now_s``."""
        candidate = np.asarray(qpos, dtype=np.float32)
        if candidate.shape != self.last_qpos.shape:
            raise ValueError(f"Expected qpos shape {self.last_qpos.shape}, got {candidate.shape}")
        if not np.isfinite(candidate).all():
            self.counters.nan_input += 1
            return self.last_qpos.copy()
        self._last_input_s = float(now_s)
        if self._estop_latched:
            return self.last_qpos.copy()
        if self._watchdog_latched:
            self._watchdog_latched = False
            self._start_ramp()

        target = np.clip(candidate, self.lower, self.upper)
        if self._ramp_index < self.ramp_frames:
            self._ramp_index += 1
            alpha = self._ramp_index / self.ramp_frames
            target = self._ramp_start + alpha * (target - self._ramp_start)
        target = np.clip(target, self.lower, self.upper)
        delta = target - self.last_qpos
        limited_delta = delta if bypass_rate_limit else np.clip(delta, -self.max_joint_step, self.max_joint_step)
        if not bypass_rate_limit and not np.array_equal(delta, limited_delta):
            self.counters.rate_limited += int(np.count_nonzero(delta != limited_delta))
        self.last_qpos = np.clip(self.last_qpos + limited_delta, self.lower, self.upper).astype(np.float32)
        return self.last_qpos.copy()


class SessionRecorder:
    """Accumulate auditable realtime samples and write one session directory."""

    def __init__(self, root: Path | str = "outputs/realtime_sessions") -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = Path(root) / stamp
        self._frames: list[dict[str, Any]] = []
        self._frozen_frames: list[dict[str, Any]] = []

    def freeze_frame(
        self,
        *,
        timestamp_s: float,
        raw_points: np.ndarray,
        normalized_tips: np.ndarray | None,
        mapped_qpos: np.ndarray,
        output_qpos: np.ndarray,
    ) -> None:
        """Store an operator-requested frozen realtime frame for offline inspection."""
        self._frozen_frames.append({
            "timestamp_s": float(timestamp_s),
            "raw_points": np.asarray(raw_points, dtype=np.float32),
            "normalized_tips": None if normalized_tips is None else np.asarray(normalized_tips, dtype=np.float32),
            "mapped_qpos": np.asarray(mapped_qpos, dtype=np.float32),
            "output_qpos": np.asarray(output_qpos, dtype=np.float32),
        })

    def append(
        self,
        *,
        timestamp_s: float,
        raw_points: np.ndarray,
        normalized_tips: np.ndarray | None,
        mapped_qpos: np.ndarray,
        refined_qpos: np.ndarray,
        output_qpos: np.ndarray,
        timings_ms: dict[str, float],
        contact: dict[str, Any] | None,
        timepoints_s: dict[str, float] | None = None,
        sender_ts_ns: int | None = None,
    ) -> None:
        self._frames.append({
            "timestamp_s": float(timestamp_s),
            "raw_points": np.asarray(raw_points, dtype=np.float32),
            "normalized_tips": None if normalized_tips is None else np.asarray(normalized_tips, dtype=np.float32),
            "mapped_qpos": np.asarray(mapped_qpos, dtype=np.float32),
            "refined_qpos": np.asarray(refined_qpos, dtype=np.float32),
            "output_qpos": np.asarray(output_qpos, dtype=np.float32),
            "timings_ms": {key: float(value) for key, value in timings_ms.items()},
            "contact": contact or {},
            "timepoints_s": {key: float(value) for key, value in (timepoints_s or {}).items()},
            "sender_ts_ns": None if sender_ts_ns is None else int(sender_ts_ns),
        })

    @staticmethod
    def _percentiles(values: list[float]) -> dict[str, float]:
        if not values:
            return {"p50": float("nan"), "p95": float("nan")}
        return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}

    def close(self, *, counters: RealtimeCounters, extra_summary: dict[str, Any] | None = None) -> Path:
        self.path.mkdir(parents=True, exist_ok=False)
        timing_keys = sorted({key for frame in self._frames for key in frame["timings_ms"]})
        def stack(name: str, fallback_shape: tuple[int, ...] = ()) -> np.ndarray:
            if not self._frames:
                return np.empty((0, *fallback_shape), dtype=np.float32)
            values = [frame[name] for frame in self._frames]
            if any(value is None for value in values):
                return np.asarray(values, dtype=object)
            return np.stack(values)
        np.savez_compressed(
            self.path / "frames.npz",
            timestamp_s=np.asarray([frame["timestamp_s"] for frame in self._frames], dtype=np.float64),
            raw_points=stack("raw_points", (21, 3)),
            normalized_tips=stack("normalized_tips"),
            mapped_qpos=stack("mapped_qpos"),
            refined_qpos=stack("refined_qpos"),
            output_qpos=stack("output_qpos"),
            contact_json=np.asarray([json.dumps(frame["contact"], sort_keys=True) for frame in self._frames]),
            t_recv_s=np.asarray([frame["timepoints_s"].get("t_recv", np.nan) for frame in self._frames], dtype=np.float64),
            t_start_s=np.asarray([frame["timepoints_s"].get("t_start", np.nan) for frame in self._frames], dtype=np.float64),
            t_map_s=np.asarray([frame["timepoints_s"].get("t_map", np.nan) for frame in self._frames], dtype=np.float64),
            t_out_s=np.asarray([frame["timepoints_s"].get("t_out", np.nan) for frame in self._frames], dtype=np.float64),
            t_render_s=np.asarray([frame["timepoints_s"].get("t_render", np.nan) for frame in self._frames], dtype=np.float64),
            sender_ts_ns=np.asarray([-1 if frame["sender_ts_ns"] is None else frame["sender_ts_ns"] for frame in self._frames], dtype=np.int64),
            **{f"timing_{key}_ms": np.asarray([frame["timings_ms"].get(key, np.nan) for frame in self._frames], dtype=np.float64) for key in timing_keys},
        )
        if self._frozen_frames:
            def frozen_stack(name: str) -> np.ndarray:
                values = [frame[name] for frame in self._frozen_frames]
                if any(value is None for value in values):
                    return np.asarray(values, dtype=object)
                return np.stack(values)
            np.savez_compressed(
                self.path / "frozen_frames.npz",
                timestamp_s=np.asarray([frame["timestamp_s"] for frame in self._frozen_frames], dtype=np.float64),
                raw_points=frozen_stack("raw_points"),
                normalized_tips=frozen_stack("normalized_tips"),
                mapped_qpos=frozen_stack("mapped_qpos"),
                output_qpos=frozen_stack("output_qpos"),
            )
        summary: dict[str, Any] = {
            "frames": len(self._frames),
            "counters": asdict(counters),
            "timings_ms": {
                key: self._percentiles([frame["timings_ms"].get(key, np.nan) for frame in self._frames])
                for key in timing_keys
            },
        }
        if extra_summary:
            summary.update(extra_summary)
        with (self.path / "summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return self.path
