import json

import numpy as np


def test_safety_controller_ramps_rate_limits_and_clamps():
    from geort.mocap.realtime_runtime import RealtimeSafetyController

    controller = RealtimeSafetyController(
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 1.0]),
        initial_qpos=np.array([0.0, 0.0]),
        ramp_frames=2,
        max_joint_step=0.6,
        watchdog_s=0.2,
    )
    first = controller.accept(np.array([4.0, -4.0]), now_s=0.0)
    second = controller.accept(np.array([4.0, -4.0]), now_s=0.01)

    np.testing.assert_allclose(first, [0.5, -0.5])
    np.testing.assert_allclose(second, [1.0, -1.0])
    assert controller.counters.rate_limited == 0


def test_safety_controller_holds_nonfinite_and_watchdog_then_reramps():
    from geort.mocap.realtime_runtime import RealtimeSafetyController

    controller = RealtimeSafetyController(
        lower=np.array([-1.0]), upper=np.array([1.0]), initial_qpos=np.array([0.0]),
        ramp_frames=2, max_joint_step=1.0, watchdog_s=0.2,
    )
    controller.accept(np.array([1.0]), now_s=0.0)
    held_nan = controller.accept(np.array([np.nan]), now_s=0.01)
    held_timeout = controller.watchdog(now_s=0.21)
    resumed = controller.accept(np.array([-1.0]), now_s=0.22)

    np.testing.assert_allclose(held_nan, [0.5])
    np.testing.assert_allclose(held_timeout, [0.5])
    np.testing.assert_allclose(resumed, [-0.25])
    assert controller.counters.nan_input == 1
    assert controller.counters.watchdog == 1


def test_safety_controller_estop_holds_and_qpos_scale_one_still_clamps():
    from geort.mocap.realtime_runtime import RealtimeSafetyController, scale_and_clamp_qpos

    controller = RealtimeSafetyController(
        lower=np.array([-1.0]), upper=np.array([1.0]), initial_qpos=np.array([0.0]),
        ramp_frames=1, max_joint_step=2.0, watchdog_s=0.2,
    )
    controller.accept(np.array([0.4]), now_s=0.0)
    controller.set_estop(True)
    held = controller.accept(np.array([-1.0]), now_s=0.01)

    np.testing.assert_allclose(held, [0.4])
    assert controller.counters.estop == 1
    np.testing.assert_allclose(
        scale_and_clamp_qpos(np.array([3.0]), np.array([-1.0]), np.array([1.0]), 1.0), [1.0]
    )



def test_session_recorder_writes_required_arrays_and_summary(tmp_path):
    from geort.mocap.realtime_runtime import RealtimeCounters, SessionRecorder

    recorder = SessionRecorder(tmp_path)
    recorder.append(
        timestamp_s=1.0,
        raw_points=np.zeros((21, 3)),
        normalized_tips=np.zeros((5, 3)),
        mapped_qpos=np.zeros(2),
        refined_qpos=np.ones(2),
        output_qpos=np.ones(2) * 0.5,
        timings_ms={"mapping": 1.5, "contact": 0.25},
        contact={"weight": 0.0},
    )
    path = recorder.close(counters=RealtimeCounters(nan_input=1), extra_summary={"stage": "1"})

    with np.load(path / "frames.npz") as frames:
        assert set(("raw_points", "normalized_tips", "mapped_qpos", "refined_qpos", "output_qpos")) <= set(frames.files)
    assert '\"nan_input\": 1' in (path / "summary.json").read_text()


def test_session_recorder_writes_frozen_frame_and_provenance_summary(tmp_path):
    from geort.mocap.realtime_runtime import RealtimeCounters, SessionRecorder

    recorder = SessionRecorder(tmp_path)
    recorder.freeze_frame(
        timestamp_s=2.0,
        raw_points=np.full((21, 3), 2.0),
        normalized_tips=np.full((5, 3), 3.0),
        mapped_qpos=np.full(20, 4.0),
        output_qpos=np.full(20, 5.0),
    )
    path = recorder.close(
        counters=RealtimeCounters(),
        extra_summary={
            "smoothing_alpha": None,
            "checkpoint_sha256": "expected-sha",
            "git_hash": "expected-git",
            "command": "python -m geort.mocap.hts_realtime_inference",
        },
    )

    with np.load(path / "frozen_frames.npz") as frozen:
        np.testing.assert_allclose(frozen["raw_points"], np.full((1, 21, 3), 2.0))
        np.testing.assert_allclose(frozen["output_qpos"], np.full((1, 20), 5.0))
    summary = json.loads((path / "summary.json").read_text())
    assert summary["smoothing_alpha"] is None
    assert summary["checkpoint_sha256"] == "expected-sha"
    assert summary["git_hash"] == "expected-git"


def test_session_recorder_writes_five_latency_timestamps(tmp_path):
    from geort.mocap.realtime_runtime import RealtimeCounters, SessionRecorder

    recorder = SessionRecorder(tmp_path)
    recorder.append(
        timestamp_s=5.0,
        raw_points=np.zeros((21, 3)), normalized_tips=np.zeros((5, 3)),
        mapped_qpos=np.zeros(2), refined_qpos=np.zeros(2), output_qpos=np.zeros(2),
        timings_ms={}, contact={},
        timepoints_s={"t_recv": 1.0, "t_start": 1.1, "t_map": 1.2, "t_out": 1.3, "t_render": 1.4},
        sender_ts_ns=123456789,
    )
    path = recorder.close(counters=RealtimeCounters())

    with np.load(path / "frames.npz") as frames:
        np.testing.assert_allclose(frames["t_recv_s"], [1.0])
        np.testing.assert_allclose(frames["t_render_s"], [1.4])
        np.testing.assert_array_equal(frames["sender_ts_ns"], [123456789])


def test_safety_bypass_rate_limit_keeps_hard_clamp_and_ramp():
    from geort.mocap.realtime_runtime import RealtimeSafetyController

    controller = RealtimeSafetyController(
        lower=np.array([-1.0]), upper=np.array([1.0]), initial_qpos=np.array([0.0]),
        ramp_frames=2, max_joint_step=0.05, watchdog_s=0.2,
    )
    first = controller.accept(np.array([3.0]), now_s=0.0, bypass_rate_limit=True)
    second = controller.accept(np.array([3.0]), now_s=0.01, bypass_rate_limit=True)

    np.testing.assert_allclose(first, [0.5])
    np.testing.assert_allclose(second, [1.0])
    assert controller.counters.rate_limited == 0
