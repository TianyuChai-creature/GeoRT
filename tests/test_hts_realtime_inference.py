import importlib
import sys
import types

import numpy as np


def load_realtime_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "sapien", types.ModuleType("sapien"))

    geort_stub = types.ModuleType("geort")
    geort_stub.get_config = lambda *_args, **_kwargs: {}
    geort_stub.load_model = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "geort", geort_stub)
    export_stub = types.ModuleType("geort.export")
    export_stub.resolve_checkpoint_dir = lambda value: value
    monkeypatch.setitem(sys.modules, "geort.export", export_stub)

    env_stub = types.ModuleType("geort.env")
    hand_stub = types.ModuleType("geort.env.hand")
    hand_stub.HandKinematicModel = object
    monkeypatch.setitem(sys.modules, "geort.env", env_stub)
    monkeypatch.setitem(sys.modules, "geort.env.hand", hand_stub)

    mocap_stub = types.ModuleType("geort.mocap")
    mocap_stub.__path__ = ["geort/mocap"]
    right_mocap_stub = types.ModuleType("geort.mocap.hts_right_mocap")
    right_mocap_stub.EXPECTED_HTS_LANDMARKS = 21
    right_mocap_stub.iter_hts_points = lambda **_kwargs: iter(())
    monkeypatch.setitem(sys.modules, "geort.mocap", mocap_stub)
    monkeypatch.setitem(sys.modules, "geort.mocap.hts_right_mocap", right_mocap_stub)

    utils_stub = types.ModuleType("geort.utils")
    config_utils_stub = types.ModuleType("geort.utils.config_utils")
    config_utils_stub.parse_config_keypoint_info = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "geort.utils", utils_stub)
    monkeypatch.setitem(sys.modules, "geort.utils.config_utils", config_utils_stub)

    sys.modules.pop("geort.mocap.hts_realtime_inference", None)
    return importlib.import_module("geort.mocap.hts_realtime_inference")


class FakeModel:
    def forward(self, points):
        return np.array([0.5, -0.5, 2.0], dtype=np.float32)


class FakeHand:
    def __init__(self):
        self.qpos_targets = []

    def get_joint_limit(self):
        return (
            np.array([-1.0, -0.75, -1.0], dtype=np.float32),
            np.array([0.55, 1.0, 2.2], dtype=np.float32),
        )

    def set_qpos_target(self, qpos):
        self.qpos_targets.append(np.asarray(qpos, dtype=np.float32))


class FakeViewerEnv:
    def update(self):
        return True


def test_realtime_inference_scales_and_clamps_qpos_targets(monkeypatch):
    realtime = load_realtime_module(monkeypatch)
    hand = FakeHand()
    points = [np.zeros((21, 3), dtype=np.float32)]

    processed = realtime.run_realtime_inference(
        model=FakeModel(),
        hand=hand,
        viewer_env=FakeViewerEnv(),
        points_iter=points,
        viewer_updates_per_frame=1,
        qpos_scale=1.2,
        fps_interval=0,
    )

    assert processed == 1
    assert len(hand.qpos_targets) == 1
    np.testing.assert_allclose(
        hand.qpos_targets[0],
        np.array([0.55, -0.6, 2.2], dtype=np.float32),
    )


def test_realtime_qpos_scale_defaults_to_one_for_c2_parity(monkeypatch):
    realtime = load_realtime_module(monkeypatch)

    args = realtime.build_arg_parser().parse_args([])

    assert args.qpos_scale == 1.0
    assert args.checkpoint == "checkpoint/custom_right_2026-07-17_12-21-39_c2b_s42"
    assert args.freeze_key == "f"


def test_realtime_contact_refinement_cli_defaults_are_opt_in(monkeypatch):
    realtime = load_realtime_module(monkeypatch)

    args = realtime.build_arg_parser().parse_args([])

    assert args.contact_refine == "off"
    assert args.contact_p_lo == 0.5
    assert args.contact_p_hi == 0.8
    assert args.contact_target_dist == 0.0
    assert args.contact_lambda == 1e-3
    assert args.contact_refine_steps == 40


def test_realtime_contact_refinement_cli_forwards_explicit_values(monkeypatch):
    realtime = load_realtime_module(monkeypatch)

    args = realtime.build_arg_parser().parse_args([
        "--contact_refine", "on",
        "--contact-model-path", "checkpoint/contact_right_d1_full/contact_models.pth",
        "--contact-p-lo", "0.45",
        "--contact-p-hi", "0.75",
        "--contact-target-dist", "0.003",
        "--contact-lambda", "0.2",
        "--contact-refine-steps", "24",
    ])

    assert args.contact_refine == "on"
    assert str(args.contact_model_path).endswith("contact_models.pth")
    assert args.contact_p_lo == 0.45
    assert args.contact_p_hi == 0.75
    assert args.contact_target_dist == 0.003
    assert args.contact_lambda == 0.2
    assert args.contact_refine_steps == 24



def test_stage_two_and_three_use_sapien_with_fixed_contact_mode(monkeypatch):
    realtime = load_realtime_module(monkeypatch)

    assert realtime.validate_stage_contact_mode("1", "off") == "off"
    assert realtime.validate_stage_contact_mode("2", "off") == "off"
    assert realtime.validate_stage_contact_mode("3", "on") == "on"


class FakeFreezeWindow:
    def key_down(self, key):
        return key == "f"


class FakeFreezeViewerEnv:
    def __init__(self):
        self.viewer = type("Viewer", (), {"window": FakeFreezeWindow()})()

    def update(self):
        return True


def test_realtime_freeze_key_writes_current_frame(monkeypatch, tmp_path):
    from geort.mocap.realtime_runtime import SessionRecorder

    realtime = load_realtime_module(monkeypatch)
    hand = FakeHand()
    buffer = realtime.LatestPointBuffer()
    buffer.put(np.zeros((21, 3), dtype=np.float32))
    recorder = SessionRecorder(tmp_path)

    processed = realtime.run_realtime_viewer_loop(
        model=FakeModel(), hand=hand, viewer_env=FakeFreezeViewerEnv(), point_buffer=buffer,
        max_frames=1, fps_interval=0, session_recorder=recorder, freeze_key="f",
    )
    path = recorder.close(counters=realtime.RealtimeSafetyController(
        lower=np.full(3, -1.0), upper=np.full(3, 1.0), initial_qpos=np.zeros(3),
    ).counters)

    assert processed == 1
    with np.load(path / "frozen_frames.npz") as frozen:
        np.testing.assert_allclose(frozen["raw_points"], np.zeros((1, 21, 3)))
        np.testing.assert_allclose(frozen["output_qpos"], hand.qpos_targets[-1][None, :])


def test_realtime_c2b_sha_guard_rejects_mismatch(monkeypatch):
    realtime = load_realtime_module(monkeypatch)

    assert realtime.require_c2b_s42_sha(realtime.C2B_S42_LAST_PTH_SHA256) == realtime.C2B_S42_LAST_PTH_SHA256
    with np.testing.assert_raises_regex(ValueError, "SHA256"):
        realtime.require_c2b_s42_sha("different")


def test_realtime_latency_diagnostic_cli_defaults(monkeypatch):
    realtime = load_realtime_module(monkeypatch)

    args = realtime.build_arg_parser().parse_args([])

    assert args.render_mode == "inline"
    assert args.diagnostic_rate_limit_bypass is False


def test_latest_point_buffer_preserves_receive_timestamps(monkeypatch):
    realtime = load_realtime_module(monkeypatch)
    buffer = realtime.LatestPointBuffer()
    buffer.put(np.zeros((21, 3), dtype=np.float32), recv_ts_s=12.5, sender_ts_ns=99)

    frame = buffer.get_latest()

    assert frame.recv_ts_s == 12.5
    assert frame.sender_ts_ns == 99
    np.testing.assert_allclose(frame.points, np.zeros((21, 3), dtype=np.float32))


def test_receiver_thread_stamps_packet_arrival_not_sdk_frame_timestamp(monkeypatch):
    realtime = load_realtime_module(monkeypatch)
    monkeypatch.setattr(realtime.time, "monotonic", lambda: 42.5)
    buffer = realtime.LatestPointBuffer()

    packet = type("Packet", (), {
        "points": np.zeros((21, 3), dtype=np.float32),
        "recv_ts_ns": 7,
        "source_ts_ns": 99,
    })()
    receiver = realtime.start_point_receiver(iter((packet,)), buffer)
    receiver.join(timeout=1.0)
    frame = buffer.get_latest()

    assert frame.recv_ts_s == 42.5
    assert frame.sender_ts_ns == 99


def test_recorded_replay_preserves_frame_order_and_receive_intervals(monkeypatch, tmp_path):
    realtime = load_realtime_module(monkeypatch)
    session = tmp_path / "session"
    session.mkdir()
    np.savez(
        session / "frames.npz",
        raw_points=np.stack((np.zeros((21, 3), dtype=np.float32), np.ones((21, 3), dtype=np.float32))),
        t_recv_s=np.array([10.0, 10.025], dtype=np.float64),
    )
    sleeps = []

    replayed = list(realtime.iter_recorded_replay(session, sleep_fn=sleeps.append))

    assert len(replayed) == 2
    np.testing.assert_allclose(replayed[0], 0.0)
    np.testing.assert_allclose(replayed[1], 1.0)
    np.testing.assert_allclose(sleeps, [0.025])


def test_replay_buffer_preserves_every_recorded_frame(monkeypatch):
    realtime = load_realtime_module(monkeypatch)
    buffer = realtime.LatestPointBuffer(preserve_order=True)
    first = np.zeros((21, 3), dtype=np.float32)
    second = np.ones((21, 3), dtype=np.float32)

    buffer.put(first, recv_ts_s=1.0)
    buffer.put(second, recv_ts_s=2.0)

    assert buffer.get_latest().recv_ts_s == 1.0
    assert buffer.get_latest().recv_ts_s == 2.0
    assert buffer.get_latest() is None


def test_viewer_loop_stops_at_explicit_max_duration(monkeypatch):
    realtime = load_realtime_module(monkeypatch)
    buffer = realtime.LatestPointBuffer()
    processed = realtime.run_realtime_viewer_loop(
        model=FakeModel(), hand=FakeHand(), viewer_env=FakeViewerEnv(), point_buffer=buffer,
        max_duration_s=0.0, fps_interval=0,
    )
    assert processed == 0


def test_viewer_loop_can_process_inputs_without_per_frame_render(monkeypatch):
    realtime = load_realtime_module(monkeypatch)

    class CountingViewer:
        def __init__(self):
            self.update_calls = 0

        def update(self):
            self.update_calls += 1
            return True

    buffer = realtime.LatestPointBuffer(preserve_order=True)
    for _ in range(3):
        buffer.put(np.zeros((21, 3), dtype=np.float32))
    viewer = CountingViewer()

    processed = realtime.run_realtime_viewer_loop(
        model=FakeModel(), hand=FakeHand(), viewer_env=viewer, point_buffer=buffer,
        max_frames=3, fps_interval=0, render_hz=0.0,
    )

    assert processed == 3
    assert viewer.update_calls == 0


def test_realtime_render_hz_defaults_to_thirty(monkeypatch):
    realtime = load_realtime_module(monkeypatch)

    assert realtime.build_arg_parser().parse_args([]).render_hz == 30.0


def test_mapping_loop_publishes_each_safe_output_without_viewer(monkeypatch):
    realtime = load_realtime_module(monkeypatch)

    class CommandHand(FakeHand):
        def __init__(self):
            super().__init__()
            self.published = []

        def set_qpos_target(self, qpos):
            self.published.append(np.asarray(qpos, dtype=np.float32))

    buffer = realtime.LatestPointBuffer(preserve_order=True)
    for _ in range(2):
        buffer.put(np.zeros((21, 3), dtype=np.float32))
    records = []
    hand = CommandHand()

    processed = realtime.run_realtime_viewer_loop(
        model=FakeModel(), hand=hand, viewer_env=FakeViewerEnv(), point_buffer=buffer,
        max_frames=2, fps_interval=0, render_hz=0.0, accepted_frame_callback=records.append,
    )

    assert processed == 2
    assert len(hand.published) == 2
    assert len(records) == 2
    assert all("output" in record and "timepoints_s" in record for record in records)
