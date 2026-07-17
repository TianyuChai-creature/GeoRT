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
