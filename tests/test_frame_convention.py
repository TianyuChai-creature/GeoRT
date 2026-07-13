from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_capture_module():
    module_path = ROOT / "geort" / "mocap" / "hts_right_mocap.py"
    spec = importlib.util.spec_from_file_location("hts_right_mocap", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_global_hand_basis_is_right_handed() -> None:
    module_path = ROOT / "geort" / "frame_convention.py"
    assert module_path.exists(), "geort.frame_convention is not implemented"
    spec = importlib.util.spec_from_file_location("frame_convention", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    basis = module.GLOBAL_HAND_BASIS

    module.validate_right_handed_basis(basis)

    x_axis, y_axis, z_axis = basis.T
    assert np.allclose(np.cross(x_axis, y_axis), z_axis)
    assert np.linalg.det(basis) == 1.0


def test_capture_uses_one_conversion_for_both_hand_sides() -> None:
    module = _load_capture_module()
    source = (ROOT / "geort" / "mocap" / "hts_right_mocap.py").read_text()

    assert "convert_hand_frame_unity_left_to_right" in source
    assert "if side ==" not in source
    assert module.CAPTURE_COORDINATE_CONVENTION == "geort_right_handed_global"


def test_step4_does_not_introduce_per_finger_frames() -> None:
    assert not (ROOT / "geort" / "frames.py").exists()
    visualizer = (ROOT / "geort" / "mocap" / "visualize_frames.py").read_text()

    assert "GLOBAL_HAND_BASIS" in visualizer
    assert "finger_frame" not in visualizer
