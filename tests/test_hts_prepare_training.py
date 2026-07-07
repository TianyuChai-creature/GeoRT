from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
geort_pkg = types.ModuleType("geort")
geort_pkg.__path__ = [str(ROOT / "geort")]
mocap_pkg = types.ModuleType("geort.mocap")
mocap_pkg.__path__ = [str(ROOT / "geort" / "mocap")]
sys.modules.setdefault("geort", geort_pkg)
sys.modules.setdefault("geort.mocap", mocap_pkg)
MODULE_PATH = ROOT / "geort" / "mocap" / "hts_prepare_training.py"
spec = importlib.util.spec_from_file_location("hts_prepare_training", MODULE_PATH)
assert spec is not None and spec.loader is not None
hts_prepare_training = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hts_prepare_training
spec.loader.exec_module(hts_prepare_training)

append_fist_boost_frames = hts_prepare_training.append_fist_boost_frames
compute_fist_curl_score = hts_prepare_training.compute_fist_curl_score
compute_mcp_weighted_fist_curl_score = hts_prepare_training.compute_mcp_weighted_fist_curl_score


def make_frames_with_middle_flexion(angles: list[float]) -> np.ndarray:
    frames = np.zeros((len(angles), 21, 3), dtype=np.float32)
    for frame_idx, angle in enumerate(angles):
        for base in (5, 9, 13, 17):
            pip = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            mcp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            dip = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
            incoming = pip - dip
            rot = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            tip = dip + rot @ incoming
            frames[frame_idx, base] = mcp
            frames[frame_idx, base + 1] = pip
            frames[frame_idx, base + 2] = dip
            frames[frame_idx, base + 3] = tip
    return frames



def make_frames_with_joint_flexion(
    *,
    mcp_angles: list[float],
    pip_angles: list[float],
    dip_angles: list[float],
) -> np.ndarray:
    frames = np.zeros((len(mcp_angles), 21, 3), dtype=np.float32)
    frames[:, 0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for frame_idx, (mcp_angle, pip_angle, dip_angle) in enumerate(zip(mcp_angles, pip_angles, dip_angles)):
        for base in (5, 9, 13, 17):
            mcp = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            pip = mcp + np.array([np.cos(mcp_angle), np.sin(mcp_angle), 0.0], dtype=np.float32)
            dip = pip + np.array([np.cos(mcp_angle + pip_angle), np.sin(mcp_angle + pip_angle), 0.0], dtype=np.float32)
            tip = dip + np.array(
                [np.cos(mcp_angle + pip_angle + dip_angle), np.sin(mcp_angle + pip_angle + dip_angle), 0.0],
                dtype=np.float32,
            )

            frames[frame_idx, base] = mcp
            frames[frame_idx, base + 1] = pip
            frames[frame_idx, base + 2] = dip
            frames[frame_idx, base + 3] = tip
    return frames

def test_compute_fist_curl_score_is_lower_for_more_curled_frames() -> None:
    frames = make_frames_with_middle_flexion([3.0, 1.2])

    score = compute_fist_curl_score(frames)

    assert score[1] < score[0]


def test_append_fist_boost_frames_repeats_lowest_score_frames() -> None:
    frames = make_frames_with_middle_flexion([3.0, 2.5, 1.2, 1.1])

    boosted, report = append_fist_boost_frames(frames, top_fraction=0.5, repeat=3)

    assert boosted.shape[0] == 10
    assert report["selected_frames"] == 2
    assert report["added_frames"] == 6
    assert np.allclose(boosted[-6:-3], frames[3])
    assert np.allclose(boosted[-3:], frames[2])


def test_mcp_weighted_fist_curl_score_prefers_mcp_curl() -> None:
    frames = make_frames_with_joint_flexion(
        mcp_angles=[2.8, 1.2],
        pip_angles=[1.2, 2.8],
        dip_angles=[1.2, 2.8],
    )

    score = compute_mcp_weighted_fist_curl_score(frames, mcp_weight=2.0, pip_weight=1.0, dip_weight=0.7)

    assert score[1] < score[0]


def test_append_fist_boost_frames_can_use_mcp_weighted_score() -> None:
    frames = make_frames_with_joint_flexion(
        mcp_angles=[2.8, 2.6, 1.4, 1.2],
        pip_angles=[1.2, 1.4, 2.6, 2.8],
        dip_angles=[1.2, 1.4, 2.6, 2.8],
    )

    boosted, report = append_fist_boost_frames(
        frames,
        top_fraction=0.25,
        repeat=2,
        score_mode="mcp_weighted",
        mcp_weight=2.0,
        pip_weight=1.0,
        dip_weight=0.7,
    )

    assert boosted.shape[0] == 6
    assert report["score_mode"] == "mcp_weighted"
    assert report["score_weights"] == {"mcp": 2.0, "pip": 1.0, "dip": 0.7}
    assert np.allclose(boosted[-2:], frames[3])
