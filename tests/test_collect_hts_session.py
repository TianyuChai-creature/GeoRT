from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "geort" / "mocap" / "collect_hts_session.py"
spec = importlib.util.spec_from_file_location("collect_hts_session", MODULE_PATH)
assert spec is not None and spec.loader is not None
collect_hts_session = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = collect_hts_session
spec.loader.exec_module(collect_hts_session)

TimedSegment = collect_hts_session.TimedSegment
build_session_paths = collect_hts_session.build_session_paths
collect_timed_segment = collect_hts_session.collect_timed_segment
save_session_outputs = collect_hts_session.save_session_outputs
validate_points = collect_hts_session.validate_points


def make_frame(value: float) -> np.ndarray:
    return np.full((21, 3), value, dtype=np.float32)


def test_build_session_paths_follow_pdf_names(tmp_path: Path) -> None:
    paths = build_session_paths(
        data_dir=tmp_path,
        name="hts",
        hand_side="right",
        session_id="20260703_quest3_v3",
    )

    assert paths.rest == tmp_path / "hts_right_20260703_quest3_v3_rest.npy"
    assert paths.motion == tmp_path / "hts_right_20260703_quest3_v3.npy"
    assert paths.metadata == tmp_path / "hts_right_20260703_quest3_v3.json"


def test_validate_points_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"Expected HTS frame shape"):
        validate_points(np.zeros((20, 3), dtype=np.float32))


def test_collect_timed_segment_records_until_duration_expires() -> None:
    frames = [make_frame(i) for i in range(5)]
    times = iter([10.0, 10.1, 10.2, 10.3, 10.4, 10.6])

    result = collect_timed_segment(
        segment_name="rest",
        points_iter=iter(frames),
        duration_s=0.25,
        now=lambda: next(times),
    )

    assert result.frames.shape == (3, 21, 3)
    assert result.summary.name == "rest"
    assert result.summary.requested_duration_s == 0.25
    assert result.summary.frame_count == 3
    assert result.summary.actual_duration_s == pytest.approx(0.3)
    assert result.summary.estimated_fps == pytest.approx(10.0)


def test_save_session_outputs_writes_arrays_and_metadata(tmp_path: Path) -> None:
    paths = build_session_paths(
        data_dir=tmp_path,
        name="hts",
        hand_side="right",
        session_id="session_a",
    )
    rest = TimedSegment.from_frames(
        name="rest",
        frames=[make_frame(1.0), make_frame(2.0)],
        requested_duration_s=30.0,
        started_monotonic_s=1.0,
        ended_monotonic_s=31.0,
    )
    motion = TimedSegment.from_frames(
        name="motion",
        frames=[make_frame(3.0), make_frame(4.0), make_frame(5.0)],
        requested_duration_s=900.0,
        started_monotonic_s=40.0,
        ended_monotonic_s=940.0,
    )

    save_session_outputs(
        paths=paths,
        rest=rest,
        motion=motion,
        session_id="session_a",
        name="hts",
        hand_side="right",
        transport="udp",
        host="0.0.0.0",
        port=9000,
        timeout_s=1.0,
        operator="tester",
        device="Quest 3",
        firmware="v1",
        notes="unit test",
    )

    assert np.load(paths.rest).shape == (2, 21, 3)
    assert np.load(paths.motion).shape == (3, 21, 3)
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["id"] == "session_a"
    assert metadata["device"] == "Quest 3"
    assert metadata["segments"]["rest"]["frame_count"] == 2
    assert metadata["segments"]["motion"]["frame_count"] == 3
