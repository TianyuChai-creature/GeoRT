from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import geort.anchor.mine_human_anchors as mine_command

from geort.anchor.mine_human_anchors import (
    build_arg_parser,
    build_report_payload,
    default_output_path,
    load_hts_frames,
    main,
    write_mining_outputs,
)
from geort.anchor.mining import LEVEL_FRACTIONS, MinedHumanAnchors


def _anchors() -> MinedHumanAnchors:
    rows = 50
    frames = np.arange(rows * 21 * 3, dtype=np.float64).reshape(rows, 21, 3)
    finger_indices = np.repeat(np.arange(5, dtype=np.int64), 10)
    finger_names = np.repeat(
        np.array(["thumb", "index", "middle", "ring", "pinky"]), 10
    )
    anchor_types = np.tile(np.repeat(np.array(["lateral", "bending"]), 5), 5)
    levels = np.tile(np.arange(5, dtype=np.int64), 10)
    return MinedHumanAnchors(
        human_frames=frames,
        human_points=frames[:, 4, :],
        source_indices=np.arange(rows, dtype=np.int64),
        finger_indices=finger_indices,
        finger_names=finger_names,
        anchor_types=anchor_types,
        levels=levels,
        trajectory_t=np.tile(LEVEL_FRACTIONS, 10),
        target_parameters=np.linspace(-0.2, 0.8, rows),
        observed_parameters=np.linspace(-0.19, 0.79, rows),
        candidate_counts=np.repeat(17, rows),
        support_counts=np.repeat(5, rows),
        group_metadata={
            f"{finger}:{anchor_type}": {"candidate_count": 17}
            for finger in ("thumb", "index", "middle", "ring", "pinky")
            for anchor_type in ("lateral", "bending")
        },
    )


def test_load_hts_frames_validates_shape(tmp_path: Path) -> None:
    good = tmp_path / "good.npy"
    bad = tmp_path / "bad.npy"
    np.save(good, np.zeros((7, 21, 3), dtype=np.float32))
    np.save(bad, np.zeros((7, 20, 3), dtype=np.float32))

    loaded = load_hts_frames(good)

    assert loaded.shape == (7, 21, 3)
    with pytest.raises(ValueError, match=r"\[T, 21, 3\]"):
        load_hts_frames(bad)


def test_default_output_uses_hand_side() -> None:
    assert default_output_path("right") == Path("data/anchors_human_right.npz")
    assert default_output_path("left") == Path("data/anchors_human_left.npz")
    with pytest.raises(ValueError, match="hand_side"):
        default_output_path("both")


def test_report_payload_has_ten_groups_and_source_sha256(tmp_path: Path) -> None:
    source = tmp_path / "hts_right.npy"
    source.write_bytes(b"D1 raw bytes")

    payload = build_report_payload(
        _anchors(),
        input_path=source,
        hand_side="right",
        config={"endpoint_low": 0.02, "endpoint_high": 0.98},
    )

    assert payload["anchor_count"] == 50
    assert payload["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(payload["groups"]) == 10
    assert payload["groups"][0]["finger"] == "thumb"
    assert payload["groups"][0]["anchor_type"] == "lateral"
    assert payload["groups"][0]["levels"] == [0, 1, 2, 3, 4]
    assert payload["groups"][0]["source_indices"] == [0, 1, 2, 3, 4]
    assert json.loads(json.dumps(payload)) == payload


def test_output_writer_is_atomic_in_contract_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hts_right.npy"
    np.save(source, np.zeros((8, 21, 3), dtype=np.float32))
    output = tmp_path / "data" / "anchors_human_right.npz"
    report_dir = tmp_path / "outputs" / "anchors"

    paths = write_mining_outputs(
        _anchors(),
        input_path=source,
        hand_side="right",
        output_path=output,
        report_dir=report_dir,
        config={"endpoint_low": 0.02, "endpoint_high": 0.98},
    )

    assert paths.npz_path == output
    assert paths.json_path == report_dir / "anchors_human_right.json"
    assert paths.html_path == report_dir / "anchors_human_right.html"
    with np.load(paths.npz_path) as bundle:
        assert bundle["human_frames"].shape == (50, 21, 3)
        assert bundle["finger_indices"].tolist() == _anchors().finger_indices.tolist()
        assert json.loads(str(bundle["metadata_json"]))["anchor_count"] == 50
    assert json.loads(paths.json_path.read_text(encoding="utf-8"))["anchor_count"] == 50
    assert "plotly" in paths.html_path.read_text(encoding="utf-8").lower()

    with pytest.raises(FileExistsError, match="--overwrite"):
        write_mining_outputs(
            _anchors(),
            input_path=source,
            hand_side="right",
            output_path=output,
            report_dir=report_dir,
            config={},
        )


def test_parser_exposes_approved_mining_controls() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--input",
            "data/hts_left.npy",
            "--hand-side",
            "left",
            "--endpoint-low",
            "0.01",
            "--endpoint-high",
            "0.99",
            "--overwrite",
        ]
    )

    assert args.input == Path("data/hts_left.npy")
    assert args.hand_side == "left"
    assert args.endpoint_low == 0.01
    assert args.endpoint_high == 0.99
    assert args.overwrite is True


def test_main_runs_synthetic_input_to_all_three_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "hts_right.npy"
    np.save(source, np.zeros((8, 21, 3), dtype=np.float32))
    monkeypatch.setattr(
        mine_command,
        "mine_human_anchor_records",
        lambda frames, **kwargs: _anchors(),
    )

    paths = main(
        [
            "--input",
            str(source),
            "--hand-side",
            "right",
            "--output",
            str(tmp_path / "anchors.npz"),
            "--report-dir",
            str(tmp_path / "report"),
        ]
    )

    assert paths.npz_path.exists()
    assert paths.json_path.exists()
    assert paths.html_path.exists()
