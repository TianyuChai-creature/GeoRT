from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geort.contact.auto_label_contacts import (
    PAIR_LANDMARKS,
    PAIR_NAMES,
    ContactLabelConfig,
    auto_label_contacts,
    build_arg_parser,
    compute_tip_distances,
    write_contact_label_outputs,
)


def _distance_trace() -> np.ndarray:
    """A low-distance contact platform, an ambiguous band, and open-hand frames."""
    contact = 0.008 + np.linspace(-0.0004, 0.0004, 600)
    ambiguous = np.linspace(0.014, 0.020, 160)
    separated = 0.032 + np.linspace(-0.001, 0.001, 700)
    return np.concatenate((contact, ambiguous, separated))


def _frames_from_pair_distances(distance: np.ndarray) -> np.ndarray:
    frames = np.zeros((len(distance), 21, 3), dtype=np.float64)
    for target in (8, 12, 16, 20):
        frames[:, target, 0] = distance
    return frames


def test_compute_tip_distances_uses_the_fixed_thumb_pairs() -> None:
    frames = _frames_from_pair_distances(np.array([0.01, 0.02]))

    distances = compute_tip_distances(frames)

    assert PAIR_NAMES == ("thumb_index", "thumb_middle", "thumb_ring", "thumb_pinky")
    assert PAIR_LANDMARKS == ((4, 8), (4, 12), (4, 16), (4, 20))
    np.testing.assert_allclose(distances, [[0.01] * 4, [0.02] * 4])


def test_parser_uses_the_contact_config_defaults() -> None:
    args = build_arg_parser().parse_args(
        [
            "--histogram-bin-mm", "2.0",
            "--smoothing-bins", "7",
            "--platform-search-quantile", "0.40",
            "--minimum-search-span-mm", "35.0",
        ]
    )

    assert args.histogram_bin_mm == pytest.approx(2.0)
    assert args.smoothing_bins == 7
    assert args.platform_search_quantile == pytest.approx(0.40)
    assert args.minimum_search_span_mm == pytest.approx(35.0)
    assert args.ambiguity_band_mm == pytest.approx(10.0)
    assert args.min_cluster_frames == 50
    assert not hasattr(args, "minimum_gap_mm")
    assert not hasattr(args, "gap_multiplier")


def test_auto_label_contacts_keeps_platform_and_open_frames_and_discards_band() -> None:
    trace = _distance_trace()
    labels = auto_label_contacts(
        _frames_from_pair_distances(trace),
        config=ContactLabelConfig(holdout_fraction=0.20),
    )

    first_pair = labels.pairs[0]
    assert first_pair.name == "thumb_index"
    assert first_pair.positive_count >= 500
    assert first_pair.negative_count >= 500
    assert np.all(first_pair.labels[:600] == 1)
    assert np.all(first_pair.labels[600:760] == -1)
    assert np.all(first_pair.labels[760:] == 0)
    assert first_pair.positive_threshold < first_pair.negative_threshold
    assert first_pair.discarded_segments == ((600, 760),)


def test_auto_label_contacts_detects_a_lower_platform_without_empty_distance_gaps() -> None:
    rng = np.random.default_rng(7)
    platform = np.clip(rng.normal(0.010, 0.002, 5000), 0.004, 0.016)
    continuous_motion = np.linspace(0.012, 0.080, 20000)

    labels = auto_label_contacts(
        _frames_from_pair_distances(np.concatenate((platform, continuous_motion))),
        config=ContactLabelConfig(holdout_fraction=0.20),
    )

    first_pair = labels.pairs[0]
    assert first_pair.positive_count >= 500
    assert first_pair.negative_count >= 500
    assert 0.008 <= first_pair.positive_threshold <= 0.020


def test_auto_label_contacts_uses_one_contiguous_trailing_held_out_block() -> None:
    labels = auto_label_contacts(
        _frames_from_pair_distances(_distance_trace()),
        config=ContactLabelConfig(holdout_fraction=0.20),
    )

    first_pair = labels.pairs[0]
    kept_indices = first_pair.frame_indices[first_pair.labels >= 0]
    held_out = first_pair.held_out[first_pair.labels >= 0]
    assert held_out.any()
    assert not held_out[0]
    assert np.all(np.diff(kept_indices[held_out]) == 1)
    assert kept_indices[held_out][0] > kept_indices[~held_out][-1]


def test_writer_emits_inspection_artifacts_and_refuses_implicit_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hts_right.npy"
    np.save(source, _frames_from_pair_distances(_distance_trace()))
    labels = auto_label_contacts(np.load(source), config=ContactLabelConfig())

    paths = write_contact_label_outputs(
        labels,
        input_path=source,
        hand_side="right",
        output_path=tmp_path / "data" / "contact_labels_right.npz",
        report_dir=tmp_path / "outputs" / "contacts",
        config=ContactLabelConfig(),
    )

    assert paths.npz_path.exists()
    assert paths.json_path.exists()
    assert paths.html_path.exists()
    with np.load(paths.npz_path) as bundle:
        assert bundle["features"].shape[1:] == (4, 6)
        assert bundle["labels"].shape[1:] == (4,)
        assert bundle["held_out"].shape[1:] == (4,)
    report = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert report["pairs"][0]["positive_count"] >= 500
    assert "thresholds" in report["pairs"][0]
    assert report["config"]["histogram_bin_m"] == pytest.approx(0.001)
    assert "gap_multiplier" not in report["config"]
    assert report["pairs"][0]["platform"]["peak_m"] < report["pairs"][0]["platform"]["valley_m"]
    assert "significant_gaps_m" not in report["pairs"][0]

    with pytest.raises(FileExistsError, match="--overwrite"):
        write_contact_label_outputs(
            labels,
            input_path=source,
            hand_side="right",
            output_path=paths.npz_path,
            report_dir=tmp_path / "outputs" / "contacts",
            config=ContactLabelConfig(),
        )
