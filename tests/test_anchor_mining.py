from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

import geort.anchor.mining as mining

from geort.anchor.mining import (
    CandidateFilterError,
    filter_motion_candidates,
    mine_human_anchor_records,
    robust_angle_targets,
    select_thumb_arc_medoids,
    select_level_medoids,
)


def test_robust_targets_are_geometric_not_empirical_cdf_levels() -> None:
    values = np.concatenate(
        (
            np.linspace(-1.2, -0.8, 40),
            np.zeros(921),
            np.linspace(2.8, 3.2, 40),
        )
    )

    result = robust_angle_targets(values)

    expected_endpoints = np.quantile(values, (0.02, 0.98))
    expected_targets = np.linspace(*expected_endpoints, 5)
    empirical_levels = np.quantile(values, (0.02, 0.26, 0.50, 0.74, 0.98))
    assert np.allclose(result.endpoints, expected_endpoints)
    assert np.allclose(result.targets, expected_targets)
    assert expected_endpoints[1] != pytest.approx(-expected_endpoints[0])
    assert np.max(np.abs(result.targets - empirical_levels)) > 0.5


def test_robust_targets_use_only_the_approved_five_level_fractions() -> None:
    result = robust_angle_targets(
        np.linspace(-0.4, 1.1, 101),
        endpoint_quantiles=(0.1, 0.9),
    )

    assert mining.LEVEL_FRACTIONS.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert np.allclose(
        result.targets,
        result.endpoints[0]
        + mining.LEVEL_FRACTIONS * np.ptp(result.endpoints),
    )
    metadata = result.endpoint_support
    assert metadata == {
        "sample_count": 101,
        "endpoint_quantiles": [0.1, 0.9],
        "endpoint_values": result.endpoints.tolist(),
        "lower_rejected_count": 10,
        "upper_rejected_count": 10,
        "retained_count": 81,
    }
    assert json.loads(json.dumps(metadata)) == metadata


def test_approved_level_fractions_are_not_publicly_customizable() -> None:
    with pytest.raises(ValueError, match="read-only"):
        mining.LEVEL_FRACTIONS[1] = 0.1
    assert "level_fractions" not in inspect.signature(robust_angle_targets).parameters
    with pytest.raises(TypeError, match="level_fractions"):
        robust_angle_targets(
            np.linspace(-1.0, 1.0, 21),
            level_fractions=(0.0, 0.1, 0.4, 0.8, 1.0),
        )


def test_robust_target_result_arrays_are_defensively_read_only() -> None:
    result = robust_angle_targets(np.linspace(-1.0, 1.0, 21))

    with pytest.raises(ValueError, match="read-only"):
        result.targets[0] = 10.0
    metadata = result.endpoint_support
    metadata["sample_count"] = -1
    metadata["endpoint_values"][0] = -99.0
    assert result.endpoint_support["sample_count"] == 21
    assert result.endpoint_support["endpoint_values"][0] != -99.0


@pytest.mark.parametrize(
    ("values", "endpoint_quantiles"),
    [
        (np.arange(4.0), (0.02, 0.98)),
        (np.zeros((5, 1)), (0.02, 0.98)),
        (np.array([0.0, 1.0, 2.0, 3.0, np.nan]), (0.02, 0.98)),
        (np.arange(5.0), (-0.1, 0.9)),
        (np.arange(5.0), (0.5, 0.5)),
        (np.arange(5.0), (0.9, 0.1)),
        (np.arange(5.0), (0.1, 1.1)),
    ],
)
def test_robust_targets_reject_malformed_inputs(
    values: np.ndarray,
    endpoint_quantiles: tuple[float, float],
) -> None:
    with pytest.raises(ValueError):
        robust_angle_targets(values, endpoint_quantiles)


def test_robust_targets_reject_degenerate_robust_range() -> None:
    values = np.concatenate(([-1.0], np.zeros(98), [1.0]))

    with pytest.raises(ValueError, match="degenerate"):
        robust_angle_targets(values)


def _separated_level_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets = np.linspace(0.0, 4.0, 5)
    offsets = np.array([-0.08, -0.04, 0.0, 0.04, 0.08])
    parameters = (targets[:, None] + offsets).reshape(-1)
    descriptors = np.tile(np.array([[-4.0], [-1.0], [0.0], [2.0], [9.0]]), (5, 1))
    source_indices = np.arange(100, 125, dtype=np.int64)
    return parameters, descriptors, source_indices, targets


def test_level_selection_uses_exact_descriptor_medoids() -> None:
    parameters, descriptors, source_indices, targets = _separated_level_data()

    result = select_level_medoids(parameters, descriptors, source_indices, targets)

    assert result.selected_row_indices.tolist() == [2, 7, 12, 17, 22]
    assert result.source_indices.tolist() == [102, 107, 112, 117, 122]
    assert np.allclose(result.observed_parameters, targets)
    assert result.support_counts.tolist() == [5, 5, 5, 5, 5]
    assert all(len(history) == 1 for history in result.expansion_history)
    assert all(history[0].factor == 1.0 for history in result.expansion_history)


def test_level_selection_scales_extreme_finite_descriptors_before_distances() -> None:
    targets = np.linspace(0.0, 4.0, 5)
    offsets = np.array([-0.08, -0.04, 0.0, 0.04, 0.08])
    parameters = (targets[:, None] + offsets).reshape(-1)
    descriptors = np.tile(
        np.array([[0.0], [10.0], [9.0], [8.0], [7.0]]) * 1e200,
        (5, 1),
    )

    result = select_level_medoids(
        parameters,
        descriptors,
        np.arange(25, dtype=np.int64),
        targets,
    )

    assert result.selected_row_indices.tolist() == [3, 8, 13, 18, 23]


def test_thumb_main_trajectory_selects_real_monotonic_arc_medoids() -> None:
    parameter = np.linspace(0.0, np.pi / 2.0, 129)
    tip_points = np.column_stack(
        (np.cos(parameter), np.sin(parameter), 0.1 * parameter)
    )
    result = select_thumb_arc_medoids(
        tip_points,
        tip_points,
        np.arange(tip_points.shape[0], dtype=np.int64),
        manifold_bins=32,
        min_support=1,
    )

    assert result.selected_row_indices.shape == (5,)
    assert len(set(result.source_indices.tolist())) == 5
    assert np.all(np.diff(result.observed_arc_fractions) > 0.0)
    assert result.populated_bin_count >= 5
    assert result.explained_variance > 0.9


def _rotate_in_palm(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
            0.0,
        ]
    )


def _direction(azimuth: np.ndarray, elevation: float) -> np.ndarray:
    return np.cos(elevation) * azimuth + np.array([0.0, 0.0, np.sin(elevation)])


def _synthetic_anchor_frame(
    *,
    lateral: float = 0.0,
    bending: float = 0.0,
    thumb_bending: float = 0.0,
) -> np.ndarray:
    frame = np.zeros((21, 3), dtype=np.float64)
    mcp_positions = {
        5: np.array([0.5, 0.8, 0.0]),
        9: np.array([0.0, 1.0, 0.0]),
        13: np.array([-0.25, 0.9, 0.0]),
        17: np.array([-0.5, 0.8, 0.0]),
    }
    for mcp_index, mcp in mcp_positions.items():
        direction = _rotate_in_palm(mcp / np.linalg.norm(mcp), lateral)
        elevations = (bending, 2.0 * bending, 2.5 * bending)
        frame[mcp_index] = mcp
        frame[mcp_index + 1] = mcp + _direction(direction, elevations[0])
        frame[mcp_index + 2] = frame[mcp_index + 1] + _direction(
            direction, elevations[1]
        )
        frame[mcp_index + 3] = frame[mcp_index + 2] + _direction(
            direction, elevations[2]
        )

    thumb_cmc = np.array([0.2, 0.25, 0.0])
    thumb_base = np.array([0.8, 0.6, 0.0])
    thumb_base /= np.linalg.norm(thumb_base)
    frame[1] = thumb_cmc
    frame[2] = thumb_cmc + thumb_base
    thumb_direction = _rotate_in_palm(thumb_base, lateral)
    frame[3] = frame[2] + _direction(thumb_direction, thumb_bending)
    frame[4] = frame[3] + _direction(thumb_direction, 2.0 * thumb_bending)
    return frame


def test_mine_human_anchor_records_returns_ordered_fifty_row_contract() -> None:
    levels = np.linspace(-0.2, 0.2, 5)
    lateral_frames = [_synthetic_anchor_frame(lateral=value) for value in levels]
    bending_frames = [
        _synthetic_anchor_frame(bending=value)
        for value in np.linspace(0.0, 0.4, 5)
    ]
    thumb_frames = [
        _synthetic_anchor_frame(thumb_bending=value)
        for value in np.linspace(0.0, 0.8, 5)
    ]
    anchors = mine_human_anchor_records(
        np.stack([*lateral_frames, *bending_frames, *thumb_frames]),
        endpoint_quantiles=(0.0, 1.0),
        min_candidates=1,
        min_level_support=1,
        thumb_manifold_bins=5,
    )

    assert anchors.human_frames.shape == (50, 21, 3)
    assert anchors.human_points.shape == (50, 3)
    assert anchors.source_indices.shape == (50,)
    assert anchors.finger_indices.tolist() == [
        finger_index
        for finger_index in range(5)
        for _anchor_type in range(2)
        for _level in range(5)
    ]
    assert anchors.anchor_types.tolist() == [
        anchor_type
        for _finger in range(5)
        for anchor_type in ("lateral", "bending")
        for _level in range(5)
    ]
    assert anchors.levels.tolist() == list(range(5)) * 10
    assert np.allclose(anchors.trajectory_t.reshape(10, 5), mining.LEVEL_FRACTIONS)
    assert len(anchors.group_metadata) == 10


def test_level_selection_breaks_medoid_ties_by_target_distance_then_source() -> None:
    targets = np.linspace(0.0, 4.0, 5)
    offsets = np.array([-0.08, -0.04, 0.04, 0.08, 0.09])
    parameters = (targets[:, None] + offsets).reshape(-1)
    descriptors = np.zeros((25, 2))
    source_indices = np.arange(500, 525, dtype=np.int64)
    for start in range(0, 25, 5):
        source_indices[start + 1] = 900 + start
        source_indices[start + 2] = 100 + start

    result = select_level_medoids(parameters, descriptors, source_indices, targets)

    assert result.selected_row_indices.tolist() == [2, 7, 12, 17, 22]
    assert result.source_indices.tolist() == [100, 105, 110, 115, 120]


def test_level_selection_caps_nearest_candidates_and_records_expansion() -> None:
    parameters = np.arange(101, dtype=np.float64)
    descriptors = np.arange(101, dtype=np.float64)[:, None]
    source_indices = np.arange(1000, 1101, dtype=np.int64)
    targets = np.linspace(0.0, 100.0, 5)

    result = select_level_medoids(
        parameters,
        descriptors,
        source_indices,
        targets,
        max_candidates=5,
    )

    first_history = result.expansion_history[0]
    assert [(item.factor, item.candidate_count) for item in first_history] == [
        (1.0, 3),
        (2.0, 6),
    ]
    assert first_history[-1].support_count == 5
    assert result.support_counts.tolist() == [5, 5, 5, 5, 5]
    assert result.selected_row_indices[0] == 2


def test_overlapping_bands_select_distinct_strictly_monotonic_frames() -> None:
    parameters = np.arange(5, dtype=np.float64)
    descriptors = np.arange(5, dtype=np.float64)[:, None]
    source_indices = np.array([50, 40, 30, 20, 10], dtype=np.int64)
    targets = np.linspace(0.0, 4.0, 5)

    result = select_level_medoids(
        parameters,
        descriptors,
        source_indices,
        targets,
        min_support=1,
        level_band_fraction=1.0,
    )

    assert result.selected_row_indices.tolist() == [0, 1, 2, 3, 4]
    assert len(set(result.source_indices.tolist())) == 5
    assert np.all(np.diff(result.observed_parameters) > 0.0)


def test_medoid_pairwise_chunk_size_does_not_change_exact_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters, descriptors, source_indices, targets = _separated_level_data()
    expected = select_level_medoids(parameters, descriptors, source_indices, targets)
    monkeypatch.setattr(mining, "_PAIRWISE_CHUNK_SIZE", 2)

    chunked = select_level_medoids(parameters, descriptors, source_indices, targets)

    assert np.array_equal(chunked.selected_row_indices, expected.selected_row_indices)


def test_level_selection_results_are_read_only() -> None:
    parameters, descriptors, source_indices, targets = _separated_level_data()
    result = select_level_medoids(parameters, descriptors, source_indices, targets)

    with pytest.raises(ValueError, match="read-only"):
        result.source_indices[0] = -1


@pytest.mark.parametrize(
    ("parameters", "descriptors", "source_indices", "targets", "kwargs"),
    [
        (np.zeros((5, 1)), np.zeros((5, 1)), np.arange(5), np.arange(5.0), {}),
        (np.arange(5.0), np.zeros(5), np.arange(5), np.arange(5.0), {}),
        (np.arange(5.0), np.zeros((4, 1)), np.arange(5), np.arange(5.0), {}),
        (np.arange(5.0), np.zeros((5, 0)), np.arange(5), np.arange(5.0), {}),
        (
            np.arange(5.0),
            np.zeros((5, 1)),
            np.array([0, 1, 2, 3, 3]),
            np.arange(5.0),
            {},
        ),
        (np.arange(5.0), np.zeros((5, 1)), np.arange(5.0), np.arange(5.0), {}),
        (np.arange(5.0), np.zeros((5, 1)), np.arange(5), np.arange(4.0), {}),
        (
            np.arange(5.0),
            np.zeros((5, 1)),
            np.arange(5),
            np.array([0.0, 1.0, 1.0, 3.0, 4.0]),
            {},
        ),
        (
            np.arange(5.0),
            np.full((5, 1), np.nan),
            np.arange(5),
            np.arange(5.0),
            {},
        ),
        (
            np.arange(5.0),
            np.zeros((5, 1)),
            np.arange(5),
            np.arange(5.0),
            {"min_support": 0},
        ),
        (
            np.arange(5.0),
            np.zeros((5, 1)),
            np.arange(5),
            np.arange(5.0),
            {"min_support": 5, "max_candidates": 4},
        ),
        (
            np.arange(5.0),
            np.zeros((5, 1)),
            np.arange(5),
            np.arange(5.0),
            {"level_band_fraction": 0.0},
        ),
        (
            np.arange(5.0),
            np.zeros((5, 1)),
            np.arange(5),
            np.arange(5.0),
            {"level_band_fraction": "bad"},
        ),
        (
            np.arange(5.0),
            np.zeros((5, 1)),
            np.arange(5),
            np.arange(5.0),
            {"band_factors": (1.0, 0.5)},
        ),
    ],
)
def test_level_selection_rejects_malformed_inputs(
    parameters: np.ndarray,
    descriptors: np.ndarray,
    source_indices: np.ndarray,
    targets: np.ndarray,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        select_level_medoids(
            parameters,
            descriptors,
            source_indices,
            targets,
            **kwargs,
        )


def test_level_selection_fails_when_a_band_cannot_reach_minimum_support() -> None:
    with pytest.raises(ValueError, match="support"):
        select_level_medoids(
            np.arange(5.0) + 100.0,
            np.zeros((5, 1)),
            np.arange(5),
            np.arange(5.0),
        )


def test_level_selection_fails_when_strict_monotonicity_is_impossible() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        select_level_medoids(
            np.zeros(5),
            np.arange(5.0)[:, None],
            np.arange(5),
            np.arange(5.0),
            min_support=1,
            level_band_fraction=1.0,
        )


def test_lateral_filter_succeeds_without_fallback_at_inclusive_tolerance() -> None:
    alpha = np.array([9.0, -4.0, 0.0, 0.0])
    beta = np.array(
        [
            [0.05, -0.02, 0.01],
            [0.10, 0.00, -0.10],
            [0.11, 0.00, 0.00],
            [0.00, 0.00, 0.00],
        ]
    )
    valid = np.array([True, True, True, False])

    result = filter_motion_candidates(
        alpha,
        beta,
        valid,
        "lateral",
        straight_tol=0.1,
        alpha_zero_tol=0.2,
        coupling_tol=0.3,
        min_candidates=2,
    )

    assert result.mask.tolist() == [True, True, False, False]
    assert result.count == 2
    assert result.factor == 1.0
    assert result.effective_straight_tol == pytest.approx(0.1)
    assert [attempt.count for attempt in result.attempted_history] == [2]


def test_lateral_filter_returns_the_first_successful_fallback_mask() -> None:
    alpha = np.zeros(5)
    beta = np.array(
        [
            [0.05, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.12, 0.0, 0.0],
            [0.14, 0.0, 0.0],
            [0.16, 0.0, 0.0],
        ]
    )

    result = filter_motion_candidates(
        alpha,
        beta,
        np.ones(5, dtype=np.bool_),
        "lateral",
        straight_tol=0.1,
        alpha_zero_tol=0.2,
        coupling_tol=0.3,
        min_candidates=4,
    )

    assert result.mask.tolist() == [True, True, True, True, False]
    assert result.count == 4
    assert result.factor == 1.5
    assert result.effective_straight_tol == pytest.approx(0.15)
    assert [(item.factor, item.count) for item in result.attempted_history] == [
        (1.0, 2),
        (1.5, 4),
    ]


def test_bending_filter_uses_alpha_and_approved_non_thumb_coupling() -> None:
    alpha = np.array([0.0, 0.1, 0.11, 0.0])
    beta = np.array(
        [
            [0.30, 0.30, 0.15],
            [0.20, 0.23, 0.10],
            [0.20, 0.20, 0.10],
            [0.20, 0.28, 0.10],
        ]
    )

    result = filter_motion_candidates(
        alpha,
        beta,
        np.ones(4, dtype=np.bool_),
        "bending",
        straight_tol=0.4,
        alpha_zero_tol=0.1,
        coupling_tol=0.05,
        min_candidates=2,
    )

    assert result.mask.tolist() == [True, True, False, False]
    assert result.factor == 1.0
    assert result.effective_alpha_zero_tol == pytest.approx(0.1)
    assert result.effective_coupling_tol == pytest.approx(0.05)


def test_filter_never_relaxes_invalid_or_nonfinite_rows() -> None:
    alpha = np.array([0.0, np.nan, 0.0, np.inf, 0.0, 0.0])
    beta = np.zeros((6, 3))
    beta[2, 0] = np.nan
    beta[4] = [1.0, 1.0, 1.0]
    beta[5] = [0.1, 0.0, 0.0]
    valid = np.array([True, True, True, True, False, True])

    with pytest.raises(CandidateFilterError) as caught:
        filter_motion_candidates(
            alpha,
            beta,
            valid,
            "lateral",
            straight_tol=0.01,
            alpha_zero_tol=0.01,
            coupling_tol=0.01,
            min_candidates=2,
        )

    assert [attempt.count for attempt in caught.value.attempted_history] == [
        1,
        1,
        1,
        1,
        1,
    ]


def test_filter_exhaustion_records_complete_attempt_history() -> None:
    with pytest.raises(CandidateFilterError, match="exhausted") as caught:
        filter_motion_candidates(
            np.zeros(5),
            np.full((5, 3), 1.0),
            np.ones(5, dtype=np.bool_),
            "lateral",
            straight_tol=0.01,
            alpha_zero_tol=0.02,
            coupling_tol=0.03,
        )

    error = caught.value
    assert [attempt.factor for attempt in error.attempted_history] == [
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
    ]
    assert [attempt.count for attempt in error.attempted_history] == [0] * 5


def test_filter_result_mask_is_read_only_and_metadata_is_json_serializable() -> None:
    result = filter_motion_candidates(
        np.zeros(5),
        np.zeros((5, 3)),
        np.ones(5, dtype=np.bool_),
        "lateral",
        straight_tol=0.1,
        alpha_zero_tol=0.2,
        coupling_tol=0.3,
    )

    with pytest.raises(ValueError, match="read-only"):
        result.mask[0] = False
    json.dumps(result.to_metadata())


@pytest.mark.parametrize(
    ("alpha", "beta", "valid", "anchor_type", "kwargs"),
    [
        (np.zeros((5, 1)), np.zeros((5, 3)), np.ones(5, dtype=bool), "lateral", {}),
        (np.zeros(5), np.zeros((5, 2)), np.ones(5, dtype=bool), "lateral", {}),
        (np.zeros(5), np.zeros((4, 3)), np.ones(5, dtype=bool), "lateral", {}),
        (np.zeros(5), np.zeros((5, 3)), np.ones(4, dtype=bool), "lateral", {}),
        (np.zeros(5), np.zeros((5, 3)), np.ones(5), "lateral", {}),
        (np.zeros(5), np.zeros((5, 3)), np.ones(5, dtype=bool), "thumb", {}),
        (
            np.zeros(5),
            np.zeros((5, 3)),
            np.ones(5, dtype=bool),
            "lateral",
            {"straight_tol": 0.0},
        ),
        (
            np.zeros(5),
            np.zeros((5, 3)),
            np.ones(5, dtype=bool),
            "lateral",
            {"straight_tol": "bad"},
        ),
        (
            np.zeros(5),
            np.zeros((5, 3)),
            np.ones(5, dtype=bool),
            "lateral",
            {"alpha_zero_tol": -1.0},
        ),
        (
            np.zeros(5),
            np.zeros((5, 3)),
            np.ones(5, dtype=bool),
            "lateral",
            {"coupling_tol": np.inf},
        ),
        (
            np.zeros(5),
            np.zeros((5, 3)),
            np.ones(5, dtype=bool),
            "lateral",
            {"min_candidates": 0},
        ),
        (
            np.zeros(5),
            np.zeros((5, 3)),
            np.ones(5, dtype=bool),
            "lateral",
            {"fallback_factors": (1.0, 0.5)},
        ),
    ],
)
def test_filter_rejects_malformed_inputs(
    alpha: np.ndarray,
    beta: np.ndarray,
    valid: np.ndarray,
    anchor_type: str,
    kwargs: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "straight_tol": 0.1,
        "alpha_zero_tol": 0.1,
        "coupling_tol": 0.1,
        **kwargs,
    }
    with pytest.raises(ValueError):
        filter_motion_candidates(
            alpha,
            beta,
            valid,
            anchor_type,
            **arguments,
        )
