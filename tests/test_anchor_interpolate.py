from __future__ import annotations

import numpy as np
import pytest

from geort.anchor.interpolate import interpolate_sparse_trajectory


SPARSE_POINTS = np.array(
    [
        [0.0, 1.0, -1.0],
        [2.0, 0.0, 3.0],
        [-1.0, 4.0, 2.0],
        [3.0, 2.0, -2.0],
        [1.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _piecewise_linear_reference(
    points: np.ndarray,
    output_count: int,
) -> np.ndarray:
    reference = np.empty((output_count, points.shape[1]), dtype=np.float64)
    for output_index, trajectory_t in enumerate(
        np.linspace(0.0, 1.0, output_count)
    ):
        scaled_t = trajectory_t * 4.0
        left_index = min(int(np.floor(scaled_t)), 3)
        segment_weight = scaled_t - left_index
        reference[output_index] = (
            (1.0 - segment_weight) * points[left_index]
            + segment_weight * points[left_index + 1]
        )
    return reference


@pytest.mark.parametrize("output_count", [50, 100])
def test_interpolates_five_knots_piecewise_linearly(output_count: int) -> None:
    result = interpolate_sparse_trajectory(SPARSE_POINTS, output_count)
    trajectory_t = np.linspace(0.0, 1.0, output_count)
    expected_points = _piecewise_linear_reference(SPARSE_POINTS, output_count)

    assert result["points"].shape == (output_count, 3)
    assert result["trajectory_t"].shape == (output_count,)
    assert result["source_sparse_indices"].shape == (output_count, 2)
    assert np.array_equal(result["points"][0], SPARSE_POINTS[0])
    assert np.array_equal(result["points"][-1], SPARSE_POINTS[-1])
    assert np.allclose(result["points"], expected_points)
    assert np.array_equal(result["trajectory_t"], trajectory_t)
    assert np.all(np.diff(result["trajectory_t"]) > 0.0)


def test_source_indices_bracket_each_output_parameter() -> None:
    output_count = 9
    result = interpolate_sparse_trajectory(SPARSE_POINTS, output_count)
    trajectory_t = np.linspace(0.0, 1.0, output_count)
    sparse_t = np.linspace(0.0, 1.0, 5)
    expected_upper = np.clip(
        np.searchsorted(sparse_t, trajectory_t, side="right"),
        1,
        4,
    )
    expected_indices = np.column_stack((expected_upper - 1, expected_upper))

    assert np.array_equal(result["source_sparse_indices"], expected_indices)


def test_interpolation_accepts_any_positive_point_dimension() -> None:
    points = SPARSE_POINTS[:, :2]

    result = interpolate_sparse_trajectory(points, 5)

    assert result["points"].shape == (5, 2)
    assert np.array_equal(result["points"], points)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((4, 3)),
        np.zeros((5, 3, 1)),
        np.zeros((5, 0)),
        np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, np.nan],
                [3.0, 3.0],
                [4.0, 4.0],
            ]
        ),
    ],
)
def test_interpolation_rejects_invalid_points(points: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"\[5, D\].*finite"):
        interpolate_sparse_trajectory(points, 5)


@pytest.mark.parametrize(
    "output_count",
    [0, 4, True, np.bool_(False), 5.0, 5.5],
)
def test_interpolation_rejects_invalid_output_counts(output_count: object) -> None:
    with pytest.raises(ValueError, match="at least 5"):
        interpolate_sparse_trajectory(SPARSE_POINTS, output_count)
