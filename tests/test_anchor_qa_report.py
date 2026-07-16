import numpy as np
import pytest

from geort.anchor.qa_report import (
    normalized_span_ratio,
    trajectory_quality,
    validate_parity_composition,
)


def test_validate_parity_composition_rejects_noncanonical_row_count():
    bundle = {
        "finger_indices": np.zeros(749, dtype=np.int64),
        "anchor_types": np.full(749, "lateral"),
    }

    with pytest.raises(ValueError, match="750"):
        validate_parity_composition(bundle)


def test_trajectory_quality_accepts_uniform_forward_sequence():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    result = trajectory_quality(points)

    assert result["all_direction_dots_positive"] is True
    assert result["step_ratio_max_min"] == pytest.approx(1.0)


def test_normalized_span_ratio_uses_each_side_stats():
    human = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    robot = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    stats = {"center": [0.0, 0.0, 0.0], "scale": 1.0}

    assert normalized_span_ratio(human, robot, stats, stats) == pytest.approx(0.5)

