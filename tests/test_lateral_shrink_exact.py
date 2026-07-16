import numpy as np


def test_exact_knots_evaluate_linear_pair_trajectory_at_quarter_levels() -> None:
    from geort.anchor.lateral_shrink_exact import exact_level_knots

    times = np.linspace(0.0, 1.0, 50)
    values = np.column_stack((3.0 * times - 1.0, times * 0.0))

    knots = exact_level_knots(values, times)
    assert np.allclose(knots[:, 0], [-1.0, -0.25, 0.5, 1.25, 2.0])
