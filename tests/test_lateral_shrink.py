import numpy as np


def test_centered_lateral_scaling_uses_ratio_over_target() -> None:
    from geort.anchor.lateral_shrink import scale_knots_to_target_ratio

    knots = np.zeros((5, 4))
    knots[:, 0] = np.linspace(-2.0, 2.0, 5)
    scaled, multiplier = scale_knots_to_target_ratio(knots, current_ratio=0.68, target_ratio=0.85)

    assert np.isclose(multiplier, 0.8)
    assert np.allclose(scaled[:, 0], np.linspace(-1.6, 1.6, 5))
    assert np.allclose(scaled[2], knots[2])
