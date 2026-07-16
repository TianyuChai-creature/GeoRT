import numpy as np


def test_choose_monotonic_pair_prefers_lowest_descriptor_rank() -> None:
    from geort.anchor.ring_lateral_monotonic import choose_monotonic_pair

    # Candidate order is already descriptor-medoid rank order.
    pair = choose_monotonic_pair(
        level2_order=np.array([10, 11]),
        level3_order=np.array([20, 21]),
        projection={1: 0.0, 10: 0.8, 11: 0.3, 20: 0.7, 21: 1.4, 4: 2.0, 5: 3.0},
        fixed_projections=(0.0, 2.0, 3.0),
    )

    assert pair == (11, 20)
