import numpy as np


def test_fast_medoid_order_matches_existing_exact_distance_sum() -> None:
    from geort.anchor.arc_bending_v2_fast import fast_medoid_order
    from geort.anchor.mining import _medoid_order

    rng = np.random.default_rng(7)
    rows = np.arange(23, dtype=np.int64)
    descriptors = rng.normal(size=(23, 12))
    parameters = rng.normal(size=23)
    sources = np.arange(100, 123, dtype=np.int64)
    target = 0.17

    expected = _medoid_order(rows, descriptors, parameters, sources, target)
    actual = fast_medoid_order(rows, descriptors, parameters, sources, target)
    assert np.array_equal(actual, expected)
