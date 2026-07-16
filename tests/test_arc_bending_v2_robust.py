import numpy as np


def test_robust_arc_selection_uses_raw_candidate_support_after_p2_p98_clip() -> None:
    from geort.anchor.arc_bending_v2_robust import select_robust_arc_medoids

    t = np.linspace(-2.0, 2.0, 101)
    tips = np.column_stack((t, t * t, np.zeros_like(t)))
    descriptors = np.column_stack((tips, tips))
    selected = select_robust_arc_medoids(tips, descriptors, np.arange(t.size))

    assert selected["domain_clip"] == "projection_quantiles_0.02_0.98"
    assert np.all(np.diff(selected["observed_arc_fractions"]) > 0.0)
    assert np.all(np.asarray(selected["support_counts"]) >= 5)
    assert selected["candidate_count"] == 101
