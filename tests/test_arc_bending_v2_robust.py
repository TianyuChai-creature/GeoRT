import numpy as np


def test_pc1_projection_is_oriented_to_increasing_beta1() -> None:
    from geort.anchor.arc_bending_v2_robust import orient_projection_to_beta1

    projection = np.array([-2.0, -0.5, 0.25, 1.5])
    beta1 = -projection

    oriented, flipped = orient_projection_to_beta1(projection, beta1)

    assert flipped is True
    assert np.dot(oriented - oriented.mean(), beta1 - beta1.mean()) > 0.0


def test_robust_arc_selection_uses_raw_candidate_support_after_p2_p98_clip() -> None:
    from geort.anchor.arc_bending_v2_robust import select_robust_arc_medoids

    t = np.linspace(-2.0, 2.0, 101)
    tips = np.column_stack((t, t * t, np.zeros_like(t)))
    descriptors = np.column_stack((tips, tips))
    selected = select_robust_arc_medoids(
        tips, descriptors, np.arange(t.size), beta1=t,
    )

    assert selected["domain_clip"] == "projection_quantiles_0.02_0.98"
    assert np.all(np.diff(selected["observed_arc_fractions"]) > 0.0)
    assert np.all(np.asarray(selected["support_counts"]) >= 5)
    assert selected["candidate_count"] == 101
