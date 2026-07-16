import torch


def test_closed_form_null_vector_matches_svd_up_to_sign():
    from geort.loss import null_vector_3x4

    torch.manual_seed(7)
    jacobian = torch.randn(1000, 3, 4)
    closed, valid = null_vector_3x4(jacobian)
    svd = torch.linalg.svd(jacobian, full_matrices=True).Vh[:, -1]
    cosine = (closed * svd).sum(-1).abs()

    assert valid.all()
    assert (1.0 - cosine).max() < 1e-5


def test_closed_form_null_vector_masks_degenerate_rows_without_nan():
    from geort.loss import null_vector_3x4

    jacobian = torch.zeros(3, 3, 4)
    vector, valid = null_vector_3x4(jacobian)

    assert not valid.any()
    assert torch.isfinite(vector).all()
    assert torch.equal(vector, torch.zeros_like(vector))
