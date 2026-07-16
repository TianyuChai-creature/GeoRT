import torch

from geort.loss import null_space_loss, nullspace_rows_used


class _ConstantChain:
    def get_joint_parameter_names(self):
        return ["a", "b", "c", "d"]

    def jacobian(self, theta):
        jacobian = torch.zeros(theta.shape[0], 6, 4, device=theta.device)
        jacobian[:, 0, 0] = 1.0
        jacobian[:, 1, 1] = 1.0
        jacobian[:, 2, 2] = 1.0
        return jacobian


def test_subsample_averages_selected_rows_not_zero_filled():
    joint = torch.full((64, 20), 0.2)
    midpoint = torch.zeros(20)
    limits = torch.ones(20)
    chains = [_ConstantChain() for _ in range(5)]
    indices = [list(range(4 * finger, 4 * finger + 4)) for finger in range(5)]

    full = null_space_loss(joint, midpoint, chains, indices, -limits, limits)
    generator = torch.Generator().manual_seed(123)
    sampled = null_space_loss(
        joint, midpoint, chains, indices, -limits, limits,
        subsample=8, generator=generator,
    )

    assert torch.allclose(sampled, full)


def test_full_nullspace_mode_reports_every_batch_row():
    assert nullspace_rows_used(batch_rows=2048, subsample=0) == 2048
    assert nullspace_rows_used(batch_rows=2048, subsample=4096) == 2048
    assert nullspace_rows_used(batch_rows=2048, subsample=256) == 256
