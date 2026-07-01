import unittest

import torch

from geort.trainer import (
    compute_finger_segment_direction_loss,
    compute_tip_pinch_loss,
    weighted_keypoint_mean,
)


class TrainerLossTest(unittest.TestCase):
    def test_weighted_keypoint_mean_normalizes_by_weight_sum(self):
        losses = torch.tensor([1.0, 3.0])
        weights = torch.tensor([1.0, 0.25])

        result = weighted_keypoint_mean(losses, weights)

        self.assertAlmostEqual(result.item(), 1.4, places=6)

    def test_tip_pinch_loss_only_uses_explicit_tip_pairs(self):
        point = torch.zeros(2, 4, 3)
        embedded = torch.zeros(2, 4, 3)

        # PIP-like points 0 and 2 are close, but they are not in pinch_pairs.
        point[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        point[:, 2, :] = torch.tensor([0.001, 0.0, 0.0])
        embedded[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        embedded[:, 2, :] = torch.tensor([10.0, 0.0, 0.0])

        # Tip pair 1 and 3 is close and should be the only contributor.
        point[:, 1, :] = torch.tensor([0.0, 0.0, 0.0])
        point[:, 3, :] = torch.tensor([0.001, 0.0, 0.0])
        embedded[:, 1, :] = torch.tensor([0.0, 0.0, 0.0])
        embedded[:, 3, :] = torch.tensor([2.0, 0.0, 0.0])

        result = compute_tip_pinch_loss(point, embedded, pinch_pairs=[(1, 3)])

        self.assertAlmostEqual(result.item(), 4.0, places=6)

    def test_tip_pinch_loss_respects_threshold(self):
        point = torch.zeros(1, 2, 3)
        embedded = torch.zeros(1, 2, 3)
        point[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        point[:, 1, :] = torch.tensor([0.02, 0.0, 0.0])
        embedded[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        embedded[:, 1, :] = torch.tensor([3.0, 0.0, 0.0])

        excluded = compute_tip_pinch_loss(point, embedded, pinch_pairs=[(0, 1)], threshold=0.015)
        included = compute_tip_pinch_loss(point, embedded, pinch_pairs=[(0, 1)], threshold=0.025)

        self.assertAlmostEqual(excluded.item(), 0.0, places=6)
        self.assertAlmostEqual(included.item(), 9.0, places=6)

    def test_finger_segment_direction_loss_is_zero_for_matching_directions(self):
        point = torch.zeros(2, 2, 3)
        embedded = torch.zeros(2, 2, 3)
        point[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        point[:, 1, :] = torch.tensor([1.0, 0.0, 0.0])
        embedded[:, 0, :] = torch.tensor([2.0, 1.0, 0.0])
        embedded[:, 1, :] = torch.tensor([5.0, 1.0, 0.0])

        result = compute_finger_segment_direction_loss(point, embedded, segment_pairs=[(0, 1)])

        self.assertAlmostEqual(result.item(), 0.0, places=6)

    def test_finger_segment_direction_loss_penalizes_opposite_directions(self):
        point = torch.zeros(1, 2, 3)
        embedded = torch.zeros(1, 2, 3)
        point[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        point[:, 1, :] = torch.tensor([1.0, 0.0, 0.0])
        embedded[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
        embedded[:, 1, :] = torch.tensor([-1.0, 0.0, 0.0])

        result = compute_finger_segment_direction_loss(point, embedded, segment_pairs=[(0, 1)])

        self.assertAlmostEqual(result.item(), 2.0, places=6)

    def test_finger_segment_direction_loss_is_zero_without_pairs(self):
        point = torch.randn(3, 2, 3)
        embedded = torch.randn(3, 2, 3)

        result = compute_finger_segment_direction_loss(point, embedded, segment_pairs=[])

        self.assertAlmostEqual(result.item(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
