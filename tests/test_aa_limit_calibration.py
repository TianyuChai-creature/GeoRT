import unittest

import numpy as np

from geort.mocap.calibrate_custom_aa_limits import (
    circular_delta,
    compute_segment_metric,
    interpolate_joint_for_metric_delta,
    robust_centered_range,
    segment_xz_projection_angle,
    suggest_limit_from_metric_range,
)


class AALimitCalibrationTest(unittest.TestCase):
    def test_compute_segment_metric_uses_xz_projection_angle(self):
        frames = np.zeros((3, 21, 3), dtype=np.float32)
        frames[0, 6, :] = [0.0, 0.0, 0.0]
        frames[0, 8, :] = [1.0, 5.0, 0.0]
        frames[1, 6, :] = [0.0, 0.0, 0.0]
        frames[1, 8, :] = [0.0, 5.0, 1.0]
        frames[2, 6, :] = [0.0, 0.0, 0.0]
        frames[2, 8, :] = [-1.0, 5.0, 0.0]

        metric = compute_segment_metric(frames, pip_id=6, tip_id=8)

        np.testing.assert_allclose(metric, np.array([0.0, np.pi / 2.0, np.pi], dtype=np.float32), atol=1e-6)

    def test_segment_xz_projection_angle_keeps_raw_atan2_angle(self):
        vectors = np.array([
            [-1.0, 0.0, 0.01],
            [-1.0, 0.0, -0.01],
        ], dtype=np.float32)

        metric = segment_xz_projection_angle(vectors)

        self.assertGreater(abs(metric[1] - metric[0]), 6.0)

    def test_circular_delta_wraps_relative_to_center(self):
        angles = np.array([np.pi - 0.01, -np.pi + 0.01], dtype=np.float32)

        delta = circular_delta(angles, center=np.pi)

        np.testing.assert_allclose(delta, np.array([-0.01, 0.01], dtype=np.float32), atol=1e-6)

    def test_robust_centered_range_returns_percentile_delta_from_median(self):
        values = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)

        result = robust_centered_range(values, low_percentile=25.0, high_percentile=75.0)

        self.assertAlmostEqual(result.center, 0.0, places=6)
        self.assertAlmostEqual(result.low_delta, -1.0, places=6)
        self.assertAlmostEqual(result.high_delta, 1.0, places=6)

    def test_interpolate_joint_for_metric_delta_handles_decreasing_metric(self):
        q_values = np.array([-0.6, 0.0, 0.6], dtype=np.float32)
        metric_delta = np.array([0.3, 0.0, -0.3], dtype=np.float32)

        q = interpolate_joint_for_metric_delta(q_values, metric_delta, target_delta=0.15)

        self.assertAlmostEqual(q, -0.3, places=6)

    def test_suggest_limit_from_metric_range_adds_margin_and_clamps_to_old_limits(self):
        q_values = np.linspace(-0.6, 0.6, 5, dtype=np.float32)
        metric_delta = q_values.copy()

        lower, upper = suggest_limit_from_metric_range(
            q_values=q_values,
            robot_metric_delta=metric_delta,
            human_low_delta=-0.2,
            human_high_delta=0.25,
            old_lower=-0.6,
            old_upper=0.6,
            margin_rad=0.05,
        )

        self.assertAlmostEqual(lower, -0.25, places=6)
        self.assertAlmostEqual(upper, 0.30, places=6)


if __name__ == "__main__":
    unittest.main()
