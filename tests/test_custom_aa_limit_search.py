import unittest

import numpy as np

from geort.mocap.search_custom_aa_limits import (
    AA_JOINT_NAMES,
    REFERENCE_AA_LIMITS,
    build_limit_comparison,
    generate_aa_limit_candidates,
    generate_lhs_aa_limit_candidates,
    generate_refined_aa_limit_candidates,
    score_limit_candidate,
)


class CustomAALimitSearchTest(unittest.TestCase):
    def test_build_limit_comparison_reports_current_candidate_and_delta(self):
        current = {
            "F2-R-MCP2": (-0.30, 0.35),
            "F3-R-MCP2": (-0.30, 0.35),
        }
        candidate = {
            "F2-R-MCP2": (-0.20, 0.25),
            "F3-R-MCP2": (-0.15, 0.20),
        }

        comparison = build_limit_comparison(current, candidate)

        self.assertEqual(comparison["F2-R-MCP2"]["current"], [-0.30, 0.35])
        self.assertEqual(comparison["F2-R-MCP2"]["candidate"], [-0.20, 0.25])
        self.assertEqual(comparison["F2-R-MCP2"]["delta"], [0.10, -0.10])

    def test_random_candidates_keep_zero_inside_and_min_width(self):
        candidates = generate_aa_limit_candidates(
            REFERENCE_AA_LIMITS,
            num_candidates=8,
            min_width=0.20,
            seed=1,
        )

        self.assertEqual(len(candidates), 8)
        for candidate in candidates:
            self.assertEqual(set(candidate), set(AA_JOINT_NAMES))
            for lower, upper in candidate.values():
                self.assertLessEqual(lower, 0.0)
                self.assertGreaterEqual(upper, 0.0)
                self.assertGreaterEqual(upper - lower, 0.20)
                self.assertGreaterEqual(lower, -0.30)
                self.assertLessEqual(upper, 0.35)

    def test_lhs_candidates_are_valid_and_record_source(self):
        candidates = generate_lhs_aa_limit_candidates(
            REFERENCE_AA_LIMITS,
            num_candidates=12,
            min_width=0.20,
            seed=3,
        )

        self.assertEqual(len(candidates), 12)
        self.assertTrue(all(item["source"] == "coarse_lhs" for item in candidates))
        first_joint_lowers = {round(item["limits"]["F2-R-MCP2"][0], 4) for item in candidates}
        self.assertGreater(len(first_joint_lowers), 6)
        for item in candidates:
            for lower, upper in item["limits"].values():
                self.assertLessEqual(lower, 0.0)
                self.assertGreaterEqual(upper, 0.0)
                self.assertGreaterEqual(upper - lower, 0.20)

    def test_refined_candidates_stay_near_parent_and_record_parent_metadata(self):
        parent = {
            "candidate_index": 7,
            "limit_comparison": {
                name: {"candidate": [-0.20, 0.25]} for name in AA_JOINT_NAMES
            },
        }

        refined = generate_refined_aa_limit_candidates(
            [parent],
            REFERENCE_AA_LIMITS,
            num_samples_per_parent=6,
            min_width=0.20,
            step_size=0.04,
            round_index=2,
            seed=4,
        )

        self.assertEqual(len(refined), 6)
        for item in refined:
            self.assertEqual(item["source"], "refine_round_2")
            self.assertEqual(item["parent_candidate"], 7)
            self.assertEqual(item["step_size"], 0.04)
            for lower, upper in item["limits"].values():
                self.assertLessEqual(lower, 0.0)
                self.assertGreaterEqual(upper, 0.0)
                self.assertGreaterEqual(upper - lower, 0.20)
                self.assertGreaterEqual(lower, -0.30)
                self.assertLessEqual(upper, 0.35)

    def test_score_limit_candidate_only_penalizes_adjacent_iou_excess(self):
        dataset_overlap = {
            "index__middle": {"iou": 0.10},
            "middle__ring": {"iou": 0.20},
        }
        underlapping_urdf = {
            "index__middle": {"iou": 0.05},
            "middle__ring": {"iou": 0.19},
        }
        over_urdf = {
            "index__middle": {"iou": 0.30},
            "middle__ring": {"iou": 0.19},
        }

        under = score_limit_candidate(
            dataset_overlap=dataset_overlap,
            urdf_overlap=underlapping_urdf,
            urdf_tips={},
            adjacent_pair_names=list(dataset_overlap),
            opposition_pair_names=[],
            iou_tolerance=0.0,
            iou_floor=0.01,
            regularization_weight=0.0,
            reach_penalty_weight=0.0,
        )
        over = score_limit_candidate(
            dataset_overlap=dataset_overlap,
            urdf_overlap=over_urdf,
            urdf_tips={},
            adjacent_pair_names=list(dataset_overlap),
            opposition_pair_names=[],
            iou_tolerance=0.0,
            iou_floor=0.01,
            regularization_weight=0.0,
            reach_penalty_weight=0.0,
        )

        self.assertEqual(under["adjacent"]["p_norm_error"], 0.0)
        self.assertGreater(over["adjacent"]["p_norm_error"], 0.0)
        self.assertEqual(over["adjacent"]["worst_pair"], "index__middle")

    def test_score_limit_candidate_uses_thumb_reach_as_constraint(self):
        reachable_urdf = {
            "thumb": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "index": np.array([[0.004, 0.0, 0.0]], dtype=np.float32),
        }
        unreachable_urdf = {
            "thumb": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "index": np.array([[0.030, 0.0, 0.0]], dtype=np.float32),
        }

        reachable = score_limit_candidate(
            dataset_overlap={},
            urdf_overlap={},
            urdf_tips=reachable_urdf,
            adjacent_pair_names=[],
            opposition_pair_names=["thumb__index"],
            contact_threshold=0.010,
            regularization_weight=0.0,
        )
        unreachable = score_limit_candidate(
            dataset_overlap={},
            urdf_overlap={},
            urdf_tips=unreachable_urdf,
            adjacent_pair_names=[],
            opposition_pair_names=["thumb__index"],
            contact_threshold=0.010,
            regularization_weight=0.0,
        )

        self.assertTrue(reachable["opposition"]["pair_metrics"]["thumb__index"]["passes"])
        self.assertFalse(unreachable["opposition"]["pair_metrics"]["thumb__index"]["passes"])
        self.assertGreater(unreachable["opposition"]["penalty"], reachable["opposition"]["penalty"])


if __name__ == "__main__":
    unittest.main()
