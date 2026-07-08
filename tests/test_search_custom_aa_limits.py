from __future__ import annotations

import numpy as np

from geort.mocap.search_custom_aa_limits import (
    REFERENCE_AA_LIMIT,
    generate_aa_limit_candidates,
    reference_aa_limits_for_joints,
    score_limit_candidate,
)


def _overlap_report(pair_names: list[str]) -> dict[str, dict[str, float]]:
    return {
        pair_name: {
            "iou": 0.0,
            "overlap_a_ratio": 0.0,
            "overlap_b_ratio": 0.0,
        }
        for pair_name in pair_names
    }


def test_reference_aa_limits_use_original_search_baseline() -> None:
    reference_limits = reference_aa_limits_for_joints(["F2-R-MCP2", "F5-R-MCP2"])

    assert REFERENCE_AA_LIMIT == (-0.61, 0.61)
    assert reference_limits["F2-R-MCP2"] == (-0.61, 0.61)
    assert reference_limits["F5-R-MCP2"] == (-0.61, 0.61)
    assert reference_limits["F2-R-MCP2"] != (-0.147353, 0.32864)


def test_reference_aa_limits_include_manual_right_closure_pose() -> None:
    reference_limits = reference_aa_limits_for_joints(["F2-R-MCP2", "F3-R-MCP2", "F4-R-MCP2", "F5-R-MCP2"])
    manual_closure_pose = {
        "F2-R-MCP2": -0.264,
        "F3-R-MCP2": 0.0,
        "F4-R-MCP2": 0.239,
        "F5-R-MCP2": 0.552,
    }

    for joint_name, value in manual_closure_pose.items():
        lower, upper = reference_limits[joint_name]
        assert lower <= value <= upper


def test_random_candidates_are_bounded_by_reference_limits_not_current_urdf_limits() -> None:
    reference_limits = reference_aa_limits_for_joints(["F2-R-MCP2"])
    current_cropped_urdf_limits = {"F2-R-MCP2": (-0.147353, 0.32864)}

    candidates = generate_aa_limit_candidates(reference_limits, num_candidates=200, min_width=0.20, seed=1)
    lowers = [candidate["F2-R-MCP2"][0] for candidate in candidates]
    uppers = [candidate["F2-R-MCP2"][1] for candidate in candidates]

    assert min(lowers) < current_cropped_urdf_limits["F2-R-MCP2"][0]
    assert max(uppers) > current_cropped_urdf_limits["F2-R-MCP2"][1]
    assert min(lowers) < -0.30
    assert max(uppers) > 0.35
    assert all(-0.61 <= lower <= 0.0 for lower in lowers)
    assert all(0.0 <= upper <= 0.61 for upper in uppers)


def test_score_limit_candidate_penalizes_missing_adjacent_finger_reach() -> None:
    adjacent_pairs = ["index__middle", "middle__ring", "ring__pinky"]
    far_tips = {
        "index": np.array([[0.00, 0.0, 0.0]], dtype=np.float32),
        "middle": np.array([[0.05, 0.0, 0.0]], dtype=np.float32),
        "ring": np.array([[0.10, 0.0, 0.0]], dtype=np.float32),
        "pinky": np.array([[0.15, 0.0, 0.0]], dtype=np.float32),
    }

    report = score_limit_candidate(
        dataset_overlap=_overlap_report(adjacent_pairs),
        urdf_overlap=_overlap_report(adjacent_pairs),
        urdf_tips=far_tips,
        adjacent_pair_names=adjacent_pairs,
        opposition_pair_names=[],
        adjacent_contact_threshold=0.02,
        adjacent_reach_penalty_weight=5.0,
        regularization_weight=0.0,
    )

    assert report["adjacent_reach"]["max_violation"] > 0.0
    assert report["score"] > 0.0


def test_score_limit_candidate_allows_adjacent_finger_reach_inside_threshold() -> None:
    adjacent_pairs = ["index__middle", "middle__ring", "ring__pinky"]
    close_tips = {
        "index": np.array([[0.000, 0.0, 0.0]], dtype=np.float32),
        "middle": np.array([[0.015, 0.0, 0.0]], dtype=np.float32),
        "ring": np.array([[0.030, 0.0, 0.0]], dtype=np.float32),
        "pinky": np.array([[0.045, 0.0, 0.0]], dtype=np.float32),
    }

    report = score_limit_candidate(
        dataset_overlap=_overlap_report(adjacent_pairs),
        urdf_overlap=_overlap_report(adjacent_pairs),
        urdf_tips=close_tips,
        adjacent_pair_names=adjacent_pairs,
        opposition_pair_names=[],
        adjacent_contact_threshold=0.02,
        adjacent_reach_penalty_weight=5.0,
        regularization_weight=0.0,
    )

    assert report["adjacent_reach"]["max_violation"] == 0.0
    assert report["score"] == 0.0


def test_score_limit_candidate_penalizes_missing_simultaneous_adjacent_closure() -> None:
    adjacent_pairs = ["index__middle", "middle__ring", "ring__pinky"]
    dataset_closure_tips = {
        "index": np.array([[0.000, 0.0, 0.0]], dtype=np.float32),
        "middle": np.array([[0.015, 0.0, 0.0]], dtype=np.float32),
        "ring": np.array([[0.030, 0.0, 0.0]], dtype=np.float32),
        "pinky": np.array([[0.045, 0.0, 0.0]], dtype=np.float32),
    }
    urdf_open_tips = {
        "index": np.array([[0.00, 0.0, 0.0]], dtype=np.float32),
        "middle": np.array([[0.05, 0.0, 0.0]], dtype=np.float32),
        "ring": np.array([[0.10, 0.0, 0.0]], dtype=np.float32),
        "pinky": np.array([[0.15, 0.0, 0.0]], dtype=np.float32),
    }

    report = score_limit_candidate(
        dataset_overlap=_overlap_report(adjacent_pairs),
        urdf_overlap=_overlap_report(adjacent_pairs),
        urdf_tips=dataset_closure_tips,
        urdf_closure_tips=urdf_open_tips,
        dataset_tips=dataset_closure_tips,
        adjacent_pair_names=adjacent_pairs,
        opposition_pair_names=[],
        closure_contact_threshold=0.02,
        closure_min_coverage_rate=1.0,
        closure_coverage_weight=3.0,
        reach_penalty_weight=0.0,
        adjacent_reach_penalty_weight=0.0,
        regularization_weight=0.0,
    )

    assert report["closure_coverage"]["dataset_frame_count"] == 1
    assert report["closure_coverage"]["urdf_coverage_rate"] == 0.0
    assert report["closure_coverage"]["penalty"] > 0.0
    assert report["score"] > 0.0


def test_score_limit_candidate_skips_closure_penalty_when_dataset_has_no_closure_frames() -> None:
    adjacent_pairs = ["index__middle", "middle__ring", "ring__pinky"]
    dataset_open_tips = {
        "index": np.array([[0.00, 0.0, 0.0]], dtype=np.float32),
        "middle": np.array([[0.05, 0.0, 0.0]], dtype=np.float32),
        "ring": np.array([[0.10, 0.0, 0.0]], dtype=np.float32),
        "pinky": np.array([[0.15, 0.0, 0.0]], dtype=np.float32),
    }

    report = score_limit_candidate(
        dataset_overlap=_overlap_report(adjacent_pairs),
        urdf_overlap=_overlap_report(adjacent_pairs),
        urdf_tips=dataset_open_tips,
        urdf_closure_tips=dataset_open_tips,
        dataset_tips=dataset_open_tips,
        adjacent_pair_names=adjacent_pairs,
        opposition_pair_names=[],
        closure_contact_threshold=0.02,
        closure_min_coverage_rate=1.0,
        closure_coverage_weight=3.0,
        reach_penalty_weight=0.0,
        adjacent_reach_penalty_weight=0.0,
        regularization_weight=0.0,
    )

    assert report["closure_coverage"]["dataset_frame_count"] == 0
    assert report["closure_coverage"]["penalty"] == 0.0
    assert report["score"] == 0.0


def test_score_limit_candidate_penalizes_missing_dataset_closure_replay() -> None:
    adjacent_pairs = ["index__middle", "middle__ring", "ring__pinky"]
    dataset_closure_tips = {
        "index": np.array([[0.000, 0.0, 0.0], [0.000, 0.0, 0.0]], dtype=np.float32),
        "middle": np.array([[0.012, 0.0, 0.0], [0.014, 0.0, 0.0]], dtype=np.float32),
        "ring": np.array([[0.024, 0.0, 0.0], [0.028, 0.0, 0.0]], dtype=np.float32),
        "pinky": np.array([[0.036, 0.0, 0.0], [0.042, 0.0, 0.0]], dtype=np.float32),
    }
    urdf_open_tips = {
        "index": np.array([[0.00, 0.0, 0.0]], dtype=np.float32),
        "middle": np.array([[0.05, 0.0, 0.0]], dtype=np.float32),
        "ring": np.array([[0.10, 0.0, 0.0]], dtype=np.float32),
        "pinky": np.array([[0.15, 0.0, 0.0]], dtype=np.float32),
    }

    report = score_limit_candidate(
        dataset_overlap=_overlap_report(adjacent_pairs),
        urdf_overlap=_overlap_report(adjacent_pairs),
        urdf_tips=dataset_closure_tips,
        urdf_closure_tips=urdf_open_tips,
        dataset_tips=dataset_closure_tips,
        adjacent_pair_names=adjacent_pairs,
        opposition_pair_names=[],
        closure_contact_threshold=0.02,
        dataset_closure_replay_tolerance=0.25,
        dataset_closure_replay_weight=4.0,
        closure_coverage_weight=0.0,
        reach_penalty_weight=0.0,
        adjacent_reach_penalty_weight=0.0,
        regularization_weight=0.0,
    )

    replay = report["dataset_closure_replay"]
    assert replay["dataset_frame_count"] == 2
    assert replay["success_rate"] == 0.0
    assert replay["penalty"] > 0.0
    assert report["score"] > 0.0


def test_score_limit_candidate_allows_matching_dataset_closure_replay() -> None:
    adjacent_pairs = ["index__middle", "middle__ring", "ring__pinky"]
    dataset_closure_tips = {
        "index": np.array([[0.000, 0.0, 0.0], [0.000, 0.0, 0.0]], dtype=np.float32),
        "middle": np.array([[0.012, 0.0, 0.0], [0.014, 0.0, 0.0]], dtype=np.float32),
        "ring": np.array([[0.024, 0.0, 0.0], [0.028, 0.0, 0.0]], dtype=np.float32),
        "pinky": np.array([[0.036, 0.0, 0.0], [0.042, 0.0, 0.0]], dtype=np.float32),
    }

    report = score_limit_candidate(
        dataset_overlap=_overlap_report(adjacent_pairs),
        urdf_overlap=_overlap_report(adjacent_pairs),
        urdf_tips=dataset_closure_tips,
        urdf_closure_tips=dataset_closure_tips,
        dataset_tips=dataset_closure_tips,
        adjacent_pair_names=adjacent_pairs,
        opposition_pair_names=[],
        closure_contact_threshold=0.02,
        dataset_closure_replay_tolerance=0.25,
        dataset_closure_replay_weight=4.0,
        closure_coverage_weight=0.0,
        reach_penalty_weight=0.0,
        adjacent_reach_penalty_weight=0.0,
        regularization_weight=0.0,
    )

    replay = report["dataset_closure_replay"]
    assert replay["dataset_frame_count"] == 2
    assert replay["success_rate"] == 1.0
    assert replay["penalty"] == 0.0
    assert report["score"] == 0.0

def test_score_limit_candidate_penalizes_limits_that_crop_manual_closure_prior() -> None:
    adjacent_pairs = ["index__middle", "middle__ring", "ring__pinky"]
    empty_tips = {
        "index": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "middle": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "ring": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "pinky": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
    }
    manual_closure_prior = {
        "F2-R-MCP2": -0.264,
        "F3-R-MCP2": 0.0,
        "F4-R-MCP2": 0.239,
        "F5-R-MCP2": 0.552,
    }
    cropped_limits = {
        "F2-R-MCP2": (-0.02, 0.24),
        "F3-R-MCP2": (-0.18, 0.11),
        "F4-R-MCP2": (-0.27, 0.35),
        "F5-R-MCP2": (-0.01, 0.24),
    }

    report = score_limit_candidate(
        dataset_overlap=_overlap_report(adjacent_pairs),
        urdf_overlap=_overlap_report(adjacent_pairs),
        urdf_tips=empty_tips,
        adjacent_pair_names=adjacent_pairs,
        opposition_pair_names=[],
        candidate_limits=cropped_limits,
        manual_closure_prior=manual_closure_prior,
        manual_closure_prior_margin=0.10,
        manual_closure_prior_weight=10.0,
        reach_penalty_weight=0.0,
        adjacent_reach_penalty_weight=0.0,
        dataset_closure_replay_weight=0.0,
        regularization_weight=0.0,
    )

    prior = report["manual_closure_prior"]
    assert prior["passes"] is False
    assert prior["joint_metrics"]["F2-R-MCP2"]["passes"] is False
    assert prior["joint_metrics"]["F5-R-MCP2"]["passes"] is False
    assert prior["penalty"] > 0.0
    assert report["score"] > 0.0

