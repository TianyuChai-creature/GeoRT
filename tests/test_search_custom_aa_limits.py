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
    reference_limits = reference_aa_limits_for_joints(["F2-R-MCP2"])

    assert reference_limits["F2-R-MCP2"] == REFERENCE_AA_LIMIT
    assert reference_limits["F2-R-MCP2"] != (-0.147353, 0.32864)


def test_random_candidates_are_bounded_by_reference_limits_not_current_urdf_limits() -> None:
    reference_limits = reference_aa_limits_for_joints(["F2-R-MCP2"], (-0.30, 0.35))
    current_cropped_urdf_limits = {"F2-R-MCP2": (-0.147353, 0.32864)}

    candidates = generate_aa_limit_candidates(reference_limits, num_candidates=200, min_width=0.20, seed=1)
    lowers = [candidate["F2-R-MCP2"][0] for candidate in candidates]
    uppers = [candidate["F2-R-MCP2"][1] for candidate in candidates]

    assert min(lowers) < current_cropped_urdf_limits["F2-R-MCP2"][0]
    assert max(uppers) > current_cropped_urdf_limits["F2-R-MCP2"][1]
    assert all(-0.30 <= lower <= 0.0 for lower in lowers)
    assert all(0.0 <= upper <= 0.35 for upper in uppers)


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
