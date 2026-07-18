"""Temporary sparse-anchor fixture for QA-report tests."""

from pathlib import Path

import numpy as np

from geort.anchor.mining import mine_human_anchor_records


def write_human_anchor_bundle(path: Path, human_data: Path) -> Path:
    """Mine a test-local 50-row sparse bundle; never writes production data."""
    anchors = mine_human_anchor_records(np.load(human_data, allow_pickle=False))
    np.savez(
        path,
        human_frames=anchors.human_frames,
        human_points=anchors.human_points,
        source_indices=anchors.source_indices,
        finger_indices=anchors.finger_indices,
        finger_names=anchors.finger_names,
        anchor_types=anchors.anchor_types,
        levels=anchors.levels,
        trajectory_t=anchors.trajectory_t,
        target_parameters=anchors.target_parameters,
        observed_parameters=anchors.observed_parameters,
        candidate_counts=anchors.candidate_counts,
        support_counts=anchors.support_counts,
    )
    return path
