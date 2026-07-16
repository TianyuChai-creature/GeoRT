import numpy as np
import torch

from geort.loss import local_motion_loss, partial_chamfer_distance
from geort.motion_frames import build_human_motion_frames, validate_rotation_matrices


def _valid_frames(count=2):
    frames = np.zeros((count, 21, 3), dtype=np.float64)
    for finger, (mcp, pip, dip, tip) in enumerate(
        ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    ):
        offset = 10.0 * finger
        frames[:, mcp] = [offset - 1.0, 1.0, 0.0]
        frames[:, pip] = [offset, 1.0, 0.0]
        frames[:, dip] = [offset, 0.0, 0.0]
        frames[:, tip] = [offset + 1.0, 0.0, 0.0]
    return frames


def test_human_frames_are_orthonormal_and_right_handed():
    rotations, report = build_human_motion_frames(_valid_frames())
    validation = validate_rotation_matrices(rotations)

    assert validation.max_orthogonality_error < 1e-6
    assert validation.max_determinant_error < 1e-6
    assert report.fallback_counts.tolist() == [0, 0, 0, 0, 0]


def test_degenerate_frame_reuses_previous_finger_frame():
    frames = _valid_frames()
    frames[1, 5] = [10.0, 0.0, 0.0]
    frames[1, 6] = [10.5, 0.0, 0.0]
    frames[1, 7] = [11.0, 0.0, 0.0]
    frames[1, 8] = [11.5, 0.0, 0.0]

    rotations, report = build_human_motion_frames(frames)

    np.testing.assert_array_equal(rotations[1, 1], rotations[0, 1])
    assert report.fallback_counts.tolist() == [0, 1, 0, 0, 0]


def test_global_rigid_rotation_left_multiplies_human_frames():
    frames = _valid_frames()
    theta = np.pi / 2.0
    q = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )

    rotations, _ = build_human_motion_frames(frames)
    rotated_rotations, _ = build_human_motion_frames(frames @ q.T)

    np.testing.assert_allclose(rotated_rotations, q @ rotations, atol=1e-6)


def test_local_motion_loss_is_invariant_when_only_human_side_is_rigidly_rotated():
    """The hard comparison: rotate only human vectors/frames, pin robot fixed."""
    human = torch.zeros(64, 5, 3)
    robot = torch.zeros(64, 5, 3)
    human[..., 0] = 1.0
    robot[..., 0] = 1.0
    human_frames = torch.eye(3).reshape(1, 1, 3, 3).expand(64, 5, 3, 3).clone()
    robot_frames = torch.eye(3).reshape(1, 1, 3, 3).expand(64, 5, 3, 3).clone()
    local_before, _ = local_motion_loss(
        human, robot, human_frames=human_frames, robot_frames=robot_frames
    )
    global_before, _ = local_motion_loss(human, robot)

    theta = torch.tensor(np.pi / 3.0)
    q = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta), 0.0], [torch.sin(theta), torch.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    local_after, _ = local_motion_loss(
        human @ q.T, robot, q @ human_frames, robot_frames
    )
    global_after, _ = local_motion_loss(human @ q.T, robot)

    assert abs((local_after - local_before).item()) < 1e-6
    assert abs((global_after - global_before).item()) > 1e-2


def test_partial_chamfer_optionally_returns_nearest_neighbor_indices():
    inputs = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    targets = torch.tensor([[[0.1, 0.0, 0.0], [1.9, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    distance, indices = partial_chamfer_distance(inputs, targets, return_indices=True)
    torch.testing.assert_close(distance, torch.tensor(0.1))
    assert indices.tolist() == [[0, 1]]
