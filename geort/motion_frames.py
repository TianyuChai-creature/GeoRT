"""Shared local coordinate-frame construction for motion preservation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


SEGMENT_NORM_EPS_M = 1e-6
COLLINEAR_SINE_EPS = 1e-3
FINGER_LANDMARKS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)


@dataclass(frozen=True)
class HumanFrameReport:
    fallback_counts: np.ndarray


@dataclass(frozen=True)
class RotationValidation:
    max_orthogonality_error: float
    max_determinant_error: float


def _normalise(vector: np.ndarray) -> tuple[np.ndarray, float]:
    length = float(np.linalg.norm(vector))
    return vector / max(length, SEGMENT_NORM_EPS_M), length


def build_human_motion_frames(frames_21: np.ndarray) -> tuple[np.ndarray, HumanFrameReport]:
    """Build column-basis TIP frames from raw [T, 21, 3] hand-base points.

    The fixed cross order is ``e2 = e1 × projected(PIP-DIP)`` and
    ``e3 = e1 × e2``.  For the positive (palmward) β convention this makes
    ``d e1/dβ · (e2 × e1)`` positive; e2/e3 are both flipped together only
    through this construction, never by a fitted numerical calibration.
    """
    frames = np.asarray(frames_21, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError(f"Expected [T, 21, 3] frames, got {frames.shape}")
    rotations = np.empty((frames.shape[0], 5, 3, 3), dtype=np.float64)
    fallback_counts = np.zeros(5, dtype=np.int64)
    previous = np.broadcast_to(np.eye(3), (5, 3, 3)).copy()

    for frame_index, frame in enumerate(frames):
        for finger_index, (mcp, pip, dip, tip) in enumerate(FINGER_LANDMARKS):
            distal_base = dip
            reference_base = pip
            if finger_index == 0:
                distal_base = pip
                reference_base = mcp
            e1, e1_norm = _normalise(frame[tip] - frame[distal_base])
            reference, reference_norm = _normalise(frame[reference_base] - frame[distal_base])
            projected = reference - np.dot(reference, e1) * e1
            projected_norm = float(np.linalg.norm(projected))
            sine = projected_norm / max(reference_norm, SEGMENT_NORM_EPS_M)
            valid = (
                np.isfinite(e1).all()
                and np.isfinite(projected).all()
                and e1_norm >= SEGMENT_NORM_EPS_M
                and reference_norm >= SEGMENT_NORM_EPS_M
                and sine >= COLLINEAR_SINE_EPS
            )
            if not valid:
                rotations[frame_index, finger_index] = previous[finger_index]
                fallback_counts[finger_index] += 1
                continue
            plane = projected / projected_norm
            e2 = np.cross(e1, plane)
            e2, _ = _normalise(e2)
            e3 = np.cross(e1, e2)
            current = np.stack((e1, e2, e3), axis=-1)
            rotations[frame_index, finger_index] = current
            previous[finger_index] = current
    return rotations, HumanFrameReport(fallback_counts=fallback_counts)


def validate_rotation_matrices(rotations: np.ndarray) -> RotationValidation:
    """Return maximum orthogonality and determinant deviations for column frames."""
    values = np.asarray(rotations, dtype=np.float64)
    if values.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [..., 3, 3] rotations, got {values.shape}")
    gram = np.swapaxes(values, -1, -2) @ values
    eye = np.eye(3, dtype=values.dtype)
    return RotationValidation(
        max_orthogonality_error=float(np.max(np.abs(gram - eye))),
        max_determinant_error=float(np.max(np.abs(np.linalg.det(values) - 1.0))),
    )


# Columns are [c1, c2, c3] in each distal-link local frame.  c1 is the exact
# configured TIP offset direction; c2 is the URDF DIP axis Gram-Schmidt
# orthogonalised against c1; c3=c1×c2.  The numeric derivative convention
# sign(d e1/d beta · (c2×e1)) is positive for all five custom_right fingers.
CUSTOM_RIGHT_TIP_FRAME_CONSTANTS = np.asarray(
    (
        ((0.6882472016, -0.1383408417, 0.7121640268),
         (0.6882472016, -0.1859183449, -0.7012489989),
         (0.2294157339,  0.9727775597, -0.0327450834)),
        ((-0.2922185356, 0.9563515711, 0.0),
         ( 0.9563515711, 0.2922185356, 0.0),
         ( 0.0,          0.0,         -1.0)),
        ((-0.2922185356, 0.9563515711, 0.0),
         ( 0.9563515711, 0.2922185356, 0.0),
         ( 0.0,          0.0,         -1.0)),
        ((-0.2922185356, 0.9563515711, 0.0),
         ( 0.9563515711, 0.2922185356, 0.0),
         ( 0.0,          0.0,         -1.0)),
        ((-0.2922185356, 0.9563515711, 0.0),
         ( 0.9563515711, 0.2922185356, 0.0),
         ( 0.0,          0.0,         -1.0)),
    ),
    dtype=np.float64,
)
CUSTOM_RIGHT_DIP_AXIS_FLIPPED = (False, False, False, False, False)
CUSTOM_RIGHT_RAW_DIP_AXIS_ANGLES_DEG = np.asarray(
    (80.9375158622, 73.0091767080, 73.0091767080, 73.0091767080, 73.0091767080),
    dtype=np.float64,
)


def robot_task_frames(
    link_rotations: torch.Tensor,
    *,
    constants: np.ndarray = CUSTOM_RIGHT_TIP_FRAME_CONSTANTS,
) -> torch.Tensor:
    """Convert distal-link rotations [B,5,3,3] to task frames R_link @ C."""
    if link_rotations.ndim != 4 or link_rotations.shape[1:] != (5, 3, 3):
        raise ValueError(f"Expected link rotations [B,5,3,3], got {tuple(link_rotations.shape)}")
    constant_tensor = torch.as_tensor(constants, dtype=link_rotations.dtype, device=link_rotations.device)
    return link_rotations @ constant_tensor


def validate_rotation_matrices_torch(rotations: torch.Tensor) -> tuple[float, float]:
    """Return max |RᵀR-I| and |det(R)-1| without changing the training graph."""
    if rotations.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [...,3,3] rotations, got {tuple(rotations.shape)}")
    with torch.no_grad():
        eye = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
        orth = (rotations.transpose(-1, -2) @ rotations - eye).abs().max().item()
        determinant = (torch.linalg.det(rotations) - 1.0).abs().max().item()
    return float(orth), float(determinant)
