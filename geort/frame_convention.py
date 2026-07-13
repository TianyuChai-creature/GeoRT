"""Shared global hand-frame contract used by mocap and robot kinematics.

GeoRT stores both HTS landmarks and URDF keypoints in one right-handed global
hand frame. The semantic axes are +X out of the palm, +Y from the palm toward
the thumb, and +Z from the palm toward the middle fingertip. Coordinates may
be translated or isotropically scaled by later preprocessing, but Step 3/4 do
not construct or apply per-finger rotations.
"""

from __future__ import annotations

import numpy as np


COORDINATE_CONVENTION = "geort_right_handed_global"
AXIS_SEMANTICS = {
    "x": "palm outward normal",
    "y": "palm center toward thumb",
    "z": "palm center toward middle fingertip",
}
GLOBAL_HAND_BASIS = np.eye(3, dtype=np.float64)


def validate_right_handed_basis(basis: np.ndarray, *, atol: float = 1e-7) -> None:
    """Raise ValueError unless columns form an orthonormal right-handed basis."""
    basis = np.asarray(basis, dtype=np.float64)
    if basis.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 basis, got {basis.shape}")
    if not np.allclose(basis.T @ basis, np.eye(3), atol=atol):
        raise ValueError("Basis is not orthonormal")
    if not np.isclose(np.linalg.det(basis), 1.0, atol=atol):
        raise ValueError("Basis is not right-handed")
