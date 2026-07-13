from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import numpy as np
import pytest

import geort.anchor.human_geometry as human_geometry

from geort.anchor.human_geometry import (
    FINGER_LANDMARKS,
    FINGER_NAMES,
    FingerAngles,
    align_hts_to_palm,
    estimate_finger_angles,
)


EXPECTED_TOPOLOGY = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _rotate_in_palm(vector: np.ndarray, angle: float) -> np.ndarray:
    x, y = vector[:2]
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [cosine * x - sine * y, sine * x + cosine * y, 0.0],
        dtype=np.float64,
    )


def _direction(azimuth: np.ndarray, elevation: float) -> np.ndarray:
    return np.cos(elevation) * azimuth + np.array(
        [0.0, 0.0, np.sin(elevation)], dtype=np.float64
    )


def _articulated_frame(
    *,
    finger: str | None = None,
    alpha: float = 0.0,
    beta: tuple[float, float, float] = (0.0, 0.0, 0.0),
    metacarpal_elevation: float = 0.0,
) -> np.ndarray:
    frame = np.zeros((21, 3), dtype=np.float64)
    frame[0] = [0.0, 0.0, 0.0]
    mcp_positions = {
        "index": np.array([0.5, 0.8, 0.0]),
        "middle": np.array([0.0, 1.0, 0.0]),
        "ring": np.array([-0.25, 0.9, 0.0]),
        "pinky": np.array([-0.5, 0.8, 0.0]),
    }

    for name in FINGER_NAMES[1:]:
        indices = EXPECTED_TOPOLOGY[name]
        mcp = mcp_positions[name]
        base_direction = mcp / np.linalg.norm(mcp)
        current_alpha = alpha if name == finger else 0.0
        current_beta = beta if name == finger else (0.0, 0.0, 0.0)
        current_metacarpal_elevation = metacarpal_elevation if name == finger else 0.0
        frame[indices[0]] = np.linalg.norm(mcp) * _direction(
            base_direction, current_metacarpal_elevation
        )
        azimuth = _rotate_in_palm(base_direction, current_alpha)
        elevations = current_metacarpal_elevation + np.cumsum(current_beta)
        frame[indices[1]] = frame[indices[0]] + _direction(azimuth, elevations[0])
        frame[indices[2]] = frame[indices[1]] + _direction(azimuth, elevations[1])
        frame[indices[3]] = frame[indices[2]] + _direction(azimuth, elevations[2])

    thumb_indices = EXPECTED_TOPOLOGY["thumb"]
    cmc = np.array([0.2, 0.25, 0.0])
    thumb_base = np.array([0.8, 0.6, 0.0])
    thumb_base /= np.linalg.norm(thumb_base)
    thumb_alpha = alpha if finger == "thumb" else 0.0
    thumb_beta = beta if finger == "thumb" else (0.0, 0.0, 0.0)
    thumb_azimuth = _rotate_in_palm(thumb_base, thumb_alpha)
    thumb_elevations = np.cumsum(thumb_beta)
    frame[thumb_indices[0]] = cmc
    frame[thumb_indices[1]] = cmc + thumb_base
    frame[thumb_indices[2]] = frame[thumb_indices[1]] + _direction(
        thumb_azimuth, thumb_elevations[1]
    )
    frame[thumb_indices[3]] = frame[thumb_indices[2]] + _direction(
        thumb_azimuth, thumb_elevations[2]
    )
    return frame


def _proper_rotation() -> np.ndarray:
    axis = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def test_fixed_hts_topology_is_explicit() -> None:
    assert FINGER_NAMES == ("thumb", "index", "middle", "ring", "pinky")
    assert FINGER_LANDMARKS == EXPECTED_TOPOLOGY


def test_align_hts_to_palm_returns_float64_without_mutating_input() -> None:
    frames = np.stack([_articulated_frame(), _articulated_frame()]).astype(np.float32)
    original = frames.copy()

    aligned, valid = align_hts_to_palm(frames)

    assert aligned.shape == (2, 21, 3)
    assert aligned.dtype == np.float64
    assert valid.shape == (2,)
    assert valid.dtype == np.bool_
    assert np.all(valid)
    assert np.array_equal(frames, original)
    assert np.allclose(aligned, frames)


def test_palm_alignment_is_invariant_to_translation_rotation_and_scale() -> None:
    frame = _articulated_frame(finger="index", alpha=0.24, beta=(0.2, 0.2, 0.1))
    rotation = _proper_rotation()
    transformed = 3.7 * (frame @ rotation.T) + np.array([4.0, -3.0, 8.0])

    aligned, valid = align_hts_to_palm(np.stack([frame, transformed]))

    assert valid.tolist() == [True, True]
    assert np.allclose(aligned[0], aligned[1], atol=1e-12)


@pytest.mark.parametrize(
    "frames",
    [
        np.zeros((21, 3)),
        np.zeros((1, 20, 3)),
        np.zeros((1, 21, 2)),
        np.zeros((1, 21, 3, 1)),
    ],
)
def test_palm_alignment_rejects_malformed_overall_shape(frames: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"\[T, 21, 3\]"):
        align_hts_to_palm(frames)


def test_palm_alignment_marks_bad_frames_invalid_with_nan_rows() -> None:
    valid_frame = _articulated_frame()
    nonfinite = valid_frame.copy()
    nonfinite[8, 1] = np.nan
    zero_width = valid_frame.copy()
    zero_width[17] = zero_width[5]
    zero_longitudinal = valid_frame.copy()
    zero_longitudinal[9] = zero_longitudinal[0]
    collinear_axes = valid_frame.copy()
    collinear_axes[5] = [1.0, 0.0, 0.0]
    collinear_axes[17] = [-1.0, 0.0, 0.0]
    collinear_axes[9] = [0.5, 0.0, 0.0]

    aligned, valid = align_hts_to_palm(
        np.stack(
            [valid_frame, nonfinite, zero_width, zero_longitudinal, collinear_axes]
        )
    )

    assert valid.tolist() == [True, False, False, False, False]
    assert np.all(np.isfinite(aligned[0]))
    assert np.all(np.isnan(aligned[1:]))


def test_palm_alignment_marks_finite_subtraction_overflow_invalid() -> None:
    frame = _articulated_frame()
    frame[0] = [-1e308, 0.0, 0.0]
    frame[5] = [1e308, 1.0, 0.0]
    frame[9] = [1e308, 2.0, 0.0]
    frame[17] = [-1e308, 1.0, 0.0]

    with np.errstate(over="ignore", invalid="ignore"):
        aligned, valid = align_hts_to_palm(frame[None])

    assert valid.tolist() == [False]
    assert np.all(np.isnan(aligned[0]))


def test_angle_contract_is_frozen_dict_like_and_read_only() -> None:
    result = estimate_finger_angles(_articulated_frame()[None])

    assert isinstance(result, FingerAngles)
    assert is_dataclass(result)
    assert result["alpha"] is result.alpha
    assert result["beta"] is result.beta
    assert result["valid"] is result.valid
    assert result.alpha.shape == (1, 5)
    assert result.beta.shape == (1, 5, 3)
    assert result.valid.shape == (1, 5)
    assert result.alpha.dtype == np.float64
    assert result.beta.dtype == np.float64
    assert result.valid.dtype == np.bool_
    assert np.all(result.valid)
    with pytest.raises(FrozenInstanceError):
        result.alpha = np.zeros((1, 5))
    with pytest.raises(ValueError):
        result.beta[0, 0, 0] = 1.0
    with pytest.raises(KeyError):
        result["unknown"]


def test_finger_angles_direct_constructor_coerces_contract_dtypes() -> None:
    result = FingerAngles(
        alpha=[[0, 1, 2, 3, 4]],
        beta=np.zeros((1, 5, 3), dtype=np.float32),
        valid=[[1, 1, 1, 1, 1]],
    )

    assert result.alpha.dtype == np.float64
    assert result.beta.dtype == np.float64
    assert result.valid.dtype == np.bool_
    assert result.alpha.shape == (1, 5)
    assert result.beta.shape == (1, 5, 3)
    assert result.valid.shape == (1, 5)
    assert not result.alpha.flags.writeable
    assert not result.beta.flags.writeable
    assert not result.valid.flags.writeable


@pytest.mark.parametrize(
    ("alpha", "beta", "valid"),
    [
        (np.zeros((1, 4)), np.zeros((1, 5, 3)), np.ones((1, 5))),
        (np.zeros((1, 5)), np.zeros((1, 5, 2)), np.ones((1, 5))),
        (np.zeros((1, 5)), np.zeros((1, 5, 3)), np.ones((1, 4))),
        (np.zeros((2, 5)), np.zeros((1, 5, 3)), np.ones((2, 5))),
    ],
)
def test_finger_angles_direct_constructor_rejects_malformed_shapes(
    alpha: np.ndarray, beta: np.ndarray, valid: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="shape|matching"):
        FingerAngles(alpha=alpha, beta=beta, valid=valid)


def test_finger_angles_direct_constructor_requires_finite_valid_values() -> None:
    alpha = np.zeros((1, 5))
    alpha[0, 2] = np.nan

    with pytest.raises(ValueError, match="finite.*valid"):
        FingerAngles(
            alpha=alpha,
            beta=np.zeros((1, 5, 3)),
            valid=np.ones((1, 5), dtype=bool),
        )


def test_finger_angles_direct_constructor_requires_nan_invalid_values() -> None:
    valid = np.ones((1, 5), dtype=bool)
    valid[0, 3] = False

    with pytest.raises(ValueError, match="invalid.*NaN"):
        FingerAngles(
            alpha=np.zeros((1, 5)),
            beta=np.zeros((1, 5, 3)),
            valid=valid,
        )


def test_straight_fingers_have_zero_alpha_and_beta() -> None:
    result = estimate_finger_angles(_articulated_frame()[None])

    assert np.allclose(result.alpha, 0.0, atol=1e-12)
    assert np.allclose(result.beta, 0.0, atol=1e-12)


@pytest.mark.parametrize("alpha", [-0.42, 0.31])
@pytest.mark.parametrize("finger", ["index", "middle", "ring", "pinky"])
def test_nonthumb_alpha_preserves_signed_in_palm_azimuth(
    finger: str, alpha: float
) -> None:
    result = estimate_finger_angles(
        _articulated_frame(finger=finger, alpha=alpha)[None]
    )
    finger_index = FINGER_NAMES.index(finger)

    assert result.valid[0, finger_index]
    assert result.alpha[0, finger_index] == pytest.approx(alpha, abs=1e-12)
    assert np.allclose(result.beta[0, finger_index], 0.0, atol=1e-12)


@pytest.mark.parametrize("beta", [(0.2, 0.2, 0.1), (-0.15, -0.15, -0.075)])
def test_nonthumb_beta_recovers_signed_known_bend_magnitudes(
    beta: tuple[float, float, float],
) -> None:
    frame = _articulated_frame(finger="index", alpha=0.27, beta=beta)

    result = estimate_finger_angles(frame[None])

    assert result.alpha[0, 1] == pytest.approx(0.27, abs=1e-12)
    assert result.beta[0, 1] == pytest.approx(beta, abs=1e-12)
    assert result.beta[0, 1, 0] == pytest.approx(result.beta[0, 1, 1])
    assert result.beta[0, 1, 0] == pytest.approx(2.0 * result.beta[0, 1, 2])


@pytest.mark.parametrize("metacarpal_elevation", [-0.3, 0.25])
def test_beta1_is_relative_to_elevated_metacarpal_baseline(
    metacarpal_elevation: float,
) -> None:
    frame = _articulated_frame(
        finger="index",
        metacarpal_elevation=metacarpal_elevation,
    )

    result = estimate_finger_angles(frame[None])

    assert result.valid[0, 1]
    assert result.beta[0, 1, 0] == pytest.approx(0.0, abs=1e-12)


def test_beta1_recovers_relative_bend_independently_of_alpha_and_baseline() -> None:
    frame = _articulated_frame(
        finger="ring",
        alpha=0.37,
        beta=(0.21, 0.21, 0.105),
        metacarpal_elevation=-0.16,
    )

    result = estimate_finger_angles(frame[None])

    assert result.valid[0, 3]
    assert result.alpha[0, 3] == pytest.approx(0.37, abs=1e-12)
    assert result.beta[0, 3] == pytest.approx((0.21, 0.21, 0.105), abs=1e-12)


@pytest.mark.parametrize("alpha", [-0.35, 0.28])
def test_thumb_alpha_uses_cmc_mcp_and_mcp_ip(alpha: float) -> None:
    frame = _articulated_frame(finger="thumb", alpha=alpha)

    result = estimate_finger_angles(frame[None])

    assert result.valid[0, 0]
    assert result.alpha[0, 0] == pytest.approx(alpha, abs=1e-12)
    assert np.all(np.isfinite(result.beta[0, 0]))


def test_angles_are_invariant_after_rigid_transform_and_scale_alignment() -> None:
    frame = _articulated_frame(finger="ring", alpha=-0.22, beta=(0.18, 0.18, 0.09))
    transformed = 2.3 * (frame @ _proper_rotation().T) + [7.0, -4.0, 2.0]
    aligned, palm_valid = align_hts_to_palm(np.stack([frame, transformed]))

    result = estimate_finger_angles(aligned)

    assert np.all(palm_valid)
    assert np.all(result.valid)
    assert np.allclose(result.alpha[0], result.alpha[1], atol=1e-12)
    assert np.allclose(result.beta[0], result.beta[1], atol=1e-12)


def test_invalid_segment_only_invalidates_its_finger() -> None:
    frame = _articulated_frame()
    frame[7] = frame[6]

    result = estimate_finger_angles(frame[None])

    assert result.valid.tolist() == [[True, False, True, True, True]]
    assert np.all(np.isnan(result.alpha[0, 1]))
    assert np.all(np.isnan(result.beta[0, 1]))
    assert np.all(np.isfinite(result.alpha[0, [0, 2, 3, 4]]))
    assert np.all(np.isfinite(result.beta[0, [0, 2, 3, 4]]))


def test_large_batches_are_consistent_across_bounded_chunks(monkeypatch) -> None:
    monkeypatch.setattr(human_geometry, "_CHUNK_SIZE", 257)
    frame = _articulated_frame(finger="middle", alpha=-0.19, beta=(0.16, 0.16, 0.08))
    frames = np.repeat(frame[None], 5_003, axis=0)
    frames[2_570, 11] = frames[2_570, 10]

    aligned, palm_valid = align_hts_to_palm(frames)
    result = estimate_finger_angles(aligned)
    expected = estimate_finger_angles(aligned[:1])

    assert np.all(palm_valid)
    assert np.allclose(result.alpha[[0, 256, 257, 5_002]], expected.alpha)
    assert np.allclose(result.beta[[0, 256, 257, 5_002]], expected.beta)
    assert result.valid[2_570].tolist() == [True, True, False, True, True]
    assert np.all(np.isnan(result.beta[2_570, 2]))


def test_angle_estimation_rejects_bad_shape_and_handles_nonfinite_finger() -> None:
    with pytest.raises(ValueError, match=r"\[T, 21, 3\]"):
        estimate_finger_angles(np.zeros((21, 3)))

    frame = _articulated_frame()
    frame[15, 0] = np.inf
    result = estimate_finger_angles(frame[None])

    assert result.valid.tolist() == [[True, True, True, False, True]]
    assert np.isnan(result.alpha[0, 3])
    assert np.all(np.isnan(result.beta[0, 3]))
