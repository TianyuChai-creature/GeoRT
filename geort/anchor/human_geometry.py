from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_LANDMARKS: Mapping[str, tuple[int, int, int, int]] = MappingProxyType(
    {
        "thumb": (1, 2, 3, 4),
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }
)

_WRIST = 0
_INDEX_MCP = 5
_MIDDLE_MCP = 9
_PINKY_MCP = 17
_COLLINEAR_TOLERANCE = 1e-12
_CHUNK_SIZE = 16_384
_FINGER_INDICES = np.array(tuple(FINGER_LANDMARKS.values()), dtype=np.intp)


def _read_only(array: NDArray[np.generic]) -> NDArray[np.generic]:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


@dataclass(frozen=True, slots=True)
class FingerAngles(Mapping[str, NDArray[np.generic]]):
    """Immutable per-frame finger angles in thumb-to-pinky order."""

    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        try:
            alpha = np.asarray(self.alpha, dtype=np.float64)
            beta = np.asarray(self.beta, dtype=np.float64)
            valid = np.asarray(self.valid, dtype=np.bool_)
        except (TypeError, ValueError) as error:
            raise ValueError("finger angle arrays must be numeric") from error

        if alpha.ndim != 2 or alpha.shape[1:] != (5,):
            raise ValueError("alpha must have shape [T, 5]")
        if beta.ndim != 3 or beta.shape[1:] != (5, 3):
            raise ValueError("beta must have shape [T, 5, 3]")
        if valid.ndim != 2 or valid.shape[1:] != (5,):
            raise ValueError("valid must have shape [T, 5]")
        if beta.shape[0] != alpha.shape[0] or valid.shape[0] != alpha.shape[0]:
            raise ValueError("alpha, beta, and valid must have matching T")

        finite = np.isfinite(alpha) & np.all(np.isfinite(beta), axis=-1)
        if not np.all(finite[valid]):
            raise ValueError("angles must be finite where valid is true")
        invalid = ~valid
        invalid_is_nan = np.isnan(alpha) & np.all(np.isnan(beta), axis=-1)
        if not np.all(invalid_is_nan[invalid]):
            raise ValueError("invalid angle entries must use NaN")

        object.__setattr__(self, "alpha", _read_only(alpha))
        object.__setattr__(self, "beta", _read_only(beta))
        object.__setattr__(self, "valid", _read_only(valid))

    @overload
    def __getitem__(self, key: Literal["alpha", "beta"]) -> NDArray[np.float64]: ...

    @overload
    def __getitem__(self, key: Literal["valid"]) -> NDArray[np.bool_]: ...

    def __getitem__(self, key: str) -> NDArray[np.generic]:
        if key == "alpha":
            return self.alpha
        if key == "beta":
            return self.beta
        if key == "valid":
            return self.valid
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("alpha", "beta", "valid"))

    def __len__(self) -> int:
        return 3


def _as_frames(frames: object) -> NDArray[np.float64]:
    try:
        array = np.asarray(frames, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("frames must have shape [T, 21, 3]") from error
    if array.ndim != 3 or array.shape[1:] != (21, 3):
        raise ValueError("frames must have shape [T, 21, 3]")
    return array


def _norms(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    scales = np.max(np.abs(vectors), axis=-1)
    usable = np.isfinite(scales) & (scales > 0.0)
    scaled = np.zeros_like(vectors)
    np.divide(vectors, scales[..., None], out=scaled, where=usable[..., None])
    norms = np.zeros(scales.shape, dtype=np.float64)
    norms[usable] = scales[usable] * np.sqrt(np.sum(np.square(scaled[usable]), axis=-1))
    return norms


def _normalize(
    vectors: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    lengths = _norms(vectors)
    valid = (
        np.all(np.isfinite(vectors), axis=-1) & np.isfinite(lengths) & (lengths > 0.0)
    )
    units = np.zeros_like(vectors)
    np.divide(vectors, lengths[..., None], out=units, where=valid[..., None])
    return units, valid


def align_hts_to_palm(
    frames: object,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Normalize HTS landmarks into a scale-free right-handed palm frame."""
    points = _as_frames(frames)
    aligned = np.full(points.shape, np.nan, dtype=np.float64)
    valid = np.zeros(points.shape[0], dtype=np.bool_)

    for start in range(0, points.shape[0], _CHUNK_SIZE):
        stop = min(start + _CHUNK_SIZE, points.shape[0])
        chunk = points[start:stop]
        with np.errstate(over="ignore", invalid="ignore"):
            origin = chunk[:, _WRIST]
            centered = chunk - origin[:, None, :]
            longitudinal = chunk[:, _MIDDLE_MCP] - origin
            lateral = chunk[:, _INDEX_MCP] - chunk[:, _PINKY_MCP]

        longitudinal_axis, longitudinal_valid = _normalize(longitudinal)
        palm_width = _norms(lateral)
        width_valid = (
            np.all(np.isfinite(lateral), axis=-1)
            & np.isfinite(palm_width)
            & (palm_width > 0.0)
        )
        lateral_orthogonal = lateral - (
            np.sum(lateral * longitudinal_axis, axis=-1)[:, None] * longitudinal_axis
        )
        lateral_length = _norms(lateral_orthogonal)
        lateral_valid = (
            np.all(np.isfinite(lateral_orthogonal), axis=-1)
            & np.isfinite(lateral_length)
            & (lateral_length > _COLLINEAR_TOLERANCE * palm_width)
        )
        lateral_axis = np.zeros_like(lateral_orthogonal)
        np.divide(
            lateral_orthogonal,
            lateral_length[:, None],
            out=lateral_axis,
            where=lateral_valid[:, None],
        )
        normal = np.cross(lateral_axis, longitudinal_axis)
        normal_axis, normal_valid = _normalize(normal)

        chunk_valid = (
            np.all(np.isfinite(chunk), axis=(1, 2))
            & np.all(np.isfinite(centered), axis=(1, 2))
            & longitudinal_valid
            & width_valid
            & lateral_valid
            & normal_valid
        )
        selected = np.flatnonzero(chunk_valid)
        if selected.size == 0:
            continue

        selected_centered = centered[selected]
        selected_width = palm_width[selected, None]
        global_indices = start + selected
        aligned[global_indices, :, 0] = (
            np.einsum(
                "tjc,tc->tj", selected_centered, lateral_axis[selected], optimize=True
            )
            / selected_width
        )
        aligned[global_indices, :, 1] = (
            np.einsum(
                "tjc,tc->tj",
                selected_centered,
                longitudinal_axis[selected],
                optimize=True,
            )
            / selected_width
        )
        aligned[global_indices, :, 2] = (
            np.einsum(
                "tjc,tc->tj", selected_centered, normal_axis[selected], optimize=True
            )
            / selected_width
        )
        finite_output = np.all(np.isfinite(aligned[global_indices]), axis=(1, 2))
        valid[global_indices[finite_output]] = True
        aligned[global_indices[~finite_output]] = np.nan

    return aligned, valid


def _azimuth(
    first_projection: NDArray[np.float64],
    second_projection: NDArray[np.float64],
) -> NDArray[np.float64]:
    dot = np.clip(np.sum(first_projection * second_projection, axis=-1), -1.0, 1.0)
    cross = (
        first_projection[..., 0] * second_projection[..., 1]
        - first_projection[..., 1] * second_projection[..., 0]
    )
    return np.arctan2(cross, dot)


def _elevation(units: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.arctan2(units[..., 2], np.hypot(units[..., 0], units[..., 1]))


def _flexion(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    first_projection: NDArray[np.float64],
) -> NDArray[np.float64]:
    cross_x = first[..., 1] * second[..., 2] - first[..., 2] * second[..., 1]
    cross_y = first[..., 2] * second[..., 0] - first[..., 0] * second[..., 2]
    signed_sine = (
        cross_x * first_projection[..., 1] - cross_y * first_projection[..., 0]
    )
    dot = np.clip(np.sum(first * second, axis=-1), -1.0, 1.0)
    return np.arctan2(signed_sine, dot)


def estimate_finger_angles(aligned: object) -> FingerAngles:
    """Estimate signed lateral and flexion angles from palm-aligned landmarks."""
    frames = _as_frames(aligned)
    alpha = np.full((frames.shape[0], 5), np.nan, dtype=np.float64)
    beta = np.full((frames.shape[0], 5, 3), np.nan, dtype=np.float64)
    valid = np.zeros((frames.shape[0], 5), dtype=np.bool_)

    for start in range(0, frames.shape[0], _CHUNK_SIZE):
        stop = min(start + _CHUNK_SIZE, frames.shape[0])
        chunk = frames[start:stop]
        with np.errstate(over="ignore", invalid="ignore"):
            joints = chunk[:, _FINGER_INDICES]
            segments = np.diff(joints, axis=2)
            baseline = np.empty_like(segments[:, :, 0])
            baseline[:, 0] = segments[:, 0, 0]
            baseline[:, 1:] = joints[:, 1:, 0] - chunk[:, None, _WRIST]

        baseline_unit, baseline_valid = _normalize(baseline)
        segment_unit, segment_valid = _normalize(segments)
        baseline_projection, baseline_projection_valid = _normalize(
            baseline_unit[..., :2]
        )
        segment_projection, segment_projection_valid = _normalize(segment_unit[..., :2])

        alpha_target_projection = segment_projection[:, :, 0].copy()
        alpha_target_valid = segment_projection_valid[:, :, 0].copy()
        alpha_target_projection[:, 0] = segment_projection[:, 0, 1]
        alpha_target_valid[:, 0] = segment_projection_valid[:, 0, 1]

        chunk_alpha = _azimuth(baseline_projection, alpha_target_projection)
        baseline_elevation = _elevation(baseline_unit)
        segment_elevation = _elevation(segment_unit)
        chunk_beta = np.empty((stop - start, 5, 3), dtype=np.float64)
        chunk_beta[:, :, 0] = segment_elevation[:, :, 0] - baseline_elevation
        chunk_beta[:, 0, 0] = baseline_elevation[:, 0]
        chunk_beta[:, :, 1] = _flexion(
            segment_unit[:, :, 0],
            segment_unit[:, :, 1],
            segment_projection[:, :, 0],
        )
        chunk_beta[:, :, 2] = _flexion(
            segment_unit[:, :, 1],
            segment_unit[:, :, 2],
            segment_projection[:, :, 1],
        )

        chunk_valid = (
            baseline_valid
            & np.all(segment_valid, axis=2)
            & baseline_projection_valid
            & alpha_target_valid
            & segment_projection_valid[:, :, 0]
            & segment_projection_valid[:, :, 1]
            & np.isfinite(chunk_alpha)
            & np.all(np.isfinite(chunk_beta), axis=-1)
        )
        alpha[start:stop] = np.where(chunk_valid, chunk_alpha, np.nan)
        beta[start:stop] = np.where(chunk_valid[..., None], chunk_beta, np.nan)
        valid[start:stop] = chunk_valid

    return FingerAngles(alpha=alpha, beta=beta, valid=valid)
