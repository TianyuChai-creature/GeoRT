from __future__ import annotations

import copy

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


_PAIRWISE_CHUNK_SIZE = 64


def _read_only(array: NDArray[np.generic]) -> NDArray[np.generic]:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


LEVEL_FRACTIONS = _read_only(
    np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
)


@dataclass(frozen=True, slots=True)
class RobustAngleTargets:
    """Robust angular endpoints and geometrically spaced target levels."""

    endpoints: NDArray[np.float64]
    targets: NDArray[np.float64]
    _endpoint_support: tuple[tuple[str, Any], ...]

    @property
    def endpoint_support(self) -> dict[str, Any]:
        """Return a defensive, JSON-serializable endpoint-support summary."""
        return copy.deepcopy(dict(self._endpoint_support))


@dataclass(frozen=True, slots=True)
class BandExpansion:
    """One attempted local parameter band."""

    factor: float
    half_width: float
    candidate_count: int
    support_count: int


@dataclass(frozen=True, slots=True)
class LevelMedoidSelection:
    """Five real source frames selected for geometric parameter levels."""

    selected_row_indices: NDArray[np.int64]
    source_indices: NDArray[np.int64]
    observed_parameters: NDArray[np.float64]
    support_counts: NDArray[np.int64]
    expansion_history: tuple[tuple[BandExpansion, ...], ...]

    def to_metadata(self) -> dict[str, Any]:
        """Return the selection diagnostics in JSON-serializable form."""
        return {
            "selected_row_indices": self.selected_row_indices.tolist(),
            "source_indices": self.source_indices.tolist(),
            "observed_parameters": self.observed_parameters.tolist(),
            "support_counts": self.support_counts.tolist(),
            "expansion_history": [
                [
                    {
                        "factor": attempt.factor,
                        "half_width": attempt.half_width,
                        "candidate_count": attempt.candidate_count,
                        "support_count": attempt.support_count,
                    }
                    for attempt in history
                ]
                for history in self.expansion_history
            ],
        }


@dataclass(frozen=True, slots=True)
class FallbackAttempt:
    """Candidate count and effective tolerances for one fallback factor."""

    factor: float
    count: int
    straight_tol: float
    alpha_zero_tol: float
    coupling_tol: float


@dataclass(frozen=True, slots=True)
class MotionCandidateSelection:
    """First candidate mask satisfying the requested minimum support."""

    mask: NDArray[np.bool_]
    count: int
    factor: float
    effective_straight_tol: float
    effective_alpha_zero_tol: float
    effective_coupling_tol: float
    attempted_history: tuple[FallbackAttempt, ...]

    def to_metadata(self) -> dict[str, Any]:
        """Return candidate diagnostics in JSON-serializable form."""
        return {
            "count": self.count,
            "factor": self.factor,
            "effective_tolerances": {
                "straight_tol": self.effective_straight_tol,
                "alpha_zero_tol": self.effective_alpha_zero_tol,
                "coupling_tol": self.effective_coupling_tol,
            },
            "attempted_fallback_history": [
                {
                    "factor": attempt.factor,
                    "count": attempt.count,
                    "straight_tol": attempt.straight_tol,
                    "alpha_zero_tol": attempt.alpha_zero_tol,
                    "coupling_tol": attempt.coupling_tol,
                }
                for attempt in self.attempted_history
            ],
        }


@dataclass(frozen=True, slots=True)
class ThumbArcMedoidSelection:
    """Five real thumb-tip frames selected along a PCA arc-length trajectory."""

    selected_row_indices: NDArray[np.int64]
    source_indices: NDArray[np.int64]
    distribution_arc_fractions: NDArray[np.float64]
    observed_arc_fractions: NDArray[np.float64]
    support_counts: NDArray[np.int64]
    populated_bin_count: int
    explained_variance: float
    level_selection: LevelMedoidSelection

    def to_metadata(self) -> dict[str, Any]:
        """Return the main-trajectory and medoid diagnostics."""
        return {
            "selected_row_indices": self.selected_row_indices.tolist(),
            "source_indices": self.source_indices.tolist(),
            "distribution_arc_fractions": self.distribution_arc_fractions.tolist(),
            "observed_arc_fractions": self.observed_arc_fractions.tolist(),
            "support_counts": self.support_counts.tolist(),
            "populated_bin_count": self.populated_bin_count,
            "explained_variance": self.explained_variance,
            "level_selection": self.level_selection.to_metadata(),
        }


@dataclass(frozen=True, slots=True)
class MinedHumanAnchors:
    """Fifty ordered human anchors mined from raw HTS landmark frames."""

    human_frames: NDArray[np.float64]
    human_points: NDArray[np.float64]
    source_indices: NDArray[np.int64]
    finger_indices: NDArray[np.int64]
    finger_names: NDArray[np.str_]
    anchor_types: NDArray[np.str_]
    levels: NDArray[np.int64]
    trajectory_t: NDArray[np.float64]
    target_parameters: NDArray[np.float64]
    observed_parameters: NDArray[np.float64]
    candidate_counts: NDArray[np.int64]
    support_counts: NDArray[np.int64]
    group_metadata: dict[str, dict[str, Any]]


class CandidateFilterError(ValueError):
    """Raised when all candidate-filter fallback factors are exhausted."""

    def __init__(
        self,
        message: str,
        attempted_history: tuple[FallbackAttempt, ...],
    ) -> None:
        super().__init__(message)
        self.attempted_history = attempted_history


def _finite_vector(
    values: object, *, name: str, minimum_size: int
) -> NDArray[np.float64]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite one-dimensional array") from error
    if array.ndim != 1 or array.size < minimum_size or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be a finite one-dimensional array with at least "
            f"{minimum_size} values"
        )
    return array


def _finite_tuple(values: object, *, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def robust_angle_targets(
    values: object,
    endpoint_quantiles: tuple[float, float] = (0.02, 0.98),
) -> RobustAngleTargets:
    """Find outlier-resistant endpoints and interpolate geometric angle levels."""
    samples = _finite_vector(values, name="values", minimum_size=5)
    quantiles = _finite_tuple(endpoint_quantiles, name="endpoint_quantiles")
    if (
        quantiles.shape != (2,)
        or quantiles[0] < 0.0
        or quantiles[1] > 1.0
        or quantiles[0] >= quantiles[1]
    ):
        raise ValueError("endpoint_quantiles must satisfy 0 <= low < high <= 1")

    endpoints = np.quantile(samples, quantiles)
    if not endpoints[1] > endpoints[0]:
        raise ValueError("robust endpoint range is degenerate")
    targets = endpoints[0] + LEVEL_FRACTIONS * (endpoints[1] - endpoints[0])
    lower_rejected = int(np.count_nonzero(samples < endpoints[0]))
    upper_rejected = int(np.count_nonzero(samples > endpoints[1]))
    support = (
        ("sample_count", int(samples.size)),
        ("endpoint_quantiles", quantiles.tolist()),
        ("endpoint_values", endpoints.tolist()),
        ("lower_rejected_count", lower_rejected),
        ("upper_rejected_count", upper_rejected),
        ("retained_count", int(samples.size - lower_rejected - upper_rejected)),
    )
    return RobustAngleTargets(
        endpoints=_read_only(endpoints),
        targets=_read_only(targets),
        _endpoint_support=support,
    )


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _medoid_order(
    rows: NDArray[np.int64],
    descriptors: NDArray[np.float64],
    parameters: NDArray[np.float64],
    source_indices: NDArray[np.int64],
    target: float,
) -> NDArray[np.int64]:
    support_descriptors = descriptors[rows]
    descriptor_scale = float(np.max(np.abs(support_descriptors)))
    if descriptor_scale > 0.0:
        support_descriptors = support_descriptors / descriptor_scale
    distance_sums = np.zeros(rows.size, dtype=np.float64)
    for row_start in range(0, rows.size, _PAIRWISE_CHUNK_SIZE):
        row_stop = min(row_start + _PAIRWISE_CHUNK_SIZE, rows.size)
        for column_start in range(0, rows.size, _PAIRWISE_CHUNK_SIZE):
            column_stop = min(column_start + _PAIRWISE_CHUNK_SIZE, rows.size)
            differences = (
                support_descriptors[row_start:row_stop, None, :]
                - support_descriptors[None, column_start:column_stop, :]
            )
            distance_sums[row_start:row_stop] += np.sqrt(
                np.sum(np.square(differences), axis=-1)
            ).sum(axis=1)
    order = np.lexsort(
        (
            source_indices[rows],
            np.abs(parameters[rows] - target),
            distance_sums,
        )
    )
    return rows[order]


def select_level_medoids(
    parameters: object,
    descriptors: object,
    source_indices: object,
    targets: object,
    min_support: int = 5,
    max_candidates: int = 256,
    level_band_fraction: float = 0.025,
    band_factors: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> LevelMedoidSelection:
    """Select distinct, monotonic real frames using exact local medoids."""
    parameter_values = _finite_vector(parameters, name="parameters", minimum_size=5)
    try:
        descriptor_values = np.asarray(descriptors, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "descriptors must be a finite array with shape [N, D]"
        ) from error
    if (
        descriptor_values.ndim != 2
        or descriptor_values.shape[0] != parameter_values.size
        or descriptor_values.shape[1] == 0
        or not np.all(np.isfinite(descriptor_values))
    ):
        raise ValueError("descriptors must be a finite array with shape [N, D]")

    raw_source_indices = np.asarray(source_indices)
    if (
        raw_source_indices.ndim != 1
        or raw_source_indices.shape != parameter_values.shape
        or raw_source_indices.dtype.kind not in "iu"
        or raw_source_indices.dtype.kind == "b"
    ):
        raise ValueError("source_indices must be an integer array with shape [N]")
    source_values = raw_source_indices.astype(np.int64, copy=False)
    if np.unique(source_values).size != source_values.size:
        raise ValueError("source_indices must be unique")

    target_values = _finite_vector(targets, name="targets", minimum_size=5)
    if target_values.shape != (5,) or np.any(np.diff(target_values) <= 0.0):
        raise ValueError("targets must contain five strictly increasing values")

    minimum = _positive_integer(min_support, name="min_support")
    maximum = _positive_integer(max_candidates, name="max_candidates")
    if maximum < minimum:
        raise ValueError("max_candidates must be at least min_support")
    band_fraction = _positive_tolerance(level_band_fraction, name="level_band_fraction")
    factors = _finite_tuple(band_factors, name="band_factors")
    if (
        factors.size == 0
        or factors[0] != 1.0
        or np.any(factors <= 0.0)
        or np.any(np.diff(factors) <= 0.0)
    ):
        raise ValueError(
            "band_factors must start at 1 and be strictly increasing and positive"
        )

    robust_range = float(target_values[-1] - target_values[0])
    base_half_width = band_fraction * robust_range

    def local_support(
        level: int, used_rows: frozenset[int]
    ) -> tuple[NDArray[np.int64], tuple[BandExpansion, ...]] | None:
        available = np.ones(parameter_values.size, dtype=np.bool_)
        if used_rows:
            available[np.fromiter(used_rows, dtype=np.int64)] = False
        history: list[BandExpansion] = []
        for factor in factors:
            half_width = base_half_width * float(factor)
            rows = np.flatnonzero(
                available
                & (np.abs(parameter_values - target_values[level]) <= half_width)
            ).astype(np.int64, copy=False)
            support_count = min(rows.size, maximum)
            history.append(
                BandExpansion(
                    factor=float(factor),
                    half_width=half_width,
                    candidate_count=int(rows.size),
                    support_count=int(support_count),
                )
            )
            if rows.size >= minimum:
                nearest_order = np.lexsort(
                    (
                        source_values[rows],
                        np.abs(parameter_values[rows] - target_values[level]),
                    )
                )
                return rows[nearest_order[:maximum]], tuple(history)
        return None

    for level in range(5):
        if local_support(level, frozenset()) is None:
            raise ValueError(
                f"level {level} cannot reach min_support within band expansion"
            )

    def search(
        level: int,
        used_rows: frozenset[int],
        previous_parameter: float,
    ) -> (
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[tuple[BandExpansion, ...], ...],
        ]
        | None
    ):
        if level == 5:
            return (), (), ()
        support_result = local_support(level, used_rows)
        if support_result is None:
            return None
        support_rows, history = support_result
        ranked_rows = _medoid_order(
            support_rows,
            descriptor_values,
            parameter_values,
            source_values,
            float(target_values[level]),
        )
        for row_value in ranked_rows:
            row = int(row_value)
            if parameter_values[row] <= previous_parameter:
                continue
            remainder = search(
                level + 1,
                used_rows | {row},
                float(parameter_values[row]),
            )
            if remainder is not None:
                rows, counts, histories = remainder
                return (
                    (row, *rows),
                    (int(support_rows.size), *counts),
                    (history, *histories),
                )
        return None

    selected = search(0, frozenset(), -np.inf)
    if selected is None:
        raise ValueError("distinct strictly monotonic medoid selection is impossible")
    selected_rows, support_counts, histories = selected
    row_array = np.asarray(selected_rows, dtype=np.int64)
    return LevelMedoidSelection(
        selected_row_indices=_read_only(row_array),
        source_indices=_read_only(source_values[row_array]),
        observed_parameters=_read_only(parameter_values[row_array]),
        support_counts=_read_only(np.asarray(support_counts, dtype=np.int64)),
        expansion_history=histories,
    )


def _positive_tolerance(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be positive and finite")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be positive and finite") from error
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return converted


def filter_motion_candidates(
    alpha: object,
    beta: object,
    valid: object,
    anchor_type: str,
    straight_tol: float,
    alpha_zero_tol: float,
    coupling_tol: float,
    min_candidates: int = 5,
    fallback_factors: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0),
) -> MotionCandidateSelection:
    """Filter lateral or non-thumb bending frames with bounded relaxation."""
    try:
        alpha_values = np.asarray(alpha, dtype=np.float64)
        beta_values = np.asarray(beta, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha and beta must be numeric arrays") from error
    if alpha_values.ndim != 1:
        raise ValueError("alpha must have shape [N]")
    if beta_values.ndim != 2 or beta_values.shape != (alpha_values.size, 3):
        raise ValueError("beta must have shape [N, 3]")

    valid_values = np.asarray(valid)
    if (
        valid_values.ndim != 1
        or valid_values.shape != alpha_values.shape
        or valid_values.dtype.kind != "b"
    ):
        raise ValueError("valid must be a boolean array with shape [N]")
    if anchor_type not in {"lateral", "bending"}:
        raise ValueError("anchor_type must be 'lateral' or 'bending'")

    straight = _positive_tolerance(straight_tol, name="straight_tol")
    alpha_zero = _positive_tolerance(alpha_zero_tol, name="alpha_zero_tol")
    coupling = _positive_tolerance(coupling_tol, name="coupling_tol")
    minimum = _positive_integer(min_candidates, name="min_candidates")
    factors = _finite_tuple(fallback_factors, name="fallback_factors")
    if (
        factors.size == 0
        or factors[0] != 1.0
        or np.any(factors <= 0.0)
        or np.any(np.diff(factors) <= 0.0)
    ):
        raise ValueError(
            "fallback_factors must start at 1 and be strictly increasing and positive"
        )

    finite_and_valid = (
        valid_values
        & np.isfinite(alpha_values)
        & np.all(np.isfinite(beta_values), axis=1)
    )
    history: list[FallbackAttempt] = []
    for factor_value in factors:
        factor = float(factor_value)
        effective_straight = straight * factor
        effective_alpha_zero = alpha_zero * factor
        effective_coupling = coupling * factor
        if anchor_type == "lateral":
            predicate = np.max(np.abs(beta_values), axis=1) <= effective_straight
        else:
            coupling_error = np.maximum(
                np.abs(beta_values[:, 0] - beta_values[:, 1]),
                np.abs(beta_values[:, 0] - 2.0 * beta_values[:, 2]),
            )
            predicate = (np.abs(alpha_values) <= effective_alpha_zero) & (
                coupling_error <= effective_coupling
            )
        mask = finite_and_valid & predicate
        count = int(np.count_nonzero(mask))
        history.append(
            FallbackAttempt(
                factor=factor,
                count=count,
                straight_tol=effective_straight,
                alpha_zero_tol=effective_alpha_zero,
                coupling_tol=effective_coupling,
            )
        )
        if count >= minimum:
            return MotionCandidateSelection(
                mask=_read_only(mask),
                count=count,
                factor=factor,
                effective_straight_tol=effective_straight,
                effective_alpha_zero_tol=effective_alpha_zero,
                effective_coupling_tol=effective_coupling,
                attempted_history=tuple(history),
            )

    raise CandidateFilterError(
        "candidate fallback factors exhausted before reaching min_candidates",
        tuple(history),
    )


def _finite_points(values: object, *, name: str, minimum_size: int) -> NDArray[np.float64]:
    try:
        points = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite array with shape [N, 3]") from error
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or points.shape[0] < minimum_size
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(f"{name} must be a finite array with shape [N, 3]")
    return points


def _source_index_vector(values: object, *, size: int) -> NDArray[np.int64]:
    raw_indices = np.asarray(values)
    if (
        raw_indices.ndim != 1
        or raw_indices.shape != (size,)
        or raw_indices.dtype.kind not in "iu"
        or raw_indices.dtype.kind == "b"
    ):
        raise ValueError("source_indices must be an integer array with shape [N]")
    indices = raw_indices.astype(np.int64, copy=False)
    if np.unique(indices).size != indices.size:
        raise ValueError("source_indices must be unique")
    return indices


def select_thumb_arc_medoids(
    tip_points: object,
    descriptors: object,
    source_indices: object,
    *,
    endpoint_quantiles: tuple[float, float] = (0.02, 0.98),
    manifold_bins: int = 64,
    min_support: int = 5,
    max_candidates: int = 256,
    level_band_fraction: float = 0.025,
    band_factors: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> ThumbArcMedoidSelection:
    """Sample five real thumb frames from a one-dimensional PCA trajectory."""
    points = _finite_points(tip_points, name="tip_points", minimum_size=5)
    try:
        descriptor_values = np.asarray(descriptors, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("descriptors must be a finite array with shape [N, D]") from error
    if (
        descriptor_values.ndim != 2
        or descriptor_values.shape[0] != points.shape[0]
        or descriptor_values.shape[1] == 0
        or not np.all(np.isfinite(descriptor_values))
    ):
        raise ValueError("descriptors must be a finite array with shape [N, D]")
    sources = _source_index_vector(source_indices, size=points.shape[0])
    bins = _positive_integer(manifold_bins, name="manifold_bins")
    minimum = _positive_integer(min_support, name="min_support")

    centered = points - np.mean(points, axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    variance = np.square(singular_values)
    total_variance = float(np.sum(variance))
    if not np.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError("thumb tip trajectory is degenerate")
    explained_variance = float(variance[0] / total_variance)
    projection = centered @ right_vectors[0]
    robust_targets = robust_angle_targets(projection, endpoint_quantiles)
    lower, upper = robust_targets.endpoints
    retained_rows = np.flatnonzero((projection >= lower) & (projection <= upper))
    if retained_rows.size < 5:
        raise ValueError("thumb trajectory retains fewer than five frames")

    edges = np.linspace(lower, upper, bins + 1)
    retained_bins = np.clip(
        np.searchsorted(edges, projection[retained_rows], side="right") - 1,
        0,
        bins - 1,
    )
    representative_rows: list[int] = []
    for bin_index in np.unique(retained_bins):
        rows = retained_rows[retained_bins == bin_index]
        target = float(np.median(projection[rows]))
        representative_rows.append(
            int(_medoid_order(rows, descriptor_values, projection, sources, target)[0])
        )
    representatives = np.asarray(representative_rows, dtype=np.int64)
    representatives = representatives[np.argsort(projection[representatives], kind="stable")]
    if representatives.size < 5:
        raise ValueError("thumb trajectory has fewer than five populated manifold bins")

    polyline = points[representatives]
    segment_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    arc_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if not arc_length[-1] > 0.0 or not np.isfinite(arc_length[-1]):
        raise ValueError("thumb trajectory arc length is degenerate")
    arc_fractions = arc_length / arc_length[-1]
    distribution_arc_fractions = np.interp(projection, projection[representatives], arc_fractions)
    selection = select_level_medoids(
        arc_fractions,
        descriptor_values[representatives],
        sources[representatives],
        LEVEL_FRACTIONS,
        min_support=minimum,
        max_candidates=max_candidates,
        level_band_fraction=level_band_fraction,
        band_factors=band_factors,
    )
    selected_rows = representatives[selection.selected_row_indices]
    return ThumbArcMedoidSelection(
        selected_row_indices=_read_only(selected_rows),
        source_indices=_read_only(sources[selected_rows]),
        observed_arc_fractions=_read_only(arc_fractions[selection.selected_row_indices]),
        support_counts=selection.support_counts,
        distribution_arc_fractions=_read_only(distribution_arc_fractions),
        populated_bin_count=int(representatives.size),
        explained_variance=explained_variance,
        level_selection=selection,
    )


def mine_human_anchor_records(
    frames: object,
    *,
    endpoint_quantiles: tuple[float, float] = (0.02, 0.98),
    straight_tol: float = np.deg2rad(15.0),
    alpha_zero_tol: float = np.deg2rad(10.0),
    coupling_tol: float = np.deg2rad(20.0),
    min_candidates: int = 5,
    min_level_support: int = 5,
    max_medoid_candidates: int = 256,
    level_band_fraction: float = 0.025,
    band_factors: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
    fallback_factors: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0),
    thumb_manifold_bins: int = 64,
) -> MinedHumanAnchors:
    """Mine ordered side-swing and flexion anchors from raw HTS frames."""
    from geort.anchor.human_geometry import (
        FINGER_LANDMARKS,
        FINGER_NAMES,
        align_hts_to_palm,
        estimate_finger_angles,
    )

    try:
        raw_frames = np.asarray(frames, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("frames must have shape [T, 21, 3]") from error
    if raw_frames.ndim != 3 or raw_frames.shape[1:] != (21, 3):
        raise ValueError("frames must have shape [T, 21, 3]")

    aligned, palm_valid = align_hts_to_palm(raw_frames)
    angles = estimate_finger_angles(aligned)
    selected_sources: list[int] = []
    selected_fingers: list[int] = []
    selected_names: list[str] = []
    selected_types: list[str] = []
    selected_levels: list[int] = []
    selected_targets: list[float] = []
    selected_observed: list[float] = []
    selected_candidates: list[int] = []
    selected_support: list[int] = []
    metadata: dict[str, dict[str, Any]] = {}

    for finger_index, finger_name in enumerate(FINGER_NAMES):
        landmarks = FINGER_LANDMARKS[finger_name]
        descriptors = aligned[:, landmarks].reshape(raw_frames.shape[0], -1)
        valid = (
            palm_valid
            & angles.valid[:, finger_index]
            & np.all(np.isfinite(descriptors), axis=1)
        )
        for anchor_type in ("lateral", "bending"):
            key = f"{finger_name}:{anchor_type}"
            if finger_index == 0 and anchor_type == "bending":
                candidate_rows = np.flatnonzero(valid)
                thumb_selection = select_thumb_arc_medoids(
                    aligned[candidate_rows, landmarks[-1]],
                    descriptors[candidate_rows],
                    candidate_rows,
                    endpoint_quantiles=endpoint_quantiles,
                    manifold_bins=thumb_manifold_bins,
                    min_support=min_level_support,
                    max_candidates=max_medoid_candidates,
                    level_band_fraction=level_band_fraction,
                    band_factors=band_factors,
                )
                group_sources = thumb_selection.source_indices
                group_targets = LEVEL_FRACTIONS
                group_observed = thumb_selection.observed_arc_fractions
                group_support = thumb_selection.support_counts
                candidate_count = int(candidate_rows.size)
                metadata[key] = {
                    "distribution_parameter": "thumb_tip_arc_fraction",
                    "distribution_values": thumb_selection.distribution_arc_fractions.tolist(),
                    "candidate_count": candidate_count,
                    "selection": thumb_selection.to_metadata(),
                }
            else:
                candidates = filter_motion_candidates(
                    angles.alpha[:, finger_index],
                    angles.beta[:, finger_index],
                    valid,
                    anchor_type,
                    straight_tol=straight_tol,
                    alpha_zero_tol=alpha_zero_tol,
                    coupling_tol=coupling_tol,
                    min_candidates=min_candidates,
                    fallback_factors=fallback_factors,
                )
                candidate_rows = np.flatnonzero(candidates.mask)
                parameters = (
                    angles.alpha[candidate_rows, finger_index]
                    if anchor_type == "lateral"
                    else angles.beta[candidate_rows, finger_index, 0]
                )
                targets = robust_angle_targets(parameters, endpoint_quantiles)
                retained = (parameters >= targets.endpoints[0]) & (
                    parameters <= targets.endpoints[1]
                )
                rows = candidate_rows[retained]
                selection = select_level_medoids(
                    parameters[retained],
                    descriptors[rows],
                    rows,
                    targets.targets,
                    min_support=min_level_support,
                    max_candidates=max_medoid_candidates,
                    level_band_fraction=level_band_fraction,
                    band_factors=band_factors,
                )
                group_sources = selection.source_indices
                group_targets = targets.targets
                group_observed = selection.observed_parameters
                group_support = selection.support_counts
                candidate_count = candidates.count
                metadata[key] = {
                    "candidate_filter": candidates.to_metadata(),
                    "robust_targets": {
                        "endpoints": targets.endpoints.tolist(),
                        "targets": targets.targets.tolist(),
                        "endpoint_support": targets.endpoint_support,
                    },
                    "selection": selection.to_metadata(),
                    "distribution_parameter": (
                        "alpha" if anchor_type == "lateral" else "beta1"
                    ),
                    "distribution_values": (
                        angles.alpha[valid, finger_index]
                        if anchor_type == "lateral" else angles.beta[valid, finger_index, 0]
                    ).tolist(),
                }

            selected_sources.extend(int(value) for value in group_sources)
            selected_fingers.extend([finger_index] * 5)
            selected_names.extend([finger_name] * 5)
            selected_types.extend([anchor_type] * 5)
            selected_levels.extend(range(5))
            selected_targets.extend(float(value) for value in group_targets)
            selected_observed.extend(float(value) for value in group_observed)
            selected_candidates.extend([candidate_count] * 5)
            selected_support.extend(int(value) for value in group_support)

    sources = np.asarray(selected_sources, dtype=np.int64)
    point_indices = np.asarray(
        [FINGER_LANDMARKS[name][-1] for name in selected_names], dtype=np.int64
    )
    return MinedHumanAnchors(
        human_frames=_read_only(raw_frames[sources]),
        human_points=_read_only(raw_frames[sources, point_indices]),
        source_indices=_read_only(sources),
        finger_indices=_read_only(np.asarray(selected_fingers, dtype=np.int64)),
        finger_names=_read_only(np.asarray(selected_names, dtype=np.str_)),
        anchor_types=_read_only(np.asarray(selected_types, dtype=np.str_)),
        levels=_read_only(np.asarray(selected_levels, dtype=np.int64)),
        trajectory_t=_read_only(np.tile(LEVEL_FRACTIONS, 10)),
        target_parameters=_read_only(np.asarray(selected_targets, dtype=np.float64)),
        observed_parameters=_read_only(np.asarray(selected_observed, dtype=np.float64)),
        candidate_counts=_read_only(np.asarray(selected_candidates, dtype=np.int64)),
        support_counts=_read_only(np.asarray(selected_support, dtype=np.int64)),
        group_metadata=copy.deepcopy(metadata),
    )
