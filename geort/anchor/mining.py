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
    level_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
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

    fractions = _finite_tuple(level_fractions, name="level_fractions")
    if (
        fractions.size < 2
        or fractions[0] != 0.0
        or fractions[-1] != 1.0
        or np.any(np.diff(fractions) <= 0.0)
    ):
        raise ValueError("level_fractions must be strictly increasing and span 0 to 1")

    endpoints = np.quantile(samples, quantiles)
    if not endpoints[1] > endpoints[0]:
        raise ValueError("robust endpoint range is degenerate")
    targets = endpoints[0] + fractions * (endpoints[1] - endpoints[0])
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
