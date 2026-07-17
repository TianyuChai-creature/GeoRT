"""Exact CPU acceleration for v2 arc medoid selection."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from geort.anchor.mining import LEVEL_FRACTIONS, robust_angle_targets, select_level_medoids
from geort.anchor.arc_bending_v2_robust import orient_projection_to_beta1


def fast_medoid_order(
    rows: np.ndarray,
    descriptors: np.ndarray,
    parameters: np.ndarray,
    source_indices: np.ndarray,
    target: float,
    *,
    chunk_size: int = 64,
) -> np.ndarray:
    """Exactly reproduce `_medoid_order`, moving distance blocks into SciPy C code."""
    rows = np.asarray(rows, dtype=np.int64)
    support = np.asarray(descriptors, dtype=np.float64)[rows]
    sums = np.zeros(rows.size, dtype=np.float64)
    for start in range(0, rows.size, chunk_size):
        stop = min(start + chunk_size, rows.size)
        sums[start:stop] = cdist(support[start:stop], support, metric="euclidean").sum(axis=1)
    return rows[np.lexsort((
        np.asarray(source_indices)[rows],
        np.abs(np.asarray(parameters)[rows] - target),
        sums,
    ))]


def select_robust_arc_medoids_fast(
    tip_points: np.ndarray,
    descriptors: np.ndarray,
    source_indices: np.ndarray,
    *,
    beta1: np.ndarray,
    manifold_bins: int = 64,
    min_support: int = 5,
    max_candidates: int = 256,
    level_band_fraction: float = 0.025,
    band_factors: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> dict:
    """P2–P98 arc selection with the same exact medoid objective as legacy code."""
    points = np.asarray(tip_points, dtype=np.float64)
    descriptors = np.asarray(descriptors, dtype=np.float64)
    sources = np.asarray(source_indices, dtype=np.int64)
    beta1_values = np.asarray(beta1, dtype=np.float64)
    if beta1_values.shape != (points.shape[0],) or not np.all(np.isfinite(beta1_values)):
        raise ValueError("beta1 must be finite [N]")
    centered = points - points.mean(axis=0, keepdims=True)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    variance = np.square(singular)
    projection = centered @ vectors[0]
    projection, direction_flipped = orient_projection_to_beta1(projection, beta1_values)
    clipped = robust_angle_targets(projection, (0.02, 0.98))
    lower, upper = clipped.endpoints
    retained = np.flatnonzero((projection >= lower) & (projection <= upper))
    edges = np.linspace(lower, upper, manifold_bins + 1)
    bin_ids = np.clip(np.searchsorted(edges, projection[retained], side="right") - 1, 0, manifold_bins - 1)
    representatives = []
    for bin_id in np.unique(bin_ids):
        rows = retained[bin_ids == bin_id]
        representatives.append(int(fast_medoid_order(rows, descriptors, projection, sources, float(np.median(projection[rows])))[0]))
    reps = np.asarray(representatives, dtype=np.int64)
    reps = reps[np.argsort(projection[reps], kind="stable")]
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points[reps], axis=0), axis=1))))
    if not np.isfinite(arc[-1]) or arc[-1] <= 0.0:
        raise ValueError("robust arc trajectory is degenerate")
    arc_fractions = arc / arc[-1]
    distribution = np.interp(projection, projection[reps], arc_fractions)
    selection = select_level_medoids(
        distribution[retained], descriptors[retained], sources[retained], LEVEL_FRACTIONS,
        min_support=min_support, max_candidates=max_candidates,
        level_band_fraction=level_band_fraction, band_factors=band_factors,
    )
    return {
        "source_indices": selection.source_indices.astype(np.int64),
        "observed_arc_fractions": selection.observed_parameters.astype(np.float64),
        "support_counts": selection.support_counts.astype(np.int64),
        "distribution_arc_fractions": distribution.astype(np.float64),
        "candidate_count": int(points.shape[0]),
        "explained_variance": float(variance[0] / variance.sum()),
        "populated_bin_count": int(reps.size),
        "domain_clip": "projection_quantiles_0.02_0.98",
        "pc1_direction_reference": "beta1_increasing",
        "pc1_direction_flipped": bool(direction_flipped),
        "endpoint_projection": clipped.endpoints.astype(float).tolist(),
        "selection": selection.to_metadata(),
    }
