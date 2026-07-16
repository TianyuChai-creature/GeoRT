"""P2–P98 robust arc-domain selection for non-thumb v2 bending anchors."""

from __future__ import annotations

import numpy as np

from geort.anchor.mining import (
    LEVEL_FRACTIONS,
    _medoid_order,
    robust_angle_targets,
    select_level_medoids,
)


def select_robust_arc_medoids(
    tip_points: np.ndarray,
    descriptors: np.ndarray,
    source_indices: np.ndarray,
    *,
    manifold_bins: int = 64,
    min_support: int = 5,
    max_candidates: int = 256,
    level_band_fraction: float = 0.025,
    band_factors: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> dict:
    """Select five medoids from raw frames after P2–P98 PC1 domain clipping.

    The 64 exact medoids still define the geometric arc polyline.  Level medoids
    are selected from the original retained D1 frames under the unchanged band
    and support parameters, avoiding artificial support loss from 64 bins.
    """
    points = np.asarray(tip_points, dtype=np.float64)
    descriptors = np.asarray(descriptors, dtype=np.float64)
    sources = np.asarray(source_indices, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 5:
        raise ValueError("tip_points must have shape [N>=5, 3]")
    if descriptors.ndim != 2 or descriptors.shape[0] != points.shape[0]:
        raise ValueError("descriptors must have matching shape [N, D]")
    if sources.shape != (points.shape[0],) or np.unique(sources).size != sources.size:
        raise ValueError("source_indices must be unique [N]")
    centered = points - points.mean(axis=0, keepdims=True)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    variance = np.square(singular)
    explained = float(variance[0] / variance.sum())
    projection = centered @ vectors[0]
    clipped = robust_angle_targets(projection, (0.02, 0.98))
    lower, upper = clipped.endpoints
    retained = np.flatnonzero((projection >= lower) & (projection <= upper))
    edges = np.linspace(lower, upper, manifold_bins + 1)
    bin_ids = np.clip(np.searchsorted(edges, projection[retained], side="right") - 1, 0, manifold_bins - 1)
    representatives: list[int] = []
    for bin_id in np.unique(bin_ids):
        rows = retained[bin_ids == bin_id]
        target = float(np.median(projection[rows]))
        representatives.append(int(_medoid_order(rows, descriptors, projection, sources, target)[0]))
    reps = np.asarray(representatives, dtype=np.int64)
    reps = reps[np.argsort(projection[reps], kind="stable")]
    if reps.size < 5:
        raise ValueError("robust arc domain has fewer than five occupied bins")
    polyline = points[reps]
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))))
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
        "explained_variance": explained,
        "populated_bin_count": int(reps.size),
        "domain_clip": "projection_quantiles_0.02_0.98",
        "endpoint_projection": clipped.endpoints.astype(float).tolist(),
        "selection": selection.to_metadata(),
    }
