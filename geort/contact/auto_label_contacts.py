"""Automatically mine D3 contact labels from an existing D1 HTS recording."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np


PAIR_NAMES = ("thumb_index", "thumb_middle", "thumb_ring", "thumb_pinky")
PAIR_LANDMARKS = ((4, 8), (4, 12), (4, 16), (4, 20))
_AMBIGUOUS_LABEL = np.int8(-1)


@dataclass(frozen=True, slots=True)
class ContactLabelConfig:
    """Controls for low-distance platform detection and temporal holdout."""

    lower_quantile: float = 0.005
    upper_quantile: float = 0.995
    histogram_bin_m: float = 0.001
    smoothing_bins: int = 5
    platform_search_quantile: float = 0.35
    minimum_search_span_m: float = 0.030
    ambiguity_band_m: float = 0.010
    min_cluster_frames: int = 50
    holdout_fraction: float = 0.20

    def validate(self) -> None:
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
        if self.minimum_search_span_m <= 0.0 or self.ambiguity_band_m <= 0.0:
            raise ValueError("distance bands must be positive")
        if self.histogram_bin_m <= 0.0 or self.smoothing_bins < 3:
            raise ValueError("histogram smoothing parameters must be positive")
        if self.smoothing_bins % 2 == 0:
            raise ValueError("smoothing_bins must be odd")
        if not self.lower_quantile < self.platform_search_quantile < self.upper_quantile:
            raise ValueError("platform_search_quantile must lie inside endpoint quantiles")
        if self.min_cluster_frames < 1:
            raise ValueError("min_cluster_frames must be positive")
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be between zero and one")


@dataclass(frozen=True, slots=True)
class ContactPairLabels:
    """All frame-level D3 decisions for one thumb-to-finger pair."""

    name: str
    landmark_indices: tuple[int, int]
    frame_indices: np.ndarray
    distances: np.ndarray
    labels: np.ndarray
    held_out: np.ndarray
    positive_threshold: float
    negative_threshold: float
    platform_peak: float
    discarded_segments: tuple[tuple[int, int], ...]

    @property
    def positive_count(self) -> int:
        return int(np.count_nonzero(self.labels == 1))

    @property
    def negative_count(self) -> int:
        return int(np.count_nonzero(self.labels == 0))

    @property
    def discarded_count(self) -> int:
        return int(np.count_nonzero(self.labels == _AMBIGUOUS_LABEL))


@dataclass(frozen=True, slots=True)
class AutoLabeledContacts:
    """D3 labels plus the raw six-coordinate feature vector for each pair."""

    features: np.ndarray
    pairs: tuple[ContactPairLabels, ...]

    @property
    def distances(self) -> np.ndarray:
        return np.column_stack([pair.distances for pair in self.pairs])

    @property
    def labels(self) -> np.ndarray:
        return np.column_stack([pair.labels for pair in self.pairs])

    @property
    def held_out(self) -> np.ndarray:
        return np.column_stack([pair.held_out for pair in self.pairs])


@dataclass(frozen=True, slots=True)
class ContactLabelOutputPaths:
    npz_path: Path
    json_path: Path
    html_path: Path


def _validate_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError(f"HTS input must have shape [T, 21, 3], got {frames.shape}")
    if len(frames) < 2 or not np.isfinite(frames).all():
        raise ValueError("HTS input must contain at least two finite frames")
    return frames


def compute_tip_distances(frames: np.ndarray) -> np.ndarray:
    """Return the four fixed thumb-to-fingertip distance traces in metres."""
    frames = _validate_frames(frames)
    return np.stack(
        [np.linalg.norm(frames[:, thumb] - frames[:, finger], axis=1)
         for thumb, finger in PAIR_LANDMARKS],
        axis=1,
    )


def _contiguous_segments(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True]))) + 1
    return tuple((int(start), int(end)) for start, end in zip(starts, ends, strict=True))


def _detect_platform(
    distances: np.ndarray,
    config: ContactLabelConfig,
) -> tuple[float, float]:
    lower, search_upper, upper = np.quantile(
        distances,
        (config.lower_quantile, config.platform_search_quantile, config.upper_quantile),
    )
    search_upper = min(
        float(upper),
        max(float(search_upper), float(lower) + config.minimum_search_span_m),
    )
    maximum_edge = np.ceil(upper / config.histogram_bin_m) * config.histogram_bin_m
    edges = np.arange(0.0, maximum_edge + 2 * config.histogram_bin_m, config.histogram_bin_m)
    counts, _ = np.histogram(distances, bins=edges)
    kernel = np.ones(config.smoothing_bins, dtype=np.float64) / config.smoothing_bins
    density = np.convolve(counts, kernel, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2.0

    peaks = np.flatnonzero(
        (density[1:-1] >= density[:-2]) & (density[1:-1] > density[2:])
    ) + 1
    peaks = peaks[
        (centers[peaks] >= lower)
        & (centers[peaks] <= search_upper)
        & (density[peaks] >= config.min_cluster_frames)
    ]
    if not len(peaks):
        raise ValueError("could not identify a supported lower-distance platform peak")
    peak = int(peaks[0])

    valleys = np.flatnonzero(
        (density[1:-1] <= density[:-2]) & (density[1:-1] < density[2:])
    ) + 1
    valleys = valleys[(valleys > peak) & (centers[valleys] <= search_upper)]
    if not len(valleys):
        raise ValueError("could not identify a density valley after the contact platform")
    valley = int(valleys[0])
    positive_threshold = float(edges[valley])
    if np.count_nonzero(distances <= positive_threshold) < config.min_cluster_frames:
        raise ValueError("contact platform has insufficient frame support")
    return float(centers[peak]), positive_threshold


def _label_distance_trace(
    name: str,
    landmark_indices: tuple[int, int],
    distances: np.ndarray,
    config: ContactLabelConfig,
) -> ContactPairLabels:
    try:
        platform_peak, positive_threshold = _detect_platform(distances, config)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error
    negative_threshold = positive_threshold + config.ambiguity_band_m
    if negative_threshold <= positive_threshold:
        raise ValueError(f"{name}: invalid platform threshold ordering")
    labels = np.full(len(distances), _AMBIGUOUS_LABEL, dtype=np.int8)
    labels[distances <= positive_threshold] = 1
    labels[distances >= negative_threshold] = 0
    frame_indices = np.arange(len(distances), dtype=np.int64)
    held_out_start = int(np.floor(len(distances) * (1.0 - config.holdout_fraction)))
    held_out = (frame_indices >= held_out_start) & (labels >= 0)
    return ContactPairLabels(
        name=name,
        landmark_indices=landmark_indices,
        frame_indices=frame_indices,
        distances=np.asarray(distances, dtype=np.float64),
        labels=labels,
        held_out=held_out,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        platform_peak=platform_peak,
        discarded_segments=_contiguous_segments(labels == _AMBIGUOUS_LABEL),
    )


def _pair_features(frames: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.concatenate((frames[:, thumb, :], frames[:, finger, :]), axis=1)
         for thumb, finger in PAIR_LANDMARKS],
        axis=1,
    )


def auto_label_contacts(
    frames: np.ndarray,
    *,
    config: ContactLabelConfig,
) -> AutoLabeledContacts:
    """Label clear D1 contact/open frames; retain ambiguous frames only as discard rows."""
    config.validate()
    frames = _validate_frames(frames)
    distances = compute_tip_distances(frames)
    pairs = tuple(
        _label_distance_trace(name, landmark_indices, distances[:, pair_index], config)
        for pair_index, (name, landmark_indices) in enumerate(zip(PAIR_NAMES, PAIR_LANDMARKS, strict=True))
    )
    return AutoLabeledContacts(features=_pair_features(frames), pairs=pairs)


def _hand_side(value: str) -> str:
    if value not in {"left", "right"}:
        raise ValueError("hand_side must be 'left' or 'right'")
    return value


def default_output_path(hand_side: str) -> Path:
    return Path("data") / f"contact_labels_{_hand_side(hand_side)}.npz"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report_payload(
    labels: AutoLabeledContacts,
    *,
    input_path: Path | str,
    hand_side: str,
    config: ContactLabelConfig,
) -> dict[str, Any]:
    """Build stable, human-readable D3 provenance and label diagnostics."""
    source = Path(input_path)
    return {
        "schema_version": 1,
        "hand_side": _hand_side(hand_side),
        "source": {"path": str(source), "sha256": _sha256_file(source)},
        "config": {
            "lower_quantile": config.lower_quantile,
            "upper_quantile": config.upper_quantile,
            "histogram_bin_m": config.histogram_bin_m,
            "smoothing_bins": config.smoothing_bins,
            "platform_search_quantile": config.platform_search_quantile,
            "minimum_search_span_m": config.minimum_search_span_m,
            "ambiguity_band_m": config.ambiguity_band_m,
            "min_cluster_frames": config.min_cluster_frames,
            "holdout_fraction": config.holdout_fraction,
        },
        "pairs": [
            {
                "name": pair.name,
                "landmark_indices": list(pair.landmark_indices),
                "positive_count": pair.positive_count,
                "negative_count": pair.negative_count,
                "discarded_count": pair.discarded_count,
                "held_out_positive_count": int(np.count_nonzero((pair.labels == 1) & pair.held_out)),
                "held_out_negative_count": int(np.count_nonzero((pair.labels == 0) & pair.held_out)),
                "thresholds": {
                    "positive_upper_m": pair.positive_threshold,
                    "negative_lower_m": pair.negative_threshold,
                },
                "platform": {
                    "peak_m": pair.platform_peak,
                    "valley_m": pair.positive_threshold,
                },
                "discarded_segments": [list(segment) for segment in pair.discarded_segments],
            }
            for pair in labels.pairs
        ],
    }


def _report_html(labels: AutoLabeledContacts, payload: dict[str, Any]) -> str:
    header = f"<h1>D1 auto-labeled contacts: {payload['hand_side']}</h1>"
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return "\n".join((
            "<!doctype html><html><head><meta charset='utf-8'></head><body>",
            header,
            "<p>Plotly unavailable; inspect the JSON report for thresholds and counts.</p>",
            f"<pre>{json.dumps(payload, indent=2, sort_keys=True)}</pre>",
            "</body></html>",
        ))
    figure = make_subplots(rows=2, cols=2, subplot_titles=list(PAIR_NAMES))
    for index, pair in enumerate(labels.pairs):
        row, col = divmod(index, 2)
        figure.add_trace(go.Histogram(x=pair.distances, nbinsx=80, name=pair.name, showlegend=False), row=row + 1, col=col + 1)
        figure.add_vline(x=pair.positive_threshold, line_color="#16a34a", row=row + 1, col=col + 1)
        figure.add_vline(x=pair.negative_threshold, line_color="#dc2626", row=row + 1, col=col + 1)
    figure.update_layout(height=760, title="Green: contact upper bound; red: open-hand lower bound")
    table_rows = "".join(
        f"<tr><td>{entry['name']}</td><td>{entry['positive_count']}</td><td>{entry['negative_count']}</td><td>{entry['discarded_count']}</td><td>{entry['thresholds']['positive_upper_m']:.5f}</td><td>{entry['thresholds']['negative_lower_m']:.5f}</td></tr>"
        for entry in payload["pairs"]
    )
    table = "<table><tr><th>pair</th><th>positive</th><th>negative</th><th>discarded</th><th>positive upper (m)</th><th>negative lower (m)</th></tr>" + table_rows + "</table>"
    return "\n".join((
        "<!doctype html><html><head><meta charset='utf-8'><style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:6px;text-align:right}th:first-child,td:first-child{text-align:left}</style></head><body>",
        header,
        table,
        figure.to_html(full_html=False, include_plotlyjs="inline"),
        "</body></html>",
    ))


def _atomic_replace(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _output_paths(output_path: Path, report_dir: Path, hand_side: str) -> ContactLabelOutputPaths:
    stem = f"contact_labels_{_hand_side(hand_side)}"
    return ContactLabelOutputPaths(output_path, report_dir / f"{stem}.json", report_dir / f"{stem}.html")


def write_contact_label_outputs(
    labels: AutoLabeledContacts,
    *,
    input_path: Path | str,
    hand_side: str,
    output_path: Path | str,
    report_dir: Path | str,
    config: ContactLabelConfig,
    overwrite: bool = False,
) -> ContactLabelOutputPaths:
    """Atomically write the D3 sample bundle plus JSON and HTML label reports."""
    paths = _output_paths(Path(output_path), Path(report_dir), hand_side)
    existing = [path for path in (paths.npz_path, paths.json_path, paths.html_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("refusing to overwrite existing output(s); pass --overwrite: " + ", ".join(map(str, existing)))
    payload = build_report_payload(labels, input_path=input_path, hand_side=hand_side, config=config)
    metadata_json = json.dumps(payload, sort_keys=True)

    def write_npz(temporary: Path) -> None:
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                features=labels.features.astype(np.float32),
                distances=labels.distances.astype(np.float32),
                labels=labels.labels,
                held_out=labels.held_out,
                pair_names=np.asarray(PAIR_NAMES),
                pair_landmarks=np.asarray(PAIR_LANDMARKS, dtype=np.int64),
                frame_indices=np.arange(len(labels.features), dtype=np.int64),
                metadata_json=np.asarray(metadata_json),
            )

    _atomic_replace(paths.npz_path, write_npz)
    _atomic_replace(paths.json_path, lambda temporary: temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"))
    _atomic_replace(paths.html_path, lambda temporary: temporary.write_text(_report_html(labels, payload), encoding="utf-8"))
    return paths


def load_hts_frames(input_path: Path | str) -> np.ndarray:
    path = Path(input_path)
    try:
        return _validate_frames(np.load(path, allow_pickle=False))
    except (OSError, ValueError) as error:
        raise ValueError(f"unable to load HTS NPY input: {path}") from error


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = ContactLabelConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/hts_right.npy"))
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/contacts"))
    parser.add_argument("--histogram-bin-mm", type=float, default=defaults.histogram_bin_m * 1000.0)
    parser.add_argument("--smoothing-bins", type=int, default=defaults.smoothing_bins)
    parser.add_argument("--platform-search-quantile", type=float, default=defaults.platform_search_quantile)
    parser.add_argument("--minimum-search-span-mm", type=float, default=defaults.minimum_search_span_m * 1000.0)
    parser.add_argument("--ambiguity-band-mm", type=float, default=defaults.ambiguity_band_m * 1000.0)
    parser.add_argument("--min-cluster-frames", type=int, default=defaults.min_cluster_frames)
    parser.add_argument("--holdout-fraction", type=float, default=defaults.holdout_fraction)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> ContactLabelOutputPaths:
    args = build_arg_parser().parse_args(argv)
    config = ContactLabelConfig(
        histogram_bin_m=args.histogram_bin_mm / 1000.0,
        smoothing_bins=args.smoothing_bins,
        platform_search_quantile=args.platform_search_quantile,
        minimum_search_span_m=args.minimum_search_span_mm / 1000.0,
        ambiguity_band_m=args.ambiguity_band_mm / 1000.0,
        min_cluster_frames=args.min_cluster_frames,
        holdout_fraction=args.holdout_fraction,
    )
    labels = auto_label_contacts(load_hts_frames(args.input), config=config)
    paths = write_contact_label_outputs(
        labels,
        input_path=args.input,
        hand_side=args.hand_side,
        output_path=args.output or default_output_path(args.hand_side),
        report_dir=args.report_dir,
        config=config,
        overwrite=args.overwrite,
    )
    print(f"Contact labels saved to {paths.npz_path}")
    print(f"Label diagnostics saved to {paths.json_path}")
    print(f"Inspection report saved to {paths.html_path}")
    return paths


if __name__ == "__main__":
    main()
