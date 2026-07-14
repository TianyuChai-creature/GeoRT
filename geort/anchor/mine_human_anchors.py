"""Mine fifty sparse human anchors from an existing D1 HTS recording."""

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

from geort.anchor.mining import MinedHumanAnchors, mine_human_anchor_records


_FINGER_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


@dataclass(frozen=True, slots=True)
class MiningOutputPaths:
    """Final paths emitted by one successful D1 anchor-mining run."""

    npz_path: Path
    json_path: Path
    html_path: Path


def _hand_side(value: str) -> str:
    if value not in {"left", "right"}:
        raise ValueError("hand_side must be 'left' or 'right'")
    return value


def default_output_path(hand_side: str) -> Path:
    """Return the ignored, conventional D2 anchor path for one hand."""
    return Path("data") / f"anchors_human_{_hand_side(hand_side)}.npz"


def load_hts_frames(input_path: Path | str) -> np.ndarray:
    """Load a D1 NPY recording without allowing pickle payloads."""
    path = Path(input_path)
    try:
        frames = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"unable to load HTS NPY input: {path}") from error
    if frames.ndim != 3 or frames.shape[1:] != (21, 3):
        raise ValueError(
            f"HTS input must have shape [T, 21, 3], got {tuple(frames.shape)}"
        )
    if not np.issubdtype(frames.dtype, np.number):
        raise ValueError("HTS input must have a numeric dtype")
    return np.asarray(frames, dtype=np.float64)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_anchor_contract(anchors: MinedHumanAnchors) -> None:
    expected_rows = 50
    expected_shapes = {
        "human_frames": (expected_rows, 21, 3),
        "human_points": (expected_rows, 3),
        "source_indices": (expected_rows,),
        "finger_indices": (expected_rows,),
        "finger_names": (expected_rows,),
        "anchor_types": (expected_rows,),
        "levels": (expected_rows,),
        "trajectory_t": (expected_rows,),
        "target_parameters": (expected_rows,),
        "observed_parameters": (expected_rows,),
        "candidate_counts": (expected_rows,),
        "support_counts": (expected_rows,),
    }
    for name, shape in expected_shapes.items():
        if np.asarray(getattr(anchors, name)).shape != shape:
            raise ValueError(f"anchors.{name} must have shape {shape}")
    if not np.all(np.isfinite(anchors.human_frames)):
        raise ValueError("anchors.human_frames must be finite")


def build_report_payload(
    anchors: MinedHumanAnchors,
    *,
    input_path: Path | str,
    hand_side: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build JSON-serializable mining diagnostics grouped by finger and type."""
    _validate_anchor_contract(anchors)
    side = _hand_side(hand_side)
    source = Path(input_path)
    groups: list[dict[str, Any]] = []
    for start in range(0, 50, 5):
        rows = slice(start, start + 5)
        finger = str(anchors.finger_names[start])
        anchor_type = str(anchors.anchor_types[start])
        key = f"{finger}:{anchor_type}"
        groups.append(
            {
                "finger": finger,
                "anchor_type": anchor_type,
                "levels": anchors.levels[rows].astype(int).tolist(),
                "trajectory_t": anchors.trajectory_t[rows].astype(float).tolist(),
                "target_parameters": (
                    anchors.target_parameters[rows].astype(float).tolist()
                ),
                "observed_parameters": (
                    anchors.observed_parameters[rows].astype(float).tolist()
                ),
                "source_indices": anchors.source_indices[rows].astype(int).tolist(),
                "candidate_counts": anchors.candidate_counts[rows].astype(int).tolist(),
                "support_counts": anchors.support_counts[rows].astype(int).tolist(),
                "diagnostics": anchors.group_metadata.get(key, {}),
            }
        )
    return {
        "schema_version": 1,
        "hand_side": side,
        "anchor_count": 50,
        "source": {
            "path": str(source),
            "sha256": _sha256_file(source),
        },
        "config": config,
        "groups": groups,
    }


def _pose_trace(go: Any, frame: np.ndarray, *, name: str) -> Any:
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for first, second in _FINGER_EDGES:
        x.extend((float(frame[first, 0]), float(frame[second, 0]), None))
        y.extend((float(frame[first, 1]), float(frame[second, 1]), None))
        z.extend((float(frame[first, 2]), float(frame[second, 2]), None))
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines+markers",
        line={"color": "#2563eb", "width": 5},
        marker={"color": "#0f172a", "size": 3},
        name=name,
        showlegend=False,
    )


def _group_figure(anchors: MinedHumanAnchors, group: dict[str, Any]) -> Any:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise RuntimeError("mine_human_anchors.py requires plotly for HTML reports") from error

    source_indices = np.asarray(group["source_indices"], dtype=np.int64)
    rows = np.array(
        [int(np.flatnonzero(anchors.source_indices == index)[0]) for index in source_indices]
    )
    titles = ["Observed parameter"] + [f"Level {level}" for level in group["levels"]]
    figure = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=titles,
    )
    figure.add_trace(
        go.Histogram(
            x=group["observed_parameters"],
            marker={"color": "#93c5fd"},
            name="selected parameters",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=group["target_parameters"],
            y=np.zeros(5),
            mode="markers",
            marker={"color": "#dc2626", "size": 9, "symbol": "x"},
            name="target",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    for level, row_index in enumerate(rows):
        plot_row = 1 if level < 2 else 2
        plot_col = level + 2 if level < 2 else level - 1
        figure.add_trace(
            _pose_trace(go, anchors.human_frames[row_index], name=f"level {level}"),
            row=plot_row,
            col=plot_col,
        )
    figure.update_layout(
        title=f"{group['finger']} {group['anchor_type']} anchors",
        height=700,
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )
    figure.update_scenes(aspectmode="data")
    figure.update_xaxes(title="parameter")
    return figure


def build_report_html(anchors: MinedHumanAnchors, payload: dict[str, Any]) -> str:
    """Render one self-contained Plotly inspection page for all ten groups."""
    source = payload["source"]
    header = (
        f"<h1>Human anchors: {payload['hand_side']}</h1>"
        f"<p>source: {source['path']}<br>sha256: {source['sha256']}</p>"
    )
    try:
        sections = []
        for index, group in enumerate(payload["groups"]):
            figure = _group_figure(anchors, group)
            sections.append(
                figure.to_html(
                    full_html=False,
                    include_plotlyjs="inline" if index == 0 else False,
                )
            )
    except RuntimeError as error:
        return "\n".join(
            (
                "<!doctype html><html><head><meta charset='utf-8'></head><body>",
                header,
                "<p>Plotly is unavailable; this fallback contains diagnostics only.</p>",
                f"<pre>{json.dumps(payload, indent=2, sort_keys=True)}</pre>",
                f"<!-- {error} -->",
                "</body></html>",
            )
        )
    return "\n".join(
        (
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<style>body{font-family:sans-serif;margin:24px;}section{margin-bottom:48px;}</style>",
            "</head><body>",
            header,
            *(f"<section>{section}</section>" for section in sections),
            "</body></html>",
        )
    )


def _atomic_replace(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _output_paths(output_path: Path, report_dir: Path, hand_side: str) -> MiningOutputPaths:
    stem = f"anchors_human_{_hand_side(hand_side)}"
    return MiningOutputPaths(
        npz_path=output_path,
        json_path=report_dir / f"{stem}.json",
        html_path=report_dir / f"{stem}.html",
    )


def write_mining_outputs(
    anchors: MinedHumanAnchors,
    *,
    input_path: Path | str,
    hand_side: str,
    output_path: Path | str,
    report_dir: Path | str,
    config: dict[str, Any],
    overwrite: bool = False,
) -> MiningOutputPaths:
    """Atomically write the D2 NPZ, JSON diagnostics, and HTML inspection page."""
    paths = _output_paths(Path(output_path), Path(report_dir), hand_side)
    existing = [path for path in (paths.npz_path, paths.json_path, paths.html_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing output(s); pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    payload = build_report_payload(
        anchors,
        input_path=input_path,
        hand_side=hand_side,
        config=config,
    )
    html = build_report_html(anchors, payload)
    metadata_json = json.dumps(payload, sort_keys=True)

    def write_npz(temporary: Path) -> None:
        with temporary.open("wb") as output_file:
            np.savez_compressed(
                output_file,
                human_frames=anchors.human_frames,
                human_points=anchors.human_points,
                source_indices=anchors.source_indices,
                finger_indices=anchors.finger_indices,
                finger_names=anchors.finger_names,
                anchor_types=anchors.anchor_types,
                levels=anchors.levels,
                trajectory_t=anchors.trajectory_t,
                target_parameters=anchors.target_parameters,
                observed_parameters=anchors.observed_parameters,
                candidate_counts=anchors.candidate_counts,
                support_counts=anchors.support_counts,
                metadata_json=np.asarray(metadata_json),
            )

    _atomic_replace(paths.npz_path, write_npz)
    _atomic_replace(
        paths.json_path,
        lambda temporary: temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        ),
    )
    _atomic_replace(
        paths.html_path,
        lambda temporary: temporary.write_text(html, encoding="utf-8"),
    )
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/hts_right.npy"))
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/anchors"))
    parser.add_argument("--endpoint-low", type=float, default=0.02)
    parser.add_argument("--endpoint-high", type=float, default=0.98)
    parser.add_argument("--straight-tol-deg", type=float, default=15.0)
    parser.add_argument("--alpha-zero-tol-deg", type=float, default=10.0)
    parser.add_argument("--coupling-tol-deg", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> MiningOutputPaths:
    args = build_arg_parser().parse_args(argv)
    if not 0.0 <= args.endpoint_low < args.endpoint_high <= 1.0:
        raise ValueError("endpoint bounds must satisfy 0 <= low < high <= 1")
    frames = load_hts_frames(args.input)
    config = {
        "endpoint_low": args.endpoint_low,
        "endpoint_high": args.endpoint_high,
        "straight_tol_deg": args.straight_tol_deg,
        "alpha_zero_tol_deg": args.alpha_zero_tol_deg,
        "coupling_tol_deg": args.coupling_tol_deg,
    }
    anchors = mine_human_anchor_records(
        frames,
        endpoint_quantiles=(args.endpoint_low, args.endpoint_high),
        straight_tol=np.deg2rad(args.straight_tol_deg),
        alpha_zero_tol=np.deg2rad(args.alpha_zero_tol_deg),
        coupling_tol=np.deg2rad(args.coupling_tol_deg),
    )
    paths = write_mining_outputs(
        anchors,
        input_path=args.input,
        hand_side=args.hand_side,
        output_path=args.output or default_output_path(args.hand_side),
        report_dir=args.report_dir,
        config=config,
        overwrite=args.overwrite,
    )
    print(f"Human anchors saved to {paths.npz_path}")
    print(f"Mining diagnostics saved to {paths.json_path}")
    print(f"Inspection report saved to {paths.html_path}")
    return paths


if __name__ == "__main__":
    main()
