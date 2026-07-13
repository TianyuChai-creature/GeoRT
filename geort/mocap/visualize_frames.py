"""Visualize the shared global hand frame for HTS data and a robot URDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geort.env.hand import HandKinematicModel
from geort.frame_convention import AXIS_SEMANTICS, GLOBAL_HAND_BASIS
from geort.utils.config_utils import get_config, parse_config_keypoint_info
from geort.utils.path import get_human_data


AXIS_COLORS = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}


def _require_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("visualize_frames.py requires plotly") from exc
    return go, make_subplots


def global_axis_traces(*, scene_name: str, length: float):
    """Build one global XYZ triad; basis columns are axis directions."""
    go, _ = _require_plotly()
    traces = []
    for index, axis_name in enumerate(("x", "y", "z")):
        endpoint = GLOBAL_HAND_BASIS[:, index] * length
        traces.append(
            go.Scatter3d(
                x=[0.0, endpoint[0]],
                y=[0.0, endpoint[1]],
                z=[0.0, endpoint[2]],
                mode="lines+text",
                text=["", f"+{axis_name.upper()}"],
                line={"color": AXIS_COLORS[axis_name], "width": 7},
                name=f"{scene_name} +{axis_name.upper()}: {AXIS_SEMANTICS[axis_name]}",
                showlegend=True,
            )
        )
    return traces


def build_global_frame_figure(human_points: np.ndarray, robot_points: np.ndarray):
    """Create side-by-side views without rotating either point set."""
    go, make_subplots = _require_plotly()
    human_points = np.asarray(human_points, dtype=np.float64)
    robot_points = np.asarray(robot_points, dtype=np.float64)
    if human_points.ndim != 2 or human_points.shape[1] != 3:
        raise ValueError(f"Expected human points [N, 3], got {human_points.shape}")
    if robot_points.ndim != 2 or robot_points.shape[1] != 3:
        raise ValueError(f"Expected robot points [N, 3], got {robot_points.shape}")

    human_points = human_points - human_points[0]
    scale = max(
        float(np.ptp(human_points, axis=0).max()),
        float(np.ptp(robot_points, axis=0).max()),
    )
    axis_length = max(scale * 0.35, 0.01)
    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("HTS landmarks: wrist origin", "URDF keypoints: base_link origin"),
    )
    figure.add_trace(
        go.Scatter3d(
            x=human_points[:, 0],
            y=human_points[:, 1],
            z=human_points[:, 2],
            mode="markers",
            marker={"size": 4, "color": "#111827"},
            name="HTS landmarks",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter3d(
            x=robot_points[:, 0],
            y=robot_points[:, 1],
            z=robot_points[:, 2],
            mode="markers",
            marker={"size": 5, "color": "#7c3aed"},
            name="URDF keypoints",
        ),
        row=1,
        col=2,
    )
    for trace in global_axis_traces(scene_name="HTS", length=axis_length):
        figure.add_trace(trace, row=1, col=1)
    for trace in global_axis_traces(scene_name="URDF", length=axis_length):
        figure.add_trace(trace, row=1, col=2)
    scene_layout = {
        "aspectmode": "data",
        "xaxis_title": "+X palm outward",
        "yaxis_title": "+Y toward thumb",
        "zaxis_title": "+Z toward middle fingertip",
    }
    figure.update_layout(
        title="GeoRT shared right-handed global hand frame",
        scene=scene_layout,
        scene2=scene_layout,
        margin={"l": 0, "r": 0, "t": 80, "b": 0},
    )
    return figure


def load_robot_keypoints(hand_name: str) -> np.ndarray:
    config = get_config(hand_name)
    info = parse_config_keypoint_info(config)
    hand = HandKinematicModel.build_from_config(config, render=False)
    hand.initialize_keypoint(info["link"], info["offset"])
    low, high = hand.get_joint_limit()
    return hand.keypoint_from_qpos((low + high) / 2.0, ret_vec=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="custom_right")
    parser.add_argument("--human-data", default="hts_right")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/frame_convention.html")
    )
    args = parser.parse_args()

    frames = np.load(get_human_data(args.human_data))
    if not -len(frames) <= args.frame_index < len(frames):
        raise IndexError(
            f"frame-index {args.frame_index} outside dataset of {len(frames)} frames"
        )
    figure = build_global_frame_figure(
        frames[args.frame_index, :, :3],
        load_robot_keypoints(args.hand),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(args.output, include_plotlyjs="cdn")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
