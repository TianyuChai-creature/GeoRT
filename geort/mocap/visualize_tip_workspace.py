"""Layered TIP workspace visualization for human dataset and URDF reachability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from geort.env.hand import HandKinematicModel
from geort.export import load_model
from geort.utils.config_utils import get_config, parse_config_keypoint_info
from geort.utils.path import get_human_data, to_package_root


FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_COLORS = {
    "thumb": "#1f77b4",
    "index": "#2ca02c",
    "middle": "#d62728",
    "ring": "#9467bd",
    "pinky": "#8c564b",
}
URDF_COLOR = "#ff7f0e"


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for HTML workspace visualization. "
            "Install plotly in the uv environment or use an environment that already includes it."
        ) from exc
    return go


def _as_point_cloud(points: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {points.shape}")
    finite = np.isfinite(points).all(axis=1)
    return points[finite]


def extract_dataset_tip_points(frames: np.ndarray, keypoint_info: dict) -> dict[str, np.ndarray]:
    """Extract dataset TIP point clouds by finger from GeoRT-format hand frames."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[2] != 3:
        raise ValueError(f"Expected frames with shape [T, N, 3], got {frames.shape}")

    tips: dict[str, np.ndarray] = {}
    for finger, keypoint_type, human_id in zip(
        keypoint_info["finger"],
        keypoint_info["type"],
        keypoint_info["human_id"],
    ):
        if keypoint_type != "tip":
            continue
        if human_id >= frames.shape[1]:
            raise ValueError(f"Human landmark id {human_id} for {finger} tip is outside frame shape {frames.shape}")
        tips[finger] = _as_point_cloud(frames[:, human_id, :], name=f"dataset_{finger}_tip")
    return _sort_fingers(tips)


def map_dataset_tip_points(
    frames: np.ndarray,
    keypoint_info: dict,
    *,
    retargeting_model,
    hand: HandKinematicModel,
    max_frames: int,
) -> dict[str, np.ndarray]:
    """Map raw hand frames through a checkpoint and exact URDF forward kinematics."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[2] != 3:
        raise ValueError(f"Expected frames with shape [T, N, 3], got {frames.shape}")
    if len(frames) == 0:
        raise ValueError("frames must not be empty")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    hand.initialize_keypoint(
        keypoint_link_names=keypoint_info["link"],
        keypoint_offsets=keypoint_info["offset"],
    )
    tip_index_by_finger = _tip_keypoint_index_by_finger(keypoint_info)
    frame_count = min(max_frames, len(frames))
    frame_indices = np.linspace(0, len(frames) - 1, frame_count, dtype=np.int64)
    tips: dict[str, list[np.ndarray]] = {
        finger: [] for finger in tip_index_by_finger
    }
    for frame_index in frame_indices:
        qpos = retargeting_model.forward(frames[frame_index])
        keypoints = hand.keypoint_from_qpos(qpos, ret_vec=True)
        for finger, tip_index in tip_index_by_finger.items():
            tips[finger].append(np.asarray(keypoints[tip_index], dtype=np.float32))
    return _sort_fingers(
        {finger: np.stack(points, axis=0) for finger, points in tips.items()}
    )


def _sort_fingers(point_clouds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ordered = {finger: point_clouds[finger] for finger in FINGER_ORDER if finger in point_clouds}
    ordered.update({finger: points for finger, points in point_clouds.items() if finger not in ordered})
    return ordered


def _tip_keypoint_index_by_finger(keypoint_info: dict) -> dict[str, int]:
    out = {}
    for idx, (finger, keypoint_type) in enumerate(zip(keypoint_info["finger"], keypoint_info["type"])):
        if keypoint_type == "tip":
            out[finger] = idx
    return out


def sample_urdf_tip_points(
    config: dict,
    keypoint_info: dict,
    *,
    samples_per_finger: int,
    seed: int = 0,
    fixed_qpos_value: float = 0.0,
) -> dict[str, np.ndarray]:
    """Sample per-finger URDF TIP workspaces under the current joint limits."""
    if samples_per_finger <= 0:
        raise ValueError("samples_per_finger must be positive")

    hand = HandKinematicModel.build_from_config(config, render=False)
    hand.initialize_keypoint(keypoint_link_names=keypoint_info["link"], keypoint_offsets=keypoint_info["offset"])
    lower, upper = hand.get_joint_limit()
    rng = np.random.default_rng(seed)

    tip_index_by_finger = _tip_keypoint_index_by_finger(keypoint_info)
    tips: dict[str, list[np.ndarray]] = {finger: [] for finger in tip_index_by_finger}

    groups_by_finger = {group["finger"]: group for group in keypoint_info["finger_groups"]}
    for finger in _sort_fingers(tip_index_by_finger).keys():
        group = groups_by_finger[finger]
        joint_indices = np.array(group["joint_indices"], dtype=np.int64)
        tip_idx = tip_index_by_finger[finger]
        qpos = np.full(hand.get_n_dof(), fixed_qpos_value, dtype=np.float32)
        qpos = np.clip(qpos, lower, upper)

        for _ in range(samples_per_finger):
            qpos[joint_indices] = rng.uniform(lower[joint_indices], upper[joint_indices])
            keypoints = hand.keypoint_from_qpos(qpos, ret_vec=True)
            tips[finger].append(keypoints[tip_idx].astype(np.float32))

    return _sort_fingers({finger: np.stack(points, axis=0) for finger, points in tips.items()})


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is required for alpha surface generation. "
            "Install open3d in the uv environment or rerun without --include_alpha_surface."
        ) from exc
    return o3d


def _downsample_points(points: np.ndarray, *, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[np.sort(indices)]


def compute_alpha_surface_mesh(
    points: np.ndarray,
    *,
    alpha: float,
    max_points: int = 2000,
    seed: int = 0,
) -> dict[str, np.ndarray] | None:
    """Return an Open3D alpha-shape triangle mesh for a point cloud.

    The mesh is intentionally optional: sparse, coplanar, or poorly tuned alpha
    inputs may not produce a valid surface. In that case callers keep the point
    cloud visualization and skip only the surface trace.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    points = _as_point_cloud(points, name="alpha_surface_points")
    if len(points) < 4:
        return None
    points = _downsample_points(points, max_points=max_points, seed=seed).astype(np.float64)

    o3d = _require_open3d()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    try:
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd,
            alpha,
            tetra_mesh,
            pt_map,
        )
    except RuntimeError:
        return None

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    if vertices.size == 0 or triangles.size == 0:
        return None
    return {"vertices": vertices, "triangles": triangles}


def _trace(points: np.ndarray, *, name: str, color: str, opacity: float, size: int):
    go = _require_plotly()
    points = _as_point_cloud(points, name=name)
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=name,
        marker={
            "size": size,
            "color": color,
            "opacity": opacity,
        },
    )


def _alpha_surface_trace(mesh: dict[str, np.ndarray], *, name: str, color: str, opacity: float):
    go = _require_plotly()
    vertices = mesh["vertices"]
    triangles = mesh["triangles"]
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        name=name,
        color=color,
        opacity=opacity,
        visible=False,
        hoverinfo="skip",
        showscale=False,
    )


def _apply_visibility_buttons(fig):
    is_alpha = [str(trace.name).endswith("_alpha") for trace in fig.data]
    if not any(is_alpha):
        return fig

    points_only = [not value for value in is_alpha]
    alpha_only = list(is_alpha)
    all_visible = [True for _ in is_alpha]
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.0,
                "y": 1.12,
                "buttons": [
                    {"label": "Points only", "method": "update", "args": [{"visible": points_only}]},
                    {"label": "Alpha only", "method": "update", "args": [{"visible": alpha_only}]},
                    {"label": "Points + Alpha", "method": "update", "args": [{"visible": all_visible}]},
                ],
            }
        ]
    )
    return fig


def _figure(title: str, traces: list):
    go = _require_plotly()
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "t": 64, "b": 0},
        legend={"itemsizing": "constant"},
        scene={
            "aspectmode": "data",
            "xaxis_title": "x (m)",
            "yaxis_title": "y (m)",
            "zaxis_title": "z (m)",
        },
    )
    return _apply_visibility_buttons(fig)


def build_layered_tip_workspace_figures(
    dataset_tips: dict[str, np.ndarray],
    urdf_tips: dict[str, np.ndarray],
    *,
    source_label: str = "Dataset",
    include_alpha_surface: bool = False,
    alpha: float = 0.03,
    surface_max_points: int = 2000,
    surface_seed: int = 0,
):
    """Build layered Plotly figures for single-finger, all-human, all-URDF, and overview views."""
    dataset_tips = _sort_fingers(dataset_tips)
    urdf_tips = _sort_fingers(urdf_tips)
    source_name = source_label.lower()
    fingers = [finger for finger in FINGER_ORDER if finger in dataset_tips and finger in urdf_tips]
    fingers.extend(
        finger for finger in sorted(set(dataset_tips) & set(urdf_tips))
        if finger not in fingers
    )
    surface_meshes: dict[str, dict[str, np.ndarray] | None] = {}

    def traces_for(
        points: np.ndarray,
        *,
        name: str,
        point_color: str,
        point_opacity: float,
        surface_color: str,
        surface_opacity: float,
    ) -> list:
        traces = [
            _trace(
                points,
                name=name,
                color=point_color,
                opacity=point_opacity,
                size=2,
            )
        ]
        if include_alpha_surface:
            if name not in surface_meshes:
                surface_meshes[name] = compute_alpha_surface_mesh(
                    points,
                    alpha=alpha,
                    max_points=surface_max_points,
                    seed=surface_seed,
                )
            mesh = surface_meshes[name]
            if mesh is not None:
                traces.append(
                    _alpha_surface_trace(
                        mesh,
                        name=f"{name}_alpha",
                        color=surface_color,
                        opacity=surface_opacity,
                    )
                )
        return traces

    figures = {}
    for finger in fingers:
        traces = []
        traces.extend(
            traces_for(
                dataset_tips[finger],
                name=f"{source_name}_{finger}_tip",
                point_color=FINGER_COLORS.get(finger, "#1f77b4"),
                point_opacity=0.65,
                surface_color=FINGER_COLORS.get(finger, "#1f77b4"),
                surface_opacity=0.22,
            )
        )
        traces.extend(
            traces_for(
                urdf_tips[finger],
                name=f"urdf_{finger}_tip",
                point_color=URDF_COLOR,
                point_opacity=0.35,
                surface_color=URDF_COLOR,
                surface_opacity=0.16,
            )
        )
        figures[f"single_{finger}"] = _figure(
            f"{finger} TIP workspace: {source_label} vs URDF", traces
        )

    dataset_all_traces = []
    for finger in dataset_tips:
        dataset_all_traces.extend(
            traces_for(
                dataset_tips[finger],
                name=f"{source_name}_{finger}_tip",
                point_color=FINGER_COLORS.get(finger, "#1f77b4"),
                point_opacity=0.65,
                surface_color=FINGER_COLORS.get(finger, "#1f77b4"),
                surface_opacity=0.20,
            )
        )
    figures["dataset_all"] = _figure(
        f"{source_label} TIP workspace: all fingers", dataset_all_traces
    )

    urdf_all_traces = []
    for finger in urdf_tips:
        urdf_all_traces.extend(
            traces_for(
                urdf_tips[finger],
                name=f"urdf_{finger}_tip",
                point_color=FINGER_COLORS.get(finger, "#1f77b4"),
                point_opacity=0.45,
                surface_color=FINGER_COLORS.get(finger, "#1f77b4"),
                surface_opacity=0.18,
            )
        )
    figures["urdf_all"] = _figure("URDF TIP workspace: all fingers", urdf_all_traces)

    overview_traces = []
    for finger in dataset_tips:
        overview_traces.extend(
            traces_for(
                dataset_tips[finger],
                name=f"{source_name}_{finger}_tip",
                point_color=FINGER_COLORS.get(finger, "#1f77b4"),
                point_opacity=0.55,
                surface_color=FINGER_COLORS.get(finger, "#1f77b4"),
                surface_opacity=0.14,
            )
        )
    for finger in urdf_tips:
        overview_traces.extend(
            traces_for(
                urdf_tips[finger],
                name=f"urdf_{finger}_tip",
                point_color=URDF_COLOR,
                point_opacity=0.22,
                surface_color=URDF_COLOR,
                surface_opacity=0.10,
            )
        )
    figures["overview_all"] = _figure(
        f"TIP workspace overview: {source_label} and URDF", overview_traces
    )
    return figures


def default_overlap_pairs(fingers: list[str] | tuple[str, ...]) -> list[tuple[str, str]]:
    """Return thumb-vs-all and adjacent four-finger overlap pairs."""
    available = set(fingers)
    pairs: list[tuple[str, str]] = []
    for other in ["index", "middle", "ring", "pinky"]:
        if "thumb" in available and other in available:
            pairs.append(("thumb", other))
    for a, b in [("index", "middle"), ("middle", "ring"), ("ring", "pinky")]:
        if a in available and b in available:
            pairs.append((a, b))
    return pairs


def _voxel_set(points: np.ndarray, *, voxel_size: float, origin: np.ndarray) -> set[tuple[int, int, int]]:
    points = _as_point_cloud(points, name="voxel_points")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if len(points) == 0:
        return set()
    voxel_indices = np.floor((points - origin.reshape(1, 3)) / voxel_size).astype(np.int64)
    return {tuple(int(v) for v in row) for row in voxel_indices}


def compute_voxel_overlap(points_a: np.ndarray, points_b: np.ndarray, *, voxel_size: float) -> dict:
    """Compute voxelized 3D overlap ratios between two point clouds."""
    points_a = _as_point_cloud(points_a, name="points_a")
    points_b = _as_point_cloud(points_b, name="points_b")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    if len(points_a) == 0 or len(points_b) == 0:
        a_voxels = 0 if len(points_a) == 0 else len(_voxel_set(points_a, voxel_size=voxel_size, origin=points_a.min(axis=0)))
        b_voxels = 0 if len(points_b) == 0 else len(_voxel_set(points_b, voxel_size=voxel_size, origin=points_b.min(axis=0)))
        return {
            "a_voxels": int(a_voxels),
            "b_voxels": int(b_voxels),
            "intersection_voxels": 0,
            "union_voxels": int(a_voxels + b_voxels),
            "overlap_a_ratio": 0.0,
            "overlap_b_ratio": 0.0,
            "iou": 0.0,
        }

    origin = np.concatenate([points_a, points_b], axis=0).min(axis=0)
    voxels_a = _voxel_set(points_a, voxel_size=voxel_size, origin=origin)
    voxels_b = _voxel_set(points_b, voxel_size=voxel_size, origin=origin)
    intersection = voxels_a & voxels_b
    union = voxels_a | voxels_b
    return {
        "a_voxels": int(len(voxels_a)),
        "b_voxels": int(len(voxels_b)),
        "intersection_voxels": int(len(intersection)),
        "union_voxels": int(len(union)),
        "overlap_a_ratio": float(len(intersection) / max(len(voxels_a), 1)),
        "overlap_b_ratio": float(len(intersection) / max(len(voxels_b), 1)),
        "iou": float(len(intersection) / max(len(union), 1)),
    }


def _pair_overlap_report(point_clouds: dict[str, np.ndarray], pairs: list[tuple[str, str]], *, voxel_size: float) -> dict:
    out = {}
    for finger_a, finger_b in pairs:
        out[f"{finger_a}__{finger_b}"] = compute_voxel_overlap(
            point_clouds[finger_a],
            point_clouds[finger_b],
            voxel_size=voxel_size,
        )
    return out


def build_workspace_overlap_report(
    dataset_tips: dict[str, np.ndarray],
    urdf_tips: dict[str, np.ndarray],
    *,
    voxel_size: float,
) -> dict:
    dataset_fingers = set(dataset_tips)
    urdf_fingers = set(urdf_tips)
    fingers = [finger for finger in FINGER_ORDER if finger in dataset_fingers and finger in urdf_fingers]
    fingers.extend(sorted((dataset_fingers & urdf_fingers) - set(fingers)))
    pairs = default_overlap_pairs(fingers)
    urdf_overlap = _pair_overlap_report(urdf_tips, pairs, voxel_size=voxel_size)
    report = {
        "space": "tip_xyz_voxel_overlap",
        "voxel_size": float(voxel_size),
        "pairs": [[a, b] for a, b in pairs],
        "dataset": _pair_overlap_report(dataset_tips, pairs, voxel_size=voxel_size),
        "urdf": urdf_overlap,
    }
    return report

def _axis_range(points: np.ndarray) -> list[list[float]]:
    points = _as_point_cloud(points, name="points")
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return [[float(lo), float(hi)] for lo, hi in zip(mins, maxs)]


def _centroid(points: np.ndarray) -> list[float]:
    points = _as_point_cloud(points, name="points")
    return [float(value) for value in points.mean(axis=0)]


def _nearest_mean(source: np.ndarray, target: np.ndarray) -> float:
    source = _as_point_cloud(source, name="source")
    target = _as_point_cloud(target, name="target")
    if len(source) == 0 or len(target) == 0:
        return float("nan")
    distances, _ = cKDTree(target).query(source, k=1)
    return float(np.mean(distances))


def summarize_workspace_alignment(dataset_tips: dict[str, np.ndarray], urdf_tips: dict[str, np.ndarray]) -> dict:
    fingers = [finger for finger in FINGER_ORDER if finger in dataset_tips or finger in urdf_tips]
    fingers.extend(finger for finger in sorted(set(dataset_tips) | set(urdf_tips)) if finger not in fingers)

    report = {
        "space": "tip_xyz",
        "fingers": {},
    }
    for finger in fingers:
        dataset = dataset_tips.get(finger, np.empty((0, 3), dtype=np.float32))
        urdf = urdf_tips.get(finger, np.empty((0, 3), dtype=np.float32))
        stats = {
            "dataset_samples": int(len(dataset)),
            "urdf_samples": int(len(urdf)),
        }
        if len(dataset):
            stats["dataset_aabb"] = _axis_range(dataset)
            stats["dataset_centroid"] = _centroid(dataset)
        if len(urdf):
            stats["urdf_aabb"] = _axis_range(urdf)
            stats["urdf_centroid"] = _centroid(urdf)
        if len(dataset) and len(urdf):
            stats["dataset_to_urdf_nn_mean"] = _nearest_mean(dataset, urdf)
            stats["urdf_to_dataset_nn_mean"] = _nearest_mean(urdf, dataset)
            stats["centroid_delta"] = [
                float(a - b)
                for a, b in zip(np.asarray(stats["dataset_centroid"]), np.asarray(stats["urdf_centroid"]))
            ]
        report["fingers"][finger] = stats
    return report


def _format_ratio(value: float) -> str:
    return f"{float(value):.4f}"


def _overlap_table_html(overlap_report: dict | None) -> str:
    if not overlap_report:
        return ""

    rows = []
    for finger_a, finger_b in overlap_report.get("pairs", []):
        pair_name = f"{finger_a}__{finger_b}"
        dataset = overlap_report.get("dataset", {}).get(pair_name, {})
        urdf = overlap_report.get("urdf", {}).get(pair_name, {})
        rows.append(
            "<tr>"
            f"<td>{pair_name}</td>"
            f"<td>{_format_ratio(dataset.get('iou', 0.0))}</td>"
            f"<td>{_format_ratio(dataset.get('overlap_a_ratio', 0.0))}</td>"
            f"<td>{_format_ratio(dataset.get('overlap_b_ratio', 0.0))}</td>"
            f"<td>{int(dataset.get('intersection_voxels', 0))}</td>"
            f"<td>{_format_ratio(urdf.get('iou', 0.0))}</td>"
            f"<td>{_format_ratio(urdf.get('overlap_a_ratio', 0.0))}</td>"
            f"<td>{_format_ratio(urdf.get('overlap_b_ratio', 0.0))}</td>"
            f"<td>{int(urdf.get('intersection_voxels', 0))}</td>"
            "</tr>"
        )

    voxel_size = overlap_report.get("voxel_size", 0.0)
    header = "<thead><tr><th>pair</th><th>dataset IoU</th><th>dataset A overlap</th><th>dataset B overlap</th><th>dataset intersect</th><th>urdf IoU</th><th>urdf A overlap</th><th>urdf B overlap</th><th>urdf intersect</th></tr></thead>"
    note = "Dataset columns use captured human data; URDF columns use the active robot limits."
    return "\n".join(
        [
            "<section class='report-section'>",
            "<h2>Workspace Overlap Summary</h2>",
            f"<p>Voxel size: {float(voxel_size):.4f} m. Ratios are computed on TIP workspace occupied voxels. {note}</p>",
            "<table>",
            header,
            "<tbody>",
            *rows,
            "</tbody>",
            "</table>",
            "</section>",
        ]
    )


def write_layered_html(figures: dict, output_path: Path | str, *, overlap_report: dict | None = None) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sections = []
    include_plotlyjs = True
    for name, fig in figures.items():
        section = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
        sections.append(f"<section><h2>{name}</h2>{section}</section>")
        include_plotlyjs = False

    html = "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>GeoRT TIP Workspace Visualization</title>",
            "<style>body{font-family:sans-serif;margin:24px;}section{margin-bottom:48px;}h1,h2{font-weight:600;}table{border-collapse:collapse;font-size:13px;}th,td{border:1px solid #ccc;padding:6px 8px;text-align:right;}th:first-child,td:first-child{text-align:left;}thead{background:#f2f2f2;}.report-section{margin-bottom:32px;}</style>",
            "</head>",
            "<body>",
            "<h1>GeoRT TIP Workspace Visualization</h1>",
            "<p>Layer order: overlap metrics, single-finger dataset/URDF overlays, dataset-only all fingers, URDF-only all fingers, final overview.</p>",
            _overlap_table_html(overlap_report),
            *sections,
            "</body>",
            "</html>",
        ]
    )
    path.write_text(html, encoding="utf-8")
    return path


def save_report(report: dict, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _resolve_output(path: str) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = to_package_root(output)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="custom_right", help="GeoRT hand config name.")
    parser.add_argument("--human_data", default="hts_right", help="Dataset name or .npy path under data/.")
    parser.add_argument(
        "--ckpt_tag",
        default=None,
        help="Checkpoint tag/path. When set, visualize checkpoint-mapped URDF TIPs instead of raw HTS TIPs.",
    )
    parser.add_argument(
        "--mapped_max_frames",
        type=int,
        default=5000,
        help="Maximum uniformly sampled HTS frames mapped through the checkpoint.",
    )
    parser.add_argument("--samples_per_finger", type=int, default=15000, help="URDF FK samples per finger.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for URDF workspace sampling.")
    parser.add_argument(
        "--include_alpha_surface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include hidden Open3D alpha-shape surface traces with Plotly visibility buttons.",
    )
    parser.add_argument("--alpha", type=float, default=0.08, help="Open3D alpha-shape radius in meters.")
    parser.add_argument(
        "--overlap_voxel_size",
        type=float,
        default=0.005,
        help="Voxel size in meters for per-finger TIP workspace overlap metrics.",
    )
    parser.add_argument(
        "--surface_max_points",
        type=int,
        default=2000,
        help="Maximum points per cloud used to build each alpha surface; 0 disables downsampling.",
    )
    parser.add_argument(
        "--output",
        default="outputs/visualizations/custom_right_tip_workspace.html",
        help="Output layered HTML path.",
    )
    parser.add_argument(
        "--report",
        default="outputs/visualizations/custom_right_tip_workspace_report.json",
        help="Output JSON report path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = get_config(args.hand)
    keypoint_info = parse_config_keypoint_info(config)
    frames = np.load(get_human_data(args.human_data))

    if args.ckpt_tag:
        mapped_hand = HandKinematicModel.build_from_config(config, render=False)
        dataset_tips = map_dataset_tip_points(
            frames,
            keypoint_info,
            retargeting_model=load_model(args.ckpt_tag),
            hand=mapped_hand,
            max_frames=args.mapped_max_frames,
        )
        workspace_source = {
            "mode": "checkpoint_mapped_urdf_tips",
            "checkpoint": args.ckpt_tag,
            "frames": len(next(iter(dataset_tips.values()))),
        }
    else:
        dataset_tips = extract_dataset_tip_points(frames, keypoint_info)
        workspace_source = {
            "mode": "raw_human_tips",
            "checkpoint": None,
            "frames": len(frames),
        }
    urdf_tips = sample_urdf_tip_points(
        config,
        keypoint_info,
        samples_per_finger=args.samples_per_finger,
        seed=args.seed,
    )
    figures = build_layered_tip_workspace_figures(
        dataset_tips,
        urdf_tips,
        source_label="Mapped" if args.ckpt_tag else "Dataset",
        include_alpha_surface=args.include_alpha_surface,
        alpha=args.alpha,
        surface_max_points=args.surface_max_points,
        surface_seed=args.seed,
    )
    report = summarize_workspace_alignment(dataset_tips, urdf_tips)
    report["workspace_source"] = workspace_source
    report["overlap"] = build_workspace_overlap_report(
        dataset_tips,
        urdf_tips,
        voxel_size=args.overlap_voxel_size,
    )
    report["alpha_surface"] = {
        "enabled": bool(args.include_alpha_surface),
        "alpha": float(args.alpha),
        "surface_max_points": int(args.surface_max_points),
    }
    report["overlap_voxel_size"] = float(args.overlap_voxel_size)

    html_path = write_layered_html(figures, _resolve_output(args.output), overlap_report=report["overlap"])
    report_path = save_report(report, _resolve_output(args.report))
    print(f"TIP workspace HTML saved to {html_path}")
    print(f"TIP workspace report saved to {report_path}")
    if args.include_alpha_surface:
        print(f"Alpha surface traces included with alpha={args.alpha} surface_max_points={args.surface_max_points}")
    print(f"Overlap voxel size={args.overlap_voxel_size}")
    for pair_name, stats in report["overlap"]["dataset"].items():
        urdf_stats = report["overlap"]["urdf"][pair_name]
        print(
            f"overlap {pair_name}: "
            f"dataset_iou={stats['iou']:.4f} "
            f"urdf_iou={urdf_stats['iou']:.4f}"
        )
    for finger, stats in report["fingers"].items():
        if "dataset_to_urdf_nn_mean" in stats:
            print(
                f"{finger}: dataset={stats['dataset_samples']} urdf={stats['urdf_samples']} "
                f"dataset->urdf nn={stats['dataset_to_urdf_nn_mean']:.5f} "
                f"urdf->dataset nn={stats['urdf_to_dataset_nn_mean']:.5f}"
            )


if __name__ == "__main__":
    main()
