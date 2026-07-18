"""CPU-only analytic/SAPIEN parity gate for sparse-anchor robot TIPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geort.anchor.anchor_spec import derive_finger_joint_layouts
from geort.anchor.compat import get_joint_limits, make_analytic_tip_callback
from geort.anchor.generate_robot_anchors import FINGER_NAMES, build_robot_sparse_knots
from geort.anchor.interpolate import interpolate_sparse_trajectory
from geort.pipeline.gates import evaluate_fk_parity_gate, load_gate_spec
from geort.pipeline.manifest import HandManifest, load_hand_manifest
from geort.utils.config_utils import get_config, parse_config_keypoint_info, select_keypoint_types


def _sapien_callback(config: dict):
    from geort.env.hand import HandKinematicModel

    tips = select_keypoint_types(parse_config_keypoint_info(config), allowed_types=("tip",))
    if tuple(tips["finger"]) != FINGER_NAMES:
        raise ValueError("TIP config must be thumb-to-pinky")
    hand = HandKinematicModel.build_from_config(config, render=False)
    hand.initialize_keypoint(tips["link"], tips["offset"])

    def evaluate(qpos: np.ndarray, finger_index: int) -> np.ndarray:
        return np.asarray(hand.keypoint_from_qpos(qpos, ret_vec=True)[finger_index], dtype=np.float64)

    return evaluate, tips["offset"]


def write_parity_qpos(path: Path, config: dict) -> Path:
    """Create canonical 750 robot qpos from 50 robot knots, without human anchors.

    This is exactly the robot interpolation performed by ``build_paired_anchors``:
    five lateral groups at 50 rows plus five bending groups at 100 rows. The
    human side never affects robot qpos, so FK parity is available before a
    hand's anchor-mining station exists.
    """
    lower, upper = get_joint_limits(config)
    sapien, offsets = _sapien_callback(config)
    knots = build_robot_sparse_knots(
        lower, upper, derive_finger_joint_layouts(config["joint_order"]), sapien
    )
    qpos_parts: list[np.ndarray] = []
    finger_parts: list[np.ndarray] = []
    type_parts: list[np.ndarray] = []
    trajectory_parts: list[np.ndarray] = []
    for group_index, (finger_index, anchor_type) in enumerate(
        (finger_index, anchor_type)
        for finger_index in range(5)
        for anchor_type in ("lateral", "bending")
    ):
        count = 50 if anchor_type == "lateral" else 100
        trajectory = interpolate_sparse_trajectory(
            knots[group_index * 5 : (group_index + 1) * 5], count
        )
        qpos_parts.append(trajectory["points"])
        finger_parts.append(np.full(count, finger_index, dtype=np.int64))
        type_parts.append(np.full(count, anchor_type))
        trajectory_parts.append(trajectory["trajectory_t"])
    robot_qpos = np.concatenate(qpos_parts, axis=0)
    finger_indices = np.concatenate(finger_parts, axis=0)
    anchor_types = np.concatenate(type_parts, axis=0)
    trajectory_t = np.concatenate(trajectory_parts, axis=0)
    if robot_qpos.shape[0] != 750:
        raise RuntimeError("parity input must contain 750 robot qpos rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        robot_qpos=robot_qpos,
        finger_indices=finger_indices,
        anchor_types=anchor_types,
        trajectory_t=trajectory_t,
        tip_offsets=np.asarray(offsets, dtype=np.float64),
        metadata_json=np.asarray(json.dumps({
            "coordinate_frame": "hand_base", "units": "m",
            "sampling": "robot_knots_5x(lateral_50+bending_100)",
        })),
    )
    return path

def compare_parity_qpos(
    path: Path,
    config: dict,
    *,
    threshold_m: float = 1e-3,
    enforce_threshold: bool = True,
    side: str = "R",
) -> dict:
    """Read one saved qpos bundle through analytic and SAPIEN TIP backends."""
    with np.load(path, allow_pickle=False) as bundle:
        qpos = np.asarray(bundle["robot_qpos"], dtype=np.float64)
        fingers = np.asarray(bundle["finger_indices"], dtype=np.int64)
        offsets = np.asarray(bundle["tip_offsets"], dtype=np.float64)
    lower, upper = get_joint_limits(config)
    sapien, config_offsets = _sapien_callback(config)
    if not np.array_equal(offsets, np.asarray(config_offsets, dtype=np.float64)):
        raise RuntimeError("parity file offsets differ from current config offsets")
    analytic = make_analytic_tip_callback(
        config, lower, upper, config_offsets, side=side
    )
    sapien_points = np.asarray([sapien(q, int(f)) for q, f in zip(qpos, fingers)])
    analytic_points = np.asarray([analytic(q, int(f)) for q, f in zip(qpos, fingers)])
    errors = np.linalg.norm(analytic_points - sapien_points, axis=1)
    report = {"threshold_m": threshold_m, "overall": {"max_m": float(errors.max()), "mean_m": float(errors.mean())}, "fingers": {}}
    for index, name in enumerate(FINGER_NAMES):
        values = errors[fingers == index]
        report["fingers"][name] = {"max_m": float(values.max()), "mean_m": float(values.mean())}
    if enforce_threshold and report["overall"]["max_m"] >= threshold_m:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report



def _write_manifest_archive(
    docs_root: Path,
    manifest: HandManifest,
    parity_qpos: Path,
    report: dict,
) -> None:
    """Write numeric JSON and readable parity gate archive together."""
    docs_root.mkdir(parents=True, exist_ok=True)
    json_path = docs_root / f"{manifest.hand_id}_fk_parity_m1.json"
    markdown_path = docs_root / f"{manifest.hand_id}_fk_parity_m1.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate = report["gate"]
    markdown_path.write_text(
        "# FK parity M1\n\n"
        f"- hand_id: `{manifest.hand_id}`\n"
        f"- manifest: `{manifest.hand_config}`\n"
        f"- parity_qpos: `{parity_qpos}`\n"
        f"- qpos_rows: `{report['qpos_rows']}`\n"
        f"- max_m: `{report['overall']['max_m']}`\n"
        f"- mean_m: `{report['overall']['mean_m']}`\n"
        f"- gate: `{gate['name']}` observed `{gate['observed_m']}` <= "
        f"`{gate['limit_m']}`: `{gate['passed']}`\n",
        encoding="utf-8",
    )


def _run_manifest_parity(manifest: HandManifest, gates_path: Path, docs_root: Path) -> dict:
    qpos_path = manifest.output_dir / "parity" / "parity_qpos.npz"
    report_path = manifest.output_dir / "parity" / "parity_report.json"
    spec = load_gate_spec(gates_path)
    config = get_config(manifest.hand_config)
    qpos_path.parent.mkdir(parents=True, exist_ok=True)
    write_parity_qpos(qpos_path, config)
    report = compare_parity_qpos(
        qpos_path,
        config,
        threshold_m=spec.fk_parity_max_m,
        enforce_threshold=False,
        side=manifest.side,
    )
    report.update({
        "hand_id": manifest.hand_id,
        "manifest": str(manifest.hand_config),
        "parity_qpos": str(qpos_path),
        "qpos_rows": 750,
        "gate": evaluate_fk_parity_gate(report, spec),
    })
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_manifest_archive(docs_root, manifest, qpos_path, report)
    if not report["gate"]["passed"]:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--gates", type=Path, default=Path("configs/gates.yaml"))
    parser.add_argument("--docs-root", type=Path, default=Path("docs/reports"))
    parser.add_argument("--hand", default=None)
    parser.add_argument("--human-anchors", type=Path, default=None)
    parser.add_argument("--parity-qpos", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.manifest is not None:
        if any(value is not None for value in (args.hand, args.human_anchors, args.parity_qpos, args.report)):
            parser.error("--manifest cannot be combined with legacy parity arguments")
        report = _run_manifest_parity(load_hand_manifest(args.manifest), args.gates, args.docs_root)
    else:
        if None in (args.hand, args.human_anchors, args.parity_qpos, args.report):
            parser.error("legacy invocation requires --hand --human-anchors --parity-qpos --report")
        if not args.human_anchors.is_file():
            parser.error(f"legacy --human-anchors does not exist: {args.human_anchors}")
        write_parity_qpos(args.parity_qpos, get_config(args.hand))
        report = compare_parity_qpos(args.parity_qpos, get_config(args.hand))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
