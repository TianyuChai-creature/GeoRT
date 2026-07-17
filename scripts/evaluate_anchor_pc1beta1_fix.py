#!/usr/bin/env python3
"""Evaluate the non-thumb PC1→beta1 anchor-direction correction.

Metrics:
- L1/L5 human beta and paired robot qpos for all five bending groups.
- H/R = ||N_H(tip_L5)-N_H(tip_L1)|| / ||N_R(tip_L5)-N_R(tip_L1)||.
- Per-finger metric-space anchor TIP residual for archived checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from geort.anchor.human_geometry import align_hts_to_palm, estimate_finger_angles
from geort.anchor.lateral_shrink_exact import exact_level_knots
from geort.analytic_fk import AnalyticFK
from geort.env.hand import HandKinematicModel
from geort.keypoint_normalization import normalize_finger_points
from geort.model import IKModel
from geort.utils.config_utils import get_config, parse_config_keypoint_info, select_keypoint_types

ROOT = Path(__file__).resolve().parents[1]
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
GROUPS = [
    {"finger": name, "keypoint_indices": [index], "joint_indices": list(range(4 * index, 4 * index + 4))}
    for index, name in enumerate(FINGERS)
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(points: np.ndarray, stats: dict) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - np.asarray(stats["center"], dtype=np.float64)) / float(stats["scale"])


def group_mask(values: dict[str, np.ndarray], finger: int, anchor_type: str) -> np.ndarray:
    return (values["finger_indices"] == finger) & (values["anchor_types"].astype(str) == anchor_type)


def make_model(checkpoint: Path, device: torch.device) -> IKModel:
    model = IKModel(finger_groups=GROUPS, n_total_joint=20).to(device)
    model.load_state_dict(torch.load(checkpoint / "last.pth", map_location=device, weights_only=True))
    model.eval()
    return model


def residuals(
    *, bundle: dict[str, np.ndarray], checkpoint: Path, fk: AnalyticFK, device: torch.device,
) -> list[dict]:
    stats = json.loads((checkpoint / "normalization.json").read_text())
    model = make_model(checkpoint, device)
    contexts = bundle["human_tip_contexts"].astype(np.float32)
    target = bundle["robot_points"].astype(np.float32)
    fingers = bundle["finger_indices"].astype(np.int64)
    normalized = normalize_finger_points(contexts, FINGERS, stats["human"])
    outputs = []
    with torch.no_grad():
        for start in range(0, len(normalized), 2048):
            point = torch.from_numpy(normalized[start:start + 2048]).to(device).float()
            outputs.append(fk(model(point)).cpu().numpy())
    tips = np.concatenate(outputs, axis=0)
    value = np.linalg.norm(tips[np.arange(len(fingers)), fingers] - target, axis=-1)
    return [
        {"finger": name, "count": int(np.count_nonzero(fingers == index)), "mean_m": float(value[fingers == index].mean()), "max_m": float(value[fingers == index].max())}
        for index, name in enumerate(FINGERS)
    ]


def directions_and_ratios(
    *, human: dict[str, np.ndarray], paired: dict[str, np.ndarray], raw: np.ndarray, stats: dict,
) -> tuple[list[dict], list[dict]]:
    aligned, _ = align_hts_to_palm(raw)
    angles = estimate_finger_angles(aligned)
    direction_rows = []
    ratio_rows = []
    for index, name in enumerate(FINGERS):
        for anchor_type in ("lateral", "bending"):
            human_mask = group_mask(human, index, anchor_type)
            paired_mask = group_mask(paired, index, anchor_type)
            human_tips = human["human_points"][human_mask]
            robot_tips = exact_level_knots(paired["robot_points"][paired_mask], paired["trajectory_t"][paired_mask])
            hn = normalize(human_tips, stats["human"][name])
            rn = normalize(robot_tips, stats["robot"][name])
            ratio_rows.append({
                "finger": name,
                "type": anchor_type,
                "human_span_normalized": float(np.linalg.norm(hn[-1] - hn[0])),
                "robot_span_normalized": float(np.linalg.norm(rn[-1] - rn[0])),
                "human_over_robot": float(np.linalg.norm(hn[-1] - hn[0]) / np.linalg.norm(rn[-1] - rn[0])),
            })
        human_mask = group_mask(human, index, "bending")
        paired_mask = group_mask(paired, index, "bending")
        sources = human["source_indices"][human_mask].astype(np.int64)
        arc = human["observed_parameters"][human_mask]
        qpos = exact_level_knots(paired["robot_qpos"][paired_mask], paired["trajectory_t"][paired_mask])
        for level in (0, 4):
            beta = angles.beta[int(sources[level]), index]
            direction_rows.append({
                "finger": name,
                "level": level + 1,
                "source_frame": int(sources[level]),
                "tip_arc_fraction": float(arc[level]),
                "human_beta1_rad": float(beta[0]), "human_beta2_rad": float(beta[1]), "human_beta3_rad": float(beta[2]),
                "robot_alpha_rad": float(qpos[level, 4 * index]), "robot_beta1_rad": float(qpos[level, 4 * index + 1]),
                "robot_beta2_rad": float(qpos[level, 4 * index + 2]), "robot_beta3_rad": float(qpos[level, 4 * index + 3]),
            })
    return direction_rows, ratio_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-human", type=Path, default=ROOT / "data/anchors_human_right_arc_bending_v2_ringmono.npz")
    parser.add_argument("--old-bundle", type=Path, default=ROOT / "data/anchors_custom_right_arc_bending_v2_lateral085_ringmono_frozenrobot.npz")
    parser.add_argument("--new-human", type=Path, default=ROOT / "data/anchors_human_right_arc_bending_v3_pc1beta1_ringmono.npz")
    parser.add_argument("--new-bundle", type=Path, default=ROOT / "data/anchors_custom_right_arc_bending_v3_pc1beta1_lateral085_ringmono_frozenrobot.npz")
    parser.add_argument("--c2b", type=Path, default=ROOT / "checkpoint/custom_right_2026-07-17_12-21-39_c2b_s42")
    parser.add_argument("--c2el", type=Path, default=ROOT / "checkpoint/custom_right_2026-07-17_17-47-22_C2eL_s42")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config("custom_right")
    tip_info = select_keypoint_types(parse_config_keypoint_info(config), allowed_types=("tip",))
    hand = HandKinematicModel.build_from_config(config)
    lower, upper = (np.asarray(value, dtype=np.float32) for value in hand.get_joint_limit())
    fk = AnalyticFK(config["urdf_path"], lower, upper, tip_offsets=tip_info["offset"]).to(device).eval()
    raw = np.load(ROOT / "data/hts_right.npy").astype(np.float64)
    old_human = dict(np.load(args.old_human, allow_pickle=True))
    old_bundle = dict(np.load(args.old_bundle, allow_pickle=True))
    new_human = dict(np.load(args.new_human, allow_pickle=True))
    new_bundle = dict(np.load(args.new_bundle, allow_pickle=True))
    stats = json.loads((args.c2b / "normalization.json").read_text())
    direction, ratios = directions_and_ratios(human=new_human, paired=new_bundle, raw=raw, stats=stats)
    result = {
        "protocol": {"units": {"joint": "rad", "residual": "m"}, "h_over_r": "norm(N_human(L5)-N_human(L1)) / norm(N_robot(L5)-N_robot(L1))"},
        "inputs": {"old_bundle": str(args.old_bundle.resolve().relative_to(ROOT)), "old_bundle_sha256": sha256(args.old_bundle), "new_bundle": str(args.new_bundle.resolve().relative_to(ROOT)), "new_bundle_sha256": sha256(args.new_bundle)},
        "new_l1_l5_direction": direction,
        "new_h_over_r": ratios,
        "new_h_over_r_scatter": {"min": float(min(row["human_over_robot"] for row in ratios)), "max": float(max(row["human_over_robot"] for row in ratios))},
        "residual_m": {"c2b_s42": {"old": residuals(bundle=old_bundle, checkpoint=args.c2b, fk=fk, device=device), "new": residuals(bundle=new_bundle, checkpoint=args.c2b, fk=fk, device=device)}, "c2el_s42": {"old": residuals(bundle=old_bundle, checkpoint=args.c2el, fk=fk, device=device), "new": residuals(bundle=new_bundle, checkpoint=args.c2el, fk=fk, device=device)}},
    }
    result["new_h_over_r_scatter"]["max_over_min"] = result["new_h_over_r_scatter"]["max"] / result["new_h_over_r_scatter"]["min"]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text("# PC1 beta1 anchor-direction fix\n\n```json\n" + json.dumps(result, indent=2, sort_keys=True) + "\n```\n")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
