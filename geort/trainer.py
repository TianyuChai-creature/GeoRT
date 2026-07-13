# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Train GeoRT with the AnyDexRT three-loss TIP correspondence objective."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Mapping
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from geort.data.prepare import normalize_finger_points
from geort.dataset import FramePointDataset, RobotKinematicsDataset
from geort.env.hand import HandKinematicModel
from geort.formatter import HandFormatter
from geort.loss import (
    anchor_align_loss,
    distance_preservation,
    motion_direction_loss,
    partial_chamfer,
)
from geort.model import FKModel, IKModel
from geort.utils.config_utils import (
    build_tip_finger_groups,
    get_config,
    parse_config_keypoint_info,
    save_json,
)
from geort.utils.path import get_data_root


DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 2048
DEFAULT_LEARNING_RATE = 1e-4


@dataclass(frozen=True)
class PreparedTrainingData:
    manifest_path: Path
    manifest: dict
    human_points: np.ndarray
    robot_points: np.ndarray
    keypoint_names: list[str]
    finger_names: list[str]
    human_ids: list[int]
    anchor_points: tuple[np.ndarray, np.ndarray] | None


def merge_dict_list(dicts):
    keys = dicts[0].keys()
    return {key: np.asarray([item[key] for item in dicts]) for key in keys}


def format_loss(value):
    return f"{value:.4e}" if math.fabs(value) < 1e-3 else f"{value:.4f}"


def should_save_epoch_checkpoint(epoch, n_epoch, save_every=0):
    del n_epoch
    return save_every > 0 and (epoch + 1) % save_every == 0


def generate_current_timestring():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def resolve_prepared_manifest(name: Path | str) -> Path:
    requested = Path(name)
    if requested.is_file():
        return requested.resolve()
    exact_name = requested.name if requested.suffix == ".json" else f"{name}.json"
    exact_path = Path(get_data_root()) / exact_name
    if exact_path.is_file():
        return exact_path.resolve()
    raise FileNotFoundError(
        f"No prepared manifest {exact_name!r}; run geort.data.prepare first"
    )


def _normalization_arrays(
    finger_names: list[str],
    stats: Mapping[str, Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(
        [stats[finger]["center"] for finger in finger_names], dtype=np.float32
    )
    scales = np.asarray(
        [stats[finger]["scale"] for finger in finger_names], dtype=np.float32
    ).reshape(1, -1, 1)
    return centers.reshape(1, -1, 3), scales


def _load_anchor_points(
    manifest: dict,
    manifest_path: Path,
    finger_names: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    anchor_spec = manifest.get("anchors")
    if anchor_spec is None:
        warnings.warn(
            "Prepared manifest has no anchors; L_align is disabled.",
            stacklevel=2,
        )
        return None

    if isinstance(anchor_spec, str):
        anchor_path = manifest_path.parent / anchor_spec
        normalized = False
    else:
        anchor_path = manifest_path.parent / anchor_spec["path"]
        normalized = bool(anchor_spec.get("normalized", False))
    with np.load(anchor_path) as anchors:
        human = np.asarray(anchors["human_points"], dtype=np.float32)
        robot = np.asarray(anchors["robot_points"], dtype=np.float32)
    expected_tail = (len(finger_names), 3)
    if human.shape[1:] != expected_tail or robot.shape[1:] != expected_tail:
        raise ValueError(
            f"Anchor points must have shape [N, {len(finger_names)}, 3]"
        )
    if not normalized:
        human = normalize_finger_points(
            human, finger_names, manifest["human"]["normalization"]
        )
        robot = normalize_finger_points(
            robot, finger_names, manifest["robot"]["normalization"]
        )
    return human, robot


def load_prepared_training_data(
    manifest_path: Path | str,
    *,
    expected_config: str,
) -> PreparedTrainingData:
    manifest_path = resolve_prepared_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config") != expected_config:
        raise ValueError(
            f"Manifest config {manifest.get('config')!r} does not match "
            f"{expected_config!r}"
        )
    prepared_path = manifest_path.parent / manifest["prepared_data"]
    with np.load(prepared_path) as prepared:
        human_points = np.asarray(prepared["human_points"], dtype=np.float32)
        robot_points = np.asarray(prepared["robot_points"], dtype=np.float32)
        keypoint_names = prepared["keypoint_names"].tolist()
        finger_names = prepared["finger_names"].tolist()
        human_ids = prepared["human_ids"].astype(int).tolist()

    if human_points.ndim != 3 or human_points.shape[1:] != (
        len(keypoint_names),
        3,
    ):
        raise ValueError("Prepared human_points has an invalid shape")
    if robot_points.ndim != 3 or robot_points.shape[1:] != (
        len(keypoint_names),
        3,
    ):
        raise ValueError("Prepared robot_points has an invalid shape")
    if keypoint_names != manifest["keypoint_names"]:
        raise ValueError("Prepared keypoint order does not match manifest")
    if finger_names != manifest["finger_names"]:
        raise ValueError("Prepared finger order does not match manifest")
    if len(set(finger_names)) != len(finger_names):
        raise ValueError("AnyDexRT prepared data must contain one TIP per finger")

    return PreparedTrainingData(
        manifest_path=manifest_path,
        manifest=manifest,
        human_points=human_points,
        robot_points=robot_points,
        keypoint_names=keypoint_names,
        finger_names=finger_names,
        human_ids=human_ids,
        anchor_points=_load_anchor_points(
            manifest, manifest_path, finger_names
        ),
    )


def compute_anydexrt_losses(
    *,
    human_points: torch.Tensor,
    mapped_points: torch.Tensor,
    jittered_human_points: torch.Tensor,
    jittered_mapped_points: torch.Tensor,
    robot_cloud: torch.Tensor,
    n_pairs: int,
) -> dict[str, torch.Tensor]:
    """Compute TIP-only AnyDexRT objectives with equal weighting."""
    return {
        "partial_chamfer": partial_chamfer(
            mapped_points.transpose(0, 1),
            robot_cloud.transpose(0, 1),
        ),
        "distance": distance_preservation(
            human_points, mapped_points, n_pairs=n_pairs
        ),
        "motion": motion_direction_loss(
            human_points,
            mapped_points,
            jittered_human_points,
            jittered_mapped_points,
        ),
    }


class GeoRTTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.hand = HandKinematicModel.build_from_config(
            self.config, render=False
        )

    def get_keypoint_info(self):
        return parse_config_keypoint_info(self.config)

    def get_robot_kinematics_dataset_path(self, postfix=False):
        output = f"data/{self.config['name']}"
        return output + ".npz" if postfix else output

    def get_robot_kinematics_dataset(self):
        dataset_path = self.get_robot_kinematics_dataset_path(postfix=True)
        if not os.path.exists(dataset_path):
            self.generate_robot_kinematics_dataset(n_total=100000, save=True)
        return RobotKinematicsDataset(
            dataset_path,
            keypoint_names=self.get_keypoint_info()["link"],
        )

    def generate_robot_kinematics_dataset(self, n_total=100000, save=True):
        info = self.get_keypoint_info()
        self.hand.initialize_keypoint(info["link"], info["offset"])
        lower, upper = self.hand.get_joint_limit()
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        all_qpos = []
        all_keypoints = []
        for _ in tqdm(range(n_total)):
            qpos = np.random.uniform(0, 1, len(lower)) * (upper - lower) + lower
            all_qpos.append(qpos)
            all_keypoints.append(self.hand.keypoint_from_qpos(qpos))
        dataset = {
            "qpos": np.asarray(all_qpos),
            "keypoint": merge_dict_list(all_keypoints),
        }
        if save:
            os.makedirs("data", exist_ok=True)
            np.savez(self.get_robot_kinematics_dataset_path(), **dataset)
        return dataset

    def get_fk_checkpoint_path(self):
        os.makedirs("checkpoint", exist_ok=True)
        return f"checkpoint/fk_model_{self.config['name']}.pth"

    def get_robot_neural_fk_model(self, force_train=False):
        info = self.get_keypoint_info()
        lower, upper = self.hand.get_joint_limit()
        formatter = HandFormatter(lower, upper)
        model = FKModel(keypoint_joints=info["joint"]).to(self.device)
        checkpoint_path = self.get_fk_checkpoint_path()
        if os.path.exists(checkpoint_path) and not force_train:
            try:
                state = torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:  # pragma: no cover
                state = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state)
        else:
            dataset = self.get_robot_kinematics_dataset()
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
            optimizer = optim.Adam(model.parameters(), lr=5e-4)
            criterion = nn.MSELoss()
            for epoch in range(200):
                total = 0.0
                for batch in dataloader:
                    keypoint = batch["keypoint"].to(self.device).float()
                    qpos = batch["qpos"].to(self.device).float()
                    predicted = model(formatter.normalize_torch(qpos))
                    loss = criterion(predicted, keypoint)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += loss.item()
                print(f"FK epoch={epoch} loss={total / len(dataloader):.6e}")
            torch.save(model.state_dict(), checkpoint_path)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _save_training_contract(
        self,
        directories: list[Path],
        data: PreparedTrainingData,
        *,
        n_epoch: int,
    ) -> None:
        lower, upper = self.hand.get_joint_limit()
        export_config = dict(self.config)
        export_config["joint"] = {
            "lower": np.asarray(lower).astype(float).tolist(),
            "upper": np.asarray(upper).astype(float).tolist(),
        }
        normalization = {
            "schema_version": 1,
            "source_manifest": data.manifest_path.name,
            "keypoint_names": data.keypoint_names,
            "finger_names": data.finger_names,
            "human_ids": data.human_ids,
            "human": data.manifest["human"]["normalization"],
            "robot": data.manifest["robot"]["normalization"],
        }
        metadata = {
            "human_data_manifest": str(data.manifest_path),
            "prepared_data": data.manifest["prepared_data"],
            "epoch": int(n_epoch),
            "batch_size": DEFAULT_BATCH_SIZE,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "objectives": [
                "partial_chamfer",
                "distance_preservation",
                "motion_direction",
            ],
            "equal_objective_weights": True,
            "anchors_enabled": data.anchor_points is not None,
            "timestamp": datetime.now().isoformat(),
        }
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            save_json(export_config, directory / "config.json")
            save_json(normalization, directory / "normalization.json")
            save_json(metadata, directory / "training_metadata.json")

    def train(self, human_data_path, **kwargs):
        info = self.get_keypoint_info()
        data = load_prepared_training_data(
            human_data_path,
            expected_config=self.config["name"],
        )
        expected_tip_names = [
            info["name"][index] for index in info["tip_indices"]
        ]
        if data.keypoint_names != expected_tip_names:
            raise ValueError(
                "Prepared TIP order does not match the hand config: "
                f"{data.keypoint_names} != {expected_tip_names}"
            )

        n_epoch = int(kwargs.get("epoch", DEFAULT_EPOCHS))
        save_every = int(kwargs.get("save_every", 0) or 0)
        update_latest = bool(kwargs.get("update_latest", True))
        tag = str(kwargs.get("tag", ""))
        if n_epoch <= 0:
            raise ValueError("epoch must be positive")

        fk_model = self.get_robot_neural_fk_model()
        ik_model = IKModel(
            finger_groups=build_tip_finger_groups(info),
            n_total_joint=len(self.config["joint_order"]),
        ).to(self.device)
        optimizer = optim.AdamW(
            ik_model.parameters(), lr=DEFAULT_LEARNING_RATE
        )

        timestamped = Path(
            f"checkpoint/{self.config['name']}_{generate_current_timestring()}"
            + (f"_{tag}" if tag else "")
        )
        latest = Path(f"checkpoint/{self.config['name']}_last")
        save_directories = [timestamped]
        if update_latest:
            save_directories.append(latest)
        self._save_training_contract(
            save_directories, data, n_epoch=n_epoch
        )

        human_dataset = FramePointDataset(data.human_points)
        batch_size = min(DEFAULT_BATCH_SIZE, len(human_dataset))
        drop_last = len(human_dataset) % batch_size == 1
        dataloader = DataLoader(
            human_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
        )
        robot_cloud = torch.from_numpy(data.robot_points)
        tip_indices = info["tip_indices"]
        robot_centers_np, robot_scales_np = _normalization_arrays(
            data.finger_names, data.manifest["robot"]["normalization"]
        )
        _, human_scales_np = _normalization_arrays(
            data.finger_names, data.manifest["human"]["normalization"]
        )
        robot_centers = torch.from_numpy(robot_centers_np).to(self.device)
        robot_scales = torch.from_numpy(robot_scales_np).to(self.device)
        human_scales = torch.from_numpy(human_scales_np).to(self.device)

        anchor_loader = None
        anchor_iterator = None
        if data.anchor_points is not None:
            anchor_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(data.anchor_points[0]),
                    torch.from_numpy(data.anchor_points[1]),
                ),
                batch_size=32,
                shuffle=True,
            )
            anchor_iterator = iter(anchor_loader)

        def map_tips(points):
            metric_keypoints = fk_model(ik_model(points))
            metric_tips = metric_keypoints[:, tip_indices, :]
            return (metric_tips - robot_centers) / robot_scales

        history = []
        for epoch in range(n_epoch):
            totals = {
                "partial_chamfer": 0.0,
                "distance": 0.0,
                "motion": 0.0,
                "anchor": 0.0,
                "total": 0.0,
            }
            for batch_index, batch in enumerate(dataloader):
                human = batch.to(self.device).float()
                mapped = map_tips(human)

                directions = F.normalize(
                    torch.randn_like(human), dim=-1, eps=1e-8
                )
                metric_step = 0.001 + torch.rand(
                    (len(human), 1, 1), device=self.device
                ) * 0.01
                jittered_human = (
                    human + directions * metric_step / human_scales
                )
                jittered_mapped = map_tips(jittered_human)

                target_count = min(DEFAULT_BATCH_SIZE, len(robot_cloud))
                target_indices = torch.randint(
                    len(robot_cloud), (target_count,)
                )
                target = robot_cloud[target_indices].to(self.device)
                losses = compute_anydexrt_losses(
                    human_points=human,
                    mapped_points=mapped,
                    jittered_human_points=jittered_human,
                    jittered_mapped_points=jittered_mapped,
                    robot_cloud=target,
                    n_pairs=min(DEFAULT_BATCH_SIZE, len(human)),
                )
                total_loss = sum(losses.values())
                anchor_loss = torch.zeros((), device=self.device)

                if anchor_iterator is not None:
                    try:
                        human_anchor, robot_anchor = next(anchor_iterator)
                    except StopIteration:
                        anchor_iterator = iter(anchor_loader)
                        human_anchor, robot_anchor = next(anchor_iterator)
                    human_anchor = human_anchor.to(self.device).float()
                    robot_anchor = robot_anchor.to(self.device).float()
                    anchor_loss = anchor_align_loss(
                        map_tips(human_anchor), robot_anchor
                    )
                    total_loss = total_loss + anchor_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                for name, value in losses.items():
                    totals[name] += value.item()
                totals["anchor"] += anchor_loss.item()
                totals["total"] += total_loss.item()

                if batch_index % 50 == 0:
                    print(
                        f"Epoch {epoch} |"
                        f" P-Chamfer {format_loss(losses['partial_chamfer'].item())}"
                        f" | Dist {format_loss(losses['distance'].item())}"
                        f" | Motion {format_loss(losses['motion'].item())}"
                        f" | Align {format_loss(anchor_loss.item())}"
                    )

            batches = len(dataloader)
            epoch_metrics = {
                name: value / batches for name, value in totals.items()
            }
            history.append(epoch_metrics)
            for directory in save_directories:
                save_json(history, directory / "training_history.json")
            state = ik_model.state_dict()
            if should_save_epoch_checkpoint(
                epoch, n_epoch, save_every=save_every
            ):
                torch.save(
                    state, timestamped / f"epoch_{epoch + 1}.pth"
                )
            torch.save(state, timestamped / "last.pth")
            if update_latest:
                torch.save(state, latest / "last.pth")
        return history


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-hand", required=True)
    parser.add_argument("-human_data", required=True, help="Prepared manifest")
    parser.add_argument("-ckpt_tag", default="")
    parser.add_argument("--save_every", type=int, default=0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    config = get_config(args.hand)
    trainer = GeoRTTrainer(config)
    manifest_path = resolve_prepared_manifest(args.human_data)
    print(f"Training with prepared manifest: {manifest_path}")
    trainer.train(
        manifest_path,
        tag=args.ckpt_tag,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
