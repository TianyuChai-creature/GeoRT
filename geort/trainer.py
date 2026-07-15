# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
# sapien is used indirectly via HandKinematicModel (geort/env/hand.py).
# No direct sapien calls remain in trainer.py after SAPIEN 3 migration.
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from geort.utils.hand_utils import get_entity_by_name, get_active_joints, get_active_joint_indices
from geort.utils.path import get_human_data
from geort.dataset_manifest import maybe_load_dataset_manifest
from geort.utils.config_utils import get_config, parse_config_keypoint_info, save_json, select_keypoint_types
from geort.model import FKModel, IKModel 
from geort.env.hand import HandKinematicModel
from geort.loss import partial_chamfer_distance, distance_preservation, local_motion_loss
from geort.keypoint_normalization import (
    fit_finger_normalization,
    normalize_finger_points,
    normalize_finger_points_torch,
    normalization_stats_to_json,
)
from geort.formatter import HandFormatter
from geort.dataset import RobotKinematicsDataset, MultiPointDataset, FramePointDataset
from geort.training_targets import build_training_metadata, resolve_chamfer_target_path, save_training_metadata
from datetime import datetime
from tqdm import tqdm 
import os
from pathlib import Path 
import math

def merge_dict_list(dl):
    keys = dl[0].keys()
    
    result = {k: [] for k in keys}
    for data in dl:
        for k in keys:
            result[k].append(data[k])
    
    result = {k: np.array(v) for k, v in result.items()}
    return result

def format_loss(value):
    return f"{value:.4e}" if math.fabs(value) < 1e-3 else f"{value:.4f}"





def compute_tip_pinch_loss(point, embedded_point, pinch_pairs, threshold=0.015):
    if not pinch_pairs:
        return torch.zeros((), device=point.device, dtype=point.dtype)

    pinch_loss = torch.zeros((), device=point.device, dtype=point.dtype)
    for i, j in pinch_pairs:
        distance = point[:, i, ...] - point[:, j, ...]
        mask = (torch.norm(distance, dim=-1) < threshold).to(point.dtype)
        e_distance = ((embedded_point[:, i, ...] - embedded_point[:, j, ...]) ** 2).sum(dim=-1)
        pinch_loss = pinch_loss + (mask * e_distance).sum() / mask.sum().clamp_min(1e-7)

    return pinch_loss



def non_thumb_mcp1_joint_indices(joint_order):
    return [
        idx
        for idx, name in enumerate(joint_order)
        if name.endswith("MCP1") and not name.startswith("F1-")
    ]


def compute_mcp1_fist_prior_loss(joint, *, fist_mask, mcp1_indices, target_alpha=0.5):
    if not mcp1_indices:
        return torch.zeros((), device=joint.device, dtype=joint.dtype)
    if target_alpha < 0.0 or target_alpha > 1.0:
        raise ValueError("target_alpha must be in [0, 1]")

    mask = fist_mask.to(device=joint.device, dtype=joint.dtype).reshape(-1)
    if mask.sum() <= 0:
        return torch.zeros((), device=joint.device, dtype=joint.dtype)

    selected = joint[:, mcp1_indices]
    target = selected.detach() + float(target_alpha) * (1.0 - selected.detach())
    loss = ((selected - target) ** 2) * mask.unsqueeze(1)
    return loss.sum() / (mask.sum().clamp_min(1e-7) * len(mcp1_indices))


def compute_mcp1_fist_prior_mask(frames, *, top_fraction=0.05, mcp_weight=2.0, pip_weight=1.0, dip_weight=0.7):
    if top_fraction <= 0.0:
        return None
    if top_fraction > 1.0:
        raise ValueError("mcp1_fist_prior_top_fraction must be <= 1.0")

    from geort.mocap.hts_prepare_training import compute_mcp_weighted_fist_curl_score

    scores = compute_mcp_weighted_fist_curl_score(
        frames,
        mcp_weight=mcp_weight,
        pip_weight=pip_weight,
        dip_weight=dip_weight,
    )
    selected_count = max(1, int(np.ceil(frames.shape[0] * top_fraction)))
    selected = np.argsort(scores, kind="stable")[:selected_count]
    mask = np.zeros(frames.shape[0], dtype=np.float32)
    mask[selected] = 1.0
    return mask

def find_human_weight_path(human_data_path):
    manifest = maybe_load_dataset_manifest(human_data_path)
    if manifest is None:
        return None
    return manifest.weights_path if manifest.weights_path and manifest.weights_path.exists() else None


def describe_human_weight_source(human_data_path):
    manifest = maybe_load_dataset_manifest(human_data_path)
    if manifest is None:
        return None
    if manifest.weights is not None:
        source = manifest.manifest_path.as_posix() if manifest.manifest_path else str(human_data_path)
        return f"inline weights in {source}"
    weight_path = find_human_weight_path(human_data_path)
    return weight_path.as_posix() if weight_path is not None else None


def resolve_human_training_input(human_data):
    manifest = maybe_load_dataset_manifest(human_data)
    if manifest is not None:
        return manifest.manifest_path
    return get_human_data(human_data)


def should_save_epoch_checkpoint(epoch, n_epoch, save_every=0):
    del n_epoch
    return save_every > 0 and (epoch + 1) % save_every == 0


def prepare_human_training_dataset(
    human_data_path,
    human_ids,
    finger_names,
    *,
    mcp1_fist_prior_enabled=False,
    mcp1_fist_prior_top_fraction=0.05,
    mcp1_fist_prior_mcp_weight=2.0,
    mcp1_fist_prior_pip_weight=1.0,
    mcp1_fist_prior_dip_weight=0.7,
):
    manifest = maybe_load_dataset_manifest(human_data_path)
    data_path = manifest.data_path if manifest is not None else Path(human_data_path)

    human_points = np.load(data_path)
    selected_points = np.array([human_points[:, idx, :3] for idx in human_ids], dtype=np.float32).transpose(1, 0, 2)

    weights = None
    if manifest is not None and manifest.weights is not None:
        weights = np.asarray(manifest.weights, dtype=np.float32)
    elif manifest is not None and manifest.weights_path is not None:
        weights = np.load(manifest.weights_path).astype(np.float32)

    if weights is not None and weights.shape != (selected_points.shape[0],):
        raise ValueError(
            f"Frame weights length {weights.shape} does not match frame count {selected_points.shape[0]}"
        )

    # Per-finger normalization: fit from the full human dataset once,
    # then normalize each finger independently into [-1, 1].
    human_stats = fit_finger_normalization(selected_points, finger_names)
    normalized_points = normalize_finger_points(selected_points, finger_names, human_stats)

    # Keep metric points for pinch loss which uses physical thresholds.
    frame_fields = {"metric_point": selected_points}
    if mcp1_fist_prior_enabled:
        mask = compute_mcp1_fist_prior_mask(
            human_points,
            top_fraction=mcp1_fist_prior_top_fraction,
            mcp_weight=mcp1_fist_prior_mcp_weight,
            pip_weight=mcp1_fist_prior_pip_weight,
            dip_weight=mcp1_fist_prior_dip_weight,
        )
        frame_fields["mcp1_fist_prior_mask"] = mask

    return FramePointDataset(normalized_points, frame_fields=frame_fields), weights, human_stats


def get_float_list_from_np(np_vector):
    float_list = np_vector.tolist()
    float_list = [float(x) for x in float_list]
    return float_list

def generate_current_timestring():
    """
        Utility Function. Generate a current timestring in the format 'YYYY-MM-DD_HH-MM-SS'.
    """
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

class GeoRTTrainer:
    def __init__(self, config):
        self.config = config
        self.hand = HandKinematicModel.build_from_config(self.config)

    def get_robot_pointcloud(self, keypoint_names, chamfer_target="uniform", chamfer_target_path=None):
        '''
            Utility getter function. Return the robot fingertip point cloud.
        '''
        kinematics_dataset = self.get_robot_kinematics_dataset(
            chamfer_target=chamfer_target,
            chamfer_target_path=chamfer_target_path,
        )
        return kinematics_dataset.export_robot_pointcloud(keypoint_names)
        
    def get_robot_kinematics_dataset(self, chamfer_target="uniform", chamfer_target_path=None):
        '''
            Utility getter function. Return the robot kinematics dataset.

            FK training always calls this with the default uniform target. IK
            chamfer can request a prebuilt human-shaped target cloud.
        '''
        target = resolve_chamfer_target_path(
            hand_name=self.config["name"],
            chamfer_target=chamfer_target,
            explicit_path=chamfer_target_path,
        )
        dataset_path = target.path.as_posix()
        if chamfer_target == "uniform" and not os.path.exists(dataset_path):
            dataset = self.generate_robot_kinematics_dataset(n_total=100000, save=True)
            dataset_path = self.get_robot_kinematics_dataset_path(postfix=True)
        
        keypoint_names = self.get_keypoint_info()["link"]

        kinematics_dataset = RobotKinematicsDataset(dataset_path, keypoint_names=keypoint_names)
        return kinematics_dataset

    def get_robot_kinematics_dataset_path(self, postfix=False):
        '''
            Utility getter function. Return the path to the robot kinematics dataset.
        '''
        data_name = self.config["name"]
        
        out = f"data/{data_name}"
        if postfix:
            out += '.npz'
        return out 

    def get_keypoint_info(self):
        return select_keypoint_types(parse_config_keypoint_info(self.config), allowed_types=("tip",))

    def generate_robot_kinematics_dataset(self, n_total=100000, save=True):
        '''
            This function will generate a (joint position, keypoint position) dataset. 
            - The joint order is specified by "joint_order" in configuration.
            - The keypoint order is specified by "fingertip_link" field in configuration.
        '''
        info = self.get_keypoint_info()
        
        self.hand.initialize_keypoint(keypoint_link_names=info["link"], keypoint_offsets=info["offset"])

        data = []
        joint_range_low, joint_range_high = self.hand.get_joint_limit() # joint order is based on user config specification.
        joint_range_low = np.array(joint_range_low)
        joint_range_high = np.array(joint_range_high)

        all_data_qpos = []
        all_data_keypoint = []
        
        for _ in tqdm(range(n_total)):
            qpos = np.random.uniform(0, 1, len(joint_range_low)) * (joint_range_high - joint_range_low) + joint_range_low
            keypoint = self.hand.keypoint_from_qpos(qpos)
            all_data_qpos.append(qpos)
            all_data_keypoint.append(keypoint)
            
        all_data_keypoint = merge_dict_list(all_data_keypoint)    
        
        dataset = {"qpos": all_data_qpos, "keypoint": all_data_keypoint}

        if save:
            # save data to disk for future use.
            os.makedirs("data", exist_ok=True)
            np.savez(self.get_robot_kinematics_dataset_path(), **dataset)

        return dataset

    def get_fk_checkpoint_path(self):
        name = self.config["name"]
        os.makedirs("checkpoint", exist_ok=True)
        return f"checkpoint/fk_model_{name}.pth"
    
    def get_robot_neural_fk_model(self, force_train=False):
        '''
            This function will return a forward kinematics model.
            If the fk model does not exist, this function will train one first.
        '''

        # Normalizer.
        joint_lower_limit, joint_upper_limit = self.hand.get_joint_limit()
        qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit)
        
        # Model.
        print(self.get_keypoint_info()["joint"])
        fk_model = FKModel(keypoint_joints=self.get_keypoint_info()["joint"]).cuda()
        
        # If the model exists, load it.
        fk_checkpoint_path = self.get_fk_checkpoint_path()
        if os.path.exists(fk_checkpoint_path) and not force_train:
            fk_model.load_state_dict(torch.load(fk_checkpoint_path))

        else:
            # If the model does not exist, train it.
            print("Train Neural Forward Kinematics (FK) from Scratch")
        
            fk_dataset = self.get_robot_kinematics_dataset()
            fk_dataloader = DataLoader(fk_dataset, batch_size=256, shuffle=True)
            fk_optim = optim.Adam(fk_model.parameters(), lr=5e-4)

            criterion_fk = nn.MSELoss()
            for epoch in range(200):
                all_fk_error = 0
                for batch_idx, batch in enumerate(fk_dataloader):
                    keypoint = batch["keypoint"].cuda().float()
                    qpos = batch["qpos"].cuda().float() 
                    qpos = qpos_normalizer.normalize_torch(qpos)
                    predicted_keypoint = fk_model(qpos)
                    fk_optim.zero_grad()
                    loss = criterion_fk(predicted_keypoint, keypoint)
                    loss.backward()
                    fk_optim.step()

                    all_fk_error += loss.item()
                
                avg_fk_error = all_fk_error / (batch_idx + 1)
                print(f"Neural FK Training Epoch: {epoch}; Training Loss: {avg_fk_error}")
            
            torch.save(fk_model.state_dict(), fk_checkpoint_path)
        
        fk_model.eval()
        return fk_model
        
    def train(self, human_data_path, **kwargs):
        '''
            This is the main trainer.
        '''

        keypoint_info = self.get_keypoint_info()
        fk_model = self.get_robot_neural_fk_model()
        ik_model = IKModel(
            finger_groups=keypoint_info["finger_groups"],
            n_total_joint=len(self.config["joint_order"]),
        ).cuda()
        os.makedirs("./checkpoint", exist_ok=True)

        ik_optim = optim.AdamW(ik_model.parameters(), lr=1e-4)

        # Workspace.
        exp_tag = kwargs.get("tag", "")
        n_epoch = kwargs.get("epoch", 200)
        hand_model_name = self.config["name"]

        w_chamfer = kwargs.get("w_chamfer", 80.0)
        w_distance = kwargs.get("w_distance", 1.0)
        w_curvature = kwargs.get("w_curvature", 0.1)
        motion_delta = float(kwargs.get("motion_delta", 0.01))
        w_collision = kwargs.get("w_collision", 0.0)
        w_pinch = kwargs.get("w_pinch", 1.0)
        pinch_threshold = kwargs.get("pinch_threshold", 0.015)
        w_mcp1_fist_prior = kwargs.get("w_mcp1_fist_prior", 0.0)
        mcp1_fist_prior_top_fraction = kwargs.get("mcp1_fist_prior_top_fraction", 0.05)
        mcp1_fist_prior_target_alpha = kwargs.get("mcp1_fist_prior_target_alpha", 0.5)
        mcp1_fist_prior_mcp_weight = kwargs.get("mcp1_fist_prior_mcp_weight", 2.0)
        mcp1_fist_prior_pip_weight = kwargs.get("mcp1_fist_prior_pip_weight", 1.0)
        mcp1_fist_prior_dip_weight = kwargs.get("mcp1_fist_prior_dip_weight", 0.7)
        save_every = int(kwargs.get("save_every", 0) or 0)
        update_latest = bool(kwargs.get("update_latest", True))
        chamfer_target = kwargs.get("chamfer_target", "uniform")
        chamfer_target_path = kwargs.get("chamfer_target_path", None)
        mold_path = kwargs.get("mold_path", None)

        save_dir = f"./checkpoint/{hand_model_name}_{generate_current_timestring()}"
        if exp_tag != '':
            save_dir += f'_{exp_tag}'
        last_save_dir = f"./checkpoint/{hand_model_name}_last"

        os.makedirs(save_dir, exist_ok=True)
        if update_latest:
            os.makedirs(last_save_dir, exist_ok=True)

        # Save the config including robot joint info to the checkpoint directory.
        joint_lower_limit, joint_upper_limit = self.hand.get_joint_limit()

        export_config = self.config.copy()
        export_config["joint"] = {
            "lower": get_float_list_from_np(joint_lower_limit),
            "upper": get_float_list_from_np(joint_upper_limit)
        }

        save_json(export_config, Path(save_dir) / "config.json")
        if update_latest:
            save_json(export_config, Path(last_save_dir) / "config.json")

        resolved_chamfer_target = resolve_chamfer_target_path(
            hand_name=hand_model_name,
            chamfer_target=chamfer_target,
            explicit_path=chamfer_target_path,
        )
        # Dataset.
        robot_keypoint_names = keypoint_info['link']
        n_keypoints = len(robot_keypoint_names)

        effective_chamfer_target_path = (
            resolved_chamfer_target.path
            if chamfer_target != "uniform" or chamfer_target_path is not None
            else None
        )
        robot_points_metric = self.get_robot_pointcloud(
            robot_keypoint_names,
            chamfer_target=chamfer_target,
            chamfer_target_path=effective_chamfer_target_path,
        )
        finger_names = keypoint_info["finger"]
        robot_stats = fit_finger_normalization(
            robot_points_metric.transpose(1, 0, 2), finger_names
        )
        robot_points = normalize_finger_points(
            robot_points_metric.transpose(1, 0, 2), finger_names, robot_stats
        ).transpose(1, 0, 2)
        metadata_target_path = (
            resolved_chamfer_target.path
            if effective_chamfer_target_path is not None
            else Path(self.get_robot_kinematics_dataset_path(postfix=True))
        )
        training_metadata = build_training_metadata(
            chamfer_target=chamfer_target,
            target_path=metadata_target_path,
            mold_path=mold_path,
            human_data_path=human_data_path,
            n_epoch=n_epoch,
            loss_weights={
                "w_chamfer": w_chamfer,
                "w_distance": w_distance,
                "w_curvature": w_curvature,
                "w_collision": w_collision,
                "w_pinch": w_pinch,
                "w_mcp1_fist_prior": w_mcp1_fist_prior,
            },
            cli_args={
                "tag": exp_tag,
                "chamfer_target": chamfer_target,
                "chamfer_target_path": str(chamfer_target_path) if chamfer_target_path else None,
                "mold_path": str(mold_path) if mold_path else None,
                "save_every": save_every,
                "update_latest": update_latest,
                "mcp1_fist_prior_top_fraction": mcp1_fist_prior_top_fraction,
                "mcp1_fist_prior_target_alpha": mcp1_fist_prior_target_alpha,
                "mcp1_fist_prior_mcp_weight": mcp1_fist_prior_mcp_weight,
                "mcp1_fist_prior_pip_weight": mcp1_fist_prior_pip_weight,
                "mcp1_fist_prior_dip_weight": mcp1_fist_prior_dip_weight,
            },
        )
        save_training_metadata(Path(save_dir) / "training_metadata.json", training_metadata)
        if update_latest:
            save_training_metadata(Path(last_save_dir) / "training_metadata.json", training_metadata)

        human_finger_idxes = keypoint_info["human_id"]
        pinch_pairs = keypoint_info["pinch_pairs"]
        mcp1_prior_joint_indices = non_thumb_mcp1_joint_indices(self.config["joint_order"])
        mcp1_fist_prior_enabled = w_mcp1_fist_prior > 0.0
        for robot_keypoint_name, human_id in zip(robot_keypoint_names, human_finger_idxes):
            print(f"Robot Keypoint {robot_keypoint_name}: Human Id: {human_id}")

        point_dataset_human, human_frame_weights, human_stats = prepare_human_training_dataset(
            human_data_path,
            human_finger_idxes,
            finger_names,
            mcp1_fist_prior_enabled=mcp1_fist_prior_enabled,
            mcp1_fist_prior_top_fraction=mcp1_fist_prior_top_fraction,
            mcp1_fist_prior_mcp_weight=mcp1_fist_prior_mcp_weight,
            mcp1_fist_prior_pip_weight=mcp1_fist_prior_pip_weight,
            mcp1_fist_prior_dip_weight=mcp1_fist_prior_dip_weight,
        )
        normalization_metadata = {
            "schema_version": 1,
            "keypoint_type": "tip",
            "keypoint_names": keypoint_info["name"],
            "keypoint_links": robot_keypoint_names,
            "human_ids": human_finger_idxes,
            "finger_names": finger_names,
            "human_data_source": str(human_data_path),
            "human": normalization_stats_to_json(human_stats),
            "robot": normalization_stats_to_json(robot_stats),
        }
        save_json(normalization_metadata, Path(save_dir) / "normalization.json")
        if update_latest:
            save_json(normalization_metadata, Path(last_save_dir) / "normalization.json")
        if human_frame_weights is not None:
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(human_frame_weights).double(),
                num_samples=len(point_dataset_human),
                replacement=True,
            )
            point_dataloader = DataLoader(point_dataset_human, batch_size=2048, sampler=sampler)
            print(f"Using frame weights from {describe_human_weight_source(human_data_path)}")
        else:
            point_dataloader = DataLoader(point_dataset_human, batch_size=2048, shuffle=True)

        # Training / Optimization
        for epoch in range(n_epoch):
            for batch_idx, batch in enumerate(point_dataloader):
                direction_loss = 0  # unused, kept for backward compat

                mcp1_fist_prior_mask = None
                if isinstance(batch, dict):
                    point = batch["point"].cuda()  # normalized [B, K, 3]
                    metric_point = batch["metric_point"].cuda()  # metric [B, K, 3]
                    if "mcp1_fist_prior_mask" in batch:
                        mcp1_fist_prior_mask = batch["mcp1_fist_prior_mask"].cuda()
                else:
                    point = batch.cuda()
                    metric_point = point
                joint = ik_model(point)  # [B, DOF]
                embedded_metric = fk_model(joint)  # metric [B, K, 3]
                embedded_point = normalize_finger_points_torch(
                    embedded_metric, finger_names, robot_stats
                )  # normalized [B, K, 3]

                # [Pinch Loss] — uses metric space (physical threshold in meters).
                pinch_loss = compute_tip_pinch_loss(metric_point, embedded_metric, pinch_pairs, threshold=pinch_threshold)

                # [Curvature loss] -- Ensuring flatness.
                # Perturb in normalized space (≈1% of [-1, 1] range).
                direction = F.normalize(torch.randn_like(point), dim=-1, p=2)
                delta1 = direction * 0.02
                point_delta_1p = point + delta1 
                point_delta_1n = point - delta1 

                embedded_point_p = normalize_finger_points_torch(
                    fk_model(ik_model(point_delta_1p)), finger_names, robot_stats
                )
                embedded_point_n = normalize_finger_points_torch(
                    fk_model(ik_model(point_delta_1n)), finger_names, robot_stats
                )
                curvature_by_keypoint = ((embedded_point_p + embedded_point_n - 2 * embedded_point) ** 2).mean(dim=(0, 2))
                curvature_loss = curvature_by_keypoint.mean()
                
                # [Chamfer loss]
                selected_idx = np.random.randint(0, robot_points.shape[1], 2048) 
                target = torch.from_numpy(robot_points[:, selected_idx, :]).permute(1, 0, 2).float().cuda()
                
                chamfer_by_keypoint = []
                for i in range(n_keypoints):
                    chamfer_by_keypoint.append(partial_chamfer_distance(embedded_point[:, i, :].unsqueeze(0), target[:, i, :].unsqueeze(0)))
                chamfer_loss = torch.stack(chamfer_by_keypoint).mean()

                # [Distance Preservation]
                # Per-finger isometry: penalize changes in pairwise distances
                # among the batch of fingertip positions before vs after mapping.
                distance_loss = distance_preservation(point, embedded_point)

                # [Local Motion Preservation]
                # T = I (identity local frame) for this version.
                # TODO(Step 4): replace with per-finger local coordinate frames.
                direction = F.normalize(torch.randn_like(point), dim=-1, p=2)
                point_delta = point + direction * motion_delta

                joint_delta = ik_model(point_delta)
                embedded_point_delta = normalize_finger_points_torch(
                    fk_model(joint_delta), finger_names, robot_stats
                )

                d_human = point_delta - point
                d_robot = embedded_point_delta - embedded_point
                motion_loss = local_motion_loss(d_human, d_robot)

                # [MCP1 fist prior]
                if mcp1_fist_prior_mask is not None:
                    mcp1_fist_prior_loss = compute_mcp1_fist_prior_loss(
                        joint,
                        fist_mask=mcp1_fist_prior_mask,
                        mcp1_indices=mcp1_prior_joint_indices,
                        target_alpha=mcp1_fist_prior_target_alpha,
                    )
                else:
                    mcp1_fist_prior_loss = torch.zeros((), device=joint.device, dtype=joint.dtype)

                # [Collision loss]
                # if classifier is not None:
                #     real_labels = torch.ones(joint.size(0), dtype=torch.long).to(joint.device)
                #     # Discriminator's output for generated data
                #     safe_logits = classifier(joint)
                #     criterion = nn.CrossEntropyLoss()
                #     # Generator loss is the cross-entropy loss between the fake outputs and the label 1 (real)
                #     collision_loss = criterion(safe_logits, real_labels)
                
                # collision Loss integration pending.
                collision_loss = torch.tensor([0.0]).cuda()

                loss = motion_loss + \
                       chamfer_loss * w_chamfer + \
                       distance_loss * w_distance + \
                       curvature_loss * w_curvature + \
                       collision_loss * w_collision + \
                       pinch_loss * w_pinch + \
                       mcp1_fist_prior_loss * w_mcp1_fist_prior

                ik_optim.zero_grad()
                loss.backward()
                ik_optim.step()

                if batch_idx % 50 == 0:
                    print(
                        f"Epoch {epoch} | Losses"
                        f" - Motion: {format_loss(motion_loss.item())}"
                        f" - P-Chamfer: {format_loss(chamfer_loss.item())}"
                        f" - Distance: {format_loss(distance_loss.item())}"
                        f" - Curvature: {format_loss(curvature_loss.item())}"
                        f" - Collision: {format_loss(collision_loss.item())}"
                        f" - Pinch: {format_loss(pinch_loss.item())}"
                        f" - MCP1FistPrior: {format_loss(mcp1_fist_prior_loss.item())}"
                    )


            # Saving the checkpoint.
            state_dict = ik_model.state_dict()
            if should_save_epoch_checkpoint(epoch, n_epoch, save_every=save_every):
                torch.save(state_dict, Path(save_dir) / f"epoch_{epoch}.pth")
            torch.save(state_dict, Path(save_dir) / "last.pth")

            if update_latest:
                torch.save(state_dict, Path(last_save_dir) / "last.pth")

        return 


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-hand', type=str, default='allegro_right')
    parser.add_argument('-human_data', type=str, default='human')
    parser.add_argument('-ckpt_tag', type=str, default='')

    parser.add_argument('--w_chamfer', type=float, default=80.0)
    parser.add_argument('--w_distance', type=float, default=1.0)
    parser.add_argument('--w_curvature', type=float, default=0.1)
    parser.add_argument('--motion_delta', type=float, default=0.01, help='Perturbation magnitude in normalized space (default 1%% of [-1,1] range).')
    parser.add_argument('--w_collision', type=float, default=0.0)
    parser.add_argument('--w_pinch', type=float, default=1.0)
    parser.add_argument('--pinch_threshold', type=float, default=0.015)
    parser.add_argument('--w_mcp1_fist_prior', type=float, default=0.0, help='Weight for strong-fist MCP1 prior loss; 0 disables it.')
    parser.add_argument('--mcp1_fist_prior_top_fraction', type=float, default=0.05, help='Fraction of strongest fist frames used by MCP1 prior.')
    parser.add_argument('--mcp1_fist_prior_target_alpha', type=float, default=0.5, help='Blend from predicted normalized MCP1 toward upper limit for prior target.')
    parser.add_argument('--mcp1_fist_prior_mcp_weight', type=float, default=2.0)
    parser.add_argument('--mcp1_fist_prior_pip_weight', type=float, default=1.0)
    parser.add_argument('--mcp1_fist_prior_dip_weight', type=float, default=0.7)
    parser.add_argument('--save_every', type=int, default=0, help='Save epoch_N.pth every N epochs; 0 keeps only last.pth.')
    parser.add_argument('--chamfer_target', choices=('uniform', 'human'), default='uniform', help='Chamfer target cloud source.')
    parser.add_argument('--chamfer_target_path', default=None, help='Explicit chamfer target .npz path. Human defaults to data/<hand>_humanshaped.npz.')
    parser.add_argument('--mold_path', default=None, help='Optional mold.json path to record in checkpoint metadata.')
    parser.add_argument('--no_update_latest', action='store_true', help='Do not update checkpoint/<hand>_last.')

    args = parser.parse_args()

    # Guard: motion_delta below ~0.002 (0.1 mm in metric space) enters the
    # float32 normalisation round-trip noise floor and corrupts the direction
    # signal (see test_analytic_fk.py noise-floor calibration).
    if args.motion_delta < 0.002:
        print(
            f"WARNING: --motion_delta={args.motion_delta} is below the 0.002 "
            f"noise-floor lower bound.  Direction signal may be dominated by "
            f"float32 round-trip noise.  Consider --motion_delta >= 0.005."
        )

    config = get_config(args.hand)
    trainer = GeoRTTrainer(config)

    human_data_path = resolve_human_training_input(args.human_data)
    print("Training with human data:", human_data_path.as_posix())
    
    trainer.train(
        human_data_path, 
        tag=args.ckpt_tag, 
        w_chamfer=args.w_chamfer,
        w_distance=args.w_distance,
        w_curvature=args.w_curvature,
        motion_delta=args.motion_delta,
        w_collision=args.w_collision,
        w_pinch=args.w_pinch,
        pinch_threshold=args.pinch_threshold,
        w_mcp1_fist_prior=args.w_mcp1_fist_prior,
        mcp1_fist_prior_top_fraction=args.mcp1_fist_prior_top_fraction,
        mcp1_fist_prior_target_alpha=args.mcp1_fist_prior_target_alpha,
        mcp1_fist_prior_mcp_weight=args.mcp1_fist_prior_mcp_weight,
        mcp1_fist_prior_pip_weight=args.mcp1_fist_prior_pip_weight,
        mcp1_fist_prior_dip_weight=args.mcp1_fist_prior_dip_weight,
        save_every=args.save_every,
        chamfer_target=args.chamfer_target,
        chamfer_target_path=args.chamfer_target_path,
        mold_path=args.mold_path,
        update_latest=not args.no_update_latest)
