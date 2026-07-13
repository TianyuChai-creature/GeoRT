# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
# sapien is used indirectly via HandKinematicModel (geort/env/hand.py).
# No direct sapien calls remain in trainer.py after SAPIEN 3 migration.
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from geort.utils.hand_utils import get_entity_by_name, get_active_joints, get_active_joint_indices
from geort.utils.path import get_human_data
from geort.utils.config_utils import get_config, parse_config_keypoint_info, save_json
from geort.model import FKModel, IKModel 
from geort.env.hand import HandKinematicModel
from geort.loss import chamfer_distance
from geort.formatter import HandFormatter
from geort.dataset import RobotKinematicsDataset, MultiPointDataset, FramePointDataset
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


def should_save_epoch_checkpoint(epoch, n_epoch, save_every=0):
    del n_epoch
    return save_every > 0 and (epoch + 1) % save_every == 0


def prepare_human_training_dataset(human_data_path, human_ids):
    human_points = np.load(Path(human_data_path))
    selected_points = np.array(
        [human_points[:, idx, :3] for idx in human_ids], dtype=np.float32
    ).transpose(1, 0, 2)
    return FramePointDataset(selected_points)


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

    def get_robot_pointcloud(self, keypoint_names):
        '''
            Utility getter function. Return the robot fingertip point cloud.
        '''
        kinematics_dataset = self.get_robot_kinematics_dataset()
        return kinematics_dataset.export_robot_pointcloud(keypoint_names)
        
    def get_robot_kinematics_dataset(self):
        '''
            Utility getter function. Return the robot kinematics dataset
        '''
        dataset_path = self.get_robot_kinematics_dataset_path(postfix=True)
        if not os.path.exists(dataset_path):
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
        return parse_config_keypoint_info(self.config)

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
        w_curvature = kwargs.get("w_curvature", 0.1)
        save_every = int(kwargs.get("save_every", 0) or 0)
        update_latest = bool(kwargs.get("update_latest", True))

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

        training_metadata = {
            "human_data_path": str(Path(human_data_path)),
            "epoch": int(n_epoch),
            "timestamp": datetime.now().isoformat(),
        }
        save_json(training_metadata, Path(save_dir) / "training_metadata.json")
        if update_latest:
            save_json(training_metadata, Path(last_save_dir) / "training_metadata.json")

        # Dataset.
        robot_keypoint_names = keypoint_info["link"]
        robot_points = self.get_robot_pointcloud(robot_keypoint_names)

        human_finger_idxes = keypoint_info["human_id"]
        tip_indices = keypoint_info["tip_indices"]
        for robot_keypoint_name, human_id in zip(robot_keypoint_names, human_finger_idxes):
            print(f"Robot Keypoint {robot_keypoint_name}: Human Id: {human_id}")

        point_dataset_human = prepare_human_training_dataset(
            human_data_path, human_finger_idxes
        )
        point_dataloader = DataLoader(
            point_dataset_human, batch_size=2048, shuffle=True
        )

        # Training / Optimization
        for epoch in range(n_epoch):
            for batch_idx, batch in enumerate(point_dataloader):
                direction_loss = 0

                point = batch.cuda() # [B, N, 3]
                joint = ik_model(point) # [B, DOF]
                embedded_point = fk_model(joint) # [B, N, 3]

                # [Curvature loss] -- Ensuring flatness.
                direction = F.normalize(torch.randn_like(point), dim=-1, p=2)
                scale = 0.002
                delta1 = direction * scale
                point_delta_1p = point + delta1 
                point_delta_1n = point - delta1 

                embedded_point_p = fk_model(ik_model(point_delta_1p))
                embedded_point_n = fk_model(ik_model(point_delta_1n))
                curvature_loss = (
                    (embedded_point_p + embedded_point_n - 2 * embedded_point)[:, tip_indices, :] ** 2
                ).mean()
                
                # [Chamfer loss]
                selected_idx = np.random.randint(0, robot_points.shape[1], 2048) 
                target = torch.from_numpy(robot_points[:, selected_idx, :]).permute(1, 0, 2).float().cuda()
                
                chamfer_by_tip = [
                    chamfer_distance(
                        embedded_point[:, i, :].unsqueeze(0),
                        target[:, i, :].unsqueeze(0),
                    )
                    for i in tip_indices
                ]
                chamfer_loss = torch.stack(chamfer_by_tip).mean()

                # [Direction Loss]
                direction = F.normalize(torch.randn_like(point), dim=-1, p=2)
                scale = 0.001 + torch.rand(point.size(0)).cuda().unsqueeze(-1).unsqueeze(-1) * 0.01
                point_delta = point + direction * scale 

                joint_delta = ik_model(point_delta)
                embedded_point_delta = fk_model(joint_delta)

                d1 = point_delta - point
                d2 = embedded_point_delta - embedded_point
                direction_by_keypoint = (
                    F.normalize(d1, dim=-1, p=2, eps=1e-5)
                    * F.normalize(d2, dim=-1, p=2, eps=1e-5)
                ).sum(-1)
                direction_loss = -direction_by_keypoint[:, tip_indices].mean()

                loss = (
                    direction_loss
                    + chamfer_loss * w_chamfer
                    + curvature_loss * w_curvature
                )

                ik_optim.zero_grad()
                loss.backward()
                ik_optim.step()

                if batch_idx % 50 == 0:
                    print(
                        f"Epoch {epoch} | Losses"
                        f" - Direction: {format_loss(direction_loss.item())}"
                        f" - Chamfer: {format_loss(chamfer_loss.item())}"
                        f" - Curvature: {format_loss(curvature_loss.item())}"
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
    parser.add_argument('--w_curvature', type=float, default=0.1)
    parser.add_argument('--save_every', type=int, default=0, help='Save epoch_N.pth every N epochs; 0 keeps only last.pth.')
    parser.add_argument('--no_update_latest', action='store_true', help='Do not update checkpoint/<hand>_last.')

    args = parser.parse_args()

    config = get_config(args.hand)
    trainer = GeoRTTrainer(config)

    human_data_path = get_human_data(args.human_data)
    print("Training with human data:", human_data_path.as_posix())
    
    trainer.train(
        human_data_path, 
        tag=args.ckpt_tag, 
        w_chamfer=args.w_chamfer, 
        w_curvature=args.w_curvature, 
        save_every=args.save_every,
        update_latest=not args.no_update_latest)
