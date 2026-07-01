# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sapien
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from geort.utils.config_utils import get_config, save_json
from geort.utils.hand_utils import get_entity_by_name, get_active_joints, get_active_joint_indices
from datetime import datetime
from tqdm import tqdm 
import os
from pathlib import Path 
import math

class HandKinematicModel:
    def __init__(self, 
                 scene=None, 
                 render=False, 
                 hand=None, 
                 hand_urdf='', 
                 n_hand_dof=16, 
                 base_link='base_link', 
                 joint_names=[],
                 # Ideally, these two guys (PD controller args) shouldn't be here. 
                 # -- There should be a controller class. I leave them here for code simplicity (maybe truth: or because I am lazy).
                 # If you see your hand model doing something weird (in the simulation viewer below), tune them.
                 kp=400.0, 
                 kd=10):
        
        if scene is None:
            physx_config = sapien.physx.PhysxSceneConfig()
            physx_config.enable_friction_every_iteration = True
            sapien.physx.set_scene_config(physx_config)
            scene = sapien.Scene()
            if render:
                print("Enable Render Mode.")

        self.scene = scene
        self.render = render

        if hand is not None:
            self.hand = hand

        else:
            loader = scene.create_urdf_loader()
            loader.set_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
            loader.set_patch_radius(0.02)
            loader.set_min_patch_radius(0.02)
            self.hand = loader.load(hand_urdf)
            self.hand.set_root_pose(sapien.Pose([0, 0, 0.35], [0.695, 0, -0.718, 0]))

        self.pmodel = self.hand.create_pinocchio_model()

        # Setup hand base link.
        self.base_link = get_entity_by_name(self.hand.get_links(), base_link)
        self.base_link_idx = self.hand.get_links().index(self.base_link)

        # Setup controllable hand dofs. Any active joint omitted from joint_names
        # remains fixed at qpos=0 in the full simulation qpos vector.
        self.all_joints = get_active_joints(self.hand, joint_names)
        all_limits = [joint.get_limits() for joint in self.all_joints]

        self.joint_names = joint_names
        self.n_sim_dof = len(self.hand.get_active_joints())
        self.fixed_qpos = np.zeros(self.n_sim_dof)
        self.user_idx_to_sim_idx = np.array(get_active_joint_indices(self.hand, joint_names), dtype=int)
        self.active_joints = self.hand.get_active_joints()
        controlled_sim_indices = set(self.user_idx_to_sim_idx.tolist())
        self.fixed_joint_indices = [
            idx for idx in range(self.n_sim_dof)
            if idx not in controlled_sim_indices
        ]
        self.fixed_joints = [self.active_joints[idx] for idx in self.fixed_joint_indices]
        print("User-to-Sim Joint", self.user_idx_to_sim_idx.tolist())

        self.joint_lower_limit = np.array([l[0][0] for l in all_limits])  # this is in user specified "joint_name" order
        self.joint_upper_limit = np.array([l[0][1] for l in all_limits])  # this is in user specified "joint_name" order
        print(self.joint_lower_limit, self.joint_upper_limit)

        init_qpos = self.convert_user_order_to_sim_order((self.joint_lower_limit + self.joint_upper_limit) / 2)
        self.hand.set_qpos(init_qpos)
        self.hand.set_qvel(0.0 * init_qpos)
        self.qpos_target = init_qpos

        for i, joint in enumerate(self.all_joints):
            print(i, self.joint_names[i], joint, self.joint_lower_limit[i], self.joint_upper_limit[i])
            # SAPIEN 3: positional args → keyword args (stiffness, damping)
            joint.set_drive_property(stiffness=kp, damping=kd, force_limit=10)

        for joint in self.fixed_joints:
            joint.set_drive_property(stiffness=kp, damping=kd, force_limit=10)
            joint.set_drive_target(0.0)


    def __del__(self):
        if hasattr(self, "scene"):
            del self.scene

    def get_n_dof(self):
        '''
            number of dof.
        '''
        return len(self.joint_lower_limit)

    def get_joint_limit(self):
        '''
            Get the hand joint limit.
        '''
        return self.joint_lower_limit, self.joint_upper_limit

    def initialize_keypoint(self, keypoint_link_names, keypoint_offsets):
        '''
            Setup keypoints to track.
        '''
        keypoint_links = [get_entity_by_name(self.hand.get_links(), link) for link in keypoint_link_names]
        print(keypoint_links)

        keypoint_links_id_dict = {link_name: (self.hand.get_links().index(keypoint_links[i]), i) for i, link_name in enumerate(keypoint_link_names)}
        self.keypoint_links = keypoint_links
        self.keypoint_links_id_dict = keypoint_links_id_dict
        self.keypoint_offsets = np.array(keypoint_offsets)

    def convert_user_order_to_sim_order(self, qpos):
        sim_qpos = self.fixed_qpos.copy()
        sim_qpos[self.user_idx_to_sim_idx] = np.asarray(qpos)
        return sim_qpos

    def keypoint_from_qpos(self, qpos, ret_vec=False):
        '''
            Get keypoints from hand qpos. qpos is specified using the user order.
        '''
        qpos = self.convert_user_order_to_sim_order(qpos)
        self.pmodel.compute_forward_kinematics(qpos)
        base_pose = self.pmodel.get_link_pose(self.base_link_idx)

        result = {} 
        vec_result = []

        for m, (link_idx, i) in self.keypoint_links_id_dict.items():
            pose = self.pmodel.get_link_pose(link_idx)
            new_pose = sapien.Pose(p=pose.p + (pose.to_transformation_matrix()[:3, :3] @ self.keypoint_offsets[i].reshape(3, 1)).reshape(-1), q=pose.q)

            x = (base_pose.inv() * new_pose).p # convert to hand base frame.
            vec_result.append(x)
            result[m] = x

        if ret_vec:
            return np.array(vec_result)
        return result

    @staticmethod
    def build_from_config(config, **kwargs):
        '''
            Build a kinematic model from user config.
        '''
        render = kwargs.get("render", False)
        urdf_path = config["urdf_path"]
        n_hand_dof = len(config["joint_order"])
        base_link = config["base_link"]
        joint_order = config["joint_order"]

        model = HandKinematicModel(hand_urdf=urdf_path, render=render, n_hand_dof=n_hand_dof,base_link=base_link, joint_names=joint_order)
        return model 

    def get_viewer_env(self):
        return HandViewerEnv(self)

    def get_scene(self):
        return self.scene

    def get_render(self):
        """Return whether render mode is enabled."""
        return self.render

    def set_qpos_target(self, qpos):
        '''
            This function is only used during visualization
        '''
        qpos = np.clip(qpos, self.joint_lower_limit + 1e-3, self.joint_upper_limit - 1e-3)
        self.qpos_target = self.convert_user_order_to_sim_order(qpos)

        for joint in self.fixed_joints:
            joint.set_drive_target(0.0)

        for i, joint in enumerate(self.all_joints):
            joint.set_drive_target(qpos[i])

class HandViewerEnv:
    def __init__(self, model):
        scene = model.get_scene()
        scene.set_timestep(1 / 100.0)
        scene.set_ambient_light([0.78, 0.78, 0.82])
        scene.add_directional_light([0, 1, -1], [1.1, 1.05, 0.95], shadow=False)
        scene.add_directional_light([-1, -0.5, -1], [0.55, 0.62, 0.75], shadow=False)
        scene.add_directional_light([1, -1, -0.5], [0.35, 0.32, 0.28], shadow=False)
        ground_material = sapien.render.RenderMaterial()
        ground_material.set_base_color([0.72, 0.74, 0.76, 1.0])
        ground_material.set_roughness(0.45)
        ground_material.set_specular(0.15)
        scene.add_ground(altitude=0, render_material=ground_material)

        # SAPIEN 3: viewer is created from scene directly; no renderer arg needed.
        viewer = scene.create_viewer()
        viewer.set_camera_xyz(x=0.1550926, y=-0.1623763, z=0.7064089)
        # SAPIEN 3: set_camera_rpy replaces set_camera_rotation (quaternion → euler r/p/y)
        viewer.set_camera_rpy(r=0, p=-0.3, y=0)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        self.model = model
        self.scene = scene
        self.viewer = viewer

    def update(self):
        if self.viewer.closed:
            return False

        self.scene.step()
        self.scene.update_render()
        self.viewer.render()
        return not self.viewer.closed

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand', type=str, default='allegro_right')

    args = parser.parse_args()

    # Load Hand Model
    config = get_config(args.hand)
    model = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = model.get_viewer_env()
   
    # Control Loop
    n_dof = model.get_n_dof()
    dof_lower, dof_upper = model.get_joint_limit()

    steps = 0
    while True:
        if not viewer_env.update():
            break

        steps += 1
        if steps % 30 == 0:
            targets = np.random.uniform(0, 1, n_dof) * (dof_upper - dof_lower - 1e-7) + dof_lower + 1e-7
            model.set_qpos_target(targets)
