# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

def get_finger_fk(n_joint=4, hidden=128):
    return nn.Sequential(
        nn.Linear(n_joint, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, 3)
    ) 

def get_finger_ik(n_joint=4, hidden=128, n_input=3):
    return nn.Sequential(
        nn.Linear(n_input, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, hidden), 
        nn.LeakyReLU(), 
        nn.BatchNorm1d(hidden),
        nn.Linear(hidden, n_joint),
        nn.Tanh()   # Normalize.
    ) 

class FKModel(nn.Module):
    def __init__(self, keypoint_joints):
        # keypoint_joints: a list of list.
        # keypoint[i] is the indices of joints that drive the i-th keypoint.
        # Example: For allegro, [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

        super().__init__()
        num_fingers = len(keypoint_joints)
        
        self.nets = []
        self.n_total_joint = 0

        for joint in keypoint_joints:
            net = get_finger_fk(n_joint=len(joint))
            self.nets.append(net)
            self.n_total_joint += len(joint)

        self.nets = nn.ModuleList(self.nets)

        self.keypoint_joints = keypoint_joints

    def forward(self, joint):
        # x: [B, DOF], joint values. normalized to [-1, 1]. 
        # out:   [B, N, 3], sequence of keypoint.
        keypoints = []
        for i, net in enumerate(self.nets):
            joint_ids = self.keypoint_joints[i]
            keypoint = net(joint[:, joint_ids])
            keypoints.append(keypoint)

        return torch.stack(keypoints, dim=1)

    
class IKModel(nn.Module):
    def __init__(self, keypoint_joints=None, finger_groups=None, n_total_joint=None):
        # finger_groups: one entry per finger. Each entry contains the keypoints
        # used as IK input and the non-overlapping joint block written by that finger.
        # Example: {"keypoint_indices": [pip_idx, tip_idx], "joint_indices": [0, 1, 2, 3]}

        super().__init__()

        if finger_groups is None:
            if keypoint_joints is None:
                raise ValueError("Either keypoint_joints or finger_groups must be provided")
            finger_groups = [
                {
                    "finger": f"keypoint_{idx}",
                    "keypoint_indices": [idx],
                    "joint_indices": list(joint_indices),
                }
                for idx, joint_indices in enumerate(keypoint_joints)
            ]

        self.finger_groups = [
            {
                "finger": group.get("finger", f"finger_{idx}"),
                "keypoint_indices": list(group["keypoint_indices"]),
                "joint_indices": list(group["joint_indices"]),
            }
            for idx, group in enumerate(finger_groups)
        ]

        if n_total_joint is None:
            max_joint_idx = max(
                joint_idx
                for group in self.finger_groups
                for joint_idx in group["joint_indices"]
            )
            n_total_joint = max_joint_idx + 1

        self.n_total_joint = n_total_joint
        self.input_dims = [3 * len(group["keypoint_indices"]) for group in self.finger_groups]
        self.output_dims = [len(group["joint_indices"]) for group in self.finger_groups]
        self.nets = nn.ModuleList([
            get_finger_ik(n_joint=output_dim, n_input=input_dim)
            for input_dim, output_dim in zip(self.input_dims, self.output_dims)
        ])
        self.keypoint_joints = [group["joint_indices"] for group in self.finger_groups]

    def forward(self, x):
        # x:   [B, N, 3], sequence of keypoints.
        # out: [B, DOF], joint values. normalized to [-1, 1].
        batch_size = x.size(0)
        out = torch.zeros((batch_size, self.n_total_joint), device=x.device, dtype=x.dtype)
        for group, net in zip(self.finger_groups, self.nets):
            keypoint_input = x[:, group["keypoint_indices"], :].reshape(batch_size, -1)
            joint = net(keypoint_input)
            out[:, group["joint_indices"]] = joint
        return out 
