# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import random
import numpy as np
import open3d as o3d

def upsample_array(x, K=50000):
    n = x.shape[0]
    ind = np.random.randint(0, n - 1, K)
    return x[ind]


class MultiPointDataset:
    def __init__(self, points):
        self.points = np.array(points) # [Num_Fingers, Num_Samples, 3]

    @staticmethod
    def from_points(points, n, resample_to=50000, resample_resolution=0.001):  
        '''
            This is the actual initialization function. 
        '''
        num_fingers = points.shape[0]
        all_points = []

        # Resampling to reduce spatial imbalance.
        for finger_id in range(num_fingers):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[finger_id])
            downpcd = pcd.voxel_down_sample(voxel_size=resample_resolution)
            resampled_points = np.asarray(downpcd.points)
            all_points.append(upsample_array(resampled_points, K=resample_to))

        return MultiPointDataset(np.array(all_points).astype(np.float32))

    def __len__(self):
        return self.points.shape[1]

    def __getitem__(self, idx):
        return self.points[:, idx]



class FramePointDataset:
    def __init__(self, points, frame_fields=None):
        self.points = np.asarray(points, dtype=np.float32)
        if self.points.ndim != 3 or self.points.shape[-1] != 3:
            raise ValueError(f"Expected frame points with shape [N, K, 3], got {self.points.shape}")
        self.frame_fields = {}
        if frame_fields:
            for name, values in frame_fields.items():
                values = np.asarray(values)
                if values.shape[0] != self.points.shape[0]:
                    raise ValueError(
                        f"Frame field {name!r} length {values.shape[0]} does not match frame count {self.points.shape[0]}"
                    )
                self.frame_fields[name] = values

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        if not self.frame_fields:
            return self.points[idx]
        item = {"point": self.points[idx]}
        for name, values in self.frame_fields.items():
            item[name] = values[idx]
        return item

class RobotKinematicsDataset:
    def __init__(self, qpos_keypoint_file, keypoint_names):
        np_array = np.load(qpos_keypoint_file,  allow_pickle=True)
        self.qpos = np_array["qpos"]
        self.keypoints = np_array["keypoint"].item()
        self.link_rotations = (
            np_array["link_rotation"].item() if "link_rotation" in np_array.files else None
        )
        self.keypoint_names = keypoint_names
        print("Keypoint Names", self.keypoint_names)
        self.n = len(self.qpos)
        return
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        qpos = self.qpos[idx]

        keypoint_data = []
        for name in self.keypoint_names:
            keypoint_data.append(self.keypoints[name][idx][:3])

        return {"qpos": self.qpos[idx].astype(np.float32), "keypoint": np.array(keypoint_data).astype(np.float32)}

    def export_robot_pointcloud(self, keypoint_names):
        all_keypoint_data = []
        for keypoint_name in keypoint_names:
            all_keypoint_data.append(self.keypoints[keypoint_name])
        return np.array(all_keypoint_data)

    def export_robot_link_rotations(self, keypoint_names):
        """Return [K,N,3,3] link rotations from a new local-frame cloud.

        Legacy point clouds intentionally have no identity fallback: local
        motion must fail loudly rather than silently change the contract.
        """
        if self.link_rotations is None:
            raise ValueError(
                "robot target cloud lacks required link_rotation field; "
                "regenerate this target cloud for --motion_frame local"
            )
        missing = [name for name in keypoint_names if name not in self.link_rotations]
        if missing:
            raise ValueError(f"robot target cloud link_rotation missing keys: {missing}")
        values = np.asarray([self.link_rotations[name] for name in keypoint_names], dtype=np.float32)
        if values.ndim != 4 or values.shape[-2:] != (3, 3):
            raise ValueError(f"robot target cloud link_rotation shape invalid: {values.shape}")
        return values

if __name__ == '__main__':
    dataset = RobotJointKeypointDataset("../data/allegro_native.npz",["link_3.0_tip", "link_3.0_tip"])
    print(dataset[0])