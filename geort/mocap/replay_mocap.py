# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from geort.utils.path import get_human_data


class ReplayMocap:
    def __init__(self, human_data):
        self.human_data_path = get_human_data(human_data)
        self.human_points = np.load(self.human_data_path)  # [T, N, 3]
        self.t = 0
        self.T = len(self.human_points)
        self.replay_indices = np.arange(self.T)

    def get(self):
        frame_idx = self.replay_indices[self.t]
        result = self.human_points[frame_idx]
        self.t = (self.t + 1) % self.T
        if self.t == 0:
            print("[ReplayMocap] I am returning to start!")
        return {"result": result, "status": "recording"}
