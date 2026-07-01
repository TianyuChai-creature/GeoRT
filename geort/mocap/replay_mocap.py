# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
from geort.utils.path import get_human_data
from geort.dataset_manifest import maybe_load_dataset_manifest


class ReplayMocap:
    def __init__(self, human_data, use_weights=False, seed=0):
        manifest = maybe_load_dataset_manifest(human_data)
        if manifest is not None:
            human_data_path = manifest.data_path
            explicit_weights = np.asarray(manifest.weights, dtype=np.float64) if manifest.weights is not None else None
            explicit_weights_path = manifest.weights_path
        else:
            human_data_path = get_human_data(human_data)
            explicit_weights = None
            explicit_weights_path = None

        self.human_data_path = Path(human_data_path)
        self.human_points = np.load(human_data_path) # [T, N, 3]
        self.t = 0
        self.T = len(self.human_points)
        self.weights_path = explicit_weights_path if use_weights else None
        self.replay_indices = np.arange(self.T)

        weights = explicit_weights if use_weights else None
        if weights is None and self.weights_path is not None:
            weights = np.load(self.weights_path).astype(np.float64)

        if weights is not None:
            if weights.shape != (self.T,):
                raise ValueError(
                    f"Replay weights shape {weights.shape} does not match frame count {self.T}"
                )
            prob = weights / weights.sum()
            rng = np.random.default_rng(seed)
            self.replay_indices = rng.choice(self.T, size=self.T, replace=True, p=prob)
            source = self.weights_path if self.weights_path is not None else "manifest inline weights"
            print(f"[ReplayMocap] Using weighted replay from {source}")
        return

    def get(self):
        frame_idx = self.replay_indices[self.t]
        result = self.human_points[frame_idx]
        self.t = (self.t + 1) % self.T
        if self.t == 0:
            print("[ReplayMocap] I am returning to start!")
        return {"result": result, "status": "recording"}
