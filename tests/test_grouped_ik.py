import json
import unittest
from pathlib import Path

import torch

from geort.model import IKModel
from geort.utils.config_utils import parse_config_keypoint_info


class GroupedIKTest(unittest.TestCase):
    def test_custom_config_exposes_tip_pip_metadata_and_groups(self):
        config = json.loads(Path("geort/config/custom_right.json").read_text())

        info = parse_config_keypoint_info(config)

        self.assertEqual(len(info["name"]), 10)
        self.assertEqual(len(info["tip_indices"]), 5)
        self.assertEqual(len(info["pip_indices"]), 5)
        self.assertEqual([info["name"][i] for i in info["tip_indices"]], [
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
        ])
        self.assertEqual([info["name"][i] for i in info["pip_indices"]], [
            "thumb_pip",
            "index_pip",
            "middle_pip",
            "ring_pip",
            "pinky_pip",
        ])
        self.assertTrue(all(info["weight"][i] == 1.0 for i in info["tip_indices"]))
        self.assertTrue(all(info["weight"][i] < 1.0 for i in info["pip_indices"]))

        self.assertEqual(len(info["finger_groups"]), 5)
        self.assertEqual(info["finger_groups"][0]["finger"], "thumb")
        self.assertEqual(info["finger_groups"][0]["keypoint_indices"], [0, 1])
        self.assertEqual(info["finger_groups"][0]["joint_indices"], [0, 1, 2, 3])

    def test_legacy_tip_only_config_defaults_to_tip_groups(self):
        config = json.loads(Path("geort/config/allegro_right.json").read_text())

        info = parse_config_keypoint_info(config)

        self.assertEqual(info["type"], ["tip", "tip", "tip", "tip"])
        self.assertEqual(info["weight"], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(len(info["finger_groups"]), 4)
        self.assertTrue(all(len(group["keypoint_indices"]) == 1 for group in info["finger_groups"]))
        self.assertTrue(all(len(group["joint_indices"]) == 4 for group in info["finger_groups"]))
        self.assertEqual(info["segment_pairs"], [])

    def test_pinch_pairs_only_include_thumb_tip_to_other_tips(self):
        config = json.loads(Path("geort/config/custom_right.json").read_text())

        info = parse_config_keypoint_info(config)

        pair_names = [(info["name"][i], info["name"][j]) for i, j in info["pinch_pairs"]]
        self.assertEqual(pair_names, [
            ("thumb_tip", "index_tip"),
            ("thumb_tip", "middle_tip"),
            ("thumb_tip", "ring_tip"),
            ("thumb_tip", "pinky_tip"),
        ])

    def test_segment_pairs_connect_pip_to_tip_per_custom_finger(self):
        config = json.loads(Path("geort/config/custom_right.json").read_text())

        info = parse_config_keypoint_info(config)

        pair_names = [(info["name"][pip], info["name"][tip]) for pip, tip in info["segment_pairs"]]
        self.assertEqual(pair_names, [
            ("thumb_pip", "thumb_tip"),
            ("index_pip", "index_tip"),
            ("middle_pip", "middle_tip"),
            ("ring_pip", "ring_tip"),
            ("pinky_pip", "pinky_tip"),
        ])

    def test_grouped_ik_uses_one_network_per_finger(self):
        finger_groups = [
            {"finger": "finger_a", "keypoint_indices": [0, 1], "joint_indices": [0, 1, 2, 3]},
            {"finger": "finger_b", "keypoint_indices": [2, 3], "joint_indices": [4, 5, 6, 7]},
        ]
        model = IKModel(finger_groups=finger_groups, n_total_joint=8)
        model.eval()

        self.assertEqual(len(model.nets), 2)
        self.assertEqual(model.n_total_joint, 8)
        self.assertEqual(model.input_dims, [6, 6])
        self.assertEqual(model.output_dims, [4, 4])

        keypoints = torch.randn(5, 4, 3)
        joints = model(keypoints)

        self.assertEqual(tuple(joints.shape), (5, 8))


if __name__ == "__main__":
    unittest.main()
