import tempfile
import unittest
from pathlib import Path

import numpy as np

from geort.mocap.hts_balance import (
    build_stage2_report,
    save_balanced_dataset,
    select_balanced_frame_indices,
)


class HTSBalanceTest(unittest.TestCase):
    def test_select_balanced_frame_indices_keeps_at_most_k_per_single_finger_voxel_source(self):
        frames = np.zeros((6, 21, 3), dtype=np.float32)
        # Index PIP/TIP stay in one voxel for first four frames, then move to two new voxels.
        frames[:4, 6, :] = [0.0, 0.0, 0.0]
        frames[:4, 8, :] = [0.0, 0.0, 0.0]
        frames[4, 6, :] = [0.02, 0.0, 0.0]
        frames[4, 8, :] = [0.02, 0.0, 0.0]
        frames[5, 6, :] = [0.04, 0.0, 0.0]
        frames[5, 8, :] = [0.04, 0.0, 0.0]

        selected, report = select_balanced_frame_indices(
            frames,
            voxel_size=0.01,
            max_per_voxel=2,
            fingers=("index",),
        )

        self.assertEqual(selected.tolist(), [0, 1, 4, 5])
        self.assertEqual(report["fingers"]["index"]["raw_frames"], 6)
        self.assertEqual(report["fingers"]["index"]["selected_by_quota"], 4)


    def test_select_balanced_frame_indices_caps_effective_density_across_fingers(self):
        frames = np.zeros((5, 21, 3), dtype=np.float32)
        # First three frames share the same thumb and index voxels; strict cap should keep only two.
        frames[:3, 2, :] = [0.0, 0.0, 0.0]
        frames[:3, 4, :] = [0.0, 0.0, 0.0]
        frames[:3, 6, :] = [0.0, 0.0, 0.0]
        frames[:3, 8, :] = [0.0, 0.0, 0.0]
        frames[3:, 2, :] = [[0.02, 0, 0], [0.04, 0, 0]]
        frames[3:, 4, :] = [[0.02, 0, 0], [0.04, 0, 0]]
        frames[3:, 6, :] = [[0.02, 0, 0], [0.04, 0, 0]]
        frames[3:, 8, :] = [[0.02, 0, 0], [0.04, 0, 0]]

        selected, _ = select_balanced_frame_indices(
            frames,
            voxel_size=0.01,
            max_per_voxel=2,
            fingers=("thumb", "index"),
        )
        report = build_stage2_report(frames, selected, voxel_size=0.01, max_per_voxel=2)

        self.assertEqual(selected.tolist(), [0, 1, 3, 4])
        self.assertLessEqual(report["fingers"]["thumb"]["effective_max_samples_in_voxel"], 2)
        self.assertLessEqual(report["fingers"]["index"]["effective_max_samples_in_voxel"], 2)

    def test_select_balanced_frame_indices_preserves_tip_contact_frames(self):
        frames = np.full((4, 21, 3), 0.5, dtype=np.float32)
        frames[:, 4, :] = [0.0, 0.0, 0.0]
        frames[:, 8, :] = [0.2, 0.0, 0.0]
        frames[:, 6, :] = [0.2, 0.0, 0.0]

        # All index PIP/TIP features share one coarse voxel, so the base quota
        # keeps only frame 0. Frame 2 must be added back because it is a
        # thumb-index tip contact.
        frames[2, 8, :] = [0.01, 0.0, 0.0]

        selected, report = select_balanced_frame_indices(
            frames,
            voxel_size=1.0,
            max_per_voxel=1,
            fingers=("index",),
            preserve_contact_pairs="all",
            contact_threshold=0.025,
        )

        self.assertEqual(selected.tolist(), [0, 2])
        contact = report["contact_preserve"]["pairs"]["thumb_tip__index_tip"]
        self.assertEqual(contact["raw_contact_count"], 1)
        self.assertEqual(contact["baseline_contact_count"], 0)
        self.assertEqual(contact["preserved_count"], 1)
        self.assertEqual(contact["final_contact_count"], 1)

    def test_select_balanced_frame_indices_preserves_tip_contacts_not_pip_contacts(self):
        frames = np.full((4, 21, 3), 0.5, dtype=np.float32)
        frames[:, 4, :] = [0.0, 0.0, 0.0]
        frames[:, 8, :] = [0.2, 0.0, 0.0]
        frames[:, 6, :] = [0.2, 0.0, 0.0]

        # Frame 2 has index PIP close to thumb tip but index TIP remains far.
        # It should not be preserved because Stage 2 contact preservation is
        # intentionally tip-tip only.
        frames[2, 6, :] = [0.01, 0.0, 0.0]

        selected, report = select_balanced_frame_indices(
            frames,
            voxel_size=1.0,
            max_per_voxel=1,
            fingers=("index",),
            preserve_contact_pairs="all",
            contact_threshold=0.025,
        )

        self.assertEqual(selected.tolist(), [0])
        contact = report["contact_preserve"]["pairs"]["thumb_tip__index_tip"]
        self.assertEqual(contact["raw_contact_count"], 0)
        self.assertEqual(contact["preserved_count"], 0)

    def test_save_balanced_dataset_writes_selected_full_frames(self):
        frames = np.arange(5 * 21 * 3, dtype=np.float32).reshape(5, 21, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = save_balanced_dataset(frames, np.array([1, 3]), Path(tmpdir) / "balanced.npy")
            loaded = np.load(out)

        self.assertEqual(loaded.shape, (2, 21, 3))
        np.testing.assert_allclose(loaded[0], frames[1])
        np.testing.assert_allclose(loaded[1], frames[3])

    def test_build_stage2_report_includes_effective_density(self):
        frames = np.zeros((3, 21, 3), dtype=np.float32)
        selected = np.array([0, 2])

        report = build_stage2_report(frames, selected, voxel_size=0.01, max_per_voxel=2)

        self.assertEqual(report["stage"], 2)
        self.assertEqual(report["raw_frames"], 3)
        self.assertEqual(report["balanced_frames"], 2)
        self.assertIn("thumb", report["fingers"])
        self.assertIn("effective_max_samples_in_voxel", report["fingers"]["thumb"])


if __name__ == "__main__":
    unittest.main()
