import tempfile
import unittest
from pathlib import Path

import numpy as np

from geort.mocap.visualize_tip_workspace import (
    build_layered_tip_workspace_figures,
    build_workspace_overlap_report,
    compute_alpha_surface_mesh,
    compute_voxel_overlap,
    default_overlap_pairs,
    extract_dataset_tip_points,
    load_aa_limit_overrides_from_search_report,
    summarize_workspace_alignment,
    write_layered_html,
)


class TipWorkspaceVisualizationTest(unittest.TestCase):
    def setUp(self):
        self.keypoint_info = {
            "finger": ["thumb", "thumb", "index", "index"],
            "type": ["pip", "tip", "pip", "tip"],
            "human_id": [3, 4, 6, 8],
            "name": ["thumb_pip", "thumb_tip", "index_pip", "index_tip"],
        }

    def test_extract_dataset_tip_points_uses_only_tip_keypoints(self):
        frames = np.zeros((2, 21, 3), dtype=np.float32)
        frames[:, 4, :] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        frames[:, 8, :] = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        frames[:, 3, :] = 99.0
        frames[:, 6, :] = 88.0

        tips = extract_dataset_tip_points(frames, self.keypoint_info)

        self.assertEqual(sorted(tips), ["index", "thumb"])
        np.testing.assert_allclose(tips["thumb"], frames[:, 4, :])
        np.testing.assert_allclose(tips["index"], frames[:, 8, :])

    def test_layered_figures_overlay_single_finger_dataset_and_urdf_traces(self):
        dataset_tips = {
            "thumb": np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float32),
            "index": np.array([[0.0, 0.02, 0.0]], dtype=np.float32),
        }
        urdf_tips = {
            "thumb": np.array([[0.0, 0.0, 0.001], [0.02, 0.0, 0.0]], dtype=np.float32),
            "index": np.array([[0.0, 0.025, 0.0]], dtype=np.float32),
        }

        figures = build_layered_tip_workspace_figures(dataset_tips, urdf_tips)

        self.assertIn("single_thumb", figures)
        thumb_trace_names = [trace.name for trace in figures["single_thumb"].data]
        self.assertEqual(thumb_trace_names, ["dataset_thumb_tip", "urdf_thumb_tip"])
        self.assertIn("dataset_all", figures)
        self.assertIn("urdf_all", figures)
        self.assertIn("overview_all", figures)
        self.assertEqual(len(figures["dataset_all"].data), 2)
        self.assertEqual(len(figures["urdf_all"].data), 2)
        self.assertEqual(len(figures["overview_all"].data), 4)


    def test_layered_figures_can_include_hidden_alpha_surface_traces(self):
        dataset_tips = {
            "thumb": np.array(
                [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.03]],
                dtype=np.float32,
            ),
        }
        urdf_tips = {
            "thumb": np.array(
                [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]],
                dtype=np.float32,
            ),
        }

        figures = build_layered_tip_workspace_figures(
            dataset_tips,
            urdf_tips,
            include_alpha_surface=True,
            alpha=0.08,
        )

        trace_names = [trace.name for trace in figures["single_thumb"].data]
        self.assertEqual(
            trace_names,
            [
                "dataset_thumb_tip",
                "dataset_thumb_tip_alpha",
                "urdf_thumb_tip",
                "urdf_thumb_tip_alpha",
            ],
        )
        self.assertTrue(figures["single_thumb"].data[0].visible is None)
        self.assertEqual(figures["single_thumb"].data[1].visible, False)
        button_labels = [button["label"] for button in figures["single_thumb"].layout.updatemenus[0].buttons]
        self.assertEqual(button_labels, ["Points only", "Alpha only", "Points + Alpha"])

    def test_compute_alpha_surface_mesh_returns_triangle_indices(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.03]],
            dtype=np.float32,
        )

        mesh = compute_alpha_surface_mesh(points, alpha=0.08)

        self.assertIsNotNone(mesh)
        self.assertEqual(mesh["vertices"].shape[1], 3)
        self.assertEqual(mesh["triangles"].shape[1], 3)


    def test_compute_voxel_overlap_reports_asymmetric_ratios_and_iou(self):
        finger_a = np.array(
            [[0.001, 0.001, 0.001], [0.011, 0.001, 0.001], [0.021, 0.001, 0.001]],
            dtype=np.float32,
        )
        finger_b = np.array(
            [[0.001, 0.001, 0.001], [0.011, 0.001, 0.001]],
            dtype=np.float32,
        )

        overlap = compute_voxel_overlap(finger_a, finger_b, voxel_size=0.01)

        self.assertEqual(overlap["intersection_voxels"], 2)
        self.assertEqual(overlap["a_voxels"], 3)
        self.assertEqual(overlap["b_voxels"], 2)
        self.assertAlmostEqual(overlap["overlap_a_ratio"], 2 / 3)
        self.assertAlmostEqual(overlap["overlap_b_ratio"], 1.0)
        self.assertAlmostEqual(overlap["iou"], 2 / 3)

    def test_default_overlap_pairs_use_thumb_against_all_and_adjacent_fingers(self):
        self.assertEqual(
            default_overlap_pairs(["thumb", "index", "middle", "ring", "pinky"]),
            [
                ("thumb", "index"),
                ("thumb", "middle"),
                ("thumb", "ring"),
                ("thumb", "pinky"),
                ("index", "middle"),
                ("middle", "ring"),
                ("ring", "pinky"),
            ],
        )

    def test_build_workspace_overlap_report_contains_dataset_and_urdf_sections(self):
        dataset_tips = {
            "index": np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float32),
            "middle": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        }
        urdf_tips = {
            "index": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "middle": np.array([[0.02, 0.0, 0.0]], dtype=np.float32),
        }

        report = build_workspace_overlap_report(dataset_tips, urdf_tips, voxel_size=0.01)

        self.assertEqual(report["voxel_size"], 0.01)
        self.assertEqual(report["pairs"], [["index", "middle"]])
        self.assertEqual(report["dataset"]["index__middle"]["intersection_voxels"], 1)
        self.assertEqual(report["urdf"]["index__middle"]["intersection_voxels"], 0)

    def test_build_workspace_overlap_report_can_include_urdf_baseline_delta(self):
        dataset_tips = {
            "index": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "middle": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        }
        urdf_tips = {
            "index": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "middle": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        }
        urdf_baseline_tips = {
            "index": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "middle": np.array([[0.02, 0.0, 0.0]], dtype=np.float32),
        }

        report = build_workspace_overlap_report(
            dataset_tips,
            urdf_tips,
            voxel_size=0.01,
            urdf_baseline_tips=urdf_baseline_tips,
        )

        pair = "index__middle"
        self.assertEqual(report["urdf_baseline"][pair]["intersection_voxels"], 0)
        self.assertEqual(report["urdf"][pair]["intersection_voxels"], 1)
        self.assertEqual(report["urdf_delta_from_baseline"][pair]["intersection_voxels"], 1)
        self.assertGreater(report["urdf_delta_from_baseline"][pair]["iou"], 0.0)


    def test_write_layered_html_embeds_overlap_summary_table(self):
        dataset_tips = {
            "index": np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float32),
            "middle": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        }
        urdf_tips = {
            "index": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "middle": np.array([[0.02, 0.0, 0.0]], dtype=np.float32),
        }
        figures = build_layered_tip_workspace_figures(dataset_tips, urdf_tips)
        overlap = build_workspace_overlap_report(dataset_tips, urdf_tips, voxel_size=0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = write_layered_html(figures, Path(tmpdir) / "workspace.html", overlap_report=overlap)
            html = html_path.read_text()

        self.assertIn("Workspace Overlap Summary", html)
        self.assertIn("index__middle", html)
        self.assertIn("dataset IoU", html)
        self.assertIn("urdf IoU", html)
        self.assertIn("1.0000", html)
        self.assertIn("0.0000", html)

    def test_write_layered_html_shows_original_and_current_urdf_when_baseline_exists(self):
        dataset_tips = {
            "index": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "middle": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        }
        urdf_tips = {
            "index": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "middle": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        }
        urdf_baseline_tips = {
            "index": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "middle": np.array([[0.02, 0.0, 0.0]], dtype=np.float32),
        }
        figures = build_layered_tip_workspace_figures(dataset_tips, urdf_tips)
        overlap = build_workspace_overlap_report(
            dataset_tips,
            urdf_tips,
            voxel_size=0.01,
            urdf_baseline_tips=urdf_baseline_tips,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = write_layered_html(figures, Path(tmpdir) / "workspace.html", overlap_report=overlap)
            html = html_path.read_text()

        self.assertIn("original URDF IoU", html)
        self.assertIn("current URDF IoU", html)
        self.assertIn("IoU delta", html)
        self.assertIn("fixed human capture baseline", html)
        self.assertIn("+1", html)


    def test_load_aa_limit_overrides_from_search_report_reads_candidate_rank(self):
        report = {
            "top_candidates": [
                {
                    "limit_comparison": {
                        "F2-R-MCP2": {"candidate": [-0.1, 0.2], "current": [-0.3, 0.35], "delta": [0.2, -0.15]},
                        "F3-R-MCP2": {"candidate": [-0.2, 0.1], "current": [-0.3, 0.35], "delta": [0.1, -0.25]},
                    }
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "search.json"
            path.write_text(__import__("json").dumps(report))
            overrides = load_aa_limit_overrides_from_search_report(path, rank=1)

        self.assertEqual(overrides["F2-R-MCP2"], (-0.1, 0.2))
        self.assertEqual(overrides["F3-R-MCP2"], (-0.2, 0.1))

    def test_summarize_workspace_alignment_reports_centroid_and_nn_distance(self):
        dataset_tips = {
            "thumb": np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float32),
        }
        urdf_tips = {
            "thumb": np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float32),
        }

        report = summarize_workspace_alignment(dataset_tips, urdf_tips)

        self.assertEqual(report["fingers"]["thumb"]["dataset_samples"], 2)
        self.assertEqual(report["fingers"]["thumb"]["urdf_samples"], 2)
        self.assertEqual(report["fingers"]["thumb"]["dataset_to_urdf_nn_mean"], 0.0)
        self.assertEqual(report["fingers"]["thumb"]["urdf_to_dataset_nn_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
