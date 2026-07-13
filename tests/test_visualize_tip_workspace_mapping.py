from __future__ import annotations

import numpy as np

from geort.mocap import visualize_tip_workspace as workspace


class _FakeRetargetingModel:
    def forward(self, frame: np.ndarray) -> np.ndarray:
        return np.array([frame[0, 0]], dtype=np.float32)


class _FakeHand:
    def __init__(self) -> None:
        self.initialized = False

    def initialize_keypoint(self, keypoint_link_names, keypoint_offsets) -> None:
        assert keypoint_link_names == ["thumb_pip", "thumb_tip", "index_tip"]
        assert len(keypoint_offsets) == 3
        self.initialized = True

    def keypoint_from_qpos(self, qpos, ret_vec=False) -> np.ndarray:
        assert self.initialized
        assert ret_vec
        value = float(qpos[0])
        return np.array(
            [
                [value, 0.0, 0.0],
                [value, 1.0, 0.0],
                [value, 2.0, 0.0],
            ],
            dtype=np.float32,
        )


def test_map_dataset_tip_points_uses_checkpoint_qpos_and_urdf_fk() -> None:
    frames = np.zeros((4, 21, 3), dtype=np.float32)
    frames[:, 0, 0] = [10.0, 20.0, 30.0, 40.0]
    keypoint_info = {
        "finger": ["thumb", "thumb", "index"],
        "type": ["pip", "tip", "tip"],
        "link": ["thumb_pip", "thumb_tip", "index_tip"],
        "offset": [[0.0, 0.0, 0.0]] * 3,
    }

    mapped = workspace.map_dataset_tip_points(
        frames,
        keypoint_info,
        retargeting_model=_FakeRetargetingModel(),
        hand=_FakeHand(),
        max_frames=2,
    )

    assert list(mapped) == ["thumb", "index"]
    assert np.allclose(mapped["thumb"], [[10.0, 1.0, 0.0], [40.0, 1.0, 0.0]])
    assert np.allclose(mapped["index"], [[10.0, 2.0, 0.0], [40.0, 2.0, 0.0]])


def test_workspace_cli_accepts_checkpoint_mapping_mode() -> None:
    args = workspace.build_arg_parser().parse_args(
        ["--ckpt_tag", "step5", "--mapped_max_frames", "123"]
    )

    assert args.ckpt_tag == "step5"
    assert args.mapped_max_frames == 123


def test_workspace_figures_label_checkpoint_mapped_cloud() -> None:
    points = {"thumb": np.array([[0.0, 0.0, 0.0]], dtype=np.float32)}

    figures = workspace.build_layered_tip_workspace_figures(
        points,
        points,
        source_label="Mapped",
    )

    assert figures["dataset_all"].layout.title.text == "Mapped TIP workspace: all fingers"
    assert figures["dataset_all"].data[0].name == "mapped_thumb_tip"
