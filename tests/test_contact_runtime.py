from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import geort.export as export_module

from geort.contact.auto_label_contacts import PAIR_LANDMARKS, PAIR_NAMES
from geort.contact.contact_model import ContactMLP


def _write_checkpoint(path: Path) -> Path:
    torch.manual_seed(7)
    pairs = {}
    for index, name in enumerate(PAIR_NAMES):
        model = ContactMLP((8, 4))
        for parameter in model.parameters():
            parameter.data.zero_()
        model.network[-1].bias.data.fill_(float(index) - 1.5)
        pairs[name] = {
            "state_dict": model.state_dict(),
            "scaler_mean": torch.zeros(6),
            "scaler_scale": torch.ones(6),
            "landmark_indices": torch.tensor(PAIR_LANDMARKS[index]),
        }
    torch.save({"schema_version": 1, "hidden_dims": (8, 4), "pairs": pairs}, path)
    return path


def _points() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    points[4] = (1.0, 0.0, 0.0)
    for pair_index, (_thumb, finger) in enumerate(PAIR_LANDMARKS):
        points[finger] = (0.0, pair_index + 1, 0.0)
    return points


def test_runtime_loads_four_models_and_extracts_metric_hand_base_features(tmp_path: Path) -> None:
    from geort.contact.runtime import ContactRefiner, extract_contact_features

    refiner = ContactRefiner.load(_write_checkpoint(tmp_path / "contact_models.pth"))
    features = extract_contact_features(_points())

    assert features.shape == (4, 6)
    np.testing.assert_array_equal(features[0], np.array([1, 0, 0, 0, 1, 0], dtype=np.float32))
    np.testing.assert_allclose(refiner.probabilities(_points()), torch.sigmoid(torch.tensor([-1.5, -0.5, 0.5, 1.5])).numpy())


def test_probability_trigger_blending_and_tie_break_are_stable(tmp_path: Path) -> None:
    from geort.contact.runtime import ContactRefiner

    refiner = ContactRefiner.load(_write_checkpoint(tmp_path / "contact_models.pth"))
    selected = refiner.select_trigger(np.array([0.8, 0.8, 0.1, 0.2]), p_lo=0.5, p_hi=0.8)
    assert selected.pair_name == "thumb_index"
    assert selected.pair_index == 0
    assert selected.weight == 1.0
    assert refiner.select_trigger(np.zeros(4), p_lo=0.5, p_hi=0.8) is None
    q_map = np.linspace(-0.3, 0.3, 20, dtype=np.float32)
    q_pinch = q_map + 0.2
    np.testing.assert_array_equal(refiner.blend_qpos(q_map, q_pinch, 0.0), q_map)
    np.testing.assert_allclose(refiner.blend_qpos(q_map, q_pinch, 0.5), q_map + 0.1)


def test_refinement_is_bounded_deterministic_and_only_changes_selected_pair(tmp_path: Path) -> None:
    from geort.contact.runtime import ContactRefiner
    from geort.utils.config_utils import get_config

    refiner = ContactRefiner.load(
        _write_checkpoint(tmp_path / "contact_models.pth"),
        hand_config=get_config("custom_right"),
        target_distance=0.0,
        regularization=0.1,
        steps=8,
    )
    q_map = ((refiner.lower + refiner.upper) / 2.0).astype(np.float32)
    q_map[0:8] = refiner.lower[0:8]
    before = refiner.tip_distance(q_map, pair_index=0)
    q_first = refiner.refine_qpos(q_map, pair_index=0)
    q_second = refiner.refine_qpos(q_map, pair_index=0)
    q_batch = refiner.refine_qpos_batch(np.stack((q_map, q_map)), pair_index=0)

    assert np.array_equal(q_first, q_second)
    np.testing.assert_array_equal(q_batch[0], q_first)
    np.testing.assert_array_equal(q_batch[1], q_first)
    assert np.all(q_first >= refiner.lower)
    assert np.all(q_first <= refiner.upper)
    np.testing.assert_array_equal(q_first[8:], q_map[8:])
    assert refiner.tip_distance(q_first, pair_index=0) <= before


def test_export_contact_off_returns_the_original_qpos_bitwise() -> None:
    from geort.export import GeoRTRetargetingModel

    model = object.__new__(GeoRTRetargetingModel)
    model.contact_refiner = None
    q_map = np.linspace(-0.3, 0.3, 20, dtype=np.float32)

    q_out = model._apply_contact_refinement(q_map, _points())

    assert np.array_equal(q_out, q_map)


def test_export_on_path_passes_raw_metric_landmarks_to_contact_refiner(monkeypatch) -> None:
    class FakeMapper:
        def forward(self, _points):
            return torch.zeros((1, 20), dtype=torch.float32)

    class FakeNormalizer:
        def unnormalize(self, _joint):
            return np.zeros((1, 20), dtype=np.float32)

    class FakeRefiner:
        def __init__(self):
            self.received = None

        def refine_from_keypoints(self, q_map, keypoints, **_kwargs):
            self.received = keypoints
            return type("Result", (), {"q_out": q_map})()

    monkeypatch.setattr(export_module, "_select_and_normalize_tips", lambda *_args: np.zeros((5, 3), dtype=np.float32))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    model = object.__new__(export_module.GeoRTRetargetingModel)
    model.human_ids = [4, 8, 12, 16, 20]
    model.finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    model.human_normalization = {}
    model.model = FakeMapper()
    model.qpos_normalizer = FakeNormalizer()
    model.contact_refiner = FakeRefiner()
    model.contact_p_lo = 0.5
    model.contact_p_hi = 0.8
    model.last_contact_refinement = None
    raw = _points()

    model.forward(raw)

    assert model.contact_refiner.received is raw
