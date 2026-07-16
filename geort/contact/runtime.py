"""CPU runtime contact prediction and bounded single-pair pinch refinement.

The classifier consumes the same raw metric, hand-base HTS landmarks as the
D1 label bundle.  Refinement operates on physical robot joint angles; only
the thumb and the selected opposing finger's four joint coordinates move.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from geort.analytic_fk import AnalyticFK
from geort.anchor.compat import get_joint_limits
from geort.contact.auto_label_contacts import PAIR_LANDMARKS, PAIR_NAMES
from geort.contact.contact_model import ContactMLP
from geort.utils.config_utils import (
    parse_config_joint_limit,
    parse_config_keypoint_info,
    select_keypoint_types,
)


@dataclass(frozen=True, slots=True)
class ContactSelection:
    """Highest-probability active contact pair and its continuous blend weight."""

    pair_index: int
    pair_name: str
    probability: float
    weight: float
    ignored_pair_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContactRefinementResult:
    """Per-frame pinch refinement result, including the v1 multi-pair deviation."""

    q_out: np.ndarray
    probabilities: np.ndarray
    selection: ContactSelection | None


def extract_contact_features(keypoints: np.ndarray) -> np.ndarray:
    """Extract four ``[thumb_tip, other_tip]`` 6D metric hand-base features."""
    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"Expected metric hand-base keypoints [21, 3], got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Contact features require finite metric hand-base keypoints")
    return np.stack(
        [np.concatenate((points[thumb], points[finger])) for thumb, finger in PAIR_LANDMARKS],
        axis=0,
    ).astype(np.float32, copy=False)


class ContactRefiner:
    """Load four contact MLPs and apply deterministic bounded single-pair IK."""

    def __init__(
        self,
        *,
        models: dict[str, ContactMLP],
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        fk: AnalyticFK | None = None,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        target_distance: float = 0.0,
        regularization: float = 1e-3,
        steps: int = 40,
    ) -> None:
        if tuple(models) != PAIR_NAMES:
            raise ValueError(f"Contact model order must be {PAIR_NAMES}, got {tuple(models)}")
        scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
        scaler_scale = np.asarray(scaler_scale, dtype=np.float32)
        if scaler_mean.shape != (4, 6) or scaler_scale.shape != (4, 6):
            raise ValueError("Contact scaler tensors must have shape [4, 6]")
        if not np.isfinite(scaler_mean).all() or not np.isfinite(scaler_scale).all() or np.any(scaler_scale <= 0.0):
            raise ValueError("Contact scaler statistics must be finite with positive scales")
        if target_distance < 0.0 or regularization < 0.0 or steps <= 0:
            raise ValueError("target_distance and regularization must be non-negative; steps must be positive")
        if (fk is None) != (lower is None) or (fk is None) != (upper is None):
            raise ValueError("Analytic FK and both joint-limit vectors must be supplied together")

        self.models = models
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.fk = fk.cpu().eval() if fk is not None else None
        self.lower = None if lower is None else np.asarray(lower, dtype=np.float32)
        self.upper = None if upper is None else np.asarray(upper, dtype=np.float32)
        if self.lower is not None and (self.lower.shape != (20,) or self.upper.shape != (20,) or np.any(self.lower >= self.upper)):
            raise ValueError("Contact refinement joint limits must be ordered [20] physical radians")
        self.target_distance = float(target_distance)
        self.regularization = float(regularization)
        self.steps = int(steps)

    @classmethod
    def load(
        cls,
        checkpoint_path: Path | str,
        *,
        hand_config: dict[str, Any] | None = None,
        target_distance: float = 0.0,
        regularization: float = 1e-3,
        steps: int = 40,
    ) -> "ContactRefiner":
        """Load the D1 four-MLP checkpoint and optional custom-right analytic FK."""
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != 1:
            raise ValueError("Unsupported contact checkpoint schema_version")
        hidden_dims = tuple(int(value) for value in checkpoint.get("hidden_dims", ()))
        pairs = checkpoint.get("pairs")
        if not hidden_dims or not isinstance(pairs, dict) or tuple(pairs) != PAIR_NAMES:
            raise ValueError("Contact checkpoint does not contain the required four ordered pair models")

        models: dict[str, ContactMLP] = {}
        means: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        for pair_index, name in enumerate(PAIR_NAMES):
            saved = pairs[name]
            landmarks = np.asarray(saved.get("landmark_indices"), dtype=np.int64)
            if not np.array_equal(landmarks, np.asarray(PAIR_LANDMARKS[pair_index])):
                raise ValueError(f"{name} landmark ordering differs from the D1 labeling contract")
            model = ContactMLP(hidden_dims)
            model.load_state_dict(saved["state_dict"])
            model.eval()
            models[name] = model
            means.append(np.asarray(saved["scaler_mean"], dtype=np.float32))
            scales.append(np.asarray(saved["scaler_scale"], dtype=np.float32))

        fk = None
        lower = upper = None
        if hand_config is not None:
            if "joint" in hand_config:
                lower, upper = parse_config_joint_limit(hand_config)
            else:
                # Current custom_right keeps physical limits in the SAPIEN
                # hand model rather than duplicating a stale config field.
                lower, upper = get_joint_limits(hand_config)
            keypoints = select_keypoint_types(
                parse_config_keypoint_info(hand_config), allowed_types=("tip",)
            )
            if tuple(keypoints["finger"]) != ("thumb", "index", "middle", "ring", "pinky"):
                raise ValueError("Contact refinement requires custom-right tip order thumb/index/middle/ring/pinky")
            fk = AnalyticFK(
                hand_config["urdf_path"], lower, upper, tip_offsets=keypoints["offset"]
            )
        return cls(
            models=models,
            scaler_mean=np.stack(means),
            scaler_scale=np.stack(scales),
            fk=fk,
            lower=lower,
            upper=upper,
            target_distance=target_distance,
            regularization=regularization,
            steps=steps,
        )

    def probabilities(self, keypoints: np.ndarray) -> np.ndarray:
        """Return the four contact probabilities for raw metric hand-base landmarks."""
        features = (extract_contact_features(keypoints) - self.scaler_mean) / self.scaler_scale
        probabilities = []
        with torch.no_grad():
            for pair_index, name in enumerate(PAIR_NAMES):
                logits = self.models[name](torch.from_numpy(features[pair_index]).unsqueeze(0))
                probabilities.append(float(torch.sigmoid(logits).item()))
        return np.asarray(probabilities, dtype=np.float32)

    @staticmethod
    def select_trigger(probabilities: np.ndarray, *, p_lo: float, p_hi: float) -> ContactSelection | None:
        """Select one pair; ties retain D1 pair order and below-low probabilities do not trigger."""
        values = np.asarray(probabilities, dtype=np.float32)
        if values.shape != (4,) or not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Contact probabilities must be finite [4] values in [0, 1]")
        if not 0.0 <= p_lo < p_hi <= 1.0:
            raise ValueError("Contact trigger thresholds must satisfy 0 <= p_lo < p_hi <= 1")
        pair_index = int(np.argmax(values))
        weight = float(np.clip((values[pair_index] - p_lo) / (p_hi - p_lo), 0.0, 1.0))
        if weight == 0.0:
            return None
        ignored = tuple(
            PAIR_NAMES[index]
            for index, probability in enumerate(values)
            if index != pair_index and probability > p_lo
        )
        return ContactSelection(
            pair_index=pair_index,
            pair_name=PAIR_NAMES[pair_index],
            probability=float(values[pair_index]),
            weight=weight,
            ignored_pair_names=ignored,
        )

    @staticmethod
    def blend_qpos(q_map: np.ndarray, q_pinch: np.ndarray, weight: float) -> np.ndarray:
        """Continuously mix mapped and pinch-refined physical qpos without a hysteresis switch."""
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Contact blend weight must lie in [0, 1]")
        mapped = np.asarray(q_map)
        pinch = np.asarray(q_pinch)
        if mapped.shape != (20,) or pinch.shape != (20,):
            raise ValueError("Contact blending requires physical qpos [20]")
        if weight == 0.0:
            return mapped.copy()
        return ((1.0 - weight) * mapped + weight * pinch).astype(mapped.dtype, copy=False)

    def _require_fk(self) -> tuple[AnalyticFK, np.ndarray, np.ndarray]:
        if self.fk is None or self.lower is None or self.upper is None:
            raise RuntimeError("Contact refinement needs hand_config so analytic FK and joint limits are available")
        return self.fk, self.lower, self.upper

    def _tip_distances(self, qpos: torch.Tensor, pair_index: int) -> torch.Tensor:
        fk, lower, upper = self._require_fk()
        if qpos.ndim != 2 or qpos.shape[1] != 20:
            raise ValueError("Contact FK requires physical qpos [B, 20]")
        lower_tensor = torch.from_numpy(lower)
        upper_tensor = torch.from_numpy(upper)
        normalized = 2.0 * (qpos - lower_tensor) / (upper_tensor - lower_tensor) - 1.0
        tips = fk(normalized)
        return (tips[:, 0] - tips[:, pair_index + 1]).norm(dim=1)

    def tip_distance(self, qpos: np.ndarray, *, pair_index: int) -> float:
        """Compute selected thumb-to-finger analytic-FK distance in metres."""
        values = np.asarray(qpos, dtype=np.float32)
        if values.shape != (20,):
            raise ValueError("Contact FK distance requires physical qpos [20]")
        with torch.no_grad():
            return float(self._tip_distances(torch.from_numpy(values).unsqueeze(0), pair_index).item())

    def refine_qpos_batch(self, q_map: np.ndarray, *, pair_index: int) -> np.ndarray:
        """Batch-equivalent projected Adam used for CPU acceptance evaluation."""
        self._require_fk()
        if pair_index not in range(4):
            raise ValueError("pair_index must identify thumb_index through thumb_pinky")
        mapped = np.asarray(q_map, dtype=np.float32)
        if mapped.ndim != 2 or mapped.shape[1] != 20 or not np.isfinite(mapped).all():
            raise ValueError("Contact refinement requires finite physical qpos [B, 20]")
        _, lower, upper = self._require_fk()
        base = torch.from_numpy(np.clip(mapped, lower, upper))
        active_indices = torch.tensor((0, 1, 2, 3, 4 + 4 * pair_index, 5 + 4 * pair_index, 6 + 4 * pair_index, 7 + 4 * pair_index))
        active_lower = torch.from_numpy(lower)[active_indices]
        active_upper = torch.from_numpy(upper)[active_indices]
        active = base[:, active_indices].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam((active,), lr=0.02)

        for _ in range(self.steps):
            optimizer.zero_grad(set_to_none=True)
            current = base.clone()
            current[:, active_indices] = active
            tip_distance = self._tip_distances(current, pair_index)
            objective = (
                (tip_distance - self.target_distance).square()
                + self.regularization * (active - base[:, active_indices]).square().sum(dim=1)
            ).sum()
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                active.clamp_(active_lower, active_upper)

        output = base.clone()
        output[:, active_indices] = active.detach()
        return output.numpy().astype(np.float32, copy=False)

    def refine_qpos(self, q_map: np.ndarray, *, pair_index: int) -> np.ndarray:
        """Run fixed-count projected Adam over exactly thumb + selected-finger DOFs."""
        mapped = np.asarray(q_map, dtype=np.float32)
        if mapped.shape != (20,):
            raise ValueError("Contact refinement requires physical qpos [20]")
        return self.refine_qpos_batch(mapped[None, :], pair_index=pair_index)[0]

    def refine_from_keypoints(
        self,
        q_map: np.ndarray,
        keypoints: np.ndarray,
        *,
        p_lo: float,
        p_hi: float,
    ) -> ContactRefinementResult:
        """Classify, select the v1 single pair, refine it, then apply continuous blending."""
        probabilities = self.probabilities(keypoints)
        selection = self.select_trigger(probabilities, p_lo=p_lo, p_hi=p_hi)
        mapped = np.asarray(q_map)
        if selection is None:
            return ContactRefinementResult(q_out=mapped.copy(), probabilities=probabilities, selection=None)
        pinch = self.refine_qpos(mapped, pair_index=selection.pair_index)
        return ContactRefinementResult(
            q_out=self.blend_qpos(mapped, pinch, selection.weight),
            probabilities=probabilities,
            selection=selection,
        )
