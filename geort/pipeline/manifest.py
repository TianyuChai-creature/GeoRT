"""Validated hand manifests for hand-agnostic pipeline stations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from geort.anchor.compat import get_joint_limits
from geort.utils.config_utils import get_config

_REQUIRED_KEYS = ("hand_id", "urdf", "hts", "hand_config", "side", "output_root")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HandManifest:
    """One robot/HTS input contract, with project-root-relative data paths."""

    hand_id: str
    urdf: Path
    hts: Path
    hand_config: str
    side: str
    output_root: Path
    anchor_bundle: Path | None = None

    @property
    def output_dir(self) -> Path:
        """Absolute output directory below this repository's ``outputs/`` root."""
        return _PROJECT_ROOT / self.output_root

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Read physical limits through the current kinematic-model compatibility API."""
        lower, upper = get_joint_limits(get_config(self.hand_config))
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _project_path(value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest {key!r} must be a non-empty path string")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"manifest {key!r} must be project-root-relative: {value!r}")
    return (_PROJECT_ROOT / path).resolve()


def load_hand_manifest(path: Path | str) -> HandManifest:
    """Load and validate the explicit five-field hand input contract."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("hand manifest must be a mapping")
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"hand manifest is missing required keys: {missing}")
    hand_id = payload["hand_id"]
    hand_config = payload["hand_config"]
    if not isinstance(hand_id, str) or not hand_id:
        raise ValueError("manifest hand_id must be a non-empty string")
    if not isinstance(hand_config, str) or not hand_config:
        raise ValueError("manifest hand_config must be a non-empty string")
    side = payload["side"]
    if side not in {"R", "L"}:
        raise ValueError("manifest side must be explicit 'R' or 'L'")
    output_root = Path(payload["output_root"])
    expected_output_root = Path("outputs") / hand_id
    if output_root.is_absolute() or output_root != expected_output_root:
        raise ValueError(
            f"manifest output_root must be exactly {expected_output_root.as_posix()!r}, "
            f"got {str(output_root)!r}"
        )
    urdf = _project_path(payload["urdf"], key="urdf")
    hts = _project_path(payload["hts"], key="hts")
    if not urdf.is_file():
        raise FileNotFoundError(f"manifest URDF does not exist: {urdf}")
    if not hts.is_file():
        raise FileNotFoundError(f"manifest HTS does not exist: {hts}")
    config = get_config(hand_config)
    config_urdf = _project_path(config["urdf_path"], key="config urdf_path")
    if config_urdf != urdf:
        raise ValueError(
            f"manifest URDF {urdf} differs from {hand_config} config URDF {config_urdf}"
        )
    anchor_value = payload.get("anchor_bundle")
    anchor_bundle = None if anchor_value is None else _project_path(anchor_value, key="anchor_bundle")
    return HandManifest(
        hand_id=hand_id,
        urdf=urdf,
        hts=hts,
        hand_config=hand_config,
        side=side,
        output_root=output_root,
        anchor_bundle=anchor_bundle,
    )
