"""Chamfer target selection and training metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:
    from geort.utils.hash_utils import sha256_file
except Exception:  # pragma: no cover - supports direct file loading in tests.
    import hashlib

    def sha256_file(path: Path | str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


@dataclass(frozen=True)
class ChamferTargetPath:
    path: Path
    requires_existing: bool


def resolve_chamfer_target_path(
    *,
    hand_name: str,
    chamfer_target: str,
    explicit_path: Path | str | None,
) -> ChamferTargetPath:
    target = chamfer_target.lower()
    if target not in ("uniform", "human"):
        raise ValueError("--chamfer_target must be one of: uniform, human")

    if explicit_path is not None:
        path = Path(explicit_path)
    elif target == "human":
        path = Path("data") / f"{hand_name}_humanshaped.npz"
    else:
        path = Path("data") / f"{hand_name}.npz"

    requires_existing = target == "human" or explicit_path is not None
    if requires_existing and not path.exists():
        kind = "human-shaped chamfer target" if target == "human" else "chamfer target"
        raise FileNotFoundError(f"{kind} file was not found: {path}")
    return ChamferTargetPath(path=path, requires_existing=requires_existing)


def build_training_metadata(
    *,
    chamfer_target: str,
    target_path: Path | str,
    mold_path: Path | str | None,
    human_data_path: Path | str,
    n_epoch: int,
    loss_weights: dict[str, float],
    cli_args: dict[str, Any],
) -> dict[str, Any]:
    target = Path(target_path)
    mold = Path(mold_path) if mold_path not in (None, "") else None
    metadata: dict[str, Any] = {
        "chamfer_target": chamfer_target,
        "target_cloud": {
            "path": target.as_posix(),
            "sha256": sha256_file(target) if target.exists() else None,
        },
        "mold": None,
        "human_data_path": Path(human_data_path).as_posix(),
        "n_epoch": int(n_epoch),
        "loss_weights": {key: float(value) for key, value in loss_weights.items()},
        "cli_args": cli_args,
    }
    if mold is not None:
        metadata["mold"] = {
            "path": mold.as_posix(),
            "sha256": sha256_file(mold) if mold.exists() else None,
        }
    return metadata


def save_training_metadata(path: Path | str, metadata: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output
