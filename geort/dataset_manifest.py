"""Dataset manifest helpers for GeoRT training inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from geort.utils.path import get_data_root, get_package_root


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    data_path: Path
    weights_path: Path | None = None
    weights: list[float] | None = None
    reports: dict[str, Path] | None = None
    transforms: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    manifest_path: Path | None = None


def _resolve_existing_manifest_path(reference: Path | str) -> Path:
    requested = Path(reference)
    if requested.is_file():
        return requested.resolve()
    if requested.is_dir() and (requested / "manifest.json").is_file():
        return (requested / "manifest.json").resolve()

    candidates = [
        Path(get_data_root()) / requested / "manifest.json",
        Path(get_package_root()) / "datasets" / requested / "manifest.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Dataset manifest {reference!r} was not found")


def _resolve_relative_path(base_dir: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_report_paths(base_dir: Path, reports: dict[str, str] | None) -> dict[str, Path]:
    if not reports:
        return {}
    return {key: _resolve_relative_path(base_dir, value) for key, value in reports.items()}


def load_dataset_manifest(reference: Path | str) -> DatasetManifest:
    manifest_path = _resolve_existing_manifest_path(reference)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "data_path" not in data:
        raise ValueError(f"Dataset manifest {manifest_path} is missing data_path")

    base_dir = manifest_path.parent
    data_path = _resolve_relative_path(base_dir, data["data_path"])
    weights_path = _resolve_relative_path(base_dir, data.get("weights_path"))
    weights = data.get("weights")
    if weights is not None:
        weights = [float(value) for value in weights]
    reports = _resolve_report_paths(base_dir, data.get("reports"))

    dataset_id = data.get("id") or manifest_path.parent.name
    return DatasetManifest(
        dataset_id=str(dataset_id),
        data_path=data_path,
        weights_path=weights_path,
        weights=weights,
        reports=reports,
        transforms=list(data.get("transforms", [])),
        metadata=dict(data.get("metadata", {})),
        manifest_path=manifest_path,
    )


def maybe_load_dataset_manifest(reference: Path | str) -> DatasetManifest | None:
    requested = Path(reference)
    if requested.suffix and requested.suffix != ".json":
        return None
    if requested.suffix == ".json" or requested.is_dir():
        return load_dataset_manifest(reference)

    try:
        return load_dataset_manifest(reference)
    except FileNotFoundError:
        return None
