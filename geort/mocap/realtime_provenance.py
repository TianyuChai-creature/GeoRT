"""Strict checkpoint provenance checks for the realtime C2 runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArchivedCheckpoint:
    run_id: str
    checkpoint: Path
    last_pth_sha256: str
    motion_frame: str
    anchor: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _relative_checkpoint(checkpoint: Path, repo_root: Path) -> str:
    try:
        return checkpoint.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"checkpoint is outside repository root: {checkpoint}") from exc


def verify_archived_checkpoint(
    checkpoint: Path | str,
    archive_root: Path | str,
    *,
    repo_root: Path | str,
) -> ArchivedCheckpoint:
    """Verify a checkpoint against the final-matrix archive or raise a named mismatch."""
    checkpoint = Path(checkpoint).resolve()
    archive_root = Path(archive_root)
    repo_root = Path(repo_root)
    ledger = _load_json(archive_root / "checkpoint_hashes.json")
    relative = _relative_checkpoint(checkpoint, repo_root)
    matching = [
        (run_id, record)
        for run_id, record in ledger.get("runs", {}).items()
        if record.get("checkpoint") == relative
    ]
    if len(matching) != 1:
        raise ValueError(f"checkpoint is not registered in final_matrix archive: {relative}")
    run_id, expected = matching[0]
    weights = checkpoint / "last.pth"
    actual_sha = sha256_file(weights)
    if actual_sha != expected.get("last_pth_sha256"):
        raise ValueError(f"checkpoint SHA256 mismatch: {actual_sha} != {expected.get('last_pth_sha256')}")
    metadata = _load_json(checkpoint / "training_metadata.json")
    actual_motion = metadata.get("cli_args", {}).get("motion_frame")
    if actual_motion != expected.get("motion_frame"):
        raise ValueError(f"motion_frame mismatch: {actual_motion!r} != {expected.get('motion_frame')!r}")
    actual_anchor = metadata.get("anchor", {})
    if actual_anchor != expected.get("anchor"):
        raise ValueError(f"anchor metadata mismatch: {actual_anchor!r} != {expected.get('anchor')!r}")
    return ArchivedCheckpoint(
        run_id=run_id,
        checkpoint=checkpoint,
        last_pth_sha256=actual_sha,
        motion_frame=actual_motion,
        anchor=actual_anchor,
    )
