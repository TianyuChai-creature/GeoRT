"""Create a SHA256 ledger for final-matrix and explicitly registered realtime checkpoints."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from geort.mocap.realtime_provenance import sha256_file


def _checkpoint_record(checkpoint: Path, repo_root: Path) -> dict:
    checkpoint = checkpoint.resolve()
    try:
        checkpoint_rel = checkpoint.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"checkpoint is outside repository root: {checkpoint}") from exc
    metadata = json.loads((checkpoint / "training_metadata.json").read_text())
    return {
        "checkpoint": checkpoint_rel,
        "last_pth_sha256": sha256_file(checkpoint / "last.pth"),
        "motion_frame": metadata["cli_args"]["motion_frame"],
        "anchor": metadata["anchor"],
    }


def build_ledger(
    final_matrix: Path,
    repo_root: Path,
    *,
    extra_checkpoints: Mapping[str, Path] | None = None,
) -> dict:
    """Return strict realtime provenance records for matrix and named extra checkpoints."""
    manifest = json.loads(final_matrix.read_text())
    runs = {}
    for run_id, run in sorted(manifest["manifest"]["runs"].items()):
        checkpoint = repo_root / run["checkpoint"]
        runs[run_id] = _checkpoint_record(checkpoint, repo_root)
    for run_id, checkpoint in sorted((extra_checkpoints or {}).items()):
        if run_id in runs:
            raise ValueError(f"duplicate realtime checkpoint run id: {run_id}")
        runs[run_id] = _checkpoint_record(Path(checkpoint), repo_root)
    return {
        "schema_version": 1,
        "source_final_matrix": final_matrix.as_posix(),
        "runs": runs,
    }


def _parse_extra_checkpoints(values: list[str]) -> dict[str, Path]:
    extras = {}
    for value in values:
        run_id, separator, path = value.partition("=")
        if not separator or not run_id or not path:
            raise ValueError("--extra-checkpoint must have RUN_ID=CHECKPOINT_DIR form")
        if run_id in extras:
            raise ValueError(f"duplicate --extra-checkpoint run id: {run_id}")
        extras[run_id] = Path(path)
    return extras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-matrix", type=Path, default=Path("outputs/final_matrix/final_matrix.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/final_matrix/checkpoint_hashes.json"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--extra-checkpoint", action="append", default=[], metavar="RUN_ID=CHECKPOINT_DIR",
        help="Additional audited realtime checkpoint, for example c2b_s42=checkpoint/custom_right_..._c2b_s42.",
    )
    args = parser.parse_args()
    ledger = build_ledger(args.final_matrix, args.repo_root, extra_checkpoints=_parse_extra_checkpoints(args.extra_checkpoint))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}: {len(ledger['runs'])} checkpoints")


if __name__ == "__main__":
    main()
