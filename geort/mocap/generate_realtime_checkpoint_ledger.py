"""Create a SHA256 ledger for final-matrix checkpoints without modifying them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geort.mocap.realtime_provenance import sha256_file


def build_ledger(final_matrix: Path, repo_root: Path) -> dict:
    """Return strict realtime provenance records for every archived matrix run."""
    manifest = json.loads(final_matrix.read_text())
    runs = {}
    for run_id, run in sorted(manifest["manifest"]["runs"].items()):
        checkpoint_rel = run["checkpoint"]
        checkpoint = repo_root / checkpoint_rel
        metadata = run["metadata"]
        runs[run_id] = {
            "checkpoint": checkpoint_rel,
            "last_pth_sha256": sha256_file(checkpoint / "last.pth"),
            "motion_frame": metadata["cli_args"]["motion_frame"],
            "anchor": metadata["anchor"],
        }
    return {
        "schema_version": 1,
        "source_final_matrix": final_matrix.as_posix(),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-matrix", type=Path, default=Path("outputs/final_matrix/final_matrix.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/final_matrix/checkpoint_hashes.json"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    ledger = build_ledger(args.final_matrix, args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}: {len(ledger['runs'])} checkpoints")


if __name__ == "__main__":
    main()
