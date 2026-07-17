"""Offline D1 parity gate for the Quest realtime C2 mapping path."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geort import load_model
from geort.mocap.hts_realtime_inference import (
    DEFAULT_C2B_S42_CHECKPOINT,
    map_realtime_frame,
    require_c2b_s42_sha,
)
from geort.mocap.realtime_provenance import verify_archived_checkpoint


def run_parity(*, checkpoint: Path, archive_root: Path, data: Path, frames: int, seed: int) -> float:
    """Return max physical-qpos difference between realtime and evaluation API paths."""
    provenance = verify_archived_checkpoint(checkpoint, archive_root, repo_root=Path.cwd())
    require_c2b_s42_sha(provenance.last_pth_sha256)
    d1 = np.load(data, mmap_mode="r")
    if frames > len(d1):
        raise ValueError(f"requested {frames} frames, dataset only has {len(d1)}")
    rows = np.random.RandomState(seed).choice(len(d1), size=frames, replace=False)
    model = load_model(str(checkpoint), contact_refine="off")
    max_abs = 0.0
    for row in rows:
        frame = np.asarray(d1[row], dtype=np.float32)
        evaluation_qpos = model.forward(frame)
        _, realtime_qpos = map_realtime_frame(model, frame)
        max_abs = max(max_abs, float(np.max(np.abs(evaluation_qpos - realtime_qpos))))
    return max_abs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_C2B_S42_CHECKPOINT))
    parser.add_argument("--archive-root", type=Path, default=Path("outputs/final_matrix"))
    parser.add_argument("--data", type=Path, default=Path("data/hts_right.npy"))
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    max_abs = run_parity(
        checkpoint=args.checkpoint, archive_root=args.archive_root, data=args.data,
        frames=args.frames, seed=args.seed,
    )
    print(f"realtime parity: frames={args.frames} seed={args.seed} max_abs_qpos_rad={max_abs:.9g}")


if __name__ == "__main__":
    main()
