"""Replace only Ring lateral human rows while copying parent robot fields bitwise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geort.anchor.arc_bending_v2 import _atomic_npz
from geort.anchor.interpolate import interpolate_sparse_trajectory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path, default=Path("data/anchors_human_right_arc_bending_v2_ringmono.npz"))
    parser.add_argument("--parent", type=Path, default=Path("data/anchors_custom_right_arc_bending_v2_lateral085_exactknots.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/anchors_custom_right_arc_bending_v2_lateral085_ringmono_frozenrobot.npz"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite versioned bundle")
    with np.load(args.human, allow_pickle=False) as bundle:
        human = {key: np.asarray(bundle[key]) for key in bundle.files if key != "metadata_json"}
        human_meta = json.loads(str(bundle["metadata_json"].item()))
    with np.load(args.parent, allow_pickle=False) as bundle:
        pair = {key: np.array(bundle[key], copy=True) for key in bundle.files if key != "metadata_json"}
        parent_meta = json.loads(str(bundle["metadata_json"].item()))
    mask_h = (human["finger_indices"] == 3) & (human["anchor_types"].astype(str) == "lateral")
    mask_p = (pair["finger_indices"] == 3) & (pair["anchor_types"].astype(str) == "lateral")
    tip_ids = np.array((4, 8, 12, 16, 20), dtype=np.int64)
    context = interpolate_sparse_trajectory(human["human_frames"][mask_h, tip_ids, :].reshape(5, -1), 50)["points"].reshape(50, 5, 3)
    pair["human_tip_contexts"][mask_p] = context
    pair["human_points"][mask_p] = context[:, 3, :]
    metadata = dict(parent_meta)
    metadata.update({"schema_version": 5, "generation": "arc_bending_v2_lateral085_ringmono_frozenrobot", "parent_bundle": str(args.parent), "human_anchor_source": str(args.human), "ring_lateral_reselection": human_meta["ring_lateral_reselection"], "robot_fields_copied_bitwise": True})
    _atomic_npz(args.output, **pair, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps({"output": str(args.output), "ring_rows": int(mask_p.sum())}, sort_keys=True))
    return args.output


if __name__ == "__main__":
    main()
