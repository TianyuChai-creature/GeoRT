# Training datasets

This branch starts from the two raw D1 recordings:

- `hts_left.npy`
- `hts_right.npy`

Step 2 reads these NPY files and produces the manifests consumed by training.
Generated preprocessing data, robot kinematics caches, reports, and checkpoints
are runtime artifacts and remain ignored by Git.

## AnyDexRT prepared data

Step 2 selects only the five fingertip landmarks, in the config's TIP order.
For every fingertip, human and robot workspaces get their own AABB center and
one positive isotropic scale. PIP landmarks are not mapper inputs or loss
targets. No rotation or per-finger coordinate frame is applied.

Commands:

```bash
  python -m geort.data.prepare --hand custom_right --human-data hts_right
  python -m geort.data.prepare --hand custom_left --human-data hts_left
```

Each command creates a generated robot kinematics cache plus
`hts_<side>_prepared.npz` and `hts_<side>_prepared.json`. The NPZ contains
normalized TIP-only `human_points`, normalized TIP-only `robot_points`,
keypoint/finger order, and source human landmark IDs. The JSON records
human/robot center and scale per fingertip and reserves nullable anchors and
contact fields. These files are reproducible runtime artifacts and remain
ignored by Git.
