# Training datasets

This branch starts from the two raw D1 recordings:

- `hts_left.npy`
- `hts_right.npy`

Training reads these NPY files directly. Generated preprocessing data, robot kinematics caches, reports, and checkpoints are runtime artifacts and remain ignored by Git.

## AnyDexRT prepared data

Step 2 prepares each hand with the existing config keypoint order. For every
finger, human and robot points get their own AABB center and one positive
isotropic scale. No rotation or per-finger coordinate frame is applied.

Commands:
  python -m geort.data.prepare --hand custom_right --human-data hts_right
  python -m geort.data.prepare --hand custom_left --human-data hts_left

Each command creates a generated robot kinematics cache plus
hts_<side>_prepared.npz and hts_<side>_prepared.json. The NPZ contains
normalized human_points, normalized robot_points, keypoint/finger order,
and source human landmark IDs. The JSON records human/robot center and scale
per finger and reserves nullable anchors and contact fields. These files
are reproducible runtime artifacts and remain ignored by Git.
