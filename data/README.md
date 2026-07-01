# Training datasets

This directory contains raw acquisition datasets that are kept in the repository:

- `human_alex.npy`
- `hts_right.npy`
- `hts_left.npy`

Generated balanced datasets, `.npz` caches, reports, frame weights, and local experiment recordings are intentionally ignored by Git. For HTS recordings, use `geort/mocap/hts_prepare_training.py` to create the two ignored training artifacts: `<name>_train.npy` and `<name>_train.json`. Pass the JSON to training when frame weights are needed. Plain `.npy` inputs do not auto-load weight files.
