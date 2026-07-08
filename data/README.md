# Training datasets

This directory keeps only the raw acquisition datasets required to reproduce the current custom left/right checkpoints:

- `hts_left.npy`
- `hts_left_20260707_anchor_v4_rest.npy`
- `hts_right_20260703_quest3_v3.npy`
- `hts_right_20260707_anchor_v4_rest.npy`

It also keeps the current derived target clouds and their mold/debug inputs:

- `custom_left.npz`
- `custom_left_humanshaped_manual_prior_mcp1boost_a06_debug/`
- `custom_right.npz`
- `custom_right_humanshaped_manual_prior_mcp1boost_a06_debug/`

Generated balanced datasets, `.npz` caches, reports, frame weights, and local experiment recordings are intentionally ignored by Git. For HTS recordings, use `geort/mocap/hts_prepare_training.py` to create the two ignored training artifacts: `<name>_train.npy` and `<name>_train.json`. Pass the JSON to training when frame weights are needed. Plain `.npy` inputs do not auto-load weight files.
