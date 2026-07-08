# Current left/right checkpoint recipe

This note records how the current usable left and right hand checkpoints were produced, and which generated artifacts are intentionally kept.

## Kept results

Right hand:

- Checkpoint: `checkpoint/custom_right_2026-07-08_11-17-07_aa_limit_manual_prior_mcp1prior_w1_a05_full`
- Latest alias: `checkpoint/custom_right_last`
- Target cloud: `data/custom_right.npz`
- Mold/debug input: `data/custom_right_humanshaped_manual_prior_mcp1boost_a06_debug/mold.json`
- Limit search report: `outputs/visualizations/custom_right_aa_limit_search_baseline061_manual_prior.json`

Left hand:

- Checkpoint: `checkpoint/custom_left_2026-07-08_14-30-25_aa_limit_manual_prior_mcp1prior_w1_a05_full`
- Latest alias: `checkpoint/custom_left_last`
- Target cloud: `data/custom_left.npz`
- Mold/debug input: `data/custom_left_humanshaped_manual_prior_mcp1boost_a06_debug/mold.json`
- Limit search report: `outputs/visualizations/custom_left_aa_limit_search_baseline061_manual_prior_rerun.json`

## Extra processing

The final checkpoints are not from the plain Quest-data AA-limit search. The MCP2 search was run from a restored mechanical baseline of `[-0.61, 0.61]` for F2-F5 MCP2, then constrained with a manually measured four-finger closure prior:

```text
F2 MCP2 = -0.264
F3 MCP2 =  0.000
F4 MCP2 =  0.239
F5 MCP2 =  0.552
```

The prior is applied to both hands. The final searched MCP2 limits are:

```text
F2 MCP2 [-0.380431, 0.087049]
F3 MCP2 [-0.242572, 0.204627]
F4 MCP2 [-0.498692, 0.477526]
F5 MCP2 [-0.440455, 0.61]
```

Search quality summary:

```text
Right: loss=4.913967, replay_success=0.956403, manual_prior_penalty=0, closure_penalty=0
Left:  loss=5.838853, replay_success=0.795043, manual_prior_penalty=0, closure_penalty=0
```

The target clouds were rebuilt with human-shaped sampling and fist strengthening:

- `--fist-mcp1-boost-top-fraction 0.08`
- `--fist-mcp1-boost-alpha 0.6`

Training also used the MCP1 fist prior:

- `--w_mcp1_fist_prior 1`
- `--mcp1_fist_prior_top_fraction 0.08`
- `--mcp1_fist_prior_target_alpha 0.5`
- `--mcp1_fist_prior_mcp_weight 2.0`
- `--mcp1_fist_prior_pip_weight 1.0`
- `--mcp1_fist_prior_dip_weight 0.7`

Other training weights:

```text
--w_chamfer 80
--w_curvature 0.1
--w_collision 0
--w_pinch 5
--pinch_threshold 0.015
--w_segment_direction 0.5
--chamfer_target human
```

## Reproduction commands

Right limit search:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python geort/mocap/search_custom_aa_limits.py \
  --hand custom_right \
  --human_data hts_right_20260703_quest3_v3 \
  --num_candidates 100 \
  --samples_per_finger 2000 \
  --top_k 10 \
  --search_mode coarse_to_fine \
  --refine_rounds 2 \
  --refine_top_k 5 \
  --refine_samples_per_parent 8 \
  --output outputs/visualizations/custom_right_aa_limit_search_baseline061_manual_prior.json
```

Left limit search:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python geort/mocap/search_custom_aa_limits.py \
  --hand custom_left \
  --human_data hts_left \
  --num_candidates 100 \
  --samples_per_finger 2000 \
  --top_k 10 \
  --search_mode coarse_to_fine \
  --refine_rounds 2 \
  --refine_top_k 5 \
  --refine_samples_per_parent 8 \
  --output outputs/visualizations/custom_left_aa_limit_search_baseline061_manual_prior_rerun.json
```

Right target cloud:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python geort/mocap/build_target_cloud.py \
  --hand custom_right \
  --motion data/hts_right_20260703_quest3_v3.npy \
  --rest data/hts_right_20260707_anchor_v4_rest.npy \
  --output data/custom_right.npz \
  --debug-dir data/custom_right_humanshaped_manual_prior_mcp1boost_a06_debug \
  --fist-mcp1-boost-top-fraction 0.08 \
  --fist-mcp1-boost-alpha 0.6
```

Left target cloud:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python geort/mocap/build_target_cloud.py \
  --hand custom_left \
  --motion data/hts_left.npy \
  --rest data/hts_left_20260707_anchor_v4_rest.npy \
  --output data/custom_left.npz \
  --debug-dir data/custom_left_humanshaped_manual_prior_mcp1boost_a06_debug \
  --fist-mcp1-boost-top-fraction 0.08 \
  --fist-mcp1-boost-alpha 0.6
```

Right training:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python geort/trainer.py \
  -hand custom_right \
  -human_data hts_right_20260703_quest3_v3 \
  -ckpt_tag aa_limit_manual_prior_mcp1prior_w1_a05_full \
  --w_chamfer 80 \
  --w_curvature 0.1 \
  --w_collision 0 \
  --w_pinch 5 \
  --pinch_threshold 0.015 \
  --w_segment_direction 0.5 \
  --w_mcp1_fist_prior 1 \
  --mcp1_fist_prior_top_fraction 0.08 \
  --mcp1_fist_prior_target_alpha 0.5 \
  --mcp1_fist_prior_mcp_weight 2.0 \
  --mcp1_fist_prior_pip_weight 1.0 \
  --mcp1_fist_prior_dip_weight 0.7 \
  --chamfer_target human \
  --chamfer_target_path data/custom_right.npz \
  --mold_path data/custom_right_humanshaped_manual_prior_mcp1boost_a06_debug/mold.json
```

Left training:

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python geort/trainer.py \
  -hand custom_left \
  -human_data hts_left \
  -ckpt_tag aa_limit_manual_prior_mcp1prior_w1_a05_full \
  --w_chamfer 80 \
  --w_curvature 0.1 \
  --w_collision 0 \
  --w_pinch 5 \
  --pinch_threshold 0.015 \
  --w_segment_direction 0.5 \
  --w_mcp1_fist_prior 1 \
  --mcp1_fist_prior_top_fraction 0.08 \
  --mcp1_fist_prior_target_alpha 0.5 \
  --mcp1_fist_prior_mcp_weight 2.0 \
  --mcp1_fist_prior_pip_weight 1.0 \
  --mcp1_fist_prior_dip_weight 0.7 \
  --chamfer_target human \
  --chamfer_target_path data/custom_left.npz \
  --mold_path data/custom_left_humanshaped_manual_prior_mcp1boost_a06_debug/mold.json
```

## Cleanup policy

Historical checkpoints, old anchor/baseline target clouds, old debug directories, and non-final limit-search reports were removed. The repository now keeps only the current usable left/right checkpoint pair and the inputs/reports required to explain or reproduce them.

The data directory was also reduced to the datasets required by the current pair:

- `data/hts_right_20260703_quest3_v3.npy`
- `data/hts_right_20260707_anchor_v4_rest.npy`
- `data/hts_left.npy`
- `data/hts_left_20260707_anchor_v4_rest.npy`
- `data/custom_right.npz`
- `data/custom_left.npz`
- the two `data/custom_*_humanshaped_manual_prior_mcp1boost_a06_debug/` directories
