# Local `L_motion` frame validation

## Constants and conventions

- `R_task = R_link @ [c1, c2, c3]`, with `c1` the configured TIP offset direction, `c2` the Gram--Schmidt DIP axis, and `c3 = c1 x c2`.
- `custom_right` DIP-axis flips: `false, false, false, false, false`.
- Raw `c1`--DIP-axis angles (degrees): `80.9375158622, 73.0091767080, 73.0091767080, 73.0091767080, 73.0091767080`.

## Recorded diagnostics

- SAPIEN versus analytic distal-link rotation, 1000 `RandomState(42)` qpos: max `6.905339541845024e-04 rad`; mean `4.5109754864824936e-05 rad`.
- Analytic link-frame max orthogonality/determinant deviations: `4.472828385448935e-07`, `4.365594521438254e-07`.
- Robot task-frame max orthogonality/determinant deviations: `5.960464477539062e-07`, `5.960464477539062e-07`.
- Human frame max orthogonality/determinant deviations: `8.881784197001252e-16`, `8.881784197001252e-16`.
- D1 (`data/hts_right.npy`, 234114 rows) human fallback counts: `0, 0, 0, 0, 0`.
- Robot β sign values: `0.9875266419, 0.9564595319, 0.9564546408, 0.9564435830, 0.9565928999`.
- Human nominal palmward-β sign values: `0.99999999999, 0.99999999999, 0.99999999999, 0.99999999999, 0.99999999999`.
- Local rigid-rotation max loss difference across ±45°/±90° on X/Y/Z: `8.381903171539307e-09`.
- Global counterpart max difference in the same synthetic diagnostic: `6.05359673500061e-09`.
- NN borrowing baseline, normalized space, 1000 `RandomState(42)` D1 rows, `seed42_null_v3_full`: p50 `0.012341046638825024`; p95 `0.023514572563671034`.
- CUDA global 50-step numerical rows compared with Commit A baseline: 50 rows, first differing row `none`.

The existing finalized target cloud and finalized anchor bundle were not rewritten. Local training requires a newly generated target cloud containing `link_rotation`; a legacy target raises an explicit error.
