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
- Human-only rigid rotation (`+60°` about Z; robot vectors and frames fixed): local before/after/delta `-1.0`, `-1.0`, `0.0`; global before/after/delta `-1.0`, `-0.5`, `0.5`.
- Collinearity threshold: normalized orthogonal component `sin = 1e-3`; all-D1 overall normalized component min/p1: `7.0532028551209625e-03`, `2.5161862652784358e-02`; raw component min/p1: `1.9460834896006842e-04 m`, `6.933849607988999e-04 m`.
- Commit-A CUDA 50-step device-refactor gate (`a4cdd46c247a4cc26a38be0310d3687a660803a4` → `d103dfe`): baseline rows `50`; candidate rows `50`; first differing row `none`.
- NN borrowing baseline, normalized space, 1000 `RandomState(42)` D1 rows, `seed42_null_v3_full`: p50 `0.012341046638825024`; p95 `0.023514572563671034`.
- CUDA global 50-step numerical rows compared with Commit A baseline: 50 rows, first differing row `none`.

The existing finalized target cloud and finalized anchor bundle were not rewritten. Local training requires a newly generated target cloud containing `link_rotation`; a legacy target raises an explicit error.

## Per-finger D1 orthogonal components

| finger | raw min (m) | raw p1 (m) | sine min | sine p1 |
| --- | ---: | ---: | ---: | ---: |
| thumb | 2.216012562736396e-03 | 2.5183201975138778e-03 | 6.822247366130546e-02 | 7.747444646651376e-02 |
| index | 1.391811532163784e-03 | 1.5201811688442673e-03 | 5.733448938755795e-02 | 6.252607347712652e-02 |
| middle | 1.9460834896006842e-04 | 3.965007253758152e-04 | 7.0532028551209625e-03 | 1.4394313908854883e-02 |
| ring | 1.0629441126600687e-03 | 1.2204765108491781e-03 | 4.008286309428726e-02 | 4.591330151259081e-02 |
| pinky | 2.007530841238008e-03 | 2.1449817249163962e-03 | 9.874947135967323e-02 | 1.056152629359259e-01 |
