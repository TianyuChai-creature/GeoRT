# Custom-right D1 contact classifier archive

Source: `data/hts_right.npy` (right hand, metric hand-base landmarks). Labels:
`data/contact_labels_right.npz`. Models:
`checkpoint/contact_right_d1_full/contact_models.pth`.

The four independent BCE MLPs use the archived Step-0 configuration: hidden
dimensions `64, 32`, 20 epochs, learning rate `1e-4`, batch size `2048`, seed
`0`, CPU. The temporal held-out split is the final 20 percent of clear frames.

| pair | positive | negative | ambiguous | positive threshold (m) | held-out | AUC | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| thumb_index | 20,422 | 202,316 | 11,376 | 0.015 | 45,311 | 0.999835 | 0.725543 | 1.000000 |
| thumb_middle | 6,936 | 219,252 | 7,926 | 0.011 | 45,735 | 0.998705 | 0.686963 | 1.000000 |
| thumb_ring | 7,676 | 221,170 | 5,268 | 0.016 | 45,738 | 0.999549 | 0.675676 | 1.000000 |
| thumb_pinky | 5,184 | 224,418 | 4,512 | 0.009 | 45,723 | 0.998690 | 0.511161 | 1.000000 |

Runtime contract: features are exactly `[thumb_tip_xyz, finger_tip_xyz]` from
the raw `[21, 3]` metric hand-base frame. The v1 runtime selects only the
highest-probability active pair; other pairs above `p_lo` are recorded in the
per-frame `ContactSelection.ignored_pair_names` deviation field. It blends
physical qpos using `clip((p-p_lo)/(p_hi-p_lo), 0, 1)` and runs a fixed 40-step
CPU projected-Adam analytic-FK solve over the selected thumb and finger's
eight DOF. This report will be supplemented by runtime acceptance measurements.


## Runtime acceptance archive

Runtime acceptance uses `checkpoint/custom_right_last` to produce `q_map`, the
Step-0 contact checkpoint above, and CPU-only classification/refinement with
`p_lo=0.5`, `p_hi=0.8`, target distance `0 m`, regularisation `0.1`, and 40
fixed projected-Adam iterations. The complete machine-readable measurements
are in [contact_runtime_acceptance.json](contact_runtime_acceptance.json).

| measurement | value |
| --- | ---: |
| held-out selected-pair positive frames | 6,635 |
| distance before, mean / p95 (m) | 0.01585003 / 0.03369935 |
| distance after, mean / p95 (m) | 0.01459729 / 0.03183056 |
| absolute q_out-q_map, max / mean (deg) | 1.21592391 / 0.08832215 |
| deterministic repeated output | `array_equal=True` |
| CPU classification / refinement (ms per frame) | 0.08821663 / 57.28298920 |
| all-D1 frames / active frames | 234,114 / 66,534 |
| all-D1 qpos bound violations | 0 |
| D1 first contiguous 1,000 frames: NaN count | 0 |
| D1 first contiguous 1,000 frames: trigger duty index/middle/ring/pinky | 0 / 0 / 0 / 0 |

