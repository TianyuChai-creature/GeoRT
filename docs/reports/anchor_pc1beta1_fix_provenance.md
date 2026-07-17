# PC1 beta1 anchor fix provenance

```json
{
  "archival_code_commit": "3e6027fccc2f6493b441db78aeee378d8cc8eaf0",
  "final_bundle": {
    "path": "data/anchors_custom_right_arc_bending_v3_pc1beta1_lateral085_ringmono_frozenrobot.npz",
    "sha256": "3bf9a190cac67cd54a6dd270adc769da01fd621218fab19c51947640bb0ca6ba"
  },
  "final_human_sparse": {
    "path": "data/anchors_human_right_arc_bending_v3_pc1beta1_ringmono.npz",
    "sha256": "df2289c4b44f9e4f1435eb7c37f644a1d3d3d85af19bf62ee47c8d929509f1ec"
  },
  "generation_commands": [
    "/home/creature/Desktop/GeoRT/.venv/bin/python -m geort.anchor.arc_bending_v2_fast_execute --input data/hts_right.npy --legacy-human data/anchors_human_right.npz --legacy-parity outputs/anchors/parity_qpos.npz --human-output data/anchors_human_right_arc_bending_v3_pc1beta1.npz --paired-output data/anchors_custom_right_arc_bending_v3_pc1beta1.npz",
    "/home/creature/Desktop/GeoRT/.venv/bin/python -m geort.anchor.lateral_shrink_exact_execute --human data/anchors_human_right_arc_bending_v3_pc1beta1.npz --paired data/anchors_custom_right_arc_bending_v3_pc1beta1.npz --output data/anchors_custom_right_arc_bending_v3_pc1beta1_lateral085_exactknots.npz",
    "/home/creature/Desktop/GeoRT/.venv/bin/python -m geort.anchor.ring_lateral_monotonic_execute --input data/hts_right.npy --human data/anchors_human_right_arc_bending_v3_pc1beta1.npz --paired data/anchors_custom_right_arc_bending_v3_pc1beta1_lateral085_exactknots.npz --human-output data/anchors_human_right_arc_bending_v3_pc1beta1_ringmono.npz --paired-output data/anchors_custom_right_arc_bending_v3_pc1beta1_lateral085_ringmono.npz",
    "/home/creature/Desktop/GeoRT/.venv/bin/python -m geort.anchor.ring_lateral_frozen_pair_v2 --human data/anchors_human_right_arc_bending_v3_pc1beta1_ringmono.npz --parent data/anchors_custom_right_arc_bending_v3_pc1beta1_lateral085_exactknots.npz --output data/anchors_custom_right_arc_bending_v3_pc1beta1_lateral085_ringmono_frozenrobot.npz"
  ],
  "generation_source_tree": {
    "head_before_generation": "9d9be9e",
    "working_tree_delta": "PC1-to-beta1 orientation patch, archived without further source change as 3e6027f"
  },
  "inputs": {
    "data/anchors_custom_right_arc_bending_v2_lateral085_ringmono_frozenrobot.npz": "f291cfc39c97bdda9e50bb670c7c14967428d3b9cbf7d83d9800fb43de51ed7e",
    "data/anchors_human_right.npz": "de1eacb4074b87c2e3c9b8c41790e74014414f8ad4a1495384e41bc507ec44a5",
    "data/hts_right.npy": "a9c783584db93110cadd546e2ab77aa2a5ac925554d8755840ed4c7d0cb96ca2",
    "outputs/anchors/parity_qpos.npz": "e0631deecf5c1a700a97a57a621eee7da5638ed00c9abbca4a399a8af1532628"
  },
  "schema_version": 1
}
```
