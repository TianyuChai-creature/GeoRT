# C2eL_s42 训练前配置差异

| 项目 | c2e_s42 | C2eL_s42 |
|---|---|---|
| hand / human data | `custom_right` / `data/hts_right.npy` | `custom_right` / `data/hts_right.npy` |
| seed / epoch / batch / lr | `42` / `200` / `2048` / `1e-4` | `42` / `200` / `2048` / `1e-4` |
| FK / chamfer mode / target | `analytic` / `partial` / `uniform` | `analytic` / `partial` / `uniform` |
| w_chamfer / w_distance / w_curvature | `1.0` / `1.0` / `0.1` | `1.0` / `1.0` / `0.1` |
| w_motion / motion_delta | `1.0` / `0.01` | `1.0` / `0.01` |
| w_pinch / w_anchor | `1.0` / `1.0` | `1.0` / `1.0` |
| nullspace / synergy / MCP1 / collision | `0` / `0` / `0` / `0` | `0` / `0` / `0` / `0` |
| anchor bundle | `data/anchors_custom_right_arc_bending_v2_lateral085_ringmono_frozenrobot.npz`, `f291cfc39c97bdda9e50bb670c7c14967428d3b9cbf7d83d9800fb43de51ed7e` | same |
| motion_frame | `global` | `local` |
| target cloud | `data/custom_right.npz`, `b978674ddec119cd0006b2ba6fb5559962317d8fe5c61be68830f0f2563208bc` | `data/custom_right_with_rot.npz`, `64eef596f901af267022c19f885a51132f6b3ab326f5612d584c0fe61e328239` |

目标点云文件 `custom_right.npz` → `custom_right_with_rot.npz`（SHA 变更），共享字段逐位等价（见 `c2el_rotation_variant_precheck.json`），仅增 `link_rotation`；这是 `motion_frame=local` 的技术性依赖。

单因子声明：除 `motion_frame` 及其技术依赖字段外零 diff。
