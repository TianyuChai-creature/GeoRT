# AnyDexRT 产物剪枝清单（2026-07-18）

本清单记录复现报告定稿后的三级处置。清单提交与实际删除分离；本文件不移动、覆盖或删除任何产物。

## 永久保留（一级）

- `checkpoint/custom_right_2026-07-17_12-21-39_c2b_s42/`：生产 C2b s42；`last.pth` SHA256 `dc9c2cc36e20bffe28736ec6111b4401631ee683c6021afe7c816768a4743e73`。
- `data/anchors_custom_right_arc_bending_v3_pc1beta1_lateral085_ringmono_frozenrobot.npz`：生产锚点；SHA256 `3bf9a190cac67cd54a6dd270adc769da01fd621218fab19c51947640bb0ca6ba`。
- `data/custom_right.npz`：SHA256 `b978674ddec119cd0006b2ba6fb5559962317d8fe5c61be68830f0f2563208bc`。
- `data/custom_right_with_rot.npz`：SHA256 `64eef596f901af267022c19f885a51132f6b3ab326f5612d584c0fe61e328239`。
- `data/hts_right.npy`、`data/hts_left.npy`、`data/contact_labels_right.npz`。
- `checkpoint/contact_right_d1_full/`（在产接触分类器，禁动）。
- `assets/custom_right/URDF_R.urdf`、`geort/config/custom_right.json`。

## 案证保留（二级）

### 批准脱水：仅删除 `last.pth`

- `checkpoint/custom_right_2026-07-16_19-38-14_c0_s42/last.pth`
- `checkpoint/custom_right_2026-07-16_20-16-08_c0_s123/last.pth`
- `checkpoint/custom_right_2026-07-16_20-53-54_c1_s42/last.pth`
- `checkpoint/custom_right_2026-07-16_21-29-07_c1_s123/last.pth`
- `checkpoint/custom_right_2026-07-16_22-04-19_c2_s42/last.pth`
- `checkpoint/custom_right_2026-07-16_22-41-48_c2_s123/last.pth`
- `checkpoint/custom_right_2026-07-17_11-03-40_c2e_s42/last.pth`
- `checkpoint/custom_right_2026-07-17_11-42-52_c2e_s123/last.pth`
- `checkpoint/custom_right_2026-07-17_13-04-50_c2b_s123/last.pth`

`C2eLf_s42`、`C2eL_s42`、`c3_s42`、`c3_s123` 的 `last.pth` 因补件 A 未过庭，本轮不动。

### 保留，不脱水

- `data/anchors_custom_right_arc_bending_v2_lateral085_ringmono_frozenrobot.npz`：毒 v2 bundle，禁删。
- `outputs/final_matrix/`、`outputs/c2_variants/rerun_20260717T110336/`、`outputs/c2_variants/motion_consistency_eval/`、`outputs/c2elf_s42/`。
- 所有 `outputs/realtime_sessions/` 非 headless/bypass 会话。
- `outputs/anchors/formal_anchor_*.log`、`outputs/anchors/` 内 QA Markdown/日志、`outputs/anchors/parity_qpos.npz`、`outputs/anchors/custom_right_fk_parity.json`。
- `checkpoint/fk_model_custom_right.pth`、`checkpoint/revalidation_logs/`。
- `outputs/contacts/` 内 `.md` 与 `.json` 文本报告。

## 批准删除（三级）

### 训练目录与中断残骸

- `checkpoint/custom_right_2026-07-15_14-31-08_anydexrt_analytic/`
- `checkpoint/custom_right_2026-07-15_14-33-00_anydexrt_neural/`
- `checkpoint/custom_right_2026-07-15_15-53-13_pca_synergy/`
- `checkpoint/custom_right_2026-07-15_16-44-15_nullspace/`
- `checkpoint/custom_right_2026-07-16_09-33-37_seed42_null_v2_invalid_subsample256/`
- `checkpoint/custom_right_2026-07-16_09-33-37_seed123_null_v2_invalid_subsample256/`
- `checkpoint/custom_right_2026-07-16_14-34-21_anchor_formal_s42/`
- `checkpoint/custom_right_2026-07-16_14-34-21_anchor_formal_s123/`
- `checkpoint/custom_right_2026-07-16_14-13-35_anchor_smoke_baseline/`
- `checkpoint/custom_right_2026-07-16_14-14-12_anchor_smoke_disabled/`
- `checkpoint/custom_right_2026-07-16_14-15-16_anchor_smoke_enabled/`
- `checkpoint/custom_right_2026-07-16_16-38-33_device_repro/`
- `checkpoint/custom_right_2026-07-16_16-40-43_device_repro_shared/`
- `checkpoint/custom_right_2026-07-16_16-42-01_device_cpu_smoke/`
- `checkpoint/custom_right_2026-07-16_16-44-17_device_repro_final/`
- `checkpoint/custom_right_2026-07-16_17-05-38_motion_global_repro/`
- `checkpoint/custom_right_2026-07-16_17-14-01_motion_local_cpu_smoke/`
- `checkpoint/custom_right_2026-07-16_17-14-32_motion_local_cpu_smoke2/`
- `checkpoint/custom_right_2026-07-16_19-10-36_parta_current_default/`
- `checkpoint/custom_right_2026-07-16_19-14-37_parta_probe_current/`
- `checkpoint/custom_right_2026-07-16_19-17-09_parta_current_default_fixed/`
- `checkpoint/custom_right_2026-07-16_19-19-02_parta_bidir_smoke/`
- `checkpoint/custom_right_2026-07-16_19-23-21_parta_current_default_a47bffa/`
- `checkpoint/custom_right_2026-07-16_19-26-17_parta_config_dump/`
- `checkpoint/custom_right_last/`（权重与早期 analytic 相同）。
- `checkpoint/interrupted/`、`outputs/c2_variants/interrupted/`、`outputs/c2el_s42_train.log`。

### Anchor 中间产物与可再生可视化

- `data/anchors_human_right*.npz`
- `data/anchors_custom_right_arc_bending_v2*.npz`，但明确排除 `data/anchors_custom_right_arc_bending_v2_lateral085_ringmono_frozenrobot.npz`。
- `data/anchors_custom_right_arc_bending_v3_pc1beta1.npz`
- `data/anchors_custom_right_arc_bending_v3_pc1beta1_lateral085_exactknots.npz`
- `data/anchors_custom_right_arc_bending_v3_pc1beta1_lateral085_ringmono.npz`
- `outputs/anchors/anchors_human_right.json`、`outputs/anchors/anchors_human_right.html`
- `outputs/contacts/contact_labels_right.html`

### 作废 realtime/缓存产物

- `outputs/realtime_sessions/20260717T064547Z/`
- `outputs/realtime_sessions/20260717T070001Z/`
- `outputs/realtime_sessions/20260717T072906Z/`
- `outputs/realtime_sessions/20260717T075120Z/`
- `outputs/realtime_sessions/20260717T075250Z/`
- `outputs/realtime_sessions/20260717T081132Z/`
- `outputs/realtime_sessions/20260717T082430Z/`
- `outputs/realtime_sessions/20260717T082930Z/`
- `outputs/final_matrix/__pycache__/`、`outputs/final_matrix/supplement/__pycache__/`

bypass session（含 `20260717T075120Z`）关键读数 `mapped` 超限位 p95 `0.452 rad` 已录入复现报告观察项；其 session 载体随三级删除。

## 补件 A：四个 local-T checkpoint 的大文件审计

四目录中的 `human_motion_frames.npy` 均为 42,140,648 B、SHA256 `823618100781362d7f88bc2df10ce753f308367e1ed377147f5df3d5f26ff0ec`；它们彼此逐位相同。

`data/custom_right_with_rot.npz` 为 40,001,728 B、SHA256 `64eef596f901af267022c19f885a51132f6b3ab326f5612d584c0fe61e328239`。二者 SHA 与大小均不同，非同源逐位副本。`human_motion_frames.npy` 的文件名表明其为训练侧预计算人手 motion frame；本轮列为待裁，不删除。

受此补件保护、不在本轮脱水的目录：

- `checkpoint/custom_right_2026-07-17_19-45-02_C2eLf_s42/`
- `checkpoint/custom_right_2026-07-17_17-47-22_C2eL_s42/`
- `checkpoint/custom_right_2026-07-16_23-19-19_c3_s42/`
- `checkpoint/custom_right_2026-07-16_23-58-23_c3_s123/`

## 补件 B：2026-07-18 四个 seed 目录来历

- `checkpoint/custom_right_2026-07-18_13-37-32_seed42_syn0/`
- `checkpoint/custom_right_2026-07-18_13-37-32_seed123_syn0/`
- `checkpoint/custom_right_2026-07-18_13-37-32_seed42_null/`
- `checkpoint/custom_right_2026-07-18_13-37-32_seed123_null/`

四目录的 `training_metadata.json` 记录 tag 与 `nullspace_weight`（syn0 为 `0.0`；null 为 `0.01`），但 `launch_command`、`run_git_commit` 与 seed 均为 `null`。全仓测试搜索没有这四个 tag 或相应 checkpoint 创建调用，未找到其为 pytest 副产物的证据。四目录保持待裁，本轮不动。
