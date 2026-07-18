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

四目录的 `training_metadata.json` 记录 tag 与 `nullspace_weight`（syn0 为 `0.0`；null 为 `0.01`），但 `launch_command`、`run_git_commit` 与 seed 均为 `null`。其来源由下方执行回执中的 `tests/test_deliverables.py:175-211` 顶层训练循环定案；四目录保持待裁，本轮不动。


## 执行回执（第二提交）

### 补件 A：逐文件清单

| 目录 | 文件 | 字节 | SHA256 |
|---|---|---:|---|
| `C2eLf_s42` | `config.json` | 6,496 | `1fb4983f2910cdc2dadcda9a4a359c7ef0c5b4ce80ad7092e6974f25dee4f5fa` |
| `C2eLf_s42` | `human_motion_frames.npy` | 42,140,648 | `823618100781362d7f88bc2df10ce753f308367e1ed377147f5df3d5f26ff0ec` |
| `C2eLf_s42` | `last.pth` | 392,945 | `6ad8a4f273486ceb20c5159a80bf964bd053219c782f52bd9a958b2d24be60f3` |
| `C2eLf_s42` | `normalization.json` | 2,866 | `c242685a73815c1b29121bdbd65cdc470e5477dd8b3dab07815520a4c41e3e6a` |
| `C2eLf_s42` | `training_metadata.json` | 4,027 | `f9ae02f74c399f1c83c59e0a821c80291c447ffa9827958aeb38e78de6a5cb3e` |
| `C2eL_s42` | `config.json` | 6,496 | `1fb4983f2910cdc2dadcda9a4a359c7ef0c5b4ce80ad7092e6974f25dee4f5fa` |
| `C2eL_s42` | `human_motion_frames.npy` | 42,140,648 | `823618100781362d7f88bc2df10ce753f308367e1ed377147f5df3d5f26ff0ec` |
| `C2eL_s42` | `last.pth` | 392,945 | `ec1641b642e7855549437aa8ac3bbd330c227329173ade26c64332a70bced59f` |
| `C2eL_s42` | `normalization.json` | 2,866 | `c242685a73815c1b29121bdbd65cdc470e5477dd8b3dab07815520a4c41e3e6a` |
| `C2eL_s42` | `training_metadata.json` | 4,023 | `ad24ca420c267ea619f55b261301ad750f14df103422d8114e9f59d4e27b2afd` |
| `c3_s42` | `config.json` | 6,496 | `1fb4983f2910cdc2dadcda9a4a359c7ef0c5b4ce80ad7092e6974f25dee4f5fa` |
| `c3_s42` | `human_motion_frames.npy` | 42,140,648 | `823618100781362d7f88bc2df10ce753f308367e1ed377147f5df3d5f26ff0ec` |
| `c3_s42` | `last.pth` | 392,945 | `f502ebb3702f9c06e7084dfa84b8de9210e59b1b86dc5afa03444be8cc63f981` |
| `c3_s42` | `normalization.json` | 2,866 | `c242685a73815c1b29121bdbd65cdc470e5477dd8b3dab07815520a4c41e3e6a` |
| `c3_s42` | `training_metadata.json` | 3,775 | `e73c8644c32d02816a739bcb2045f8256531175c27130cc57a68cfd0bac6ece1` |
| `c3_s123` | `config.json` | 6,496 | `1fb4983f2910cdc2dadcda9a4a359c7ef0c5b4ce80ad7092e6974f25dee4f5fa` |
| `c3_s123` | `human_motion_frames.npy` | 42,140,648 | `823618100781362d7f88bc2df10ce753f308367e1ed377147f5df3d5f26ff0ec` |
| `c3_s123` | `last.pth` | 392,945 | `d910a1071eee0178e8ad6f9b9e734bb71e57d3af240d430a73a4c917af1e5c5f` |
| `c3_s123` | `normalization.json` | 2,866 | `c242685a73815c1b29121bdbd65cdc470e5477dd8b3dab07815520a4c41e3e6a` |
| `c3_s123` | `training_metadata.json` | 3,780 | `be62027ffffe7a1701099ebad7ffdc98f85616f78b4fd35391c6e6dc78a95ca9` |

四份 `human_motion_frames.npy` 是相同的 `float32 (234114, 5, 3, 3)` 数组；与 `data/custom_right_with_rot.npz` 的 SHA、体积和文件格式均不同。本轮未删除这四个数组及所在目录的 `last.pth`。

### 补件 B：来源定案

`tests/test_deliverables.py:175-211` 在模块顶层循环 `seed in [42, 123]` 与 `variant in [("syn0", 0.0, 0.0), ("null", 0.01, 0.0)]`，通过 `subprocess.Popen(..., shell=True)` 启动 `GeoRTTrainer.train(..., tag='seed{seed}_{variant}', epoch=200)`。这与四个 `2026-07-18_13-37-32_seed{42,123}_{syn0,null}` 目录的 tag、权重和时间相符；它们是 `tests/test_deliverables.py` 的 pytest 副产物。

### 已执行处置与删后核验

- 已删除批准的三级目录/文件，以及未受补件 A 保护的九个二级 `last.pth`；逻辑文件字节回收量为 `169,024,079 B`。
- 已复核一级和升一级的 C2b、v3 bundle、两份目标点云、`hts_right.npy`、`hts_left.npy`、`contact_labels_right.npz`、contact MLP、URDF、config 均在位；已知基准 SHA 与清单逐位一致。
- 已复核补件 A 四目录与补件 B 四目录仍在位。
- pytest 命令：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest`。结果：收集 `260 items / 1 error`，退出码 `2`；`tests/test_deliverables.py:63` 读取已按本清单删除的 `checkpoint/custom_right_last/last.pth`。该测试还在顶层启动四个后台训练，属于补件 B 所确认的副产物来源；本轮未改测试或恢复 checkpoint。


## 第三阶段授权与预核算

### `test_deliverables.py` 处置

退役 `tests/test_deliverables.py`。案底：文件由 `af6b96f` 引入，是无 pytest 断言的交付冒烟脚本；在模块顶层读取已淘汰的 `custom_right_last`/早期 nullspace checkpoint，并启动四个 200 epoch 后台训练。其每次运行先删除同 tag 目录再重建，因此仅保留最近一次（13:37）的四组副产物。测试收集期不得产生训练或 checkpoint 写入。

### 已批准本批删除

- 四个 local-T checkpoint 中的 `human_motion_frames.npy` 与 `last.pth`；保留各自 `config.json`、`normalization.json`、`training_metadata.json`。
- `checkpoint/custom_right_2026-07-18_13-37-32_seed{42,123}_{syn0,null}/` 四目录。
- `checkpoint/custom_right_2026-07-16_10-08-30_seed42_null_v3_full/last.pth` 与 `seed123_null_v3_full/last.pth`；保留各自 `config.json`。
- `tests/test_deliverables.py`。

本批删除前逻辑字节核算：`172,544,647 B`。其中四份 `human_motion_frames.npy` 合计 `168,562,592 B`；四个 13:37 pytest 副产物合计 `1,614,940 B`。与第二阶段 `169,024,079 B` 合计预计逻辑回收 `341,568,726 B`。
