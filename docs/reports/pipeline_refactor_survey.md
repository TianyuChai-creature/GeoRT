# 左手线：手无关产线工程化第一阶段普查

日期：2026-07-18  
范围：只读盘点；本文件是本阶段唯一新增文件。未修改 `geort/`、`data/`、`checkpoint/` 或既有产物。

目标产线的目标接口为：`hand_X.urdf + hts_X.npy` → FK 校验 → 目标点云 → 锚点与方向门 → C2b 双 seed 训练 → 评测与验收报告（数字、门判定、SHA manifest）。本表记录当前右手实现离该接口的差距，左手为第一个目标客户。

## 1. 前提输入

| 项目 | 路径 | SHA256 / 结构 |
|---|---|---|
| 左手 URDF | `assets/custom_left/URDF_L.urdf` | `96c948aacc22458449931b538467f2d776e053bc7c03f983287be7956052968b` |
| 左手 HTS | `data/hts_left.npy` | shape `(15527, 21, 3)`；`float32`；15,527 帧；3,912,804 B；SHA256 `bb2c7020f95cc4d3d9c57f0c4cec0eee5bafb0d75d7f8829919c2a89745be5ff` |
| 右手 HTS | `data/hts_right.npy` | shape `(234114, 21, 3)`；`float32`；234,114 帧；58,996,728 B；SHA256 `a9c783584db93110cadd546e2ab77aa2a5ac925554d8755840ed4c7d0cb96ca2` |

格式 diff：两者 rank 均为 3，布局均为 `[T, 21, 3]`，dtype 均为 `float32`；只有 `T`、文件字节数和内容哈希不同。`geort/config/custom_left.json:2-3` 已有 `custom_left` 名称及该 URDF 路径；与 `custom_right.json` 同为 20 DOF、五个 TIP（thumb→pinky）配置，但关节名、link 名和 offsets 是手侧专属值。

## 2. 硬编码普查

检索口径：仓内 `*.py`、`*.yaml`、`*.json` 的 `custom_right|hts_right|anchors_*right|contact_right|custom_left|hts_left`；下列为会绑定手名、输入路径或输出目录的命中。仅含测试夹具、历史归档或注释的项已标明，避免把 API 名中的 `right` 误计为产线依赖。

### 2.1 产线代码

| 文件:行号 | 原文（节录） | 绑定性质 |
|---|---|---|
| `configs/geort_equiv.yaml:2-3` | `hand: custom_right`; `human_data: data/hts_right.npy` | 基线 YAML 固定右手 |
| `geort/anchor/parity.py:85,88` | `--hand default="custom_right"`; `--report ...custom_right_fk_parity.json` | 默认手名、报告名 |
| `geort/anchor/mine_human_anchors.py:381-382` | `--input data/hts_right.npy`; `--hand-side ... default="right"` | 默认采集与侧别 |
| `geort/anchor/arc_bending_v2.py:1,152,184,243-255` | `Versioned custom_right...`; `human_data_source: data/hts_right.npy`; `get_config("custom_right")` | v2 锚点主脚本固定右手 |
| `geort/anchor/arc_bending_v2_runner.py:87,102-114` | `human_data_source: data/hts_right.npy`; 右手输入/输出默认；`get_config("custom_right")` | 同上，runner 变体 |
| `geort/anchor/arc_bending_v2_execute.py:1,77,88-97` | `Fixed executable for versioned custom_right...`; 右手路径；`get_config("custom_right")` | 同上，execute 变体 |
| `geort/anchor/arc_bending_v2_robust_execute.py:1,71,84-93` | `Build versioned custom_right...`; 右手路径；`get_config("custom_right")` | P2–P98 弧长版本固定右手 |
| `geort/anchor/lateral_shrink_execute.py:1,38-41,65,74` | `custom_right ... H/R = 0.85`; 右手 bundle/normalization；`get_config("custom_right")` | lateral085 专用拼接 |
| `geort/anchor/lateral_shrink_exact_execute.py:32-35,43,64` | 右手 human/paired/normalization/output；`get_config("custom_right")` | exact-knot 专用拼接 |
| `geort/anchor/ring_lateral_monotonic_execute.py:107-132` | `data/hts_right.npy`、右手 bundle 默认；`get_config("custom_right")`; metadata `human_data_source` 固定右手 | ringmono 专用重选 |
| `geort/anchor/ring_lateral_frozen_pair.py:17-19`; `_v2.py:16-18` | 三个 `anchors_*right*` 默认路径 | frozenrobot 专用拼接 |
| `geort/anchor/qa_report_runner.py:1,217,236-238,252,284` | `custom_right` QA；`if inputs.hand != "custom_right": raise` | QA 被显式锁右手 |
| `geort/anchor/qa_report.py:1` | `metrics shared by the custom_right anchor QA report` | QA 命名/说明右手化 |
| `geort/motion_frames.py:102-133` | `CUSTOM_RIGHT_TIP_FRAME_CONSTANTS`; `CUSTOM_RIGHT_DIP_AXIS_FLIPPED`; 默认 constants | local-T 常量只为右手存证 |
| `geort/mocap/build_target_cloud.py:560-564` | `--hand`, `--motion`, `--rest`, `--output` | **无手名硬编码**，已是可传入口 |
| `geort/mocap/hts_prepare_training.py:265` | `--input data/hts_right_20260703_quest3_v3.npy` | 历史处理入口默认右手 |
| `geort/mocap/hts_balance.py:254-256` | 右手 input/output/report 默认 | 历史平衡入口 |
| `geort/mocap/hts_stage3.py:186-190` | 右手 input/weights/report/manifest/dataset-id 默认 | 历史 stage3 入口 |
| `geort/mocap/hts_coverage.py:158-159` | 右手 input/output 默认 | 历史 coverage 入口 |
| `geort/mocap/measure_hts_aa_pose.py:10` | `from ...hts_right_mocap import ... iter_right_hts_points` | 右手几何读取器 |
| `geort/mocap/collect_hts_session.py:16` | `from ...hts_right_mocap import ...` | 采集层引用右手读取器 |
| `geort/mocap/verify_realtime_c2.py:41` | `--data ... data/hts_right.npy` | C2 验证默认右手 |
| `geort/mocap/visualize_tip_workspace.py:718-754` | `--hand custom_right`; `hts_right_train`; `outputs/...custom_right...` | 工作空间诊断默认右手 |
| `geort/mocap/search_custom_aa_limits.py:1,1462-1494` | `custom_right`; `hts_right_train`; `outputs/...custom_right...` | AA 限位搜索专用右手 |
| `geort/mocap/calibrate_custom_aa_limits.py:1,261-263` | `custom_right`; `docs/custom_right_aa_limit_calibration.md` | AA 标定默认右手 |
| `geort/contact/auto_label_contacts.py:400` | `--input data/hts_right.npy` | 接触标注默认右手 |
| `geort/contact/contact_model.py:266-275` | `--hand-side default="right"`、tag/output 参数 | 接触训练侧别默认右手 |
| `geort/contact/runtime.py:142` | `Current custom_right keeps physical limits ...` | runtime 兼容分支的右手语义 |

### 2.2 评测与报告代码

| 文件:行号 | 原文（节录） | 绑定性质 |
|---|---|---|
| `outputs/final_matrix/evaluate_final_matrix.py:18,53,114-119` | 右手 dataset/anchor/checkpoint manifest；写 `final_matrix.json/.md` | 终验矩阵脚本固定 C0–C3 右手目录 |
| `scripts/evaluate_c2el_s42.py:46,67-68` | `custom_right_with_rot.npz`; `get_config('custom_right')`; `data/hts_right.npy` | C2eL 正典评测固定右手 |
| `scripts/evaluate_anchor_pc1beta1_fix.py:123-138` | v2/v3 right bundle、C2b/C2eL checkpoint、right config/data 默认 | PC1 修复专项固定右手 |
| `geort/anchor/qa_report_runner.py:236-238` | `this QA report is intentionally restricted to custom_right` | 锚点报告不接受左手 |
| `geort/mocap/generate_realtime_checkpoint_ledger.py:70` | `checkpoint/custom_right_..._c2b_s42` 示例 | 账本示例右手化 |

### 2.3 Realtime

| 文件:行号 | 原文（节录） | 绑定性质 |
|---|---|---|
| `geort/mocap/hts_realtime_inference.py:22` | `from geort.mocap.hts_right_mocap import ...` | Quest 关键点读取器右手专用 |
| `geort/mocap/hts_realtime_inference.py:47` | `DEFAULT_C2B_S42_CHECKPOINT = "checkpoint/custom_right_..._c2b_s42"` | 默认生产权重右手 |
| `geort/mocap/hts_realtime_inference.py:649,652,655-656` | `--hand custom_right`；checkpoint/archive/session-root 默认 | 启动默认及归档目录固定右手基线 |
| `geort/mocap/hts_realtime_inference.py:723` | `checkpoint/contact_right_d1_full/contact_models.pth` | 接触模型路径固定右手 |

### 2.4 测试

| 文件:行号 | 原文（节录） | 性质 |
|---|---|---|
| `tests/test_anchor_analytic_fk.py:27,45`; `test_analytic_fk.py:25,45,123`; `test_analytic_fk_rotations.py:12,19`; `test_fk_direction_accuracy.py:13`; `test_trainer_device_contract.py:19` | `get_config("custom_right")` / `URDF_R.urdf` | 右手 FK fixture |
| `tests/test_anchor_qa_report_{runner,integration}.py:11-24` | right human data、right parity、right checkpoint/data | right QA integration fixture |
| `tests/test_anchor_runtime_loader.py:16,18`; `test_anchor_trainer_contract.py:31` | `human_data_source: data/hts_right.npy` | loader 契约 fixture |
| `tests/test_generate_robot_anchors.py:110`; `test_build_target_cloud.py:38`; `test_training_targets.py:41,46,53` | `custom_right` | 右手训练目标 fixture |
| `tests/test_hts_realtime_inference.py:27,31,96,118` | `hts_right_mocap` stub、C2b checkpoint、contact model | realtime 默认契约 fixture |
| `tests/test_auto_label_contacts.py:117`; `test_contact_runtime.py:72` | right HTS/config | contact fixture |
| `tests/test_mine_human_anchors.py:67-174` | right output fixture，另含 left CLI case | anchor CLI 默认与左右侧行为测试 |
| `tests/test_search_custom_aa_limits.py:51,61-67`; `test_custom_right_urdf_limits.py:20-36` | left/right AA/MCP2 fixtures | 两侧 URDF 既有验收 |
| `tests/test_collect_hts_session.py:37-39` | `hts_right_*` 录制名 | 历史采集命名 fixture |

### 2.5 死码／历史归档

| 文件:行号 | 原文（节录） | 归类依据 |
|---|---|---|
| `scripts/archive/tmp_evaluations/evaluate_c2be_motion.py:33,38,42,102-103` | right config/data/rot、`/tmp` 评测路径 | 明确位于 `scripts/archive`，非当前正典入口 |
| `geort/anchor/arc_bending_v2_execute.py`、`arc_bending_v2_runner.py`、`arc_bending_v2_robust_execute.py` | 三个版本化右手可执行入口 | 版本试产脚本；当前定稿 v3 流程并未由单一编排器调用，先列为待整合的历史专项，不删除 |
| `geort/mocap/hts_balance.py`、`hts_stage3.py`、`hts_coverage.py` | 右手会话版本文件名 | 历史数据整备路径；未接入当前 C2b 正典训练启动器 |

## 3. 七工位对号入座

| 工位 | 现有入口 | 当前参数化程度 | 改造点 |
|---|---|---|---|
| intake | `geort/config/custom_left.json`、`custom_right.json`；`get_config()` | hand config 可按名称读；URDF 路径藏于 config；HTS 由各脚本各自默认 | 定义一个 hand manifest，显式给 `hand_id/urdf/hts/config/output_root`，不再由脚本猜命名。 |
| FK parity | `python -m geort.anchor.parity --hand ... --human-anchors ...` | hand/human anchors/parity file 可传；默认 hand/report 右手 | 把输出目录、TIP offsets、阈值和 750 qpos 生成统一纳入 manifest；去除右手默认名。 |
| 点云 | `python -m geort.mocap.build_target_cloud --hand --motion --rest --output` | CLI 已可传全部主要输入 | 给它上游标准化 HTS 和固定 seed/provenance wrapper；当前 motion/rest 二输入与目标“一份 hts_X.npy”接口不一致。 |
| 锚点＋方向门 | `mine_human_anchors.py`；`generate_robot_anchors.py`；arc/lateral/ring 版本脚本；`qa_report_runner.py` | 通用库函数、robot generator 可传 `--hand`；试产 execute 默认路径与 `get_config("custom_right")` 固定；QA 明确拒绝非 right | 合成一个 manifest 驱动的单入口；将 PC1 β1 对齐、lateral H/R、ring 单调、parity 与 QA 变成可配置 gate。 |
| 训练 | `python -m geort.trainer -hand ... -human_data ...`；`outputs/c2_variants/.../run_four.sh` | 大部分 CLI/YAML 可传；C2b launcher 固定右手 data/bundle/tag/output | 提供 `recipe=c2b` 的手无关 launcher，派生 seed42/123、明确 run root、保存 resolved config+input SHA。 |
| 评测＋报告 | `outputs/final_matrix/evaluate_final_matrix.py`；`scripts/evaluate_c2el_s42.py`；`qa_report_runner.py` | 各专项脚本各有固定 right 数据、checkpoint 名和输出；部分能传 checkpoint | 统一 evaluator CLI 与 gate spec：按 hand manifest 找数据和 bundle，写单一 `metrics.json`、`report.md`、`sha_manifest.json`。 |
| 接触（可选） | `geort.contact.auto_label_contacts`、`geort.contact.contact_model`、`geort.contact.runtime` | label/train 有 input/side CLI；realtime model 默认固定 right path | 作为 optional station：按 hand output root 产 labels/models，runtime 从 manifest 加载而非硬编码 checkpoint。 |

## 4. 参数普查

### 4.1 锚点筛选与方向门

| 参数 | 当前值 | 存放位置 |
|---|---:|---|
| 稀疏等级 | `[0, 0.25, 0.5, 0.75, 1]` | `geort/anchor/mining.py:22`，`LEVEL_FRACTIONS` |
| 非拇指 bending 参数化 | TIP-PC1 弧长分数；PC1 域 P2–P98 | `arc_bending_v2_robust.py:35-101`；metadata `non_thumb_arc_domain` |
| PC1 方向约定 | projection 与 β1 相关性为负则翻转；方向为 β1 increasing | `arc_bending_v2_robust.py:15-27,65,100-101` |
| 弧线 bin 数 | 64 | `arc_bending_v2_robust.py:38,68` |
| level band | 0.025 | `arc_bending_v2_robust.py:41,88-90` |
| support 下限 | 5 | `arc_bending_v2_robust.py:39,88-90` |
| candidate 上限 | 256 | `arc_bending_v2_robust.py:40,88-90` |
| band 逐级放宽 | `[1, 2, 4, 8]` | `arc_bending_v2_robust.py:42,89-90` |
| lateral robot 收缩 | 非拇指目标 H/R = 0.85；拇指冻结 | `lateral_shrink_execute.py:1`、`ring_lateral_monotonic_execute.py:107-132`；目前路径/手名写死 |
| ring lateral 单调方向 | `dot(projection-mean, alpha-mean) < 0` 时翻转 | `ring_lateral_monotonic_execute.py:55` |
| 插值数 | lateral `K=50`；bending `K=100`；总 750 对 | `anchor_spec.py`/`interpolate.py` 调用与 `ring_lateral_monotonic_execute.py:132` metadata |
| FK parity gate | max `< 1e-3 m` | `geort/anchor/parity.py:60,74,78` |
| QA gate | 方向内积正、重复阈值 `1e-10`、H/R span 比 `<=3.0` | `geort/anchor/qa_report_runner.py` 与 `geort/anchor/qa_report.py` |

### 4.2 C2b 训练配方（当前归档启动器）

来源：`outputs/c2_variants/rerun_20260717T110336/run_four.sh:13-18`。共同项与 C2b 增量如下；seed 为 42/123，tag 为 `c2b_s42`/`c2b_s123`。

| 项 | 值 |
|---|---|
| hand / human data | `custom_right` / `data/hts_right.npy` |
| FK / chamfer | `analytic` / `bidirectional` |
| `w_chamfer`, `w_distance`, `w_curvature`, `w_motion` | `80.0`, `1.0`, `0.1`, `1.0` |
| motion | `motion_frame=global`, `motion_delta=0.01` |
| anchor | `data/anchors_custom_right_arc_bending_v2_lateral085_ringmono_frozenrobot.npz`; `w_anchor=1.0` |
| pinch / collision / MCP1 | `w_pinch=1.0`; `w_collision=0`; `w_mcp1_fist_prior=0` |
| nullspace / synergy | `nullspace_weight=0`; `nullspace_subsample=0`; `synergy_weight=0`; `synergy_lambda=2.0` |
| optimization | batch `2048`; lr `1e-4`; epoch `200`; `max_steps=0`; device `cuda` |
| run behavior | `contact_refine=off`; `chamfer_target=uniform`; `save_every=0`; `no_update_latest`; `run_git_commit=4827adb` |

Trainer 通用 CLI 的默认位置：`geort/trainer.py:495-545,1008-1095`。其中 `anchor_batch_size=32` 目前是训练内部常量（`geort/trainer.py:529`），不是 CLI/config。`configs/geort_equiv.yaml:2-3` 还把 equivalence recipe 固定为 custom_right/hts_right。

### 4.3 评测阈值／固定口径

| 项 | 当前值 | 存放位置 |
|---|---:|---|
| D1 采样 | `RandomState(42)`、1000 帧 | `outputs/final_matrix/evaluate_final_matrix.py`；`scripts/evaluate_c2el_s42.py:68` |
| 开手度十桶边界 | `[0.074412, 0.170439]` | `scripts/evaluate_c2el_s42.py` 的正典评测实现与归档 JSON |
| 云形状采样 | D1 至多 50k，随机 10k 点对 | `scripts/evaluate_c2el_s42.py` |
| anchor 残差 | 750 对，mean/max，米制 hand-base | 正典 evaluator 及 anchor bundle metadata |
| local LMC | 统一 local-T evaluator（而非 checkpoint `motion_frame`） | `outputs/final_matrix/supplement/run_supplement.py` / `supplement.json` |
| checksum 比对 | input SHA、checkpoint SHA、git chain | `outputs/final_matrix/checkpoint_hashes.json`、`effective_configs.json`、`ledger.json` |

现状：上述“阈值”大多是专项评测脚本里的固定协议或报告中已写死的门，未集中成可供 left/right 共用的 machine-readable gate spec。

## 5. 报告工位现状与缺口

已有归档：`outputs/final_matrix/final_matrix.json` + `final_matrix.md`、`effective_configs.json`、`checkpoint_hashes.json`、`ledger.json`、`supplement/supplement.{json,md}`；专项结果在 `docs/reports/*_canonical_evaluation.{json,md}`、`anchor_pc1beta1_fix*`、`deviations.md`。现有格式能保存数字、命令、配置和部分 SHA，但其输入与实验标签均由 right-specific 脚本固定。

| 自动验收报告能力 | 当前状态 | 缺口 |
|---|---|---|
| 数字汇总 | final matrix / canonical evaluator 可写 JSON+MD | 没有按 `hand_id` 和 recipe 自动发现 seed42/123 产物的汇总器。 |
| 门判定逻辑 | FK parity、QA runner 和各任务书里各自存在 | 没有单一 declarative gate 表，也没有将 gate 结果写进统一报告。 |
| SHA manifest | final_matrix 已有 target/anchor/checkpoint 哈希与 ledger | 未覆盖 URDF、HTS、normalization、contact model、anchor 中间物；没有通用输入清单。 |
| provenance | command/config/log 分散在 `outputs/final_matrix/` 和 checkpoint metadata | 没有一个 hand-specific output root 下的原子 manifest。 |
| 人类可读报告 | 多份 MD 专项报告 | 缺手无关模板（输入、各工位输出、门、数值、SHA、失败原因）。 |

## 6. 第一阶段结论性造册（不执行改造）

当前最接近手无关的组成件是 config loader、analytic FK、`build_target_cloud.py`、`generate_robot_anchors.py`、trainer 的 `-hand/-human_data` CLI，以及 contact 的 label/train CLI。阻塞左手一键运行的不是左手 URDF 或 HTS 格式，而是：(1) v2/v3 anchor 试产链和 QA runner 把 right 路径/`get_config("custom_right")`/元数据写死；(2) C2b launcher 与终验脚本将 bundle、data、checkpoint tag 和 output root 固定；(3) local-T 只存 `CUSTOM_RIGHT_*` 常量；(4) realtime/接触默认路径固定右手；(5) 门规则和 SHA 记录未被统一成 manifest 驱动的报告工位。

本文件不包含任何改动提案的实现，不改变现有右手产线行为。
