# AnyDexRT 改造执行手册（Codex 友好版）

<aside>
📋

用法：每个 Step 作为一个独立的 Codex 任务（一个 session / 一个 PR），按顺序执行，通过验收后再进入下一步。标注【人工】的事项由操作者完成，其余全部交给 Codex。仓库：[TianyuChai-creature/GeoRT](https://github.com/TianyuChai-creature/GeoRT)，工作分支 `anydexrt`，`main` 冻结为基线。

</aside>

# 人工任务总览（共 5 项，其余全自动）

| # | 人工任务 | 发生在 | 耗时 |
| --- | --- | --- | --- |
| H1 | 确认 main 基线 checkpoint 与评测报告已存档 | Step 0 | 10 分钟 |
| H2 | 目视检查人手/机器人局部坐标系可视化方向一致 | Step 4 | 5 分钟 |
| H3 | D2 锚点采集（50 次记录，按主计划 2.4 手册） | Step 6 | 10–15 分钟 |
| H4 | D3 接触标签采集（4 指对正负段，按 2.5 手册） | Step 7 | 10–15 分钟 |
| H5 | 端到端遥操作实测主观评估 | Step 8 | 30 分钟 |

---

# Step 0 · 建分支 + 基线留底（已完成）

**Codex 做什么**

- [x]  从 `main` 建分支 `anydexrt`。
- [x]  初始化分支 `data/`（从空开始）：拷入已有 raw `.npy` 作 D1（✅ 已决策：直接复用，不重录；左、右手 raw 均可）；**不拷** `*_train.npy`、`*_train.json`、`*_humanshaped.npz`（全部作废）。完整数据清单见主计划 2.0。
- [x]  在 `main` 上确认/补跑基线：用现有 raw HTS 数据训练一版旧管线 checkpoint（若 `checkpoint/custom_right_last` 已是最新可跳过），跑 `replay_evaluation.py` 与 `visualize_tip_workspace.py`，产物归档到 `outputs/baseline_main/`。

**人工（H1，已确认）**：基线采用 `main` 分支现有 checkpoint；基线 checkpoint、replay 记录与 workspace 报告已归档。

**验收**：`outputs/baseline_main/` 包含 checkpoint 路径记录 + workspace 报告 + replay 结果；分支 `anydexrt` 存在。

# Step 1 · 清除旧补丁（已完成）

**Codex 做什么**（在 `anydexrt` 分支）

- [x]  **整文件删除**（职责全部属于魔改管线，Step 2 的新 `prepare.py` 与新 manifest 整体取代）：
    - `geort/mocap/search_custom_aa_limits.py`（AA 限位搜索）
    - `geort/mocap/hts_balance.py`（体素平衡化：`select_balanced_frame_indices`/`build_stage2_report`）
    - `geort/mocap/hts_stage3.py`（接触加权 frame weights：`compute_frame_weights`/`build_stage3_report`）
    - `geort/mocap/hts_prepare_training.py`（平衡化 + **fist boost 重复帧**：`append_fist_boost_frames`、`compute_fist_curl_score`、`compute_mcp_weighted_fist_curl_score` 与全部 `--fist-boost-*` CLI + weights 输出）
    - `geort/training_targets.py`（chamfer 目标选择 + `mold` 机制 + metadata 构建）
    - `geort/dataset_manifest.py`（weights/weights_path/reports/transforms 的 manifest 机制）
- [x]  `trainer.py` 同步删除（保持官方 chamfer+direction+curvature 主循环可跑）：
    - 函数：`compute_tip_pinch_loss`、`compute_finger_segment_direction_loss`、`non_thumb_mcp1_joint_indices`、`compute_mcp1_fist_prior_loss`、`compute_mcp1_fist_prior_mask`、`find_human_weight_path`、`describe_human_weight_source`、`resolve_human_training_input`
    - `prepare_human_training_dataset`：weights 加载与 `mcp1_fist_prior_*` 参数全删，`WeightedRandomSampler` 分支改普通 shuffle，输入直接吃 raw `.npy` 路径
    - 主循环：pinch / segment_direction / mcp1_fist_prior 三个损失项与对应打印
    - metadata：`build_training_metadata`/`save_training_metadata` 调用改为写最小 JSON（数据路径 + epoch + 时间戳）
    - CLI：`--w_pinch`、`--pinch_threshold`、`--w_segment_direction`、`--w_mcp1_fist_prior` 与 5 个 `--mcp1_fist_prior_*`、`--chamfer_target`、`--chamfer_target_path`、`--mold_path`
- [x]  `visualize_tip_workspace.py` 删 AA 限位相关（保留其余评测功能）：`load_aa_limit_overrides_from_search_report`、`sample_urdf_tip_points` 的 `joint_limit_overrides` 参数、`urdf_baseline_tips`/`_overlap_delta`/HTML 报表 baseline 列、CLI `--aa_limit_search_report`/`--aa_limit_rank`、report 的 `joint_limit_overrides`/`joint_limit_override_source` 字段。
- [x]  `dataset.py`：删 `FramePointDataset` 的 `frame_fields` 机制（只为 mcp1 mask 存在），退化为纯点数据集。
- [x]  **PIP 监督体系**（扫描确认：不是独立 loss 项，而是 PIP 关键点以 `loss_weight: 0.25` 进入 chamfer/direction/curvature 的关键点加权）：
    - `trainer.py`：删 `weighted_keypoint_mean` 与 `keypoint_weights`，chamfer/direction/curvature 只对 `tip_indices` 计算、等权
    - `geort/utils/config_utils.py`：删 `parse_config_keypoint_info` 中 `loss_weight`/`weight`、`segment_pairs`、`pinch_pairs` 的解析输出（`pip_indices`/`tip_indices`/`finger_groups` 保留）
    - `custom_right.json` 等 config：删各 keypoint 的 `loss_weight` 字段；PIP 关键点条目本身保留（IK 输入维度与可视化仍用），是否从映射输入中移除在 Step 5 决策
- [x]  数据产物停用（不删历史文件）：`data/*_train.npy`/`*_train.json`（平衡集 + weights）、`data/*_humanshaped.npz`（human-shaped 目标云）、`outputs/` 的 AA 搜索报告；分支上训练输入一律回到 raw `.npy`。
- [x]  更新 README：删去已移除功能的章节。

**人工**：无（review diff）。

**验收**：`grep -rnE "aa_limit|mold|chamfer_target|fist|pinch|segment_direction|loss_weight|weights_path|WeightedRandomSampler|hts_balance|hts_stage3|dataset_manifest" geort/` 无残留（无关词如 pipeline 命中可忽略）；`python geort/env/hand.py --hand custom_right` 正常；trainer 直接读取 raw `.npy`，仅保留 chamfer+direction+curvature。按本轮决策，当前分支不重训，基线复用 `main` 已有 checkpoint。

# Step 2 · 新数据预处理管线

**Codex 做什么**

- [ ]  新建 `geort/data/prepare.py`：输入 raw HTS `.npy`，按 config 的 `human_hand_id` 提取逐指指尖 + 指根关键点；实现 AnyDexRT 归一化（逐指中心化 → 最大轴范围各向同性缩放到 $[-1,1]$）；输出 `<name>_prepared.npz` + 精简 manifest JSON（数据路径、逐指 center/scale、预留 anchors/contact 字段）。
- [ ]  robot 侧同法：对 `generate_robot_kinematics_dataset` 产出的指尖点云计算并保存归一化参数。
- [ ]  单测 `tests/test_prepare.py`：① 归一化后每指点云 ⊆ [-1,1] 且最大轴恰为 [-1,1]；② 各向同性（三轴同一 scale）；③ 反变换往返误差 < 1e-6。

**人工**：无（用旧 raw `hts_right.npy` 验证）。

**验收**：pytest 通过；对旧 raw 跑出 manifest，抽样打印归一化参数合理（scale 量级 ≈ 指尖行程半径，单位米）。

# Step 3 · 损失函数重写（`geort/loss.py`）

**Codex 做什么**

- [ ]  `partial_chamfer(mapped_human, robot_cloud)`：只算 human→robot 半边。
- [ ]  `distance_preservation(points, mapped_points, n_pairs)`：batch 内采样点对，映射前后距离差平方。
- [ ]  `local_motion_loss(x, fx, dx, dfx, T_human, T_robot)`：局部系内归一化方向负内积。
- [ ]  `anchor_align_loss(mapped_human_anchor, robot_anchor)`：成对 L2。
- [ ]  单测 `tests/test_losses.py`（关键行为验证）：① 目标云多出冗余区域时 partial chamfer 不变（对比双向会增大）；② 刚体平移/旋转映射下 L_dist = 0；③ 恒等映射下 L_motion = -1（最优）；④ 锚点完全对齐时 L_align = 0。

**人工**：无。

**验收**：pytest 通过。

# Step 4 · 局部坐标系模块

**Codex 做什么**

- [ ]  新建 `geort/frames.py`：人手侧从 HTS 指根/掌心关键点构建逐指局部系 $\mathbf{T}(x)$；机器人侧从 URDF 指根 link 位姿取局部系；两侧方向约定写成文档字符串常量。
- [ ]  可视化脚本 `geort/mocap/visualize_frames.py`：同屏渲染人手关键点+逐指局部系坐标轴 vs 机器人手+指根坐标轴。

**人工（H2）**：运行可视化，目视确认两侧同名轴方向语义一致（弯曲方向、侧旋方向、指尖外伸方向）。

**验收**：H2 确认通过；单测：局部系正交性 + 行列式=1。

# Step 5 · Trainer 重写（首版：三损失，无锚点）

**Codex 做什么**

- [ ]  重写 `trainer.py` 主循环：保留 `FK(IK(x))` 架构（`model.py` 不动）；损失 = P-Chamfer + L_dist + L_motion 等权；扰动机制复用旧 direction loss 骨架但在局部系计算；20 epoch、lr 1e-4、batch 2048。
- [ ]  L_align 接口预留：manifest 有 anchors 字段则自动启用（batch 32），无则跳过并警告。
- [ ]  CLI 精简为：`-hand`、`-human_data`（manifest）、`-ckpt_tag`、`--save_every`。
- [ ]  checkpoint 写入归一化参数 + 精简 metadata；`geort.load_model` 推理路径同步：归一化 → 映射 → qpos。

**人工**：无。

**验收**：用旧 raw 准备的 manifest 训练跑通 20 epoch，三项损失均下降且末期稳定；`replay_evaluation.py` 回放目视无坍缩/翻转异常；`visualize_tip_workspace.py` 中映射后人手云落在机器人 TIP 空间内且分布自然（无被冗余区拉扯的形变）。

# Step 6 · 锚点系统（D2）

**Codex 做什么**

- [ ]  `geort/anchor/anchor_spec.py`：侧旋/弯曲锚点定义（$K_0=5$，$\lambda=2$，$\beta_1: 0\to\pi/2$ 步长 $\pi/8$），参数可配。
- [ ]  `geort/anchor/collect_human_anchors.py`：SAPIEN viewer 逐个展示参考构型（含拇指预生成弯曲轨迹的 5 个采样位姿），同时读 HTS 流；按键采集 1–2 秒窗口均值，存人手锚点。
- [ ]  `geort/anchor/generate_robot_anchors.py`（P3）+ `interpolate.py`（P2，侧旋 K=50 / 弯曲 K=100），输出 `anchors_<hand>.npz` 并写入 manifest。
- [ ]  trainer 接入 L_align（Step 5 已预留）。

**人工（H3）**：按主计划 2.4 手册执行 50 次锚点记录（手平放桌面侧旋、五档卷曲、拇指轨迹模仿）。

**验收**：`anchors_custom_right.npz` 含 5 指×两类、插值后数量正确；可视化人手锚点序列与机器人锚点序列形状趋势一致；重新训练后 L_align 明显下降且其余三项损失不变差；回放对比 Step 5 版：握拳/张手等歧义易发姿态的映射更稳定可预测。

# Step 7 · 接触分类器 + 捏合推理（D3）

**Codex 做什么**

- [ ]  `geort/contact/collect_contact_labels.py`：按段录制（按键标记段边界 + y∈{0,1}），输出逐指对样本集。
- [ ]  `geort/contact/contact_model.py`：逐指对 MLP 二分类器，BCE，与 f_m 并行训练（独立优化器，20 epoch lr 1e-4 batch 2048）。
- [ ]  推理端集成：`load_model` 返回对象增加接触检测 → 触发时在映射位置附近搜索可行捏合姿态（拇指–目标指指尖距最小化，关节限位内）。

**人工（H4）**：按 2.5 手册录制 4 指对正/负段（含 5–15mm 悬停难负样本）。

**验收**：留出整段 held-out（按段划分，非按帧）评估，各指对 F1 ≥ 0.9；悬停段误触发率接近 0；实测小物体捏取：触发时机与预期一致。

# Step 8 · 端到端回归与基线对比

**Codex 做什么**

- [ ]  评测脚本：同一段 raw HTS 分别过 `main` 基线 checkpoint 与 `anydexrt` checkpoint，输出并排回放 + LMC/GMC 指标表 + workspace 对比报告。
- [ ]  更新 README：新管线的采集→准备→训练→部署全流程命令。

**人工（H5）**：遥操作实测 30 分钟，主观评估直观性/可预测性/捏合可靠性，记录结论。

**验收（合入 main 的门槛）**：LMC 优于基线；GMC 不劣于基线；捏合成功率不劣于旧 15mm 阈值方案；H5 主观评估正向。达标后 PR 合入 `main`。

# Step 9（可选后续）· 阶段二架构决策

- 若 Step 8 发现冻结神经 FK 误差是主要瓶颈（FK 预测 vs 仿真真值指尖误差 > 映射本身误差量级），启动显式 $f_m$（指尖→指尖）+ 解析/优化 IK 改造；否则维持 `FK(IK)` 架构。
- 验收：决策结论写回主计划页 M5 节。