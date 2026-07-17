# Quest 实时驱动 C2 定稿权重：实施计划

> 基座：`4827adb` 的训练/权重/数据语义；本计划对应的设计记录为 `1563fd5`。
> 范围：仅实时映射、运行时安全、离线验收与文档；不修改 `geort/trainer.py`、`geort/loss.py`、任何 checkpoint 或数据产物。

## 约束与固定输入

- 默认 checkpoint：`checkpoint/custom_right_2026-07-16_22-04-19_c2_s42/last.pth`。
- `--checkpoint` 只允许切换到 `outputs/final_matrix/checkpoint_hashes.json` 中登记的终验 checkpoint；启动时对 `last.pth` SHA256、`motion_frame`、anchor 元数据与终验归档逐项核对，不匹配或未登记即拒绝启动。
- Quest/HTS 输入继续由 `geort.mocap.hts_right_mocap.frame_to_geort_points` 完成 Unity-left 到 GeoRT right-handed 的转换；归一化与评测统一通过 `geort.export.GeoRTRetargetingModel.forward`（内部调用 `_select_and_normalize_tips` / `normalize_finger_points`），实时脚本不得复制坐标或归一化公式。
- 映射路径只有 `f(x)` 加可选现有接触精修；SAPIEN 只用于 Stage 1 的镜像显示，不参与映射计算。

## 1. 先用测试锁定运行时安全组件

文件：新建 `tests/test_realtime_safety.py`、`tests/test_realtime_provenance.py`。

1. 为纯 NumPy `RealtimeSafetyController` 写失败测试：
   - 初始 100 帧从当前 qpos 线性缓升；
   - 单关节每帧变化截到 `0.05 rad`；
   - 输入 NaN/Inf 保持上一帧且计数；
   - 200 ms 超时冻结，下一有效帧重新进入 100 帧缓升；
   - 急停锁存后保持姿态，解除后重新缓升；
   - 无论 `qpos_scale` 是否为 1，所有输出都做硬限位钳制。
2. 为 provenance 解析器写失败测试：
   - 匹配的归档条目返回 checkpoint SHA、`motion_frame` 和 anchor 摘要；
   - SHA、配置字段、anchor 数量或路径不一致均抛带字段名的异常；
   - 未登记的 checkpoint 拒绝运行。
3. 运行以上测试，确保它们在实现前失败且失败原因对应缺失组件。

## 2. 实现可测试的实时运行时层

文件：新建 `geort/mocap/realtime_runtime.py`、`geort/mocap/realtime_provenance.py`。

1. `RealtimeSafetyController` 保存 `last_qpos`、启动/恢复 ramp 状态、watchdog/NaN/速率限制/急停计数；接收单调时钟时间，返回经过 `clip(lower, upper)`、ramp、速率限制后的 qpos。
2. 明确执行次序：有限性检查 → 限位钳制 → ramp → 限位钳制 → 速率限制 → 限位钳制。任何异常输入或 watchdog/急停都输出最后一个已安全输出。
3. `SessionRecorder` 预分配/追加以下每帧字段，并在退出时写 `outputs/realtime_sessions/<UTC 时间戳>/frames.npz` 与 `summary.json`：时间戳、原始 `[21,3]` 关键点、模型归一化输入、`f(x)` 输出、接触精修输出（off 时与前者相同）、安全层输出、各阶段耗时、四对接触概率/混合权重（若可用）。
4. `realtime_provenance` 读取 `outputs/final_matrix/final_matrix.json` 与新增的 `checkpoint_hashes.json`；以仓库相对 checkpoint 路径匹配归档，计算当前 `last.pth` SHA256，再读取 training metadata 中的 `motion_frame` 与 anchor 字段完成严格比对。
5. 新增只读归档文件 `outputs/final_matrix/checkpoint_hashes.json`：记录八个终验 `last.pth` 的 SHA256、相对路径、`motion_frame`、anchor 摘要及生成命令。该文件不改变 checkpoint、数据或训练代码。

## 3. 扩展导出 API 的可观测性，不分叉映射

文件：`geort/export.py`。

1. 给 `GeoRTRetargetingModel` 增加只读的最近一次调用诊断：`last_normalized_tips`、`last_mapped_qpos`、`last_refined_qpos`、各阶段耗时。归一化仍直接调用已有 `_select_and_normalize_tips`；不重写任何公式。
2. 将设备选择显式保留为现有模型行为；实时入口只调用 `load_model`，不自行加载网络或 checkpoint state dict。
3. 接触关闭时 `last_refined_qpos` 精确等于 `last_mapped_qpos`；开启时复用现有 `ContactRefiner`，不改变其参数、模型或优化器。
4. 增加导出 API 的单测，断言诊断字段不改变 `forward` 返回值，且归一化字段与 `_select_and_normalize_tips` 完全相同。

## 4. 接入 Quest/SAPIEN 实时脚本

文件：`geort/mocap/hts_realtime_inference.py`、`tests/test_hts_realtime_inference.py`。

1. 保留旧 `--ckpt_tag` 作为兼容别名，新增主接口 `--checkpoint`，默认 C2 s42；新增 `--stage {1,2,3}`、`--watchdog-ms 200`、`--ramp-frames 100`、`--max-joint-step 0.05`、`--session-root`、`--archive-root`、`--estop-key`。`--contact_refine` 默认仍为 `off`；Stage 2 拒绝 `on`，Stage 3 要求 `on`。
2. 启动先完成 provenance 核验，随后打印一行：checkpoint 路径、SHA256、`motion_frame`、`anchor: <N> pairs from <path>`、接触状态和完整生效实时配置。任一比对差异在连接 Quest 前退出。
3. 保留 `iter_hts_points` 输入链和 `validate_live_points`；每一有效帧仅 `model.forward(raw_points)`，从模型诊断读取归一化、mapped/refined 输出并送入安全控制器。修正 `scale_and_clamp_qpos`，使 scale 为 1 时也永远执行硬限位钳制。
4. Stage 1：SAPIEN 镜像可视化，只向模拟手下发安全输出；Stage 2/3：调用现有下游实时目标接口，Stage 3 启用接触精修。若本仓库没有实机 actuator 接口，Stage 2/3 在初始化处明确报“未产出+缺少 actuator adapter”，不以 SAPIEN 冒充实机。
5. 在 SAPIEN viewer 循环中检查可用的键盘事件；`--estop-key` 按下时锁存急停、立即冻结。`Ctrl-C` 同样先锁存再退出并写 session。
6. 结束时打印映射/接触/安全/总耗时 p50/p95、实际帧率、rate-limit/watchdog/NaN/急停计数；contact on 时另打印四对占空比及接触精修 p50/p95。

## 5. 离线硬门与验收脚本

文件：新建 `geort/mocap/verify_realtime_c2.py`、新建 `tests/test_realtime_parity.py`。

1. `--offline-parity` 使用 D1 `RandomState(42)` 的 1000 帧，经完整 realtime mapper（跳过 Quest receiver、SAPIEN 和安全 ramp）与评测调用的同一 `load_model(...).forward` 对拍，报告 `max |Δq|`；门值 `≤1e-6 rad`。
2. 用合成 NaN、Inf、超限和超时序列验证安全守卫；输出每项计数与最终 qpos，确保没有 NaN 下发。
3. 对默认 C2 运行离线 Stage 1 并落一个 session；检查所有必需字段、JSON 可读、输出 qpos 全部在限位内。
4. 全部 CPU 验收：单元测试、离线 parity、session schema/summary。无训练、无 checkpoint 写入、无 GPU 需求。

## 6. 文档、提交与分级上线交付

文件：`docs/realtime_c2_quest.md`（新建），必要时更新脚本 module docstring。

1. 写明三阶段唯一允许命令（解释器显式为 `/home/creature/Desktop/GeoRT/.venv/bin/python`）、各 CLI 参数、默认 C2 SHA、启动自检预期格式、急停/恢复行为、session 位置与审计方法。
2. Stage 1 完成后只交付离线/SAPIEN数值；Stage 2、3 必须等待用户在真实 Quest/actuator 环境确认再运行，不自行连接实机。
3. 最后复核 `git diff --name-only` 不包含 `geort/trainer.py`、`geort/loss.py`、`data/` 或 checkpoint 路径；提交仅含实时运行时、测试和文档。

## 验收命令

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/creature/Desktop/GeoRT/.venv/bin/python -m pytest \
  tests/test_realtime_safety.py tests/test_realtime_provenance.py \
  tests/test_hts_realtime_inference.py tests/test_realtime_parity.py -q

/home/creature/Desktop/GeoRT/.venv/bin/python -m geort.mocap.verify_realtime_c2 \
  --checkpoint checkpoint/custom_right_2026-07-16_22-04-19_c2_s42 \
  --archive-root outputs/final_matrix --frames 1000 --seed 42 --device cpu
```

预期交付仅报数：离线 `max |Δq|`、NaN/限位计数、各阶段 p50/p95、实际帧率、触发计数和 session 路径；不对 Stage 2/3 作虚构实机结论。
