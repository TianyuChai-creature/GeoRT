# Quest 实时驱动 C2

默认生产权重是 C2 seed 42：

`checkpoint/custom_right_2026-07-16_22-04-19_c2_s42/last.pth`

启动会将其 SHA256、`motion_frame` 与锚点元数据同
`outputs/final_matrix/checkpoint_hashes.json` 的终验归档逐项比对。权重不在归档中、权重哈希不同、或 metadata 不同都会在连接 Quest 前退出。

先在仓库根目录生成或刷新只读账本：

```bash
/home/creature/Desktop/GeoRT/.venv/bin/python -m geort.mocap.generate_realtime_checkpoint_ledger \
  --final-matrix outputs/final_matrix/final_matrix.json \
  --output outputs/final_matrix/checkpoint_hashes.json \
  --repo-root /home/creature/Desktop/GeoRT/.worktrees/AnyDexRT
```

## Stage 1：SAPIEN 镜像

```bash
/home/creature/Desktop/GeoRT/.venv/bin/python -m geort.mocap.hts_realtime_inference \
  --stage 1 \
  --checkpoint checkpoint/custom_right_2026-07-16_22-04-19_c2_s42 \
  --archive-root outputs/final_matrix \
  --hand custom_right --hand-side right --transport udp --host 0.0.0.0 --port 9000 \
  --qpos-scale 1.0 --watchdog-ms 200 --ramp-frames 100 --max-joint-step 0.05 \
  --contact_refine off
```

HTS 输入通过 `frame_to_geort_points` 统一转换为 GeoRT 的右手系；映射调用评测同一 `load_model(...).forward`，内部仍使用 checkpoint 的 `normalization.json`。启动日志包含：checkpoint SHA256、`motion_frame`、锚点对数/路径和接触配置。

空格键会锁存急停并冻结上一安全姿态；`Ctrl-C` 也会锁存后退出。输入断流超过 200 ms 时保持当前姿态，下一有效帧再经过 100 帧缓升。每个输出永远限位钳制，单关节每帧最多变动 0.05 rad。

每次运行都写入 `outputs/realtime_sessions/<UTC 时间戳>/`：`frames.npz` 含原始/归一化关键点、map/refine/safe qpos、接触 JSON 与分阶段时间；`summary.json` 含 p50/p95 和安全事件计数。

## 离线 parity

```bash
/home/creature/Desktop/GeoRT/.venv/bin/python -m geort.mocap.verify_realtime_c2 \
  --checkpoint checkpoint/custom_right_2026-07-16_22-04-19_c2_s42 \
  --archive-root outputs/final_matrix --data data/hts_right.npy --frames 1000 --seed 42
```

## Stage 2 / Stage 3

当前仓库未提供真实机器人 actuator adapter；`--stage 2` 与 `--stage 3` 会明确报“未产出+原因”，不会把 SAPIEN 的 `set_qpos_target` 当成实机下发。接入经审查的 adapter 后，Stage 2 固定 `--contact_refine off`，Stage 3 固定 `--contact_refine on`，其余安全和审计参数不变。
