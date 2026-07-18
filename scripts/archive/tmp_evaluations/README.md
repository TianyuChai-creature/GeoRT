# `/tmp` 正典评测脚本归档

## 归档范围与溯源

本目录保存曾在 `/tmp` 中执行、且结果已进入 C2 变体正典评测归档的脚本。归档时 `/tmp` 中已不存在该脚本；从 Codex 会话记录中提取其原始 heredoc 正文，未改动内容。

| 项目 | 值 |
| --- | --- |
| 原始路径 | `/tmp/evaluate_c2be_motion.py` |
| 会话记录 | `/home/creature/.codex/sessions/2026/07/15/rollout-2026-07-15T17-42-54-019f6528-683e-76b3-8822-cd2b8ee0e2c5.jsonl`，`custom_tool_call` 第 6373 条 |
| 归档路径 | `scripts/archive/tmp_evaluations/evaluate_c2be_motion.py` |
| 归档文件 SHA256 | `6db8ef3e34a50c839a5258529b671cc8680678451759e8798998ca0bc42a9357` |
| 当时命令 | `/home/creature/Desktop/GeoRT/.venv/bin/python /tmp/evaluate_c2be_motion.py --device cuda --frames 1000 --random-state 42 --perturbations X/Y/Z:{-15,-5,+5,+15}` |

仓库及输出归档中对 `/tmp/*.py` 的正典评测引用仅发现上述脚本；`outputs/final_matrix/build_custom_right_with_rot.py` 的 `/tmp/custom_right_with_rot.npz` 是临时输出路径，不是评测脚本。

## 指标定义

脚本导入 `outputs/final_matrix/evaluate_final_matrix.py`，以 `data/hts_right.npy` 的 `RandomState(42)` 抽样 1000 帧为公共输入：

| 指标 | 实现口径 |
| --- | --- |
| GMC | `evaluate_final_matrix.cos_metric(human_norm, robot_norm)`；人、机归一化后的运动方向余弦指标。 |
| 统一 local-T LMC | 人手局部旋转与机器人点云最近邻借旋，调用评测侧 `local_motion_metric`。 |
| 扰动注入 | 绕 hand-base X/Y/Z 轴各 `-15,-5,+5,+15` 度，记录相对无扰动基线的逐指绝对误差差值。 |

## 抽验复算

归档脚本以原参数重新执行，`c2b_s42` GMC：`0.940441788081173`。在案值：`0.940441788081173`，来源 `outputs/c2_variants/motion_consistency_eval/c2be_motion_consistency.json`。
