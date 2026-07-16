# custom_right 锚点质检报告设计

## 目标

从已落盘的 `custom_right / hts_right.npy` 人手锚点、机器人锚点与 FK 对拍产物生成一份静态 Markdown 判决报告。报告用于裁决机器人侧五级锚点是否必须按人手观测跨度收缩；它不改变锚点、挖掘算法、训练配置或任何损失。

## 输入契约

- 人手原始数据：`data/hts_right.npy`（D1）。
- 人手锚点 bundle：`data/anchors_human_right.npz` 及其生成 metadata。
- 机器人锚点：现有 custom_right 生成产物；若没有可读 bundle，报告明确 FAIL，不从其他手型替代。
- 对拍复现输入：`outputs/anchors/parity_qpos.npz`。
- FK 门记录：`outputs/anchors/custom_right_fk_parity.json`；报告引用既有 `max < 1e-3 m` 结果而不重跑。
- 归一化契约：本次重验训练目录的 `normalization.json`，并记录其 `human_data_source`。

所有相对位置均以米制 hand-base 空间解释；所有跨度比在同一份归一化契约下计算。

## 报告内容

### A. 裁决表

1. 每根手指 × lateral/bending 的五个等级参数在 D1 全有效帧分布中的百分位。非拇指使用 lateral 的 alpha 与 bending 的 beta1；拇指使用主轨迹弧长分数。
2. 每根手指 × 动作类型的 L1→L5 TIP 行程（归一化空间）：人手、机器人与 human/robot 比值。

### B. 人手锚点自检

- 各等级 medoid 候选帧数（小于 10 标黄）；
- MCP1≈PIP=b、DIP≈b/2 的残差分布，异常值标黄；
- 同组 TIP 单调性、相邻间距与 max/min 比、重复或退化检测。

### C. 机器人与配对自检

- 机器人五级在关节空间和 TIP 空间的相邻间距；
- 拇指主轨迹弧长的等距性偏差；
- `750 = 250 lateral + 500 bending`、逐指 `50/100` 构成核验；
- 人手插值步长均匀性与相邻方向内积正值检查；
- 已通过的解析 FK / SAPIEN FK 对拍门引用。

### D. 契约存证与图

- 坐标空间、normalization 来源、human_data_source、生成 git hash、输出文件清单；
- 逐指静态 3D 图：人手锚点叠 D1 TIP 点云；机器人锚点叠机器人可达 TIP 点云。

## 输出与失败语义

输出目录为 `outputs/anchors/qa_custom_right/`，其中主报告为 `anchor_qa_report.md`、图像为 `figures/*.png`、数值中间表为 `metrics.json`。缺少任何必需机器人 bundle、normalization 契约或可判定字段时，报告对应小节以 FAIL 标出缺失路径和原因；不得伪造数字或以不同 hand/side 数据补齐。

## 验证

报告生成器必须：

1. 断言输入 hand/side 为 `custom_right/right` 与 `hts_right`；
2. 断言每一张 A/C 表的预期行数；
3. 断言对拍 qpos 总数与动作构成；
4. 断言图文件、JSON 指标和 Markdown 均成功写出；
5. 使用 main 主分支 CPython 3.12 解释器，以匹配 `open3d==0.19.0`。
