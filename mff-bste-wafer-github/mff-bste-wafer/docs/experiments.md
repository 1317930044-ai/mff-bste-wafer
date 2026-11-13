# 实验复现与报告指南

从 [README](../README.md) 完成安装和 WM-811K 数据准备。使用同一份原始 ZIP、[默认配置](../configs/default.yaml) 和种子 `42` 可以复跑课程实验。模型通过验证 Macro-F1 选定，训练拟合结束、候选冻结后再评估测试集。报告同时记录验证指标、冻结测试指标、每类支持数和拆分方式。

## 实验协议

- 数据读取会核验字段、嵌套标签和晶粒状态。WM-811K 中 `trianTestLabel` 是原始字段拼写。分类特征来自晶圆矩阵，`lotName` 只用于分组审计。无标签样本不进入分类实验，None 是单独的有标签正常类。
- 分层与按 lot 拆分的目标比例为训练 60%、验证 20%、测试 20%。分组实验搜索各类均有覆盖的拆分，同一 lot 只落入一个分区。`standard` 在训练部分设类别上限，验证和测试保留原分区。`quick` 对三个分区均设类别上限，其表现用于小规模复验。
- 特征包括有效区空间密度、八连通域几何、离散 Radon、有效邻域纹理和行列投影。仅 Radon 的内部网格使用最近邻重采样。`standard` 默认输出 99 维，`quick` 默认 93 维。
- 决策树、随机森林、Extra Trees、Bagging、HistGradientBoosting、AdaBoost 和 GradientBoosting 构成基础比较。标准运行还比较特征消融、类权重、随机过采样、SMOTE、Balanced RF 与两种融合。
- Soft Voting 用验证集估计组合权重。MFF-BSTE 由类权重平衡的随机森林、Extra Trees 和 HistGradientBoosting 生成训练集折外概率，再拟合轻量逻辑回归元模型。分组 Stacking 的折内外也按 lot 隔离。重采样只在当前训练数据上执行。

## 原交付版实测基线

以下数值是此前在 Windows 11、Intel Core Ultra 5 225H、Python 3.14.2、scikit-learn 1.8.0 上运行的汇总。当前源码仓库提供复现脚本，实验文件需按 README 重新生成。主实验为 `wm811k-standard`，八类分层拆分，训练、验证、测试分别为 13,003、5,104、5,104 张。23 组实验完成后按验证 Macro-F1 冻结主模型。

| 模型 | 验证 Macro-F1 | 冻结测试 Macro-F1 | 冻结测试 Accuracy |
| --- | ---: | ---: | ---: |
| Soft Voting（验证集选定） | 0.9077 | 0.9033 | 93.04% |
| Stacking（MFF-BSTE） | 0.9058 | 0.9044 | 93.42% |
| HistGradientBoosting（验证集选定的最佳单模型） | 0.9049 | 0.9066 | 93.46% |
| Random Forest | 0.8893 | 0.8812 | 91.44% |

Soft Voting 对 Near-full 正确识别 29/30 张。标准分层和 lot 分组实验的随机森林测试 Macro-F1 分别为 0.8812 和 0.8832，测试样本数分别为 5,104 和 5,058。两种拆分的测试样本不同。真实九类 Quick 实验使用 162 张测试晶圆，验证集选定的 Stacking 测试 Macro-F1 为 0.8274，结果对应 Quick 抽样分布。

## 输出文件与五章对应

| 输出 | 可追溯信息 |
| --- | --- |
| `artifacts/runs/<run-id>/run.json`、`frozen.json` | 运行配置、随机种子、环境、选择依据、冻结模型 |
| `split_manifest.csv`、`oof.csv`、`predictions/` | 每个样本的分区、折外预测及冻结测试预测 |
| `leaderboard.csv`、`per_class.csv`、`metrics.json` | 指标对照、少数类支持数与混淆计数 |
| `feature_dictionary.csv`、`importance.csv`、`errors.csv` | 特征公式和计时、验证集置换重要性与错例 |
| `progress.jsonl`、`boosting.csv`、`benchmark.json` | 真实阶段日志、Boosting 损失与独立 CPU 基准。`benchmark.json` 由 `benchmark` 命令生成。 |
| `artifacts/reports/runs/<run-id>/` | 章节子目录、PNG/SVG 图、对应图注、数据表和 `figure_catalog.csv` |

1. **问题描述和需求分析**。F01–F06 包含类别示例、数量、占比、尺寸、掩码和状态编码。结合 `data_audit.json` 说明少数类及数据来源。
2. **算法原理和技术路线**。F07–F15 说明五路特征、最近邻预处理、树模型及 OOF 集成。结合 `feature_dictionary.csv` 和 `F13_sample_features.csv` 解释关键特征。
3. **问题求解分析和步骤**。F16–F31、F41 依次覆盖拆分、特征、性能、混淆矩阵、消融、置换重要性、误判和 lot 对照。F41 需要用 `--compare-run` 指定同配置的另一拆分运行。F30 为可选 SHAP 预留项。
4. **流程图和程序代码**。F32–F39 包含三端依赖、CLI 阶段日志、Boosting 曲线和真实预测卡。可截取网页识别、桌面和网页看板为 F36–F38，将截图依次命名为 `F36.png`、`F37.png`、`F38.png`，放入 `artifacts/runs/<run-id>/screenshots/`，再运行 `figures` 纳入索引。
5. **个人心得**。F40、F42 适合讨论 CPU 特征成本和 Radon 取舍。可运行 `benchmark` 比较冷、热缓存和分支开关，记录自己遇到的具体问题与证据。

## 核心代码入口

| 文件 | 职责 |
| --- | --- |
| [`wafer/data.py`](../wafer/data.py) | 加载旧版 PKL、标签解析、矩阵校验、示例生成与导出 |
| [`wafer/preparation.py`](../wafer/preparation.py) | 分层与分组拆分、特征缓存、样本 ID 对齐 |
| [`wafer/features.py`](../wafer/features.py) | 五路特征、特征定义及计时 |
| [`wafer/models.py`](../wafer/models.py)、[`wafer/experiment.py`](../wafer/experiment.py) | 拟合、折外预测、加权融合、冻结和指标计算 |
| [`wafer/inference.py`](../wafer/inference.py)、[`wafer/figures.py`](../wafer/figures.py) | 三端共享推理、批量预测、识别卡与报告素材 |
