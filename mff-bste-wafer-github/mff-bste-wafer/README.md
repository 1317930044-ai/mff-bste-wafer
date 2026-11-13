# 晶圆缺陷识别 · MFF-BSTE

输入半导体晶圆的离散晶粒状态图，上提取空间、几何、Radon、纹理和投影特征，再用决策树及其集成模型识别缺陷。命令行、Streamlit 网页和 Tkinter 桌面程序共用输入校验和推理代码。

- `defect8` 识别 Center、Donut、Edge-Loc、Edge-Ring、Loc、Random、Scratch、Near-full 八种缺陷。
- `all9` 在八类缺陷上增加 None 正常类。
- 真实实验采用 WM-811K。项目支持固定拆分、按生产批次 `lotName` 分组、逐类指标、折外 Stacking、消融和报告图表重建。

## 1. 下载源码与安装

**Code → Download ZIP**，解压后在包含 `main.py` 的目录打开 Windows PowerShell。推荐64位Python 3.11或3.12使用桌面端时安装Python的Tcl/Tk组件。所有命令都从项目根目录运行。

```powershell
py -3.11 -m venv .venv
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -m pip install -r requirements.txt
& $py main.py env
```

使用 Python 3.12 时，将第一行改为 `py -3.12 -m venv .venv`。如果系统通过 `python` 调用所需版本，可用 `python -m venv .venv`。新开 PowerShell 后重新执行 `$py = ...` 这一行即可。`main.py env` 会列出 Python、依赖和 Tkinter 版本。

## 2. 无数据快速演示

下面的命令即时生成带标记的合成晶圆、训练八类模型并生成图表。首次下载源码后，可以先用这条路径验收完整流程。合成数据的指标用于程序测试。

```powershell
& $py main.py train-all --synthetic --task defect8 --mode quick --run-id demo-synthetic
& $py main.py evaluate --run-id demo-synthetic
& $py main.py figures --run-id demo-synthetic
```

训练后启动网页或桌面程序，选八类、`random_forest · demo-synthetic` 和一个项目样本即可识别。模型、样本与日志都在 `artifacts/runs/demo-synthetic/`，图表在 `artifacts/reports/runs/demo-synthetic/`。

## 3. 获取并加载真实数据

1. 打开 Kaggle 的 WM-811K 数据页，点击 **Download**。按页面提示登录。下载包包含 `LSWMD.pkl`，数据页将该版本标为 CC0。[WM-811K 数据页](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
2. 在项目根目录运行 `New-Item -ItemType Directory -Force data/raw`。将下载包保存为 `data/raw/wm811k.zip`。如果已经解压，也可直接放置 `data/raw/LSWMD.pkl`。程序默认先找 PKL，再找 ZIP，ZIP 可直接读取。用 `--data` 可指定其他本地路径。
3. 运行以下命令检查文件、统计标签并导出识别示例。若放置的是 PKL，将两条命令中的文件名改为 `data/raw/LSWMD.pkl`。

```powershell
Get-FileHash -Algorithm SHA256 data/raw/wm811k.zip
& $py main.py inspect --data data/raw/wm811k.zip
& $py main.py export-samples --data data/raw/wm811k.zip
```

原实验使用的 Kaggle ZIP 的 SHA-256 为 `0e4bd4316a110a04e475a88071f5e1c7fe4303b12ee53c851d2cae4d995a93e2`。审计输出位于 `artifacts/reports/data_audit/wm811k/`，示例位于 `artifacts/samples/wm811k/`。原实验实际读取到 811,457 张晶圆，其中 172,950 张有可用标签。首次读取需要加载较大的原始文件，可能花费数分钟。后续训练可以复用本地缓存。请从可信来源获取原始 PKL 和本地模型文件，网页仅接收晶圆矩阵或规范 PNG。

## 4. 提取特征、训练和复现报告图表

先尝试真实数据的 Quick 训练，再根据需要运行八类标准实验和 lot 分组对照。`extract` 可单独运行以提前准备特征，训练也会自动提取。命令默认读取第 3 节放置的数据。

```powershell
& $py main.py train --task defect8 --mode quick --model random_forest --run-id wm811k-quick

& $py main.py extract --task defect8 --mode standard
& $py main.py train-all --task defect8 --mode standard --run-id wm811k-standard
& $py main.py train --task defect8 --mode standard --split grouped --model random_forest --run-id wm811k-grouped
& $py main.py evaluate --run-id wm811k-standard
& $py main.py figures --run-id wm811k-standard --compare-run wm811k-grouped
```

九类和 CPU 基准可独立运行。

```powershell
& $py main.py train-all --task all9 --mode quick --run-id wm811k-quick-all9
& $py main.py figures --run-id wm811k-quick-all9
& $py main.py benchmark --task defect8 --mode quick --run-id bench-radon
& $py main.py benchmark --task defect8 --mode quick --no-radon --run-id bench-no-radon
```

| 模式 | 使用场景 |
| --- | --- |
| `quick` | 拆分后限制每类样本数。适合核对安装、训练及演示。 |
| `standard` | 主实验。限制训练集每类最多 3,500 张，保留验证和测试分区的原始类别分布。执行基础模型、均衡策略、特征消融与集成对照。 |
| `full` | 保留全部可用样本，增加树数与折数。CPU 和磁盘开销较大。 |

重跑训练时使用新的 `--run-id`。默认配置见 [configs/default.yaml](configs/default.yaml)，可复制后通过 `--config` 指定。需要额外尝试 XGBoost、LightGBM 或 CatBoost 时，安装 `requirements-optional.txt`，再在标准实验命令中加入 `--external`。主要评价指标为 Macro-F1，训练与验证的用途、复现实测基线和报告素材位置见 [实验复现与报告指南](docs/experiments.md)。

## 5. 三端识别

先完成第 2 节或第 4 节的任意一次训练。网页与桌面会扫描本地已完成的运行。

### Streamlit 网页

```powershell
& $py -m streamlit run web_app.py
```

打开 `http://127.0.0.1:8501`。选择任务和模型，然后选择项目样本或上传文件，点击“识别晶圆”。页面显示原图、缺陷掩码、Top-3、特征解释和耗时。批量页可下载预测 CSV，“实验与训练日志”页可刷新查看训练状态、结果与图表。使用 `Ctrl+C` 停止网页服务。

### Tkinter 桌面 / PyCharm

```powershell
& $py desktop_app.py
```

在 PyCharm 中打开项目根目录，将解释器设为 `.venv\Scripts\python.exe`，右键运行 `desktop_app.py`。选择模型和项目样本，也可以点击“打开文件…”选择晶圆图，再点击“识别晶圆”。

### 命令行

下面使用第 2 节自动生成的模型和样本。使用真实模型时，将 `$run` 改为 `wm811k-quick` 等已训练的运行 ID，并对应修改模型文件名。运行样本目录内的同一晶圆通常同时有 PNG 和 NPY，两者各产生一条批量记录。

```powershell
$run = 'demo-synthetic'
$model = "artifacts/runs/$run/models/random_forest.joblib"
$samples = "artifacts/runs/$run/samples"
$sample = (Get-ChildItem -LiteralPath $samples -Filter '*_Center.png' | Select-Object -First 1).FullName

& $py main.py predict --model-path $model --input $sample --out artifacts/predictions/single.json --card artifacts/predictions/card.png
& $py main.py predict-batch --model-path $model --input-dir $samples --out artifacts/predictions/batch.csv
```

三端支持二维数值 `.npy`、仅含 `wafer` 数组的 `.npz`、以及项目导出的固定调色板 PNG。晶粒状态为 0（背景）、1（正常）、2（缺陷）。PNG 色值依次为 `#17222F`、`#C7E2DF`、`#E95842`，请保留原始尺寸和颜色。普通照片、截图和任意配色的 PNG 缺少可靠的晶粒状态编码。上传限制为每文件 8 MiB，矩阵最多 1,048,576 个元素。

## 6. 命令、输出和测试

| 命令 | 产物或用途 |
| --- | --- |
| `env` | 环境与依赖检查 |
| `inspect`、`export-samples` | 源数据审计、类别统计和规范图导出 |
| `extract` | 拆分清单、按 ID 对齐的分支特征、特征字典与缓存 |
| `train`、`train-all` | 模型文件、冻结记录、训练日志、指标与逐样本预测 |
| `evaluate`、`figures` | 从本地运行重算测试指标、重建五章图表与索引 |
| `benchmark` | 峰值内存采样、CPU 耗时及推理记录 |
| `predict`、`predict-batch` | 单张 JSON、PNG 识别卡及批量 CSV |

```powershell
& $py main.py --help
& $py main.py train-all --help
& $py -m unittest discover -s tests -v
& $py -m tests.web_smoke
```

最后一条需要先完成第 2 节的 Quick 训练，用于检查网页的识别、文件输入、批量与日志组件。正式报告可从 `artifacts/reports/figure_catalog.csv` 定位生成的图表；运行参数、折外预测、分类报告和逐模型结果在 `artifacts/runs/<run-id>/`。

## 7. 仓库结构

```text
.
├── README.md                  # 下载、安装、训练、推理与复现入口
├── requirements.txt           # 主流程依赖
├── requirements-optional.txt  # 可选 Boosting 依赖
├── configs/default.yaml       # 随机种子、资源档和模型参数
├── main.py                    # CLI
├── web_app.py                  # Streamlit 网页
├── desktop_app.py              # Tkinter 桌面程序
├── wafer/                     # 数据、特征、模型、评估、推理和绘图
├── tests/                     # 核心测试和网页组件测试
├── docs/experiments.md         # 实验协议、原实测基线与报告指南
└── .streamlit/config.toml      # 本地网页配置
```

`data/` 和 `artifacts/` 由下载与运行创建，并由 `.gitignore` 排除。上传时将这个源码目录的内容放到仓库根目录，确保 `README.md`、`.gitignore` 和 `.streamlit/config.toml` 同时上传。教师下载仓库后可以自行运行第 2 节或第 3–4 节。

## 数据来源与引用

Ming-Ju Wu, Jyh-Shing R. Jang, and Jui-Long Chen. *Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets*. IEEE Transactions on Semiconductor Manufacturing, 28(1), 1–12, 2015. [IEEE 论文与 DOI](https://doi.org/10.1109/TSM.2014.2364237) · [Kaggle 数据发布页](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
