import os
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, patches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .common import ASSETS, CLASSES, Progress, SYNTHETIC, read_csv, read_json, write_csv, write_json
from .data import PALETTE, read_map
from .features import Wafer, components, extract, geometry, projections, sinogram
from .inference import load_model, predict, predict_directory

TEAL, INK, CORAL, BLUE, MUTE = "#127C80", "#17222F", "#E95842", "#4569A5", "#688096"
CHAPTERS = {1: "01_问题描述和需求分析", 2: "02_算法原理和技术路线", 3: "03_问题求解分析和步骤",
            4: "04_流程图和程序代码", 5: "05_个人心得"}

CATALOG = {
    "F01": (1, "九种晶圆类别样例", "各类别展示一个已标注样本，晶粒状态以统一调色板显示。晶圆尺寸和有效轮廓随产品而变，同一类别也存在不同形态。图可用于介绍八种缺陷和正常类；样例只用于观察形态，样本之间的视觉差异不能作为定量性能依据。"),
    "F02": (1, "八类和九类样本量", "比较两项任务对标签集合的使用范围，横轴采用对数尺度，便于同时读出数量最多和最少的类别。None 仅进入九类任务，Near-full 的样本支持数应与后续逐类召回一起解释。该统计来自完整可用标注，独立于训练抽样。"),
    "F03": (1, "标注类别的比例", "将各类可用标注数除以九类样本合计得到比例。由于主要类别占比较高，整体准确率会受类别分布影响，报告同时应给出 Macro-F1 和逐类召回。未标注样本没有被归入正常类，柱形比例的分母仅包含可用标注。"),
    "F04": (1, "晶圆尺寸与有效晶粒分布", "左图比较标注样本的高宽组合，右图按有效晶粒计数分箱。尺寸和有效晶粒数量会随样本改变，因此空间密度使用各区域自身的有效晶粒数作分母。绘图汇总仅覆盖有可用标注的样本，不能据此估计未标注数据的分布。"),
    "F05": (1, "晶粒状态与两种掩码", "同一晶圆依次展示原始状态、有效区和缺陷区。状态 0 为无效背景，1 为有效正常晶粒，2 为有效缺陷晶粒。有效区决定统计分母，缺陷区决定计数分子。掩码保留离散状态，示例中的晶圆图外观随输入矩阵而变。"),
    "F06": (1, "多样尺寸的随机晶圆实例", "按固定随机种子从带标注样例清单抽取若干晶圆，标注类别和原始尺寸。不同尺寸与有效区形状说明预处理应保留无效区边界，只有方向特征内部在必要时作最近邻变换。此图用于提出输入需求，样本未按总体频率抽取。"),
    "F07": (2, "整体技术路线", "系统由离散晶圆输入、掩码构造、五路特征、树模型、集成和三端展示组成。特征表依样本 ID 对齐，验证集用于冻结候选，测试指标在冻结后计算。箭头表示程序的数据流，图中不包含任何模型精度或运行速度数据。"),
    "F08": (2, "离散晶圆预处理", "解析结构化数组或固定调色板 PNG 后，检查矩阵维度、像素状态与有效晶粒。随后分别构建有效掩码和缺陷掩码，并保持原始尺寸。仅 Radon 网格采用保持长宽比的最近邻重采样；计算失败会在特征提取前给出原因。"),
    "F09": (2, "有效晶圆的同心环与扇区", "以有效晶粒质心为圆心，使用有效区的最大质心距离归一化半径，画出三个径向区和八个扇区。径向区可能包含无效背景，各区域缺陷率均以区域内有效晶粒数为分母。几何叠加用于辅助理解，无需把晶圆边界拟合成圆。"),
    "F10": (2, "缺陷质心、包围盒和主轴", "在样本上标出有效区质心、全缺陷质心，以及最大八连通域的包围盒和协方差主轴。用质心偏移表示整体位置，用最大域主轴表示局部方向。单像素或主轴不唯一时方向特征置为零，避免对退化形状作过度解读。"),
    "F11": (2, "八连通域及面积分布", "将缺陷掩码按八邻域连通关系编号，以不同色块显示，并绘制连通域面积最大的若干项。大小分布可以辅助区分团块、划痕和随机散点；邻接编号只是计算标识，颜色无物理意义，极小连通域也可能来自噪声。"),
    "F12": (2, "行列投影及有效区密度", "计算每行、每列的缺陷计数和有效晶粒数，将二者相除得到投影密度。条带或局部团块会使对应方向出现峰值，归一化可缓解不同有效长度的影响。无有效晶粒的行列在图中留空，特征汇总也排除这些位置。"),
    "F13": (2, "离散 Radon 正弦图", "在等间隔角度下，将缺陷晶粒投影到离散距离格并累加，得到角度与距离组成的二维图。各角度的投影集中程度可反映划痕等方向模式，角度描述投影法线。网格缩小可能损失细线，需要用 Radon 消融判断实际收益。"),
    "F14": (2, "决策树、随机森林与 Boosting", "单树通过特征阈值递归划分，随机森林通过抽样和随机子特征拟合多棵树，Boosting 按轮添加基学习器修正当前模型。结构图描述训练关系，树的数量、深度和分裂特征应从本次运行参数表中引用。"),
    "F15": (2, "Soft Voting 与严格折外 Stacking", "Soft Voting 用验证集确定基础模型权重。Stacking 将训练样本逐折留出，使用未见过该折的基础树输出构建元训练表，再用完整训练集重拟合基础树。分组实验的折内外 lot 同时隔离。两条路线的性能需由同一冻结测试集比较。"),
    "F16": (3, "各类别的训练、验证和测试样本数", "依据冻结样本 ID 清单统计三个集合的每类计数，并保留类别支持数。纵向比较可以发现极少类是否在每个集合中出现；分组实验的具体 lot 边界可在清单审计。Quick 使用上限抽样，不能代表完整总体比例。"),
    "F17": (3, "训练集标准化特征热力图", "从训练子集读取特征，以列均值和标准差标准化后绘图，展示有限个样本和特征。色块体现统计量的相对偏离，便于观察分支间尺度和常量特征。标准化只用于可视化，训练树模型使用原始特征，图中不包含测试样本。"),
    "F18": (3, "训练特征的二维 PCA 投影", "在训练样本上对标准化特征拟合 PCA，绘出前两个主成分及各自解释方差。不同颜色表示标签，用于观察可分结构和混叠。二维投影丢失其余维度，也不会反映树模型的非线性边界，因此不能用散点分离程度代替性能。"),
    "F19": (3, "候选模型的 Macro-F1", "以同一数据拆分比较候选模型在验证和冻结测试集上的 Macro-F1。每类 F1 具有同等权重。图中排序依据验证分数，测试分数保留用于报告差异。各模型可能拥有不同训练开销，结论应与效率、类别支持数同时引用。"),
    "F20": (3, "Accuracy 与 Balanced Accuracy", "同一冻结测试集上比较整体准确率和各类召回的平均值。两者差距较大时需结合逐类召回检查类别分布与错误。指标来自同一预测清单，分母与类别集合均可追溯，分组与随机拆分的结果应分图或明确标明。"),
    "F21": (3, "训练耗时与压缩模型大小", "并排显示训练墙钟时间和压缩 joblib 文件体积。Stacking 计入折外训练及基础树重拟合，Voting 计入其基础成员的训练总成本。结果依赖硬件、线程、样本量和压缩配置，复跑时应保持资源条件或单独记录差异。"),
    "F22": (3, "模型单条推理时延", "在每个模型完成热身后，随机抽取最多 24 个已提取的特征向量，逐条记录概率预测时延并报告中位数和 P95。该图隔离了模型开销，读取文件、建掩码和提取特征的端到端时间见识别卡与批量 CSV。"),
    "F23": (3, "验证集选定模型的混淆计数", "矩阵按真实类别组织行、按预测类别组织列，显示冻结测试集的误判数。对角线是正确预测，非对角线反映错误流向。模型依据验证 Macro-F1 选定。类别支持相差较大时应同时查看按行归一化矩阵。"),
    "F24": (3, "按真实类别归一化的混淆矩阵", "将每个混淆计数除以所在真实类别的样本数，便于比较不同类的召回率和误判去向。少数类别的比例对一两条记录很敏感，引用比例时宜同时注明真实测试支持数，并辅以错例图核查形态重叠。"),
    "F25": (3, "逐类别 Precision、Recall 和 F1", "展示主模型在冻结测试集上的逐类查准率、召回率和 F1，并在类别名后标注样本数。对极少类，可以将支持数与计数混淆矩阵并用，避免只凭总体准确率下结论。预测为空时该类精确率记为零。"),
    "F26": (3, "特征分支的单用和移除消融", "固定随机森林配置、样本拆分和种子，比较五路齐用、单独使用某一分支，以及从全集移除某一分支的结果。移除差值衡量该配置下的边际作用，单用分数表示独立区分能力。互相相关的分支可能共享信息。"),
    "F27": (3, "类别均衡策略消融", "在相同特征和拆分下比较无均衡、类权重、随机过采样、SMOTE 与 Balanced RF。采样只对当前拟合的训练数据应用。Macro-F1 反映平均效果；若总体分数接近，还应检查逐类召回和误报方向。Balanced RF 同时改变了抽样模型。"),
    "F28": (3, "单模型、Soft Voting 和 Stacking", "比较按验证分数选出的最佳独立模型与两种融合。Voting 在验证集估计权重，Stacking 仅以折外概率拟合元分类器。显示验证与冻结测试分数，便于讨论融合的收益与成本；排名可能随拆分、任务和超参数改变。"),
    "F29": (3, "验证集置换重要性前 20 项", "在验证样本上依次打乱一个特征，记录 Macro-F1 的降低幅度，并显示变化最大的特征。正值表示在该样本下有预测贡献，负值可能源于噪声或估计方差。相关特征可能互相替代，排序不等于因果或工艺诊断。"),
    "F30": (3, "SHAP 解释", "树模型的全局或单样本 SHAP 可作为扩展分析。当前交付以验证集置换重要性、预测卡片的可读特征和错例为主，SHAP 未纳入实测图。后续若做，需记录模型版本、背景样本来源、输出概率空间和解释器参数。"),
    "F31": (3, "主模型的代表性误分类", "从冻结测试集的误判中按预测概率排序，每个真实类别最多取两个样本，标注真实标签、预测与概率。可对照有效区、径向位置、团块和细线观察混淆线索。这里没有工艺复判记录，图示不推断具体制造故障。"),
    "F32": (4, "三端与核心模块结构", "网页、桌面和 CLI 调用同一推理模块；数据、特征、训练和绘图各自保存可复现产物。图中展示运行入口与关键依赖关系，可配合代码解说定位输入校验、特征顺序检查、模型持久化和报告输出的实现。"),
    "F33": (4, "从数据到结果的程序执行顺序", "原始数据经审计和拆分后生成缓存，训练命令输出参数、折外预测、冻结记录与测试结果。推理命令复用模型中记录的特征设置。评估和图表能够从保存的记录重建，训练与模型加载各有独立入口。"),
    "F34": (4, "CLI 训练日志的阶段时间轴", "从磁盘 JSONL 中读取各事件的相对秒数与阶段，绘制训练、交叉验证、冻结和测试事件在时间上的位置。点对应真实日志记录，图中不把普通树的拟合过程表述为迭代轮次。日志也用于网页运行看板。"),
    "F35": (4, "Boosting 训练与验证损失", "从实际 Boosting 迭代记录重建对数损失曲线。HistGradient 使用显式验证集执行早停，曲线包含初始轮次。训练与验证差距可用于讨论过拟合，但测试集没有参与曲线或早停；不同实现的最佳迭代可在各模型参数表中核对。"),
    "F36": (4, "网页识别实录", "在本地网页选择带来源的规范样本和已训练模型，执行识别后查看晶圆原图、缺陷掩码、类别和概率。截图记录的是当次真实界面状态；截图中的单图结果仅用于演示交互，分类性能以独立测试指标为准。"),
    "F37": (4, "Tkinter 桌面识别实录", "从独立桌面程序选择模型与合法晶圆文件，点击识别后查看掩码、Top-3 概率与耗时。PyCharm 的右键运行使用同一 Python 入口；截图取自实际界面运行，课堂演示前可用项目提供的同格式样本复查。"),
    "F38": (4, "网页训练日志看板实录", "看板通过读取 CLI 生成的 JSONL 和运行元数据更新阶段、模型名、训练样本量及验证分数。页面可手动刷新，刷新不改变训练进程。截图记录已有运行状态，某个时点只能显示当时已写入磁盘的日志。"),
    "F39": (4, "真实推理结果卡与批量 CSV", "在验证集选定的模型上运行单张示例和同目录批量推理，记录类别概率、重点特征和单图分段耗时。卡片用于说明输入到预测的交互链路；批量样例属于演示数据，评估结果应从冻结测试集预测清单计算。"),
    "F40": (5, "五路特征的 CPU 耗时", "将实际特征提取计时按模块汇总，并用处理张数计算平均开销。该图可用于讨论离散 Radon 网格、邻域运算和缓存的工程取舍。分支耗时不包含读数据、写特征表、模型拟合或加载，报告应注明模式、线程和样本量。"),
    "F41": (3, "分层随机与按 lot 拆分的泛化对比", "在相同任务、模式、特征和配置下比较随机分层与按 lot 拆分的共享模型。分组实验不让同一 lot 跨集合，但测试样本和类别支持也随拆分变化。分数差异用于讨论泛化难度，不能归因于单一因素。"),
    "F42": (5, "Radon 开关的 CPU 与效果对照", "在相同样本拆分和随机森林设置下，将全集特征与去掉 Radon 特征的模型作对照。比较实测特征分支计时之和、模型训练耗时和冻结测试 Macro-F1。分支计时不包含磁盘读写或框架启动，适合讨论方向特征的计算取舍。"),
}


def setup_style():
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for path in (fonts / "msyh.ttc", fonts / "simhei.ttf", Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")):
        if path.exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white", "axes.labelcolor": INK,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
                         "svg.fonttype": "path", "axes.grid": True, "grid.alpha": .16, "axes.axisbelow": True})


def new(rows=1, cols=1, size=(8.5, 4.4)):
    return plt.subplots(rows, cols, figsize=size, layout="constrained", squeeze=False)


def wafer_panel(ax, a, title=""):
    ax.imshow(PALETTE[a], interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def mask_panel(ax, a, title):
    ax.imshow(a, interpolation="nearest", cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def footer(fig, text):
    fig.supxlabel(text, fontsize=8, color=MUTE, ha="left", x=.025)


def result_card(w, result, path):
    setup_style()
    fig, axes = new(2, 2, (9.2, 6.2))
    wafer_panel(axes[0, 0], w.a, "晶圆状态")
    mask_panel(axes[1, 0], w.defect, "缺陷掩码")
    ax = axes[0, 1]
    names, probabilities = zip(*reversed(result["top3"]))
    ax.barh(names, probabilities, color=[MUTE, BLUE, TEAL])
    ax.set(xlim=(0, 1), xlabel="预测概率", title=f"识别结果  {result['prediction']}")
    ax = axes[1, 1]
    ax.axis("off")
    lines = [f"总耗时  {result['elapsed_ms']:.2f} ms", ""]
    lines += [f"{r['feature'].split('.')[-1]}  {r['value']:.3f}  {r['description']}" for r in result["explanations"][:5]]
    ax.text(.01, .96, "\n".join(lines), va="top", linespacing=1.5, transform=ax.transAxes, fontsize=8.8)
    source = SYNTHETIC if result["source"] == "synthetic" else "WM-811K 模型"
    footer(fig, f"{source}  |  {result['run_id']}  |  {result['model']}")
    if not hasattr(path, "write"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, format="png")
    plt.close(fig)


class Report:
    def __init__(self, run):
        self.run = run
        self.meta = read_json(run / "run.json")
        self.metrics = read_json(run / "metrics.json")
        self.frozen = read_json(run / "frozen.json")
        self.progress = Progress()
        self.out = ASSETS / "runs" / run.name
        self.synthetic = self.meta["source"]["kind"] == "synthetic"
        self.source = self.meta["source"]["title"]
        self.catalog = []
        self.samples = read_csv(run / "samples" / "samples.csv")

    def path(self, figure_id, suffix):
        directory = self.out / CHAPTERS[CATALOG[figure_id][0]]
        directory.mkdir(parents=True, exist_ok=True)
        return directory / (figure_id + suffix)

    def add(self, figure_id, fig=None, source="", conclusion="", kind=None, data=None, existing=None, limitation=""):
        chapter, title, caption = CATALOG[figure_id]
        kind = kind or ("合成测试" if self.synthetic else "实测")
        png, vector = self.path(figure_id, ".png"), self.path(figure_id, ".svg")
        if fig is not None:
            fig.suptitle(title, fontsize=15, color=INK, fontweight="bold")
            label = "结构示意" if kind == "示意" else self.source
            footer(fig, f"{figure_id}   ·   {label}   ·   {self.run.name}")
            fig.savefig(png, dpi=300)
            fig.savefig(vector)
            plt.close(fig)
        if existing is not None and Path(existing) != png:
            shutil.copyfile(existing, png)
        if data is not None:
            write_csv(self.path(figure_id, ".csv"), data)
        record = dict(figure_id=figure_id, title_zh=title, chapter=CHAPTERS[chapter], kind=kind, run_id=self.run.name,
                      png=png.relative_to(self.out).as_posix() if png.exists() else "",
                      svg=vector.relative_to(self.out).as_posix() if vector.exists() else "",
                      source=source, source_csv=self.path(figure_id, ".csv").relative_to(self.out).as_posix() if data is not None else "",
                      generator="wafer/figures.py", caption=caption, key_finding=conclusion or "见图注。",
                      limitation=limitation or ("仅用于合成测试，不代表真实晶圆。" if self.synthetic and kind != "示意" else "适用条件见图注。"))
        write_json(self.path(figure_id, ".json"), record)
        self.catalog.append(record)
        self.progress("figure", f"{figure_id} · {title} · {kind}")

    def empty(self, figure_id, reason):
        self.path(figure_id, ".md").write_text(f"# {figure_id} {CATALOG[figure_id][1]}\n\n{reason}\n", encoding="utf-8")
        self.add(figure_id, kind="待补", limitation=reason, conclusion="未生成实测图。")

    def sample(self, row):
        return read_map(self.run / "samples" / row["npy"])

    def diagram(self, figure_id, lanes):
        fig, aa = new(size=(10.8, max(3.4, len(lanes) * 1.55)))
        ax = aa[0, 0]
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        rows = []
        for j, (label, boxes) in enumerate(lanes):
            y = 1 - (j + .59) / len(lanes)
            ax.text(.02, y + .17 / len(lanes), label, color=TEAL, fontsize=11, fontweight="bold", va="bottom")
            width = min(.22, .78 / len(boxes))
            centers = np.linspace(.13, .87, len(boxes))
            for i, (x, box) in enumerate(zip(centers, boxes)):
                height = .27 / len(lanes)
                ax.add_patch(patches.FancyBboxPatch((x-width/2, y-height/2), width, height, boxstyle="round,pad=0.02",
                                                   fc="#EAF5F4" if i % 2 == 0 else "#EDF1FA", ec="none"))
                ax.text(x, y, box, ha="center", va="center", fontsize=9)
                if i:
                    ax.annotate("", (x-width/2-.024, y), (centers[i-1]+width/2+.024, y),
                                arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.4))
                rows.append(dict(lane=label, order=i+1, node=box))
        self.add(figure_id, fig, "项目源码与运行协议", kind="示意", data=rows)


def input_figures(r):
    fig, axes = new(3, 3, (8, 8))
    examples = [next((row for row in r.samples if row["label"] == c), None) for c in CLASSES]
    for ax, label, row in zip(axes.ravel(), CLASSES, examples):
        if row:
            wafer_panel(ax, r.sample(row), f"{label} · {row['id']}")
        else:
            ax.set_title(f"{label} · 无样本")
            ax.axis("off")
    r.add("F01", fig, "samples/samples.csv", data=[row for row in examples if row])
    counts = read_csv(r.run / "classes.csv")
    names = [c["label"] for c in counts]
    f, aa = new(size=(9.8, 5.7))
    ax = aa[0, 0]
    idx = np.arange(len(names))
    for offset, field, color in ((-.18, "defect8", TEAL), (.18, "all9", BLUE)):
        values = [int(c[field]) for c in counts]
        ax.barh(idx+offset, values, .32, color=color, label=field)
    ax.set(yticks=idx, yticklabels=names, xscale="log", xlabel="样本数（对数坐标）")
    ax.invert_yaxis()
    ax.legend()
    rare = min((c for c in counts if int(c['all9']) > 0), key=lambda c: int(c['all9']))
    r.add("F02", f, "classes.csv", f"最少类 {rare['label']} 共 {rare['all9']} 张。", data=counts)
    f, aa = new(size=(9.4, 5.4))
    total = sum(int(c["all9"]) for c in counts)
    pct = [100 * int(c["all9"]) / total for c in counts]
    bars = aa[0, 0].barh(names, pct, color=TEAL)
    aa[0, 0].bar_label(bars, fmt="%.2f%%", padding=4)
    aa[0, 0].set(xlim=(0, max(pct)*1.15), xlabel="占可用标注样本比例（%）")
    aa[0, 0].invert_yaxis()
    r.add("F03", f, "classes.csv", data=[dict(label=n, percent=p) for n, p in zip(names, pct)])
    rows = read_csv(r.run / "dimensions.csv")
    f, aa = new(1, 2, (10, 4.2))
    h, w, n, weights = (np.array([int(row[key]) for row in rows]) for key in ("rows", "columns", "valid_cells", "count"))
    colors = aa[0, 0].scatter(w, h, c=np.log10(weights), cmap="viridis", s=14)
    f.colorbar(colors, ax=aa[0, 0], label="组合频数 log10")
    aa[0, 0].set(xlabel="宽（晶粒）", ylabel="高（晶粒）")
    aa[0, 1].hist(n, weights=weights, bins=24, color=TEAL)
    aa[0, 1].set(xlabel="有效晶粒数", ylabel="样本数")
    r.add("F04", f, "dimensions.csv")
    row = next((s for s in r.samples if s["label"] == "Scratch"), r.samples[0])
    a = r.sample(row)
    f, aa = new(1, 3, (9.4, 3.8))
    wafer_panel(aa[0, 0], a, f"状态图 · {row['id']}")
    mask_panel(aa[0, 1], a != 0, "有效晶粒  1 / 2")
    mask_panel(aa[0, 2], a == 2, "缺陷晶粒  2")
    r.add("F05", f, f"samples/{row['npy']}")
    f, aa = new(2, 3, (8.5, 6.4))
    pick = np.random.default_rng(r.meta["config"]["seed"]).choice(len(r.samples), min(6, len(r.samples)), replace=False)
    selected = [r.samples[i] for i in pick]
    for ax, s in zip(aa.ravel(), selected):
        image = r.sample(s)
        wafer_panel(ax, image, f"{s['label']} · {image.shape[0]} × {image.shape[1]}")
    r.add("F06", f, "samples/samples.csv", data=selected)
    return Wafer(a), row


def feature_figures(r, w, row):
    source = f"samples/{row['npy']}"
    f, aa = new(size=(6.2, 5.3))
    ax = aa[0, 0]
    wafer_panel(ax, w.a, row["id"])
    cy, cx, radius, _, _ = w.spatial_grid()
    for fraction in (1/3, 2/3, 1):
        ax.add_patch(patches.Circle((cx, cy), radius*fraction, ec="#F5CA52", lw=1.5, fc="none"))
    for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
        ax.plot([cx, cx+radius*np.cos(theta)], [cy, cy+radius*np.sin(theta)], lw=1.2, color="#F5CA52")
    ax.set(xlim=(-.5, w.a.shape[1]-.5), ylim=(w.a.shape[0]-.5, -.5))
    r.add("F09", f, source, data=[dict(center_row=cy, center_column=cx, radius=radius)])
    indexed, sizes = components(w)
    f, aa = new(size=(6.2, 5.3))
    ax = aa[0, 0]
    wafer_panel(ax, w.a, row["id"])
    ax.scatter(cx, cy, marker="+", s=100, color=BLUE, label="有效区质心")
    stats = []
    if sizes.size:
        yy, xx = np.where(w.defect)
        ax.scatter(xx.mean(), yy.mean(), marker="x", s=65, color="#F5CA52", label="缺陷质心")
        y, x = np.where(indexed == sizes.argmax()+1)
        ax.add_patch(patches.Rectangle((x.min()-.5, y.min()-.5), np.ptp(x)+1, np.ptp(y)+1, ec=TEAL, lw=2, fill=False))
        g = geometry(w, {})
        direction = .5*np.arctan2(g["axis_sin2"], g["axis_cos2"])
        length = max(np.ptp(x), np.ptp(y)) / 2
        if g["axis_sin2"] or g["axis_cos2"]:
            ax.plot(x.mean()+np.array([-1,1])*length*np.cos(direction), y.mean()+np.array([-1,1])*length*np.sin(direction),
                    color="white", lw=1.5, label="最大域主轴")
        stats = [dict(feature=k, value=v) for k, v in g.items()]
    ax.legend(loc="upper right", fontsize=8)
    r.add("F10", f, source, data=stats)
    f, aa = new(1, 2, (9.3, 4.5))
    aa[0, 0].imshow(np.ma.masked_where(indexed == 0, indexed % 19 + 1), cmap="tab20", interpolation="nearest")
    aa[0, 0].axis("off")
    aa[0, 0].set_title("连通域编号着色")
    ranked = np.sort(sizes)[::-1][:18]
    aa[0, 1].bar(np.arange(1, len(ranked)+1), ranked, color=TEAL)
    aa[0, 1].set(xlabel="按面积排序的连通域", ylabel="晶粒数")
    r.add("F11", f, source, data=[dict(component=i, size=int(s)) for i, s in enumerate(sizes, 1)])
    f, aa = new(1, 2, (10, 4.1))
    rows = []
    for ax, name, (bad, support, rates) in zip(aa.ravel(), ("行", "列"), projections(w)):
        ax.plot(np.where(support, rates, np.nan), color=TEAL, lw=1.8)
        ax.set(xlabel=f"{name}索引", ylabel="缺陷密度", ylim=(0, 1.02))
        rows += [dict(axis=name, position=i, defect=int(b), valid=int(s), density=float(d)) for i, (b,s,d) in enumerate(zip(bad,support,rates))]
    r.add("F12", f, source, data=rows)
    f, aa = new(1, 2, (10, 4.4))
    wafer_panel(aa[0, 0], w.a, f"{row['label']} · {row['id']}")
    settings = r.meta["signature"]["settings"]
    rays, valid = sinogram(w, settings)
    angle_step = 180/rays.shape[0]
    colors = aa[0, 1].imshow(rays, cmap="magma", aspect="auto", origin="lower",
                            extent=(-.5, rays.shape[1]-.5, -angle_step/2, 180-angle_step/2))
    aa[0, 1].set(xlabel="距离格索引", ylabel="投影法线角度（°）")
    f.colorbar(colors, ax=aa[0, 1], label="缺陷投影计数")
    r.add("F13", f, source, data=[dict(angle_deg=i*angle_step, bin=j, defects=int(rays[i,j]), valid=int(valid[i,j]))
                                  for i, j in np.ndindex(rays.shape)])
    _, groups, times = extract(w.a, settings, tuple(r.meta["signature"]["branches"]))
    write_csv(r.path("F13", "_sample_features.csv"), [dict(id=row["id"], branch=b, name=k, value=v, branch_ms=1000*times[b])
                                                      for b, values in groups.items() for k, v in values.items()])


def diagrams(r):
    r.diagram("F07", [("输入与表示", ["离散晶圆矩阵", "有效 / 缺陷掩码", "五路特征", "ID 对齐融合"]),
                      ("训练与评价", ["训练树模型", "验证集冻结", "测试集评估", "三端推理和图表"])])
    r.diagram("F08", [("输入检查", ["NPY / NPZ / PNG", "检查尺寸与状态", "构造两种掩码"]),
                      ("特征输入", ["原始网格统计", "Radon 最近邻网格", "提取特征并校验字段"])])
    r.diagram("F14", [("决策树", ["带标签特征", "选择分裂阈值", "递归节点", "叶节点概率"]),
                      ("随机森林", ["抽样与随机子特征", "独立训练多棵树", "汇总概率"]),
                      ("Boosting", ["初始化预测", "拟合更新方向", "逐轮累加", "分类概率"])])
    r.diagram("F15", [("Soft Voting", ["训练基础树", "验证集估计权重", "加权平均概率"]),
                      ("Stacking · 元训练", ["切分训练折", "用其余折拟合基础树", "生成留出折概率", "拟合元分类器"]),
                      ("Stacking · 推理", ["全训练集重拟合", "产生基础树概率", "元分类器输出"])])
    r.diagram("F32", [("交互依赖", ["网页 / Tkinter / CLI", "inference", "data + features", "模型概率"]),
                      ("实验依赖", ["preparation", "models + experiment", "运行产物", "figures"])])
    r.diagram("F33", [("训练", ["inspect", "extract", "train-all", "冻结模型", "evaluate"]),
                      ("推理", ["格式解析", "掩码", "特征", "模型概率", "结果卡 / CSV"]),
                      ("回放", ["模型和日志", "figures", "报告五章素材"])])


def bar_compare(r, figure_id, rows, fields, labels, source="leaderboard.csv", conclusion=""):
    if not rows:
        r.empty(figure_id, "运行没有该组实验。请使用 standard 或 full 模式。")
        return
    f, aa = new(size=(9.7, max(3.6, 1.95 + .46*len(rows))))
    ax = aa[0, 0]
    y = np.arange(len(rows))
    width = .73 / len(fields)
    for i, (field, label) in enumerate(zip(fields, labels)):
        values = [float(row[field]) for row in rows]
        bars = ax.barh(y + (i-(len(fields)-1)/2)*width, values, width*.91,
                      color=(TEAL, BLUE, CORAL)[i], label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=3)
    ax.set(yticks=y, yticklabels=[row["model"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(float(row[field]) for field in fields for row in rows)*1.14 + .001)
    ax.legend(loc="lower right", fontsize=8)
    r.add(figure_id, f, source, conclusion, data=rows)


def training_views(r):
    classes = r.metrics["classes"]
    f, aa = new(size=(10, 4.6))
    ax = aa[0, 0]
    x = np.arange(len(classes))
    rows = []
    for i, (part, by_class) in enumerate(r.meta["split_counts"].items()):
        counts = [by_class.get(c, 0) for c in classes]
        ax.bar(x+(i-1)*.25, counts, .23, color=(TEAL, BLUE, CORAL)[i], label=part)
        rows += [dict(part=part, label=c, count=n) for c, n in zip(classes, counts)]
    ax.set(xticks=x, xticklabels=classes, ylabel="样本数（对数坐标）", yscale="log")
    ax.legend()
    r.add("F16", f, "split_manifest.csv", data=rows)
    with np.load(r.run / "train_plot.npz", allow_pickle=False) as npz:
        matrix, y, names = (npz[k] for k in ("x", "y", "names"))
    scaled = StandardScaler().fit_transform(matrix)
    f, aa = new(size=(12.5, 5.6))
    cols = np.unique(np.linspace(0, len(names)-1, min(25, len(names)), dtype=int))
    picked = np.argsort(y, kind="stable")[np.linspace(0, len(y)-1, min(90, len(y)), dtype=int)]
    view = scaled[np.ix_(picked, cols)]
    colors = aa[0, 0].imshow(view, cmap="coolwarm", vmin=-3, vmax=3, aspect="auto", interpolation="nearest")
    aa[0, 0].set(xticks=np.arange(len(cols)), xticklabels=names[cols], ylabel="按类别排序的训练样本")
    aa[0, 0].tick_params(axis="x", labelrotation=66, labelsize=7)
    f.colorbar(colors, ax=aa[0, 0], label="z-score")
    r.add("F17", f, "train_plot.npz", data=[dict(label=classes[y[i]], **{names[c]: float(scaled[i,c]) for c in cols}) for i in picked])
    f, aa = new(size=(8.9, 5))
    pca = PCA(n_components=2)
    xy = pca.fit_transform(scaled)
    ax = aa[0, 0]
    for i, label in enumerate(classes):
        ax.scatter(*xy[y == i].T, label=label, s=16, alpha=.64)
    p1, p2 = 100*pca.explained_variance_ratio_
    ax.set(xlabel=f"PC1（{p1:.1f}%）", ylabel=f"PC2（{p2:.1f}%）")
    ax.legend(loc="best", fontsize=8, ncol=2)
    r.add("F18", f, "train_plot.npz", f"前两个主成分合计解释 {p1+p2:.1f}% 的训练特征方差。",
          data=[dict(label=classes[i], pc1=float(a), pc2=float(b)) for i, (a,b) in zip(y, xy)])


def model_views(r):
    best = r.metrics["best"]
    rows = r.metrics["results"]
    main = [row for row in rows if row["group"] in {"model", "ensemble"}]
    bar_compare(r, "F19", main, ("val_macro_f1", "macro_f1"), ("验证", "冻结测试"),
                conclusion=f"按验证分数选定 {best}。")
    bar_compare(r, "F20", main, ("accuracy", "balanced_accuracy"), ("Accuracy", "Balanced Accuracy"))
    f, aa = new(1, 2, (11, max(4, 1.8+.42*len(main))))
    for ax, field, title, color in zip(aa.ravel(), ("train_s", "size_kib"), ("训练秒数", "压缩模型体积 KiB"), (TEAL, BLUE)):
        ax.barh([row["model"] for row in main], [row[field] for row in main], color=color)
        ax.invert_yaxis()
        ax.set_xlabel(title)
    r.add("F21", f, "leaderboard.csv", data=main)
    bar_compare(r, "F22", main, ("infer_p50_ms", "infer_p95_ms"), ("中位数 ms", "P95 ms"))
    classes = r.metrics["classes"]
    matrix = np.asarray(r.metrics["confusion"][best])
    for figure_id, table, fmt in (("F23", matrix, "d"), ("F24", matrix / matrix.sum(axis=1, keepdims=True), ".2f")):
        f, aa = new(size=(8, 6.3))
        ax = aa[0, 0]
        colors = ax.imshow(table, cmap="Blues")
        ax.grid(False)
        for i, j in np.ndindex(table.shape):
            ax.text(j, i, f"{table[i,j]:{fmt}}", ha="center", va="center", fontsize=8.5,
                    color="white" if table[i,j] > table.max()*.57 else INK)
        ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)), xticklabels=classes, yticklabels=classes,
               xlabel="预测标签", ylabel="真实标签", title=best)
        ax.tick_params(axis="x", labelrotation=30)
        f.colorbar(colors, ax=ax)
        r.add(figure_id, f, "metrics.json", data=[dict(label=c, **dict(zip(classes, line.tolist()))) for c, line in zip(classes, table)])
    cls_rows = [dict(row, model=f"{row['label']} (n={row['support']})") for row in read_csv(r.run / "per_class.csv") if row["model"] == best]
    bar_compare(r, "F25", cls_rows, ("precision", "recall", "f1"), ("Precision", "Recall", "F1"), "per_class.csv")
    baseline = [row for row in rows if row["model"] == "random_forest"]
    for figure_id, group in (("F26", "features"), ("F27", "balance")):
        extra = [row for row in rows if row["group"] == group]
        bar_compare(r, figure_id, baseline+extra if extra else [], ("val_macro_f1", "macro_f1"), ("验证", "冻结测试"))
    ensemble = [row for row in rows if row["model"] == r.metrics["best_single"] or row["group"] == "ensemble"]
    bar_compare(r, "F28", ensemble if len(ensemble) >= 2 else [], ("val_macro_f1", "macro_f1"), ("验证", "冻结测试"))
    importance = read_csv(r.run / "importance.csv")[:20]
    f, aa = new(size=(10.2, 7.2))
    values = [float(row["validation_delta"]) for row in importance]
    ax = aa[0, 0]
    ax.barh([row["feature"] for row in importance], values, color=[TEAL if v >= 0 else CORAL for v in values])
    ax.axvline(0, color=MUTE, linewidth=.8)
    ax.invert_yaxis()
    ax.set_xlabel("打乱前后 Macro-F1 差值")
    r.add("F29", f, "importance.csv", data=importance)
    r.empty("F30", "可选 SHAP 实验尚未执行。可使用 F29 和 F31 配合讨论。")
    errors = read_csv(r.run / "errors.csv")
    if errors:
        f, aa = new(int(np.ceil(len(errors)/4)), 4, (11, max(3.3, 2.45*np.ceil(len(errors)/4))))
        for ax in aa.ravel():
            ax.axis("off")
        for ax, row in zip(aa.ravel(), errors):
            wafer_panel(ax, read_map(r.run / "errors" / row["file"]),
                        f"{row['truth']} → {row['prediction']}\n{row['id']} · {float(row['confidence']):.1%}")
        r.add("F31", f, "errors.csv + errors/*.png", data=errors)
    else:
        r.empty("F31", "本运行冻结测试集没有误分类。")


def process_views(r):
    logs = [json.loads(line) for line in (r.run / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    stages = list(dict.fromkeys(row["stage"] for row in logs))
    f, aa = new(size=(10.3, 4.6))
    ax = aa[0, 0]
    for i, stage in enumerate(stages):
        times = [row["elapsed_s"] for row in logs if row["stage"] == stage]
        ax.scatter(times, [i]*len(times), color=(TEAL, BLUE, CORAL)[i%3], alpha=.72, s=30)
    ax.set(yticks=np.arange(len(stages)), yticklabels=stages, xlabel="累计墙钟时间（秒）")
    ax.invert_yaxis()
    r.add("F34", f, "progress.jsonl", data=logs)
    curves = read_csv(r.run / "boosting.csv")
    models = list(dict.fromkeys(row["model"] for row in curves))
    if models:
        f, aa = new(int(np.ceil(len(models)/2)), min(2, len(models)), (11, 3.8*int(np.ceil(len(models)/2))))
        for ax, name in zip(aa.ravel(), models):
            rows = [row for row in curves if row["model"] == name]
            for key, label, color in (("train", "训练", TEAL), ("val", "验证", CORAL)):
                ax.plot([int(row["iteration"]) for row in rows], [float(row[key]) for row in rows], color=color, label=label)
            ax.set(xlabel="Boosting 轮次", ylabel="Log loss", title=name)
            ax.legend(fontsize=8)
        for ax in aa.ravel()[len(models):]:
            ax.axis("off")
        r.add("F35", f, "boosting.csv", data=curves)
    else:
        r.empty("F35", "该运行无 Boosting 迭代日志。")
    for figure_id in ("F36", "F37", "F38"):
        screenshot = r.run / "screenshots" / f"{figure_id}.png"
        if screenshot.is_file():
            r.add(figure_id, source=f"screenshots/{figure_id}.png", kind="截图", existing=screenshot)
        else:
            r.empty(figure_id, "此运行没有保存该界面的实际截图。按 README 的交互验收步骤操作即可补录。")
    model_file = r.run / "models" / f"{r.metrics['best']}.joblib"
    if model_file.is_file():
        bundle = load_model(model_file)
        sample = next(row for row in r.samples if row["label"] in bundle["classes"])
        w, result = predict(bundle, path=r.run / "samples" / sample["npy"])
        write_json(r.path("F39", "_prediction.json"), result)
        result_card(w, result, r.path("F39", ".png"))
        csv_path = r.path("F39", "_batch.csv")
        batch = predict_directory(bundle, r.run / "samples", csv_path, Progress(quiet=True))
        r.add("F39", source=f"samples/{sample['npy']} + models/{model_file.name}",
              conclusion=f"示例识别为 {result['prediction']}，批量共处理 {len(batch)} 个文件（PNG / NPY 各一份）。", data=batch)
    else:
        r.empty("F39", "缺少主模型文件。请重训或使用完整交付包。")
    f, aa = new(size=(8.6, 4.6))
    n = sum(sum(p.values()) for p in r.meta["split_counts"].values())
    timings = r.meta["feature_seconds"]
    data = [dict(branch=k, total_seconds=v, mean_ms=1000*v/n, n=n) for k, v in timings.items()]
    aa[0, 0].barh([d["branch"] for d in data], [d["mean_ms"] for d in data], color=TEAL)
    aa[0, 0].invert_yaxis()
    aa[0, 0].set_xlabel("每张平均提取耗时（ms）")
    r.add("F40", f, "run.json", data=data)


def comparison(r, compare_run):
    if compare_run is None:
        r.empty("F41", "使用 --compare-run 提供同模式的另一拆分运行。")
    else:
        other = read_json(compare_run / "run.json")
        keys = ("source", "task", "seed", "mode", "settings", "branches", "fractions", "version")
        if any(r.meta["signature"][k] != other["signature"][k] for k in keys) or r.meta["split"] == other["split"] or r.meta["config"] != other["config"]:
            raise ValueError("比较要求相同数据、特征、任务、模式、配置及不同拆分策略。")
        a, b = r.metrics["results"], read_json(compare_run / "metrics.json")["results"]
        second = {row["model"]: row for row in b}
        rows = [dict(model=row["model"], primary=row["macro_f1"], compare=second[row["model"]]["macro_f1"],
                     primary_n=row["test_n"], compare_n=second[row["model"]]["test_n"])
                for row in a if row["group"] in {"model", "ensemble"} and row["model"] in second]
        bar_compare(r, "F41", rows, ("primary", "compare"), (r.meta["split"], other["split"]),
                    f"{r.run.name}/metrics.json + {compare_run.name}/metrics.json")
    ablation = next((x for x in r.metrics["results"] if x["model"] == "without_radon"), None)
    baseline = next((x for x in r.metrics["results"] if x["model"] == "random_forest"), None)
    if ablation and baseline:
        source_time = r.meta["feature_seconds"]
        total = sum(source_time.values())
        rows = [dict(model="五路齐用", measured_branch_sum_s=total, train_s=baseline["train_s"], macro_f1=baseline["macro_f1"]),
                dict(model="移除 Radon", measured_branch_sum_s=total-source_time["radon"], train_s=ablation["train_s"], macro_f1=ablation["macro_f1"])]
        f, aa = new(1, 3, (11.5, 3.7))
        for ax, key, title in zip(aa.ravel(), ("measured_branch_sum_s", "train_s", "macro_f1"), ("分支计时合计（秒）", "训练秒数", "测试 Macro-F1")):
            bars = ax.bar([row["model"] for row in rows], [row[key] for row in rows], color=[TEAL, BLUE])
            ax.bar_label(bars, fmt="%.3f")
            ax.set_title(title)
            ax.margins(y=.16)
        r.add("F42", f, "run.json + leaderboard.csv", data=rows,
              limitation="提取时间为各实测分支计时之和；移除项用减去该分支计时估计，不是另一次冷启动。")
    else:
        r.empty("F42", "需运行包含 without_radon 的 standard 或 full 模式。")


def make_figures(run, compare_run=None):
    setup_style()
    report = Report(run)
    w, row = input_figures(report)
    diagrams(report)
    feature_figures(report, w, row)
    training_views(report)
    model_views(report)
    process_views(report)
    comparison(report, compare_run)
    write_csv(report.out / "figure_catalog.csv", sorted(report.catalog, key=lambda row: row["figure_id"]))
    combined = []
    for path in sorted((ASSETS / "runs").glob("*/figure_catalog.csv")):
        for row in read_csv(path):
            for key in ("png", "svg", "source_csv"):
                if row[key]:
                    row[key] = f"runs/{path.parent.name}/{row[key]}"
            combined.append(row)
    write_csv(ASSETS / "figure_catalog.csv", combined)
    tables = report.out / "tables"
    tables.mkdir(exist_ok=True)
    for file in ("classes.csv", "dimensions.csv", "split_manifest.csv", "feature_dictionary.csv", "leaderboard.csv",
                 "per_class.csv", "importance.csv", "errors.csv", "boosting.csv", "frozen.json", "run.json", "data_audit.json"):
        shutil.copyfile(run / file, tables / file)
    env = report.meta["environment"]
    write_csv(tables / "environment.csv", [dict(item=k, value=v) for k, v in env.items() if k != "packages"] +
              [dict(item=k, value=v) for k, v in env["packages"].items()])
    print(report.out)
    return report.out
