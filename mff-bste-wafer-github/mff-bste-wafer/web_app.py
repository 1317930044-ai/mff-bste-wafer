import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from wafer.common import ASSETS, RUNS, SYNTHETIC, labels, read_csv, read_json
from wafer.data import PALETTE
from wafer.features import projections, sinogram
from wafer.inference import available_models, available_samples, batch_csv, batch_row, load_model, predict

st.set_page_config(page_title="晶圆缺陷识别", page_icon="◉", layout="wide")
st.markdown("""<style>
[data-testid="stMainBlockContainer"]{max-width:1320px;padding-top:1.8rem}
[data-testid="stSidebar"]{border-right:1px solid #dce7ed}
[data-testid="stMetric"]{background:#edf3f6;border-radius:12px;padding:12px 16px}
[data-testid="stMetricValue"]{font-size:1.55rem}
[data-testid="stImage"] img{image-rendering:pixelated;border-radius:8px}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model(path, mtime):
    return load_model(Path(path))


def source_notice(bundle):
    if bundle["source"] == "synthetic":
        st.warning(f"{SYNTHETIC}。此模型用于检查流程，真实晶圆实验请使用 WM-811K 运行。")


def show_prediction(w, result, settings, branches):
    images, panel = st.columns((1.16, 1), gap="large")
    original, mask = images.columns(2)
    original.image(PALETTE[w.a], caption="原始状态图", width="stretch", output_format="PNG")
    mask.image(w.defect.astype(np.uint8) * 255, caption="缺陷掩码", width="stretch", output_format="PNG")
    images.caption(f"{w.a.shape[0]} × {w.a.shape[1]} 晶粒 · 有效 {w.valid_count:,} · 缺陷 {w.defect_count:,}")
    with panel:
        a, b = st.columns(2)
        a.metric("预测类别", result["prediction"])
        ms = result["elapsed_ms"]
        b.metric("端到端耗时", f"{ms/1000:.2f} s" if ms >= 1000 else f"{ms:.1f} ms")
        st.markdown("**Top-3 概率**")
        for name, p in result["top3"]:
            st.progress(min(1.0, max(0.0, p)), text=f"{name} · {p:.1%}")
    with st.expander("预处理、投影和特征解释"):
        panels = st.columns(3)
        panels[0].image(w.valid.astype(np.uint8) * 255, caption="有效区掩码", width=165)
        for col, key, (_, _, rates) in zip(panels[1:], ("行密度", "列密度"), projections(w)):
            col.line_chart(pd.DataFrame({key: rates}), height=160)
        if "radon" in branches:
            rays, _ = sinogram(w, settings)
            st.image(np.rint(255*rays/max(1, rays.max())).astype(np.uint8), caption="Radon 缺陷投影（行对应角度）", width=310)
        st.dataframe(pd.DataFrame(result["explanations"]).rename(columns={"feature": "特征", "value": "数值", "description": "含义"}), hide_index=True)
        st.dataframe(pd.DataFrame([{"阶段": name, "毫秒": ms} for name, ms in result["timings_ms"].items()]), hide_index=True)
    with st.expander("下载结果"):
        st.download_button("下载推理 JSON", json.dumps(result, indent=2, ensure_ascii=False), "wafer_result.json", "application/json")
        if st.button("生成 PNG 结果卡"):
            from wafer.figures import result_card

            buffer = io.BytesIO()
            result_card(w, result, buffer)
            st.session_state["result_card"] = (result["run_id"], result["elapsed_ms"], buffer.getvalue())
        card = st.session_state.get("result_card")
        if card and card[:2] == (result["run_id"], result["elapsed_ms"]):
            st.download_button("下载结果卡", card[2], "wafer_result.png", "image/png")


def single_page(bundle, task):
    choice, field = st.columns((1, 1.65), gap="medium")
    mode = choice.radio("输入方式", ("项目样本", "上传文件"), horizontal=True)
    samples = [row for row in available_samples() if row["label"] in labels(task)]
    sample = upload = None
    if mode == "项目样本" and samples:
        chosen = field.selectbox("晶圆样本", range(len(samples)), format_func=lambda i: f"{samples[i]['label']} · {samples[i]['id']} · {samples[i]['source']}")
        sample = samples[chosen]
    else:
        upload = field.file_uploader("晶圆文件", type=["npy", "npz", "png"], help="NPZ 使用 wafer 键。PNG 必须使用项目固定颜色。")
    if sample and sample["kind"] == "synthetic":
        st.info(SYNTHETIC)
    if upload is None and sample is None:
        return
    input_key = (mode, upload.file_id if upload else str(sample["path"]), bundle["run_id"], bundle["name"])
    if st.button("识别晶圆", type="primary"):
        try:
            with st.spinner("解析输入、构建掩码、提取特征并预测…"):
                w, result = predict(bundle, path=sample["path"]) if sample else predict(bundle, content=upload.getvalue(), filename=upload.name)
            st.session_state["prediction"] = (input_key, w, result)
        except (ValueError, OSError, TypeError, EOFError) as exc:
            st.error(f"输入读取失败。{exc}")
            st.session_state.pop("prediction", None)
    saved = st.session_state.get("prediction")
    if saved and saved[0] == input_key:
        show_prediction(saved[1], saved[2], bundle["settings"], bundle["branches"])


def batch_page(bundle, task):
    mode = st.radio("批量来源", ("项目样本（有标签）", "上传多张"), horizontal=True)
    if mode == "项目样本（有标签）":
        kind = bundle["source"]
        items = [(row["path"], row["label"]) for row in available_samples() if row["kind"] == kind and row["label"] in labels(task)]
        st.caption(f"{len(items)} 张 · {SYNTHETIC if kind == 'synthetic' else 'WM-811K'}")
    else:
        files = st.file_uploader("批量晶圆文件", type=["npy", "npz", "png"], accept_multiple_files=True)
        items = [(file, None) for file in files]
    batch_key = (bundle["run_id"], bundle["name"], mode, tuple(getattr(file, "file_id", file.name) for file, _ in items))
    if len(items) > 100:
        st.error("每批最多处理 100 个文件。")
        return
    if st.button("批量识别", disabled=not items):
        progress = st.progress(0, text="准备识别")
        rows = []
        for i, (file, truth) in enumerate(items, 1):
            rows.append(batch_row(bundle, file, truth))
            progress.progress(i/len(items), text=f"已处理 {i}/{len(items)}")
        st.session_state["batch"] = (batch_key, rows)
    saved = st.session_state.get("batch")
    if saved and saved[0] == batch_key:
        rows = saved[1]
        distribution = Counter(row["prediction"] for row in rows if not row["error"])
        if distribution:
            st.bar_chart(pd.Series(distribution, name="预测数"), height=180)
        st.dataframe(pd.DataFrame(rows).replace("", None), hide_index=True)
        errors = [row for row in rows if row["correct"] is False or row["error"]]
        if errors:
            with st.expander(f"错误或无法解析的样本（{len(errors)}）"):
                st.dataframe(pd.DataFrame(errors).replace("", None), hide_index=True)
        st.download_button("下载批量 CSV", batch_csv(rows), "wafer_batch.csv", "text/csv")


def run_page():
    runs = sorted((path.parent for path in RUNS.glob("*/progress.jsonl")), key=lambda p: (p.name.startswith("demo-"), p.name))
    if not runs:
        st.info("暂无训练记录。请从 CLI 运行 `python main.py train-all --synthetic --run-id demo-web`。")
        return
    selected = st.selectbox("运行 ID", runs, format_func=lambda p: p.name)
    st.button("刷新日志")
    log = []
    for line in (selected / "progress.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            log.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not log:
        return
    last = log[-1]
    most_recent_model = next((r for r in reversed(log) if r.get("model")), {})
    sample_count = next((r["training_samples"] for r in reversed(log) if "training_samples" in r), "—")
    metadata = read_json(selected / "run.json") if (selected / "run.json").exists() else {}
    if metadata.get("source", {}).get("kind") == "synthetic" or selected.name.startswith("demo-synthetic"):
        st.warning(SYNTHETIC)
    stats = st.columns(4)
    stats[0].metric("阶段", last["stage"])
    stats[1].metric("最近模型", most_recent_model.get("model", "—"))
    stats[2].metric("训练样本", sample_count)
    stats[3].metric("累计耗时", f"{last['elapsed_s']:.1f} s")
    if last.get("total"):
        st.progress(min(1, last["done"]/last["total"]), text=last["message"])
    st.dataframe(pd.DataFrame(log[-100:][::-1]), hide_index=True, height=292)
    if (selected / "leaderboard.csv").is_file():
        st.subheader("模型对比")
        st.dataframe(pd.read_csv(selected / "leaderboard.csv").drop(columns=["algorithm", "group"]), hide_index=True)
    catalog = ASSETS / "runs" / selected.name / "figure_catalog.csv"
    if catalog.is_file():
        rows = [row for row in read_csv(catalog) if row["png"]]
        item = st.selectbox("浏览报告图表", rows, format_func=lambda r: f"{r['figure_id']} · {r['title_zh']}")
        if item:
            st.image(str(catalog.parent / item["png"]), caption=item["caption"])


with st.sidebar:
    st.header("模型与输入")
    task = st.radio("分类任务", ("defect8", "all9"), format_func=lambda value: {"defect8": "8 类缺陷", "all9": "9 类（含正常）"}[value])
    models = available_models(task)
    selected = st.selectbox("本地模型", models, format_func=lambda pair: f"{pair[0].stem} · {pair[0].parent.parent.name}") if models else None
    if selected:
        st.caption(f"训练来源 · {selected[1]}")
    st.markdown("**输入格式**")
    st.caption("NPY、含 wafer 的 NPZ 或项目导出的原尺寸 PNG。每个文件最多 8 MiB。普通照片和截图没有晶粒状态编码。")

st.title("晶圆缺陷识别")
st.markdown("离散晶粒图 · 多视角特征融合 · 决策树集成")
single, batch, runs = st.tabs(("单张识别", "批量识别", "实验与训练日志"))
bundle = None
if selected:
    path = selected[0]
    try:
        bundle = get_model(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError, ImportError) as exc:
        st.error(f"无法加载本地模型。{exc}")
else:
    st.warning("该任务暂无已完成的模型。请按 README 训练，或先用合成数据试跑。")
if bundle:
    source_notice(bundle)
    with single:
        single_page(bundle, task)
    with batch:
        batch_page(bundle, task)
with runs:
    run_page()
