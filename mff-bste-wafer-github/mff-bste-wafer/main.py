import argparse
import json
import multiprocessing
import sys
import threading
import time
from pathlib import Path

from wafer.common import (ARTIFACTS, BASE, BRANCHES, CONFIG, OPTIONAL, ROOT, STRATEGIES, Progress, environment,
                          load_config, run_path, write_json)


def source_args(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=Path, help="可信 LSWMD.pkl 或包含该文件的 ZIP")
    source.add_argument("--synthetic", action="store_true", help="使用明确标记的合成测试数据")


def experiment_args(parser):
    source_args(parser)
    parser.add_argument("--task", choices=("defect8", "all9"), default="defect8")
    parser.add_argument("--mode", choices=("quick", "standard", "full"), default="quick")
    parser.add_argument("--split", choices=("stratified", "grouped"), default="stratified")
    parser.add_argument("--no-radon", action="store_true", help="从特征提取和训练中关闭 Radon")
    parser.add_argument("--config", type=Path, default=CONFIG)


def parser_for_cli():
    parser = argparse.ArgumentParser(description="晶圆缺陷识别 · MFF-BSTE · Windows CPU")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env", help="检查 Python、依赖和 Tkinter")
    for name, description in (("inspect", "审计原始标签、矩阵和尺寸"), ("export-samples", "导出规范 PNG 和 NPY")):
        command = sub.add_parser(name, help=description)
        source_args(command)
        command.add_argument("--out", type=Path, help="输出目录")
        command.add_argument("--seed", type=int, default=42)
        if name == "export-samples":
            command.add_argument("--per-class", type=int, default=2)
    experiment_args(sub.add_parser("extract", help="确定性拆分、提取五路特征并缓存"))
    for name, description in (("train", "训练单个模型"), ("train-all", "运行模型矩阵、融合与消融"),
                              ("benchmark", "实际测量提取、训练、推理和峰值内存")):
        command = sub.add_parser(name, help=description)
        experiment_args(command)
        command.add_argument("--run-id", required=True, help="新运行的唯一 ID")
        if name == "train-all":
            command.add_argument("--external", action="store_true", help="尝试额外安装的三种 Boosting")
        else:
            command.add_argument("--model", default="random_forest", choices=(*BASE, *OPTIONAL, "soft_voting", "stacking"))
            command.add_argument("--balance", default="none", choices=STRATEGIES)
    for name, description in (("evaluate", "用冻结预测清单重算测试指标"), ("figures", "从实际运行输出重建报告图表")):
        command = sub.add_parser(name, help=description)
        command.add_argument("--run-id", required=True)
        if name == "figures":
            command.add_argument("--compare-run", help="比较同一数据/模式的另一拆分运行")
    single = sub.add_parser("predict", help="单张推理，可保存 JSON 与 PNG 卡片")
    single.add_argument("--model-path", type=Path, required=True)
    single.add_argument("--input", type=Path, required=True)
    single.add_argument("--out", type=Path, help="结果 JSON 路径")
    single.add_argument("--card", type=Path, help="结果卡 PNG 路径")
    batch = sub.add_parser("predict-batch", help="目录批量推理并保存 CSV")
    batch.add_argument("--model-path", type=Path, required=True)
    batch.add_argument("--input-dir", type=Path, required=True)
    batch.add_argument("--out", type=Path, default=ARTIFACTS / "predictions" / "batch.csv")
    return parser


def default_data():
    raw = ROOT / "data" / "raw"
    return next((p for p in (raw / "LSWMD.pkl", raw / "wm811k.zip") if p.exists()), raw / "LSWMD.pkl")


def benchmark(args, config, branches):
    import psutil

    from wafer.experiment import run_experiment
    from wafer.common import read_csv, read_json
    from wafer.inference import load_model, predict

    proc = psutil.Process()
    memory = [proc.memory_info().rss]
    done = threading.Event()

    def sample():
        while not done.wait(.05):
            memory[0] = max(memory[0], proc.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    args.command = "train"
    try:
        out = run_experiment(args, config, branches)
    finally:
        done.set()
        monitor.join()
    meta = read_json(out / "run.json")
    result = read_json(out / "metrics.json")["results"][0]
    bundle = load_model(out / "models" / f"{result['model']}.joblib")
    samples = [s for s in read_csv(out / "samples" / "samples.csv") if s["label"] in bundle["classes"]]
    end_to_end = [predict(bundle, path=out / "samples" / s["npy"])[1]["elapsed_ms"] for s in samples]
    log = (out / "progress.jsonl").read_text(encoding="utf-8")
    write_json(out / "benchmark.json", dict(environment=environment(), run_wall_s=time.perf_counter() - started,
               sampled_peak_rss_mib=memory[0] / 2**20, sample_interval_s=.05, jobs=config["jobs"],
               reused_feature_cache="复用已校验的特征缓存" in log, original_cache_extract_s=meta["extract_s"],
               feature_seconds=meta["feature_seconds"], end_to_end_ms=end_to_end, metrics=result))


def main():
    args = parser_for_cli().parse_args()
    cmd = args.command
    if cmd == "env":
        try:
            import tkinter
            tk_version = tkinter.TkVersion
        except ImportError:
            tk_version = "未安装"
        print(json.dumps(environment() | {"tk": tk_version}, indent=2, ensure_ascii=False))
        return

    from wafer.common import ASSETS, SAMPLES
    from wafer.data import collect, describe, export_samples
    from wafer.experiment import evaluate, run_experiment
    from wafer.inference import load_model, predict, predict_directory
    from wafer.preparation import prepare

    if hasattr(args, "synthetic"):
        args.data = args.data or default_data()
        if not args.synthetic and not args.data.is_file():
            raise ValueError(f"找不到 {args.data}。请下载官方数据，或显式使用 --synthetic。")
    if cmd in {"inspect", "export-samples"}:
        data, meta = collect(args.data, args.synthetic, args.seed, Progress(), audit=cmd == "inspect")
        if cmd == "inspect":
            dest = args.out or ASSETS / "data_audit" / meta["kind"]
            describe(data, meta, dest)
        else:
            dest = args.out or SAMPLES / meta["kind"]
            export_samples(data, meta, dest, args.seed, args.per_class)
        print(dest)
    elif cmd in {"extract", "train", "train-all", "benchmark"}:
        config = load_config(args.config)
        branches = tuple(b for b in BRANCHES if b != "radon" or not args.no_radon)
        if cmd == "extract":
            print(prepare(args.data, args.synthetic, args.task, args.mode, args.split, config, branches, Progress()).path)
        elif cmd == "benchmark":
            benchmark(args, config, branches)
        else:
            run_experiment(args, config, branches)
    elif cmd == "evaluate":
        print(json.dumps(evaluate(run_path(args.run_id))[:3], ensure_ascii=False, indent=2))
    elif cmd == "figures":
        from wafer.figures import make_figures

        make_figures(run_path(args.run_id), run_path(args.compare_run) if args.compare_run else None)
    elif cmd == "predict":
        bundle = load_model(args.model_path)
        w, result = predict(bundle, path=args.input)
        if args.out:
            write_json(args.out, result)
        if args.card:
            from wafer.figures import result_card

            result_card(w, result, args.card)
        print(json.dumps({k: v for k, v in result.items() if k != "features"}, ensure_ascii=False, indent=2))
    elif cmd == "predict-batch":
        rows = predict_directory(load_model(args.model_path), args.input_dir, args.out, Progress())
        print(json.dumps(dict(ok=sum(not r["error"] for r in rows), total=len(rows), csv=str(args.out)), ensure_ascii=False))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        main()
    except (OSError, ValueError, ImportError) as exc:
        sys.exit(f"错误：{exc}")
