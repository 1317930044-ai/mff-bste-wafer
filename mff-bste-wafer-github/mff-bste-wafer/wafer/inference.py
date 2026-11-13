import csv
import io
import os
import time
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .common import FEATURE_VERSION, RUNS, SAMPLES, read_csv, read_json, write_csv
from .data import FORMATS, decode, read_map
from .features import describe_feature, extract
from .models import probability


def load_model(path):
    bundle = joblib.load(path)  # Only locally produced / explicitly trusted files.
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("模型特征版本与程序不一致，请重新训练。")
    return bundle


def available_models(task):
    models = []
    for meta_path in RUNS.glob("*/run.json"):
        try:
            meta = read_json(meta_path)
            if meta["task"] != task or meta["status"] != "complete":
                continue
            metric = read_json(meta_path.parent / "metrics.json")
            for row in metric["results"]:
                file = meta_path.parent / "models" / f"{row['model']}.joblib"
                if file.is_file():
                    order = (meta["source"]["kind"] != "wm811k", meta["mode"] == "quick", meta["split"] == "grouped",
                             row["model"] != metric["best"], row["group"] in {"features", "balance"}, str(file))
                    models.append((order, file, meta["source"]["title"]))
        except (OSError, ValueError, KeyError):
            continue
    return [(file, source) for _, file, source in sorted(models)]


def available_samples():
    rows = []
    seen = set()
    manifests = sorted(SAMPLES.glob("*/samples.csv")) + sorted(RUNS.glob("*/samples/samples.csv"))
    for path in manifests:
        for row in read_csv(path):
            key = (row["id"], row["kind"])
            if key not in seen and (path.parent / row["npy"]).is_file():
                seen.add(key)
                rows.append(dict(row, path=path.parent / row["npy"]))
    return sorted(rows, key=lambda r: (r["kind"] != "wm811k", r["label"], r["id"]))


def predict(bundle, *, path=None, content=None, filename=None):
    start = time.perf_counter()
    a = read_map(path) if path is not None else decode(content, filename)
    input_seconds = time.perf_counter() - start
    w, groups, times = extract(a, bundle["settings"], bundle["branches"], validated=True)
    features = {key: value for group in groups.values() for key, value in group.items()}
    if list(features) != bundle["names"]:
        raise ValueError("提取特征的字段顺序与模型记录不一致。")
    x = np.asarray([list(features.values())], dtype=np.float32)
    fit_started = time.perf_counter()
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
    with threadpool_limits(limits=2):
        p = probability(bundle["model"], x, len(bundle["classes"]))[0]
    model_seconds = time.perf_counter() - fit_started
    prediction = bundle["classes"][int(p.argmax())]
    ranking = sorted(zip(bundle["classes"], p.tolist()), key=lambda row: -row[1])
    selected = [k for k in ("spatial.defect_share", "spatial.radial_mean", "spatial.centroid_distance",
                           "geometry.largest_share", "radon.directionality", "texture.isolated_share", "projection.row_max")
                if k in features] or list(features)[:5]
    explanations = [dict(feature=k, value=features[k], description=describe_feature(k)[2]) for k in selected]
    return w, dict(prediction=prediction, top3=ranking[:3], probability=dict(ranking), features=features,
                   explanations=explanations, timings_ms={k: 1000 * v for k, v in dict(input=input_seconds, **times,
                                                                                   prediction=model_seconds).items()},
                   elapsed_ms=1000 * (time.perf_counter() - start), model=bundle["name"], run_id=bundle["run_id"],
                   source=bundle["source"], task=bundle["task"])


def batch_row(bundle, source, truth=None):
    name = source.name
    in_scope = truth in bundle["classes"] if truth else ""
    try:
        if isinstance(source, Path):
            _, result = predict(bundle, path=source)
        else:
            _, result = predict(bundle, content=source.getvalue(), filename=name)
        return dict(file=name, prediction=result["prediction"], truth=truth or "",
                    truth_in_task=in_scope, correct=(result["prediction"] == truth) if in_scope else "", confidence=result["top3"][0][1],
                    elapsed_ms=result["elapsed_ms"], error="", **{f"p_{k}": v for k, v in result["probability"].items()})
    except (ValueError, OSError, TypeError, EOFError) as exc:
        return dict(file=name, prediction="", truth=truth or "", truth_in_task=in_scope, correct="", confidence="", elapsed_ms="", error=str(exc))


def batch_csv(rows):
    file = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(file, fields)
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + file.getvalue()


def predict_directory(bundle, directory, out, progress):
    items = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in (*FORMATS, ".jpg", ".jpeg"))
    if not items:
        raise ValueError("目录没有晶圆输入文件。")
    manifest = directory / "samples.csv"
    truth = {row[k]: row["label"] for row in read_csv(manifest) for k in ("npy", "png")} if manifest.exists() else {}
    rows = []
    for i, item in enumerate(items, 1):
        rows.append(batch_row(bundle, item, truth.get(item.name)))
        progress("predict", item.name, i, len(items))
    write_csv(out, rows)
    return rows
