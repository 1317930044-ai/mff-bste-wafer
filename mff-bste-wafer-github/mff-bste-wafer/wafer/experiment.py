import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from threadpoolctl import threadpool_limits

from .common import (FEATURE_VERSION, ROOT, Progress, environment, read_csv, read_json, run_path,
                     utc_now, write_csv, write_json)
from .data import export_png
from .models import BASE, OPTIONAL, STACK_BASES, STRATEGIES, fit_base, fit_stack, fit_voting, learning_curve, probability
from .preparation import prepare


@dataclass(frozen=True)
class Spec:
    key: str
    algorithm: str
    branches: tuple
    balance: str = "none"
    group: str = "model"


def design(args, branches):
    if args.command == "train":
        return [Spec(args.model, args.model, branches, args.balance)]
    options = BASE if args.mode != "quick" else ("decision_tree", "random_forest", "extra_trees", "hist_gradient")
    specs = [Spec(name, name, branches) for name in options]
    if args.mode != "quick":
        specs.extend(Spec(f"only_{b}", "random_forest", (b,), group="features") for b in branches)
        specs.extend(Spec(f"without_{b}", "random_forest", tuple(k for k in branches if b != k), group="features")
                     for b in branches if len(branches) > 1)
        specs.extend(Spec(f"rf_{s}", "random_forest", branches, s, "balance") for s in STRATEGIES[1:])
        specs.append(Spec("balanced_rf", "balanced_rf", branches, group="balance"))
        if args.mode == "full" or args.external:
            specs.extend(Spec(name, name, branches) for name in OPTIONAL[1:])
    specs += [Spec("soft_voting", "soft_voting", branches, group="ensemble"),
              Spec("stacking", "stacking", branches, "class_weight", "ensemble")]
    return specs


def metrics(y, pred, names):
    classes = np.arange(len(names))
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    scores = dict(macro_f1=float(np.mean(f1)), weighted_f1=float(np.average(f1, weights=support)),
                  accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred))
    by_class = [dict(label=name, precision=float(p), recall=float(r), f1=float(f), support=int(n))
                for name, p, r, f, n in zip(names, precision, recall, f1, support)]
    return scores, by_class, confusion_matrix(y, pred, labels=classes).tolist()


def params(model):
    return {key: (value if isinstance(value, (str, int, float, bool, type(None))) else str(value))
            for key, value in model.get_params(deep=False).items()}


def run_experiment(args, config, branches):
    out = run_path(args.run_id)
    out.mkdir(parents=True, exist_ok=False)
    progress = Progress(out / "progress.jsonl")
    try:
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(config["jobs"]))
        progress("start", f"{args.run_id} · {args.task} · {args.mode} · {args.split}")
        data = prepare(args.data, args.synthetic, args.task, args.mode, args.split, config, branches, progress)
        for name in ("split_manifest.csv", "feature_dictionary.csv", "classes.csv", "dimensions.csv", "data_audit.json"):
            shutil.copyfile(data.path / name, out / name)
        shutil.copytree(data.path / "samples", out / "samples")
        write_json(out / "run.json", dict(**data.meta, run_id=args.run_id, config=config, environment=environment(),
                   started_utc=utc_now(), task=args.task, mode=args.mode, split=args.split, status="training"))
        with threadpool_limits(limits=config["jobs"]), joblib.parallel_backend("threading", n_jobs=config["jobs"]):
            fit_and_evaluate(args, config, data, out, progress, branches)
        meta = read_json(out / "run.json")
        meta.update(status="complete", completed_utc=utc_now(), wall_s=time.perf_counter() - progress.start)
        write_json(out / "run.json", meta)
        progress("complete", f"运行成果：{out.relative_to(ROOT).as_posix()}")
    except Exception as exc:
        progress("failed", str(exc))
        raise
    return out


def fit_and_evaluate(args, config, data, out, progress, branches):
    train, val, test = (data.parts[p] for p in ("train", "val", "test"))
    settings, classes = config["modes"][args.mode], data.classes
    rng = np.random.default_rng(config["seed"])
    specs = design(args, branches)
    models, details, skipped, curves = {}, {}, [], []
    models_dir = out / "models"
    models_dir.mkdir()
    for number, spec in enumerate(specs, 1):
        progress("train", f"{spec.key} · 训练 {len(train)} 张", number, len(specs), model=spec.key, training_samples=len(train))
        cols = data.columns(spec.branches)
        xt, xv, yt, yv = data.x[np.ix_(train, cols)], data.x[np.ix_(val, cols)], data.y[train], data.y[val]
        started = time.perf_counter()
        try:
            if spec.algorithm == "soft_voting":
                base = {name: models[name][0] if name in models else fit_base(name, "none", xt, yt, config, settings, (xv, yv))
                        for name in STACK_BASES}
                model, weights = fit_voting(base, xv, yv, len(classes))
                configuration = dict(weights=weights, training_seconds_excludes_reused_members=all(name in models for name in base),
                                     bases={name: params(member) for name, member in base.items()})
            elif spec.algorithm == "stacking":
                model, oof = fit_stack(xt, yt, data.ids[train], data.lots[train], args.split == "grouped", config, settings, progress)
                write_csv(out / "oof.csv", oof)
                configuration = dict(bases={k: params(v) for k, v in zip(STACK_BASES, model.models)}, meta=params(model.meta),
                                     folds=len({row["fold"] for row in oof}))
            else:
                model = fit_base(spec.algorithm, spec.balance, xt, yt, config, settings, (xv, yv))
                configuration = params(model)
        except (ImportError, ValueError) as exc:
            is_optional = spec.algorithm in OPTIONAL or spec.balance in {"oversample", "smote"} or spec.algorithm == "stacking"
            if not is_optional or args.command == "train":
                raise
            skipped.append(dict(model=spec.key, reason=str(exc)))
            progress("skip", f"{spec.key} · {exc}", model=spec.key)
            continue
        seconds = time.perf_counter() - started
        if spec.group == "model" or spec.algorithm in {"hist_gradient", "xgboost", "lightgbm", "catboost"}:
            curves.extend(dict(model=spec.key, **row) for row in learning_curve(spec.algorithm, model, xt, yt, (xv, yv)))
        prob_val = probability(model, xv, len(classes))
        scores, _, _ = metrics(yv, prob_val.argmax(axis=1), classes)
        destination = models_dir / f"{spec.key}.joblib"
        bundle = dict(model=model, name=spec.key, classes=classes, task=args.task, source=data.meta["source"]["kind"],
                      run_id=args.run_id, feature_version=FEATURE_VERSION, branches=spec.branches, settings=settings,
                      names=[data.names[i] for i in cols])
        joblib.dump(bundle, destination, compress=3)
        models[spec.key] = (model, cols)
        if spec.algorithm == "soft_voting":
            seconds += sum(details[name]["train_s"] for name in STACK_BASES if name in details)
        details[spec.key] = dict(model=spec.key, algorithm=spec.algorithm, group=spec.group, balance=spec.balance,
                                 features=len(cols), train_s=seconds, size_kib=destination.stat().st_size / 1024,
                                 val_macro_f1=scores["macro_f1"], parameters=configuration, branches=spec.branches)
        progress("validation", f"{spec.key} · Macro-F1={scores['macro_f1']:.4f}", model=spec.key,
                 val_macro_f1=scores["macro_f1"], training_samples=len(train))
    if not models:
        raise ValueError("没有成功训练的模型。")
    singles = [key for key, d in details.items() if d["group"] == "model" and d["algorithm"] not in {"soft_voting", "stacking"}]
    best_single = max(singles, key=lambda k: details[k]["val_macro_f1"]) if singles else None
    candidates = [k for k, d in details.items() if d["group"] in {"model", "ensemble"}]
    best = max(candidates, key=lambda k: details[k]["val_macro_f1"])
    write_json(out / "frozen.json", dict(frozen_utc=utc_now(), selection="validation_macro_f1", best=best,
               best_single=best_single, models=details, skipped=skipped, split_sha256=data.meta["split_sha256"]))
    write_csv(out / "boosting.csv", curves, ["model", "iteration", "train", "val", "metric"])
    progress("freeze", f"按验证集冻结全部模型 · 主模型 {best}")

    pred_dir = out / "predictions"
    for i, (key, (model, cols)) in enumerate(models.items(), 1):
        x = data.x[np.ix_(test, cols)]
        probs = probability(model, x, len(classes))
        predictions = probs.argmax(axis=1)
        write_csv(pred_dir / f"{key}.csv", [dict(id=data.ids[idx], truth=classes[data.y[idx]], prediction=classes[pred],
                  **{f"p_{name}": float(p) for name, p in zip(classes, prob)})
                  for idx, pred, prob in zip(test, predictions, probs)])
        single_rows = rng.choice(len(x), min(24, len(x)), replace=False)
        for j in single_rows[:2]:
            probability(model, x[j:j+1], len(classes))
        durations = []
        for j in single_rows:
            start = time.perf_counter()
            probability(model, x[j:j+1], len(classes))
            durations.append(1000 * (time.perf_counter() - start))
        details[key].update(infer_p50_ms=float(np.median(durations)), infer_p95_ms=float(np.percentile(durations, 95)),
                            latency_repeats=len(durations))
        progress("test", f"{key} · {len(test)} 个冻结测试样本", i, len(models), model=key)
        if key == best:
            save_errors(data, test, predictions, probs, out)
    write_json(out / "timings.json", {key: {k: value for k, value in row.items() if k.startswith("infer_") or k == "latency_repeats"}
                                     for key, row in details.items()})
    evaluate(out)
    model, cols = models[best]
    select = rng.choice(val, min(len(val), 400), replace=False)
    imp = permutation_importance(model, data.x[np.ix_(select, cols)], data.y[select],
                                 scoring="f1_macro", n_repeats=2, random_state=config["seed"]) if hasattr(model, "fit") else None
    if imp is None:
        x, y = data.x[np.ix_(select, cols)], data.y[select]
        score = f1_score(y, model.predict(x), average="macro")
        means = []
        for col in range(x.shape[1]):
            xx = x.copy()
            xx[:, col] = rng.permutation(xx[:, col])
            means.append(score - f1_score(y, model.predict(xx), average="macro"))
    else:
        means = imp.importances_mean
    write_csv(out / "importance.csv", sorted([dict(feature=data.names[c], validation_delta=float(value), validation_n=len(select))
              for c, value in zip(cols, means)], key=lambda row: -row["validation_delta"]))
    train_plot = rng.choice(train, min(len(train), 1100), replace=False)
    np.savez_compressed(out / "train_plot.npz", x=data.x[train_plot], y=data.y[train_plot], names=data.names)


def save_errors(data, test, predictions, probs, out):
    order = np.argsort(-probs.max(axis=1), kind="stable")
    seen, rows = Counter(), []
    error_dir = out / "errors"
    error_dir.mkdir()
    for pos in order:
        i, pred = test[pos], predictions[pos]
        truth = data.y[i]
        if truth == pred or seen[truth] >= 2:
            continue
        seen[truth] += 1
        a = data.image(i)
        file = f"{data.ids[i]}.png"
        export_png(a, error_dir / file)
        rows.append(dict(id=data.ids[i], truth=data.classes[truth], prediction=data.classes[pred], file=file,
                         confidence=float(probs[pos, pred]), truth_probability=float(probs[pos, truth]),
                         defect_share=float((a == 2).sum() / (a != 0).sum()),
                         note="对照缺陷比例、径向位置和连通域；成因需要工艺记录核验。"))
    write_csv(out / "errors.csv", rows, ["id", "truth", "prediction", "file", "confidence", "truth_probability", "defect_share", "note"])


def evaluate(out):
    run, frozen, timings = (read_json(out / name) for name in ("run.json", "frozen.json", "timings.json"))
    classes = run["classes"]
    test = {row["id"]: row["label"] for row in read_csv(out / "split_manifest.csv") if row["part"] == "test"}
    leaderboard, all_classes, confusion = [], [], {}
    for name, details in frozen["models"].items():
        rows = read_csv(out / "predictions" / f"{name}.csv")
        ids = {row["id"]: row["truth"] for row in rows}
        if len(ids) != len(rows) or ids != test:
            raise ValueError(f"{name} 的预测清单与冻结测试集不一致。")
        scores, per_class, matrix = metrics([classes.index(r["truth"]) for r in rows],
                                            [classes.index(r["prediction"]) for r in rows], classes)
        row = {k: v for k, v in details.items() if k not in {"parameters", "branches"}}
        leaderboard.append(dict(**row, **scores, **timings[name], test_n=len(rows)))
        all_classes.extend(dict(model=name, **record) for record in per_class)
        confusion[name] = matrix
    leaderboard.sort(key=lambda row: -row["val_macro_f1"])
    write_csv(out / "leaderboard.csv", leaderboard)
    write_csv(out / "per_class.csv", all_classes)
    write_json(out / "metrics.json", dict(best=frozen["best"], best_single=frozen["best_single"], classes=classes,
               results=leaderboard, confusion=confusion, skipped=frozen["skipped"], source=run["source"]["kind"]))
    return leaderboard
