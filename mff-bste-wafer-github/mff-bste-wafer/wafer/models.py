import numpy as np
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .common import BASE, OPTIONAL, STRATEGIES

STACK_BASES = ("random_forest", "extra_trees", "hist_gradient")


def estimator(name, config, settings, strategy):
    seed, jobs, n, iters = config["seed"], config["jobs"], settings["n_estimators"], settings["boost_iterations"]
    weight = "balanced" if strategy == "class_weight" else None
    parameters = {"random_state": seed}
    if name in {"decision_tree", "random_forest", "extra_trees", "hist_gradient"}:
        parameters["class_weight"] = weight
    if name in {"random_forest", "extra_trees", "bagging", "balanced_rf"}:
        parameters.update(n_estimators=n, n_jobs=jobs)
    if name in {"ada_boost", "gradient_boost"}:
        parameters["n_estimators"] = iters
    if name == "hist_gradient":
        parameters.update(max_iter=iters, early_stopping=False, scoring="loss", n_iter_no_change=12)
    if name == "bagging":
        parameters["estimator"] = DecisionTreeClassifier(max_depth=16, min_samples_leaf=2, class_weight=weight)
    if name == "gradient_boost":
        parameters.update(max_depth=2, learning_rate=.12)
    choices = dict(zip(BASE, (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier,
                              BaggingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, GradientBoostingClassifier)))
    if name == "balanced_rf":
        from imblearn.ensemble import BalancedRandomForestClassifier
        choices[name] = BalancedRandomForestClassifier
    if name == "xgboost":
        from xgboost import XGBClassifier
        choices[name] = XGBClassifier
        parameters.update(n_estimators=iters, n_jobs=jobs, tree_method="hist", eval_metric="mlogloss")
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        choices[name] = LGBMClassifier
        parameters.update(n_estimators=iters, n_jobs=jobs, verbosity=-1, class_weight=weight, deterministic=True,
                          force_col_wise=True)
    if name == "catboost":
        from catboost import CatBoostClassifier
        choices[name] = CatBoostClassifier
        parameters = dict(random_seed=seed, iterations=iters, thread_count=jobs, verbose=False,
                          allow_writing_files=False, auto_class_weights="Balanced" if weight else None)
    parameters.update(config["model_params"].get(name, {}))
    return choices[name](**parameters)


def probability(model, x, n):
    p = model.predict_proba(x)
    if np.array_equal(model.classes_, np.arange(n)):
        return p
    result = np.zeros((len(x), n))
    result[:, np.asarray(model.classes_, dtype=int)] = p
    return result


def fit_base(name, strategy, x, y, config, settings, val=None):
    model = estimator(name, config, settings, strategy)
    if strategy == "oversample":
        from imblearn.over_sampling import RandomOverSampler
        x, y = RandomOverSampler(random_state=config["seed"]).fit_resample(x, y)
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE
        k = min(5, int(np.bincount(y).min()) - 1)
        if k < 1:
            raise ValueError("SMOTE 的每类训练样本至少需要 2 张。")
        x, y = SMOTE(random_state=config["seed"], k_neighbors=k).fit_resample(x, y)
    kw = {}
    if strategy == "class_weight" and name in {"ada_boost", "gradient_boost", "xgboost"}:
        from sklearn.utils.class_weight import compute_sample_weight
        kw["sample_weight"] = compute_sample_weight("balanced", y)
    if name == "hist_gradient" and val is not None:
        model.set_params(early_stopping=True)
        kw.update(X_val=val[0], y_val=val[1])
    if name == "xgboost" and val is not None:
        model.set_params(early_stopping_rounds=12)
        kw.update(eval_set=[(x, y), val], verbose=False)
    if name == "lightgbm" and val is not None:
        from lightgbm import early_stopping
        kw.update(eval_set=[(x, y), val], eval_metric="multi_logloss", callbacks=[early_stopping(12, verbose=False)])
    if name == "catboost" and val is not None:
        kw.update(eval_set=val, early_stopping_rounds=12)
    model.fit(x, y, **kw)
    return model


def learning_curve(name, model, x, y, val):
    if name == "hist_gradient" and len(model.validation_score_):
        return [dict(iteration=i, train=-float(train), val=-float(validation), metric="log_loss")
                for i, (train, validation) in enumerate(zip(model.train_score_, model.validation_score_))]
    if name == "xgboost":
        history, keys, metric = model.evals_result(), ("validation_0", "validation_1"), "mlogloss"
    elif name == "lightgbm":
        history, keys, metric = model.evals_result_, ("training", "valid_1"), "multi_logloss"
    elif name == "catboost":
        history, keys, metric = model.get_evals_result(), ("learn", "validation"), "MultiClass"
    elif name == "gradient_boost":
        classes = model.classes_
        return [dict(iteration=i, train=float(log_loss(y, a, labels=classes)),
                     val=float(log_loss(val[1], b, labels=classes)), metric="log_loss")
                for i, (a, b) in enumerate(zip(model.staged_predict_proba(x), model.staged_predict_proba(val[0])), 1)]
    else:
        return []
    return [dict(iteration=i, train=float(a), val=float(b), metric="log_loss")
            for i, (a, b) in enumerate(zip(history[keys[0]][metric], history[keys[1]][metric]), 1)]


class Voting:
    def __init__(self, models, weights, n):
        self.models, self.weights, self.classes_ = models, weights, np.arange(n)

    def predict_proba(self, x):
        return np.average([probability(model, x, len(self.classes_)) for model in self.models], axis=0, weights=self.weights)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


class Stacking:
    def __init__(self, models, meta, n):
        self.models, self.meta, self.classes_ = models, meta, np.arange(n)

    def predict_proba(self, x):
        joined = np.column_stack([probability(model, x, len(self.classes_)) for model in self.models])
        return self.meta.predict_proba(joined)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


def oof_folds(y, lots, grouped, folds, seed):
    n = min(folds, int(np.bincount(y).min()))
    if n < 2:
        raise ValueError("OOF 需要每类至少两个训练样本。")
    x = np.zeros(len(y))
    if not grouped:
        return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(x, y))
    for candidate in range(n, 1, -1):
        for offset in range(16):
            cv = StratifiedGroupKFold(candidate, shuffle=True, random_state=seed + offset)
            folds_ = list(cv.split(x, y, lots))
            if all(len(np.unique(y[fit])) == len(np.unique(y)) for fit, _ in folds_):
                return folds_
    raise ValueError("训练集的独立 lot 不足以保证 OOF 拟合覆盖所有类别。")


def fit_stack(x, y, ids, lots, grouped, config, settings, progress):
    folds = oof_folds(y, lots, grouped, settings["folds"], config["seed"])
    n = len(np.unique(y))
    oof = np.full((len(y), len(STACK_BASES) * n), np.nan)
    assigned = np.zeros(len(y), dtype=int)
    manifest = []
    for i, (fit, hold) in enumerate(folds, 1):
        if np.intersect1d(fit, hold).size or (grouped and np.intersect1d(lots[fit], lots[hold]).size):
            raise RuntimeError("OOF 拆分交集异常。")
        assigned[hold] += 1
        for j, name in enumerate(STACK_BASES):
            progress("oof", f"{name} · 拟合 {len(fit)}，折外预测 {len(hold)}", i, len(folds), model="stacking")
            model = fit_base(name, "class_weight", x[fit], y[fit], config, settings)
            p = probability(model, x[hold], n)
            oof[hold, j*n:(j+1)*n] = p
        manifest.extend(dict(id=ids[k], fold=i, lot=lots[k], label=int(y[k]),
                             **{f"{name}_p{c}": float(oof[k, j*n+c]) for j, name in enumerate(STACK_BASES) for c in range(n)})
                        for k in hold)
    if not np.all(assigned == 1) or not np.isfinite(oof).all():
        raise RuntimeError("训练样本需要恰好一次折外预测。")
    meta = LogisticRegression(max_iter=500, class_weight="balanced", random_state=config["seed"]).fit(oof, y)
    models = [fit_base(name, "class_weight", x, y, config, settings) for name in STACK_BASES]
    return Stacking(models, meta, n), manifest


def fit_voting(models, x, y, n):
    names = list(models)
    weights = [max(.01, f1_score(y, probability(models[name], x, n).argmax(axis=1), average="macro")) for name in names]
    weights = (np.asarray(weights) / sum(weights)).tolist()
    return Voting(list(models.values()), weights, n), dict(zip(names, weights))
