import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .common import BRANCHES, CACHE, FEATURE_VERSION, digest, labels, read_csv, read_json, sha256, write_csv, write_json
from .data import collect, describe, export_samples
from .features import describe_feature, extract

PARTS = ("train", "val", "test")


def split_indices(y, lots, fractions, seed, strategy):
    train_share, val_share, test_share = fractions
    if min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("拆分比例必须为正数且总和为 1。")
    ix = np.arange(len(y))
    classes = set(y)
    if strategy == "stratified":
        train, hold = train_test_split(ix, test_size=1 - train_share, random_state=seed, stratify=y)
        val, test = train_test_split(hold, test_size=test_share / (val_share + test_share),
                                    random_state=seed, stratify=y[hold])
        partitions = (train, val, test)
    else:
        if not np.all(lots != ""):
            raise ValueError("存在缺失 lotName 的样本，无法确认分组隔离。请使用 stratified。")
        for c in classes:
            if len(set(lots[y == c])) < 3:
                raise ValueError(f"类别 {c} 的独立 lot 少于 3 个，无法进行三分组实验。")
        winner, best = None, float("inf")
        count = np.bincount(y)
        candidates = GroupShuffleSplit(24, train_size=train_share, random_state=seed)
        for number, (train, hold) in enumerate(candidates.split(ix, y, lots)):
            inner = GroupShuffleSplit(4, test_size=test_share / (val_share + test_share), random_state=seed + number)
            for v, t in inner.split(hold, y[hold], lots[hold]):
                parts = train, hold[v], hold[t]
                counts = np.array([np.bincount(y[part], minlength=len(count)) for part in parts])
                if np.any(counts == 0):
                    continue
                score = np.mean(np.abs(counts / count - np.asarray(fractions)[:, None]))
                if score < best:
                    winner, best = parts, score
        if winner is None:
            raise ValueError("96 个候选 lot 拆分均未覆盖全部类别。请用 stratified 或修改种子。")
        partitions = winner
    if any(set(y[part]) != classes for part in partitions):
        raise ValueError("训练、验证、测试未同时覆盖全部类别。")
    return dict(zip(PARTS, partitions))


class Prepared:
    def __init__(self, directory):
        self.path = directory
        self.meta = read_json(directory / "metadata.json")
        self.rows = read_csv(directory / "split_manifest.csv")
        with np.load(directory / "features.npz", allow_pickle=False) as stored:
            self.x = stored["x"]
            self.names = stored["names"].tolist()
        self.classes = self.meta["classes"]
        self.y = np.array([self.classes.index(row["label"]) for row in self.rows])
        self.ids = np.array([row["id"] for row in self.rows])
        self.lots = np.array([row["lot"] for row in self.rows])
        self.parts = {p: np.array([i for i, row in enumerate(self.rows) if row["part"] == p]) for p in PARTS}
        if len(self.x) != len(self.rows) or len(set(self.ids)) != len(self.ids):
            raise ValueError("特征缓存与样本清单不一致。")

    def columns(self, branches):
        return np.array([i for i, name in enumerate(self.names) if name.split(".")[0] in branches])

    def image(self, i):
        row = self.rows[i]
        shape = int(row["height"]), int(row["width"])
        with (self.path / "maps.bin").open("rb") as handle:
            handle.seek(int(row["offset"]))
            return np.frombuffer(handle.read(shape[0] * shape[1]), dtype=np.uint8).reshape(shape)


def prepare(path, synthetic, task, mode, strategy, config, branches, progress):
    settings = config["modes"][mode]
    seed = config["seed"]
    source = "synthetic-v1" if synthetic else sha256(path)
    signature = dict(source=source, task=task, seed=seed, mode=mode, strategy=strategy,
                     settings=settings, branches=branches, fractions=config["split_fractions"], version=FEATURE_VERSION)
    key = digest(signature)[:20]
    out = CACHE / key
    if (out / "metadata.json").exists():
        progress("extract", f"复用已校验的特征缓存 {key}")
        return Prepared(out)

    started = time.perf_counter()
    data, source_meta = collect(path, synthetic, seed, progress)
    classes = list(labels(task))
    chosen = [row for row in data if row[1] in classes]
    y = np.array([classes.index(row[1]) for row in chosen])
    lots = np.array([row[2] for row in chosen])
    counts = np.bincount(y, minlength=len(classes))
    if np.any(counts < 5):
        raise ValueError(f"类别样本不足：{dict(zip(classes, counts.tolist()))}")
    progress("split", f"冻结 {strategy} 拆分 · seed={seed}")
    partitions = split_indices(y, lots, config["split_fractions"], seed, strategy)
    rng = np.random.default_rng(seed)
    cap = settings["per_class_cap"]
    if cap:
        for part, share in zip(PARTS, config["split_fractions"]):
            if part != "train" and mode != "quick":
                continue
            ix = partitions[part]
            limit = max(3, int(round(cap * share))) if mode == "quick" else cap
            partitions[part] = np.concatenate([rng.choice(pool, min(len(pool), limit), replace=False)
                                               for c in range(len(classes)) if len(pool := ix[y[ix] == c])])

    stage = CACHE / f".{key}-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        describe(data, source_meta, stage)
        export_samples(data, source_meta, stage / "samples", seed)
        rows, values, names, timing = [], [], None, Counter()
        total = sum(map(len, partitions.values()))
        with (stage / "maps.bin").open("wb") as raw:
            for part, indices in partitions.items():
                for i in sorted(indices):
                    sample_id, label, lot, a = chosen[i]
                    _, groups, times = extract(a, settings, branches, validated=True)
                    row = {key: value for group in groups.values() for key, value in group.items()}
                    names = names or list(row)
                    if list(row) != names or not np.isfinite(list(row.values())).all():
                        raise ValueError(f"特征字段或数值异常：{sample_id}")
                    values.append(list(row.values()))
                    timing.update(times)
                    rows.append(dict(id=sample_id, label=label, lot=lot, part=part, height=a.shape[0],
                                     width=a.shape[1], offset=raw.tell()))
                    raw.write(a.tobytes())
                    if len(rows) % max(1, total // 20) == 0 or len(rows) == total:
                        progress("extract", f"{part} · {len(names)} 维", len(rows), total)
        x = np.asarray(values, dtype=np.float32)
        np.savez_compressed(stage / "features.npz", x=x, names=names)
        write_csv(stage / "split_manifest.csv", rows)
        for branch in branches:
            cols = [(j, n) for j, n in enumerate(names) if n.startswith(branch + ".")]
            write_csv(stage / f"features_{branch}.csv", (dict(id=row["id"], **{n: float(x[i, j]) for j, n in cols})
                      for i, row in enumerate(rows)))
        write_csv(stage / "feature_dictionary.csv", [dict(name=n, branch=n.split(".")[0], formula=f, range=r,
                  explanation=e, branch_mean_ms=1000 * timing[n.split(".")[0]] / total)
                  for n in names for f, r, e in [describe_feature(n)]])
        write_json(stage / "metadata.json", dict(signature=signature, cache_key=key, source=source_meta, classes=classes,
                   split_counts={p: dict(Counter(classes[y[i]] for i in ix)) for p, ix in partitions.items()},
                   split_sha256=sha256(stage / "split_manifest.csv"), extract_s=time.perf_counter() - started,
                   feature_seconds=dict(timing)))
        try:
            stage.rename(out)
        except OSError:
            if not (out / "metadata.json").exists():
                raise
    finally:
        if stage.exists():
            # The temporary directory contains only the flat set of outputs created above.
            for file in stage.glob("*"):
                if file.is_dir():
                    for child in file.iterdir():
                        child.unlink()
                    file.rmdir()
                else:
                    file.unlink()
            stage.rmdir()
    return Prepared(out)
