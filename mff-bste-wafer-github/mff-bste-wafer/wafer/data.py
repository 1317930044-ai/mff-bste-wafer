import io
import pickle
import time
import uuid
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pandas.compat.pickle_compat import _class_locations_map
from pandas.core.internals import BlockManager

from .common import CACHE, CLASSES, SOURCE_URL, SYNTHETIC, sha256, write_csv, write_json

PALETTE = np.array([[23, 34, 47], [199, 226, 223], [233, 88, 66]], dtype=np.uint8)
FORMATS = (".npy", ".npz", ".png")
MAX_UPLOAD = 8 * 1024 * 1024
MAX_CELLS = 1024 * 1024
REQUIRED = {"waferMap", "failureType", "lotName"}


class LegacyBlockManager(BlockManager):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *(args or ((), [], False)), **kwargs)


class LegacyWaferUnpickler(pickle.Unpickler):
    """Resolve the pre-0.20 Pandas modules used in the Kaggle pickle."""

    def find_class(self, module, name):
        if module.startswith("pandas.indexes"):
            module = module.replace("pandas.indexes", "pandas.core.indexes", 1)
        if name in {"Int64Index", "UInt64Index", "Float64Index"} and module.startswith("pandas."):
            return pd.Index
        if name == "BlockManager" and module == "pandas.core.internals":
            return LegacyBlockManager
        module, name = _class_locations_map.get((module, name), (module, name))
        return super().find_class(module, name)


def parse_label(raw):
    while isinstance(raw, (np.ndarray, list, tuple)):
        if isinstance(raw, np.ndarray):
            if raw.size != 1:
                return None
            raw = raw.flat[0]
        else:
            if len(raw) != 1:
                return None
            raw = raw[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        key = raw.strip().lower().replace("_", "-").replace(" ", "-")
        return {label.lower(): label for label in CLASSES}.get(key)
    return None


def validate_map(value):
    a = np.asarray(value)
    if a.ndim != 2 or not a.size or a.size > MAX_CELLS or a.dtype.kind not in "biuf":
        raise ValueError("晶圆必须为不超过 1,048,576 个元素的二维数值矩阵。")
    if a.dtype.kind in "biu":
        allowed = a.min() >= 0 and a.max() <= 2
    else:
        allowed = np.isin(a, [0, 1, 2]).all()
    if not allowed:
        raise ValueError("晶圆仅允许状态 0（外部）、1（正常）、2（缺陷）。")
    if not np.any(a):
        raise ValueError("晶圆没有有效晶粒。")
    return a.astype(np.uint8, copy=False)


def read_npy(stream):
    version = np.lib.format.read_magic(stream)
    readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
    if version not in readers:
        raise ValueError("仅支持 NumPy NPY v1/v2 数值晶圆。")
    shape, _, dtype = readers[version](stream)
    if len(shape) != 2 or min(shape) <= 0 or shape[0] * shape[1] > MAX_CELLS or dtype.kind not in "biuf":
        raise ValueError("NPY 头部不是尺寸合法的二维数值晶圆。")
    stream.seek(0)
    return validate_map(np.load(stream, allow_pickle=False))


def decode(content, filename):
    suffix = Path(filename).suffix.lower()
    if suffix not in FORMATS:
        raise ValueError("请选择 .npy、.npz 或项目导出的固定调色板 PNG。")
    if not 0 < len(content) <= MAX_UPLOAD:
        raise ValueError("文件为空或超过 8 MiB。")
    stream = io.BytesIO(content)
    if suffix == ".npy":
        return read_npy(stream)
    if suffix == ".npz":
        with zipfile.ZipFile(stream) as archive:
            info = archive.infolist()
            if len(info) != 1 or info[0].filename != "wafer.npy" or info[0].file_size > MAX_UPLOAD:
                raise ValueError("NPZ 仅应含一个 wafer 数组，解压后不得超过 8 MiB。")
            return read_npy(io.BytesIO(archive.read(info[0])))
    with Image.open(stream) as png:
        if png.format != "PNG" or png.width * png.height > MAX_CELLS or getattr(png, "n_frames", 1) != 1:
            raise ValueError("需要单帧、尺寸合法的 PNG 文件。")
        rgba = np.asarray(png.convert("RGBA"))
        colors = (rgba[..., None, :3] == PALETTE).all(axis=-1)
        if not np.all(rgba[..., 3] == 255) or not colors.any(axis=-1).all():
            raise ValueError("颜色或透明度与项目固定调色板不一致。请使用 export-samples 导出的 PNG。")
        return validate_map(colors.argmax(axis=-1))


def read_map(path):
    path = Path(path)
    with path.open("rb") as handle:
        return decode(handle.read(MAX_UPLOAD + 1), path.name)


def export_png(a, path):
    Image.fromarray(PALETTE[validate_map(a)]).save(path)


def load_frame(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [n for n in archive.namelist() if Path(n).name.lower() == "lswmd.pkl"]
            if len(candidates) != 1:
                raise ValueError("ZIP 内必须有且仅有一个 LSWMD.pkl。")
            with archive.open(candidates[0]) as handle:
                frame = unpickle_frame(handle)
    elif path.suffix.lower() == ".pkl":
        with path.open("rb") as handle:
            frame = unpickle_frame(handle)
    else:
        raise ValueError("数据必须为受信任的 LSWMD.pkl 或包含它的 ZIP。")
    if not isinstance(frame, pd.DataFrame) or not REQUIRED.issubset(frame.columns):
        raise ValueError(f"数据缺少必需字段：{sorted(REQUIRED)}")
    return frame


def unpickle_frame(handle):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"dtype\(\): align should be passed.*")
        return LegacyWaferUnpickler(handle, encoding="latin1").load()


def synthetic_map(label, rng):
    h, w = rng.integers(23, 52, size=2)
    y, x = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    radius = np.hypot(x, y)
    angle = np.arctan2(y, x)
    direction = rng.uniform(-np.pi, np.pi)
    spread = rng.uniform(.12, .4)
    local = np.abs(np.angle(np.exp(1j * (angle - direction)))) < rng.uniform(.35, .85)
    patterns = {
        "Center": radius < spread,
        "Donut": np.abs(radius - rng.uniform(.42, .65)) < spread / 2,
        "Edge-Loc": (radius > rng.uniform(.63, .82)) & local,
        "Edge-Ring": radius > rng.uniform(.72, .92),
        "Loc": np.hypot(x - .47 * np.cos(direction), y - .47 * np.sin(direction)) < spread,
        "Random": rng.random(radius.shape) < rng.uniform(.03, .26),
        "Scratch": np.abs(x * np.cos(direction) + y * np.sin(direction) - rng.uniform(-.45, .45)) < spread / 3,
        "Near-full": rng.random(radius.shape) < rng.uniform(.75, .99),
        "None": rng.random(radius.shape) < rng.uniform(0, .02),
    }
    defect = patterns[label] ^ (rng.random(radius.shape) < rng.uniform(0, .045))
    return np.where(radius <= 1, 1 + defect, 0).astype(np.uint8)


def collect(path, synthetic, seed, progress, audit=False):
    start = time.perf_counter()
    if synthetic:
        rng = np.random.default_rng(seed)
        data = [(f"s{i}-{j:03d}", label, f"synthetic-lot-{j % 12:02d}", synthetic_map(label, rng))
                for i, (label, count) in enumerate(zip(CLASSES, (72, 44, 104, 90, 70, 36, 48, 18, 155)))
                for j in range(count)]
        meta = dict(kind="synthetic", title=SYNTHETIC, seed=seed, rows=len(data), invalid=0, unlabeled=0,
                    columns=["waferMap", "failureType", "lotName"], states=[0, 1, 2], audited_rows=len(data))
    else:
        path = Path(path)
        source_hash = sha256(path)
        local = CACHE / f"labeled-v1-{source_hash[:20]}.pkl"
        if local.exists() and not audit:
            progress("load", f"复用同源标注晶圆缓存 {source_hash[:12]}")
            with local.open("rb") as handle:
                return pickle.load(handle)
        progress("load", f"读取可信源 {path.name}")
        frame = load_frame(path)
        progress("load", f"字段 {', '.join(frame.columns)}，共 {len(frame):,} 行")
        data, missing, unknown, invalid, invalid_all = [], 0, Counter(), 0, 0
        observed, audited = set(), 0
        values = zip(*(frame[key].to_numpy(copy=False) for key in ("waferMap", "failureType", "lotName")))
        for i, (a, raw_label, lot) in enumerate(values):
            label = parse_label(raw_label)
            if label is None:
                if isinstance(raw_label, (str, bytes)) and str(raw_label).strip():
                    unknown[str(raw_label)[:100]] += 1
                elif (isinstance(raw_label, np.ndarray) and raw_label.size) or (isinstance(raw_label, (list, tuple)) and raw_label):
                    unknown[str(raw_label)[:100]] += 1
                else:
                    missing += 1
            if audit or label:
                audited += 1
                try:
                    a = validate_map(a)
                except ValueError:
                    invalid_all += 1
                    invalid += label is not None
                else:
                    observed.update(np.bincount(a.ravel(), minlength=3).nonzero()[0].tolist())
                    if label:
                        lot_name = str(lot).strip() if pd.notna(lot) else ""
                        data.append((f"w{i:06d}", label, lot_name, a))
            if (i + 1) % 50000 == 0:
                progress("scan", f"已验证 {audited:,} 张矩阵", i + 1, len(frame))
        meta = dict(kind="wm811k", title="WM-811K", url=SOURCE_URL, source_file=path.name, sha256=source_hash,
                    rows=len(frame), columns=list(frame.columns), states=sorted(observed), audited_rows=audited,
                    invalid=invalid, invalid_audited=invalid_all, unlabeled=missing, unknown_labels=dict(unknown))
    if not data:
        raise ValueError("没有通过校验的已标注晶圆。")
    meta.update(counts=dict(Counter(row[1] for row in data)), lots=len({r[2] for r in data if r[2]}),
                collect_seconds=round(time.perf_counter() - start, 3))
    if not synthetic:
        del values
        del frame
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_name(f".{local.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                pickle.dump((data, meta), handle, protocol=5)
            temporary.replace(local)
        finally:
            temporary.unlink(missing_ok=True)
    progress("scan", f"{meta['title']} · 可用已标注 {len(data):,} 张", meta["rows"], meta["rows"])
    return data, meta


def describe(data, meta, out):
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "data_audit.json", meta)
    sizes = Counter((a.shape[0], a.shape[1], int(np.count_nonzero(a))) for _, _, _, a in data)
    write_csv(out / "dimensions.csv", [dict(rows=h, columns=w, valid_cells=n, count=count)
                                     for (h, w, n), count in sizes.most_common()])
    write_csv(out / "classes.csv", [dict(label=c, defect8=0 if c == "None" else meta["counts"].get(c, 0),
                                         all9=meta["counts"].get(c, 0)) for c in CLASSES])


def export_samples(data, meta, out, seed, per_class=2):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    for label in CLASSES:
        pool = [row for row in data if row[1] == label]
        for i in rng.choice(len(pool), min(per_class, len(pool)), replace=False):
            sample_id, _, _, a = pool[i]
            stem = f"{sample_id}_{label}"
            np.save(out / f"{stem}.npy", a, allow_pickle=False)
            export_png(a, out / f"{stem}.png")
            records.append(dict(id=sample_id, label=label, kind=meta["kind"], source=meta["title"],
                                npy=f"{stem}.npy", png=f"{stem}.png"))
    write_csv(out / "samples.csv", records)
